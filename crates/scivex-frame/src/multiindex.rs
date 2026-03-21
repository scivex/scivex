//! Hierarchical (multi-level) row/column indexing, similar to pandas `MultiIndex`.

use std::collections::HashMap;
use std::fmt;

use crate::error::{FrameError, Result};

/// A hierarchical index with multiple levels, like pandas `MultiIndex`.
///
/// Each level has a set of distinct labels, and each row is described by a
/// tuple of codes — one per level — that indexes into the level labels.
///
/// # Examples
///
/// ```
/// use scivex_frame::MultiIndex;
///
/// let mi = MultiIndex::from_tuples(
///     &["year", "quarter"],
///     &[
///         vec!["2024", "Q1"],
///         vec!["2024", "Q2"],
///         vec!["2025", "Q1"],
///     ],
/// ).unwrap();
///
/// assert_eq!(mi.nlevels(), 2);
/// assert_eq!(mi.len(), 3);
/// ```
#[derive(Debug, Clone)]
pub struct MultiIndex {
    /// Level names.
    names: Vec<String>,
    /// Level values: `levels[i]` contains the distinct labels for level `i`.
    levels: Vec<Vec<String>>,
    /// Codes: `codes[i][j]` is the index into `levels[i]` for row `j`.
    codes: Vec<Vec<usize>>,
    /// Number of rows.
    len: usize,
}

impl MultiIndex {
    // -- Constructors --------------------------------------------------------

    /// Create a `MultiIndex` from pre-computed names, levels, and codes.
    ///
    /// Returns an error if dimensions are inconsistent.
    pub fn new(
        names: Vec<String>,
        levels: Vec<Vec<String>>,
        codes: Vec<Vec<usize>>,
    ) -> Result<Self> {
        let nlevels = names.len();
        if levels.len() != nlevels {
            return Err(FrameError::InvalidValue {
                reason: format!(
                    "MultiIndex: expected {} level arrays, got {}",
                    nlevels,
                    levels.len()
                ),
            });
        }
        if codes.len() != nlevels {
            return Err(FrameError::InvalidValue {
                reason: format!(
                    "MultiIndex: expected {} code arrays, got {}",
                    nlevels,
                    codes.len()
                ),
            });
        }

        // All code arrays must have the same length.
        let len = codes.first().map_or(0, Vec::len);
        for (i, c) in codes.iter().enumerate() {
            if c.len() != len {
                return Err(FrameError::RowCountMismatch {
                    expected: len,
                    got: c.len(),
                });
            }
            // Each code must be a valid index into its level.
            for &code in c {
                if code >= levels[i].len() {
                    return Err(FrameError::IndexOutOfBounds {
                        index: code,
                        length: levels[i].len(),
                    });
                }
            }
        }

        Ok(Self {
            names,
            levels,
            codes,
            len,
        })
    }

    /// Build a `MultiIndex` from a list of row tuples.
    ///
    /// Each element of `tuples` must have the same length as `names`.
    ///
    /// ```
    /// use scivex_frame::MultiIndex;
    /// let mi = MultiIndex::from_tuples(
    ///     &["city", "year"],
    ///     &[vec!["NYC", "2024"], vec!["NYC", "2025"], vec!["LA", "2024"]],
    /// ).unwrap();
    /// assert_eq!(mi.len(), 3);
    /// ```
    pub fn from_tuples(names: &[&str], tuples: &[Vec<&str>]) -> Result<Self> {
        let nlevels = names.len();

        for (row_idx, t) in tuples.iter().enumerate() {
            if t.len() != nlevels {
                return Err(FrameError::InvalidValue {
                    reason: format!(
                        "MultiIndex::from_tuples: tuple at row {} has {} elements, expected {}",
                        row_idx,
                        t.len(),
                        nlevels
                    ),
                });
            }
        }

        // Collect distinct labels per level and build codes.
        let mut levels: Vec<Vec<String>> = vec![Vec::new(); nlevels];
        let mut label_maps: Vec<HashMap<String, usize>> = vec![HashMap::new(); nlevels];
        let mut codes: Vec<Vec<usize>> = vec![Vec::with_capacity(tuples.len()); nlevels];

        for t in tuples {
            for (lev, val) in t.iter().enumerate() {
                let val_str = val.to_string();
                let next_id = label_maps[lev].len();
                let id = *label_maps[lev].entry(val_str.clone()).or_insert_with(|| {
                    levels[lev].push(val_str);
                    next_id
                });
                codes[lev].push(id);
            }
        }

        let owned_names = names.iter().map(ToString::to_string).collect();
        Self::new(owned_names, levels, codes)
    }

    /// Build a `MultiIndex` from the Cartesian product of the given levels.
    ///
    /// ```
    /// use scivex_frame::MultiIndex;
    /// let mi = MultiIndex::from_product(
    ///     &["color", "size"],
    ///     &[vec!["red", "blue"], vec!["S", "M", "L"]],
    /// ).unwrap();
    /// assert_eq!(mi.len(), 6); // 2 * 3
    /// ```
    pub fn from_product(names: &[&str], level_values: &[Vec<&str>]) -> Result<Self> {
        let nlevels = names.len();
        if level_values.len() != nlevels {
            return Err(FrameError::InvalidValue {
                reason: format!(
                    "MultiIndex::from_product: expected {} level arrays, got {}",
                    nlevels,
                    level_values.len()
                ),
            });
        }

        if nlevels == 0 {
            return Self::new(Vec::new(), Vec::new(), Vec::new());
        }

        // Compute total number of rows.
        let total: usize = level_values.iter().map(Vec::len).product();

        let levels: Vec<Vec<String>> = level_values
            .iter()
            .map(|lv| lv.iter().map(ToString::to_string).collect())
            .collect();

        let mut codes: Vec<Vec<usize>> = vec![Vec::with_capacity(total); nlevels];

        // Build codes using repeated tiling.
        // For level i, each code value is repeated `repeat` times, and the
        // whole pattern tiles `tile` times.
        let mut repeat = total;
        for lev in 0..nlevels {
            let n = levels[lev].len();
            if n == 0 {
                // Product with an empty level yields 0 rows — already handled
                // by `total == 0`.
                break;
            }
            repeat /= n;
            for _ in 0..(total / (n * repeat)) {
                for code in 0..n {
                    for _ in 0..repeat {
                        codes[lev].push(code);
                    }
                }
            }
        }

        let owned_names = names.iter().map(ToString::to_string).collect();
        Self::new(owned_names, levels, codes)
    }

    /// Build a `MultiIndex` from parallel arrays (one per level).
    ///
    /// Each array must have the same length; this is the number of rows.
    ///
    /// ```
    /// use scivex_frame::MultiIndex;
    /// let mi = MultiIndex::from_arrays(
    ///     &["letter", "number"],
    ///     &[
    ///         vec!["A".into(), "A".into(), "B".into()],
    ///         vec!["1".into(), "2".into(), "1".into()],
    ///     ],
    /// ).unwrap();
    /// assert_eq!(mi.len(), 3);
    /// ```
    pub fn from_arrays(names: &[&str], arrays: &[Vec<String>]) -> Result<Self> {
        let nlevels = names.len();
        if arrays.len() != nlevels {
            return Err(FrameError::InvalidValue {
                reason: format!(
                    "MultiIndex::from_arrays: expected {} arrays, got {}",
                    nlevels,
                    arrays.len()
                ),
            });
        }

        let len = arrays.first().map_or(0, Vec::len);
        for (i, arr) in arrays.iter().enumerate() {
            if arr.len() != len {
                return Err(FrameError::InvalidValue {
                    reason: format!(
                        "MultiIndex::from_arrays: array {} has length {}, expected {}",
                        i,
                        arr.len(),
                        len
                    ),
                });
            }
        }

        // Deduplicate labels per level.
        let mut levels: Vec<Vec<String>> = vec![Vec::new(); nlevels];
        let mut label_maps: Vec<HashMap<String, usize>> = vec![HashMap::new(); nlevels];
        let mut codes: Vec<Vec<usize>> = vec![Vec::with_capacity(len); nlevels];

        #[allow(clippy::needless_range_loop)]
        for row in 0..len {
            for lev in 0..nlevels {
                let val = &arrays[lev][row];
                let next_id = label_maps[lev].len();
                let id = *label_maps[lev].entry(val.clone()).or_insert_with(|| {
                    levels[lev].push(val.clone());
                    next_id
                });
                codes[lev].push(id);
            }
        }

        let owned_names = names.iter().map(ToString::to_string).collect();
        Self::new(owned_names, levels, codes)
    }

    // -- Accessors -----------------------------------------------------------

    /// Number of index levels.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::MultiIndex;
    /// let mi = MultiIndex::from_tuples(
    ///     &["a", "b"],
    ///     &[vec!["x", "1"], vec!["y", "2"]],
    /// ).unwrap();
    /// assert_eq!(mi.nlevels(), 2);
    /// ```
    #[inline]
    pub fn nlevels(&self) -> usize {
        self.names.len()
    }

    /// Number of rows in the index.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::MultiIndex;
    /// let mi = MultiIndex::from_tuples(
    ///     &["a"],
    ///     &[vec!["x"], vec!["y"]],
    /// ).unwrap();
    /// assert_eq!(mi.len(), 2);
    /// ```
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the index has zero rows.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Level names.
    #[inline]
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// Return the distinct labels at the given `level`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::MultiIndex;
    /// let mi = MultiIndex::from_tuples(
    ///     &["city", "year"],
    ///     &[vec!["NYC", "2024"], vec!["LA", "2024"]],
    /// ).unwrap();
    /// let cities = mi.get_level(0).unwrap();
    /// assert!(cities.contains(&"NYC"));
    /// ```
    pub fn get_level(&self, level: usize) -> Result<Vec<&str>> {
        if level >= self.nlevels() {
            return Err(FrameError::IndexOutOfBounds {
                index: level,
                length: self.nlevels(),
            });
        }
        Ok(self.levels[level].iter().map(String::as_str).collect())
    }

    /// Return the label tuple for a given row.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::MultiIndex;
    /// let mi = MultiIndex::from_tuples(
    ///     &["city", "year"],
    ///     &[vec!["NYC", "2024"], vec!["LA", "2025"]],
    /// ).unwrap();
    /// assert_eq!(mi.get_label(0).unwrap(), vec!["NYC", "2024"]);
    /// ```
    pub fn get_label(&self, row: usize) -> Result<Vec<&str>> {
        if row >= self.len {
            return Err(FrameError::IndexOutOfBounds {
                index: row,
                length: self.len,
            });
        }
        Ok(self
            .codes
            .iter()
            .enumerate()
            .map(|(lev, c)| self.levels[lev][c[row]].as_str())
            .collect())
    }

    // -- Selection -----------------------------------------------------------

    /// Return the row indices where `level` has the given `label`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::MultiIndex;
    /// let mi = MultiIndex::from_tuples(
    ///     &["region", "year"],
    ///     &[vec!["East", "2023"], vec!["West", "2023"], vec!["East", "2024"]],
    /// ).unwrap();
    /// assert_eq!(mi.select(0, "East"), vec![0, 2]);
    /// ```
    pub fn select(&self, level: usize, label: &str) -> Vec<usize> {
        if level >= self.nlevels() {
            return Vec::new();
        }
        // Find the code for `label` in this level.
        let Some(code) = self.levels[level].iter().position(|s| s == label) else {
            return Vec::new();
        };
        self.codes[level]
            .iter()
            .enumerate()
            .filter(|&(_, c)| *c == code)
            .map(|(i, _)| i)
            .collect()
    }

    /// Return the row indices where all levels match the given `labels` tuple.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::MultiIndex;
    /// let mi = MultiIndex::from_tuples(
    ///     &["region", "year"],
    ///     &[vec!["East", "2023"], vec!["West", "2024"]],
    /// ).unwrap();
    /// assert_eq!(mi.select_tuple(&["East", "2023"]), vec![0]);
    /// ```
    pub fn select_tuple(&self, labels: &[&str]) -> Vec<usize> {
        if labels.len() != self.nlevels() {
            return Vec::new();
        }
        // Resolve each label to its code.
        let mut target_codes: Vec<usize> = Vec::with_capacity(self.nlevels());
        for (lev, label) in labels.iter().enumerate() {
            match self.levels[lev].iter().position(|s| s == label) {
                Some(c) => target_codes.push(c),
                None => return Vec::new(),
            }
        }
        (0..self.len)
            .filter(|&row| {
                self.codes
                    .iter()
                    .zip(target_codes.iter())
                    .all(|(c, &tc)| c[row] == tc)
            })
            .collect()
    }

    // -- Transformations -----------------------------------------------------

    /// Swap two levels, returning a new `MultiIndex`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::MultiIndex;
    /// let mi = MultiIndex::from_tuples(
    ///     &["A", "B"],
    ///     &[vec!["a1", "b1"]],
    /// ).unwrap();
    /// let swapped = mi.swap_levels(0, 1).unwrap();
    /// assert_eq!(swapped.names(), &["B", "A"]);
    /// ```
    pub fn swap_levels(&self, i: usize, j: usize) -> Result<MultiIndex> {
        if i >= self.nlevels() {
            return Err(FrameError::IndexOutOfBounds {
                index: i,
                length: self.nlevels(),
            });
        }
        if j >= self.nlevels() {
            return Err(FrameError::IndexOutOfBounds {
                index: j,
                length: self.nlevels(),
            });
        }
        let mut names = self.names.clone();
        let mut levels = self.levels.clone();
        let mut codes = self.codes.clone();
        names.swap(i, j);
        levels.swap(i, j);
        codes.swap(i, j);
        Self::new(names, levels, codes)
    }

    /// Remove a level, returning a new `MultiIndex`.
    ///
    /// Returns an error if there is only one level (cannot drop the last one)
    /// or if the index is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::MultiIndex;
    /// let mi = MultiIndex::from_tuples(
    ///     &["A", "B", "C"],
    ///     &[vec!["a1", "b1", "c1"]],
    /// ).unwrap();
    /// let dropped = mi.droplevel(1).unwrap();
    /// assert_eq!(dropped.nlevels(), 2);
    /// assert_eq!(dropped.names(), &["A", "C"]);
    /// ```
    pub fn droplevel(&self, level: usize) -> Result<MultiIndex> {
        if level >= self.nlevels() {
            return Err(FrameError::IndexOutOfBounds {
                index: level,
                length: self.nlevels(),
            });
        }
        if self.nlevels() <= 1 {
            return Err(FrameError::InvalidValue {
                reason: "MultiIndex::droplevel: cannot drop the only remaining level".into(),
            });
        }
        let mut names = self.names.clone();
        let mut levels = self.levels.clone();
        let mut codes = self.codes.clone();
        names.remove(level);
        levels.remove(level);
        codes.remove(level);
        Self::new(names, levels, codes)
    }

    /// Flatten the multi-level labels into single strings joined by `_`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::MultiIndex;
    /// let mi = MultiIndex::from_tuples(
    ///     &["color", "size"],
    ///     &[vec!["red", "S"], vec!["blue", "M"]],
    /// ).unwrap();
    /// assert_eq!(mi.to_flat_index(), vec!["red_S", "blue_M"]);
    /// ```
    pub fn to_flat_index(&self) -> Vec<String> {
        (0..self.len)
            .map(|row| {
                let parts: Vec<&str> = self
                    .codes
                    .iter()
                    .enumerate()
                    .map(|(lev, c)| self.levels[lev][c[row]].as_str())
                    .collect();
                parts.join("_")
            })
            .collect()
    }

    /// Sort the index lexicographically.
    ///
    /// Returns the sorted `MultiIndex` and the permutation vector mapping new
    /// positions to old row indices.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::MultiIndex;
    /// let mi = MultiIndex::from_tuples(
    ///     &["letter"],
    ///     &[vec!["C"], vec!["A"], vec!["B"]],
    /// ).unwrap();
    /// let (sorted, perm) = mi.sort();
    /// assert_eq!(sorted.get_label(0).unwrap(), vec!["A"]);
    /// assert_eq!(perm, vec![1, 2, 0]);
    /// ```
    pub fn sort(&self) -> (MultiIndex, Vec<usize>) {
        let mut perm: Vec<usize> = (0..self.len).collect();
        perm.sort_by(|&a, &b| {
            for (lev, c) in self.codes.iter().enumerate() {
                let la = &self.levels[lev][c[a]];
                let lb = &self.levels[lev][c[b]];
                match la.cmp(lb) {
                    std::cmp::Ordering::Equal => {}
                    other => return other,
                }
            }
            std::cmp::Ordering::Equal
        });

        // Rebuild codes in sorted order.
        let codes: Vec<Vec<usize>> = self
            .codes
            .iter()
            .map(|c| perm.iter().map(|&p| c[p]).collect())
            .collect();

        let sorted = MultiIndex {
            names: self.names.clone(),
            levels: self.levels.clone(),
            codes,
            len: self.len,
        };

        (sorted, perm)
    }
}

// ---------------------------------------------------------------------------
// Display
// ---------------------------------------------------------------------------

impl fmt::Display for MultiIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Column widths: max of name length and all label lengths per level.
        let mut widths: Vec<usize> = self.names.iter().map(String::len).collect();
        for (lev, lvl) in self.levels.iter().enumerate() {
            for label in lvl {
                if label.len() > widths[lev] {
                    widths[lev] = label.len();
                }
            }
        }

        // Header
        for (lev, name) in self.names.iter().enumerate() {
            if lev > 0 {
                write!(f, "  ")?;
            }
            write!(f, "{:>width$}", name, width = widths[lev])?;
        }
        writeln!(f)?;

        // Separator
        for (lev, &w) in widths.iter().enumerate() {
            if lev > 0 {
                write!(f, "  ")?;
            }
            for _ in 0..w {
                write!(f, "-")?;
            }
        }
        writeln!(f)?;

        // Rows
        for row in 0..self.len {
            for (lev, c) in self.codes.iter().enumerate() {
                if lev > 0 {
                    write!(f, "  ")?;
                }
                let label = &self.levels[lev][c[row]];
                write!(f, "{:>width$}", label, width = widths[lev])?;
            }
            writeln!(f)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multiindex_from_tuples() {
        let mi = MultiIndex::from_tuples(
            &["city", "year"],
            &[vec!["NYC", "2024"], vec!["NYC", "2025"], vec!["LA", "2024"]],
        )
        .unwrap();

        assert_eq!(mi.nlevels(), 2);
        assert_eq!(mi.len(), 3);
        assert!(!mi.is_empty());
        assert_eq!(mi.names(), &["city", "year"]);

        // Check labels.
        assert_eq!(mi.get_label(0).unwrap(), vec!["NYC", "2024"]);
        assert_eq!(mi.get_label(1).unwrap(), vec!["NYC", "2025"]);
        assert_eq!(mi.get_label(2).unwrap(), vec!["LA", "2024"]);

        // Out-of-bounds access returns error.
        assert!(mi.get_label(3).is_err());

        // Get distinct level labels.
        let cities = mi.get_level(0).unwrap();
        assert!(cities.contains(&"NYC"));
        assert!(cities.contains(&"LA"));
    }

    #[test]
    fn test_multiindex_from_product() {
        let mi = MultiIndex::from_product(
            &["color", "size"],
            &[vec!["red", "blue"], vec!["S", "M", "L"]],
        )
        .unwrap();

        assert_eq!(mi.len(), 6);
        assert_eq!(mi.nlevels(), 2);

        // First row should be (red, S), last should be (blue, L).
        assert_eq!(mi.get_label(0).unwrap(), vec!["red", "S"]);
        assert_eq!(mi.get_label(5).unwrap(), vec!["blue", "L"]);

        // All six combos should be present.
        let flat = mi.to_flat_index();
        assert_eq!(flat.len(), 6);
        assert!(flat.contains(&"red_S".to_string()));
        assert!(flat.contains(&"blue_L".to_string()));
    }

    #[test]
    fn test_multiindex_select() {
        let mi = MultiIndex::from_tuples(
            &["region", "year"],
            &[
                vec!["East", "2023"],
                vec!["East", "2024"],
                vec!["West", "2023"],
                vec!["West", "2024"],
            ],
        )
        .unwrap();

        // Select by level 0.
        let east_rows = mi.select(0, "East");
        assert_eq!(east_rows, vec![0, 1]);

        let west_rows = mi.select(0, "West");
        assert_eq!(west_rows, vec![2, 3]);

        // Select by level 1.
        let y2023 = mi.select(1, "2023");
        assert_eq!(y2023, vec![0, 2]);

        // Non-existent label returns empty.
        let none = mi.select(0, "North");
        assert!(none.is_empty());

        // Out-of-bounds level returns empty.
        let oob = mi.select(5, "East");
        assert!(oob.is_empty());
    }

    #[test]
    fn test_multiindex_select_tuple() {
        let mi = MultiIndex::from_tuples(
            &["region", "year"],
            &[
                vec!["East", "2023"],
                vec!["East", "2024"],
                vec!["West", "2023"],
                vec!["West", "2024"],
            ],
        )
        .unwrap();

        let rows = mi.select_tuple(&["East", "2024"]);
        assert_eq!(rows, vec![1]);

        let rows = mi.select_tuple(&["West", "2023"]);
        assert_eq!(rows, vec![2]);

        // Wrong number of labels.
        let empty = mi.select_tuple(&["East"]);
        assert!(empty.is_empty());

        // Non-existent combo.
        let empty = mi.select_tuple(&["North", "2023"]);
        assert!(empty.is_empty());
    }

    #[test]
    fn test_multiindex_swap_levels() {
        let mi =
            MultiIndex::from_tuples(&["A", "B"], &[vec!["a1", "b1"], vec!["a2", "b2"]]).unwrap();

        let swapped = mi.swap_levels(0, 1).unwrap();
        assert_eq!(swapped.names(), &["B", "A"]);
        assert_eq!(swapped.get_label(0).unwrap(), vec!["b1", "a1"]);
        assert_eq!(swapped.get_label(1).unwrap(), vec!["b2", "a2"]);

        // Out-of-bounds level is an error.
        assert!(mi.swap_levels(0, 5).is_err());
    }

    #[test]
    fn test_multiindex_droplevel() {
        let mi = MultiIndex::from_tuples(
            &["A", "B", "C"],
            &[vec!["a1", "b1", "c1"], vec!["a2", "b2", "c2"]],
        )
        .unwrap();

        // Drop level 1 ("B").
        let dropped = mi.droplevel(1).unwrap();
        assert_eq!(dropped.nlevels(), 2);
        assert_eq!(dropped.names(), &["A", "C"]);
        assert_eq!(dropped.get_label(0).unwrap(), vec!["a1", "c1"]);
        assert_eq!(dropped.get_label(1).unwrap(), vec!["a2", "c2"]);

        // Cannot drop the last level.
        let two_level = mi.droplevel(0).unwrap();
        let one_level = two_level.droplevel(0).unwrap();
        assert!(one_level.droplevel(0).is_err());

        // Out-of-bounds is an error.
        assert!(mi.droplevel(5).is_err());
    }
}
