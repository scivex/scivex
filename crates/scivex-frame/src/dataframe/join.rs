//! Join operations for [`DataFrame`].

use std::collections::HashMap;

use crate::dataframe::DataFrame;
use crate::error::{FrameError, Result};
use crate::series::AnySeries;

/// Type of join to perform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinType {
    /// Keep only rows with matching keys in both data frames.
    Inner,
    /// Keep all rows from the left data frame; fill with nulls for non-matches.
    Left,
    /// Keep all rows from the right data frame; fill with nulls for non-matches.
    Right,
    /// Keep all rows from both data frames; fill with nulls for non-matches.
    Outer,
}

impl DataFrame {
    /// Join two data frames on columns with the same name.
    pub fn join(&self, other: &DataFrame, on: &[&str], how: JoinType) -> Result<DataFrame> {
        self.join_on(other, on, on, how)
    }

    /// Join two data frames on columns that may have different names.
    pub fn join_on(
        &self,
        other: &DataFrame,
        left_on: &[&str],
        right_on: &[&str],
        how: JoinType,
    ) -> Result<DataFrame> {
        if left_on.len() != right_on.len() {
            return Err(FrameError::JoinKeyMismatch {
                left: left_on.len(),
                right: right_on.len(),
            });
        }

        // Validate columns exist
        for &col in left_on {
            self.column(col)?;
        }
        for &col in right_on {
            other.column(col)?;
        }

        // Build hash map for right DF: composite key → Vec<row index>
        let right_nrows = other.nrows();
        let mut right_map: HashMap<String, Vec<usize>> = HashMap::new();
        for row in 0..right_nrows {
            let key = composite_key(other, right_on, row);
            right_map.entry(key).or_default().push(row);
        }

        // Probe left and collect index pairs
        let left_nrows = self.nrows();
        let mut pairs: Vec<(Option<usize>, Option<usize>)> = Vec::new();

        let mut right_matched = vec![false; right_nrows];

        for left_row in 0..left_nrows {
            let key = composite_key(self, left_on, left_row);
            if let Some(right_rows) = right_map.get(&key) {
                for &right_row in right_rows {
                    pairs.push((Some(left_row), Some(right_row)));
                    right_matched[right_row] = true;
                }
            } else if matches!(how, JoinType::Left | JoinType::Outer) {
                pairs.push((Some(left_row), None));
            }
        }

        // For Right/Outer: add unmatched right rows
        if matches!(how, JoinType::Right | JoinType::Outer) {
            for (right_row, &matched) in right_matched.iter().enumerate() {
                if !matched {
                    pairs.push((None, Some(right_row)));
                }
            }
        }

        // Build result columns
        let mut result_cols: Vec<Box<dyn AnySeries>> = Vec::new();

        // Determine which right columns are key columns (to avoid duplication)
        let right_key_set: Vec<&str> = right_on.to_vec();

        // Left key column names for suffix logic
        let left_key_set: Vec<&str> = left_on.to_vec();

        // Collect right non-key column names for suffix check
        let left_names: Vec<&str> = self.column_names();
        let right_non_key_names: Vec<&str> = other
            .column_names()
            .into_iter()
            .filter(|n| !right_key_set.contains(n))
            .collect();

        let left_indices: Vec<Option<usize>> = pairs.iter().map(|(l, _)| *l).collect();
        let right_indices: Vec<Option<usize>> = pairs.iter().map(|(_, r)| *r).collect();

        // Add all left columns
        for col in self.columns() {
            let col_name = col.name();
            let new_col = col.take_optional(&left_indices);
            // If this left column name (non-key) also appears in right non-key, add suffix
            let needs_suffix =
                !left_key_set.contains(&col_name) && right_non_key_names.contains(&col_name);
            if needs_suffix {
                result_cols.push(new_col.rename_box(&format!("{col_name}_left")));
            } else {
                result_cols.push(new_col);
            }
        }

        // Add right non-key columns
        for col in other.columns() {
            let col_name = col.name();
            if right_key_set.contains(&col_name) {
                continue;
            }
            let new_col = col.take_optional(&right_indices);
            let needs_suffix = left_names.contains(&col_name);
            if needs_suffix {
                result_cols.push(new_col.rename_box(&format!("{col_name}_right")));
            } else {
                result_cols.push(new_col);
            }
        }

        DataFrame::new(result_cols)
    }
}

/// Build a composite string key from multiple columns at a given row.
fn composite_key(df: &DataFrame, cols: &[&str], row: usize) -> String {
    cols.iter()
        .map(|&c| df.column(c).unwrap().display_value(row))
        .collect::<Vec<_>>()
        .join("\x00")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::series::Series;
    use crate::series::string::StringSeries;

    fn left_df() -> DataFrame {
        DataFrame::new(vec![
            Box::new(StringSeries::from_strs("key", &["a", "b", "c", "d"])),
            Box::new(Series::new("lval", vec![1_i32, 2, 3, 4])),
        ])
        .unwrap()
    }

    fn right_df() -> DataFrame {
        DataFrame::new(vec![
            Box::new(StringSeries::from_strs("key", &["b", "c", "c", "e"])),
            Box::new(Series::new("rval", vec![20_i32, 30, 31, 50])),
        ])
        .unwrap()
    }

    #[test]
    fn test_inner_join() {
        let result = left_df()
            .join(&right_df(), &["key"], JoinType::Inner)
            .unwrap();
        // b matches 1 row, c matches 2 rows → 3 result rows
        assert_eq!(result.nrows(), 3);
        assert_eq!(result.ncols(), 3); // key, lval, rval
    }

    #[test]
    fn test_left_join() {
        let result = left_df()
            .join(&right_df(), &["key"], JoinType::Left)
            .unwrap();
        // a(no match), b(1), c(2), d(no match) → 5 rows
        assert_eq!(result.nrows(), 5);
        // rval should be null for unmatched rows
        let rval = result.column("rval").unwrap();
        // Row for "a" should be null
        assert!(rval.is_null(0));
    }

    #[test]
    fn test_right_join() {
        let result = left_df()
            .join(&right_df(), &["key"], JoinType::Right)
            .unwrap();
        // b(1), c(2), e(no match from left) → 4 rows
        assert_eq!(result.nrows(), 4);
    }

    #[test]
    fn test_outer_join() {
        let result = left_df()
            .join(&right_df(), &["key"], JoinType::Outer)
            .unwrap();
        // a(1), b(1), c(2), d(1), e(1) → 6 rows
        assert_eq!(result.nrows(), 6);
    }

    #[test]
    fn test_multi_key_join() {
        let left = DataFrame::new(vec![
            Box::new(StringSeries::from_strs("k1", &["a", "a", "b"])),
            Box::new(StringSeries::from_strs("k2", &["x", "y", "x"])),
            Box::new(Series::new("v", vec![1_i32, 2, 3])),
        ])
        .unwrap();
        let right = DataFrame::new(vec![
            Box::new(StringSeries::from_strs("k1", &["a", "b"])),
            Box::new(StringSeries::from_strs("k2", &["x", "x"])),
            Box::new(Series::new("w", vec![10_i32, 30])),
        ])
        .unwrap();
        let result = left.join(&right, &["k1", "k2"], JoinType::Inner).unwrap();
        assert_eq!(result.nrows(), 2); // (a,x) and (b,x)
    }

    #[test]
    fn test_duplicate_column_suffixes() {
        let left = DataFrame::new(vec![
            Box::new(StringSeries::from_strs("key", &["a", "b"])),
            Box::new(Series::new("val", vec![1_i32, 2])),
        ])
        .unwrap();
        let right = DataFrame::new(vec![
            Box::new(StringSeries::from_strs("key", &["a", "b"])),
            Box::new(Series::new("val", vec![10_i32, 20])),
        ])
        .unwrap();
        let result = left.join(&right, &["key"], JoinType::Inner).unwrap();
        let names = result.column_names();
        assert!(names.contains(&"val_left"));
        assert!(names.contains(&"val_right"));
    }

    #[test]
    fn test_different_key_names() {
        let left = DataFrame::new(vec![
            Box::new(StringSeries::from_strs("lk", &["a", "b"])),
            Box::new(Series::new("lval", vec![1_i32, 2])),
        ])
        .unwrap();
        let right = DataFrame::new(vec![
            Box::new(StringSeries::from_strs("rk", &["b", "a"])),
            Box::new(Series::new("rval", vec![20_i32, 10])),
        ])
        .unwrap();
        let result = left
            .join_on(&right, &["lk"], &["rk"], JoinType::Inner)
            .unwrap();
        assert_eq!(result.nrows(), 2);
    }

    #[test]
    fn test_empty_df_join() {
        let left = DataFrame::new(vec![
            Box::new(StringSeries::from_strs("key", &[])),
            Box::new(Series::new("v", Vec::<i32>::new())),
        ])
        .unwrap();
        let right = right_df();
        let result = left.join(&right, &["key"], JoinType::Inner).unwrap();
        assert_eq!(result.nrows(), 0);
    }
}
