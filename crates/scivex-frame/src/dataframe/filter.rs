//! Row filtering, slicing, and sorting for [`DataFrame`].

use super::DataFrame;
use crate::error::{FrameError, Result};
use crate::series::AnySeries;

impl DataFrame {
    /// First `n` rows.
    ///
    /// # Examples
    ///
    /// ```
    /// use scivex_frame::{DataFrame, Series};
    /// let df = DataFrame::builder()
    ///     .add_column("x", vec![10_i32, 20, 30, 40])
    ///     .build()
    ///     .unwrap();
    /// let top = df.head(2);
    /// assert_eq!(top.nrows(), 2);
    /// ```
    pub fn head(&self, n: usize) -> DataFrame {
        let n = n.min(self.nrows());
        let cols = self.columns.iter().map(|c| c.slice(0, n)).collect();
        DataFrame { columns: cols }
    }

    /// Last `n` rows.
    ///
    /// # Examples
    ///
    /// ```
    /// use scivex_frame::{DataFrame, Series};
    /// let df = DataFrame::builder()
    ///     .add_column("x", vec![10_i32, 20, 30, 40])
    ///     .build()
    ///     .unwrap();
    /// let bottom = df.tail(2);
    /// assert_eq!(bottom.nrows(), 2);
    /// ```
    pub fn tail(&self, n: usize) -> DataFrame {
        let total = self.nrows();
        let n = n.min(total);
        let offset = total - n;
        let cols = self.columns.iter().map(|c| c.slice(offset, n)).collect();
        DataFrame { columns: cols }
    }

    /// Contiguous row slice.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::prelude::*;
    /// let df = DataFrame::builder()
    ///     .add_column("x", vec![10_i32, 20, 30, 40])
    ///     .build()
    ///     .unwrap();
    /// let middle = df.slice(1, 2);
    /// assert_eq!(middle.nrows(), 2);
    /// ```
    pub fn slice(&self, offset: usize, length: usize) -> DataFrame {
        let cols = self
            .columns
            .iter()
            .map(|c| c.slice(offset, length))
            .collect();
        DataFrame { columns: cols }
    }

    /// Keep rows where `mask[i]` is true.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::prelude::*;
    /// let df = DataFrame::builder()
    ///     .add_column("x", vec![10_i32, 20, 30])
    ///     .build()
    ///     .unwrap();
    /// let filtered = df.filter(&[true, false, true]).unwrap();
    /// assert_eq!(filtered.nrows(), 2);
    /// ```
    pub fn filter(&self, mask: &[bool]) -> Result<DataFrame> {
        if mask.len() != self.nrows() {
            return Err(FrameError::RowCountMismatch {
                expected: self.nrows(),
                got: mask.len(),
            });
        }
        let cols = self.columns.iter().map(|c| c.filter_mask(mask)).collect();
        Ok(DataFrame { columns: cols })
    }

    /// Sort all rows by a single column.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::prelude::*;
    /// let df = DataFrame::builder()
    ///     .add_column("x", vec![3_i32, 1, 2])
    ///     .build()
    ///     .unwrap();
    /// let sorted = df.sort_by("x", true).unwrap();
    /// assert_eq!(sorted.column_typed::<i32>("x").unwrap().as_slice(), &[1, 2, 3]);
    /// ```
    pub fn sort_by(&self, column: &str, ascending: bool) -> Result<DataFrame> {
        let col = self.column(column)?;
        // Build sort indices via display_value comparison (simple MVP approach).
        // For numeric columns this works because the AnySeries stores real data
        // and we can access via take_indices.
        let n = col.len();
        let mut indices: Vec<usize> = (0..n).collect();

        // We need a way to compare arbitrary column values. For MVP, we use
        // display_value string comparison which works correctly for integers.
        // For proper numeric sort we use the concrete typed approach below.
        // Try to get a typed sort key first.
        sort_indices_by_column(col, &mut indices, ascending);

        let cols = self
            .columns
            .iter()
            .map(|c| c.take_indices(&indices))
            .collect();
        Ok(DataFrame { columns: cols })
    }
}

/// Sort indices by a column's values using string representation.
fn sort_indices_by_column(col: &dyn AnySeries, indices: &mut [usize], ascending: bool) {
    indices.sort_by(|&a, &b| {
        let va = col.display_value(a);
        let vb = col.display_value(b);
        // Try numeric parse first for proper numeric ordering.
        let cmp = match (va.parse::<f64>(), vb.parse::<f64>()) {
            (Ok(fa), Ok(fb)) => fa.partial_cmp(&fb).unwrap_or(core::cmp::Ordering::Equal),
            _ => va.cmp(&vb),
        };
        if ascending { cmp } else { cmp.reverse() }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::series::Series;

    fn sample_df() -> DataFrame {
        DataFrame::new(vec![
            Box::new(Series::new("a", vec![3_i32, 1, 2])),
            Box::new(Series::new("b", vec![30.0_f64, 10.0, 20.0])),
        ])
        .unwrap()
    }

    #[test]
    fn test_head() {
        let df = sample_df();
        let h = df.head(2);
        assert_eq!(h.nrows(), 2);
        let col = h.column_typed::<i32>("a").unwrap();
        assert_eq!(col.as_slice(), &[3, 1]);
    }

    #[test]
    fn test_tail() {
        let df = sample_df();
        let t = df.tail(2);
        assert_eq!(t.nrows(), 2);
        let col = t.column_typed::<i32>("a").unwrap();
        assert_eq!(col.as_slice(), &[1, 2]);
    }

    #[test]
    fn test_slice() {
        let df = sample_df();
        let s = df.slice(1, 1);
        assert_eq!(s.nrows(), 1);
        let col = s.column_typed::<i32>("a").unwrap();
        assert_eq!(col.as_slice(), &[1]);
    }

    #[test]
    fn test_filter() {
        let df = sample_df();
        let filtered = df.filter(&[true, false, true]).unwrap();
        assert_eq!(filtered.nrows(), 2);
        let col = filtered.column_typed::<i32>("a").unwrap();
        assert_eq!(col.as_slice(), &[3, 2]);
    }

    #[test]
    fn test_filter_length_mismatch() {
        let df = sample_df();
        assert!(df.filter(&[true]).is_err());
    }

    #[test]
    fn test_sort_by_ascending() {
        let df = sample_df();
        let sorted = df.sort_by("a", true).unwrap();
        let col = sorted.column_typed::<i32>("a").unwrap();
        assert_eq!(col.as_slice(), &[1, 2, 3]);
        // Check that column b is reordered correspondingly.
        let col_b = sorted.column_typed::<f64>("b").unwrap();
        assert_eq!(col_b.as_slice(), &[10.0, 20.0, 30.0]);
    }

    #[test]
    fn test_sort_by_descending() {
        let df = sample_df();
        let sorted = df.sort_by("a", false).unwrap();
        let col = sorted.column_typed::<i32>("a").unwrap();
        assert_eq!(col.as_slice(), &[3, 2, 1]);
    }

    #[test]
    fn test_head_larger_than_nrows() {
        let df = sample_df();
        let h = df.head(100);
        assert_eq!(h.nrows(), 3);
    }

    #[test]
    fn test_tail_larger_than_nrows() {
        let df = sample_df();
        let t = df.tail(100);
        assert_eq!(t.nrows(), 3);
        let col = t.column_typed::<i32>("a").unwrap();
        assert_eq!(col.as_slice(), &[3, 1, 2]);
    }

    #[test]
    fn test_empty_df_head_tail() {
        let df = DataFrame::empty();
        let h = df.head(5);
        assert_eq!(h.nrows(), 0);
        let t = df.tail(5);
        assert_eq!(t.nrows(), 0);
    }

    #[test]
    fn test_sort_by_nonexistent_column() {
        let df = sample_df();
        assert!(df.sort_by("nonexistent", true).is_err());
    }

    #[test]
    fn test_filter_empty_df() {
        let df = DataFrame::new(vec![Box::new(Series::new("a", Vec::<i32>::new()))]).unwrap();
        let filtered = df.filter(&[]).unwrap();
        assert_eq!(filtered.nrows(), 0);
    }

    #[test]
    fn test_slice_beyond_bounds() {
        let df = sample_df();
        let s = df.slice(1, 100);
        assert_eq!(s.nrows(), 2);
    }

    #[test]
    fn test_head_zero() {
        let df = sample_df();
        let h = df.head(0);
        assert_eq!(h.nrows(), 0);
    }

    #[test]
    fn test_tail_zero() {
        let df = sample_df();
        let t = df.tail(0);
        assert_eq!(t.nrows(), 0);
    }
}
