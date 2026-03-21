//! Column selection, mutation, and reordering for [`DataFrame`].

use super::DataFrame;
use crate::error::{FrameError, Result};
use crate::series::AnySeries;

impl DataFrame {
    /// Return a new `DataFrame` with only the named columns.
    ///
    /// # Examples
    ///
    /// ```
    /// use scivex_frame::{DataFrame, Series};
    /// let df = DataFrame::builder()
    ///     .add_column("a", vec![1_i32, 2])
    ///     .add_column("b", vec![3_i32, 4])
    ///     .add_column("c", vec![5_i32, 6])
    ///     .build()
    ///     .unwrap();
    /// let sub = df.select(&["c", "a"]).unwrap();
    /// assert_eq!(sub.column_names(), vec!["c", "a"]);
    /// ```
    pub fn select(&self, names: &[&str]) -> Result<DataFrame> {
        let mut cols: Vec<Box<dyn AnySeries>> = Vec::with_capacity(names.len());
        for &name in names {
            let idx = self.column_index(name)?;
            cols.push(self.columns[idx].clone_box());
        }
        // Names are already validated as existing, but check for duplicates in
        // the requested list.
        DataFrame::new(cols)
    }

    /// Return a new `DataFrame` without the named columns.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::prelude::*;
    /// let df = DataFrame::builder()
    ///     .add_column("a", vec![1_i32, 2])
    ///     .add_column("b", vec![3_i32, 4])
    ///     .build()
    ///     .unwrap();
    /// let dropped = df.drop_columns(&["b"]).unwrap();
    /// assert_eq!(dropped.column_names(), vec!["a"]);
    /// ```
    pub fn drop_columns(&self, names: &[&str]) -> Result<DataFrame> {
        // Validate that all names exist.
        for &name in names {
            self.column_index(name)?;
        }
        let cols: Vec<Box<dyn AnySeries>> = self
            .columns
            .iter()
            .filter(|c| !names.contains(&c.name()))
            .map(|c| c.clone_box())
            .collect();
        Ok(DataFrame { columns: cols })
    }

    /// Rename a column in place.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::prelude::*;
    /// let mut df = DataFrame::builder()
    ///     .add_column("a", vec![1_i32, 2])
    ///     .build()
    ///     .unwrap();
    /// df.rename("a", "alpha").unwrap();
    /// assert_eq!(df.column_names()[0], "alpha");
    /// ```
    pub fn rename(&mut self, old: &str, new: &str) -> Result<()> {
        // Check the new name doesn't already exist (unless it's the same column).
        if old != new && self.columns.iter().any(|c| c.name() == new) {
            return Err(FrameError::DuplicateColumnName {
                name: new.to_string(),
            });
        }
        let idx = self.column_index(old)?;
        self.columns[idx] = self.columns[idx].rename_box(new);
        Ok(())
    }

    /// Add a column. Fails if the name is a duplicate or if the length
    /// doesn't match existing columns.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::prelude::*;
    /// let mut df = DataFrame::builder()
    ///     .add_column("a", vec![1_i32, 2])
    ///     .build()
    ///     .unwrap();
    /// df.add_column(Box::new(Series::new("b", vec![3_i32, 4]))).unwrap();
    /// assert_eq!(df.ncols(), 2);
    /// ```
    pub fn add_column(&mut self, col: Box<dyn AnySeries>) -> Result<()> {
        if self.columns.iter().any(|c| c.name() == col.name()) {
            return Err(FrameError::DuplicateColumnName {
                name: col.name().to_string(),
            });
        }
        if !self.columns.is_empty() && col.len() != self.nrows() {
            return Err(FrameError::RowCountMismatch {
                expected: self.nrows(),
                got: col.len(),
            });
        }
        self.columns.push(col);
        Ok(())
    }

    /// Remove and return a column by name.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::prelude::*;
    /// let mut df = DataFrame::builder()
    ///     .add_column("a", vec![1_i32, 2])
    ///     .add_column("b", vec![3_i32, 4])
    ///     .build()
    ///     .unwrap();
    /// let removed = df.remove_column("b").unwrap();
    /// assert_eq!(removed.name(), "b");
    /// assert_eq!(df.ncols(), 1);
    /// ```
    pub fn remove_column(&mut self, name: &str) -> Result<Box<dyn AnySeries>> {
        let idx = self.column_index(name)?;
        Ok(self.columns.remove(idx))
    }
}

#[cfg(test)]
mod tests {
    use crate::series::Series;

    use super::*;

    fn sample_df() -> DataFrame {
        DataFrame::new(vec![
            Box::new(Series::new("a", vec![1_i32, 2, 3])),
            Box::new(Series::new("b", vec![4_i32, 5, 6])),
            Box::new(Series::new("c", vec![7_i32, 8, 9])),
        ])
        .unwrap()
    }

    #[test]
    fn test_select() {
        let df = sample_df();
        let selected = df.select(&["c", "a"]).unwrap();
        assert_eq!(selected.column_names(), vec!["c", "a"]);
        assert_eq!(selected.ncols(), 2);
    }

    #[test]
    fn test_select_not_found() {
        let df = sample_df();
        assert!(df.select(&["z"]).is_err());
    }

    #[test]
    fn test_drop_columns() {
        let df = sample_df();
        let dropped = df.drop_columns(&["b"]).unwrap();
        assert_eq!(dropped.column_names(), vec!["a", "c"]);
    }

    #[test]
    fn test_rename() {
        let mut df = sample_df();
        df.rename("a", "alpha").unwrap();
        assert_eq!(df.column_names()[0], "alpha");
    }

    #[test]
    fn test_rename_duplicate() {
        let mut df = sample_df();
        assert!(df.rename("a", "b").is_err());
    }

    #[test]
    fn test_add_column() {
        let mut df = sample_df();
        df.add_column(Box::new(Series::new("d", vec![10_i32, 11, 12])))
            .unwrap();
        assert_eq!(df.ncols(), 4);
    }

    #[test]
    fn test_add_column_length_mismatch() {
        let mut df = sample_df();
        assert!(
            df.add_column(Box::new(Series::new("d", vec![1_i32])))
                .is_err()
        );
    }

    #[test]
    fn test_add_column_duplicate() {
        let mut df = sample_df();
        assert!(
            df.add_column(Box::new(Series::new("a", vec![1_i32, 2, 3])))
                .is_err()
        );
    }

    #[test]
    fn test_remove_column() {
        let mut df = sample_df();
        let removed = df.remove_column("b").unwrap();
        assert_eq!(removed.name(), "b");
        assert_eq!(df.ncols(), 2);
    }

    #[test]
    fn test_drop_columns_nonexistent() {
        let df = sample_df();
        assert!(df.drop_columns(&["nonexistent"]).is_err());
    }

    #[test]
    fn test_remove_column_nonexistent() {
        let mut df = sample_df();
        assert!(df.remove_column("nonexistent").is_err());
    }

    #[test]
    fn test_select_empty_list() {
        let df = sample_df();
        let selected = df.select(&[]).unwrap();
        assert_eq!(selected.ncols(), 0);
    }

    #[test]
    fn test_rename_same_name() {
        let mut df = sample_df();
        // Renaming to the same name should succeed
        df.rename("a", "a").unwrap();
        assert_eq!(df.column_names()[0], "a");
    }
}
