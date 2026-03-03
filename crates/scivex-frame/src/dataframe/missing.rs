//! Missing data handling for [`DataFrame`].

use crate::dataframe::DataFrame;
use crate::error::Result;

impl DataFrame {
    /// Drop rows where any column has a null value.
    pub fn drop_nulls(&self) -> Result<DataFrame> {
        let nrows = self.nrows();
        if nrows == 0 {
            return Ok(self.clone());
        }

        let mask: Vec<bool> = (0..nrows)
            .map(|row| self.columns.iter().all(|col| !col.is_null(row)))
            .collect();

        self.filter(&mask)
    }

    /// Drop rows where any of the specified columns has a null value.
    pub fn drop_nulls_subset(&self, cols: &[&str]) -> Result<DataFrame> {
        // Validate columns exist.
        let col_refs: Vec<&dyn crate::series::AnySeries> = cols
            .iter()
            .map(|&name| self.column(name))
            .collect::<Result<Vec<_>>>()?;

        let nrows = self.nrows();
        if nrows == 0 {
            return Ok(self.clone());
        }

        let mask: Vec<bool> = (0..nrows)
            .map(|row| col_refs.iter().all(|col| !col.is_null(row)))
            .collect();

        self.filter(&mask)
    }

    /// Count of null values per column.
    pub fn null_count_per_column(&self) -> Vec<(&str, usize)> {
        self.columns
            .iter()
            .map(|col| (col.name(), col.null_count()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::series::Series;

    #[test]
    fn test_drop_nulls() {
        let s1 = Series::with_nulls("a", vec![1_i32, 0, 3], vec![false, true, false]).unwrap();
        let s2 = Series::new("b", vec![10_i32, 20, 30]);
        let df = DataFrame::new(vec![Box::new(s1), Box::new(s2)]).unwrap();
        let result = df.drop_nulls().unwrap();
        assert_eq!(result.nrows(), 2);
    }

    #[test]
    fn test_drop_nulls_subset() {
        let s1 = Series::with_nulls("a", vec![1_i32, 0, 3], vec![false, true, false]).unwrap();
        let s2 = Series::with_nulls("b", vec![10_i32, 20, 0], vec![false, false, true]).unwrap();
        let df = DataFrame::new(vec![Box::new(s1), Box::new(s2)]).unwrap();

        // Only drop where "a" is null
        let result = df.drop_nulls_subset(&["a"]).unwrap();
        assert_eq!(result.nrows(), 2);

        // Drop where any of a or b is null
        let result2 = df.drop_nulls_subset(&["a", "b"]).unwrap();
        assert_eq!(result2.nrows(), 1);
    }

    #[test]
    fn test_null_count_per_column() {
        let s1 = Series::with_nulls("a", vec![1_i32, 0, 3], vec![false, true, false]).unwrap();
        let s2 = Series::new("b", vec![10_i32, 20, 30]);
        let df = DataFrame::new(vec![Box::new(s1), Box::new(s2)]).unwrap();
        let counts = df.null_count_per_column();
        assert_eq!(counts, vec![("a", 1), ("b", 0)]);
    }
}
