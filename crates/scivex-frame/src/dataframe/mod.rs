//! The [`DataFrame`] type — a collection of named, typed columns.

pub mod display;
pub mod filter;
pub mod join;
pub mod missing;
pub mod pivot;
pub mod select;

use crate::dtype::{DType, HasDType};
use crate::error::{FrameError, Result};
use crate::series::{AnySeries, Series};

use scivex_core::Scalar;

/// A columnar data frame: a collection of named, type-erased columns of equal
/// length.
#[derive(Debug, Clone)]
pub struct DataFrame {
    columns: Vec<Box<dyn AnySeries>>,
}

impl DataFrame {
    /// Create a `DataFrame` from pre-built columns.
    ///
    /// Returns an error if column names are duplicated or if column lengths
    /// do not match.
    pub fn new(columns: Vec<Box<dyn AnySeries>>) -> Result<Self> {
        // Check for duplicate names.
        for (i, col) in columns.iter().enumerate() {
            for other in &columns[i + 1..] {
                if col.name() == other.name() {
                    return Err(FrameError::DuplicateColumnName {
                        name: col.name().to_string(),
                    });
                }
            }
        }

        // Check that all columns have the same length.
        if let Some(first) = columns.first() {
            let expected = first.len();
            for col in &columns[1..] {
                if col.len() != expected {
                    return Err(FrameError::RowCountMismatch {
                        expected,
                        got: col.len(),
                    });
                }
            }
        }

        Ok(Self { columns })
    }

    /// Create an empty `DataFrame` with no columns.
    pub fn empty() -> Self {
        Self {
            columns: Vec::new(),
        }
    }

    /// Alias for [`new`](Self::new).
    pub fn from_series(series: Vec<Box<dyn AnySeries>>) -> Result<Self> {
        Self::new(series)
    }

    /// Start building a `DataFrame` column-by-column.
    pub fn builder() -> DataFrameBuilder {
        DataFrameBuilder::new()
    }

    // -- Column access ------------------------------------------------------

    /// Look up a column by name.
    pub fn column(&self, name: &str) -> Result<&dyn AnySeries> {
        self.columns
            .iter()
            .find(|c| c.name() == name)
            .map(AsRef::as_ref)
            .ok_or_else(|| FrameError::ColumnNotFound {
                name: name.to_string(),
            })
    }

    /// Look up a column and downcast to `Series<T>`.
    pub fn column_typed<T: Scalar + HasDType + 'static>(&self, name: &str) -> Result<&Series<T>> {
        let col = self.column(name)?;
        col.as_any()
            .downcast_ref::<Series<T>>()
            .ok_or(FrameError::TypeMismatch {
                expected: std::any::type_name::<Series<T>>(),
                got: std::any::type_name::<dyn AnySeries>(),
            })
    }

    /// Index of a column by name.
    pub fn column_index(&self, name: &str) -> Result<usize> {
        self.columns
            .iter()
            .position(|c| c.name() == name)
            .ok_or_else(|| FrameError::ColumnNotFound {
                name: name.to_string(),
            })
    }

    /// All columns.
    pub fn columns(&self) -> &[Box<dyn AnySeries>] {
        &self.columns
    }

    /// Column names in order.
    pub fn column_names(&self) -> Vec<&str> {
        self.columns.iter().map(|c| c.name()).collect()
    }

    /// Data types of all columns in order.
    pub fn dtypes(&self) -> Vec<DType> {
        self.columns.iter().map(|c| c.dtype()).collect()
    }

    // -- Shape --------------------------------------------------------------

    /// Number of rows.
    pub fn nrows(&self) -> usize {
        self.columns.first().map_or(0, |c| c.len())
    }

    /// Number of columns.
    pub fn ncols(&self) -> usize {
        self.columns.len()
    }

    /// `(nrows, ncols)`.
    pub fn shape(&self) -> (usize, usize) {
        (self.nrows(), self.ncols())
    }

    /// Whether the data frame has no columns.
    pub fn is_empty(&self) -> bool {
        self.columns.is_empty()
    }
}

// ---------------------------------------------------------------------------
// DataFrameBuilder
// ---------------------------------------------------------------------------

/// Incremental builder for [`DataFrame`].
pub struct DataFrameBuilder {
    columns: Vec<Box<dyn AnySeries>>,
}

impl DataFrameBuilder {
    fn new() -> Self {
        Self {
            columns: Vec::new(),
        }
    }

    /// Add a typed column.
    pub fn add_column<T: Scalar + HasDType + 'static>(
        mut self,
        name: impl Into<String>,
        data: Vec<T>,
    ) -> Self {
        self.columns.push(Box::new(Series::new(name, data)));
        self
    }

    /// Add a pre-built boxed column.
    pub fn add_boxed(mut self, col: Box<dyn AnySeries>) -> Self {
        self.columns.push(col);
        self
    }

    /// Finalize, validating lengths and uniqueness.
    pub fn build(self) -> Result<DataFrame> {
        DataFrame::new(self.columns)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::series::string::StringSeries;

    #[test]
    fn test_dataframe_new() {
        let df = DataFrame::new(vec![
            Box::new(Series::new("a", vec![1_i32, 2, 3])),
            Box::new(Series::new("b", vec![4.0_f64, 5.0, 6.0])),
        ])
        .unwrap();
        assert_eq!(df.nrows(), 3);
        assert_eq!(df.ncols(), 2);
        assert_eq!(df.shape(), (3, 2));
        assert_eq!(df.column_names(), vec!["a", "b"]);
    }

    #[test]
    fn test_dataframe_empty() {
        let df = DataFrame::empty();
        assert!(df.is_empty());
        assert_eq!(df.nrows(), 0);
    }

    #[test]
    fn test_dataframe_duplicate_column_name() {
        let result = DataFrame::new(vec![
            Box::new(Series::new("x", vec![1_i32])),
            Box::new(Series::new("x", vec![2_i32])),
        ]);
        assert!(matches!(
            result,
            Err(FrameError::DuplicateColumnName { .. })
        ));
    }

    #[test]
    fn test_dataframe_row_count_mismatch() {
        let result = DataFrame::new(vec![
            Box::new(Series::new("a", vec![1_i32, 2])),
            Box::new(Series::new("b", vec![3_i32])),
        ]);
        assert!(matches!(result, Err(FrameError::RowCountMismatch { .. })));
    }

    #[test]
    fn test_column_typed() {
        let df = DataFrame::new(vec![Box::new(Series::new("x", vec![1.0_f64, 2.0]))]).unwrap();
        let col = df.column_typed::<f64>("x").unwrap();
        assert_eq!(col.get(0), Some(1.0));
    }

    #[test]
    fn test_column_typed_mismatch() {
        let df = DataFrame::new(vec![Box::new(Series::new("x", vec![1_i32, 2]))]).unwrap();
        assert!(df.column_typed::<f64>("x").is_err());
    }

    #[test]
    fn test_column_not_found() {
        let df = DataFrame::empty();
        assert!(matches!(
            df.column("nope"),
            Err(FrameError::ColumnNotFound { .. })
        ));
    }

    #[test]
    fn test_dtypes() {
        let df = DataFrame::new(vec![
            Box::new(Series::new("a", vec![1_i32])),
            Box::new(StringSeries::from_strs("b", &["hi"])),
        ])
        .unwrap();
        assert_eq!(df.dtypes(), vec![DType::I32, DType::Str]);
    }

    #[test]
    fn test_builder() {
        let df = DataFrame::builder()
            .add_column("x", vec![1_i32, 2, 3])
            .add_column("y", vec![4.0_f64, 5.0, 6.0])
            .build()
            .unwrap();
        assert_eq!(df.ncols(), 2);
        assert_eq!(df.nrows(), 3);
    }
}
