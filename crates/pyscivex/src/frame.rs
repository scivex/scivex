//! Python bindings for [`scivex_frame::DataFrame`].

use pyo3::prelude::*;
use scivex_frame::{DataFrame, Series};

/// Convert a `FrameError` into a Python `ValueError`.
#[allow(clippy::needless_pass_by_value)]
fn frame_err(e: scivex_frame::FrameError) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}

/// A columnar data frame.
#[pyclass(name = "DataFrame")]
#[derive(Clone)]
pub struct PyDataFrame {
    inner: DataFrame,
}

#[pymethods]
#[allow(clippy::needless_pass_by_value)]
impl PyDataFrame {
    /// Create an empty data frame.
    #[new]
    fn new() -> Self {
        Self {
            inner: DataFrame::empty(),
        }
    }

    /// Add a column of `f64` values.
    fn add_column(&mut self, name: String, data: Vec<f64>) -> PyResult<()> {
        let series: Box<dyn scivex_frame::AnySeries> = Box::new(Series::new(name, data));
        self.inner.add_column(series).map_err(frame_err)?;
        Ok(())
    }

    /// Number of rows.
    fn nrows(&self) -> usize {
        self.inner.nrows()
    }

    /// Number of columns.
    fn ncols(&self) -> usize {
        self.inner.ncols()
    }

    /// Column names.
    fn column_names(&self) -> Vec<String> {
        self.inner
            .column_names()
            .into_iter()
            .map(String::from)
            .collect()
    }

    /// Get a column's data as a list of floats.
    ///
    /// Attempts to downcast the column to `Series<f64>`. Returns an error
    /// if the column is not found or is not of type `f64`.
    fn column(&self, name: &str) -> PyResult<Vec<f64>> {
        let col = self.inner.column_typed::<f64>(name).map_err(frame_err)?;
        Ok(col.as_slice().to_vec())
    }

    /// Return the first `n` rows as a new data frame.
    fn head(&self, n: usize) -> Self {
        Self {
            inner: self.inner.head(n),
        }
    }

    /// Select a subset of columns by name.
    fn select(&self, columns: Vec<String>) -> PyResult<Self> {
        let refs: Vec<&str> = columns.iter().map(String::as_str).collect();
        let selected = self.inner.select(&refs).map_err(frame_err)?;
        Ok(Self { inner: selected })
    }

    /// Summary string (uses the Display implementation).
    fn describe(&self) -> String {
        format!("{}", self.inner)
    }

    fn __repr__(&self) -> String {
        format!("{}", self.inner)
    }

    fn __len__(&self) -> usize {
        self.inner.nrows()
    }
}
