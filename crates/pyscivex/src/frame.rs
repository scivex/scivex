//! Python bindings for [`scivex_frame::DataFrame`].

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use scivex_frame::dataframe::join::JoinType;
use scivex_frame::groupby::AggFunc;
use scivex_frame::lazy::LazyFrame;
use scivex_frame::lazy::expr;
use scivex_frame::series::datetime::DateTimeSeries;
use scivex_frame::series::string::StringSeries;
use scivex_frame::series::window::RollingWindow;
use scivex_frame::{DataFrame, Series};

use crate::tensor::PyTensor;

/// Convert a `FrameError` into a Python `ValueError`.
#[allow(clippy::needless_pass_by_value)]
fn frame_err(e: scivex_frame::FrameError) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}

/// A columnar data frame.
#[pyclass(name = "DataFrame")]
#[derive(Clone)]
pub struct PyDataFrame {
    pub(crate) inner: DataFrame,
}

impl PyDataFrame {
    pub fn from_inner(inner: DataFrame) -> Self {
        Self { inner }
    }
}

// ---------------------------------------------------------------------------
// GroupBy result wrapper
// ---------------------------------------------------------------------------

#[pyclass(name = "GroupBy")]
struct PyGroupByResult {
    df: DataFrame,
    by: Vec<String>,
}

#[pymethods]
impl PyGroupByResult {
    fn sum(&self) -> PyResult<PyDataFrame> {
        let refs: Vec<&str> = self.by.iter().map(String::as_str).collect();
        let gb = self.df.groupby(&refs).map_err(frame_err)?;
        let result = gb.sum().map_err(frame_err)?;
        Ok(PyDataFrame { inner: result })
    }

    fn mean(&self) -> PyResult<PyDataFrame> {
        let refs: Vec<&str> = self.by.iter().map(String::as_str).collect();
        let gb = self.df.groupby(&refs).map_err(frame_err)?;
        let result = gb.mean().map_err(frame_err)?;
        Ok(PyDataFrame { inner: result })
    }

    fn min(&self) -> PyResult<PyDataFrame> {
        let refs: Vec<&str> = self.by.iter().map(String::as_str).collect();
        let gb = self.df.groupby(&refs).map_err(frame_err)?;
        let result = gb.min().map_err(frame_err)?;
        Ok(PyDataFrame { inner: result })
    }

    fn max(&self) -> PyResult<PyDataFrame> {
        let refs: Vec<&str> = self.by.iter().map(String::as_str).collect();
        let gb = self.df.groupby(&refs).map_err(frame_err)?;
        let result = gb.max().map_err(frame_err)?;
        Ok(PyDataFrame { inner: result })
    }

    fn count(&self) -> PyResult<PyDataFrame> {
        let refs: Vec<&str> = self.by.iter().map(String::as_str).collect();
        let gb = self.df.groupby(&refs).map_err(frame_err)?;
        let result = gb.count().map_err(frame_err)?;
        Ok(PyDataFrame { inner: result })
    }

    fn first(&self) -> PyResult<PyDataFrame> {
        let refs: Vec<&str> = self.by.iter().map(String::as_str).collect();
        let gb = self.df.groupby(&refs).map_err(frame_err)?;
        let result = gb.first().map_err(frame_err)?;
        Ok(PyDataFrame { inner: result })
    }

    fn last(&self) -> PyResult<PyDataFrame> {
        let refs: Vec<&str> = self.by.iter().map(String::as_str).collect();
        let gb = self.df.groupby(&refs).map_err(frame_err)?;
        let result = gb.last().map_err(frame_err)?;
        Ok(PyDataFrame { inner: result })
    }
}

#[pymethods]
#[allow(clippy::needless_pass_by_value)]
impl PyDataFrame {
    // ======================================================================
    // Constructors
    // ======================================================================

    /// Create an empty DataFrame, or from a dict of columns.
    ///
    /// Usage:
    ///   df = DataFrame()                                      # empty
    ///   df = DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]})   # from dict
    #[new]
    #[pyo3(signature = (data=None))]
    fn new(data: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        match data {
            None => Ok(Self {
                inner: DataFrame::empty(),
            }),
            Some(dict) => {
                let mut boxed_cols: Vec<Box<dyn scivex_frame::AnySeries>> = Vec::new();

                for (key, value) in dict.iter() {
                    let name: String = key.extract()?;
                    let list = value.downcast::<PyList>().map_err(|_| {
                        pyo3::exceptions::PyTypeError::new_err(
                            "DataFrame dict values must be lists",
                        )
                    })?;

                    if list.is_empty() {
                        boxed_cols.push(Box::new(Series::<f64>::new(&name, vec![])));
                        continue;
                    }

                    // Detect type from first element
                    let first = list.get_item(0)?;
                    if first.extract::<String>().is_ok() || first.extract::<&str>().is_ok() {
                        // String column
                        let mut vals = Vec::with_capacity(list.len());
                        for i in 0..list.len() {
                            let item = list.get_item(i)?;
                            let s: String = item.extract()?;
                            vals.push(s);
                        }
                        boxed_cols.push(Box::new(StringSeries::new(&name, vals)));
                    } else if first.extract::<f64>().is_ok() {
                        // Float column
                        let mut vals = Vec::with_capacity(list.len());
                        for i in 0..list.len() {
                            let item = list.get_item(i)?;
                            let v: f64 = item.extract()?;
                            vals.push(v);
                        }
                        boxed_cols.push(Box::new(Series::new(&name, vals)));
                    } else if first.extract::<i64>().is_ok() {
                        // Integer column
                        let mut vals = Vec::with_capacity(list.len());
                        for i in 0..list.len() {
                            let item = list.get_item(i)?;
                            let v: i64 = item.extract()?;
                            vals.push(v);
                        }
                        boxed_cols.push(Box::new(Series::new(&name, vals)));
                    } else if first.extract::<bool>().is_ok() {
                        // Bool → i32 column (0/1)
                        let mut vals = Vec::with_capacity(list.len());
                        for i in 0..list.len() {
                            let item = list.get_item(i)?;
                            let v: bool = item.extract()?;
                            vals.push(if v { 1_i32 } else { 0 });
                        }
                        boxed_cols.push(Box::new(Series::new(&name, vals)));
                    } else {
                        return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                            "unsupported column type for '{name}'"
                        )));
                    }
                }

                let inner = DataFrame::from_series(boxed_cols).map_err(frame_err)?;
                Ok(Self { inner })
            }
        }
    }

    // ======================================================================
    // Column operations
    // ======================================================================

    /// Add a column of f64 values.
    fn add_column(&mut self, name: String, data: Vec<f64>) -> PyResult<()> {
        let series: Box<dyn scivex_frame::AnySeries> = Box::new(Series::new(name, data));
        self.inner.add_column(series).map_err(frame_err)?;
        Ok(())
    }

    /// Add a column of string values.
    fn add_string_column(&mut self, name: String, data: Vec<String>) -> PyResult<()> {
        let series: Box<dyn scivex_frame::AnySeries> = Box::new(StringSeries::new(name, data));
        self.inner.add_column(series).map_err(frame_err)?;
        Ok(())
    }

    /// Add a column of integer values.
    fn add_int_column(&mut self, name: String, data: Vec<i64>) -> PyResult<()> {
        let series: Box<dyn scivex_frame::AnySeries> = Box::new(Series::new(name, data));
        self.inner.add_column(series).map_err(frame_err)?;
        Ok(())
    }

    // ======================================================================
    // Shape & accessors
    // ======================================================================

    fn nrows(&self) -> usize {
        self.inner.nrows()
    }

    fn ncols(&self) -> usize {
        self.inner.ncols()
    }

    fn shape(&self) -> (usize, usize) {
        self.inner.shape()
    }

    fn column_names(&self) -> Vec<String> {
        self.inner
            .column_names()
            .into_iter()
            .map(String::from)
            .collect()
    }

    fn dtypes(&self) -> Vec<String> {
        self.inner
            .dtypes()
            .into_iter()
            .map(|d| format!("{d:?}"))
            .collect()
    }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Get a column's data as a list of floats.
    fn column(&self, name: &str) -> PyResult<Vec<f64>> {
        let col = self.inner.column_typed::<f64>(name).map_err(frame_err)?;
        Ok(col.as_slice().to_vec())
    }

    /// Get a string column's data.
    fn column_str(&self, name: &str, py: Python<'_>) -> PyResult<PyObject> {
        let col = self.inner.column(name).map_err(frame_err)?;
        let nrows = col.len();
        let list = PyList::empty(py);
        for i in 0..nrows {
            list.append(col.display_value(i))?;
        }
        Ok(list.into_any().unbind())
    }

    // ======================================================================
    // Selection & filtering
    // ======================================================================

    fn select(&self, columns: Vec<String>) -> PyResult<Self> {
        let refs: Vec<&str> = columns.iter().map(String::as_str).collect();
        let selected = self.inner.select(&refs).map_err(frame_err)?;
        Ok(Self { inner: selected })
    }

    fn drop_columns(&self, columns: Vec<String>) -> PyResult<Self> {
        let refs: Vec<&str> = columns.iter().map(String::as_str).collect();
        let result = self.inner.drop_columns(&refs).map_err(frame_err)?;
        Ok(Self { inner: result })
    }

    fn head(&self, n: usize) -> Self {
        Self {
            inner: self.inner.head(n),
        }
    }

    fn tail(&self, n: usize) -> Self {
        Self {
            inner: self.inner.tail(n),
        }
    }

    fn slice(&self, offset: usize, length: usize) -> Self {
        Self {
            inner: self.inner.slice(offset, length),
        }
    }

    fn filter(&self, mask: Vec<bool>) -> PyResult<Self> {
        let result = self.inner.filter(&mask).map_err(frame_err)?;
        Ok(Self { inner: result })
    }

    // ======================================================================
    // Sorting
    // ======================================================================

    #[pyo3(signature = (by, ascending=true))]
    fn sort_values(&self, by: String, ascending: bool) -> PyResult<Self> {
        let result = self.inner.sort_by(&by, ascending).map_err(frame_err)?;
        Ok(Self { inner: result })
    }

    // ======================================================================
    // GroupBy
    // ======================================================================

    fn groupby(&self, by: Vec<String>) -> PyResult<PyGroupByResult> {
        // Validate columns exist
        let refs: Vec<&str> = by.iter().map(String::as_str).collect();
        let _gb = self.inner.groupby(&refs).map_err(frame_err)?;
        Ok(PyGroupByResult {
            df: self.inner.clone(),
            by,
        })
    }

    // ======================================================================
    // Joins
    // ======================================================================

    #[pyo3(signature = (other, on, how="inner"))]
    fn join(&self, other: &PyDataFrame, on: Vec<String>, how: &str) -> PyResult<Self> {
        let join_type = match how {
            "inner" => JoinType::Inner,
            "left" => JoinType::Left,
            "right" => JoinType::Right,
            "outer" => JoinType::Outer,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "how must be 'inner', 'left', 'right', or 'outer'",
                ));
            }
        };
        let refs: Vec<&str> = on.iter().map(String::as_str).collect();
        let result = self
            .inner
            .join(&other.inner, &refs, join_type)
            .map_err(frame_err)?;
        Ok(Self { inner: result })
    }

    // ======================================================================
    // Pivot
    // ======================================================================

    fn pivot(
        &self,
        index: Vec<String>,
        columns: String,
        values: String,
        agg_func: &str,
    ) -> PyResult<Self> {
        let agg = match agg_func {
            "sum" => AggFunc::Sum,
            "mean" => AggFunc::Mean,
            "min" => AggFunc::Min,
            "max" => AggFunc::Max,
            "count" => AggFunc::Count,
            "first" => AggFunc::First,
            "last" => AggFunc::Last,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "agg_func must be 'sum', 'mean', 'min', 'max', 'count', 'first', or 'last'",
                ));
            }
        };
        let idx_refs: Vec<&str> = index.iter().map(String::as_str).collect();
        let result = self
            .inner
            .pivot(&idx_refs, &columns, &values, agg)
            .map_err(frame_err)?;
        Ok(Self { inner: result })
    }

    // ======================================================================
    // Missing data
    // ======================================================================

    fn drop_nulls(&self) -> PyResult<Self> {
        let result = self.inner.drop_nulls().map_err(frame_err)?;
        Ok(Self { inner: result })
    }

    fn drop_nulls_subset(&self, columns: Vec<String>) -> PyResult<Self> {
        let refs: Vec<&str> = columns.iter().map(String::as_str).collect();
        let result = self.inner.drop_nulls_subset(&refs).map_err(frame_err)?;
        Ok(Self { inner: result })
    }

    // ======================================================================
    // SQL
    // ======================================================================

    fn sql(&self, query: &str) -> PyResult<Self> {
        let result = scivex_frame::sql(&self.inner, query).map_err(frame_err)?;
        Ok(Self { inner: result })
    }

    // ======================================================================
    // Tensor interop
    // ======================================================================

    /// Convert all f64 columns to a Tensor (nrows × ncols).
    fn to_tensor(&self) -> PyResult<PyTensor> {
        let nrows = self.inner.nrows();
        let names = self.inner.column_names();
        let ncols = names.len();

        let mut data = Vec::with_capacity(nrows * ncols);
        for row in 0..nrows {
            for name in &names {
                let col = self.inner.column_typed::<f64>(name).map_err(frame_err)?;
                data.push(col.as_slice()[row]);
            }
        }

        let inner = scivex_core::Tensor::from_vec(data, vec![nrows, ncols])
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(PyTensor::from_f64(inner))
    }

    // ======================================================================
    // Display
    // ======================================================================

    fn describe(&self) -> String {
        format!("{}", self.inner)
    }

    fn __repr__(&self) -> String {
        format!("{}", self.inner)
    }

    fn __len__(&self) -> usize {
        self.inner.nrows()
    }

    fn __getitem__(&self, key: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let py = key.py();
        // String key → column data
        if let Ok(name) = key.extract::<String>() {
            // Try f64 first
            if let Ok(col) = self.inner.column_typed::<f64>(&name) {
                let list = PyList::new(py, col.as_slice())?;
                return Ok(list.into_any().unbind());
            }
            // Try i64
            if let Ok(col) = self.inner.column_typed::<i64>(&name) {
                let list = PyList::new(py, col.as_slice())?;
                return Ok(list.into_any().unbind());
            }
            // Try i32
            if let Ok(col) = self.inner.column_typed::<i32>(&name) {
                let list = PyList::new(py, col.as_slice())?;
                return Ok(list.into_any().unbind());
            }
            // Fallback: display values
            let col = self.inner.column(&name).map_err(frame_err)?;
            let list = PyList::empty(py);
            for i in 0..col.len() {
                list.append(col.display_value(i))?;
            }
            return Ok(list.into_any().unbind());
        }
        // List of strings → select columns
        if let Ok(names) = key.extract::<Vec<String>>() {
            let refs: Vec<&str> = names.iter().map(String::as_str).collect();
            let selected = self.inner.select(&refs).map_err(frame_err)?;
            return Ok(Self { inner: selected }
                .into_pyobject(py)?
                .into_any()
                .unbind());
        }
        Err(pyo3::exceptions::PyTypeError::new_err(
            "DataFrame index must be a column name (str) or list of column names",
        ))
    }

    // ======================================================================
    // Rolling window operations
    // ======================================================================

    /// Compute rolling mean for a float column.
    ///
    /// Returns the rolling mean values as a list of floats.
    /// Elements with insufficient window data are returned as `None`.
    fn rolling_mean(&self, col: &str, window: usize, py: Python<'_>) -> PyResult<PyObject> {
        let series = self.inner.column_typed::<f64>(col).map_err(frame_err)?;
        let w = RollingWindow::new(window);
        let result = series.rolling_mean(&w).map_err(frame_err)?;
        series_to_py_list(&result, py)
    }

    /// Compute rolling sum for a float column.
    ///
    /// Returns the rolling sum values as a list of floats.
    /// Elements with insufficient window data are returned as `None`.
    fn rolling_sum(&self, col: &str, window: usize, py: Python<'_>) -> PyResult<PyObject> {
        let series = self.inner.column_typed::<f64>(col).map_err(frame_err)?;
        let w = RollingWindow::new(window);
        let result = series.rolling_sum(&w).map_err(frame_err)?;
        series_to_py_list(&result, py)
    }

    /// Compute rolling standard deviation (population) for a float column.
    ///
    /// Returns the rolling std values as a list of floats.
    /// Elements with insufficient window data are returned as `None`.
    fn rolling_std(&self, col: &str, window: usize, py: Python<'_>) -> PyResult<PyObject> {
        let series = self.inner.column_typed::<f64>(col).map_err(frame_err)?;
        let w = RollingWindow::new(window);
        let result = series.rolling_std(&w).map_err(frame_err)?;
        series_to_py_list(&result, py)
    }

    /// Compute rolling minimum for a float column.
    ///
    /// Returns the rolling min values as a list of floats.
    /// Elements with insufficient window data are returned as `None`.
    fn rolling_min(&self, col: &str, window: usize, py: Python<'_>) -> PyResult<PyObject> {
        let series = self.inner.column_typed::<f64>(col).map_err(frame_err)?;
        let w = RollingWindow::new(window);
        let result = series.rolling_min(&w).map_err(frame_err)?;
        series_to_py_list(&result, py)
    }

    /// Compute rolling maximum for a float column.
    ///
    /// Returns the rolling max values as a list of floats.
    /// Elements with insufficient window data are returned as `None`.
    fn rolling_max(&self, col: &str, window: usize, py: Python<'_>) -> PyResult<PyObject> {
        let series = self.inner.column_typed::<f64>(col).map_err(frame_err)?;
        let w = RollingWindow::new(window);
        let result = series.rolling_max(&w).map_err(frame_err)?;
        series_to_py_list(&result, py)
    }

    /// Compute exponentially weighted moving average for a float column.
    ///
    /// `alpha` must be in `(0, 1]`. Returns the EWM values as a list of floats.
    fn ewm_mean(&self, col: &str, alpha: f64, py: Python<'_>) -> PyResult<PyObject> {
        let series = self.inner.column_typed::<f64>(col).map_err(frame_err)?;
        let result = series.ewm_mean(alpha).map_err(frame_err)?;
        series_to_py_list(&result, py)
    }

    // ======================================================================
    // String operations
    // ======================================================================

    /// Convert all string values in a column to uppercase.
    ///
    /// Returns a new DataFrame with the column replaced by its uppercase version.
    fn str_upper(&self, col: &str) -> PyResult<Vec<String>> {
        let series = get_string_series(&self.inner, col)?;
        let result = series.to_uppercase();
        Ok(result.as_slice().to_vec())
    }

    /// Convert all string values in a column to lowercase.
    ///
    /// Returns a new DataFrame with the column replaced by its lowercase version.
    fn str_lower(&self, col: &str) -> PyResult<Vec<String>> {
        let series = get_string_series(&self.inner, col)?;
        let result = series.to_lowercase();
        Ok(result.as_slice().to_vec())
    }

    /// Boolean mask: which string values contain the given pattern.
    fn str_contains(&self, col: &str, pattern: &str) -> PyResult<Vec<bool>> {
        let series = get_string_series(&self.inner, col)?;
        Ok(series.contains(pattern))
    }

    /// Replace all occurrences of `old` with `new_val` in each string value.
    fn str_replace(&self, col: &str, old: &str, new_val: &str) -> PyResult<Vec<String>> {
        let series = get_string_series(&self.inner, col)?;
        let result = series.replace_all(old, new_val);
        Ok(result.as_slice().to_vec())
    }

    /// Character length of each string value in a column.
    fn str_len(&self, col: &str) -> PyResult<Vec<usize>> {
        let series = get_string_series(&self.inner, col)?;
        Ok(series.len_chars())
    }

    // ======================================================================
    // DateTime operations
    // ======================================================================

    /// Extract the year component from each datetime in a column.
    ///
    /// Returns a list of optional year values (`None` for null entries).
    fn dt_year(&self, col: &str, py: Python<'_>) -> PyResult<PyObject> {
        let series = get_datetime_series(&self.inner, col)?;
        let values = series.year();
        option_vec_to_py_list(py, &values)
    }

    /// Extract the month component (1-12) from each datetime in a column.
    ///
    /// Returns a list of optional month values (`None` for null entries).
    fn dt_month(&self, col: &str, py: Python<'_>) -> PyResult<PyObject> {
        let series = get_datetime_series(&self.inner, col)?;
        let values = series.month();
        option_vec_to_py_list(py, &values)
    }

    /// Extract the day-of-month component (1-31) from each datetime in a column.
    ///
    /// Returns a list of optional day values (`None` for null entries).
    fn dt_day(&self, col: &str, py: Python<'_>) -> PyResult<PyObject> {
        let series = get_datetime_series(&self.inner, col)?;
        let values = series.day();
        option_vec_to_py_list(py, &values)
    }

    /// Extract the hour component (0-23) from each datetime in a column.
    ///
    /// Returns a list of optional hour values (`None` for null entries).
    fn dt_hour(&self, col: &str, py: Python<'_>) -> PyResult<PyObject> {
        let series = get_datetime_series(&self.inner, col)?;
        let values = series.hour();
        option_vec_to_py_list(py, &values)
    }

    /// Extract the minute component (0-59) from each datetime in a column.
    ///
    /// Returns a list of optional minute values (`None` for null entries).
    fn dt_minute(&self, col: &str, py: Python<'_>) -> PyResult<PyObject> {
        let series = get_datetime_series(&self.inner, col)?;
        let values = series.minute();
        option_vec_to_py_list(py, &values)
    }

    /// Extract the second component (0-59) from each datetime in a column.
    ///
    /// Returns a list of optional second values (`None` for null entries).
    fn dt_second(&self, col: &str, py: Python<'_>) -> PyResult<PyObject> {
        let series = get_datetime_series(&self.inner, col)?;
        let values = series.second();
        option_vec_to_py_list(py, &values)
    }

    // ======================================================================
    // Lazy API
    // ======================================================================

    /// Convert this DataFrame into a LazyFrame for deferred evaluation.
    ///
    /// Operations on the LazyFrame are not executed until `collect()` is called.
    fn lazy(&self) -> PyLazyFrame {
        PyLazyFrame {
            inner: self.inner.clone().lazy(),
        }
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Convert a `Series<f64>` to a Python list, using `None` for null elements.
fn series_to_py_list(series: &Series<f64>, py: Python<'_>) -> PyResult<PyObject> {
    let list = PyList::empty(py);
    for i in 0..series.len() {
        if series.is_null_at(i) {
            list.append(py.None())?;
        } else {
            list.append(series.as_slice()[i])?;
        }
    }
    Ok(list.into_any().unbind())
}

/// Downcast a DataFrame column to `StringSeries`.
fn get_string_series<'a>(df: &'a DataFrame, col: &str) -> PyResult<&'a StringSeries> {
    let any_col = df.column(col).map_err(frame_err)?;
    any_col
        .as_any()
        .downcast_ref::<StringSeries>()
        .ok_or_else(|| {
            pyo3::exceptions::PyTypeError::new_err(format!("column '{col}' is not a string column"))
        })
}

/// Downcast a DataFrame column to `DateTimeSeries`.
fn get_datetime_series<'a>(df: &'a DataFrame, col: &str) -> PyResult<&'a DateTimeSeries> {
    let any_col = df.column(col).map_err(frame_err)?;
    any_col
        .as_any()
        .downcast_ref::<DateTimeSeries>()
        .ok_or_else(|| {
            pyo3::exceptions::PyTypeError::new_err(format!(
                "column '{col}' is not a datetime column"
            ))
        })
}

/// Convert `Vec<Option<T>>` to a Python list with `None` for missing values.
fn option_vec_to_py_list<T>(py: Python<'_>, values: &[Option<T>]) -> PyResult<PyObject>
where
    T: Copy,
    for<'a> T: IntoPyObject<'a>,
{
    let list = PyList::empty(py);
    for v in values {
        match v {
            Some(val) => list.append(*val)?,
            None => list.append(py.None())?,
        }
    }
    Ok(list.into_any().unbind())
}

// ---------------------------------------------------------------------------
// PyLazyFrame
// ---------------------------------------------------------------------------

/// A lazy wrapper around a DataFrame that defers execution.
///
/// Build a chain of operations (filter, select, groupby, sort, limit)
/// and call `collect()` to execute them all at once.
#[pyclass(name = "LazyFrame")]
#[derive(Clone)]
pub struct PyLazyFrame {
    inner: LazyFrame,
}

#[pymethods]
impl PyLazyFrame {
    /// Filter rows by a boolean predicate.
    ///
    /// `col` is the column name, `op` is one of `">"`, `"<"`, `">="`, `"<="`,
    /// `"=="`, `"!="`, and `value` is the comparison threshold (float).
    #[pyo3(signature = (col, op, value))]
    fn filter(&self, col: &str, op: &str, value: f64) -> PyResult<Self> {
        let col_expr = expr::col(col);
        let lit = expr::lit_f64(value);
        let predicate = match op {
            ">" => col_expr.gt(lit),
            "<" => col_expr.lt(lit),
            ">=" => col_expr.gt_eq(lit),
            "<=" => col_expr.lt_eq(lit),
            "==" => col_expr.eq(lit),
            "!=" => col_expr.neq(lit),
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "op must be '>', '<', '>=', '<=', '==', or '!='",
                ));
            }
        };
        Ok(Self {
            inner: self.inner.clone().filter(predicate),
        })
    }

    /// Select columns by name.
    ///
    /// Pass a list of column name strings.
    fn select(&self, columns: Vec<String>) -> Self {
        let exprs: Vec<_> = columns.iter().map(|c| expr::col(c)).collect();
        Self {
            inner: self.inner.clone().select(&exprs),
        }
    }

    /// Group by columns and aggregate.
    ///
    /// `by` is a list of grouping column names. `agg` is a dict mapping
    /// column names to aggregation functions (`"sum"`, `"mean"`, `"min"`,
    /// `"max"`, `"count"`, `"first"`, `"last"`).
    fn groupby(&self, by: Vec<String>, agg: &Bound<'_, PyDict>) -> PyResult<Self> {
        let group_refs: Vec<&str> = by.iter().map(String::as_str).collect();
        let mut agg_exprs = Vec::new();

        for (key, value) in agg.iter() {
            let col_name: String = key.extract()?;
            let func_name: String = value.extract()?;
            let col_expr = expr::col(&col_name);
            let agg_expr = match func_name.as_str() {
                "sum" => col_expr.sum(),
                "mean" => col_expr.mean(),
                "min" => col_expr.min(),
                "max" => col_expr.max(),
                "count" => col_expr.count(),
                "first" => col_expr.first(),
                "last" => col_expr.last(),
                _ => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "unknown aggregation function: '{func_name}'"
                    )));
                }
            };
            agg_exprs.push(agg_expr);
        }

        Ok(Self {
            inner: self.inner.clone().groupby_agg(&group_refs, &agg_exprs),
        })
    }

    /// Sort by a column.
    ///
    /// `ascending` defaults to `True`.
    #[pyo3(signature = (by, ascending=true))]
    fn sort(&self, by: &str, ascending: bool) -> Self {
        Self {
            inner: self.inner.clone().sort(by, ascending),
        }
    }

    /// Limit to the first `n` rows.
    fn limit(&self, n: usize) -> Self {
        Self {
            inner: self.inner.clone().limit(n),
        }
    }

    /// Execute the lazy plan and return a materialized DataFrame.
    fn collect(&self) -> PyResult<PyDataFrame> {
        let df = self.inner.clone().collect().map_err(frame_err)?;
        Ok(PyDataFrame { inner: df })
    }

    fn __repr__(&self) -> String {
        format!("LazyFrame(plan={:?})", self.inner.plan())
    }
}

// ---------------------------------------------------------------------------
// PySeries — typed f64 series
// ---------------------------------------------------------------------------

/// A named 1-D array of `f64` values with optional null tracking.
#[pyclass(name = "Series")]
#[derive(Clone)]
pub struct PySeries {
    inner: Series<f64>,
}

impl PySeries {
    pub fn from_inner(inner: Series<f64>) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PySeries {
    /// Create a new Series from a name and list of float values.
    #[new]
    fn new(name: String, data: Vec<f64>) -> Self {
        Self {
            inner: Series::new(name, data),
        }
    }

    /// Column name.
    fn name(&self) -> &str {
        self.inner.name()
    }

    /// Number of elements (including nulls).
    fn len(&self) -> usize {
        self.inner.len()
    }

    /// Whether the series is empty.
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Get a single element by index.  Returns `None` for out-of-bounds or null.
    fn get(&self, i: usize) -> Option<f64> {
        self.inner.get(i)
    }

    /// Return all values as a Python list (nulls become `None`).
    fn to_list(&self, py: Python<'_>) -> PyResult<PyObject> {
        series_to_py_list(&self.inner, py)
    }

    /// Arithmetic mean of non-null elements.
    fn mean(&self) -> f64 {
        self.inner.mean()
    }

    /// Sum of non-null elements.
    fn sum(&self) -> f64 {
        self.inner.sum()
    }

    /// Minimum of non-null elements, or `None` if empty.
    fn min(&self) -> Option<f64> {
        self.inner.min()
    }

    /// Maximum of non-null elements, or `None` if empty.
    fn max(&self) -> Option<f64> {
        self.inner.max()
    }

    /// Population standard deviation of non-null elements.
    fn std(&self) -> f64 {
        self.inner.std()
    }

    /// Population variance of non-null elements.
    fn var(&self) -> f64 {
        self.inner.var()
    }

    /// Return a sorted copy.
    #[pyo3(signature = (ascending=true))]
    fn sort(&self, ascending: bool) -> Self {
        Self {
            inner: self.inner.sort(ascending),
        }
    }

    /// Rename the series (returns a new copy).
    fn rename(&self, name: String) -> Self {
        let mut cloned = self.inner.clone();
        cloned.rename(name);
        Self { inner: cloned }
    }

    /// Data type as a string.
    fn dtype(&self) -> String {
        format!("{:?}", self.inner.dtype())
    }

    /// Number of null entries.
    fn null_count(&self) -> usize {
        self.inner.null_count()
    }

    /// Number of non-null entries.
    fn count(&self) -> usize {
        self.inner.count()
    }

    /// Compute the rolling mean with the given window size.
    ///
    /// Returns a new Series where elements with insufficient window data are null.
    fn rolling_mean(&self, window: usize) -> PyResult<Self> {
        let w = RollingWindow::new(window);
        let result = self.inner.rolling_mean(&w).map_err(frame_err)?;
        Ok(Self { inner: result })
    }

    /// Compute the rolling sum with the given window size.
    ///
    /// Returns a new Series where elements with insufficient window data are null.
    fn rolling_sum(&self, window: usize) -> PyResult<Self> {
        let w = RollingWindow::new(window);
        let result = self.inner.rolling_sum(&w).map_err(frame_err)?;
        Ok(Self { inner: result })
    }

    /// Compute the rolling standard deviation with the given window size.
    fn rolling_std(&self, window: usize) -> PyResult<Self> {
        let w = RollingWindow::new(window);
        let result = self.inner.rolling_std(&w).map_err(frame_err)?;
        Ok(Self { inner: result })
    }

    /// Compute the rolling minimum with the given window size.
    fn rolling_min(&self, window: usize) -> PyResult<Self> {
        let w = RollingWindow::new(window);
        let result = self.inner.rolling_min(&w).map_err(frame_err)?;
        Ok(Self { inner: result })
    }

    /// Compute the rolling maximum with the given window size.
    fn rolling_max(&self, window: usize) -> PyResult<Self> {
        let w = RollingWindow::new(window);
        let result = self.inner.rolling_max(&w).map_err(frame_err)?;
        Ok(Self { inner: result })
    }

    fn __repr__(&self) -> String {
        format!(
            "Series(name='{}', len={}, dtype={:?})",
            self.inner.name(),
            self.inner.len(),
            self.inner.dtype()
        )
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __getitem__(&self, index: usize) -> PyResult<Option<f64>> {
        if index >= self.inner.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "index out of range",
            ));
        }
        Ok(self.inner.get(index))
    }
}
