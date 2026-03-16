//! Python bindings for descriptive statistics and correlation.

use pyo3::prelude::*;

/// Convert a `StatsError` into a Python `ValueError`.
#[allow(clippy::needless_pass_by_value)]
fn stats_err(e: scivex_stats::StatsError) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}

/// Arithmetic mean of a list of values.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub fn mean(data: Vec<f64>) -> PyResult<f64> {
    scivex_stats::mean(&data).map_err(stats_err)
}

/// Population variance of a list of values.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub fn variance(data: Vec<f64>) -> PyResult<f64> {
    scivex_stats::variance(&data).map_err(stats_err)
}

/// Population standard deviation of a list of values.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub fn std_dev(data: Vec<f64>) -> PyResult<f64> {
    scivex_stats::std_dev(&data).map_err(stats_err)
}

/// Median of a list of values.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub fn median(data: Vec<f64>) -> PyResult<f64> {
    scivex_stats::median(&data).map_err(stats_err)
}

/// Pearson correlation coefficient between two lists.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub fn pearson(x: Vec<f64>, y: Vec<f64>) -> PyResult<f64> {
    scivex_stats::pearson(&x, &y).map_err(stats_err)
}
