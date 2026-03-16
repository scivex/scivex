//! Python bindings for ML models (linear regression, k-means).

use pyo3::prelude::*;
use scivex_core::Tensor;
use scivex_ml::traits::Predictor;

use crate::tensor::PyTensor;

/// Convert an `MlError` into a Python `ValueError`.
#[allow(clippy::needless_pass_by_value)]
fn ml_err(e: scivex_ml::MlError) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}

/// Convert a `CoreError` into a Python `ValueError`.
#[allow(clippy::needless_pass_by_value)]
fn core_err(e: scivex_core::CoreError) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}

// ---------------------------------------------------------------------------
// LinearRegression
// ---------------------------------------------------------------------------

/// Ordinary least-squares linear regression.
#[pyclass(name = "LinearRegression")]
pub struct PyLinearRegression {
    inner: scivex_ml::linear::LinearRegression<f64>,
}

#[pymethods]
impl PyLinearRegression {
    /// Create a new, unfitted model.
    #[new]
    fn new() -> Self {
        Self {
            inner: scivex_ml::linear::LinearRegression::new(),
        }
    }

    /// Fit the model to training data.
    fn fit(&mut self, x: &PyTensor, y: &PyTensor) -> PyResult<()> {
        self.inner.fit(&x.inner, &y.inner).map_err(ml_err)
    }

    /// Predict target values for the given input.
    fn predict(&self, x: &PyTensor) -> PyResult<PyTensor> {
        let result = self.inner.predict(&x.inner).map_err(ml_err)?;
        Ok(PyTensor { inner: result })
    }

    /// Return the fitted weights, or `None` if unfitted.
    fn weights(&self) -> Option<Vec<f64>> {
        self.inner.weights().map(<[f64]>::to_vec)
    }

    /// Return the fitted bias (intercept), or `None` if unfitted.
    fn bias(&self) -> Option<f64> {
        self.inner.bias()
    }
}

// ---------------------------------------------------------------------------
// KMeans
// ---------------------------------------------------------------------------

/// K-Means clustering.
#[pyclass(name = "KMeans")]
pub struct PyKMeans {
    inner: scivex_ml::cluster::KMeans<f64>,
}

#[pymethods]
impl PyKMeans {
    /// Create a new K-Means model.
    #[new]
    #[pyo3(signature = (n_clusters, max_iter=100, n_init=3, seed=42))]
    fn new(n_clusters: usize, max_iter: usize, n_init: usize, seed: u64) -> PyResult<Self> {
        let inner = scivex_ml::cluster::KMeans::new(n_clusters, max_iter, 1e-6, n_init, seed)
            .map_err(ml_err)?;
        Ok(Self { inner })
    }

    /// Fit the model to data.
    fn fit(&mut self, x: &PyTensor) -> PyResult<()> {
        self.inner.fit(&x.inner).map_err(ml_err)
    }

    /// Predict cluster labels for the given data.
    fn predict(&self, x: &PyTensor) -> PyResult<PyTensor> {
        let result = self.inner.predict(&x.inner).map_err(ml_err)?;
        Ok(PyTensor { inner: result })
    }

    /// Return the inertia (sum of squared distances to centroids), or `None`.
    fn inertia(&self) -> Option<f64> {
        self.inner.inertia()
    }

    /// Return the fitted centroids as a flat tensor, or `None`.
    fn centroids(&self) -> PyResult<Option<PyTensor>> {
        match self.inner.centroids() {
            Some(data) => {
                let t = Tensor::from_vec(data.to_vec(), vec![data.len()]).map_err(core_err)?;
                Ok(Some(PyTensor { inner: t }))
            }
            None => Ok(None),
        }
    }
}
