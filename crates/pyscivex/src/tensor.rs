//! Python bindings for [`scivex_core::Tensor<f64>`].

use pyo3::prelude::*;
use scivex_core::Tensor;

/// Convert a `CoreError` into a Python `ValueError`.
#[allow(clippy::needless_pass_by_value)]
fn core_err(e: scivex_core::CoreError) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}

/// A multidimensional tensor of `f64` values.
#[pyclass(name = "Tensor")]
#[derive(Clone)]
pub struct PyTensor {
    pub(crate) inner: Tensor<f64>,
}

#[pymethods]
#[allow(clippy::needless_pass_by_value)]
impl PyTensor {
    /// Create a tensor from flat data and a shape.
    #[new]
    fn new(data: Vec<f64>, shape: Vec<usize>) -> PyResult<Self> {
        let inner = Tensor::from_vec(data, shape).map_err(core_err)?;
        Ok(Self { inner })
    }

    /// Create a tensor filled with zeros.
    #[staticmethod]
    fn zeros(shape: Vec<usize>) -> Self {
        Self {
            inner: Tensor::zeros(shape),
        }
    }

    /// Create a tensor filled with ones.
    #[staticmethod]
    fn ones(shape: Vec<usize>) -> Self {
        Self {
            inner: Tensor::ones(shape),
        }
    }

    /// Create an identity matrix of size `n x n`.
    #[staticmethod]
    fn eye(n: usize) -> Self {
        Self {
            inner: Tensor::eye(n),
        }
    }

    /// Return the shape as a list.
    fn shape(&self) -> Vec<usize> {
        self.inner.shape().to_vec()
    }

    /// Return the number of dimensions.
    fn ndim(&self) -> usize {
        self.inner.ndim()
    }

    /// Return the total number of elements.
    fn numel(&self) -> usize {
        self.inner.numel()
    }

    /// Return a flat list of all elements.
    fn to_list(&self) -> Vec<f64> {
        self.inner.as_slice().to_vec()
    }

    /// Element-wise addition.
    fn __add__(&self, other: &PyTensor) -> PyResult<Self> {
        let result = self
            .inner
            .zip_map(&other.inner, |a, b| a + b)
            .map_err(core_err)?;
        Ok(Self { inner: result })
    }

    /// Element-wise subtraction.
    fn __sub__(&self, other: &PyTensor) -> PyResult<Self> {
        let result = self
            .inner
            .zip_map(&other.inner, |a, b| a - b)
            .map_err(core_err)?;
        Ok(Self { inner: result })
    }

    /// Matrix multiplication.
    fn __matmul__(&self, other: &PyTensor) -> PyResult<Self> {
        let result = self.inner.matmul(&other.inner).map_err(core_err)?;
        Ok(Self { inner: result })
    }

    /// Sum of all elements.
    fn sum(&self) -> f64 {
        self.inner.sum()
    }

    /// Mean of all elements.
    fn mean(&self) -> f64 {
        self.inner.mean()
    }

    /// Transpose a 2-D tensor.
    fn transpose(&self) -> PyResult<Self> {
        let result = self.inner.transpose().map_err(core_err)?;
        Ok(Self { inner: result })
    }

    /// Reshape to a new shape (returns a new tensor).
    fn reshape(&self, shape: Vec<usize>) -> PyResult<Self> {
        let result = self.inner.reshaped(shape).map_err(core_err)?;
        Ok(Self { inner: result })
    }

    /// Dot product with another 1-D tensor.
    fn dot(&self, other: &PyTensor) -> PyResult<f64> {
        self.inner.dot(&other.inner).map_err(core_err)
    }

    fn __repr__(&self) -> String {
        format!("{}", self.inner)
    }
}
