//! Python bindings for scivex-gpu — GPU-accelerated tensor operations.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use scivex_gpu::{GpuBackend, GpuDevice, GpuTensor, ops};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn py_err(e: impl std::fmt::Display) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}

// ---------------------------------------------------------------------------
// PyGpuDevice
// ---------------------------------------------------------------------------

/// GPU device context for accelerated computation.
#[pyclass(name = "Device", unsendable)]
pub struct PyGpuDevice {
    inner: GpuDevice,
}

#[pymethods]
impl PyGpuDevice {
    #[new]
    fn new() -> PyResult<Self> {
        let inner = GpuDevice::new().map_err(py_err)?;
        Ok(Self { inner })
    }

    fn info(&self) -> PyResult<PyObject> {
        let info = self.inner.info();
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            dict.set_item("name", &info.name)?;
            dict.set_item("backend", &info.backend)?;
            dict.set_item("device_type", &info.device_type)?;
            Ok(dict.into_any().unbind())
        })
    }

    fn __repr__(&self) -> String {
        let info = self.inner.info();
        format!("Device({})", info)
    }
}

// ---------------------------------------------------------------------------
// PyGpuTensor
// ---------------------------------------------------------------------------

/// GPU-resident tensor (f32). Transfer data to/from CPU via to_list().
#[pyclass(name = "GpuTensor", unsendable)]
pub struct PyGpuTensor {
    inner: GpuTensor,
}

impl PyGpuTensor {
    fn wrap(t: GpuTensor) -> Self {
        Self { inner: t }
    }
}

#[pymethods]
impl PyGpuTensor {
    /// Create a GPU tensor from flat data and shape.
    #[new]
    fn new(device: &PyGpuDevice, data: Vec<f32>, shape: Vec<usize>) -> PyResult<Self> {
        let t = GpuTensor::from_slice(&device.inner, &data, shape).map_err(py_err)?;
        Ok(Self::wrap(t))
    }

    /// Create a GPU tensor of zeros.
    #[staticmethod]
    fn zeros(device: &PyGpuDevice, shape: Vec<usize>) -> Self {
        Self::wrap(GpuTensor::zeros(&device.inner, shape))
    }

    /// Transfer back to CPU as a flat list.
    fn to_list(&self) -> PyResult<Vec<f32>> {
        let tensor = self.inner.to_tensor().map_err(py_err)?;
        Ok(tensor.as_slice().to_vec())
    }

    fn shape(&self) -> Vec<usize> {
        self.inner.shape().to_vec()
    }

    fn ndim(&self) -> usize {
        self.inner.ndim()
    }

    fn numel(&self) -> usize {
        self.inner.numel()
    }

    fn __repr__(&self) -> String {
        format!(
            "GpuTensor(shape={:?}, numel={})",
            self.inner.shape(),
            self.inner.numel()
        )
    }
}

// ---------------------------------------------------------------------------
// Element-wise operations
// ---------------------------------------------------------------------------

/// Element-wise addition of two GPU tensors.
#[pyfunction]
fn add(a: &PyGpuTensor, b: &PyGpuTensor) -> PyResult<PyGpuTensor> {
    let r = ops::add(&a.inner, &b.inner).map_err(py_err)?;
    Ok(PyGpuTensor::wrap(r))
}

/// Element-wise subtraction of two GPU tensors.
#[pyfunction]
fn sub(a: &PyGpuTensor, b: &PyGpuTensor) -> PyResult<PyGpuTensor> {
    let r = ops::sub(&a.inner, &b.inner).map_err(py_err)?;
    Ok(PyGpuTensor::wrap(r))
}

/// Element-wise multiplication of two GPU tensors.
#[pyfunction]
fn mul(a: &PyGpuTensor, b: &PyGpuTensor) -> PyResult<PyGpuTensor> {
    let r = ops::mul(&a.inner, &b.inner).map_err(py_err)?;
    Ok(PyGpuTensor::wrap(r))
}

/// Element-wise division of two GPU tensors.
#[pyfunction]
fn div(a: &PyGpuTensor, b: &PyGpuTensor) -> PyResult<PyGpuTensor> {
    let r = ops::div(&a.inner, &b.inner).map_err(py_err)?;
    Ok(PyGpuTensor::wrap(r))
}

/// Add a scalar to every element of a GPU tensor.
#[pyfunction]
fn add_scalar(input: &PyGpuTensor, scalar: f32) -> PyResult<PyGpuTensor> {
    let r = ops::add_scalar(&input.inner, scalar).map_err(py_err)?;
    Ok(PyGpuTensor::wrap(r))
}

/// Multiply every element of a GPU tensor by a scalar.
#[pyfunction]
fn mul_scalar(input: &PyGpuTensor, scalar: f32) -> PyResult<PyGpuTensor> {
    let r = ops::mul_scalar(&input.inner, scalar).map_err(py_err)?;
    Ok(PyGpuTensor::wrap(r))
}

/// Subtract a scalar from every element of a GPU tensor.
#[pyfunction]
fn sub_scalar(input: &PyGpuTensor, scalar: f32) -> PyResult<PyGpuTensor> {
    let r = ops::sub_scalar(&input.inner, scalar).map_err(py_err)?;
    Ok(PyGpuTensor::wrap(r))
}

// ---------------------------------------------------------------------------
// Linear algebra
// ---------------------------------------------------------------------------

/// Matrix multiplication on GPU: [m,k] @ [k,n] -> [m,n].
#[pyfunction]
fn matmul(a: &PyGpuTensor, b: &PyGpuTensor) -> PyResult<PyGpuTensor> {
    let r = ops::matmul(&a.inner, &b.inner).map_err(py_err)?;
    Ok(PyGpuTensor::wrap(r))
}

/// Transpose a 2D GPU tensor.
#[pyfunction]
fn transpose(input: &PyGpuTensor) -> PyResult<PyGpuTensor> {
    let r = ops::transpose(&input.inner).map_err(py_err)?;
    Ok(PyGpuTensor::wrap(r))
}

// ---------------------------------------------------------------------------
// Reductions
// ---------------------------------------------------------------------------

/// Sum all elements of a GPU tensor.
#[pyfunction]
fn sum(input: &PyGpuTensor) -> PyResult<f32> {
    ops::sum(&input.inner).map_err(py_err)
}

/// Mean of all elements of a GPU tensor.
#[pyfunction]
fn mean(input: &PyGpuTensor) -> PyResult<f32> {
    ops::mean(&input.inner).map_err(py_err)
}

// ---------------------------------------------------------------------------
// Activations & unary
// ---------------------------------------------------------------------------

/// Element-wise ReLU activation on GPU.
#[pyfunction]
fn relu(input: &PyGpuTensor) -> PyResult<PyGpuTensor> {
    let r = ops::relu(&input.inner).map_err(py_err)?;
    Ok(PyGpuTensor::wrap(r))
}

/// Element-wise sigmoid activation on GPU.
#[pyfunction]
fn sigmoid(input: &PyGpuTensor) -> PyResult<PyGpuTensor> {
    let r = ops::sigmoid(&input.inner).map_err(py_err)?;
    Ok(PyGpuTensor::wrap(r))
}

/// Element-wise tanh activation on GPU.
#[pyfunction]
fn tanh(input: &PyGpuTensor) -> PyResult<PyGpuTensor> {
    let r = ops::tanh_op(&input.inner).map_err(py_err)?;
    Ok(PyGpuTensor::wrap(r))
}

/// Element-wise exponential on GPU.
#[pyfunction]
fn exp(input: &PyGpuTensor) -> PyResult<PyGpuTensor> {
    let r = ops::exp(&input.inner).map_err(py_err)?;
    Ok(PyGpuTensor::wrap(r))
}

/// Element-wise natural logarithm on GPU.
#[pyfunction]
fn log(input: &PyGpuTensor) -> PyResult<PyGpuTensor> {
    let r = ops::log(&input.inner).map_err(py_err)?;
    Ok(PyGpuTensor::wrap(r))
}

/// Negate every element of a GPU tensor.
#[pyfunction]
fn negate(input: &PyGpuTensor) -> PyResult<PyGpuTensor> {
    let r = ops::negate(&input.inner).map_err(py_err)?;
    Ok(PyGpuTensor::wrap(r))
}

/// Create a GPU tensor filled with a constant value.
#[pyfunction]
fn fill(device: &PyGpuDevice, shape: Vec<usize>, value: f32) -> PyResult<PyGpuTensor> {
    let r = ops::fill(&device.inner, shape, value).map_err(py_err)?;
    Ok(PyGpuTensor::wrap(r))
}

// ---------------------------------------------------------------------------
// Backend detection
// ---------------------------------------------------------------------------

/// Detect the best available GPU backend (e.g. "Metal", "Vulkan", "DX12").
#[pyfunction]
fn detect_backend() -> String {
    GpuBackend::detect().name().to_string()
}

// ---------------------------------------------------------------------------
// GPU Optimizers
// ---------------------------------------------------------------------------

/// GPU on-device SGD optimizer with optional momentum and weight decay.
///
/// All parameter updates run entirely on the GPU via WGSL compute shaders.
#[pyclass(name = "GpuSGD", unsendable)]
pub struct PyGpuSGD {
    inner: scivex_gpu::GpuSGD,
}

#[pymethods]
impl PyGpuSGD {
    /// Create a new GPU SGD optimizer.
    ///
    /// # Arguments
    ///
    /// * `device` — The GPU device to run on.
    /// * `n_params` — Number of parameter groups.
    /// * `lr` — Learning rate.
    /// * `momentum` — Momentum coefficient (default 0.0).
    /// * `weight_decay` — Weight decay / L2 penalty (default 0.0).
    #[new]
    #[pyo3(signature = (device, n_params, lr, momentum=0.0, weight_decay=0.0))]
    fn new(
        device: &PyGpuDevice,
        n_params: usize,
        lr: f32,
        momentum: f32,
        weight_decay: f32,
    ) -> Self {
        let inner = scivex_gpu::GpuSGD::new(&device.inner, n_params, lr)
            .with_momentum(momentum)
            .with_weight_decay(weight_decay);
        Self { inner }
    }

    /// Perform one SGD step — updates `params` in-place on the GPU.
    ///
    /// `params` and `grads` must be lists of `GpuTensor` of the same length.
    fn step(
        &mut self,
        py: Python<'_>,
        params: Vec<Py<PyGpuTensor>>,
        grads: Vec<Py<PyGpuTensor>>,
    ) -> PyResult<()> {
        let mut param_tensors: Vec<GpuTensor> =
            params.iter().map(|p| p.borrow(py).inner.clone()).collect();
        let grad_tensors: Vec<GpuTensor> =
            grads.iter().map(|g| g.borrow(py).inner.clone()).collect();
        self.inner
            .step(&mut param_tensors, &grad_tensors)
            .map_err(py_err)?;
        for (py_p, updated) in params.iter().zip(param_tensors.into_iter()) {
            py_p.borrow_mut(py).inner = updated;
        }
        Ok(())
    }

    fn __repr__(&self) -> &'static str {
        "GpuSGD()"
    }
}

/// GPU on-device Adam optimizer — all updates run on the GPU.
///
/// Implements Adam with bias correction via WGSL compute shaders.
#[pyclass(name = "GpuAdam", unsendable)]
pub struct PyGpuAdam {
    inner: scivex_gpu::GpuAdam,
}

#[pymethods]
impl PyGpuAdam {
    /// Create a new GPU Adam optimizer.
    ///
    /// # Arguments
    ///
    /// * `device` — The GPU device to run on.
    /// * `n_params` — Number of parameter groups.
    /// * `lr` — Learning rate.
    /// * `beta1` — First moment decay (default 0.9).
    /// * `beta2` — Second moment decay (default 0.999).
    /// * `eps` — Epsilon for numerical stability (default 1e-8).
    #[new]
    #[pyo3(signature = (device, n_params, lr, beta1=0.9, beta2=0.999, eps=1e-8))]
    fn new(
        device: &PyGpuDevice,
        n_params: usize,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
    ) -> Self {
        let inner = scivex_gpu::GpuAdam::new(&device.inner, n_params, lr)
            .with_beta1(beta1)
            .with_beta2(beta2)
            .with_eps(eps);
        Self { inner }
    }

    /// Perform one Adam step — updates `params` in-place on the GPU.
    ///
    /// `params` and `grads` must be lists of `GpuTensor` of the same length.
    fn step(
        &mut self,
        py: Python<'_>,
        params: Vec<Py<PyGpuTensor>>,
        grads: Vec<Py<PyGpuTensor>>,
    ) -> PyResult<()> {
        let mut param_tensors: Vec<GpuTensor> =
            params.iter().map(|p| p.borrow(py).inner.clone()).collect();
        let grad_tensors: Vec<GpuTensor> =
            grads.iter().map(|g| g.borrow(py).inner.clone()).collect();
        self.inner
            .step(&mut param_tensors, &grad_tensors)
            .map_err(py_err)?;
        for (py_p, updated) in params.iter().zip(param_tensors.into_iter()) {
            py_p.borrow_mut(py).inner = updated;
        }
        Ok(())
    }

    fn __repr__(&self) -> &'static str {
        "GpuAdam()"
    }
}

// ---------------------------------------------------------------------------
// Register submodule
// ---------------------------------------------------------------------------

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "gpu")?;

    // Core
    m.add_class::<PyGpuDevice>()?;
    m.add_class::<PyGpuTensor>()?;

    // Element-wise
    m.add_function(wrap_pyfunction!(add, &m)?)?;
    m.add_function(wrap_pyfunction!(sub, &m)?)?;
    m.add_function(wrap_pyfunction!(mul, &m)?)?;
    m.add_function(wrap_pyfunction!(div, &m)?)?;
    m.add_function(wrap_pyfunction!(add_scalar, &m)?)?;
    m.add_function(wrap_pyfunction!(mul_scalar, &m)?)?;
    m.add_function(wrap_pyfunction!(sub_scalar, &m)?)?;

    // Linear algebra
    m.add_function(wrap_pyfunction!(matmul, &m)?)?;
    m.add_function(wrap_pyfunction!(transpose, &m)?)?;

    // Reductions
    m.add_function(wrap_pyfunction!(sum, &m)?)?;
    m.add_function(wrap_pyfunction!(mean, &m)?)?;

    // Activations
    m.add_function(wrap_pyfunction!(relu, &m)?)?;
    m.add_function(wrap_pyfunction!(sigmoid, &m)?)?;
    m.add_function(wrap_pyfunction!(tanh, &m)?)?;
    m.add_function(wrap_pyfunction!(exp, &m)?)?;
    m.add_function(wrap_pyfunction!(log, &m)?)?;
    m.add_function(wrap_pyfunction!(negate, &m)?)?;
    m.add_function(wrap_pyfunction!(fill, &m)?)?;

    // Backend
    m.add_function(wrap_pyfunction!(detect_backend, &m)?)?;

    // GPU Optimizers
    m.add_class::<PyGpuSGD>()?;
    m.add_class::<PyGpuAdam>()?;

    parent.add_submodule(&m)?;
    Ok(())
}
