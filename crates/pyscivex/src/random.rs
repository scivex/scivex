//! Python bindings for `scivex_core::random`.

use pyo3::prelude::*;
use scivex_core::random;

use crate::tensor::PyTensor;

#[allow(clippy::needless_pass_by_value)]
fn core_err(e: scivex_core::CoreError) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}

/// Pseudo-random number generator (xoshiro256**).
#[pyclass(name = "Rng")]
pub struct PyRng {
    inner: random::Rng,
}

#[pymethods]
impl PyRng {
    #[new]
    fn new(seed: u64) -> Self {
        Self {
            inner: random::Rng::new(seed),
        }
    }

    /// Re-seed the generator.
    fn seed(&mut self, seed: u64) {
        self.inner.seed(seed);
    }

    /// Next uniform f64 in [0, 1).
    fn next_f64(&mut self) -> f64 {
        self.inner.next_f64()
    }

    /// Next standard normal (Gaussian) f64.
    fn next_normal_f64(&mut self) -> f64 {
        self.inner.next_normal_f64()
    }

    /// Next random u64.
    fn next_u64(&mut self) -> u64 {
        self.inner.next_u64()
    }

    /// Uniform tensor in [0, 1).
    fn uniform(&mut self, shape: Vec<usize>) -> PyTensor {
        PyTensor::from_f64(random::uniform(&mut self.inner, shape))
    }

    /// Uniform tensor in [low, high).
    fn uniform_range(&mut self, shape: Vec<usize>, low: f64, high: f64) -> PyResult<PyTensor> {
        let t = random::uniform_range(&mut self.inner, shape, low, high).map_err(core_err)?;
        Ok(PyTensor::from_f64(t))
    }

    /// Normal (Gaussian) tensor.
    fn normal(&mut self, shape: Vec<usize>, mean: f64, std_dev: f64) -> PyTensor {
        PyTensor::from_f64(random::normal(&mut self.inner, shape, mean, std_dev))
    }

    /// Standard normal tensor (mean=0, std=1).
    fn standard_normal(&mut self, shape: Vec<usize>) -> PyTensor {
        PyTensor::from_f64(random::standard_normal(&mut self.inner, shape))
    }

    /// Random integer tensor in [low, high).
    fn randint(&mut self, shape: Vec<usize>, low: i64, high: i64) -> PyResult<PyTensor> {
        let inner = random::randint::<i64>(&mut self.inner, shape, low, high).map_err(core_err)?;
        // Convert i64 tensor to f64 tensor for Python compatibility
        let data: Vec<f64> = inner.as_slice().iter().map(|&v| v as f64).collect();
        let t = scivex_core::Tensor::from_vec(data, inner.shape().to_vec()).map_err(core_err)?;
        Ok(PyTensor::from_f64(t))
    }
}

/// Register the random submodule.
pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let m = PyModule::new(py, "random")?;
    m.add_class::<PyRng>()?;
    parent.add_submodule(&m)?;
    Ok(())
}
