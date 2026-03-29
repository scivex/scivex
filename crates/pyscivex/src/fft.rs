//! Python bindings for `scivex_core::fft`.

use pyo3::prelude::*;
use scivex_core::fft;

use crate::tensor::PyTensor;

#[allow(clippy::needless_pass_by_value)]
fn core_err(e: scivex_core::CoreError) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}

/// 1-D FFT (complex input, complex output).
#[pyfunction]
pub fn fft_1d(input: &PyTensor) -> PyResult<PyTensor> {
    let result = fft::fft(input.as_f64()?).map_err(core_err)?;
    Ok(PyTensor::from_f64(result))
}

/// 1-D inverse FFT.
#[pyfunction]
pub fn ifft_1d(input: &PyTensor) -> PyResult<PyTensor> {
    let result = fft::ifft(input.as_f64()?).map_err(core_err)?;
    Ok(PyTensor::from_f64(result))
}

/// Real-to-complex FFT.
#[pyfunction]
pub fn rfft(input: &PyTensor) -> PyResult<PyTensor> {
    let result = fft::rfft(input.as_f64()?).map_err(core_err)?;
    Ok(PyTensor::from_f64(result))
}

/// Complex-to-real inverse FFT.
#[pyfunction]
pub fn irfft(input: &PyTensor, n: usize) -> PyResult<PyTensor> {
    let result = fft::irfft(input.as_f64()?, n).map_err(core_err)?;
    Ok(PyTensor::from_f64(result))
}

/// 2-D FFT.
#[pyfunction]
pub fn fft2(input: &PyTensor) -> PyResult<PyTensor> {
    let result = fft::fft2(input.as_f64()?).map_err(core_err)?;
    Ok(PyTensor::from_f64(result))
}

/// 2-D inverse FFT.
#[pyfunction]
pub fn ifft2(input: &PyTensor) -> PyResult<PyTensor> {
    let result = fft::ifft2(input.as_f64()?).map_err(core_err)?;
    Ok(PyTensor::from_f64(result))
}

/// Register the fft submodule.
pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let m = PyModule::new(py, "fft")?;

    m.add_function(wrap_pyfunction!(fft_1d, &m)?)?;
    m.add_function(wrap_pyfunction!(ifft_1d, &m)?)?;
    m.add_function(wrap_pyfunction!(rfft, &m)?)?;
    m.add_function(wrap_pyfunction!(irfft, &m)?)?;
    m.add_function(wrap_pyfunction!(fft2, &m)?)?;
    m.add_function(wrap_pyfunction!(ifft2, &m)?)?;

    parent.add_submodule(&m)?;
    Ok(())
}
