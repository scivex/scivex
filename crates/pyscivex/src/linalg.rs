//! Python bindings for `scivex_core::linalg` — decompositions and BLAS.

use pyo3::prelude::*;
use scivex_core::linalg::decomp::{
    cholesky::CholeskyDecomposition, eig::EigDecomposition, lu::LuDecomposition,
    qr::QrDecomposition, svd::SvdDecomposition,
};

use crate::tensor::PyTensor;

#[allow(clippy::needless_pass_by_value)]
fn core_err(e: scivex_core::CoreError) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}

// ---------------------------------------------------------------------------
// LU Decomposition
// ---------------------------------------------------------------------------

#[pyclass(name = "LU")]
pub struct PyLU {
    inner: LuDecomposition<f64>,
}

#[pymethods]
impl PyLU {
    #[staticmethod]
    fn decompose(a: &PyTensor) -> PyResult<Self> {
        let inner = LuDecomposition::decompose(a.as_f64()?).map_err(core_err)?;
        Ok(Self { inner })
    }

    fn l(&self) -> PyTensor {
        PyTensor::from_f64(self.inner.l())
    }

    fn u(&self) -> PyTensor {
        PyTensor::from_f64(self.inner.u())
    }

    fn p(&self) -> PyTensor {
        PyTensor::from_f64(self.inner.p())
    }

    fn det(&self) -> f64 {
        self.inner.det()
    }

    fn solve(&self, b: &PyTensor) -> PyResult<PyTensor> {
        let result = self.inner.solve(b.as_f64()?).map_err(core_err)?;
        Ok(PyTensor::from_f64(result))
    }

    fn inverse(&self) -> PyResult<PyTensor> {
        let result = self.inner.inverse().map_err(core_err)?;
        Ok(PyTensor::from_f64(result))
    }
}

// ---------------------------------------------------------------------------
// QR Decomposition
// ---------------------------------------------------------------------------

#[pyclass(name = "QR")]
pub struct PyQR {
    inner: QrDecomposition<f64>,
}

#[pymethods]
impl PyQR {
    #[staticmethod]
    fn decompose(a: &PyTensor) -> PyResult<Self> {
        let inner = QrDecomposition::decompose(a.as_f64()?).map_err(core_err)?;
        Ok(Self { inner })
    }

    fn q(&self) -> PyTensor {
        PyTensor::from_f64(self.inner.q())
    }

    fn r(&self) -> PyTensor {
        PyTensor::from_f64(self.inner.r())
    }

    fn is_full_rank(&self) -> bool {
        self.inner.is_full_rank()
    }

    fn solve(&self, b: &PyTensor) -> PyResult<PyTensor> {
        let result = self.inner.solve(b.as_f64()?).map_err(core_err)?;
        Ok(PyTensor::from_f64(result))
    }
}

// ---------------------------------------------------------------------------
// SVD Decomposition
// ---------------------------------------------------------------------------

#[pyclass(name = "SVD")]
pub struct PySVD {
    inner: SvdDecomposition<f64>,
}

#[pymethods]
impl PySVD {
    #[staticmethod]
    fn decompose(a: &PyTensor) -> PyResult<Self> {
        let inner = SvdDecomposition::decompose(a.as_f64()?).map_err(core_err)?;
        Ok(Self { inner })
    }

    fn singular_values(&self) -> Vec<f64> {
        self.inner.singular_values().to_vec()
    }

    fn u(&self) -> PyTensor {
        PyTensor::from_f64(self.inner.u())
    }

    fn vt(&self) -> PyTensor {
        PyTensor::from_f64(self.inner.vt())
    }

    fn s(&self) -> PyTensor {
        PyTensor::from_f64(self.inner.s())
    }

    fn rank(&self, tol: f64) -> usize {
        self.inner.rank(tol)
    }

    fn condition_number(&self) -> f64 {
        self.inner.condition_number()
    }
}

// ---------------------------------------------------------------------------
// Eigenvalue Decomposition (symmetric)
// ---------------------------------------------------------------------------

#[pyclass(name = "Eig")]
pub struct PyEig {
    inner: EigDecomposition<f64>,
}

#[pymethods]
impl PyEig {
    #[staticmethod]
    fn decompose_symmetric(a: &PyTensor) -> PyResult<Self> {
        let inner = EigDecomposition::decompose_symmetric(a.as_f64()?).map_err(core_err)?;
        Ok(Self { inner })
    }

    fn eigenvalues(&self) -> Vec<f64> {
        self.inner.eigenvalues().to_vec()
    }

    fn eigenvectors(&self) -> PyTensor {
        PyTensor::from_f64(self.inner.eigenvectors())
    }
}

// ---------------------------------------------------------------------------
// Cholesky Decomposition
// ---------------------------------------------------------------------------

#[pyclass(name = "Cholesky")]
pub struct PyCholesky {
    inner: CholeskyDecomposition<f64>,
}

#[pymethods]
impl PyCholesky {
    #[staticmethod]
    fn decompose(a: &PyTensor) -> PyResult<Self> {
        let inner = CholeskyDecomposition::decompose(a.as_f64()?).map_err(core_err)?;
        Ok(Self { inner })
    }

    fn l(&self) -> PyTensor {
        PyTensor::from_f64(self.inner.l())
    }

    fn solve(&self, b: &PyTensor) -> PyResult<PyTensor> {
        let result = self.inner.solve(b.as_f64()?).map_err(core_err)?;
        Ok(PyTensor::from_f64(result))
    }

    fn inverse(&self) -> PyResult<PyTensor> {
        let result = self.inner.inverse().map_err(core_err)?;
        Ok(PyTensor::from_f64(result))
    }

    fn log_det(&self) -> f64 {
        self.inner.log_det()
    }
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Solve the linear system A x = b.
#[pyfunction]
pub fn solve(a: &PyTensor, b: &PyTensor) -> PyResult<PyTensor> {
    let result = scivex_core::linalg::solve(a.as_f64()?, b.as_f64()?).map_err(core_err)?;
    Ok(PyTensor::from_f64(result))
}

/// Compute the inverse of a square matrix.
#[pyfunction]
pub fn inv(a: &PyTensor) -> PyResult<PyTensor> {
    let result = scivex_core::linalg::inv(a.as_f64()?).map_err(core_err)?;
    Ok(PyTensor::from_f64(result))
}

/// Compute the determinant of a square matrix.
#[pyfunction]
pub fn det(a: &PyTensor) -> PyResult<f64> {
    scivex_core::linalg::det(a.as_f64()?).map_err(core_err)
}

/// Compute the L2 norm of a vector.
#[pyfunction]
pub fn norm(x: &PyTensor) -> PyResult<f64> {
    scivex_core::linalg::blas::nrm2(x.as_f64()?).map_err(core_err)
}

/// Least-squares solve: minimize ||Ax - b||.
#[pyfunction]
pub fn lstsq(a: &PyTensor, b: &PyTensor) -> PyResult<PyTensor> {
    let result =
        scivex_core::linalg::decomp::qr::lstsq(a.as_f64()?, b.as_f64()?).map_err(core_err)?;
    Ok(PyTensor::from_f64(result))
}

/// Register all linalg classes and functions into a submodule.
pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let m = PyModule::new(py, "linalg")?;

    m.add_class::<PyLU>()?;
    m.add_class::<PyQR>()?;
    m.add_class::<PySVD>()?;
    m.add_class::<PyEig>()?;
    m.add_class::<PyCholesky>()?;

    m.add_function(wrap_pyfunction!(solve, &m)?)?;
    m.add_function(wrap_pyfunction!(inv, &m)?)?;
    m.add_function(wrap_pyfunction!(det, &m)?)?;
    m.add_function(wrap_pyfunction!(norm, &m)?)?;
    m.add_function(wrap_pyfunction!(lstsq, &m)?)?;

    parent.add_submodule(&m)?;
    Ok(())
}
