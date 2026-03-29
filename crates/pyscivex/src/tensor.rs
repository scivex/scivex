//! Python bindings for [`scivex_core::Tensor`] with multi-dtype support,
//! sparse matrices, and Einstein summation.
//!
//! `PyTensor` wraps tensors of four element types (`f64`, `f32`, `i64`, `i32`)
//! via the [`TensorData`] enum-dispatch.  The default dtype is `"f64"` for full
//! backward compatibility.

use pyo3::prelude::*;
use pyo3::types::PyList;
use scivex_core::Tensor;
use scivex_core::linalg::sparse::{CooMatrix, CscMatrix, CsrMatrix};
use scivex_core::tensor::einsum::einsum;
use scivex_core::tensor::indexing::SliceRange;

/// Convert a `CoreError` into a Python `ValueError`.
#[allow(clippy::needless_pass_by_value)]
fn core_err(e: scivex_core::CoreError) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}

/// Return a `TypeError` for an unsupported dtype string.
fn dtype_err(dtype: &str) -> PyErr {
    pyo3::exceptions::PyTypeError::new_err(format!(
        "unsupported dtype '{dtype}'; expected one of 'f64', 'f32', 'i64', 'i32'"
    ))
}

/// Return a `TypeError` when an operation requires matching dtypes.
fn dtype_mismatch_err(lhs: &str, rhs: &str) -> PyErr {
    pyo3::exceptions::PyTypeError::new_err(format!(
        "dtype mismatch: left is '{lhs}', right is '{rhs}'"
    ))
}

/// Return a `TypeError` when a float-only operation is called on an integer tensor.
fn float_only_err(op: &str, dtype: &str) -> PyErr {
    pyo3::exceptions::PyTypeError::new_err(format!(
        "'{op}' requires a float dtype (f64 or f32), but tensor has dtype '{dtype}'"
    ))
}

// =========================================================================
// TensorData — enum dispatch over supported element types
// =========================================================================

/// Storage for a tensor whose element type is chosen at runtime.
#[derive(Clone, Debug)]
pub(crate) enum TensorData {
    /// 64-bit floating point (default).
    F64(Tensor<f64>),
    /// 32-bit floating point.
    F32(Tensor<f32>),
    /// 64-bit signed integer.
    I64(Tensor<i64>),
    /// 32-bit signed integer.
    I32(Tensor<i32>),
}

impl TensorData {
    /// Return the dtype name as a static string.
    fn dtype_str(&self) -> &'static str {
        match self {
            Self::F64(_) => "f64",
            Self::F32(_) => "f32",
            Self::I64(_) => "i64",
            Self::I32(_) => "i32",
        }
    }

    /// Shape of the inner tensor.
    fn shape(&self) -> &[usize] {
        match self {
            Self::F64(t) => t.shape(),
            Self::F32(t) => t.shape(),
            Self::I64(t) => t.shape(),
            Self::I32(t) => t.shape(),
        }
    }

    /// Strides of the inner tensor.
    fn strides(&self) -> &[usize] {
        match self {
            Self::F64(t) => t.strides(),
            Self::F32(t) => t.strides(),
            Self::I64(t) => t.strides(),
            Self::I32(t) => t.strides(),
        }
    }

    /// Number of dimensions.
    fn ndim(&self) -> usize {
        match self {
            Self::F64(t) => t.ndim(),
            Self::F32(t) => t.ndim(),
            Self::I64(t) => t.ndim(),
            Self::I32(t) => t.ndim(),
        }
    }

    /// Total number of elements.
    fn numel(&self) -> usize {
        match self {
            Self::F64(t) => t.numel(),
            Self::F32(t) => t.numel(),
            Self::I64(t) => t.numel(),
            Self::I32(t) => t.numel(),
        }
    }

    /// Whether the tensor is empty.
    fn is_empty(&self) -> bool {
        match self {
            Self::F64(t) => t.is_empty(),
            Self::F32(t) => t.is_empty(),
            Self::I64(t) => t.is_empty(),
            Self::I32(t) => t.is_empty(),
        }
    }

    /// Convert to an `f64` tensor, casting if necessary.
    fn to_f64(&self) -> Tensor<f64> {
        match self {
            Self::F64(t) => t.clone(),
            Self::F32(t) => t.cast::<f64>(),
            Self::I64(t) => {
                let data: Vec<f64> = t.as_slice().iter().map(|&v| v as f64).collect();
                Tensor::from_vec(data, t.shape().to_vec()).unwrap()
            }
            Self::I32(t) => {
                let data: Vec<f64> = t.as_slice().iter().map(|&v| f64::from(v)).collect();
                Tensor::from_vec(data, t.shape().to_vec()).unwrap()
            }
        }
    }

    /// Convert to an `f32` tensor, casting if necessary.
    fn to_f32(&self) -> Tensor<f32> {
        match self {
            Self::F64(t) => t.cast::<f32>(),
            Self::F32(t) => t.clone(),
            Self::I64(t) => {
                let data: Vec<f32> = t.as_slice().iter().map(|&v| v as f32).collect();
                Tensor::from_vec(data, t.shape().to_vec()).unwrap()
            }
            Self::I32(t) => {
                let data: Vec<f32> = t.as_slice().iter().map(|&v| v as f32).collect();
                Tensor::from_vec(data, t.shape().to_vec()).unwrap()
            }
        }
    }

    /// Convert to the given dtype string.
    fn to_dtype(&self, dtype: &str) -> PyResult<Self> {
        match dtype {
            "f64" => Ok(Self::F64(self.to_f64())),
            "f32" => Ok(Self::F32(self.to_f32())),
            "i64" => {
                let f = self.to_f64();
                #[allow(clippy::cast_possible_truncation)]
                let data: Vec<i64> = f.as_slice().iter().map(|&v| v as i64).collect();
                Ok(Self::I64(
                    Tensor::from_vec(data, f.shape().to_vec()).map_err(core_err)?,
                ))
            }
            "i32" => {
                let f = self.to_f64();
                #[allow(clippy::cast_possible_truncation)]
                let data: Vec<i32> = f.as_slice().iter().map(|&v| v as i32).collect();
                Ok(Self::I32(
                    Tensor::from_vec(data, f.shape().to_vec()).map_err(core_err)?,
                ))
            }
            other => Err(dtype_err(other)),
        }
    }
}

// =========================================================================
// Helper: display for TensorData
// =========================================================================

impl std::fmt::Display for TensorData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::F64(t) => write!(f, "{t}"),
            Self::F32(t) => write!(f, "{t}"),
            Self::I64(t) => write!(f, "{t}"),
            Self::I32(t) => write!(f, "{t}"),
        }
    }
}

// =========================================================================
// PyTensor
// =========================================================================

/// A multidimensional tensor supporting multiple element types.
///
/// Supported dtypes: ``"f64"`` (default), ``"f32"``, ``"i64"``, ``"i32"``.
///
/// Constructors accept an optional ``dtype`` parameter.  When omitted the
/// tensor is created with ``f64`` elements for backward compatibility.
#[pyclass(name = "Tensor")]
#[derive(Clone)]
pub struct PyTensor {
    pub(crate) data: TensorData,
}

impl PyTensor {
    /// Construct a `PyTensor` from an `f64` tensor (convenience for other
    /// pyscivex modules that still work exclusively with `f64`).
    pub(crate) fn from_f64(t: Tensor<f64>) -> Self {
        Self {
            data: TensorData::F64(t),
        }
    }

    /// Borrow the inner `Tensor<f64>`, returning a `PyResult` error if the
    /// dtype is not `f64`.  Other pyscivex modules that need raw `Tensor<f64>`
    /// access should call this.
    pub(crate) fn as_f64(&self) -> PyResult<&Tensor<f64>> {
        match &self.data {
            TensorData::F64(t) => Ok(t),
            other => Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "expected dtype 'f64', got '{}'",
                other.dtype_str()
            ))),
        }
    }

    /// Get an owned `Tensor<f64>`, converting from other dtypes if needed.
    /// This never fails.
    pub(crate) fn to_f64_tensor(&self) -> Tensor<f64> {
        self.data.to_f64()
    }
}

// ---------------------------------------------------------------------------
// Helpers: recursively flatten a nested Python list into a flat Vec + shape
// ---------------------------------------------------------------------------

/// Flatten a nested Python list into `(Vec<f64>, shape)`.
pub(crate) fn nested_list_to_flat(obj: &Bound<'_, PyAny>) -> PyResult<(Vec<f64>, Vec<usize>)> {
    if let Ok(list) = obj.downcast::<PyList>() {
        if list.is_empty() {
            return Ok((vec![], vec![0]));
        }
        let first = list.get_item(0)?;
        if first.downcast::<PyList>().is_ok() {
            let mut all_data = Vec::new();
            let mut child_shape: Option<Vec<usize>> = None;
            for i in 0..list.len() {
                let item = list.get_item(i)?;
                let (data, shape) = nested_list_to_flat(&item)?;
                if let Some(ref cs) = child_shape {
                    if *cs != shape {
                        return Err(pyo3::exceptions::PyValueError::new_err(
                            "jagged nested lists are not supported; all sub-lists must have the same shape",
                        ));
                    }
                } else {
                    child_shape = Some(shape);
                }
                all_data.extend(data);
            }
            let mut full_shape = vec![list.len()];
            if let Some(cs) = child_shape {
                full_shape.extend(cs);
            }
            Ok((all_data, full_shape))
        } else {
            let mut data = Vec::with_capacity(list.len());
            for i in 0..list.len() {
                let item = list.get_item(i)?;
                let val: f64 = item.extract()?;
                data.push(val);
            }
            Ok((data, vec![list.len()]))
        }
    } else {
        let val: f64 = obj.extract()?;
        Ok((vec![val], vec![]))
    }
}

/// Recursively build a nested Python list from flat `f64` data + shape.
fn flat_to_nested_list_f64(py: Python<'_>, data: &[f64], shape: &[usize]) -> PyResult<PyObject> {
    if shape.is_empty() {
        return Ok(data[0].into_pyobject(py)?.into_any().unbind());
    }
    if shape.len() == 1 {
        let list = PyList::new(py, data)?;
        return Ok(list.into_any().unbind());
    }
    let dim = shape[0];
    let sub_shape = &shape[1..];
    let sub_size: usize = sub_shape.iter().product();
    let list = PyList::empty(py);
    for i in 0..dim {
        let sub_data = &data[i * sub_size..(i + 1) * sub_size];
        let sub_obj = flat_to_nested_list_f64(py, sub_data, sub_shape)?;
        list.append(sub_obj)?;
    }
    Ok(list.into_any().unbind())
}

/// Recursively build a nested Python list from flat `f32` data + shape.
fn flat_to_nested_list_f32(py: Python<'_>, data: &[f32], shape: &[usize]) -> PyResult<PyObject> {
    if shape.is_empty() {
        return Ok(data[0].into_pyobject(py)?.into_any().unbind());
    }
    if shape.len() == 1 {
        let list = PyList::new(py, data.iter().map(|&v| f64::from(v)).collect::<Vec<_>>())?;
        return Ok(list.into_any().unbind());
    }
    let dim = shape[0];
    let sub_shape = &shape[1..];
    let sub_size: usize = sub_shape.iter().product();
    let list = PyList::empty(py);
    for i in 0..dim {
        let sub_data = &data[i * sub_size..(i + 1) * sub_size];
        let sub_obj = flat_to_nested_list_f32(py, sub_data, sub_shape)?;
        list.append(sub_obj)?;
    }
    Ok(list.into_any().unbind())
}

/// Recursively build a nested Python list from flat `i64` data + shape.
fn flat_to_nested_list_i64(py: Python<'_>, data: &[i64], shape: &[usize]) -> PyResult<PyObject> {
    if shape.is_empty() {
        return Ok(data[0].into_pyobject(py)?.into_any().unbind());
    }
    if shape.len() == 1 {
        let list = PyList::new(py, data)?;
        return Ok(list.into_any().unbind());
    }
    let dim = shape[0];
    let sub_shape = &shape[1..];
    let sub_size: usize = sub_shape.iter().product();
    let list = PyList::empty(py);
    for i in 0..dim {
        let sub_data = &data[i * sub_size..(i + 1) * sub_size];
        let sub_obj = flat_to_nested_list_i64(py, sub_data, sub_shape)?;
        list.append(sub_obj)?;
    }
    Ok(list.into_any().unbind())
}

/// Recursively build a nested Python list from flat `i32` data + shape.
fn flat_to_nested_list_i32(py: Python<'_>, data: &[i32], shape: &[usize]) -> PyResult<PyObject> {
    if shape.is_empty() {
        return Ok(data[0].into_pyobject(py)?.into_any().unbind());
    }
    if shape.len() == 1 {
        let list = PyList::new(py, data)?;
        return Ok(list.into_any().unbind());
    }
    let dim = shape[0];
    let sub_shape = &shape[1..];
    let sub_size: usize = sub_shape.iter().product();
    let list = PyList::empty(py);
    for i in 0..dim {
        let sub_data = &data[i * sub_size..(i + 1) * sub_size];
        let sub_obj = flat_to_nested_list_i32(py, sub_data, sub_shape)?;
        list.append(sub_obj)?;
    }
    Ok(list.into_any().unbind())
}

// =========================================================================
// Macro to reduce boilerplate for dispatching over TensorData variants
// =========================================================================

/// Dispatch a method call on every variant, returning a new `PyTensor`.
/// Usage: `dispatch_all!(self, t => t.method_call())`
macro_rules! dispatch_all {
    ($self:expr, $t:ident => $body:expr) => {
        match &$self.data {
            TensorData::F64($t) => PyTensor {
                data: TensorData::F64($body),
            },
            TensorData::F32($t) => PyTensor {
                data: TensorData::F32($body),
            },
            TensorData::I64($t) => PyTensor {
                data: TensorData::I64($body),
            },
            TensorData::I32($t) => PyTensor {
                data: TensorData::I32($body),
            },
        }
    };
}

/// Dispatch a fallible method call on every variant, returning `PyResult<PyTensor>`.
macro_rules! dispatch_all_try {
    ($self:expr, $t:ident => $body:expr) => {
        match &$self.data {
            TensorData::F64($t) => Ok(PyTensor {
                data: TensorData::F64($body.map_err(core_err)?),
            }),
            TensorData::F32($t) => Ok(PyTensor {
                data: TensorData::F32($body.map_err(core_err)?),
            }),
            TensorData::I64($t) => Ok(PyTensor {
                data: TensorData::I64($body.map_err(core_err)?),
            }),
            TensorData::I32($t) => Ok(PyTensor {
                data: TensorData::I32($body.map_err(core_err)?),
            }),
        }
    };
}

/// Dispatch a method call only on float variants, erroring for integer types.
macro_rules! dispatch_float {
    ($self:expr, $op:expr, $t:ident => $body:expr) => {
        match &$self.data {
            TensorData::F64($t) => Ok(PyTensor {
                data: TensorData::F64($body),
            }),
            TensorData::F32($t) => Ok(PyTensor {
                data: TensorData::F32($body),
            }),
            other => Err(float_only_err($op, other.dtype_str())),
        }
    };
}

/// Dispatch a fallible method call only on float variants.
macro_rules! dispatch_float_try {
    ($self:expr, $op:expr, $t:ident => $body:expr) => {
        match &$self.data {
            TensorData::F64($t) => Ok(PyTensor {
                data: TensorData::F64($body.map_err(core_err)?),
            }),
            TensorData::F32($t) => Ok(PyTensor {
                data: TensorData::F32($body.map_err(core_err)?),
            }),
            other => Err(float_only_err($op, other.dtype_str())),
        }
    };
}

// =========================================================================
// #[pymethods] implementation
// =========================================================================

#[pymethods]
#[allow(clippy::needless_pass_by_value)]
impl PyTensor {
    // ======================================================================
    // Constructors
    // ======================================================================

    /// Create a tensor from data and an optional shape.
    ///
    /// Parameters
    /// ----------
    /// data : list or nested list
    ///     Flat data with explicit ``shape``, or a nested list whose shape is
    ///     inferred automatically.
    /// shape : list[int], optional
    ///     Explicit shape.  Required when ``data`` is a flat list.
    /// dtype : str, optional
    ///     Element type: ``"f64"`` (default), ``"f32"``, ``"i64"``, ``"i32"``.
    ///
    /// Examples
    /// --------
    /// >>> Tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
    /// >>> Tensor([[1, 2], [3, 4]], dtype="i32")
    #[new]
    #[pyo3(signature = (data, shape=None, dtype=None))]
    fn new(
        data: &Bound<'_, PyAny>,
        shape: Option<Vec<usize>>,
        dtype: Option<&str>,
    ) -> PyResult<Self> {
        let dtype = dtype.unwrap_or("f64");

        // Parse flat data + shape, or nested list.
        let (flat_f64, resolved_shape) = if let Some(shape) = shape {
            let flat: Vec<f64> = data.extract()?;
            (flat, shape)
        } else {
            let (flat, inferred) = nested_list_to_flat(data)?;
            (flat, inferred)
        };

        // Build the correct TensorData variant.
        match dtype {
            "f64" => {
                if resolved_shape.is_empty() {
                    Ok(Self {
                        data: TensorData::F64(Tensor::scalar(flat_f64[0])),
                    })
                } else {
                    Ok(Self {
                        data: TensorData::F64(
                            Tensor::from_vec(flat_f64, resolved_shape).map_err(core_err)?,
                        ),
                    })
                }
            }
            "f32" => {
                #[allow(clippy::cast_possible_truncation)]
                let flat: Vec<f32> = flat_f64.iter().map(|&v| v as f32).collect();
                if resolved_shape.is_empty() {
                    Ok(Self {
                        data: TensorData::F32(Tensor::scalar(flat[0])),
                    })
                } else {
                    Ok(Self {
                        data: TensorData::F32(
                            Tensor::from_vec(flat, resolved_shape).map_err(core_err)?,
                        ),
                    })
                }
            }
            "i64" => {
                #[allow(clippy::cast_possible_truncation)]
                let flat: Vec<i64> = flat_f64.iter().map(|&v| v as i64).collect();
                if resolved_shape.is_empty() {
                    Ok(Self {
                        data: TensorData::I64(Tensor::scalar(flat[0])),
                    })
                } else {
                    Ok(Self {
                        data: TensorData::I64(
                            Tensor::from_vec(flat, resolved_shape).map_err(core_err)?,
                        ),
                    })
                }
            }
            "i32" => {
                #[allow(clippy::cast_possible_truncation)]
                let flat: Vec<i32> = flat_f64.iter().map(|&v| v as i32).collect();
                if resolved_shape.is_empty() {
                    Ok(Self {
                        data: TensorData::I32(Tensor::scalar(flat[0])),
                    })
                } else {
                    Ok(Self {
                        data: TensorData::I32(
                            Tensor::from_vec(flat, resolved_shape).map_err(core_err)?,
                        ),
                    })
                }
            }
            other => Err(dtype_err(other)),
        }
    }

    /// Create a tensor filled with zeros.
    ///
    /// Parameters
    /// ----------
    /// shape : list[int]
    /// dtype : str, optional
    #[staticmethod]
    #[pyo3(signature = (shape, dtype=None))]
    fn zeros(shape: Vec<usize>, dtype: Option<&str>) -> PyResult<Self> {
        match dtype.unwrap_or("f64") {
            "f64" => Ok(Self {
                data: TensorData::F64(Tensor::zeros(shape)),
            }),
            "f32" => Ok(Self {
                data: TensorData::F32(Tensor::zeros(shape)),
            }),
            "i64" => Ok(Self {
                data: TensorData::I64(Tensor::zeros(shape)),
            }),
            "i32" => Ok(Self {
                data: TensorData::I32(Tensor::zeros(shape)),
            }),
            other => Err(dtype_err(other)),
        }
    }

    /// Create a tensor filled with ones.
    ///
    /// Parameters
    /// ----------
    /// shape : list[int]
    /// dtype : str, optional
    #[staticmethod]
    #[pyo3(signature = (shape, dtype=None))]
    fn ones(shape: Vec<usize>, dtype: Option<&str>) -> PyResult<Self> {
        match dtype.unwrap_or("f64") {
            "f64" => Ok(Self {
                data: TensorData::F64(Tensor::ones(shape)),
            }),
            "f32" => Ok(Self {
                data: TensorData::F32(Tensor::ones(shape)),
            }),
            "i64" => Ok(Self {
                data: TensorData::I64(Tensor::ones(shape)),
            }),
            "i32" => Ok(Self {
                data: TensorData::I32(Tensor::ones(shape)),
            }),
            other => Err(dtype_err(other)),
        }
    }

    /// Create a tensor filled with a given value.
    ///
    /// Parameters
    /// ----------
    /// shape : list[int]
    /// value : float
    /// dtype : str, optional
    #[staticmethod]
    #[pyo3(signature = (shape, value, dtype=None))]
    fn full(shape: Vec<usize>, value: f64, dtype: Option<&str>) -> PyResult<Self> {
        match dtype.unwrap_or("f64") {
            "f64" => Ok(Self {
                data: TensorData::F64(Tensor::full(shape, value)),
            }),
            #[allow(clippy::cast_possible_truncation)]
            "f32" => Ok(Self {
                data: TensorData::F32(Tensor::full(shape, value as f32)),
            }),
            #[allow(clippy::cast_possible_truncation)]
            "i64" => Ok(Self {
                data: TensorData::I64(Tensor::full(shape, value as i64)),
            }),
            #[allow(clippy::cast_possible_truncation)]
            "i32" => Ok(Self {
                data: TensorData::I32(Tensor::full(shape, value as i32)),
            }),
            other => Err(dtype_err(other)),
        }
    }

    /// Create an identity matrix of size ``n x n``.
    ///
    /// Parameters
    /// ----------
    /// n : int
    /// dtype : str, optional
    #[staticmethod]
    #[pyo3(signature = (n, dtype=None))]
    fn eye(n: usize, dtype: Option<&str>) -> PyResult<Self> {
        match dtype.unwrap_or("f64") {
            "f64" => Ok(Self {
                data: TensorData::F64(Tensor::eye(n)),
            }),
            "f32" => Ok(Self {
                data: TensorData::F32(Tensor::eye(n)),
            }),
            "i64" => Ok(Self {
                data: TensorData::I64(Tensor::eye(n)),
            }),
            "i32" => Ok(Self {
                data: TensorData::I32(Tensor::eye(n)),
            }),
            other => Err(dtype_err(other)),
        }
    }

    /// Create a 1-D tensor with values ``[0, 1, ..., n-1]``.
    ///
    /// Parameters
    /// ----------
    /// n : int
    /// dtype : str, optional
    #[staticmethod]
    #[pyo3(signature = (n, dtype=None))]
    fn arange(n: usize, dtype: Option<&str>) -> PyResult<Self> {
        match dtype.unwrap_or("f64") {
            "f64" => Ok(Self {
                data: TensorData::F64(Tensor::<f64>::arange(n)),
            }),
            "f32" => Ok(Self {
                data: TensorData::F32(Tensor::<f32>::arange(n)),
            }),
            "i64" => Ok(Self {
                data: TensorData::I64(Tensor::<i64>::arange(n)),
            }),
            "i32" => Ok(Self {
                data: TensorData::I32(Tensor::<i32>::arange(n)),
            }),
            other => Err(dtype_err(other)),
        }
    }

    /// Create a 1-D tensor of ``n`` evenly spaced values from ``start`` to ``end``.
    ///
    /// Always produces an ``f64`` tensor (or ``f32`` if requested).
    #[staticmethod]
    #[pyo3(signature = (start, end, n, dtype=None))]
    fn linspace(start: f64, end: f64, n: usize, dtype: Option<&str>) -> PyResult<Self> {
        match dtype.unwrap_or("f64") {
            "f64" => {
                let inner = Tensor::linspace(start, end, n).map_err(core_err)?;
                Ok(Self {
                    data: TensorData::F64(inner),
                })
            }
            #[allow(clippy::cast_possible_truncation)]
            "f32" => {
                let inner = Tensor::linspace(start as f32, end as f32, n).map_err(core_err)?;
                Ok(Self {
                    data: TensorData::F32(inner),
                })
            }
            other => Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "linspace only supports float dtypes ('f64', 'f32'), got '{other}'"
            ))),
        }
    }

    /// Create a scalar tensor.
    ///
    /// Parameters
    /// ----------
    /// value : float
    /// dtype : str, optional
    #[staticmethod]
    #[pyo3(signature = (value, dtype=None))]
    fn scalar(value: f64, dtype: Option<&str>) -> PyResult<Self> {
        match dtype.unwrap_or("f64") {
            "f64" => Ok(Self {
                data: TensorData::F64(Tensor::scalar(value)),
            }),
            #[allow(clippy::cast_possible_truncation)]
            "f32" => Ok(Self {
                data: TensorData::F32(Tensor::scalar(value as f32)),
            }),
            #[allow(clippy::cast_possible_truncation)]
            "i64" => Ok(Self {
                data: TensorData::I64(Tensor::scalar(value as i64)),
            }),
            #[allow(clippy::cast_possible_truncation)]
            "i32" => Ok(Self {
                data: TensorData::I32(Tensor::scalar(value as i32)),
            }),
            other => Err(dtype_err(other)),
        }
    }

    // ======================================================================
    // Dtype inspection and conversion
    // ======================================================================

    /// Return the dtype of the tensor as a string.
    ///
    /// Returns one of ``"f64"``, ``"f32"``, ``"i64"``, ``"i32"``.
    #[getter]
    fn dtype(&self) -> &'static str {
        self.data.dtype_str()
    }

    /// Convert the tensor to the specified dtype, returning a new tensor.
    ///
    /// Parameters
    /// ----------
    /// dtype : str
    ///     Target dtype (``"f64"``, ``"f32"``, ``"i64"``, ``"i32"``).
    fn to_dtype(&self, dtype: &str) -> PyResult<Self> {
        Ok(Self {
            data: self.data.to_dtype(dtype)?,
        })
    }

    /// Convert to ``f64`` dtype.
    fn as_f64_py(&self) -> Self {
        Self {
            data: TensorData::F64(self.data.to_f64()),
        }
    }

    /// Convert to ``f32`` dtype.
    fn as_f32(&self) -> Self {
        Self {
            data: TensorData::F32(self.data.to_f32()),
        }
    }

    // ======================================================================
    // Accessors
    // ======================================================================

    /// Return the shape as a list.
    fn shape(&self) -> Vec<usize> {
        self.data.shape().to_vec()
    }

    /// Return the strides as a list.
    fn strides(&self) -> Vec<usize> {
        self.data.strides().to_vec()
    }

    /// Return the number of dimensions.
    fn ndim(&self) -> usize {
        self.data.ndim()
    }

    /// Return the total number of elements.
    fn numel(&self) -> usize {
        self.data.numel()
    }

    /// Return whether the tensor is empty.
    fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Return a flat Python list of all elements.
    ///
    /// For float tensors the values are Python floats; for integer tensors
    /// they are Python ints.
    fn to_list(&self, py: Python<'_>) -> PyResult<PyObject> {
        match &self.data {
            TensorData::F64(t) => {
                let list = PyList::new(py, t.as_slice())?;
                Ok(list.into_any().unbind())
            }
            TensorData::F32(t) => {
                let vals: Vec<f64> = t.as_slice().iter().map(|&v| f64::from(v)).collect();
                let list = PyList::new(py, &vals)?;
                Ok(list.into_any().unbind())
            }
            TensorData::I64(t) => {
                let list = PyList::new(py, t.as_slice())?;
                Ok(list.into_any().unbind())
            }
            TensorData::I32(t) => {
                let list = PyList::new(py, t.as_slice())?;
                Ok(list.into_any().unbind())
            }
        }
    }

    /// Return a nested Python list matching the tensor's shape.
    fn tolist(&self, py: Python<'_>) -> PyResult<PyObject> {
        match &self.data {
            TensorData::F64(t) => flat_to_nested_list_f64(py, t.as_slice(), t.shape()),
            TensorData::F32(t) => flat_to_nested_list_f32(py, t.as_slice(), t.shape()),
            TensorData::I64(t) => flat_to_nested_list_i64(py, t.as_slice(), t.shape()),
            TensorData::I32(t) => flat_to_nested_list_i32(py, t.as_slice(), t.shape()),
        }
    }

    /// Get a single element by multi-dimensional index.
    fn get(&self, py: Python<'_>, index: Vec<usize>) -> PyResult<PyObject> {
        match &self.data {
            TensorData::F64(t) => {
                let v = *t.get(&index).map_err(core_err)?;
                Ok(v.into_pyobject(py)?.into_any().unbind())
            }
            TensorData::F32(t) => {
                let v = *t.get(&index).map_err(core_err)?;
                Ok(f64::from(v).into_pyobject(py)?.into_any().unbind())
            }
            TensorData::I64(t) => {
                let v = *t.get(&index).map_err(core_err)?;
                Ok(v.into_pyobject(py)?.into_any().unbind())
            }
            TensorData::I32(t) => {
                let v = *t.get(&index).map_err(core_err)?;
                Ok(v.into_pyobject(py)?.into_any().unbind())
            }
        }
    }

    /// Set a single element by multi-dimensional index.
    fn set(&mut self, index: Vec<usize>, value: f64) -> PyResult<()> {
        match &mut self.data {
            TensorData::F64(t) => t.set(&index, value).map_err(core_err),
            #[allow(clippy::cast_possible_truncation)]
            TensorData::F32(t) => t.set(&index, value as f32).map_err(core_err),
            #[allow(clippy::cast_possible_truncation)]
            TensorData::I64(t) => t.set(&index, value as i64).map_err(core_err),
            #[allow(clippy::cast_possible_truncation)]
            TensorData::I32(t) => t.set(&index, value as i32).map_err(core_err),
        }
    }

    // ======================================================================
    // Arithmetic operators
    // ======================================================================

    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(t) = other.extract::<PyRef<PyTensor>>() {
            if self.data.dtype_str() != t.data.dtype_str() {
                return Err(dtype_mismatch_err(
                    self.data.dtype_str(),
                    t.data.dtype_str(),
                ));
            }
            match (&self.data, &t.data) {
                (TensorData::F64(a), TensorData::F64(b)) => Ok(Self {
                    data: TensorData::F64(a.add_checked(b).map_err(core_err)?),
                }),
                (TensorData::F32(a), TensorData::F32(b)) => Ok(Self {
                    data: TensorData::F32(a.add_checked(b).map_err(core_err)?),
                }),
                (TensorData::I64(a), TensorData::I64(b)) => Ok(Self {
                    data: TensorData::I64(a.add_checked(b).map_err(core_err)?),
                }),
                (TensorData::I32(a), TensorData::I32(b)) => Ok(Self {
                    data: TensorData::I32(a.add_checked(b).map_err(core_err)?),
                }),
                _ => unreachable!(),
            }
        } else {
            let scalar: f64 = other.extract()?;
            match &self.data {
                TensorData::F64(t) => Ok(Self {
                    data: TensorData::F64(t + scalar),
                }),
                #[allow(clippy::cast_possible_truncation)]
                TensorData::F32(t) => Ok(Self {
                    data: TensorData::F32(t + scalar as f32),
                }),
                #[allow(clippy::cast_possible_truncation)]
                TensorData::I64(t) => Ok(Self {
                    data: TensorData::I64(t + scalar as i64),
                }),
                #[allow(clippy::cast_possible_truncation)]
                TensorData::I32(t) => Ok(Self {
                    data: TensorData::I32(t + scalar as i32),
                }),
            }
        }
    }

    fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.__add__(other)
    }

    fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(t) = other.extract::<PyRef<PyTensor>>() {
            if self.data.dtype_str() != t.data.dtype_str() {
                return Err(dtype_mismatch_err(
                    self.data.dtype_str(),
                    t.data.dtype_str(),
                ));
            }
            match (&self.data, &t.data) {
                (TensorData::F64(a), TensorData::F64(b)) => Ok(Self {
                    data: TensorData::F64(a.sub_checked(b).map_err(core_err)?),
                }),
                (TensorData::F32(a), TensorData::F32(b)) => Ok(Self {
                    data: TensorData::F32(a.sub_checked(b).map_err(core_err)?),
                }),
                (TensorData::I64(a), TensorData::I64(b)) => Ok(Self {
                    data: TensorData::I64(a.sub_checked(b).map_err(core_err)?),
                }),
                (TensorData::I32(a), TensorData::I32(b)) => Ok(Self {
                    data: TensorData::I32(a.sub_checked(b).map_err(core_err)?),
                }),
                _ => unreachable!(),
            }
        } else {
            let scalar: f64 = other.extract()?;
            match &self.data {
                TensorData::F64(t) => Ok(Self {
                    data: TensorData::F64(t - scalar),
                }),
                #[allow(clippy::cast_possible_truncation)]
                TensorData::F32(t) => Ok(Self {
                    data: TensorData::F32(t - scalar as f32),
                }),
                #[allow(clippy::cast_possible_truncation)]
                TensorData::I64(t) => Ok(Self {
                    data: TensorData::I64(t - scalar as i64),
                }),
                #[allow(clippy::cast_possible_truncation)]
                TensorData::I32(t) => Ok(Self {
                    data: TensorData::I32(t - scalar as i32),
                }),
            }
        }
    }

    fn __rsub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(t) = other.extract::<PyRef<PyTensor>>() {
            if self.data.dtype_str() != t.data.dtype_str() {
                return Err(dtype_mismatch_err(
                    t.data.dtype_str(),
                    self.data.dtype_str(),
                ));
            }
            match (&t.data, &self.data) {
                (TensorData::F64(a), TensorData::F64(b)) => Ok(Self {
                    data: TensorData::F64(a.sub_checked(b).map_err(core_err)?),
                }),
                (TensorData::F32(a), TensorData::F32(b)) => Ok(Self {
                    data: TensorData::F32(a.sub_checked(b).map_err(core_err)?),
                }),
                (TensorData::I64(a), TensorData::I64(b)) => Ok(Self {
                    data: TensorData::I64(a.sub_checked(b).map_err(core_err)?),
                }),
                (TensorData::I32(a), TensorData::I32(b)) => Ok(Self {
                    data: TensorData::I32(a.sub_checked(b).map_err(core_err)?),
                }),
                _ => unreachable!(),
            }
        } else {
            let scalar: f64 = other.extract()?;
            match &self.data {
                TensorData::F64(t) => Ok(Self {
                    data: TensorData::F64(t.map(|x| scalar - x)),
                }),
                #[allow(clippy::cast_possible_truncation)]
                TensorData::F32(t) => {
                    let s = scalar as f32;
                    Ok(Self {
                        data: TensorData::F32(t.map(move |x| s - x)),
                    })
                }
                #[allow(clippy::cast_possible_truncation)]
                TensorData::I64(t) => {
                    let s = scalar as i64;
                    Ok(Self {
                        data: TensorData::I64(t.map(move |x| s - x)),
                    })
                }
                #[allow(clippy::cast_possible_truncation)]
                TensorData::I32(t) => {
                    let s = scalar as i32;
                    Ok(Self {
                        data: TensorData::I32(t.map(move |x| s - x)),
                    })
                }
            }
        }
    }

    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(t) = other.extract::<PyRef<PyTensor>>() {
            if self.data.dtype_str() != t.data.dtype_str() {
                return Err(dtype_mismatch_err(
                    self.data.dtype_str(),
                    t.data.dtype_str(),
                ));
            }
            match (&self.data, &t.data) {
                (TensorData::F64(a), TensorData::F64(b)) => Ok(Self {
                    data: TensorData::F64(a.mul_checked(b).map_err(core_err)?),
                }),
                (TensorData::F32(a), TensorData::F32(b)) => Ok(Self {
                    data: TensorData::F32(a.mul_checked(b).map_err(core_err)?),
                }),
                (TensorData::I64(a), TensorData::I64(b)) => Ok(Self {
                    data: TensorData::I64(a.mul_checked(b).map_err(core_err)?),
                }),
                (TensorData::I32(a), TensorData::I32(b)) => Ok(Self {
                    data: TensorData::I32(a.mul_checked(b).map_err(core_err)?),
                }),
                _ => unreachable!(),
            }
        } else {
            let scalar: f64 = other.extract()?;
            match &self.data {
                TensorData::F64(t) => Ok(Self {
                    data: TensorData::F64(t * scalar),
                }),
                #[allow(clippy::cast_possible_truncation)]
                TensorData::F32(t) => Ok(Self {
                    data: TensorData::F32(t * scalar as f32),
                }),
                #[allow(clippy::cast_possible_truncation)]
                TensorData::I64(t) => Ok(Self {
                    data: TensorData::I64(t * scalar as i64),
                }),
                #[allow(clippy::cast_possible_truncation)]
                TensorData::I32(t) => Ok(Self {
                    data: TensorData::I32(t * scalar as i32),
                }),
            }
        }
    }

    fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.__mul__(other)
    }

    fn __truediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(t) = other.extract::<PyRef<PyTensor>>() {
            if self.data.dtype_str() != t.data.dtype_str() {
                return Err(dtype_mismatch_err(
                    self.data.dtype_str(),
                    t.data.dtype_str(),
                ));
            }
            match (&self.data, &t.data) {
                (TensorData::F64(a), TensorData::F64(b)) => Ok(Self {
                    data: TensorData::F64(a.div_checked(b).map_err(core_err)?),
                }),
                (TensorData::F32(a), TensorData::F32(b)) => Ok(Self {
                    data: TensorData::F32(a.div_checked(b).map_err(core_err)?),
                }),
                (TensorData::I64(a), TensorData::I64(b)) => Ok(Self {
                    data: TensorData::I64(a.div_checked(b).map_err(core_err)?),
                }),
                (TensorData::I32(a), TensorData::I32(b)) => Ok(Self {
                    data: TensorData::I32(a.div_checked(b).map_err(core_err)?),
                }),
                _ => unreachable!(),
            }
        } else {
            let scalar: f64 = other.extract()?;
            match &self.data {
                TensorData::F64(t) => Ok(Self {
                    data: TensorData::F64(t / scalar),
                }),
                #[allow(clippy::cast_possible_truncation)]
                TensorData::F32(t) => Ok(Self {
                    data: TensorData::F32(t / scalar as f32),
                }),
                #[allow(clippy::cast_possible_truncation)]
                TensorData::I64(t) => Ok(Self {
                    data: TensorData::I64(t / scalar as i64),
                }),
                #[allow(clippy::cast_possible_truncation)]
                TensorData::I32(t) => Ok(Self {
                    data: TensorData::I32(t / scalar as i32),
                }),
            }
        }
    }

    fn __rtruediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(t) = other.extract::<PyRef<PyTensor>>() {
            if self.data.dtype_str() != t.data.dtype_str() {
                return Err(dtype_mismatch_err(
                    t.data.dtype_str(),
                    self.data.dtype_str(),
                ));
            }
            match (&t.data, &self.data) {
                (TensorData::F64(a), TensorData::F64(b)) => Ok(Self {
                    data: TensorData::F64(a.div_checked(b).map_err(core_err)?),
                }),
                (TensorData::F32(a), TensorData::F32(b)) => Ok(Self {
                    data: TensorData::F32(a.div_checked(b).map_err(core_err)?),
                }),
                (TensorData::I64(a), TensorData::I64(b)) => Ok(Self {
                    data: TensorData::I64(a.div_checked(b).map_err(core_err)?),
                }),
                (TensorData::I32(a), TensorData::I32(b)) => Ok(Self {
                    data: TensorData::I32(a.div_checked(b).map_err(core_err)?),
                }),
                _ => unreachable!(),
            }
        } else {
            let scalar: f64 = other.extract()?;
            match &self.data {
                TensorData::F64(t) => Ok(Self {
                    data: TensorData::F64(t.map(|x| scalar / x)),
                }),
                #[allow(clippy::cast_possible_truncation)]
                TensorData::F32(t) => {
                    let s = scalar as f32;
                    Ok(Self {
                        data: TensorData::F32(t.map(move |x| s / x)),
                    })
                }
                #[allow(clippy::cast_possible_truncation)]
                TensorData::I64(t) => {
                    let s = scalar as i64;
                    Ok(Self {
                        data: TensorData::I64(t.map(move |x| s / x)),
                    })
                }
                #[allow(clippy::cast_possible_truncation)]
                TensorData::I32(t) => {
                    let s = scalar as i32;
                    Ok(Self {
                        data: TensorData::I32(t.map(move |x| s / x)),
                    })
                }
            }
        }
    }

    /// Unary negation (float dtypes only).
    fn __neg__(&self) -> PyResult<Self> {
        dispatch_float!(self, "negation", t => -t)
    }

    /// Matrix multiplication (requires matching dtype; float only for linalg).
    fn __matmul__(&self, other: &PyTensor) -> PyResult<Self> {
        if self.data.dtype_str() != other.data.dtype_str() {
            return Err(dtype_mismatch_err(
                self.data.dtype_str(),
                other.data.dtype_str(),
            ));
        }
        match (&self.data, &other.data) {
            (TensorData::F64(a), TensorData::F64(b)) => Ok(Self {
                data: TensorData::F64(a.matmul(b).map_err(core_err)?),
            }),
            (TensorData::F32(a), TensorData::F32(b)) => Ok(Self {
                data: TensorData::F32(a.matmul(b).map_err(core_err)?),
            }),
            (TensorData::I64(a), TensorData::I64(b)) => Ok(Self {
                data: TensorData::I64(a.matmul(b).map_err(core_err)?),
            }),
            (TensorData::I32(a), TensorData::I32(b)) => Ok(Self {
                data: TensorData::I32(a.matmul(b).map_err(core_err)?),
            }),
            _ => unreachable!(),
        }
    }

    /// Exponentiation (float dtypes only).
    fn __pow__(
        &self,
        exp: &Bound<'_, PyAny>,
        _modulo: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let e: f64 = exp.extract()?;
        match &self.data {
            TensorData::F64(t) => Ok(Self {
                data: TensorData::F64(t.powf(e)),
            }),
            #[allow(clippy::cast_possible_truncation)]
            TensorData::F32(t) => Ok(Self {
                data: TensorData::F32(t.powf(e as f32)),
            }),
            other => Err(float_only_err("__pow__", other.dtype_str())),
        }
    }

    // ======================================================================
    // Python protocols
    // ======================================================================

    fn __len__(&self) -> usize {
        let shape = self.data.shape();
        if shape.is_empty() { 0 } else { shape[0] }
    }

    fn __bool__(&self) -> PyResult<bool> {
        if self.data.numel() != 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "truth value of a tensor with more than one element is ambiguous",
            ));
        }
        match &self.data {
            TensorData::F64(t) => Ok(t.as_slice()[0] != 0.0),
            TensorData::F32(t) => Ok(t.as_slice()[0] != 0.0),
            TensorData::I64(t) => Ok(t.as_slice()[0] != 0),
            TensorData::I32(t) => Ok(t.as_slice()[0] != 0),
        }
    }

    fn __float__(&self) -> PyResult<f64> {
        if self.data.numel() != 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "only tensors with one element can be converted to float",
            ));
        }
        match &self.data {
            TensorData::F64(t) => Ok(t.as_slice()[0]),
            TensorData::F32(t) => Ok(f64::from(t.as_slice()[0])),
            TensorData::I64(t) => Ok(t.as_slice()[0] as f64),
            TensorData::I32(t) => Ok(f64::from(t.as_slice()[0])),
        }
    }

    fn __int__(&self) -> PyResult<i64> {
        if self.data.numel() != 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "only tensors with one element can be converted to int",
            ));
        }
        match &self.data {
            #[allow(clippy::cast_possible_truncation)]
            TensorData::F64(t) => Ok(t.as_slice()[0] as i64),
            #[allow(clippy::cast_possible_truncation)]
            TensorData::F32(t) => Ok(t.as_slice()[0] as i64),
            TensorData::I64(t) => Ok(t.as_slice()[0]),
            TensorData::I32(t) => Ok(i64::from(t.as_slice()[0])),
        }
    }

    fn __getitem__(&self, key: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let py = key.py();
        // Integer index — select along axis 0
        if let Ok(idx) = key.extract::<isize>() {
            let n = self.data.shape()[0];
            let idx = normalize_index(idx, n)?;
            return match &self.data {
                TensorData::F64(t) => {
                    let result = t.select(0, idx).map_err(core_err)?;
                    if result.ndim() == 0 {
                        Ok(result.as_slice()[0].into_pyobject(py)?.into_any().unbind())
                    } else {
                        Ok(Self {
                            data: TensorData::F64(result),
                        }
                        .into_pyobject(py)?
                        .into_any()
                        .unbind())
                    }
                }
                TensorData::F32(t) => {
                    let result = t.select(0, idx).map_err(core_err)?;
                    if result.ndim() == 0 {
                        Ok(f64::from(result.as_slice()[0])
                            .into_pyobject(py)?
                            .into_any()
                            .unbind())
                    } else {
                        Ok(Self {
                            data: TensorData::F32(result),
                        }
                        .into_pyobject(py)?
                        .into_any()
                        .unbind())
                    }
                }
                TensorData::I64(t) => {
                    let result = t.select(0, idx).map_err(core_err)?;
                    if result.ndim() == 0 {
                        Ok(result.as_slice()[0].into_pyobject(py)?.into_any().unbind())
                    } else {
                        Ok(Self {
                            data: TensorData::I64(result),
                        }
                        .into_pyobject(py)?
                        .into_any()
                        .unbind())
                    }
                }
                TensorData::I32(t) => {
                    let result = t.select(0, idx).map_err(core_err)?;
                    if result.ndim() == 0 {
                        Ok(result.as_slice()[0].into_pyobject(py)?.into_any().unbind())
                    } else {
                        Ok(Self {
                            data: TensorData::I32(result),
                        }
                        .into_pyobject(py)?
                        .into_any()
                        .unbind())
                    }
                }
            };
        }
        // Tuple of ints — multi-dim index
        if let Ok(indices) = key.extract::<Vec<isize>>() {
            return self.getitem_multi(py, &indices);
        }
        Err(pyo3::exceptions::PyTypeError::new_err(
            "tensor indices must be integers or tuples of integers",
        ))
    }

    fn __setitem__(&mut self, key: Vec<usize>, value: f64) -> PyResult<()> {
        self.set(key, value)
    }

    fn __iter__(&self, py: Python<'_>) -> PyResult<PyObject> {
        if self.data.ndim() == 0 {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "cannot iterate over a 0-d tensor",
            ));
        }
        let list = PyList::empty(py);
        let n = self.data.shape()[0];
        match &self.data {
            TensorData::F64(t) => {
                for i in 0..n {
                    let sub = t.select(0, i).map_err(core_err)?;
                    if sub.ndim() == 0 {
                        list.append(sub.as_slice()[0])?;
                    } else {
                        list.append(
                            Self {
                                data: TensorData::F64(sub),
                            }
                            .into_pyobject(py)?,
                        )?;
                    }
                }
            }
            TensorData::F32(t) => {
                for i in 0..n {
                    let sub = t.select(0, i).map_err(core_err)?;
                    if sub.ndim() == 0 {
                        list.append(f64::from(sub.as_slice()[0]))?;
                    } else {
                        list.append(
                            Self {
                                data: TensorData::F32(sub),
                            }
                            .into_pyobject(py)?,
                        )?;
                    }
                }
            }
            TensorData::I64(t) => {
                for i in 0..n {
                    let sub = t.select(0, i).map_err(core_err)?;
                    if sub.ndim() == 0 {
                        list.append(sub.as_slice()[0])?;
                    } else {
                        list.append(
                            Self {
                                data: TensorData::I64(sub),
                            }
                            .into_pyobject(py)?,
                        )?;
                    }
                }
            }
            TensorData::I32(t) => {
                for i in 0..n {
                    let sub = t.select(0, i).map_err(core_err)?;
                    if sub.ndim() == 0 {
                        list.append(sub.as_slice()[0])?;
                    } else {
                        list.append(
                            Self {
                                data: TensorData::I32(sub),
                            }
                            .into_pyobject(py)?,
                        )?;
                    }
                }
            }
        }
        Ok(list.call_method0("__iter__")?.unbind())
    }

    fn __repr__(&self) -> String {
        if self.data.dtype_str() == "f64" {
            format!("{}", self.data)
        } else {
            format!("{} (dtype={})", self.data, self.data.dtype_str())
        }
    }

    fn __eq__(&self, other: &PyTensor) -> bool {
        if self.data.dtype_str() != other.data.dtype_str() {
            return false;
        }
        match (&self.data, &other.data) {
            (TensorData::F64(a), TensorData::F64(b)) => {
                a.shape() == b.shape() && a.as_slice() == b.as_slice()
            }
            (TensorData::F32(a), TensorData::F32(b)) => {
                a.shape() == b.shape() && a.as_slice() == b.as_slice()
            }
            (TensorData::I64(a), TensorData::I64(b)) => {
                a.shape() == b.shape() && a.as_slice() == b.as_slice()
            }
            (TensorData::I32(a), TensorData::I32(b)) => {
                a.shape() == b.shape() && a.as_slice() == b.as_slice()
            }
            _ => false,
        }
    }

    // ======================================================================
    // Element-wise math (float dtypes only)
    // ======================================================================

    /// Absolute value (float dtypes only).
    fn abs(&self) -> PyResult<Self> {
        dispatch_float!(self, "abs", t => t.abs())
    }

    /// Square root (float dtypes only).
    fn sqrt(&self) -> PyResult<Self> {
        dispatch_float!(self, "sqrt", t => t.sqrt())
    }

    /// Sine (float dtypes only).
    fn sin(&self) -> PyResult<Self> {
        dispatch_float!(self, "sin", t => t.sin())
    }

    /// Cosine (float dtypes only).
    fn cos(&self) -> PyResult<Self> {
        dispatch_float!(self, "cos", t => t.cos())
    }

    /// Tangent (float dtypes only).
    fn tan(&self) -> PyResult<Self> {
        dispatch_float!(self, "tan", t => t.tan())
    }

    /// Exponential (float dtypes only).
    fn exp(&self) -> PyResult<Self> {
        dispatch_float!(self, "exp", t => t.exp())
    }

    /// Natural logarithm (float dtypes only).
    fn ln(&self) -> PyResult<Self> {
        dispatch_float!(self, "ln", t => t.ln())
    }

    /// Base-2 logarithm (float dtypes only).
    fn log2(&self) -> PyResult<Self> {
        dispatch_float!(self, "log2", t => t.log2())
    }

    /// Base-10 logarithm (float dtypes only).
    fn log10(&self) -> PyResult<Self> {
        dispatch_float!(self, "log10", t => t.log10())
    }

    /// Floor (float dtypes only).
    fn floor(&self) -> PyResult<Self> {
        dispatch_float!(self, "floor", t => t.floor())
    }

    /// Ceiling (float dtypes only).
    fn ceil(&self) -> PyResult<Self> {
        dispatch_float!(self, "ceil", t => t.ceil())
    }

    /// Round (float dtypes only).
    fn round(&self) -> PyResult<Self> {
        dispatch_float!(self, "round", t => t.round())
    }

    /// Element-wise float exponentiation (float dtypes only).
    fn powf(&self, exponent: f64) -> PyResult<Self> {
        match &self.data {
            TensorData::F64(t) => Ok(Self {
                data: TensorData::F64(t.powf(exponent)),
            }),
            #[allow(clippy::cast_possible_truncation)]
            TensorData::F32(t) => Ok(Self {
                data: TensorData::F32(t.powf(exponent as f32)),
            }),
            other => Err(float_only_err("powf", other.dtype_str())),
        }
    }

    /// Element-wise integer exponentiation (float dtypes only).
    fn powi(&self, n: i32) -> PyResult<Self> {
        match &self.data {
            TensorData::F64(t) => Ok(Self {
                data: TensorData::F64(t.powi(n)),
            }),
            TensorData::F32(t) => Ok(Self {
                data: TensorData::F32(t.powi(n)),
            }),
            other => Err(float_only_err("powi", other.dtype_str())),
        }
    }

    /// Clamp elements to ``[min, max]`` (float dtypes only).
    fn clamp(&self, min: f64, max: f64) -> PyResult<Self> {
        match &self.data {
            TensorData::F64(t) => Ok(Self {
                data: TensorData::F64(t.clamp(min, max)),
            }),
            #[allow(clippy::cast_possible_truncation)]
            TensorData::F32(t) => Ok(Self {
                data: TensorData::F32(t.clamp(min as f32, max as f32)),
            }),
            other => Err(float_only_err("clamp", other.dtype_str())),
        }
    }

    // ======================================================================
    // Reductions
    // ======================================================================

    /// Sum of all elements.
    fn sum(&self, py: Python<'_>) -> PyResult<PyObject> {
        match &self.data {
            TensorData::F64(t) => Ok(t.sum().into_pyobject(py)?.into_any().unbind()),
            TensorData::F32(t) => Ok(f64::from(t.sum()).into_pyobject(py)?.into_any().unbind()),
            TensorData::I64(t) => Ok(t.sum().into_pyobject(py)?.into_any().unbind()),
            TensorData::I32(t) => Ok(t.sum().into_pyobject(py)?.into_any().unbind()),
        }
    }

    /// Mean of all elements (float dtypes only).
    fn mean(&self) -> PyResult<f64> {
        match &self.data {
            TensorData::F64(t) => Ok(t.mean()),
            TensorData::F32(t) => Ok(f64::from(t.mean())),
            other => Err(float_only_err("mean", other.dtype_str())),
        }
    }

    /// Product of all elements.
    fn product(&self, py: Python<'_>) -> PyResult<PyObject> {
        match &self.data {
            TensorData::F64(t) => Ok(t.product().into_pyobject(py)?.into_any().unbind()),
            TensorData::F32(t) => Ok(f64::from(t.product())
                .into_pyobject(py)?
                .into_any()
                .unbind()),
            TensorData::I64(t) => Ok(t.product().into_pyobject(py)?.into_any().unbind()),
            TensorData::I32(t) => Ok(t.product().into_pyobject(py)?.into_any().unbind()),
        }
    }

    /// Minimum element.
    fn min(&self, py: Python<'_>) -> PyResult<PyObject> {
        let empty_err = || pyo3::exceptions::PyValueError::new_err("empty tensor");
        match &self.data {
            TensorData::F64(t) => {
                let v = t.min_element().ok_or_else(empty_err)?;
                Ok(v.into_pyobject(py)?.into_any().unbind())
            }
            TensorData::F32(t) => {
                let v = t.min_element().ok_or_else(empty_err)?;
                Ok(f64::from(v).into_pyobject(py)?.into_any().unbind())
            }
            TensorData::I64(t) => {
                let v = t.min_element().ok_or_else(empty_err)?;
                Ok(v.into_pyobject(py)?.into_any().unbind())
            }
            TensorData::I32(t) => {
                let v = t.min_element().ok_or_else(empty_err)?;
                Ok(v.into_pyobject(py)?.into_any().unbind())
            }
        }
    }

    /// Maximum element.
    fn max(&self, py: Python<'_>) -> PyResult<PyObject> {
        let empty_err = || pyo3::exceptions::PyValueError::new_err("empty tensor");
        match &self.data {
            TensorData::F64(t) => {
                let v = t.max_element().ok_or_else(empty_err)?;
                Ok(v.into_pyobject(py)?.into_any().unbind())
            }
            TensorData::F32(t) => {
                let v = t.max_element().ok_or_else(empty_err)?;
                Ok(f64::from(v).into_pyobject(py)?.into_any().unbind())
            }
            TensorData::I64(t) => {
                let v = t.max_element().ok_or_else(empty_err)?;
                Ok(v.into_pyobject(py)?.into_any().unbind())
            }
            TensorData::I32(t) => {
                let v = t.max_element().ok_or_else(empty_err)?;
                Ok(v.into_pyobject(py)?.into_any().unbind())
            }
        }
    }

    /// Sum along an axis.
    fn sum_axis(&self, axis: usize) -> PyResult<Self> {
        dispatch_all_try!(self, t => t.sum_axis(axis))
    }

    // ======================================================================
    // Shape manipulation
    // ======================================================================

    /// Reshape the tensor.
    fn reshape(&self, shape: Vec<usize>) -> PyResult<Self> {
        dispatch_all_try!(self, t => t.reshaped(shape.clone()))
    }

    /// Flatten the tensor to 1-D.
    fn flatten(&self) -> Self {
        dispatch_all!(self, t => t.flattened())
    }

    /// Transpose a 2-D tensor.
    fn transpose(&self) -> PyResult<Self> {
        dispatch_all_try!(self, t => t.transpose())
    }

    /// Permute dimensions.
    fn permute(&self, axes: Vec<usize>) -> PyResult<Self> {
        dispatch_all_try!(self, t => t.permute(&axes))
    }

    /// Insert a dimension of size 1 at ``axis``.
    fn unsqueeze(&self, axis: usize) -> PyResult<Self> {
        dispatch_all_try!(self, t => t.clone().unsqueeze(axis))
    }

    /// Remove all dimensions of size 1.
    fn squeeze(&self) -> Self {
        dispatch_all!(self, t => t.clone().squeeze())
    }

    /// Concatenate tensors along an axis.
    #[staticmethod]
    fn concat(tensors: Vec<PyRef<'_, PyTensor>>, axis: usize) -> PyResult<Self> {
        if tensors.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "concat requires at least one tensor",
            ));
        }
        let dtype = tensors[0].data.dtype_str();
        for t in &tensors {
            if t.data.dtype_str() != dtype {
                return Err(dtype_mismatch_err(dtype, t.data.dtype_str()));
            }
        }
        match dtype {
            "f64" => {
                let refs: Vec<&Tensor<f64>> = tensors
                    .iter()
                    .map(|t| match &t.data {
                        TensorData::F64(inner) => inner,
                        _ => unreachable!(),
                    })
                    .collect();
                Ok(Self {
                    data: TensorData::F64(Tensor::concat(&refs, axis).map_err(core_err)?),
                })
            }
            "f32" => {
                let refs: Vec<&Tensor<f32>> = tensors
                    .iter()
                    .map(|t| match &t.data {
                        TensorData::F32(inner) => inner,
                        _ => unreachable!(),
                    })
                    .collect();
                Ok(Self {
                    data: TensorData::F32(Tensor::concat(&refs, axis).map_err(core_err)?),
                })
            }
            "i64" => {
                let refs: Vec<&Tensor<i64>> = tensors
                    .iter()
                    .map(|t| match &t.data {
                        TensorData::I64(inner) => inner,
                        _ => unreachable!(),
                    })
                    .collect();
                Ok(Self {
                    data: TensorData::I64(Tensor::concat(&refs, axis).map_err(core_err)?),
                })
            }
            "i32" => {
                let refs: Vec<&Tensor<i32>> = tensors
                    .iter()
                    .map(|t| match &t.data {
                        TensorData::I32(inner) => inner,
                        _ => unreachable!(),
                    })
                    .collect();
                Ok(Self {
                    data: TensorData::I32(Tensor::concat(&refs, axis).map_err(core_err)?),
                })
            }
            _ => unreachable!(),
        }
    }

    /// Stack tensors along a new axis.
    #[staticmethod]
    fn stack(tensors: Vec<PyRef<'_, PyTensor>>, axis: usize) -> PyResult<Self> {
        if tensors.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "stack requires at least one tensor",
            ));
        }
        let dtype = tensors[0].data.dtype_str();
        for t in &tensors {
            if t.data.dtype_str() != dtype {
                return Err(dtype_mismatch_err(dtype, t.data.dtype_str()));
            }
        }
        match dtype {
            "f64" => {
                let refs: Vec<&Tensor<f64>> = tensors
                    .iter()
                    .map(|t| match &t.data {
                        TensorData::F64(inner) => inner,
                        _ => unreachable!(),
                    })
                    .collect();
                Ok(Self {
                    data: TensorData::F64(Tensor::stack(&refs, axis).map_err(core_err)?),
                })
            }
            "f32" => {
                let refs: Vec<&Tensor<f32>> = tensors
                    .iter()
                    .map(|t| match &t.data {
                        TensorData::F32(inner) => inner,
                        _ => unreachable!(),
                    })
                    .collect();
                Ok(Self {
                    data: TensorData::F32(Tensor::stack(&refs, axis).map_err(core_err)?),
                })
            }
            "i64" => {
                let refs: Vec<&Tensor<i64>> = tensors
                    .iter()
                    .map(|t| match &t.data {
                        TensorData::I64(inner) => inner,
                        _ => unreachable!(),
                    })
                    .collect();
                Ok(Self {
                    data: TensorData::I64(Tensor::stack(&refs, axis).map_err(core_err)?),
                })
            }
            "i32" => {
                let refs: Vec<&Tensor<i32>> = tensors
                    .iter()
                    .map(|t| match &t.data {
                        TensorData::I32(inner) => inner,
                        _ => unreachable!(),
                    })
                    .collect();
                Ok(Self {
                    data: TensorData::I32(Tensor::stack(&refs, axis).map_err(core_err)?),
                })
            }
            _ => unreachable!(),
        }
    }

    // ======================================================================
    // Slicing & indexing
    // ======================================================================

    /// Slice the tensor: ``t.slice([[start, stop], [start, stop], ...])``
    fn slice(&self, ranges: Vec<Vec<usize>>) -> PyResult<Self> {
        let slice_ranges: Vec<SliceRange> = ranges
            .iter()
            .map(|r| match r.len() {
                2 => Ok(SliceRange::range(r[0], r[1])),
                3 => Ok(SliceRange::new(r[0], r[1], r[2])),
                _ => Err(pyo3::exceptions::PyValueError::new_err(
                    "each range must be [start, stop] or [start, stop, step]",
                )),
            })
            .collect::<PyResult<_>>()?;
        dispatch_all_try!(self, t => t.slice(&slice_ranges))
    }

    /// Select along an axis at a single index, reducing dimensionality.
    fn select(&self, axis: usize, index: usize) -> PyResult<Self> {
        dispatch_all_try!(self, t => t.select(axis, index))
    }

    /// Select elements along an axis using an array of indices.
    fn index_select(&self, axis: usize, indices: Vec<usize>) -> PyResult<Self> {
        dispatch_all_try!(self, t => t.index_select(axis, &indices))
    }

    /// Select elements where mask is True, returning a flat 1-D tensor.
    fn masked_select(&self, mask: Vec<bool>) -> PyResult<Self> {
        dispatch_all_try!(self, t => t.masked_select(&mask))
    }

    // ======================================================================
    // Sorting
    // ======================================================================

    /// Sort elements in ascending order, returning a flat 1-D tensor.
    fn sort(&self) -> Self {
        dispatch_all!(self, t => t.sort())
    }

    /// Return indices that would sort all elements (flat).
    fn argsort(&self) -> Vec<usize> {
        match &self.data {
            TensorData::F64(t) => t.argsort().as_slice().to_vec(),
            TensorData::F32(t) => t.argsort().as_slice().to_vec(),
            TensorData::I64(t) => t.argsort().as_slice().to_vec(),
            TensorData::I32(t) => t.argsort().as_slice().to_vec(),
        }
    }

    /// Sort along an axis.
    fn sort_axis(&self, axis: usize) -> PyResult<Self> {
        dispatch_all_try!(self, t => t.sort_axis(axis))
    }

    // ======================================================================
    // Linear algebra (on tensor directly)
    // ======================================================================

    /// Dot product (all dtypes).
    fn dot(&self, other: &PyTensor, py: Python<'_>) -> PyResult<PyObject> {
        if self.data.dtype_str() != other.data.dtype_str() {
            return Err(dtype_mismatch_err(
                self.data.dtype_str(),
                other.data.dtype_str(),
            ));
        }
        match (&self.data, &other.data) {
            (TensorData::F64(a), TensorData::F64(b)) => Ok(a
                .dot(b)
                .map_err(core_err)?
                .into_pyobject(py)?
                .into_any()
                .unbind()),
            (TensorData::F32(a), TensorData::F32(b)) => Ok(f64::from(a.dot(b).map_err(core_err)?)
                .into_pyobject(py)?
                .into_any()
                .unbind()),
            (TensorData::I64(a), TensorData::I64(b)) => Ok(a
                .dot(b)
                .map_err(core_err)?
                .into_pyobject(py)?
                .into_any()
                .unbind()),
            (TensorData::I32(a), TensorData::I32(b)) => Ok(a
                .dot(b)
                .map_err(core_err)?
                .into_pyobject(py)?
                .into_any()
                .unbind()),
            _ => unreachable!(),
        }
    }

    /// Euclidean norm (float dtypes only).
    fn norm(&self) -> PyResult<f64> {
        match &self.data {
            TensorData::F64(t) => t.norm().map_err(core_err),
            TensorData::F32(t) => t.norm().map(f64::from).map_err(core_err),
            other => Err(float_only_err("norm", other.dtype_str())),
        }
    }

    /// Determinant (float dtypes only).
    fn det(&self) -> PyResult<f64> {
        match &self.data {
            TensorData::F64(t) => t.det().map_err(core_err),
            TensorData::F32(t) => t.det().map(f64::from).map_err(core_err),
            other => Err(float_only_err("det", other.dtype_str())),
        }
    }

    /// Matrix inverse (float dtypes only).
    fn inv(&self) -> PyResult<Self> {
        dispatch_float_try!(self, "inv", t => t.inv())
    }

    /// Solve linear system ``self @ x = b`` (float dtypes only).
    fn solve(&self, b: &PyTensor) -> PyResult<Self> {
        if self.data.dtype_str() != b.data.dtype_str() {
            return Err(dtype_mismatch_err(
                self.data.dtype_str(),
                b.data.dtype_str(),
            ));
        }
        match (&self.data, &b.data) {
            (TensorData::F64(a), TensorData::F64(bv)) => Ok(Self {
                data: TensorData::F64(a.solve(bv).map_err(core_err)?),
            }),
            (TensorData::F32(a), TensorData::F32(bv)) => Ok(Self {
                data: TensorData::F32(a.solve(bv).map_err(core_err)?),
            }),
            _ => Err(float_only_err("solve", self.data.dtype_str())),
        }
    }

    /// Least squares solution (float dtypes only).
    fn lstsq(&self, b: &PyTensor) -> PyResult<Self> {
        if self.data.dtype_str() != b.data.dtype_str() {
            return Err(dtype_mismatch_err(
                self.data.dtype_str(),
                b.data.dtype_str(),
            ));
        }
        match (&self.data, &b.data) {
            (TensorData::F64(a), TensorData::F64(bv)) => Ok(Self {
                data: TensorData::F64(a.lstsq(bv).map_err(core_err)?),
            }),
            (TensorData::F32(a), TensorData::F32(bv)) => Ok(Self {
                data: TensorData::F32(a.lstsq(bv).map_err(core_err)?),
            }),
            _ => Err(float_only_err("lstsq", self.data.dtype_str())),
        }
    }

    /// Matrix-vector multiply.
    fn matvec(&self, x: &PyTensor) -> PyResult<Self> {
        if self.data.dtype_str() != x.data.dtype_str() {
            return Err(dtype_mismatch_err(
                self.data.dtype_str(),
                x.data.dtype_str(),
            ));
        }
        match (&self.data, &x.data) {
            (TensorData::F64(a), TensorData::F64(b)) => Ok(Self {
                data: TensorData::F64(a.matvec(b).map_err(core_err)?),
            }),
            (TensorData::F32(a), TensorData::F32(b)) => Ok(Self {
                data: TensorData::F32(a.matvec(b).map_err(core_err)?),
            }),
            (TensorData::I64(a), TensorData::I64(b)) => Ok(Self {
                data: TensorData::I64(a.matvec(b).map_err(core_err)?),
            }),
            (TensorData::I32(a), TensorData::I32(b)) => Ok(Self {
                data: TensorData::I32(a.matvec(b).map_err(core_err)?),
            }),
            _ => unreachable!(),
        }
    }

    // ======================================================================
    // Map operations
    // ======================================================================

    /// Apply a Python callable element-wise (always uses ``f64``).
    fn apply(&self, _py: Python<'_>, func: &Bound<'_, PyAny>) -> PyResult<Self> {
        // Convert to f64 for the callback, then wrap back as f64.
        let f64_tensor = self.data.to_f64();
        let data = f64_tensor.as_slice();
        let mut new_data = Vec::with_capacity(data.len());
        for &val in data {
            let result = func.call1((val,))?;
            let new_val: f64 = result.extract()?;
            new_data.push(new_val);
        }
        let inner = Tensor::from_vec(new_data, f64_tensor.shape().to_vec()).map_err(core_err)?;
        Ok(Self {
            data: TensorData::F64(inner),
        })
    }
}

// =========================================================================
// Private helpers for PyTensor
// =========================================================================

impl PyTensor {
    /// Multi-dimensional getitem helper.
    fn getitem_multi(&self, py: Python<'_>, indices: &[isize]) -> PyResult<PyObject> {
        macro_rules! impl_getitem_multi {
            ($t:expr, $variant:ident) => {{
                let ndim = $t.ndim();
                if indices.len() == ndim {
                    let mut idx = Vec::with_capacity(ndim);
                    for (i, &raw) in indices.iter().enumerate() {
                        idx.push(normalize_index(raw, $t.shape()[i])?);
                    }
                    let val = $t.get(&idx).map_err(core_err)?;
                    return self.scalar_to_py(py, &TensorData::$variant(Tensor::scalar(*val)));
                }
                // Partial indexing
                let mut result = $t.clone();
                for &raw in indices {
                    let dim = result.shape()[0];
                    let idx = normalize_index(raw, dim)?;
                    result = result.select(0, idx).map_err(core_err)?;
                }
                if result.ndim() == 0 {
                    self.scalar_to_py(py, &TensorData::$variant(result))
                } else {
                    Ok(Self {
                        data: TensorData::$variant(result),
                    }
                    .into_pyobject(py)?
                    .into_any()
                    .unbind())
                }
            }};
        }

        match &self.data {
            TensorData::F64(t) => impl_getitem_multi!(t, F64),
            TensorData::F32(t) => impl_getitem_multi!(t, F32),
            TensorData::I64(t) => impl_getitem_multi!(t, I64),
            TensorData::I32(t) => impl_getitem_multi!(t, I32),
        }
    }

    /// Convert a scalar (0-d) TensorData to a Python scalar object.
    fn scalar_to_py(&self, py: Python<'_>, td: &TensorData) -> PyResult<PyObject> {
        match td {
            TensorData::F64(t) => Ok(t.as_slice()[0].into_pyobject(py)?.into_any().unbind()),
            TensorData::F32(t) => Ok(f64::from(t.as_slice()[0])
                .into_pyobject(py)?
                .into_any()
                .unbind()),
            TensorData::I64(t) => Ok(t.as_slice()[0].into_pyobject(py)?.into_any().unbind()),
            TensorData::I32(t) => Ok(t.as_slice()[0].into_pyobject(py)?.into_any().unbind()),
        }
    }
}

/// Normalize a possibly-negative Python index to a valid positive index.
fn normalize_index(idx: isize, dim: usize) -> PyResult<usize> {
    #[allow(clippy::cast_possible_wrap)]
    let len = dim as isize;
    let normalized = if idx < 0 { idx + len } else { idx };
    if normalized < 0 || normalized >= len {
        return Err(pyo3::exceptions::PyIndexError::new_err(format!(
            "index {idx} is out of bounds for axis with size {dim}"
        )));
    }
    #[allow(clippy::cast_sign_loss)]
    Ok(normalized as usize)
}

// =========================================================================
// P2.11 — Sparse matrix bindings
// =========================================================================

/// A sparse matrix in CSR (Compressed Sparse Row) format.
///
/// Wraps [`scivex_core::linalg::sparse::CsrMatrix<f64>`] for use from Python.
#[pyclass(name = "CsrMatrix")]
#[derive(Clone)]
pub struct PyCsrMatrix {
    inner: CsrMatrix<f64>,
}

#[pymethods]
impl PyCsrMatrix {
    /// Build a CSR matrix from triplet arrays and a shape.
    ///
    /// Parameters
    /// ----------
    /// rows : list[int]
    ///     Row indices of non-zero entries.
    /// cols : list[int]
    ///     Column indices of non-zero entries.
    /// values : list[float]
    ///     Values of non-zero entries.
    /// shape : tuple[int, int]
    ///     ``(nrows, ncols)`` dimensions of the matrix.
    #[new]
    fn new(
        rows: Vec<usize>,
        cols: Vec<usize>,
        values: Vec<f64>,
        shape: (usize, usize),
    ) -> PyResult<Self> {
        let inner =
            CsrMatrix::from_triplets(shape.0, shape.1, rows, cols, values).map_err(core_err)?;
        Ok(Self { inner })
    }

    /// Build a CSR matrix from triplet arrays and a shape (static constructor).
    #[staticmethod]
    fn from_triplets(
        rows: Vec<usize>,
        cols: Vec<usize>,
        values: Vec<f64>,
        shape: (usize, usize),
    ) -> PyResult<Self> {
        let inner =
            CsrMatrix::from_triplets(shape.0, shape.1, rows, cols, values).map_err(core_err)?;
        Ok(Self { inner })
    }

    /// Convert to a dense ``Tensor``.
    fn to_dense(&self) -> PyTensor {
        PyTensor::from_f64(self.inner.to_dense())
    }

    /// Multiply this sparse matrix by a dense vector.
    ///
    /// Parameters
    /// ----------
    /// x : Tensor
    ///     A 1-D tensor of length ``ncols``.
    ///
    /// Returns
    /// -------
    /// Tensor
    ///     A 1-D tensor of length ``nrows``.
    fn matvec(&self, x: &PyTensor) -> PyResult<PyTensor> {
        let xt = x.as_f64()?;
        let result = self.inner.matvec(xt).map_err(core_err)?;
        Ok(PyTensor::from_f64(result))
    }

    /// Return the shape ``(nrows, ncols)`` as a list.
    fn shape(&self) -> Vec<usize> {
        let (r, c) = self.inner.shape();
        vec![r, c]
    }

    /// Return the number of stored non-zero entries.
    fn nnz(&self) -> usize {
        self.inner.nnz()
    }

    fn __repr__(&self) -> String {
        let (r, c) = self.inner.shape();
        format!("CsrMatrix(shape=[{}, {}], nnz={})", r, c, self.inner.nnz())
    }
}

/// A sparse matrix in CSC (Compressed Sparse Column) format.
///
/// Wraps [`scivex_core::linalg::sparse::CscMatrix<f64>`] for use from Python.
#[pyclass(name = "CscMatrix")]
#[derive(Clone)]
pub struct PyCscMatrix {
    inner: CscMatrix<f64>,
}

#[pymethods]
impl PyCscMatrix {
    /// Build a CSC matrix from triplet arrays and a shape.
    ///
    /// Parameters
    /// ----------
    /// rows : list[int]
    ///     Row indices of non-zero entries.
    /// cols : list[int]
    ///     Column indices of non-zero entries.
    /// values : list[float]
    ///     Values of non-zero entries.
    /// shape : tuple[int, int]
    ///     ``(nrows, ncols)`` dimensions of the matrix.
    #[new]
    fn new(
        rows: Vec<usize>,
        cols: Vec<usize>,
        values: Vec<f64>,
        shape: (usize, usize),
    ) -> PyResult<Self> {
        let inner =
            CscMatrix::from_triplets(shape.0, shape.1, rows, cols, values).map_err(core_err)?;
        Ok(Self { inner })
    }

    /// Build a CSC matrix from triplet arrays and a shape (static constructor).
    #[staticmethod]
    fn from_triplets(
        rows: Vec<usize>,
        cols: Vec<usize>,
        values: Vec<f64>,
        shape: (usize, usize),
    ) -> PyResult<Self> {
        let inner =
            CscMatrix::from_triplets(shape.0, shape.1, rows, cols, values).map_err(core_err)?;
        Ok(Self { inner })
    }

    /// Convert to a dense ``Tensor``.
    fn to_dense(&self) -> PyTensor {
        PyTensor::from_f64(self.inner.to_dense())
    }

    /// Multiply this sparse matrix by a dense vector.
    ///
    /// Parameters
    /// ----------
    /// x : Tensor
    ///     A 1-D tensor of length ``ncols``.
    ///
    /// Returns
    /// -------
    /// Tensor
    ///     A 1-D tensor of length ``nrows``.
    fn matvec(&self, x: &PyTensor) -> PyResult<PyTensor> {
        let xt = x.as_f64()?;
        let result = self.inner.matvec(xt).map_err(core_err)?;
        Ok(PyTensor::from_f64(result))
    }

    /// Return the shape ``(nrows, ncols)`` as a list.
    fn shape(&self) -> Vec<usize> {
        let (r, c) = self.inner.shape();
        vec![r, c]
    }

    /// Return the number of stored non-zero entries.
    fn nnz(&self) -> usize {
        self.inner.nnz()
    }

    fn __repr__(&self) -> String {
        let (r, c) = self.inner.shape();
        format!("CscMatrix(shape=[{}, {}], nnz={})", r, c, self.inner.nnz())
    }
}

/// A sparse matrix in COO (coordinate / triplet) format.
///
/// Wraps [`scivex_core::linalg::sparse::CooMatrix<f64>`] for use from Python.
#[pyclass(name = "CooMatrix")]
#[derive(Clone)]
pub struct PyCooMatrix {
    inner: CooMatrix<f64>,
}

#[pymethods]
impl PyCooMatrix {
    /// Build a COO matrix from triplet arrays and a shape.
    ///
    /// Parameters
    /// ----------
    /// rows : list[int]
    ///     Row indices of non-zero entries.
    /// cols : list[int]
    ///     Column indices of non-zero entries.
    /// values : list[float]
    ///     Values of non-zero entries.
    /// shape : tuple[int, int]
    ///     ``(nrows, ncols)`` dimensions of the matrix.
    #[new]
    fn new(
        rows: Vec<usize>,
        cols: Vec<usize>,
        values: Vec<f64>,
        shape: (usize, usize),
    ) -> PyResult<Self> {
        let inner =
            CooMatrix::from_triplets(shape.0, shape.1, rows, cols, values).map_err(core_err)?;
        Ok(Self { inner })
    }

    /// Build a COO matrix from triplet arrays and a shape (static constructor).
    #[staticmethod]
    fn from_triplets(
        rows: Vec<usize>,
        cols: Vec<usize>,
        values: Vec<f64>,
        shape: (usize, usize),
    ) -> PyResult<Self> {
        let inner =
            CooMatrix::from_triplets(shape.0, shape.1, rows, cols, values).map_err(core_err)?;
        Ok(Self { inner })
    }

    /// Convert to a dense ``Tensor``.
    fn to_dense(&self) -> PyTensor {
        PyTensor::from_f64(self.inner.to_dense())
    }

    /// Multiply this sparse matrix by a dense vector (via CSR conversion).
    ///
    /// Parameters
    /// ----------
    /// x : Tensor
    ///     A 1-D tensor of length ``ncols``.
    ///
    /// Returns
    /// -------
    /// Tensor
    ///     A 1-D tensor of length ``nrows``.
    fn matvec(&self, x: &PyTensor) -> PyResult<PyTensor> {
        let xt = x.as_f64()?;
        let csr = self.inner.to_csr();
        let result = csr.matvec(xt).map_err(core_err)?;
        Ok(PyTensor::from_f64(result))
    }

    /// Return the shape ``(nrows, ncols)`` as a list.
    fn shape(&self) -> Vec<usize> {
        let (r, c) = self.inner.shape();
        vec![r, c]
    }

    /// Return the number of stored entries (may include duplicates).
    fn nnz(&self) -> usize {
        self.inner.nnz()
    }

    fn __repr__(&self) -> String {
        let (r, c) = self.inner.shape();
        format!("CooMatrix(shape=[{}, {}], nnz={})", r, c, self.inner.nnz())
    }
}

// =========================================================================
// P2.13 — Einstein summation
// =========================================================================

/// Perform Einstein summation on one or more tensors.
///
/// The ``subscripts`` string follows NumPy-style notation, e.g.
/// ``"ij,jk->ik"`` for matrix multiplication.
///
/// Parameters
/// ----------
/// subscripts : str
///     Einstein summation subscript string.
/// operands : list[Tensor]
///     Input tensors whose ranks match the subscript indices.
///
/// Returns
/// -------
/// Tensor
///     The result of the contraction.
///
/// Examples
/// --------
/// >>> a = sv.Tensor([[1, 2], [3, 4]])
/// >>> b = sv.Tensor([[5, 6], [7, 8]])
/// >>> sv.einsum("ij,jk->ik", [a, b])
#[pyfunction]
#[pyo3(name = "einsum")]
pub fn py_einsum(subscripts: &str, operands: Vec<PyRef<'_, PyTensor>>) -> PyResult<PyTensor> {
    // einsum currently only supports f64; convert if needed.
    let f64_tensors: Vec<Tensor<f64>> = operands.iter().map(|t| t.to_f64_tensor()).collect();
    let refs: Vec<&Tensor<f64>> = f64_tensors.iter().collect();
    let result = einsum(subscripts, &refs).map_err(core_err)?;
    Ok(PyTensor::from_f64(result))
}

/// Register sparse matrix classes and Einstein summation into the parent module.
pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    parent.add_class::<PyCsrMatrix>()?;
    parent.add_class::<PyCscMatrix>()?;
    parent.add_class::<PyCooMatrix>()?;
    parent.add_function(wrap_pyfunction!(py_einsum, parent)?)?;
    Ok(())
}
