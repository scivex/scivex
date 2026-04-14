//! Tensor bindings for JavaScript.

use scivex_core::Tensor;
use wasm_bindgen::prelude::*;

/// A multi-dimensional tensor of f64 values.
#[wasm_bindgen]
pub struct WasmTensor {
    inner: Tensor<f64>,
}

#[wasm_bindgen]
impl WasmTensor {
    /// Create a tensor from a flat Float64Array and a shape.
    #[wasm_bindgen(js_name = "fromArray")]
    pub fn from_array(data: &[f64], shape: &[usize]) -> Result<WasmTensor, JsError> {
        let t = Tensor::from_vec(data.to_vec(), shape.to_vec())
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(WasmTensor { inner: t })
    }

    /// Create a tensor of zeros with the given shape.
    #[wasm_bindgen(js_name = "zeros")]
    pub fn zeros(shape: &[usize]) -> WasmTensor {
        WasmTensor {
            inner: Tensor::zeros(shape.to_vec()),
        }
    }

    /// Create a tensor of ones with the given shape.
    #[wasm_bindgen(js_name = "ones")]
    pub fn ones(shape: &[usize]) -> WasmTensor {
        WasmTensor {
            inner: Tensor::ones(shape.to_vec()),
        }
    }

    /// Create an identity matrix of size n×n.
    #[wasm_bindgen(js_name = "eye")]
    pub fn eye(n: usize) -> WasmTensor {
        WasmTensor {
            inner: Tensor::eye(n),
        }
    }

    /// Return the shape.
    pub fn shape(&self) -> Vec<usize> {
        self.inner.shape().to_vec()
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.inner.shape().len()
    }

    /// Total number of elements.
    pub fn size(&self) -> usize {
        self.inner.as_slice().len()
    }

    /// Return the data as a Float64Array.
    #[wasm_bindgen(js_name = "toArray")]
    pub fn to_array(&self) -> Vec<f64> {
        self.inner.as_slice().to_vec()
    }

    /// Get a single element by flat index.
    pub fn get(&self, index: usize) -> Result<f64, JsError> {
        let s = self.inner.as_slice();
        if index >= s.len() {
            return Err(JsError::new("index out of bounds"));
        }
        Ok(s[index])
    }

    /// Reshape the tensor (returns a new tensor).
    pub fn reshape(&self, new_shape: &[usize]) -> Result<WasmTensor, JsError> {
        let t = self
            .inner
            .reshaped(new_shape.to_vec())
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(WasmTensor { inner: t })
    }

    /// Transpose (2-D only).
    pub fn transpose(&self) -> Result<WasmTensor, JsError> {
        let t = self
            .inner
            .transpose()
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(WasmTensor { inner: t })
    }

    /// Element-wise addition.
    pub fn add(&self, other: &WasmTensor) -> Result<WasmTensor, JsError> {
        let t = self
            .inner
            .add_checked(&other.inner)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(WasmTensor { inner: t })
    }

    /// Element-wise subtraction.
    pub fn sub(&self, other: &WasmTensor) -> Result<WasmTensor, JsError> {
        let t = self
            .inner
            .sub_checked(&other.inner)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(WasmTensor { inner: t })
    }

    /// Element-wise multiplication.
    pub fn mul(&self, other: &WasmTensor) -> Result<WasmTensor, JsError> {
        let t = self
            .inner
            .mul_checked(&other.inner)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(WasmTensor { inner: t })
    }

    /// Matrix multiplication.
    pub fn matmul(&self, other: &WasmTensor) -> Result<WasmTensor, JsError> {
        let t = self
            .inner
            .matmul(&other.inner)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(WasmTensor { inner: t })
    }

    /// Dot product of two 1-D tensors.
    pub fn dot(&self, other: &WasmTensor) -> Result<f64, JsError> {
        self.inner
            .dot(&other.inner)
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Add a scalar to every element.
    #[wasm_bindgen(js_name = "addScalar")]
    pub fn add_scalar(&self, val: f64) -> WasmTensor {
        WasmTensor {
            inner: self.inner.map(|x| x + val),
        }
    }

    /// Multiply every element by a scalar.
    #[wasm_bindgen(js_name = "mulScalar")]
    pub fn mul_scalar(&self, val: f64) -> WasmTensor {
        WasmTensor {
            inner: &self.inner * val,
        }
    }

    /// Sum all elements.
    pub fn sum(&self) -> f64 {
        self.inner.as_slice().iter().sum()
    }

    /// Mean of all elements.
    pub fn mean(&self) -> f64 {
        let s = self.inner.as_slice();
        if s.is_empty() {
            return 0.0;
        }
        s.iter().sum::<f64>() / s.len() as f64
    }

    /// Determinant (square matrix only).
    pub fn det(&self) -> Result<f64, JsError> {
        scivex_core::linalg::det(&self.inner).map_err(|e| JsError::new(&e.to_string()))
    }

    /// String representation.
    #[wasm_bindgen(js_name = "toString")]
    pub fn to_string_js(&self) -> String {
        format!(
            "Tensor(shape={:?}, data={:?})",
            self.inner.shape(),
            self.inner.as_slice()
        )
    }
}

impl WasmTensor {
    pub(crate) fn from_inner(inner: Tensor<f64>) -> Self {
        Self { inner }
    }

    pub(crate) fn inner(&self) -> &Tensor<f64> {
        &self.inner
    }
}
