//! Python bindings for [`scivex_nn`] — neural networks with autograd.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use scivex_core::Tensor;
use scivex_core::random::Rng;
use scivex_nn::layer::gnn::{GATConv, GCNConv, SAGEConv};
use scivex_nn::layer::{
    AvgPool1d, AvgPool2d, BatchNorm1d, BatchNorm2d, Conv1d, Conv2d, Conv3d, Dropout, Embedding,
    FlashAttention, Flatten, GRU, GroupedQueryAttention, LSTM, Layer, LayerNorm, Linear, MaxPool1d,
    MaxPool2d, MultiHeadAttention, MultiQueryAttention, ReLU, RotaryPositionalEncoding, Sequential,
    Sigmoid, SimpleRNN, SinusoidalPositionalEncoding, Tanh, TransformerDecoderLayer,
    TransformerEncoderLayer, causal_mask,
};
use scivex_nn::onnx::{OnnxInferenceSession, load_onnx};
use scivex_nn::optim::scheduler::LrScheduler;
use scivex_nn::optim::{self, Optimizer};
use scivex_nn::training::{Callback, CallbackAction, EarlyStopping, ModelCheckpoint};
use scivex_nn::variable::Variable;

use crate::tensor::{PyTensor, nested_list_to_flat};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn nn_err(e: impl std::fmt::Debug) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(format!("{e:?}"))
}

fn consumed_err() -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err("layer consumed by Sequential")
}

// ---------------------------------------------------------------------------
// Variable — autograd tensor
// ---------------------------------------------------------------------------

/// Autograd-enabled tensor variable for neural network computation.
#[pyclass(name = "Variable", unsendable)]
pub struct PyVariable {
    pub(crate) inner: Variable<f64>,
}

#[pymethods]
impl PyVariable {
    /// Create a new Variable from data (nested list, flat list, or Tensor).
    ///
    /// Parameters: data — input tensor data, requires_grad — whether to track gradients.
    #[new]
    #[pyo3(signature = (data, requires_grad=false))]
    fn new(data: &Bound<'_, PyAny>, requires_grad: bool) -> PyResult<Self> {
        let tensor = extract_tensor(data)?;
        Ok(Self {
            inner: Variable::new(tensor, requires_grad),
        })
    }

    /// Underlying tensor data.
    fn data(&self) -> PyTensor {
        PyTensor::from_f64(self.inner.data())
    }

    /// Shape of the variable.
    fn shape(&self) -> Vec<usize> {
        self.inner.shape()
    }

    /// Gradient tensor (None if not computed yet).
    fn grad(&self) -> Option<PyTensor> {
        self.inner.grad().map(PyTensor::from_f64)
    }

    /// Whether this variable tracks gradients.
    fn requires_grad(&self) -> bool {
        self.inner.requires_grad()
    }

    /// Run backward pass (reverse-mode autodiff).
    fn backward(&self) {
        self.inner.backward();
    }

    /// Reset gradient to zero.
    fn zero_grad(&self) {
        self.inner.zero_grad();
    }

    /// Detach from computation graph.
    fn detach(&self) -> Self {
        Self {
            inner: self.inner.detach(),
        }
    }

    /// Return a string representation of this Variable.
    fn __repr__(&self) -> String {
        let d = self.inner.data();
        format!(
            "Variable(shape={:?}, requires_grad={})",
            d.shape(),
            self.inner.requires_grad()
        )
    }
}

/// Extract a Tensor<f64> from a Python object (nested list, flat list, or PyTensor).
fn extract_tensor(obj: &Bound<'_, PyAny>) -> PyResult<Tensor<f64>> {
    // Try PyTensor first
    if let Ok(t) = obj.extract::<PyTensor>() {
        return Ok(t.to_f64_tensor());
    }
    // Try nested / flat list
    let (flat, shape) = nested_list_to_flat(obj)?;
    if shape.is_empty() {
        Ok(Tensor::scalar(flat[0]))
    } else {
        Tensor::from_vec(flat, shape)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }
}

// ---------------------------------------------------------------------------
// Convenience: nn.tensor()
// ---------------------------------------------------------------------------

/// Create a Variable from data.
#[pyfunction]
#[pyo3(signature = (data, requires_grad=false))]
fn tensor(data: &Bound<'_, PyAny>, requires_grad: bool) -> PyResult<PyVariable> {
    PyVariable::new(data, requires_grad)
}

// ===========================================================================
// LAYER CLASSES
// ===========================================================================

// ---- Macro for layers with Option<T> inner ----

macro_rules! stateful_layer {
    (
        $py_name:ident, $py_str:literal, $rust_ty:ty,
        #[pyo3(signature = ($($sig:tt)*))]
        new($($p:ident : $pt:ty),*) => $body:expr
        $(, extra { $($extra:tt)* })?
    ) => {
        #[pyclass(name = $py_str, unsendable)]
        pub struct $py_name {
            inner: Option<$rust_ty>,
        }

        #[pymethods]
        impl $py_name {
            #[new]
            #[pyo3(signature = ($($sig)*))]
            #[allow(clippy::new_without_default, unused_mut)]
            fn new($($p: $pt),*) -> PyResult<Self> {
                let layer: $rust_ty = $body;
                Ok(Self { inner: Some(layer) })
            }

            /// Run a forward pass through this layer.
            ///
            /// Parameters: x — input Variable. Returns the output Variable.
            fn forward(&self, x: &PyVariable) -> PyResult<PyVariable> {
                let l = self.inner.as_ref().ok_or_else(consumed_err)?;
                let out = l.forward(&x.inner).map_err(nn_err)?;
                Ok(PyVariable { inner: out })
            }

            /// Return all trainable parameters of this layer.
            fn parameters(&self) -> Vec<PyVariable> {
                self.inner.as_ref()
                    .map(|l| Layer::parameters(l).into_iter()
                        .map(|v| PyVariable { inner: v }).collect())
                    .unwrap_or_default()
            }

            /// Set the layer to training mode.
            fn train(&mut self) {
                if let Some(ref mut l) = self.inner { Layer::train(l); }
            }

            /// Set the layer to evaluation mode.
            fn eval(&mut self) {
                if let Some(ref mut l) = self.inner { Layer::eval(l); }
            }

            $($($extra)*)?
        }
    };
}

// ---- Linear ----
stateful_layer!(
    PyNnLinear, "Linear", Linear<f64>,
    #[pyo3(signature = (in_features, out_features, bias=None, seed=None))]
    new(in_features: usize, out_features: usize, bias: Option<bool>, seed: Option<u64>) => {
        let mut rng = Rng::new(seed.unwrap_or(42));
        Linear::new(in_features, out_features, bias.unwrap_or(true), &mut rng)
    }
);

// ---- Conv1d ----
stateful_layer!(
    PyConv1d, "Conv1d", Conv1d<f64>,
    #[pyo3(signature = (in_channels, out_channels, kernel_size, bias=None, seed=None))]
    new(in_channels: usize, out_channels: usize, kernel_size: usize,
        bias: Option<bool>, seed: Option<u64>) => {
        let mut rng = Rng::new(seed.unwrap_or(42));
        Conv1d::new(in_channels, out_channels, kernel_size, bias.unwrap_or(true), &mut rng)
    }
);

// ---- Conv2d ----
stateful_layer!(
    PyConv2d, "Conv2d", Conv2d<f64>,
    #[pyo3(signature = (in_channels, out_channels, kernel_h, kernel_w, bias=None, seed=None))]
    new(in_channels: usize, out_channels: usize, kernel_h: usize, kernel_w: usize,
        bias: Option<bool>, seed: Option<u64>) => {
        let mut rng = Rng::new(seed.unwrap_or(42));
        Conv2d::new(in_channels, out_channels, (kernel_h, kernel_w), bias.unwrap_or(true), &mut rng)
    }
);

// ---- BatchNorm1d ----
stateful_layer!(
    PyBatchNorm1d, "BatchNorm1d", BatchNorm1d<f64>,
    #[pyo3(signature = (num_features))]
    new(num_features: usize) => { BatchNorm1d::new(num_features) }
);

// ---- BatchNorm2d ----
stateful_layer!(
    PyBatchNorm2d, "BatchNorm2d", BatchNorm2d<f64>,
    #[pyo3(signature = (num_channels))]
    new(num_channels: usize) => { BatchNorm2d::new(num_channels) }
);

// ---- Dropout ----
stateful_layer!(
    PyDropout, "Dropout", Dropout<f64>,
    #[pyo3(signature = (p, seed=None))]
    new(p: f64, seed: Option<u64>) => {
        let rng = Rng::new(seed.unwrap_or(42));
        Dropout::new(p, rng)
    }
);

// ---- Embedding ----
stateful_layer!(
    PyEmbedding, "Embedding", Embedding<f64>,
    #[pyo3(signature = (num_embeddings, dim, seed=None))]
    new(num_embeddings: usize, dim: usize, seed: Option<u64>) => {
        let mut rng = Rng::new(seed.unwrap_or(42));
        Embedding::new(num_embeddings, dim, &mut rng)
    }
);

// ---- LayerNorm ----
stateful_layer!(
    PyLayerNorm, "LayerNorm", LayerNorm<f64>,
    #[pyo3(signature = (num_features))]
    new(num_features: usize) => { LayerNorm::new(num_features) }
);

// ---- LSTM ----
stateful_layer!(
    PyLSTM, "LSTM", LSTM<f64>,
    #[pyo3(signature = (input_size, hidden_size, seq_len, seed=None))]
    new(input_size: usize, hidden_size: usize, seq_len: usize, seed: Option<u64>) => {
        let mut rng = Rng::new(seed.unwrap_or(42));
        LSTM::new(input_size, hidden_size, seq_len, &mut rng)
    }
);

// ---- GRU ----
stateful_layer!(
    PyGRU, "GRU", GRU<f64>,
    #[pyo3(signature = (input_size, hidden_size, seq_len, seed=None))]
    new(input_size: usize, hidden_size: usize, seq_len: usize, seed: Option<u64>) => {
        let mut rng = Rng::new(seed.unwrap_or(42));
        GRU::new(input_size, hidden_size, seq_len, &mut rng)
    }
);

// ---- MultiHeadAttention ----
stateful_layer!(
    PyMultiHeadAttention, "MultiHeadAttention", MultiHeadAttention<f64>,
    #[pyo3(signature = (d_model, num_heads, seq_len, seed=None))]
    new(d_model: usize, num_heads: usize, seq_len: usize, seed: Option<u64>) => {
        let mut rng = Rng::new(seed.unwrap_or(42));
        MultiHeadAttention::new(d_model, num_heads, seq_len, &mut rng)
    }
);

// ---- Conv3d ----
stateful_layer!(
    PyConv3d, "Conv3d", Conv3d<f64>,
    #[pyo3(signature = (in_channels, out_channels, kernel_d, kernel_h, kernel_w, bias=None, seed=None))]
    new(in_channels: usize, out_channels: usize, kernel_d: usize, kernel_h: usize, kernel_w: usize,
        bias: Option<bool>, seed: Option<u64>) => {
        let mut rng = Rng::new(seed.unwrap_or(42));
        Conv3d::new(in_channels, out_channels, (kernel_d, kernel_h, kernel_w), bias.unwrap_or(true), &mut rng)
    }
);

// ---- SimpleRNN ----
stateful_layer!(
    PySimpleRNN, "SimpleRNN", SimpleRNN<f64>,
    #[pyo3(signature = (input_size, hidden_size, seq_len, seed=None))]
    new(input_size: usize, hidden_size: usize, seq_len: usize, seed: Option<u64>) => {
        let mut rng = Rng::new(seed.unwrap_or(42));
        SimpleRNN::new(input_size, hidden_size, seq_len, &mut rng)
    }
);

// ---- TransformerEncoderLayer ----
stateful_layer!(
    PyTransformerEncoderLayer, "TransformerEncoderLayer", TransformerEncoderLayer<f64>,
    #[pyo3(signature = (d_model, num_heads, d_ff, seq_len, seed=None))]
    new(d_model: usize, num_heads: usize, d_ff: usize, seq_len: usize, seed: Option<u64>) => {
        let mut rng = Rng::new(seed.unwrap_or(42));
        TransformerEncoderLayer::new(d_model, num_heads, d_ff, seq_len, &mut rng)
    }
);

// ---- TransformerDecoderLayer ----
stateful_layer!(
    PyTransformerDecoderLayer, "TransformerDecoderLayer", TransformerDecoderLayer<f64>,
    #[pyo3(signature = (d_model, num_heads, d_ff, seq_len, pre_norm=None, seed=None))]
    new(d_model: usize, num_heads: usize, d_ff: usize, seq_len: usize,
        pre_norm: Option<bool>, seed: Option<u64>) => {
        let mut rng = Rng::new(seed.unwrap_or(42));
        TransformerDecoderLayer::new(d_model, num_heads, d_ff, seq_len, pre_norm.unwrap_or(false), &mut rng)
            .map_err(nn_err)?
    }
);

// ---- MultiQueryAttention ----
stateful_layer!(
    PyMultiQueryAttention, "MultiQueryAttention", MultiQueryAttention<f64>,
    #[pyo3(signature = (d_model, num_heads, seq_len, seed=None))]
    new(d_model: usize, num_heads: usize, seq_len: usize, seed: Option<u64>) => {
        let mut rng = Rng::new(seed.unwrap_or(42));
        MultiQueryAttention::new(d_model, num_heads, seq_len, &mut rng).map_err(nn_err)?
    }
);

// ---- GroupedQueryAttention ----
stateful_layer!(
    PyGroupedQueryAttention, "GroupedQueryAttention", GroupedQueryAttention<f64>,
    #[pyo3(signature = (d_model, num_heads, num_kv_heads, seq_len, seed=None))]
    new(d_model: usize, num_heads: usize, num_kv_heads: usize, seq_len: usize, seed: Option<u64>) => {
        let mut rng = Rng::new(seed.unwrap_or(42));
        GroupedQueryAttention::new(d_model, num_heads, num_kv_heads, seq_len, &mut rng).map_err(nn_err)?
    }
);

// ---- FlashAttention ----
stateful_layer!(
    PyFlashAttention, "FlashAttention", FlashAttention<f64>,
    #[pyo3(signature = (d_model, num_heads, seq_len, block_size=None, seed=None))]
    new(d_model: usize, num_heads: usize, seq_len: usize, block_size: Option<usize>, seed: Option<u64>) => {
        let mut rng = Rng::new(seed.unwrap_or(42));
        FlashAttention::new(d_model, num_heads, seq_len, block_size.unwrap_or(64), &mut rng).map_err(nn_err)?
    }
);

// ---- SinusoidalPositionalEncoding ----
stateful_layer!(
    PySinusoidalPositionalEncoding, "SinusoidalPositionalEncoding", SinusoidalPositionalEncoding<f64>,
    #[pyo3(signature = (d_model, max_len=None))]
    new(d_model: usize, max_len: Option<usize>) => {
        SinusoidalPositionalEncoding::new(d_model, max_len.unwrap_or(5000))
    }
);

// ---- RotaryPositionalEncoding ----
/// Rotary Positional Encoding (RoPE) layer.
///
/// Unlike other layers, RoPE requires a `seq_len` argument in its `forward` call
/// because the rotation depends on the sequence length.
#[pyclass(name = "RotaryPositionalEncoding", unsendable)]
pub struct PyRotaryPositionalEncoding {
    inner: Option<RotaryPositionalEncoding<f64>>,
}

#[pymethods]
impl PyRotaryPositionalEncoding {
    /// Create a new RoPE encoder.
    ///
    /// Parameters: d_model — dimensionality (must be even), base — frequency base (default 10000.0).
    #[new]
    #[pyo3(signature = (d_model, base=None))]
    fn new(d_model: usize, base: Option<f64>) -> PyResult<Self> {
        let rope =
            RotaryPositionalEncoding::new(d_model, base.unwrap_or(10000.0)).map_err(nn_err)?;
        Ok(Self { inner: Some(rope) })
    }

    /// Apply rotary positional encoding to the input Variable.
    ///
    /// Parameters: x — input Variable [batch, seq_len * d_model], seq_len — sequence length.
    fn forward(&self, x: &PyVariable, seq_len: usize) -> PyResult<PyVariable> {
        let l = self.inner.as_ref().ok_or_else(consumed_err)?;
        let out = l.apply(&x.inner, seq_len).map_err(nn_err)?;
        Ok(PyVariable { inner: out })
    }

    /// Return trainable parameters (empty for RoPE).
    fn parameters(&self) -> Vec<PyVariable> {
        vec![]
    }
    /// Set the layer to training mode (no-op).
    fn train(&self) {}
    /// Set the layer to evaluation mode (no-op).
    fn eval(&self) {}
}

// ---- Pooling layers (no learnable params, configurable) ----

/// 1-D max pooling layer.
#[pyclass(name = "MaxPool1d", unsendable)]
pub struct PyMaxPool1d {
    inner: MaxPool1d,
}

#[pymethods]
impl PyMaxPool1d {
    /// Create a new MaxPool1d layer.
    ///
    /// Parameters: kernel_size — pooling window size, stride — step size (default: kernel_size).
    #[new]
    #[pyo3(signature = (kernel_size, stride=None))]
    fn new(kernel_size: usize, stride: Option<usize>) -> Self {
        let mut pool = MaxPool1d::new(kernel_size);
        if let Some(s) = stride {
            pool = pool.set_stride(s);
        }
        Self { inner: pool }
    }

    /// Run a forward pass through this layer.
    fn forward(&self, x: &PyVariable) -> PyResult<PyVariable> {
        let out = Layer::<f64>::forward(&self.inner, &x.inner).map_err(nn_err)?;
        Ok(PyVariable { inner: out })
    }

    /// Return trainable parameters (empty for pooling).
    fn parameters(&self) -> Vec<PyVariable> {
        vec![]
    }
    /// Set the layer to training mode (no-op).
    fn train(&self) {}
    /// Set the layer to evaluation mode (no-op).
    fn eval(&self) {}
}

/// 2-D max pooling layer.
#[pyclass(name = "MaxPool2d", unsendable)]
pub struct PyMaxPool2d {
    inner: MaxPool2d,
}

#[pymethods]
impl PyMaxPool2d {
    /// Create a new MaxPool2d layer.
    ///
    /// Parameters: kernel_h, kernel_w — pooling window size,
    /// stride_h, stride_w — step size (default: kernel size),
    /// padding_h, padding_w — zero padding (default: 0).
    #[new]
    #[pyo3(signature = (kernel_h, kernel_w, stride_h=None, stride_w=None, padding_h=None, padding_w=None))]
    fn new(
        kernel_h: usize,
        kernel_w: usize,
        stride_h: Option<usize>,
        stride_w: Option<usize>,
        padding_h: Option<usize>,
        padding_w: Option<usize>,
    ) -> Self {
        let mut pool = MaxPool2d::new((kernel_h, kernel_w));
        if stride_h.is_some() || stride_w.is_some() {
            pool = pool.set_stride((stride_h.unwrap_or(kernel_h), stride_w.unwrap_or(kernel_w)));
        }
        if padding_h.is_some() || padding_w.is_some() {
            pool = pool.set_padding((padding_h.unwrap_or(0), padding_w.unwrap_or(0)));
        }
        Self { inner: pool }
    }

    /// Run a forward pass through this layer.
    fn forward(&self, x: &PyVariable) -> PyResult<PyVariable> {
        let out = Layer::<f64>::forward(&self.inner, &x.inner).map_err(nn_err)?;
        Ok(PyVariable { inner: out })
    }

    /// Return trainable parameters (empty for pooling).
    fn parameters(&self) -> Vec<PyVariable> {
        vec![]
    }
    /// Set the layer to training mode (no-op).
    fn train(&self) {}
    /// Set the layer to evaluation mode (no-op).
    fn eval(&self) {}
}

/// 1-D average pooling layer.
#[pyclass(name = "AvgPool1d", unsendable)]
pub struct PyAvgPool1d {
    inner: AvgPool1d,
}

#[pymethods]
impl PyAvgPool1d {
    /// Create a new AvgPool1d layer.
    ///
    /// Parameters: kernel_size — pooling window size, stride — step size (default: kernel_size).
    #[new]
    #[pyo3(signature = (kernel_size, stride=None))]
    fn new(kernel_size: usize, stride: Option<usize>) -> Self {
        let mut pool = AvgPool1d::new(kernel_size);
        if let Some(s) = stride {
            pool = pool.set_stride(s);
        }
        Self { inner: pool }
    }

    /// Run a forward pass through this layer.
    fn forward(&self, x: &PyVariable) -> PyResult<PyVariable> {
        let out = Layer::<f64>::forward(&self.inner, &x.inner).map_err(nn_err)?;
        Ok(PyVariable { inner: out })
    }

    /// Return trainable parameters (empty for pooling).
    fn parameters(&self) -> Vec<PyVariable> {
        vec![]
    }
    /// Set the layer to training mode (no-op).
    fn train(&self) {}
    /// Set the layer to evaluation mode (no-op).
    fn eval(&self) {}
}

/// 2-D average pooling layer.
#[pyclass(name = "AvgPool2d", unsendable)]
pub struct PyAvgPool2d {
    inner: AvgPool2d,
}

#[pymethods]
impl PyAvgPool2d {
    /// Create a new AvgPool2d layer.
    ///
    /// Parameters: kernel_h, kernel_w — pooling window size,
    /// stride_h, stride_w — step size (default: kernel size),
    /// padding_h, padding_w — zero padding (default: 0).
    #[new]
    #[pyo3(signature = (kernel_h, kernel_w, stride_h=None, stride_w=None, padding_h=None, padding_w=None))]
    fn new(
        kernel_h: usize,
        kernel_w: usize,
        stride_h: Option<usize>,
        stride_w: Option<usize>,
        padding_h: Option<usize>,
        padding_w: Option<usize>,
    ) -> Self {
        let mut pool = AvgPool2d::new((kernel_h, kernel_w));
        if stride_h.is_some() || stride_w.is_some() {
            pool = pool.set_stride((stride_h.unwrap_or(kernel_h), stride_w.unwrap_or(kernel_w)));
        }
        if padding_h.is_some() || padding_w.is_some() {
            pool = pool.set_padding((padding_h.unwrap_or(0), padding_w.unwrap_or(0)));
        }
        Self { inner: pool }
    }

    /// Run a forward pass through this layer.
    fn forward(&self, x: &PyVariable) -> PyResult<PyVariable> {
        let out = Layer::<f64>::forward(&self.inner, &x.inner).map_err(nn_err)?;
        Ok(PyVariable { inner: out })
    }

    /// Return trainable parameters (empty for pooling).
    fn parameters(&self) -> Vec<PyVariable> {
        vec![]
    }
    /// Set the layer to training mode (no-op).
    fn train(&self) {}
    /// Set the layer to evaluation mode (no-op).
    fn eval(&self) {}
}

// ---- causal_mask function ----

/// Generate a causal (lower-triangular) attention mask.
///
/// Returns a Tensor of shape [seq_len, seq_len] where position (i, j) is 1.0
/// if j <= i (allowed) and 0.0 otherwise (masked).
#[pyfunction]
fn py_causal_mask(seq_len: usize) -> PyVariable {
    PyVariable {
        inner: causal_mask::<f64>(seq_len),
    }
}

// ---- Stateless activation layers (no inner state to consume) ----

macro_rules! stateless_layer {
    ($py_name:ident, $py_str:literal, $rust_val:expr) => {
        #[pyclass(name = $py_str)]
        #[derive(Clone)]
        pub struct $py_name;

        #[pymethods]
        impl $py_name {
            /// Create a new activation layer instance.
            #[new]
            fn new() -> Self {
                Self
            }

            /// Apply the activation function to the input Variable.
            fn forward(&self, x: &PyVariable) -> PyResult<PyVariable> {
                let layer: &dyn Layer<f64> = &$rust_val;
                let out = layer.forward(&x.inner).map_err(nn_err)?;
                Ok(PyVariable { inner: out })
            }

            /// Return trainable parameters (empty for activation layers).
            fn parameters(&self) -> Vec<PyVariable> {
                vec![]
            }
            /// Set the layer to training mode (no-op for activations).
            fn train(&self) {}
            /// Set the layer to evaluation mode (no-op for activations).
            fn eval(&self) {}
        }
    };
}

stateless_layer!(PyReLULayer, "ReLU", ReLU);
stateless_layer!(PySigmoidLayer, "Sigmoid", Sigmoid);
stateless_layer!(PyTanhLayer, "Tanh", Tanh);
stateless_layer!(PyFlattenLayer, "Flatten", Flatten::new());

// ===========================================================================
// SEQUENTIAL
// ===========================================================================

/// A sequential container that chains layers in order.
#[pyclass(name = "Sequential", unsendable)]
pub struct PySequential {
    inner: Option<Sequential<f64>>,
}

/// Try to extract a `Box<dyn Layer<f64>>` from a Python layer object.
fn extract_layer(obj: &Bound<'_, PyAny>) -> PyResult<Box<dyn Layer<f64>>> {
    // Stateless layers (no inner to take)
    if obj.downcast::<PyReLULayer>().is_ok() {
        return Ok(Box::new(ReLU));
    }
    if obj.downcast::<PySigmoidLayer>().is_ok() {
        return Ok(Box::new(Sigmoid));
    }
    if obj.downcast::<PyTanhLayer>().is_ok() {
        return Ok(Box::new(Tanh));
    }
    if obj.downcast::<PyFlattenLayer>().is_ok() {
        return Ok(Box::new(Flatten::new()));
    }
    // Stateful layers — take inner
    macro_rules! try_take {
        ($py_ty:ty) => {
            if let Ok(cell) = obj.downcast::<$py_ty>() {
                let mut borrow = cell.borrow_mut();
                return borrow
                    .inner
                    .take()
                    .map(|l| Box::new(l) as Box<dyn Layer<f64>>)
                    .ok_or_else(consumed_err);
            }
        };
    }
    try_take!(PyNnLinear);
    try_take!(PyConv1d);
    try_take!(PyConv2d);
    try_take!(PyBatchNorm1d);
    try_take!(PyBatchNorm2d);
    try_take!(PyDropout);
    try_take!(PyEmbedding);
    try_take!(PyLayerNorm);
    try_take!(PyLSTM);
    try_take!(PyGRU);
    try_take!(PyMultiHeadAttention);
    try_take!(PyConv3d);
    try_take!(PySimpleRNN);
    try_take!(PyTransformerEncoderLayer);
    try_take!(PyTransformerDecoderLayer);
    try_take!(PyMultiQueryAttention);
    try_take!(PyGroupedQueryAttention);
    try_take!(PyFlashAttention);
    try_take!(PySinusoidalPositionalEncoding);
    // Note: Pooling layers (MaxPool1d, MaxPool2d, AvgPool1d, AvgPool2d) and
    // RotaryPositionalEncoding are not supported in Sequential because they
    // either lack Clone or have non-standard forward signatures.
    Err(pyo3::exceptions::PyTypeError::new_err(
        "unsupported layer type for Sequential",
    ))
}

#[pymethods]
impl PySequential {
    /// Create a Sequential model from an optional list of layers.
    ///
    /// Parameters: layers — optional Python list of layer objects.
    #[new]
    #[pyo3(signature = (layers=None))]
    fn new(layers: Option<&Bound<'_, PyList>>) -> PyResult<Self> {
        let rust_layers = if let Some(list) = layers {
            let mut v = Vec::with_capacity(list.len());
            for item in list.iter() {
                v.push(extract_layer(&item)?);
            }
            v
        } else {
            Vec::new()
        };
        Ok(Self {
            inner: Some(Sequential::new(rust_layers)),
        })
    }

    /// Run a forward pass through all layers in sequence.
    ///
    /// Parameters: x — input Variable. Returns the output Variable.
    fn forward(&self, x: &PyVariable) -> PyResult<PyVariable> {
        let seq = self.inner.as_ref().ok_or_else(consumed_err)?;
        let out = seq.forward(&x.inner).map_err(nn_err)?;
        Ok(PyVariable { inner: out })
    }

    /// Return all trainable parameters from every layer in the sequence.
    fn parameters(&self) -> Vec<PyVariable> {
        self.inner
            .as_ref()
            .map(|s| {
                Layer::parameters(s)
                    .into_iter()
                    .map(|v| PyVariable { inner: v })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Set all layers to training mode.
    fn train(&mut self) {
        if let Some(ref mut s) = self.inner {
            Layer::train(s);
        }
    }

    /// Set all layers to evaluation mode.
    fn eval(&mut self) {
        if let Some(ref mut s) = self.inner {
            Layer::eval(s);
        }
    }

    /// Return the number of layers in the sequence.
    fn __len__(&self) -> usize {
        self.inner.as_ref().map_or(0, |s| s.len())
    }

    /// Return a string representation of the Sequential model.
    fn __repr__(&self) -> String {
        let n = self.__len__();
        format!("Sequential({n} layers)")
    }
}

// ===========================================================================
// FUNCTIONAL ACTIVATIONS
// ===========================================================================

/// Apply the ReLU activation function element-wise: max(0, x).
#[pyfunction]
fn relu(x: &PyVariable) -> PyVariable {
    PyVariable {
        inner: scivex_nn::functional::relu(&x.inner),
    }
}

/// Apply the sigmoid activation function element-wise: 1 / (1 + exp(-x)).
#[pyfunction]
fn sigmoid(x: &PyVariable) -> PyVariable {
    PyVariable {
        inner: scivex_nn::functional::sigmoid(&x.inner),
    }
}

/// Apply the tanh activation function element-wise.
#[pyfunction]
fn tanh_act(x: &PyVariable) -> PyVariable {
    PyVariable {
        inner: scivex_nn::functional::tanh_fn(&x.inner),
    }
}

/// Apply the softmax function, normalizing the input into a probability distribution.
#[pyfunction]
fn softmax(x: &PyVariable) -> PyResult<PyVariable> {
    Ok(PyVariable {
        inner: scivex_nn::functional::softmax(&x.inner).map_err(nn_err)?,
    })
}

/// Apply log-softmax: log(softmax(x)), numerically stable.
#[pyfunction]
fn log_softmax(x: &PyVariable) -> PyResult<PyVariable> {
    Ok(PyVariable {
        inner: scivex_nn::functional::log_softmax(&x.inner).map_err(nn_err)?,
    })
}

// ===========================================================================
// LOSS FUNCTIONS
// ===========================================================================

/// Compute mean squared error loss between predictions and targets.
#[pyfunction]
fn mse_loss(pred: &PyVariable, target: &PyVariable) -> PyResult<PyVariable> {
    Ok(PyVariable {
        inner: scivex_nn::loss::mse_loss(&pred.inner, &target.inner).map_err(nn_err)?,
    })
}

/// Compute cross-entropy loss from logits and target labels.
#[pyfunction]
fn cross_entropy_loss(logits: &PyVariable, targets: &PyVariable) -> PyResult<PyVariable> {
    Ok(PyVariable {
        inner: scivex_nn::loss::cross_entropy_loss(&logits.inner, &targets.inner)
            .map_err(nn_err)?,
    })
}

/// Compute binary cross-entropy loss between predictions and binary targets.
#[pyfunction]
fn bce_loss(pred: &PyVariable, target: &PyVariable) -> PyResult<PyVariable> {
    Ok(PyVariable {
        inner: scivex_nn::loss::bce_loss(&pred.inner, &target.inner).map_err(nn_err)?,
    })
}

/// Compute Huber loss (smooth L1), transitioning from quadratic to linear at delta.
///
/// Parameters: pred — predictions, target — targets, delta — threshold (default 1.0).
#[pyfunction]
#[pyo3(signature = (pred, target, delta=1.0))]
fn huber_loss(pred: &PyVariable, target: &PyVariable, delta: f64) -> PyResult<PyVariable> {
    Ok(PyVariable {
        inner: scivex_nn::loss::huber_loss(&pred.inner, &target.inner, delta).map_err(nn_err)?,
    })
}

/// Compute focal loss for class-imbalanced classification.
///
/// Parameters: logits, targets, gamma — focusing parameter (default 2.0), alpha — balancing factor (default 0.25).
#[pyfunction]
#[pyo3(signature = (logits, targets, gamma=2.0, alpha=0.25))]
fn focal_loss(
    logits: &PyVariable,
    targets: &PyVariable,
    gamma: f64,
    alpha: f64,
) -> PyResult<PyVariable> {
    Ok(PyVariable {
        inner: scivex_nn::loss::focal_loss(&logits.inner, &targets.inner, gamma, alpha)
            .map_err(nn_err)?,
    })
}

// ===========================================================================
// OPTIMIZERS
// ===========================================================================

fn extract_vars(params: &Bound<'_, PyList>) -> PyResult<Vec<Variable<f64>>> {
    let mut vars = Vec::with_capacity(params.len());
    for item in params.iter() {
        let cell = item
            .downcast::<PyVariable>()
            .map_err(|_| pyo3::exceptions::PyTypeError::new_err("expected list of Variable"))?;
        vars.push(cell.borrow().inner.clone());
    }
    Ok(vars)
}

// ---- SGD ----

/// Stochastic Gradient Descent optimizer with optional momentum and weight decay.
#[pyclass(name = "SGD", unsendable)]
pub struct PySGD {
    inner: optim::SGD<f64>,
}

#[pymethods]
impl PySGD {
    /// Create an SGD optimizer.
    ///
    /// Parameters: params — list of Variables, lr — learning rate,
    /// momentum — momentum factor (default 0.0), weight_decay — L2 penalty (default 0.0).
    #[new]
    #[pyo3(signature = (params, lr, momentum=0.0, weight_decay=0.0))]
    fn new(
        params: &Bound<'_, PyList>,
        lr: f64,
        momentum: f64,
        weight_decay: f64,
    ) -> PyResult<Self> {
        let vars = extract_vars(params)?;
        let opt = optim::SGD::new(vars, lr)
            .with_momentum(momentum)
            .with_weight_decay(weight_decay);
        Ok(Self { inner: opt })
    }

    /// Perform a single optimization step, updating all parameters.
    fn step(&mut self) {
        Optimizer::step(&mut self.inner);
    }

    /// Zero out gradients for all tracked parameters.
    fn zero_grad(&mut self) {
        Optimizer::zero_grad(&mut self.inner);
    }
}

// ---- Adam ----

/// Adam optimizer with adaptive learning rates.
#[pyclass(name = "Adam", unsendable)]
pub struct PyAdam {
    inner: optim::Adam<f64>,
}

#[pymethods]
impl PyAdam {
    /// Create an Adam optimizer.
    ///
    /// Parameters: params — list of Variables, lr — learning rate (default 0.001).
    #[new]
    #[pyo3(signature = (params, lr=0.001))]
    fn new(params: &Bound<'_, PyList>, lr: f64) -> PyResult<Self> {
        let vars = extract_vars(params)?;
        Ok(Self {
            inner: optim::Adam::new(vars, lr),
        })
    }

    /// Perform a single optimization step, updating all parameters.
    fn step(&mut self) {
        Optimizer::step(&mut self.inner);
    }

    /// Zero out gradients for all tracked parameters.
    fn zero_grad(&mut self) {
        Optimizer::zero_grad(&mut self.inner);
    }
}

// ---- AdamW ----

/// AdamW optimizer with decoupled weight decay regularization.
#[pyclass(name = "AdamW", unsendable)]
pub struct PyAdamW {
    inner: optim::AdamW<f64>,
}

#[pymethods]
impl PyAdamW {
    /// Create an AdamW optimizer.
    ///
    /// Parameters: params — list of Variables, lr — learning rate (default 0.001),
    /// weight_decay — decoupled weight decay (default 0.01).
    #[new]
    #[pyo3(signature = (params, lr=0.001, weight_decay=0.01))]
    fn new(params: &Bound<'_, PyList>, lr: f64, weight_decay: f64) -> PyResult<Self> {
        let vars = extract_vars(params)?;
        let _ = weight_decay; // AdamW constructor has default 0.01
        Ok(Self {
            inner: optim::AdamW::new(vars, lr),
        })
    }

    /// Perform a single optimization step, updating all parameters.
    fn step(&mut self) {
        Optimizer::step(&mut self.inner);
    }

    /// Zero out gradients for all tracked parameters.
    fn zero_grad(&mut self) {
        Optimizer::zero_grad(&mut self.inner);
    }
}

// ---- RMSprop ----

/// RMSprop optimizer with adaptive per-parameter learning rates.
#[pyclass(name = "RMSprop", unsendable)]
pub struct PyRMSprop {
    inner: optim::RMSprop<f64>,
}

#[pymethods]
impl PyRMSprop {
    /// Create an RMSprop optimizer.
    ///
    /// Parameters: params — list of Variables, lr — learning rate (default 0.01).
    #[new]
    #[pyo3(signature = (params, lr=0.01))]
    fn new(params: &Bound<'_, PyList>, lr: f64) -> PyResult<Self> {
        let vars = extract_vars(params)?;
        Ok(Self {
            inner: optim::RMSprop::new(vars, lr),
        })
    }

    /// Perform a single optimization step, updating all parameters.
    fn step(&mut self) {
        Optimizer::step(&mut self.inner);
    }

    /// Zero out gradients for all tracked parameters.
    fn zero_grad(&mut self) {
        Optimizer::zero_grad(&mut self.inner);
    }
}

// ===========================================================================
// LR SCHEDULERS
// ===========================================================================

/// Step-based learning rate scheduler that decays LR by gamma every step_size epochs.
#[pyclass(name = "StepLR", unsendable)]
pub struct PyStepLR {
    inner: scivex_nn::optim::scheduler::StepLR<f64>,
    epoch: usize,
}

#[pymethods]
impl PyStepLR {
    /// Create a StepLR scheduler.
    ///
    /// Parameters: base_lr — initial learning rate, step_size — epochs between decays, gamma — decay factor.
    #[new]
    fn new(base_lr: f64, step_size: usize, gamma: f64) -> Self {
        Self {
            inner: scivex_nn::optim::scheduler::StepLR::new(base_lr, step_size, gamma),
            epoch: 0,
        }
    }

    /// Return the current learning rate for this epoch.
    fn get_lr(&self) -> f64 {
        self.inner.get_lr(self.epoch)
    }

    /// Advance the scheduler by one epoch.
    fn step(&mut self) {
        self.epoch += 1;
    }
}

/// Cosine annealing learning rate scheduler that decays LR following a cosine curve.
#[pyclass(name = "CosineAnnealingLR", unsendable)]
pub struct PyCosineAnnealingLR {
    inner: scivex_nn::optim::scheduler::CosineAnnealingLR<f64>,
    epoch: usize,
}

#[pymethods]
impl PyCosineAnnealingLR {
    /// Create a CosineAnnealingLR scheduler.
    ///
    /// Parameters: base_lr — initial LR, t_max — maximum epochs, eta_min — minimum LR (default 0.0).
    #[new]
    #[pyo3(signature = (base_lr, t_max, eta_min=0.0))]
    fn new(base_lr: f64, t_max: usize, eta_min: f64) -> Self {
        Self {
            inner: scivex_nn::optim::scheduler::CosineAnnealingLR::new(base_lr, t_max, eta_min),
            epoch: 0,
        }
    }

    /// Return the current learning rate for this epoch.
    fn get_lr(&self) -> f64 {
        self.inner.get_lr(self.epoch)
    }

    /// Advance the scheduler by one epoch.
    fn step(&mut self) {
        self.epoch += 1;
    }
}

/// Reduce learning rate when a monitored metric has stopped improving.
#[pyclass(name = "ReduceLROnPlateau", unsendable)]
pub struct PyReduceLROnPlateau {
    inner: scivex_nn::optim::scheduler::ReduceLROnPlateau<f64>,
}

#[pymethods]
impl PyReduceLROnPlateau {
    /// Create a ReduceLROnPlateau scheduler.
    ///
    /// Parameters: initial_lr, factor — LR reduction factor (default 0.1),
    /// patience — epochs to wait before reducing (default 10), min_lr — floor (default 0.0).
    #[new]
    #[pyo3(signature = (initial_lr, factor=0.1, patience=10, min_lr=0.0))]
    fn new(initial_lr: f64, factor: f64, patience: usize, min_lr: f64) -> Self {
        Self {
            inner: scivex_nn::optim::scheduler::ReduceLROnPlateau::new(
                initial_lr, factor, patience, min_lr, true,
            ),
        }
    }

    /// Report metric value and get current learning rate.
    fn step(&mut self, metric: f64) -> f64 {
        self.inner.report(metric)
    }

    /// Return the current learning rate.
    fn get_lr(&self) -> f64 {
        self.inner.current_lr()
    }
}

// ===========================================================================
// DATA LOADING
// ===========================================================================

/// A dataset wrapping input (x) and target (y) tensors for batched training.
#[pyclass(name = "TensorDataset", unsendable)]
pub struct PyTensorDataset {
    inner: scivex_nn::data::TensorDataset<f64>,
}

#[pymethods]
impl PyTensorDataset {
    /// Create a TensorDataset from input tensor x and target tensor y.
    #[new]
    fn new(x: &PyTensor, y: &PyTensor) -> PyResult<Self> {
        let ds = scivex_nn::data::TensorDataset::new(x.to_f64_tensor(), y.to_f64_tensor())
            .map_err(nn_err)?;
        Ok(Self { inner: ds })
    }

    /// Return the number of samples in the dataset.
    fn __len__(&self) -> usize {
        scivex_nn::data::Dataset::len(&self.inner)
    }

    /// Get the (x, y) sample pair at the given index.
    fn get(&self, index: usize) -> PyResult<(PyTensor, PyTensor)> {
        let (xi, yi) = scivex_nn::data::Dataset::get(&self.inner, index).map_err(nn_err)?;
        Ok((PyTensor::from_f64(xi), PyTensor::from_f64(yi)))
    }
}

// ===========================================================================
// WEIGHT PERSISTENCE
// ===========================================================================

/// Save model weights (list of Variables) to a file at the given path.
#[pyfunction]
fn save_weights(path: &str, params: &Bound<'_, PyList>) -> PyResult<()> {
    let vars = extract_vars(params)?;
    scivex_nn::persist::save_weights(path, &vars).map_err(nn_err)
}

/// Load model weights from a file, returning a list of Tensors.
#[pyfunction]
fn load_weights(path: &str) -> PyResult<Vec<PyTensor>> {
    let tensors = scivex_nn::persist::load_weights::<f64>(path).map_err(nn_err)?;
    Ok(tensors.into_iter().map(PyTensor::from_f64).collect())
}

// ===========================================================================
// TRAINING UTILITIES
// ===========================================================================

/// Early stopping callback that monitors a metric and signals when to stop training.
///
/// Tracks whether the metric has improved by at least `min_delta` within the
/// last `patience` checks.
#[pyclass(name = "EarlyStopping", unsendable)]
pub struct PyEarlyStopping {
    inner: EarlyStopping<f64>,
}

#[pymethods]
impl PyEarlyStopping {
    /// Create a new EarlyStopping monitor.
    ///
    /// Parameters: patience — number of checks without improvement before stopping,
    /// min_delta — minimum decrease to count as improvement (default 0.0).
    #[new]
    #[pyo3(signature = (patience, min_delta=0.0))]
    fn new(patience: usize, min_delta: f64) -> Self {
        Self {
            inner: EarlyStopping::new(patience, min_delta),
        }
    }

    /// Check a metric value and return True if training should stop.
    ///
    /// Parameters: metric — the current metric value (e.g. validation loss).
    /// Returns True if patience has been exhausted without improvement.
    fn check(&mut self, metric: f64) -> bool {
        self.inner.on_epoch_end(0, metric) == CallbackAction::Stop
    }

    /// Reset the internal state so the monitor can be reused.
    fn reset(&mut self) {
        self.inner.on_train_begin();
    }

    /// Return a string representation.
    fn __repr__(&self) -> String {
        "EarlyStopping(...)".to_string()
    }
}

/// Model checkpoint callback that tracks the epoch with the best (lowest) metric.
#[pyclass(name = "ModelCheckpoint", unsendable)]
pub struct PyModelCheckpoint {
    inner: ModelCheckpoint<f64>,
}

#[pymethods]
impl PyModelCheckpoint {
    /// Create a new ModelCheckpoint tracker.
    #[new]
    fn new() -> Self {
        Self {
            inner: ModelCheckpoint::new(),
        }
    }

    /// Report a metric value for a training step.
    ///
    /// Internally records whether this is the best metric seen so far.
    ///
    /// Parameters: metric — the current metric value (e.g. validation loss).
    fn step(&mut self, metric: f64) {
        self.inner.on_epoch_end(0, metric);
    }

    /// Return the epoch index that achieved the best (lowest) metric.
    fn best_epoch(&self) -> usize {
        self.inner.best_epoch()
    }

    /// Return the best metric value, or None if no step has been recorded.
    fn best_loss(&self) -> Option<f64> {
        self.inner.best_loss()
    }

    /// Reset the internal state so the tracker can be reused.
    fn reset(&mut self) {
        self.inner.on_train_begin();
    }

    /// Return a string representation.
    fn __repr__(&self) -> String {
        format!(
            "ModelCheckpoint(best_epoch={}, best_loss={:?})",
            self.inner.best_epoch(),
            self.inner.best_loss()
        )
    }
}

// ===========================================================================
// ONNX IMPORT
// ===========================================================================

/// An ONNX model loaded from a file, ready for inference.
#[pyclass(name = "OnnxModel", unsendable)]
pub struct PyOnnxModel {
    session: OnnxInferenceSession<f64>,
    input_names: Vec<String>,
    output_names: Vec<String>,
}

#[pymethods]
impl PyOnnxModel {
    /// Run inference on the model with the given input tensor.
    ///
    /// The input is fed as the first declared input of the ONNX graph.
    /// Returns the first output tensor.
    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        if self.input_names.is_empty() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "ONNX model has no declared inputs",
            ));
        }
        let name = &self.input_names[0];
        let outputs = self
            .session
            .run(&[(name.as_str(), input.to_f64_tensor())])
            .map_err(nn_err)?;
        let first = outputs.into_iter().next().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("ONNX model produced no outputs")
        })?;
        Ok(PyTensor::from_f64(first))
    }

    /// Return the list of input tensor names declared by the ONNX graph.
    fn input_names(&self) -> Vec<String> {
        self.input_names.clone()
    }

    /// Return the list of output tensor names declared by the ONNX graph.
    fn output_names(&self) -> Vec<String> {
        self.output_names.clone()
    }

    /// Return a string representation.
    fn __repr__(&self) -> String {
        format!(
            "OnnxModel(inputs={:?}, outputs={:?})",
            self.input_names, self.output_names
        )
    }
}

/// Load an ONNX model from a file path and return a ready-to-use OnnxModel.
///
/// Parameters: path — path to the `.onnx` file.
#[pyfunction]
fn load_onnx_model(path: &str) -> PyResult<PyOnnxModel> {
    let bytes = std::fs::read(path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("cannot read ONNX file: {e}")))?;
    let model = load_onnx(&bytes).map_err(nn_err)?;

    let input_names: Vec<String> = model
        .graph
        .inputs
        .iter()
        .map(|vi| vi.name.clone())
        .collect();
    let output_names: Vec<String> = model
        .graph
        .outputs
        .iter()
        .map(|vi| vi.name.clone())
        .collect();

    let session = OnnxInferenceSession::from_model(model).map_err(nn_err)?;

    Ok(PyOnnxModel {
        session,
        input_names,
        output_names,
    })
}

// ===========================================================================
// SAFETENSORS
// ===========================================================================

/// Save a dictionary of named tensors to a SafeTensors file.
///
/// Parameters: path — output file path, tensors — dict mapping names to Tensors.
#[pyfunction]
fn save_safetensors(path: &str, tensors: &Bound<'_, PyDict>) -> PyResult<()> {
    let mut pairs: Vec<(String, Tensor<f64>)> = Vec::with_capacity(tensors.len());
    for (key, val) in tensors.iter() {
        let name: String = key.extract()?;
        let tensor: PyTensor = val.extract()?;
        pairs.push((name, tensor.to_f64_tensor()));
    }
    let refs: Vec<(&str, &Tensor<f64>)> = pairs.iter().map(|(n, t)| (n.as_str(), t)).collect();
    scivex_nn::serialize::save_safetensors(path, &refs).map_err(nn_err)
}

/// Load tensors from a SafeTensors file, returning a dict of {name: Tensor}.
///
/// Parameters: path — path to the `.safetensors` file.
#[pyfunction]
fn load_safetensors(py: Python<'_>, path: &str) -> PyResult<Py<PyDict>> {
    let loaded = scivex_nn::serialize::load_safetensors::<f64>(path).map_err(nn_err)?;
    let dict = PyDict::new(py);
    for (name, tensor) in loaded {
        let py_tensor = Py::new(py, PyTensor::from_f64(tensor))?;
        dict.set_item(name, py_tensor)?;
    }
    Ok(dict.into())
}

// ===========================================================================
// GNN LAYERS
// ===========================================================================

/// Graph Convolutional Network layer (Kipf & Welling, 2017).
///
/// Computes `out = D^{-1/2} A_hat D^{-1/2} X W + b` where `A_hat = A + I`.
#[pyclass(name = "GCNConv", unsendable)]
pub struct PyGCNConv {
    inner: GCNConv<f64>,
}

#[pymethods]
impl PyGCNConv {
    /// Create a new GCN convolution layer.
    ///
    /// Parameters: in_features — input feature dimension per node,
    /// out_features — output feature dimension per node,
    /// bias — whether to include a bias term (default True),
    /// seed — RNG seed for weight initialization (default 42).
    #[new]
    #[pyo3(signature = (in_features, out_features, bias=true, seed=42))]
    fn new(in_features: usize, out_features: usize, bias: bool, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        Self {
            inner: GCNConv::new(in_features, out_features, bias, &mut rng),
        }
    }

    /// Forward pass through the GCN layer.
    ///
    /// Parameters: x — node features Variable [n_nodes, in_features],
    /// edge_index — adjacency matrix Tensor [n_nodes, n_nodes].
    /// Returns output Variable [n_nodes, out_features].
    fn forward(&self, x: &PyVariable, edge_index: &PyTensor) -> PyResult<PyVariable> {
        let out = self
            .inner
            .forward(&x.inner, edge_index.as_f64()?)
            .map_err(nn_err)?;
        Ok(PyVariable { inner: out })
    }

    /// Return all learnable parameters of this layer.
    fn parameters(&self) -> Vec<PyVariable> {
        self.inner
            .parameters()
            .into_iter()
            .map(|v| PyVariable { inner: v })
            .collect()
    }

    /// Return a string representation.
    fn __repr__(&self) -> String {
        "GCNConv(...)".to_string()
    }
}

/// Graph Attention Network layer (Velickovic et al., 2018).
///
/// Single-head attention: transforms features, computes attention coefficients
/// using LeakyReLU, applies masked softmax over neighbors, and aggregates.
#[pyclass(name = "GATConv", unsendable)]
pub struct PyGATConv {
    inner: GATConv<f64>,
}

#[pymethods]
impl PyGATConv {
    /// Create a new GAT convolution layer.
    ///
    /// Parameters: in_features — input feature dimension per node,
    /// out_features — output feature dimension per node,
    /// seed — RNG seed for weight initialization (default 42).
    #[new]
    #[pyo3(signature = (in_features, out_features, seed=42))]
    fn new(in_features: usize, out_features: usize, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        Self {
            inner: GATConv::new(in_features, out_features, &mut rng),
        }
    }

    /// Forward pass through the GAT layer.
    ///
    /// Parameters: x — node features Variable [n_nodes, in_features],
    /// edge_index — adjacency matrix Tensor [n_nodes, n_nodes].
    /// Returns output Variable [n_nodes, out_features].
    fn forward(&self, x: &PyVariable, edge_index: &PyTensor) -> PyResult<PyVariable> {
        let out = self
            .inner
            .forward(&x.inner, edge_index.as_f64()?)
            .map_err(nn_err)?;
        Ok(PyVariable { inner: out })
    }

    /// Return all learnable parameters of this layer.
    fn parameters(&self) -> Vec<PyVariable> {
        self.inner
            .parameters()
            .into_iter()
            .map(|v| PyVariable { inner: v })
            .collect()
    }

    /// Return a string representation.
    fn __repr__(&self) -> String {
        "GATConv(...)".to_string()
    }
}

/// GraphSAGE mean-aggregator layer (Hamilton et al., 2017).
///
/// Aggregates neighbor features via mean pooling, concatenates with self
/// features, and applies a linear transformation.
#[pyclass(name = "SAGEConv", unsendable)]
pub struct PySAGEConv {
    inner: SAGEConv<f64>,
}

#[pymethods]
impl PySAGEConv {
    /// Create a new GraphSAGE convolution layer.
    ///
    /// Parameters: in_features — input feature dimension per node,
    /// out_features — output feature dimension per node,
    /// bias — whether to include a bias term (default True),
    /// seed — RNG seed for weight initialization (default 42).
    #[new]
    #[pyo3(signature = (in_features, out_features, bias=true, seed=42))]
    fn new(in_features: usize, out_features: usize, bias: bool, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        Self {
            inner: SAGEConv::new(in_features, out_features, bias, &mut rng),
        }
    }

    /// Forward pass through the GraphSAGE layer.
    ///
    /// Parameters: x — node features Variable [n_nodes, in_features],
    /// edge_index — adjacency matrix Tensor [n_nodes, n_nodes].
    /// Returns output Variable [n_nodes, out_features].
    fn forward(&self, x: &PyVariable, edge_index: &PyTensor) -> PyResult<PyVariable> {
        let out = self
            .inner
            .forward(&x.inner, edge_index.as_f64()?)
            .map_err(nn_err)?;
        Ok(PyVariable { inner: out })
    }

    /// Return all learnable parameters of this layer.
    fn parameters(&self) -> Vec<PyVariable> {
        self.inner
            .parameters()
            .into_iter()
            .map(|v| PyVariable { inner: v })
            .collect()
    }

    /// Return a string representation.
    fn __repr__(&self) -> String {
        "SAGEConv(...)".to_string()
    }
}

// ===========================================================================
// REGISTER
// ===========================================================================

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let m = PyModule::new(py, "nn")?;

    // Variable
    m.add_class::<PyVariable>()?;
    m.add_function(wrap_pyfunction!(tensor, &m)?)?;

    // Layers
    m.add_class::<PyNnLinear>()?;
    m.add_class::<PyConv1d>()?;
    m.add_class::<PyConv2d>()?;
    m.add_class::<PyBatchNorm1d>()?;
    m.add_class::<PyBatchNorm2d>()?;
    m.add_class::<PyDropout>()?;
    m.add_class::<PyEmbedding>()?;
    m.add_class::<PyLayerNorm>()?;
    m.add_class::<PyLSTM>()?;
    m.add_class::<PyGRU>()?;
    m.add_class::<PyMultiHeadAttention>()?;
    m.add_class::<PyConv3d>()?;
    m.add_class::<PySimpleRNN>()?;
    m.add_class::<PyTransformerEncoderLayer>()?;
    m.add_class::<PyTransformerDecoderLayer>()?;
    m.add_class::<PyMultiQueryAttention>()?;
    m.add_class::<PyGroupedQueryAttention>()?;
    m.add_class::<PyFlashAttention>()?;
    m.add_class::<PySinusoidalPositionalEncoding>()?;
    m.add_class::<PyRotaryPositionalEncoding>()?;
    m.add_class::<PyMaxPool1d>()?;
    m.add_class::<PyMaxPool2d>()?;
    m.add_class::<PyAvgPool1d>()?;
    m.add_class::<PyAvgPool2d>()?;
    m.add_class::<PyReLULayer>()?;
    m.add_class::<PySigmoidLayer>()?;
    m.add_class::<PyTanhLayer>()?;
    m.add_class::<PyFlattenLayer>()?;
    m.add_class::<PySequential>()?;

    // Functional activations
    m.add_function(wrap_pyfunction!(relu, &m)?)?;
    m.add_function(wrap_pyfunction!(sigmoid, &m)?)?;
    m.add_function(wrap_pyfunction!(tanh_act, &m)?)?;
    m.add_function(wrap_pyfunction!(softmax, &m)?)?;
    m.add_function(wrap_pyfunction!(log_softmax, &m)?)?;

    // Causal mask
    m.add_function(wrap_pyfunction!(py_causal_mask, &m)?)?;

    // Loss functions
    m.add_function(wrap_pyfunction!(mse_loss, &m)?)?;
    m.add_function(wrap_pyfunction!(cross_entropy_loss, &m)?)?;
    m.add_function(wrap_pyfunction!(bce_loss, &m)?)?;
    m.add_function(wrap_pyfunction!(huber_loss, &m)?)?;
    m.add_function(wrap_pyfunction!(focal_loss, &m)?)?;

    // Optimizers
    m.add_class::<PySGD>()?;
    m.add_class::<PyAdam>()?;
    m.add_class::<PyAdamW>()?;
    m.add_class::<PyRMSprop>()?;

    // LR Schedulers
    m.add_class::<PyStepLR>()?;
    m.add_class::<PyCosineAnnealingLR>()?;
    m.add_class::<PyReduceLROnPlateau>()?;

    // Data
    m.add_class::<PyTensorDataset>()?;

    // Persistence
    m.add_function(wrap_pyfunction!(save_weights, &m)?)?;
    m.add_function(wrap_pyfunction!(load_weights, &m)?)?;

    // Training utilities
    m.add_class::<PyEarlyStopping>()?;
    m.add_class::<PyModelCheckpoint>()?;

    // ONNX import
    m.add_class::<PyOnnxModel>()?;
    m.add_function(wrap_pyfunction!(load_onnx_model, &m)?)?;

    // SafeTensors
    m.add_function(wrap_pyfunction!(save_safetensors, &m)?)?;
    m.add_function(wrap_pyfunction!(load_safetensors, &m)?)?;

    // GNN layers
    m.add_class::<PyGCNConv>()?;
    m.add_class::<PyGATConv>()?;
    m.add_class::<PySAGEConv>()?;

    parent.add_submodule(&m)?;
    Ok(())
}
