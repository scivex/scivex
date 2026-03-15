//! # scivex-nn
//!
//! Neural networks with reverse-mode automatic differentiation for the
//! Scivex ecosystem.
//!
//! ## Modules
//!
//! | Module | Contents |
//! |--------|----------|
//! | [`variable`] | `Variable<T>` — autograd computation graph node |
//! | [`ops`] | Differentiable operations (add, mul, matmul, etc.) |
//! | [`functional`] | Activation functions (relu, sigmoid, tanh, softmax) |
//! | [`layer`] | `Layer` trait, `Linear`, `Conv1d/2d`, pooling, RNN/LSTM/GRU, attention, `Sequential` |
//! | [`optim`] | `Optimizer` trait, `SGD`, `Adam`, LR schedulers |
//! | [`loss`] | Loss functions (MSE, cross-entropy, BCE) |
//! | [`init`] | Weight initialization (Xavier, Kaiming) |
//! | [`persist`] | `save_weights`, `load_weights` — binary weight persistence |
//! | [`data`] | `Dataset` trait, `TensorDataset`, `DataLoader` |

/// Dataset, DataLoader, and batching utilities.
pub mod data;
/// Neural network error types.
pub mod error;
/// Activation functions (ReLU, sigmoid, tanh, softmax).
pub mod functional;
/// Weight initialization strategies (Xavier, Kaiming).
pub mod init;
/// Layer trait and built-in layers (Linear, BatchNorm, Dropout).
pub mod layer;
/// Loss functions (MSE, cross-entropy, BCE).
pub mod loss;
/// Differentiable tensor operations for the autograd graph.
pub mod ops;
/// Optimizers (SGD, Adam).
pub mod optim;
/// Weight persistence: save and load model parameters.
pub mod persist;
/// Autograd computation graph node.
pub mod variable;

pub use error::{NnError, Result};
pub use variable::Variable;

/// Convenience re-exports.
pub mod prelude {
    pub use crate::data::{DataLoader, Dataset, TensorDataset};
    pub use crate::error::{NnError, Result};
    pub use crate::functional::{log_softmax, relu, sigmoid, softmax, tanh_fn};
    pub use crate::init::{kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform};
    pub use crate::layer::{
        AvgPool1d, AvgPool2d, BatchNorm1d, Conv1d, Conv2d, Dropout, Embedding, Flatten, GRU, LSTM,
        Layer, LayerNorm, Linear, MaxPool1d, MaxPool2d, MultiHeadAttention, ReLU, Sequential,
        Sigmoid, SimpleRNN, Tanh, TransformerEncoderLayer,
    };
    pub use crate::loss::{bce_loss, cross_entropy_loss, mse_loss};
    pub use crate::ops::{add, add_bias, matmul, mean, mul, neg, pow, scalar_mul, sub, sum};
    pub use crate::optim::{
        Adam, CosineAnnealingLR, ExponentialLR, LinearLR, LrScheduler, Optimizer,
        ReduceLROnPlateau, SGD, StepLR, WarmupCosineDecay,
    };
    pub use crate::persist::{load_weights, save_weights};
    pub use crate::variable::Variable;
}
