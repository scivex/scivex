//! Neural network layers.

mod activation;
mod attention;
mod batchnorm;
mod conv;
mod dropout;
mod embedding;
mod flatten;
pub(crate) mod layernorm;
pub(crate) mod linear;
mod pool;
mod positional;
mod rnn;
mod sequential;
mod transformer_decoder;

pub use activation::{ReLU, Sigmoid, Tanh};
pub use attention::{MultiHeadAttention, TransformerEncoderLayer};
pub use batchnorm::BatchNorm1d;
pub use conv::{Conv1d, Conv2d};
pub use dropout::Dropout;
pub use embedding::Embedding;
pub use flatten::Flatten;
pub use layernorm::LayerNorm;
pub use linear::Linear;
pub use pool::{AvgPool1d, AvgPool2d, MaxPool1d, MaxPool2d};
pub use positional::{RotaryPositionalEncoding, SinusoidalPositionalEncoding, causal_mask};
pub use rnn::{GRU, LSTM, SimpleRNN};
pub use sequential::Sequential;
pub use transformer_decoder::TransformerDecoderLayer;

use scivex_core::Float;

use crate::error::Result;
use crate::variable::Variable;

/// A neural network layer.
pub trait Layer<T: Float> {
    /// Forward pass.
    fn forward(&self, x: &Variable<T>) -> Result<Variable<T>>;

    /// Return all learnable parameters.
    fn parameters(&self) -> Vec<Variable<T>>;

    /// Set training mode.
    fn train(&mut self);

    /// Set evaluation mode.
    fn eval(&mut self);
}
