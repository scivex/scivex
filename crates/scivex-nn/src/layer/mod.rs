//! Neural network layers.

mod activation;
pub(crate) mod attention;
mod attention_variants;
mod batchnorm;
mod batchnorm2d;
mod conv;
mod conv3d;
mod dropout;
mod embedding;
mod flatten;
pub mod gnn;
pub(crate) mod layernorm;
pub(crate) mod linear;
mod pool;
mod positional;
mod rnn;
mod sequential;
mod transformer_decoder;

pub use activation::{ReLU, Sigmoid, Tanh};
pub use attention::{MultiHeadAttention, TransformerEncoderLayer};
pub use attention_variants::{FlashAttention, GroupedQueryAttention, MultiQueryAttention};
pub use batchnorm::BatchNorm1d;
pub use batchnorm2d::BatchNorm2d;
pub use conv::{Conv1d, Conv2d};
pub use conv3d::Conv3d;
pub use dropout::Dropout;
pub use embedding::Embedding;
pub use flatten::Flatten;
pub use gnn::{GATConv, GCNConv, SAGEConv};
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
///
/// # Examples
///
/// ```
/// # use scivex_nn::layer::{Linear, Layer};
/// # use scivex_nn::variable::Variable;
/// # use scivex_core::{Tensor, random::Rng};
/// let mut rng = Rng::new(42);
/// let layer = Linear::<f64>::new(4, 2, true, &mut rng);
/// let x = Variable::new(Tensor::ones(vec![1, 4]), false);
/// let y = layer.forward(&x).unwrap();
/// assert_eq!(y.shape(), vec![1, 2]);
/// ```
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
