//! Neural network layers.

mod activation;
mod batchnorm;
mod dropout;
mod linear;
mod sequential;

pub use activation::{ReLU, Sigmoid, Tanh};
pub use batchnorm::BatchNorm1d;
pub use dropout::Dropout;
pub use linear::Linear;
pub use sequential::Sequential;

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
