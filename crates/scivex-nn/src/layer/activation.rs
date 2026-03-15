//! Activation layers (stateless wrappers around functional activations).

use scivex_core::Float;

use crate::error::Result;
use crate::functional;
use crate::variable::Variable;

use super::Layer;

/// ReLU activation layer.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct ReLU;

impl<T: Float> Layer<T> for ReLU {
    fn forward(&self, x: &Variable<T>) -> Result<Variable<T>> {
        Ok(functional::relu(x))
    }

    fn parameters(&self) -> Vec<Variable<T>> {
        Vec::new()
    }

    fn train(&mut self) {}
    fn eval(&mut self) {}
}

/// Sigmoid activation layer.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct Sigmoid;

impl<T: Float> Layer<T> for Sigmoid {
    fn forward(&self, x: &Variable<T>) -> Result<Variable<T>> {
        Ok(functional::sigmoid(x))
    }

    fn parameters(&self) -> Vec<Variable<T>> {
        Vec::new()
    }

    fn train(&mut self) {}
    fn eval(&mut self) {}
}

/// Tanh activation layer.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct Tanh;

impl<T: Float> Layer<T> for Tanh {
    fn forward(&self, x: &Variable<T>) -> Result<Variable<T>> {
        Ok(functional::tanh_fn(x))
    }

    fn parameters(&self) -> Vec<Variable<T>> {
        Vec::new()
    }

    fn train(&mut self) {}
    fn eval(&mut self) {}
}
