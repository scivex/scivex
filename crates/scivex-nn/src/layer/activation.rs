//! Activation layers (stateless wrappers around functional activations).

use scivex_core::Float;

use crate::error::Result;
use crate::functional;
use crate::variable::Variable;

use super::Layer;

/// ReLU activation layer.
///
/// # Examples
///
/// ```
/// # use scivex_nn::layer::{ReLU, Layer};
/// # use scivex_nn::variable::Variable;
/// # use scivex_core::Tensor;
/// let relu = ReLU;
/// let x = Variable::new(
///     Tensor::from_vec(vec![-1.0_f64, 0.0, 1.0, 2.0], vec![1, 4]).unwrap(),
///     false,
/// );
/// let y = relu.forward(&x).unwrap();
/// assert_eq!(y.data().as_slice(), &[0.0, 0.0, 1.0, 2.0]);
/// ```
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
///
/// # Examples
///
/// ```
/// # use scivex_nn::layer::{Sigmoid, Layer};
/// # use scivex_nn::variable::Variable;
/// # use scivex_core::Tensor;
/// let sig = Sigmoid;
/// let x = Variable::new(Tensor::from_vec(vec![0.0_f64], vec![1, 1]).unwrap(), false);
/// let y = sig.forward(&x).unwrap();
/// assert!((y.data().as_slice()[0] - 0.5).abs() < 1e-10);
/// ```
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
///
/// # Examples
///
/// ```
/// # use scivex_nn::layer::{Tanh, Layer};
/// # use scivex_nn::variable::Variable;
/// # use scivex_core::Tensor;
/// let tanh = Tanh;
/// let x = Variable::new(Tensor::from_vec(vec![0.0_f64], vec![1, 1]).unwrap(), false);
/// let y = tanh.forward(&x).unwrap();
/// assert!(y.data().as_slice()[0].abs() < 1e-10);
/// ```
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
