//! Sequential container that chains layers.

use scivex_core::Float;

use crate::error::Result;
use crate::variable::Variable;

use super::Layer;

/// A sequential container that runs layers in order.
///
/// # Examples
///
/// ```
/// # use scivex_nn::layer::{Sequential, Linear, ReLU, Layer};
/// # use scivex_nn::variable::Variable;
/// # use scivex_core::{Tensor, random::Rng};
/// let mut rng = Rng::new(42);
/// let model: Sequential<f64> = Sequential::new(vec![
///     Box::new(Linear::new(4, 3, true, &mut rng)),
///     Box::new(ReLU),
///     Box::new(Linear::new(3, 2, true, &mut rng)),
/// ]);
/// let x = Variable::new(Tensor::ones(vec![1, 4]), false);
/// let y = model.forward(&x).unwrap();
/// assert_eq!(y.shape(), vec![1, 2]);
/// ```
pub struct Sequential<T: Float> {
    layers: Vec<Box<dyn Layer<T>>>,
}

impl<T: Float> Sequential<T> {
    /// Create a new sequential container from a list of layers.
    pub fn new(layers: Vec<Box<dyn Layer<T>>>) -> Self {
        Self { layers }
    }

    /// Add a layer to the end of the sequence.
    pub fn push(&mut self, layer: Box<dyn Layer<T>>) {
        self.layers.push(layer);
    }

    /// Number of layers.
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Whether the container is empty.
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }
}

impl<T: Float> Layer<T> for Sequential<T> {
    fn forward(&self, x: &Variable<T>) -> Result<Variable<T>> {
        let mut out = x.clone();
        for layer in &self.layers {
            out = layer.forward(&out)?;
        }
        Ok(out)
    }

    fn parameters(&self) -> Vec<Variable<T>> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }

    fn train(&mut self) {
        for layer in &mut self.layers {
            layer.train();
        }
    }

    fn eval(&mut self) {
        for layer in &mut self.layers {
            layer.eval();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layer::{Linear, ReLU};
    use scivex_core::Tensor;
    use scivex_core::random::Rng;

    #[test]
    fn test_sequential_forward() {
        let mut rng = Rng::new(42);
        let model: Sequential<f64> = Sequential::new(vec![
            Box::new(Linear::new(4, 3, true, &mut rng)),
            Box::new(ReLU),
            Box::new(Linear::new(3, 2, true, &mut rng)),
        ]);
        let x = Variable::new(Tensor::ones(vec![2, 4]), true);
        let y = model.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![2, 2]);
    }

    #[test]
    fn test_sequential_parameters() {
        let mut rng = Rng::new(42);
        let model: Sequential<f64> = Sequential::new(vec![
            Box::new(Linear::new(4, 3, true, &mut rng)),
            Box::new(ReLU),
            Box::new(Linear::new(3, 2, true, &mut rng)),
        ]);
        // 2 linear layers with bias = 4 parameters total
        assert_eq!(model.parameters().len(), 4);
    }
}
