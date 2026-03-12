//! Flatten layer — reshapes multi-dimensional input to 2-D.

use scivex_core::{Float, Tensor};

use crate::error::{NnError, Result};
use crate::variable::Variable;

use super::Layer;

/// Flattens all dimensions except the batch dimension into a single vector.
///
/// Input: `[batch, d1, d2, ...]`
/// Output: `[batch, d1 * d2 * ...]`
///
/// Commonly placed between convolutional/pooling layers and fully-connected
/// layers.
pub struct Flatten;

impl Flatten {
    /// Create a new Flatten layer.
    pub fn new() -> Self {
        Self
    }
}

impl Default for Flatten {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float> Layer<T> for Flatten {
    fn forward(&self, x: &Variable<T>) -> Result<Variable<T>> {
        let shape = x.shape();
        if shape.is_empty() {
            return Err(NnError::EmptyInput);
        }
        let n = shape[0];
        let flat_dim: usize = shape[1..].iter().product();
        if flat_dim == 0 {
            return Err(NnError::EmptyInput);
        }

        let data = x.data();
        let out =
            Tensor::from_vec(data.as_slice().to_vec(), vec![n, flat_dim]).map_err(NnError::from)?;

        let orig_shape = shape.clone();
        let grad_fn = Box::new(move |g: &Tensor<T>| {
            vec![
                Tensor::from_vec(g.as_slice().to_vec(), orig_shape.clone()).expect("valid reshape"),
            ]
        });

        Ok(Variable::from_op(out, vec![x.clone()], grad_fn))
    }

    fn parameters(&self) -> Vec<Variable<T>> {
        Vec::new()
    }

    fn train(&mut self) {}
    fn eval(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flatten_4d() {
        let flat = Flatten::new();
        // [2, 3, 4, 5] → [2, 60]
        let x = Variable::new(Tensor::<f64>::ones(vec![2, 3, 4, 5]), true);
        let y = flat.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![2, 60]);
    }

    #[test]
    fn test_flatten_3d() {
        let flat = Flatten::new();
        let x = Variable::new(Tensor::<f64>::ones(vec![4, 16, 8]), true);
        let y = flat.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![4, 128]);
    }

    #[test]
    fn test_flatten_preserves_data() {
        let flat = Flatten::new();
        let x = Variable::new(
            Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 2, 3]).unwrap(),
            true,
        );
        let y = flat.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![1, 6]);
        assert_eq!(y.data().as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_flatten_backward() {
        let flat = Flatten::new();
        let x = Variable::new(Tensor::<f64>::ones(vec![2, 3, 4]), true);
        let y = flat.forward(&x).unwrap();
        let loss = crate::ops::sum(&y);
        loss.backward();
        let gx = x.grad().unwrap();
        // Gradient should have original shape
        assert_eq!(gx.shape(), &[2, 3, 4]);
    }
}
