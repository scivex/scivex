//! Dropout regularization layer.

use std::cell::RefCell;

use scivex_core::random::Rng;
use scivex_core::{Float, Tensor};

use crate::error::Result;
use crate::variable::Variable;

use super::Layer;

/// Dropout layer.
///
/// During training, randomly zeros elements with probability `p` and scales
/// the remaining elements by `1 / (1 - p)` (inverted dropout).
///
/// During evaluation, this is an identity function.
pub struct Dropout<T: Float> {
    p: T,
    training: bool,
    rng: RefCell<Rng>,
}

impl<T: Float> Dropout<T> {
    /// Create a new dropout layer.
    ///
    /// - `p`: probability of zeroing an element (0.0 to 1.0)
    /// - `rng`: random number generator
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_core::{Tensor, random::Rng};
    /// # use scivex_nn::variable::Variable;
    /// # use scivex_nn::layer::{Dropout, Layer};
    /// let mut dropout = Dropout::<f64>::new(0.5, Rng::new(42));
    /// dropout.eval(); // disable dropout for inference
    /// let x = Variable::new(Tensor::ones(vec![2, 3]), false);
    /// let y = dropout.forward(&x).unwrap();
    /// // In eval mode, output equals input
    /// assert_eq!(y.data().as_slice(), x.data().as_slice());
    /// ```
    pub fn new(p: T, rng: Rng) -> Self {
        Self {
            p,
            training: true,
            rng: RefCell::new(rng),
        }
    }
}

impl<T: Float> Layer<T> for Dropout<T> {
    fn forward(&self, x: &Variable<T>) -> Result<Variable<T>> {
        if !self.training {
            return Ok(x.clone());
        }

        let x_data = x.data();
        let shape = x_data.shape().to_vec();
        let numel = x_data.numel();

        let scale = T::one() / (T::one() - self.p);
        let p_threshold = self.p;

        let mut rng = self.rng.borrow_mut();
        let mut mask_data = Vec::with_capacity(numel);
        let mut out_data = Vec::with_capacity(numel);
        let x_slice = x_data.as_slice();

        for &xv in x_slice {
            let r = T::from_f64(rng.next_f64());
            if r < p_threshold {
                mask_data.push(T::zero());
                out_data.push(T::zero());
            } else {
                mask_data.push(scale);
                out_data.push(xv * scale);
            }
        }

        let mask = Tensor::from_vec(mask_data, shape.clone())?;
        let out = Tensor::from_vec(out_data, shape)?;

        Ok(Variable::from_op(
            out,
            vec![x.clone()],
            Box::new(move |g: &Tensor<T>| {
                vec![
                    g.zip_map(&mask, |gi, mi| gi * mi)
                        .expect("shapes match from forward pass"),
                ]
            }),
        ))
    }

    fn parameters(&self) -> Vec<Variable<T>> {
        Vec::new()
    }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scivex_core::Tensor;

    #[test]
    fn test_dropout_eval_identity() {
        let mut dropout = Dropout::<f64>::new(0.5, Rng::new(42));
        dropout.eval();
        let x = Variable::new(Tensor::ones(vec![10]), true);
        let y = dropout.forward(&x).unwrap();
        assert_eq!(y.data().as_slice(), x.data().as_slice());
    }

    #[test]
    fn test_dropout_training_zeros_some() {
        let dropout = Dropout::<f64>::new(0.5, Rng::new(42));
        let x = Variable::new(Tensor::ones(vec![1000]), true);
        let y = dropout.forward(&x).unwrap();
        let zeros = y.data().as_slice().iter().filter(|&&v| v == 0.0).count();
        // With p=0.5, roughly half should be zero.
        assert!(zeros > 300 && zeros < 700, "zeros={zeros}");
    }
}
