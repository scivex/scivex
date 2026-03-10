//! Fully-connected (dense) layer.

use scivex_core::Float;
use scivex_core::random::Rng;

use crate::error::Result;
use crate::init;
use crate::ops;
use crate::variable::Variable;

use super::Layer;

/// A fully-connected linear layer: `y = x @ W^T + b`.
pub struct Linear<T: Float> {
    weight: Variable<T>,
    bias: Option<Variable<T>>,
}

impl<T: Float> Linear<T> {
    /// Create a new linear layer.
    ///
    /// - `in_features`: size of each input sample
    /// - `out_features`: size of each output sample
    /// - `use_bias`: whether to include a bias term
    /// - `rng`: random number generator for initialization
    pub fn new(in_features: usize, out_features: usize, use_bias: bool, rng: &mut Rng) -> Self {
        let w_data = init::kaiming_uniform::<T>(&[out_features, in_features], rng);
        let weight = Variable::new(w_data, true);

        let bias = if use_bias {
            // Initialize bias to zeros.
            let b_data = scivex_core::Tensor::zeros(vec![out_features]);
            Some(Variable::new(b_data, true))
        } else {
            None
        };

        Self { weight, bias }
    }

    /// Return the weight variable.
    pub fn weight(&self) -> &Variable<T> {
        &self.weight
    }

    /// Return the bias variable, if present.
    pub fn bias(&self) -> Option<&Variable<T>> {
        self.bias.as_ref()
    }
}

impl<T: Float> Layer<T> for Linear<T> {
    fn forward(&self, x: &Variable<T>) -> Result<Variable<T>> {
        // x: [batch, in], weight: [out, in]
        // y = x @ W^T
        let wt_data = self.weight.data().transpose()?;
        let wt = Variable::new(wt_data, false);

        // We need matmul that participates in the graph for weight gradients.
        // Use ops::matmul(x, wt_var) where wt_var carries the weight connection.
        let wt_var = Variable::from_op(
            self.weight.data().transpose()?,
            vec![self.weight.clone()],
            Box::new(|g: &scivex_core::Tensor<T>| {
                // grad of transpose is transpose of grad
                vec![g.transpose().expect("2-D from forward pass")]
            }),
        );
        let _ = wt; // drop unused

        let y = ops::matmul(x, &wt_var);

        match &self.bias {
            Some(b) => Ok(ops::add_bias(&y, b)),
            None => Ok(y),
        }
    }

    fn parameters(&self) -> Vec<Variable<T>> {
        let mut params = vec![self.weight.clone()];
        if let Some(b) = &self.bias {
            params.push(b.clone());
        }
        params
    }

    fn train(&mut self) {}
    fn eval(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use scivex_core::Tensor;

    #[test]
    fn test_linear_output_shape() {
        let mut rng = Rng::new(42);
        let layer = Linear::<f64>::new(5, 3, true, &mut rng);
        let x = Variable::new(Tensor::ones(vec![4, 5]), true);
        let y = layer.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![4, 3]);
    }

    #[test]
    fn test_linear_parameters_count() {
        let mut rng = Rng::new(42);
        let layer = Linear::<f64>::new(5, 3, true, &mut rng);
        assert_eq!(layer.parameters().len(), 2);

        let layer_no_bias = Linear::<f64>::new(5, 3, false, &mut rng);
        assert_eq!(layer_no_bias.parameters().len(), 1);
    }

    #[test]
    fn test_linear_backward() {
        let mut rng = Rng::new(42);
        let layer = Linear::<f64>::new(2, 1, true, &mut rng);
        let x = Variable::new(Tensor::from_vec(vec![1.0, 2.0], vec![1, 2]).unwrap(), true);
        let y = layer.forward(&x).unwrap();
        let loss = crate::ops::sum(&y);
        loss.backward();
        // Weight should have gradient.
        assert!(layer.weight().grad().is_some());
    }
}
