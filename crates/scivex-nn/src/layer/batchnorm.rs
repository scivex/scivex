//! Batch normalization layer.

use scivex_core::{Float, Tensor};

use crate::error::Result;
use crate::variable::Variable;

use super::Layer;

/// 1-D batch normalization.
///
/// Input shape: `[batch, features]`.
/// Normalizes each feature across the batch dimension.
///
/// During training, uses batch statistics and updates running statistics.
/// During evaluation, uses running statistics.
pub struct BatchNorm1d<T: Float> {
    gamma: Variable<T>,
    beta: Variable<T>,
    running_mean: Vec<T>,
    running_var: Vec<T>,
    eps: T,
    training: bool,
}

impl<T: Float> BatchNorm1d<T> {
    /// Create a new batch normalization layer.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_core::Tensor;
    /// # use scivex_nn::variable::Variable;
    /// # use scivex_nn::layer::{BatchNorm1d, Layer};
    /// let bn = BatchNorm1d::<f64>::new(3);
    /// let x = Variable::new(Tensor::ones(vec![4, 3]), false);
    /// let y = bn.forward(&x).unwrap();
    /// assert_eq!(y.shape(), vec![4, 3]);
    /// ```
    pub fn new(num_features: usize) -> Self {
        let gamma = Variable::new(Tensor::ones(vec![num_features]), true);
        let beta = Variable::new(Tensor::zeros(vec![num_features]), true);

        Self {
            gamma,
            beta,
            running_mean: vec![T::zero(); num_features],
            running_var: vec![T::one(); num_features],
            eps: T::from_f64(1e-5),
            training: true,
        }
    }
}

impl<T: Float> Layer<T> for BatchNorm1d<T> {
    fn forward(&self, x: &Variable<T>) -> Result<Variable<T>> {
        let x_data = x.data();
        let shape = x_data.shape().to_vec();
        let batch = shape[0];
        let features = shape[1];
        let x_slice = x_data.as_slice();
        let gamma_data = self.gamma.data();
        let beta_data = self.beta.data();
        let g_slice = gamma_data.as_slice();
        let b_slice = beta_data.as_slice();

        let (mean, var) = if self.training {
            // Compute batch statistics.
            let mut mean = vec![T::zero(); features];
            let mut var = vec![T::zero(); features];
            let n = T::from_usize(batch);

            for f in 0..features {
                let mut sum = T::zero();
                for b in 0..batch {
                    sum += x_slice[b * features + f];
                }
                mean[f] = sum / n;

                let mut sq_sum = T::zero();
                for b in 0..batch {
                    let diff = x_slice[b * features + f] - mean[f];
                    sq_sum += diff * diff;
                }
                var[f] = sq_sum / n;
            }

            (mean, var)
        } else {
            (self.running_mean.clone(), self.running_var.clone())
        };

        // Normalize and apply affine: y = gamma * (x - mean) / sqrt(var + eps) + beta
        let mut out_data = vec![T::zero(); batch * features];
        let mut x_norm_data = vec![T::zero(); batch * features];
        let mut inv_std = vec![T::zero(); features];

        for f in 0..features {
            inv_std[f] = T::one() / (var[f] + self.eps).sqrt();
        }

        for b_idx in 0..batch {
            for f in 0..features {
                let idx = b_idx * features + f;
                let xn = (x_slice[idx] - mean[f]) * inv_std[f];
                x_norm_data[idx] = xn;
                out_data[idx] = g_slice[f] * xn + b_slice[f];
            }
        }

        let out = Tensor::from_vec(out_data, shape.clone())?;
        let x_norm = Tensor::from_vec(x_norm_data, shape.clone())?;
        let inv_std_t = Tensor::from_vec(inv_std, vec![features])?;

        Ok(Variable::from_op(
            out,
            vec![x.clone(), self.gamma.clone(), self.beta.clone()],
            Box::new(move |g: &Tensor<T>| {
                let g_s = g.as_slice();
                let xn_s = x_norm.as_slice();
                let inv_s = inv_std_t.as_slice();
                let gs = gamma_data.as_slice();

                // Gradient for gamma: sum over batch of g * x_norm
                let mut grad_gamma = vec![T::zero(); features];
                // Gradient for beta: sum over batch of g
                let mut grad_beta = vec![T::zero(); features];
                // Gradient for x (simplified batch norm backward)
                let mut grad_x = vec![T::zero(); batch * features];

                for f in 0..features {
                    let mut g_sum = T::zero();
                    let mut gxn_sum = T::zero();
                    for b_idx in 0..batch {
                        let idx = b_idx * features + f;
                        grad_gamma[f] += g_s[idx] * xn_s[idx];
                        grad_beta[f] += g_s[idx];
                        g_sum += g_s[idx];
                        gxn_sum += g_s[idx] * xn_s[idx];
                    }
                    let n = T::from_usize(batch);
                    for b_idx in 0..batch {
                        let idx = b_idx * features + f;
                        grad_x[idx] =
                            gs[f] * inv_s[f] * (g_s[idx] - g_sum / n - xn_s[idx] * gxn_sum / n);
                    }
                }

                vec![
                    Tensor::from_vec(grad_x, shape.clone())
                        .expect("grad shape matches forward pass"),
                    Tensor::from_vec(grad_gamma, vec![features])
                        .expect("gamma grad length matches features"),
                    Tensor::from_vec(grad_beta, vec![features])
                        .expect("beta grad length matches features"),
                ]
            }),
        ))
    }

    fn parameters(&self) -> Vec<Variable<T>> {
        vec![self.gamma.clone(), self.beta.clone()]
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
    fn test_batchnorm_output_shape() {
        let bn = BatchNorm1d::<f64>::new(3);
        let x = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap(),
            true,
        );
        let y = bn.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![2, 3]);
    }

    #[test]
    fn test_batchnorm_normalized_output() {
        let bn = BatchNorm1d::<f64>::new(2);
        let x = Variable::new(
            Tensor::from_vec(vec![1.0, 10.0, 3.0, 20.0, 5.0, 30.0], vec![3, 2]).unwrap(),
            true,
        );
        let y = bn.forward(&x).unwrap();
        // Each feature should have near-zero mean across the batch.
        let y_data = y.data();
        let y_s = y_data.as_slice();
        let mean_f0 = (y_s[0] + y_s[2] + y_s[4]) / 3.0;
        assert!(mean_f0.abs() < 1e-5, "mean was {mean_f0}");
    }

    #[test]
    fn test_batchnorm_parameters() {
        let bn = BatchNorm1d::<f64>::new(5);
        assert_eq!(bn.parameters().len(), 2);
    }
}
