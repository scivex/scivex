//! Layer normalization.

use scivex_core::{Float, Tensor};

use crate::error::{NnError, Result};
use crate::variable::Variable;

use super::Layer;

/// Layer normalization over the last dimension.
///
/// Input: `[batch, features]`
/// Output: `[batch, features]`
///
/// Normalizes each sample independently: `y = gamma * (x - mean) / sqrt(var + eps) + beta`
pub struct LayerNorm<T: Float> {
    gamma: Variable<T>,
    beta: Variable<T>,
    eps: T,
    num_features: usize,
}

impl<T: Float> LayerNorm<T> {
    /// Create a new LayerNorm for the given feature dimension.
    pub fn new(num_features: usize) -> Self {
        Self {
            gamma: Variable::new(Tensor::ones(vec![num_features]), true),
            beta: Variable::new(Tensor::zeros(vec![num_features]), true),
            eps: T::from_f64(1e-5),
            num_features,
        }
    }
}

impl<T: Float> Layer<T> for LayerNorm<T> {
    fn forward(&self, x: &Variable<T>) -> Result<Variable<T>> {
        let shape = x.shape();
        if shape.len() != 2 || shape[1] != self.num_features {
            return Err(NnError::ShapeMismatch {
                expected: vec![0, self.num_features],
                got: shape,
            });
        }
        let n = shape[0];
        let d = shape[1];
        let xd = x.data();
        let xs = xd.as_slice();
        let gamma_d = self.gamma.data();
        let gs = gamma_d.as_slice();
        let beta_d = self.beta.data();
        let bs = beta_d.as_slice();
        let eps = self.eps;

        // Compute mean and variance per sample
        let mut means = vec![T::zero(); n];
        let mut vars = vec![T::zero(); n];
        for i in 0..n {
            let row = &xs[i * d..(i + 1) * d];
            let mut sum = T::zero();
            for &v in row {
                sum += v;
            }
            let mean = sum / T::from_f64(d as f64);
            means[i] = mean;
            let mut var_sum = T::zero();
            for &v in row {
                let diff = v - mean;
                var_sum += diff * diff;
            }
            vars[i] = var_sum / T::from_f64(d as f64);
        }

        // Normalize: x_hat = (x - mean) / sqrt(var + eps)
        let mut x_hat = vec![T::zero(); n * d];
        let mut out = vec![T::zero(); n * d];
        for i in 0..n {
            let inv_std = (vars[i] + eps).sqrt().recip();
            for j in 0..d {
                let idx = i * d + j;
                x_hat[idx] = (xs[idx] - means[i]) * inv_std;
                out[idx] = gs[j] * x_hat[idx] + bs[j];
            }
        }

        let out_tensor = Tensor::from_vec(out, vec![n, d]).expect("valid shape");

        let gamma_clone = gs.to_vec();
        let x_hat_clone = x_hat;
        let vars_clone = vars;

        #[allow(clippy::needless_range_loop)]
        let grad_fn = Box::new(move |g: &Tensor<T>| {
            let gd = g.as_slice();
            let mut gx = vec![T::zero(); n * d];
            let mut gg = vec![T::zero(); d];
            let mut gb = vec![T::zero(); d];

            for i in 0..n {
                let inv_std = (vars_clone[i] + eps).sqrt().recip();

                // Accumulate gamma/beta gradients
                for j in 0..d {
                    let idx = i * d + j;
                    gg[j] += gd[idx] * x_hat_clone[idx];
                    gb[j] += gd[idx];
                }

                // Input gradient via layer norm backward
                // dx = (1/d) * inv_std * (d * dy_scaled - sum(dy_scaled)
                //       - x_hat * sum(dy_scaled * x_hat))
                // where dy_scaled = gamma * dy
                let mut sum_dy = T::zero();
                let mut sum_dy_xhat = T::zero();
                for j in 0..d {
                    let idx = i * d + j;
                    let dy_s = gamma_clone[j] * gd[idx];
                    sum_dy += dy_s;
                    sum_dy_xhat += dy_s * x_hat_clone[idx];
                }
                let inv_d = T::from_f64(1.0 / d as f64);
                for j in 0..d {
                    let idx = i * d + j;
                    let dy_s = gamma_clone[j] * gd[idx];
                    gx[idx] = inv_std
                        * inv_d
                        * (T::from_f64(d as f64) * dy_s - sum_dy - x_hat_clone[idx] * sum_dy_xhat);
                }
            }

            vec![
                Tensor::from_vec(gx, vec![n, d]).expect("valid shape"),
                Tensor::from_vec(gg, vec![d]).expect("valid shape"),
                Tensor::from_vec(gb, vec![d]).expect("valid shape"),
            ]
        });

        Ok(Variable::from_op(
            out_tensor,
            vec![x.clone(), self.gamma.clone(), self.beta.clone()],
            grad_fn,
        ))
    }

    fn parameters(&self) -> Vec<Variable<T>> {
        vec![self.gamma.clone(), self.beta.clone()]
    }

    fn train(&mut self) {}
    fn eval(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layernorm_output_shape() {
        let ln = LayerNorm::<f64>::new(8);
        let x = Variable::new(Tensor::ones(vec![4, 8]), true);
        let y = ln.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![4, 8]);
    }

    #[test]
    fn test_layernorm_normalizes() {
        let ln = LayerNorm::<f64>::new(4);
        let x = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 4]).unwrap(),
            true,
        );
        let y = ln.forward(&x).unwrap();
        let yd = y.data();
        let ys = yd.as_slice();
        // Each row should have mean ≈ 0 (before gamma/beta, but gamma=1, beta=0)
        for row in 0..2 {
            let mut sum = 0.0;
            for j in 0..4 {
                sum += ys[row * 4 + j];
            }
            assert!((sum / 4.0).abs() < 1e-6, "Row {row} mean = {}", sum / 4.0);
        }
    }

    #[test]
    fn test_layernorm_backward() {
        let ln = LayerNorm::<f64>::new(4);
        let x = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]).unwrap(),
            true,
        );
        let y = ln.forward(&x).unwrap();
        let loss = crate::ops::sum(&y);
        loss.backward();
        let gx = x.grad().unwrap();
        assert_eq!(gx.shape(), &[1, 4]);
    }

    #[test]
    fn test_layernorm_parameters() {
        let ln = LayerNorm::<f64>::new(16);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_wrong_shape() {
        let ln = LayerNorm::<f64>::new(8);
        let x = Variable::new(Tensor::ones(vec![4, 10]), true);
        assert!(ln.forward(&x).is_err());
    }
}
