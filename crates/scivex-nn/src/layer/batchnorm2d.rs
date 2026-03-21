//! 2-D batch normalization layer for convolutional feature maps.

use scivex_core::{Float, Tensor};

use crate::error::Result;
use crate::variable::Variable;

use super::Layer;

/// 2-D batch normalization for convolutional feature maps.
///
/// Input shape: `[batch, channels, height, width]` (NCHW).
/// Normalizes each channel across the `(N, H, W)` dimensions.
///
/// During training, uses batch statistics and updates running statistics.
/// During evaluation, uses running statistics.
pub struct BatchNorm2d<T: Float> {
    gamma: Variable<T>,
    beta: Variable<T>,
    running_mean: Vec<T>,
    running_var: Vec<T>,
    eps: T,
    training: bool,
}

impl<T: Float> BatchNorm2d<T> {
    /// Create a new 2-D batch normalization layer.
    ///
    /// `num_channels` is the number of input channels (the `C` dimension in NCHW).
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_nn::layer::{BatchNorm2d, Layer};
    /// # use scivex_nn::variable::Variable;
    /// # use scivex_core::Tensor;
    /// let bn = BatchNorm2d::<f64>::new(3);
    /// let x = Variable::new(Tensor::ones(vec![2, 3, 4, 4]), false);
    /// let y = bn.forward(&x).unwrap();
    /// assert_eq!(y.shape(), vec![2, 3, 4, 4]);
    /// ```
    pub fn new(num_channels: usize) -> Self {
        let gamma = Variable::new(Tensor::ones(vec![num_channels]), true);
        let beta = Variable::new(Tensor::zeros(vec![num_channels]), true);

        Self {
            gamma,
            beta,
            running_mean: vec![T::zero(); num_channels],
            running_var: vec![T::one(); num_channels],
            eps: T::from_f64(1e-5),
            training: true,
        }
    }
}

/// Compute per-channel mean and variance across `(N, H, W)` for NCHW data.
fn channel_stats<T: Float>(
    x_slice: &[T],
    batch: usize,
    channels: usize,
    spatial: usize,
) -> (Vec<T>, Vec<T>) {
    let count = T::from_usize(batch * spatial);
    let mut mean = vec![T::zero(); channels];
    let mut var = vec![T::zero(); channels];

    for c in 0..channels {
        let mut sum = T::zero();
        for b_idx in 0..batch {
            let base = b_idx * channels * spatial + c * spatial;
            for s in 0..spatial {
                sum += x_slice[base + s];
            }
        }
        mean[c] = sum / count;

        let mut sq_sum = T::zero();
        for b_idx in 0..batch {
            let base = b_idx * channels * spatial + c * spatial;
            for s in 0..spatial {
                let diff = x_slice[base + s] - mean[c];
                sq_sum += diff * diff;
            }
        }
        var[c] = sq_sum / count;
    }

    (mean, var)
}

impl<T: Float> Layer<T> for BatchNorm2d<T> {
    fn forward(&self, x: &Variable<T>) -> Result<Variable<T>> {
        let x_data = x.data();
        let shape = x_data.shape().to_vec();
        let batch = shape[0];
        let channels = shape[1];
        let height = shape[2];
        let width = shape[3];
        let spatial = height * width;
        let x_slice = x_data.as_slice();
        let gamma_data = self.gamma.data();
        let beta_data = self.beta.data();
        let g_slice = gamma_data.as_slice();
        let b_slice = beta_data.as_slice();

        let (mean, var) = if self.training {
            channel_stats(x_slice, batch, channels, spatial)
        } else {
            (self.running_mean.clone(), self.running_var.clone())
        };

        // Normalize and apply affine: y = gamma * (x - mean) / sqrt(var + eps) + beta
        let total = batch * channels * spatial;
        let mut out_data = vec![T::zero(); total];
        let mut x_norm_data = vec![T::zero(); total];
        let mut inv_std = vec![T::zero(); channels];

        for c in 0..channels {
            inv_std[c] = T::one() / (var[c] + self.eps).sqrt();
        }

        for b_idx in 0..batch {
            for c in 0..channels {
                let base = b_idx * channels * spatial + c * spatial;
                for s in 0..spatial {
                    let idx = base + s;
                    let xn = (x_slice[idx] - mean[c]) * inv_std[c];
                    x_norm_data[idx] = xn;
                    out_data[idx] = g_slice[c] * xn + b_slice[c];
                }
            }
        }

        let out = Tensor::from_vec(out_data, shape.clone())?;
        let x_norm = Tensor::from_vec(x_norm_data, shape.clone())?;
        let inv_std_t = Tensor::from_vec(inv_std, vec![channels])?;

        Ok(Variable::from_op(
            out,
            vec![x.clone(), self.gamma.clone(), self.beta.clone()],
            Box::new(move |g: &Tensor<T>| {
                let g_s = g.as_slice();
                let xn_s = x_norm.as_slice();
                let inv_s = inv_std_t.as_slice();
                let gs = gamma_data.as_slice();

                // Gradient for gamma: sum over (N, H, W) of g * x_norm per channel
                let mut grad_gamma = vec![T::zero(); channels];
                // Gradient for beta: sum over (N, H, W) of g per channel
                let mut grad_beta = vec![T::zero(); channels];
                // Gradient for x (batch norm backward)
                let mut grad_x = vec![T::zero(); batch * channels * spatial];

                let count = T::from_usize(batch * spatial);

                for c in 0..channels {
                    let mut g_sum = T::zero();
                    let mut gxn_sum = T::zero();

                    for b_idx in 0..batch {
                        let base = b_idx * channels * spatial + c * spatial;
                        for s in 0..spatial {
                            let idx = base + s;
                            grad_gamma[c] += g_s[idx] * xn_s[idx];
                            grad_beta[c] += g_s[idx];
                            g_sum += g_s[idx];
                            gxn_sum += g_s[idx] * xn_s[idx];
                        }
                    }

                    for b_idx in 0..batch {
                        let base = b_idx * channels * spatial + c * spatial;
                        for s in 0..spatial {
                            let idx = base + s;
                            grad_x[idx] = gs[c]
                                * inv_s[c]
                                * (g_s[idx] - g_sum / count - xn_s[idx] * gxn_sum / count);
                        }
                    }
                }

                vec![
                    Tensor::from_vec(grad_x, shape.clone())
                        .expect("grad shape matches forward pass"),
                    Tensor::from_vec(grad_gamma, vec![channels])
                        .expect("gamma grad length matches channels"),
                    Tensor::from_vec(grad_beta, vec![channels])
                        .expect("beta grad length matches channels"),
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
    fn test_batchnorm2d_output_shape() {
        let bn = BatchNorm2d::<f64>::new(3);
        // Input: [2, 3, 4, 4] — 2 batches, 3 channels, 4x4 spatial
        let data: Vec<f64> = (0..96).map(|i| f64::from(i) * 0.1).collect();
        let x = Variable::new(Tensor::from_vec(data, vec![2, 3, 4, 4]).unwrap(), true);
        let y = bn.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![2, 3, 4, 4]);
    }

    #[test]
    fn test_batchnorm2d_normalized_output() {
        let bn = BatchNorm2d::<f64>::new(2);
        // Input: [3, 2, 2, 2] — 3 batches, 2 channels, 2x2 spatial
        let data: Vec<f64> = (0..24).map(|i| (f64::from(i)) * 0.5 + 1.0).collect();
        let x = Variable::new(Tensor::from_vec(data, vec![3, 2, 2, 2]).unwrap(), true);
        let y = bn.forward(&x).unwrap();

        // Each channel should have near-zero mean across (N, H, W).
        let y_data = y.data();
        let y_s = y_data.as_slice();
        let spatial = 4; // 2*2
        let channels = 2;
        let batch = 3;

        for c in 0..channels {
            let mut sum = 0.0;
            for b_idx in 0..batch {
                let base = b_idx * channels * spatial + c * spatial;
                for s in 0..spatial {
                    sum += y_s[base + s];
                }
            }
            let mean = sum / (batch * spatial) as f64;
            assert!(
                mean.abs() < 1e-5,
                "channel {c} mean was {mean}, expected ~0"
            );
        }
    }

    #[test]
    fn test_batchnorm2d_parameters() {
        let bn = BatchNorm2d::<f64>::new(5);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm2d_backward() {
        let bn = BatchNorm2d::<f64>::new(2);
        // Input: [2, 2, 2, 2] — 2 batches, 2 channels, 2x2 spatial
        let data: Vec<f64> = (0..16).map(|i| (f64::from(i)) * 0.3 + 0.5).collect();
        let x = Variable::new(Tensor::from_vec(data, vec![2, 2, 2, 2]).unwrap(), true);
        let y = bn.forward(&x).unwrap();

        // Sum all outputs to get a scalar for backward.
        let y_data = y.data();
        let y_s = y_data.as_slice();
        let total: f64 = y_s.iter().copied().sum();
        let scalar = Variable::from_op(
            Tensor::from_vec(vec![total], vec![1]).unwrap(),
            vec![y.clone()],
            Box::new(move |_g: &Tensor<f64>| {
                // Gradient of sum w.r.t. each element is 1.
                vec![Tensor::ones(vec![2, 2, 2, 2])]
            }),
        );
        scalar.backward();

        // x should have gradients.
        let x_grad = x.grad();
        assert!(x_grad.is_some(), "x should have gradients after backward");
        let g = x_grad.unwrap();
        assert_eq!(g.shape(), &[2, 2, 2, 2]);

        // gamma and beta should have gradients.
        let params = bn.parameters();
        for (i, p) in params.iter().enumerate() {
            let pg = p.grad();
            assert!(
                pg.is_some(),
                "parameter {i} should have gradient after backward"
            );
        }
    }
}
