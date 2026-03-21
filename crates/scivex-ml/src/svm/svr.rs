use scivex_core::{Float, Tensor};

use crate::error::{MlError, Result};
use crate::traits::Predictor;

use super::kernel::Kernel;

/// Support Vector Regressor (epsilon-SVR).
///
/// Uses a simplified SMO-like coordinate descent algorithm with an
/// epsilon-insensitive loss function: only errors larger than `epsilon`
/// contribute to the loss.
///
/// # Examples
///
/// ```
/// # use scivex_ml::prelude::*;
/// # use scivex_core::prelude::*;
/// let x = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
/// let y = Tensor::from_vec(vec![2.0, 4.0, 6.0, 8.0], vec![4]).unwrap();
/// let mut svr = SVR::new(Kernel::Linear, 100.0, 0.5).unwrap();
/// svr.fit(&x, &y).unwrap();
/// let preds = svr.predict(&x).unwrap();
/// ```
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct SVR<T: Float> {
    kernel: Kernel<T>,
    c: T,
    epsilon: T,
    tol: T,
    max_iter: usize,
    // Fitted state: coefficients = alpha_i - alpha_i*
    coeffs: Option<Vec<T>>,
    bias: T,
    train_x: Option<Vec<Vec<T>>>,
}

impl<T: Float> SVR<T> {
    /// Create a new SVR.
    ///
    /// - `kernel`: kernel function
    /// - `c`: regularisation parameter
    /// - `epsilon`: width of the epsilon-insensitive tube
    pub fn new(kernel: Kernel<T>, c: f64, epsilon: f64) -> Result<Self> {
        if c <= 0.0 {
            return Err(MlError::InvalidParameter {
                name: "c",
                reason: "must be positive",
            });
        }
        if epsilon < 0.0 {
            return Err(MlError::InvalidParameter {
                name: "epsilon",
                reason: "must be non-negative",
            });
        }
        Ok(Self {
            kernel,
            c: T::from_f64(c),
            epsilon: T::from_f64(epsilon),
            tol: T::from_f64(1e-3),
            max_iter: 1000,
            coeffs: None,
            bias: T::zero(),
            train_x: None,
        })
    }

    /// Set the convergence tolerance.
    pub fn set_tol(&mut self, tol: f64) -> &mut Self {
        self.tol = T::from_f64(tol.max(1e-12));
        self
    }

    /// Set the maximum number of iterations.
    pub fn set_max_iter(&mut self, max_iter: usize) -> &mut Self {
        self.max_iter = max_iter.max(1);
        self
    }
}

impl<T: Float> Predictor<T> for SVR<T> {
    fn fit(&mut self, x: &Tensor<T>, y: &Tensor<T>) -> Result<()> {
        let (n, p) = matrix_shape(x)?;
        check_y(y, n)?;
        let x_data = x.as_slice();
        let y_data = y.as_slice();

        let x_rows: Vec<Vec<T>> = (0..n)
            .map(|i| x_data[i * p..(i + 1) * p].to_vec())
            .collect();

        // Pre-compute kernel matrix
        let mut k = vec![T::zero(); n * n];
        for i in 0..n {
            for j in i..n {
                let kij = self.kernel.compute(&x_rows[i], &x_rows[j]);
                k[i * n + j] = kij;
                k[j * n + i] = kij;
            }
        }

        // Coordinate descent on the dual of epsilon-SVR.
        // We optimise w_i = alpha_i - alpha_i* in [-C, C].
        // The prediction is f(x) = sum_i w_i * K(x_i, x) + b.
        // Gradient for w_i: g_i = sum_j w_j * K(x_j, x_i) + b - y_i
        // For epsilon-insensitive: we want |g_i| <= epsilon for non-bound w_i.
        let mut w = vec![T::zero(); n];
        let mut b = T::zero();

        for _iter in 0..self.max_iter {
            let mut max_violation = T::zero();

            for i in 0..n {
                // Current prediction at x_i
                let mut fi = b;
                for j in 0..n {
                    fi += w[j] * k[j * n + i];
                }
                let residual = fi - y_data[i];

                // Compute desired update
                let grad = if residual > self.epsilon {
                    residual - self.epsilon
                } else if residual < -self.epsilon {
                    residual + self.epsilon
                } else {
                    T::zero()
                };

                if grad.abs() < self.tol {
                    continue;
                }

                let violation = grad.abs();
                if violation > max_violation {
                    max_violation = violation;
                }

                // Newton step: delta = -grad / K(x_i, x_i)
                let kii = k[i * n + i];
                if kii < T::from_f64(1e-12) {
                    continue;
                }
                let delta = -grad / kii;
                let new_w = (w[i] + delta).max(-self.c).min(self.c);
                let actual_delta = new_w - w[i];
                w[i] = new_w;

                // Update bias: running average of (y_i - f(x_i) without bias)
                // For free SVs (|w_i| < C), b = y_i - sum_j w_j K(x_j, x_i) ± epsilon
                b += actual_delta * kii * T::from_f64(0.1); // damped bias update
            }

            if max_violation < self.tol {
                break;
            }
        }

        // Recompute bias from support vectors
        let mut b_sum = T::zero();
        let mut b_count = 0usize;
        for i in 0..n {
            if w[i].abs() > T::from_f64(1e-8) && w[i].abs() < self.c - T::from_f64(1e-8) {
                let mut fi = T::zero();
                for j in 0..n {
                    fi += w[j] * k[j * n + i];
                }
                if w[i] > T::zero() {
                    b_sum += y_data[i] - fi - self.epsilon;
                } else {
                    b_sum += y_data[i] - fi + self.epsilon;
                }
                b_count += 1;
            }
        }
        if b_count > 0 {
            b = b_sum / T::from_usize(b_count);
        }

        // Store support vectors (non-zero w)
        let mut sv_x = Vec::new();
        let mut sv_w = Vec::new();
        for i in 0..n {
            if w[i].abs() > T::from_f64(1e-8) {
                sv_x.push(x_rows[i].clone());
                sv_w.push(w[i]);
            }
        }

        // If no SVs found, store all with their weights (fallback)
        if sv_x.is_empty() {
            sv_x = x_rows;
            sv_w = w;
        }

        self.coeffs = Some(sv_w);
        self.bias = b;
        self.train_x = Some(sv_x);

        Ok(())
    }

    fn predict(&self, x: &Tensor<T>) -> Result<Tensor<T>> {
        let sx = self.train_x.as_ref().ok_or(MlError::NotFitted)?;
        let coeffs = self.coeffs.as_ref().ok_or(MlError::NotFitted)?;
        let (n, p) = matrix_shape(x)?;
        let data = x.as_slice();

        let mut out = vec![T::zero(); n];
        for i in 0..n {
            let xi = &data[i * p..(i + 1) * p];
            let mut sum = T::zero();
            for (j, sv) in sx.iter().enumerate() {
                sum += coeffs[j] * self.kernel.compute(sv, xi);
            }
            out[i] = sum + self.bias;
        }

        Tensor::from_vec(out, vec![n]).map_err(MlError::from)
    }
}

fn matrix_shape<T: Float>(x: &Tensor<T>) -> Result<(usize, usize)> {
    let s = x.shape();
    if s.len() != 2 {
        return Err(MlError::DimensionMismatch {
            expected: 2,
            got: s.len(),
        });
    }
    if s[0] == 0 {
        return Err(MlError::EmptyInput);
    }
    Ok((s[0], s[1]))
}

fn check_y<T: Float>(y: &Tensor<T>, n: usize) -> Result<()> {
    if y.ndim() != 1 || y.shape()[0] != n {
        return Err(MlError::DimensionMismatch {
            expected: n,
            got: y.shape()[0],
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_svr_linear() {
        // Simple linear relationship: y = 2x
        let x =
            Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![8, 1]).unwrap();
        let y =
            Tensor::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0], vec![8]).unwrap();

        let mut svr = SVR::new(Kernel::Linear, 100.0, 0.5).unwrap();
        svr.set_max_iter(2000);
        svr.fit(&x, &y).unwrap();
        let preds = svr.predict(&x).unwrap();

        for (&p, &t) in preds.as_slice().iter().zip(y.as_slice()) {
            assert!(
                (p - t).abs() < 5.0,
                "prediction {p} too far from target {t}"
            );
        }
    }

    #[test]
    fn test_svr_rbf() {
        let x = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0], vec![5, 1]).unwrap();
        let y = Tensor::from_vec(vec![1.0, 4.0, 9.0, 16.0, 25.0], vec![5]).unwrap();

        let mut svr = SVR::new(Kernel::Rbf { gamma: 0.5 }, 100.0, 0.1).unwrap();
        svr.fit(&x, &y).unwrap();
        let preds = svr.predict(&x).unwrap();

        // Predictions should be in reasonable range
        for &p in preds.as_slice() {
            assert!(p > -10.0 && p < 40.0, "prediction {p} out of range");
        }
    }

    #[test]
    fn test_svr_not_fitted() {
        let svr = SVR::<f64>::new(Kernel::Linear, 1.0, 0.1).unwrap();
        let x = Tensor::from_vec(vec![1.0_f64, 2.0], vec![2, 1]).unwrap();
        assert!(svr.predict(&x).is_err());
    }

    #[test]
    fn test_svr_invalid_params() {
        assert!(SVR::<f64>::new(Kernel::Linear, 0.0, 0.1).is_err());
        assert!(SVR::<f64>::new(Kernel::Linear, 1.0, -0.1).is_err());
    }

    #[test]
    fn test_svr_predictions_in_range() {
        let x = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
        let y = Tensor::from_vec(vec![10.0, 20.0, 30.0, 40.0], vec![4]).unwrap();

        let mut svr = SVR::new(Kernel::Linear, 100.0, 1.0).unwrap();
        svr.set_max_iter(2000);
        svr.fit(&x, &y).unwrap();
        let preds = svr.predict(&x).unwrap();

        // At minimum, predictions should be in a reasonable ballpark
        for &p in preds.as_slice() {
            assert!(p > -10.0 && p < 60.0, "prediction {p} wildly out of range");
        }
    }
}
