use scivex_core::{Float, Tensor, linalg};

use crate::error::{MlError, Result};
use crate::traits::Predictor;

/// Ordinary least-squares linear regression.
///
/// Fits `y = Xw + b` by solving the normal equations via QR-based least squares.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct LinearRegression<T: Float> {
    pub(crate) weights: Option<Vec<T>>,
    pub(crate) bias: Option<T>,
}

impl<T: Float> Default for LinearRegression<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float> LinearRegression<T> {
    /// Create a new, unfitted linear regression model.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_core::Tensor;
    /// # use scivex_ml::linear::LinearRegression;
    /// # use scivex_ml::traits::Predictor;
    /// let x = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
    /// let y = Tensor::from_vec(vec![3.0, 5.0, 7.0, 9.0], vec![4]).unwrap();
    /// let mut model = LinearRegression::<f64>::new();
    /// model.fit(&x, &y).unwrap();
    /// let preds = model.predict(&x).unwrap();
    /// assert!((preds.as_slice()[0] - 3.0).abs() < 1e-6);
    /// ```
    pub fn new() -> Self {
        Self {
            weights: None,
            bias: None,
        }
    }

    /// Fitted coefficients (excluding bias).
    pub fn weights(&self) -> Option<&[T]> {
        self.weights.as_deref()
    }

    /// Fitted intercept.
    pub fn bias(&self) -> Option<T> {
        self.bias
    }
}

impl<T: Float> Predictor<T> for LinearRegression<T> {
    fn fit(&mut self, x: &Tensor<T>, y: &Tensor<T>) -> Result<()> {
        let (n, p) = matrix_shape(x)?;
        check_y(y, n)?;

        // Build augmented matrix [X | 1] for intercept
        let aug = augment_ones(x, n, p)?;
        let y_col = y.clone();
        let coeffs = linalg::lstsq(&aug, &y_col)?;
        let c = coeffs.as_slice();
        self.weights = Some(c[..p].to_vec());
        self.bias = Some(c[p]);
        Ok(())
    }

    fn predict(&self, x: &Tensor<T>) -> Result<Tensor<T>> {
        let w = self.weights.as_ref().ok_or(MlError::NotFitted)?;
        let b = self.bias.ok_or(MlError::NotFitted)?;
        let (n, p) = matrix_shape(x)?;
        if p != w.len() {
            return Err(MlError::DimensionMismatch {
                expected: w.len(),
                got: p,
            });
        }
        let data = x.as_slice();
        let mut out = vec![T::zero(); n];
        for i in 0..n {
            let mut val = b;
            for j in 0..p {
                val += data[i * p + j] * w[j];
            }
            out[i] = val;
        }
        Ok(Tensor::from_vec(out, vec![n])?)
    }
}

/// Ridge regression (L2 regularisation).
///
/// Solves `(X^T X + alpha * I) w = X^T y`.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct Ridge<T: Float> {
    pub(crate) alpha: T,
    pub(crate) weights: Option<Vec<T>>,
    pub(crate) bias: Option<T>,
}

impl<T: Float> Ridge<T> {
    /// Create a new Ridge regression with the given regularisation strength `alpha`.
    pub fn new(alpha: T) -> Result<Self> {
        if alpha < T::zero() {
            return Err(MlError::InvalidParameter {
                name: "alpha",
                reason: "must be non-negative",
            });
        }
        Ok(Self {
            alpha,
            weights: None,
            bias: None,
        })
    }

    /// Fitted coefficients (excluding bias).
    pub fn weights(&self) -> Option<&[T]> {
        self.weights.as_deref()
    }

    /// Fitted intercept.
    pub fn bias(&self) -> Option<T> {
        self.bias
    }
}

impl<T: Float> Predictor<T> for Ridge<T> {
    fn fit(&mut self, x: &Tensor<T>, y: &Tensor<T>) -> Result<()> {
        let (n, p) = matrix_shape(x)?;
        check_y(y, n)?;

        // Centre y by mean to compute intercept separately
        let y_data = y.as_slice();
        let y_mean = y_data.iter().copied().fold(T::zero(), |a, b| a + b) / T::from_usize(n);

        // Compute column means of X
        let x_data = x.as_slice();
        let mut x_mean = vec![T::zero(); p];
        for i in 0..n {
            for j in 0..p {
                x_mean[j] += x_data[i * p + j];
            }
        }
        for m in &mut x_mean {
            *m /= T::from_usize(n);
        }

        // Compute X^T X + alpha * I (on centred data)
        let mut xtx = vec![T::zero(); p * p];
        for i in 0..n {
            for j in 0..p {
                let xij = x_data[i * p + j] - x_mean[j];
                for k in 0..p {
                    let xik = x_data[i * p + k] - x_mean[k];
                    xtx[j * p + k] += xij * xik;
                }
            }
        }
        // Add alpha * I
        for j in 0..p {
            xtx[j * p + j] += self.alpha;
        }

        // Compute X^T y (centred)
        let mut xty = vec![T::zero(); p];
        for i in 0..n {
            let yi = y_data[i] - y_mean;
            for j in 0..p {
                let xij = x_data[i * p + j] - x_mean[j];
                xty[j] += xij * yi;
            }
        }

        let xtx_tensor = Tensor::from_vec(xtx, vec![p, p])?;
        let xty_tensor = Tensor::from_vec(xty, vec![p])?;
        let w = linalg::solve(&xtx_tensor, &xty_tensor)?;
        let w_slice = w.as_slice();

        // intercept = y_mean - x_mean . w
        let mut bias = y_mean;
        for j in 0..p {
            bias -= x_mean[j] * w_slice[j];
        }

        self.weights = Some(w_slice.to_vec());
        self.bias = Some(bias);
        Ok(())
    }

    fn predict(&self, x: &Tensor<T>) -> Result<Tensor<T>> {
        let w = self.weights.as_ref().ok_or(MlError::NotFitted)?;
        let b = self.bias.ok_or(MlError::NotFitted)?;
        let (n, p) = matrix_shape(x)?;
        if p != w.len() {
            return Err(MlError::DimensionMismatch {
                expected: w.len(),
                got: p,
            });
        }
        let data = x.as_slice();
        let mut out = vec![T::zero(); n];
        for i in 0..n {
            let mut val = b;
            for j in 0..p {
                val += data[i * p + j] * w[j];
            }
            out[i] = val;
        }
        Ok(Tensor::from_vec(out, vec![n])?)
    }
}

// ── helpers ──

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

/// Append a column of ones to a 2-D tensor.
fn augment_ones<T: Float>(x: &Tensor<T>, n: usize, p: usize) -> Result<Tensor<T>> {
    let src = x.as_slice();
    let mut aug = vec![T::zero(); n * (p + 1)];
    for i in 0..n {
        for j in 0..p {
            aug[i * (p + 1) + j] = src[i * p + j];
        }
        aug[i * (p + 1) + p] = T::one();
    }
    Ok(Tensor::from_vec(aug, vec![n, p + 1])?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_regression_simple() {
        // y = 2*x + 1
        let x = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0], vec![5, 1]).unwrap();
        let y = Tensor::from_vec(vec![3.0, 5.0, 7.0, 9.0, 11.0], vec![5]).unwrap();
        let mut lr = LinearRegression::new();
        lr.fit(&x, &y).unwrap();
        let w = lr.weights().unwrap();
        assert!((w[0] - 2.0).abs() < 1e-8);
        assert!((lr.bias().unwrap() - 1.0).abs() < 1e-8);
    }

    #[test]
    fn test_ridge_alpha_zero_matches_ols() {
        // Use non-degenerate data: y = 2*x1 + 3*x2 + 1
        let x = Tensor::from_vec(
            vec![1.0_f64, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 0.5, 0.5],
            vec![5, 2],
        )
        .unwrap();
        let y = Tensor::from_vec(vec![3.0, 4.0, 6.0, 8.0, 3.5], vec![5]).unwrap();

        let mut ols = LinearRegression::new();
        ols.fit(&x, &y).unwrap();

        let mut ridge = Ridge::new(1e-12).unwrap();
        ridge.fit(&x, &y).unwrap();

        // With near-zero alpha, Ridge should closely match OLS predictions
        let x_test = Tensor::from_vec(vec![1.0_f64, 1.0], vec![1, 2]).unwrap();
        let ols_pred = ols.predict(&x_test).unwrap();
        let ridge_pred = ridge.predict(&x_test).unwrap();
        assert!(
            (ols_pred.as_slice()[0] - ridge_pred.as_slice()[0]).abs() < 0.1,
            "OLS pred={}, Ridge pred={}",
            ols_pred.as_slice()[0],
            ridge_pred.as_slice()[0]
        );
    }

    #[test]
    fn test_predict_before_fit() {
        let lr = LinearRegression::<f64>::new();
        let x = Tensor::from_vec(vec![1.0], vec![1, 1]).unwrap();
        assert!(lr.predict(&x).is_err());
    }
}
