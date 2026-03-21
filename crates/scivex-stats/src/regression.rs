//! Ordinary Least Squares (OLS) regression.

use scivex_core::Float;
use scivex_core::Tensor;

use crate::descriptive::mean;
use crate::distributions::{Distribution, StudentT};
use crate::error::{Result, StatsError};

/// Results of an OLS regression.
///
/// # Examples
///
/// ```
/// # use scivex_core::Tensor;
/// # use scivex_stats::regression::ols;
/// let x = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0], vec![5, 1]).unwrap();
/// let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
/// let res = ols(&x, &y).unwrap();
/// assert_eq!(res.n_obs, 5);
/// assert!(res.r_squared > 0.99);
/// ```
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct OlsResult<T: Float> {
    /// Estimated coefficients (including intercept at index 0).
    pub coefficients: Vec<T>,
    /// Standard errors of the coefficients.
    pub std_errors: Vec<T>,
    /// t-statistics for each coefficient.
    pub t_statistics: Vec<T>,
    /// Two-tailed p-values for each coefficient.
    pub p_values: Vec<T>,
    /// Coefficient of determination.
    pub r_squared: T,
    /// Adjusted R-squared.
    pub r_squared_adj: T,
    /// F-statistic for overall significance.
    pub f_statistic: T,
    /// p-value for the F-statistic.
    pub f_p_value: T,
    /// Residuals (y - y_hat).
    pub residuals: Vec<T>,
    /// Number of observations.
    pub n_obs: usize,
    /// Residual degrees of freedom (n - k - 1).
    pub df_resid: usize,
}

/// Ordinary Least Squares regression.
///
/// `x` is a 2-D tensor `[n_obs x n_features]` (without intercept — it will be
/// prepended automatically). `y` is a slice of `n_obs` target values.
///
/// Returns the full regression summary.
///
/// # Examples
///
/// ```
/// # use scivex_core::Tensor;
/// # use scivex_stats::regression::ols;
/// let x = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0], vec![5, 1]).unwrap();
/// let y: Vec<f64> = vec![2.0, 4.0, 6.0, 8.0, 10.0]; // y = 2x
/// let result = ols(&x, &y).unwrap();
/// assert!(result.r_squared > 0.99);
/// ```
#[allow(clippy::too_many_lines)]
pub fn ols<T: Float>(x: &Tensor<T>, y: &[T]) -> Result<OlsResult<T>> {
    if x.ndim() != 2 {
        return Err(StatsError::InvalidParameter {
            name: "x",
            reason: "must be a 2-D tensor",
        });
    }
    let n = x.shape()[0];
    let p = x.shape()[1]; // number of predictors (not counting intercept)

    if y.len() != n {
        return Err(StatsError::LengthMismatch {
            expected: n,
            got: y.len(),
        });
    }
    if n <= p + 1 {
        return Err(StatsError::InsufficientData {
            need: p + 2,
            got: n,
        });
    }

    let one = T::from_f64(1.0);
    let k = p + 1; // total columns including intercept

    // Build design matrix X = [1 | x]
    let x_slice = x.as_slice();
    let mut design = Vec::with_capacity(n * k);
    for i in 0..n {
        design.push(one); // intercept
        for j in 0..p {
            design.push(x_slice[i * p + j]);
        }
    }
    let x_mat = Tensor::from_vec(design, vec![n, k])?;

    // Solve via least squares: beta = (X^T X)^{-1} X^T y
    let y_tensor = Tensor::from_slice(y, vec![n])?;
    let beta_tensor = x_mat.lstsq(&y_tensor)?;
    let beta: Vec<T> = beta_tensor.as_slice().to_vec();

    // Predicted values and residuals
    let y_hat_tensor = x_mat.matvec(&beta_tensor)?;
    let y_hat = y_hat_tensor.as_slice();
    let residuals: Vec<T> = y
        .iter()
        .zip(y_hat.iter())
        .map(|(&yi, &yh)| yi - yh)
        .collect();

    // RSS = sum(residuals^2)
    let rss: T = residuals.iter().map(|&r| r * r).sum();

    // TSS = sum((y - mean(y))^2)
    let y_mean = mean(y)?;
    let tss: T = y.iter().map(|&yi| (yi - y_mean) * (yi - y_mean)).sum();

    let df_resid = n - k;
    let zero = T::from_f64(0.0);

    // R-squared
    let r_squared = if tss > zero { one - rss / tss } else { one };

    // Adjusted R-squared
    let nf = T::from_f64(n as f64);
    let kf = T::from_f64(k as f64);
    let r_squared_adj = if tss > zero && n > k {
        one - (one - r_squared) * (nf - one) / (nf - kf)
    } else {
        r_squared
    };

    // MSE and variance of coefficients
    let mse = rss / T::from_f64(df_resid as f64);

    // Compute (X^T X)^{-1}
    let xt = x_mat.transpose()?;
    let xtx = xt.matmul(&x_mat)?;
    let xtx_inv = xtx.inv().map_err(|_| StatsError::SingularMatrix)?;

    // Standard errors = sqrt(MSE * diag(XTX_inv))
    let xtx_inv_slice = xtx_inv.as_slice();
    let mut std_errors = Vec::with_capacity(k);
    for i in 0..k {
        let var_i = mse * xtx_inv_slice[i * k + i];
        std_errors.push(if var_i > zero { var_i.sqrt() } else { zero });
    }

    // t-statistics and p-values
    let t_dist = StudentT::new(T::from_f64(df_resid as f64))?;
    let two = T::from_f64(2.0);
    let mut t_statistics = Vec::with_capacity(k);
    let mut p_values = Vec::with_capacity(k);
    for i in 0..k {
        if std_errors[i] > zero {
            let t = beta[i] / std_errors[i];
            let p = two * (one - t_dist.cdf(t.abs()));
            t_statistics.push(t);
            p_values.push(p);
        } else {
            t_statistics.push(zero);
            p_values.push(one);
        }
    }

    // F-statistic: (R²/p) / ((1-R²)/(n-p-1))
    let pf = T::from_f64(p as f64);
    let (f_statistic, f_p_value) = if p > 0 && tss > zero && df_resid > 0 {
        let f = (r_squared / pf) / ((one - r_squared) / T::from_f64(df_resid as f64));
        // p-value from F-dist via regularized beta
        let half = T::from_f64(0.5);
        let x_f = pf * f / (pf * f + T::from_f64(df_resid as f64));
        let ib =
            crate::special::regularized_beta(x_f, pf * half, T::from_f64(df_resid as f64) * half)?;
        (f, one - ib)
    } else {
        (zero, one)
    };

    Ok(OlsResult {
        coefficients: beta,
        std_errors,
        t_statistics,
        p_values,
        r_squared,
        r_squared_adj,
        f_statistic,
        f_p_value,
        residuals,
        n_obs: n,
        df_resid,
    })
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn test_ols_exact_linear() {
        // y = 2 + 3*x  (exact)
        let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y: Vec<f64> = x_data.iter().map(|&xi| 2.0 + 3.0 * xi).collect();
        let x = Tensor::from_vec(x_data, vec![5, 1]).unwrap();
        let result = ols(&x, &y).unwrap();
        assert!((result.coefficients[0] - 2.0).abs() < 1e-6);
        assert!((result.coefficients[1] - 3.0).abs() < 1e-6);
        assert!((result.r_squared - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_ols_noisy() {
        let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y = vec![2.1, 3.9, 6.2, 7.8, 10.1, 12.0, 13.9, 16.1, 18.0, 20.2];
        let x = Tensor::from_vec(x_data, vec![10, 1]).unwrap();
        let result = ols(&x, &y).unwrap();
        // Should be close to y = 0 + 2*x
        assert!((result.coefficients[1] - 2.0).abs() < 0.2);
        assert!(result.r_squared > 0.99);
        // Slope should be significant
        assert!(result.p_values[1] < 0.05);
    }

    #[test]
    fn test_ols_residuals_sum_near_zero() {
        let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y = vec![2.3, 4.1, 5.9, 8.2, 9.8, 12.1, 14.0, 15.9];
        let x = Tensor::from_vec(x_data, vec![8, 1]).unwrap();
        let result = ols(&x, &y).unwrap();
        let resid_sum: f64 = result.residuals.iter().sum();
        assert!(resid_sum.abs() < 1e-8);
    }

    #[test]
    fn test_ols_multiple_predictors() {
        // y = 1 + 2*x1 + 3*x2
        let x_data = vec![1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 2.0];
        let y = vec![6.0, 8.0, 10.0, 9.0, 11.0, 13.0];
        let x = Tensor::from_vec(x_data, vec![6, 2]).unwrap();
        let result = ols(&x, &y).unwrap();
        assert!((result.coefficients[0] - 1.0).abs() < 1e-6); // intercept
        assert!((result.coefficients[1] - 2.0).abs() < 1e-6); // x1
        assert!((result.coefficients[2] - 3.0).abs() < 1e-6); // x2
    }

    #[test]
    fn test_ols_insufficient_data() {
        let x = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let y = vec![1.0, 2.0];
        assert!(ols(&x, &y).is_err());
    }
}
