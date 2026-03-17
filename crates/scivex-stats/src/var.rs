//! Vector Autoregression (VAR) models for multivariate time series.
//!
//! Provides [`VarModel`] for fitting and forecasting multiple interrelated
//! time series, plus [`GrangerResult`] for Granger causality testing.

use scivex_core::Float;

use crate::error::{Result, StatsError};

// ── Granger causality result ────────────────────────────────────────────

/// Result of a Granger causality test.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone, Copy)]
pub struct GrangerResult<T: Float> {
    /// F-statistic for the Granger causality test.
    pub f_statistic: T,
    /// Approximate p-value (chi-squared approximation).
    pub p_value: T,
    /// Whether the result is significant at the 5% level.
    pub significant: bool,
}

// ── VAR model ───────────────────────────────────────────────────────────

/// Vector Autoregression model of order `p`.
///
/// Models `k` interrelated time series where each variable is regressed on
/// its own lagged values and the lagged values of all other variables.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct VarModel<T: Float> {
    /// VAR order (number of lags).
    p: usize,
    /// Number of variables (time series).
    k: usize,
    /// Coefficient matrices: `p` matrices, each `k × k`.
    /// `coefficients[lag][row][col]` is the effect of variable `col` at lag
    /// `lag+1` on variable `row`.
    coefficients: Vec<Vec<Vec<T>>>,
    /// Intercept vector of length `k`.
    intercept: Vec<T>,
    /// Whether the model has been fitted.
    fitted: bool,
    /// Stored data for forecasting (last `p` observations per variable).
    history: Vec<Vec<T>>,
}

impl<T: Float> VarModel<T> {
    /// Create a new VAR model of order `p`.
    ///
    /// # Errors
    ///
    /// Returns [`StatsError::InvalidParameter`] if `p < 1`.
    pub fn new(p: usize) -> Result<Self> {
        if p < 1 {
            return Err(StatsError::InvalidParameter {
                name: "p",
                reason: "VAR order must be >= 1",
            });
        }
        Ok(Self {
            p,
            k: 0,
            coefficients: Vec::new(),
            intercept: Vec::new(),
            fitted: false,
            history: Vec::new(),
        })
    }

    /// Fit the VAR(p) model to multivariate data via OLS (normal equations).
    ///
    /// `data` is a slice of `k` vectors, each of equal length `n` (the number
    /// of time observations). The model requires `n > k * p + 1`.
    ///
    /// # Errors
    ///
    /// - [`StatsError::InvalidParameter`] if `data` is empty.
    /// - [`StatsError::LengthMismatch`] if the series have different lengths.
    /// - [`StatsError::InsufficientData`] if there are not enough observations.
    /// - [`StatsError::SingularMatrix`] if the normal equations are singular.
    #[allow(clippy::too_many_lines)]
    pub fn fit(&mut self, data: &[Vec<T>]) -> Result<()> {
        let k = data.len();
        if k == 0 {
            return Err(StatsError::InvalidParameter {
                name: "data",
                reason: "must contain at least one variable",
            });
        }

        let n = data[0].len();
        for series in data.iter().skip(1) {
            if series.len() != n {
                return Err(StatsError::LengthMismatch {
                    expected: n,
                    got: series.len(),
                });
            }
        }

        let p = self.p;
        let usable = n.saturating_sub(p);
        let n_regs = k * p + 1; // intercept + k*p lagged values

        if usable < n_regs + 1 {
            return Err(StatsError::InsufficientData {
                need: p + n_regs + 1,
                got: n,
            });
        }

        self.k = k;

        // Build design matrix Z (usable × n_regs) and response matrix Y (usable × k).
        // For each time t = p..n:
        //   row of Z = [1, y_1(t-1), y_2(t-1), ..., y_k(t-1), y_1(t-2), ..., y_k(t-p)]
        //   row of Y = [y_1(t), y_2(t), ..., y_k(t)]
        let m = usable; // number of observations for regression

        // Z stored in row-major: z[row * n_regs + col]
        let mut z = vec![T::zero(); m * n_regs];
        // Y stored in row-major: y_mat[row * k + col]
        let mut y_mat = vec![T::zero(); m * k];

        for (row, t) in (p..n).enumerate() {
            // Intercept
            z[row * n_regs] = T::one();

            // Lagged values
            for lag in 0..p {
                for var in 0..k {
                    z[row * n_regs + 1 + lag * k + var] = data[var][t - 1 - lag];
                }
            }

            // Response
            for var in 0..k {
                y_mat[row * k + var] = data[var][t];
            }
        }

        // Solve via normal equations: (Z'Z) B = Z'Y
        // where B is n_regs × k
        let nr = n_regs;

        // Compute Z'Z (nr × nr)
        let mut ztz = vec![T::zero(); nr * nr];
        for i in 0..nr {
            for j in i..nr {
                let mut s = T::zero();
                for r in 0..m {
                    s += z[r * nr + i] * z[r * nr + j];
                }
                ztz[i * nr + j] = s;
                ztz[j * nr + i] = s; // symmetric
            }
        }

        // Compute Z'Y (nr × k)
        let mut zty = vec![T::zero(); nr * k];
        for i in 0..nr {
            for j in 0..k {
                let mut s = T::zero();
                for r in 0..m {
                    s += z[r * nr + i] * y_mat[r * k + j];
                }
                zty[i * k + j] = s;
            }
        }

        // Solve for each column of B separately: (Z'Z) b_j = (Z'Y)_j
        let mut intercept = vec![T::zero(); k];
        let mut coefficients = vec![vec![vec![T::zero(); k]; k]; p];

        for eq in 0..k {
            // Extract column eq of Z'Y
            let mut rhs = vec![T::zero(); nr];
            for i in 0..nr {
                rhs[i] = zty[i * k + eq];
            }

            let beta = solve_linear_system(&ztz, &rhs, nr)?;

            intercept[eq] = beta[0];

            for lag in 0..p {
                for var in 0..k {
                    coefficients[lag][eq][var] = beta[1 + lag * k + var];
                }
            }
        }

        self.intercept = intercept;
        self.coefficients = coefficients;

        // Store last p observations for forecasting
        self.history = Vec::with_capacity(k);
        for series in data {
            let start = n.saturating_sub(p);
            self.history.push(series[start..].to_vec());
        }

        self.fitted = true;
        Ok(())
    }

    /// Forecast `steps` steps ahead from the end of the fitted data.
    ///
    /// Returns `k` vectors, each of length `steps`.
    ///
    /// # Errors
    ///
    /// Returns [`StatsError::InvalidParameter`] if the model has not been fitted.
    pub fn forecast(&self, steps: usize) -> Result<Vec<Vec<T>>> {
        if !self.fitted {
            return Err(StatsError::InvalidParameter {
                name: "model",
                reason: "model must be fitted before forecasting",
            });
        }

        let k = self.k;
        let p = self.p;

        // Build an extended history: start with the last p observations,
        // then append forecasts as we go.
        let mut extended: Vec<Vec<T>> = self.history.clone();

        for _ in 0..steps {
            let t = extended[0].len();
            let mut new_vals = vec![T::zero(); k];
            for (eq, pred_val) in new_vals.iter_mut().enumerate() {
                let mut pred = self.intercept[eq];
                for lag in 0..p {
                    let idx = t - 1 - lag;
                    if idx < extended[0].len() {
                        for (var, ext_series) in extended.iter().enumerate() {
                            pred += self.coefficients[lag][eq][var] * ext_series[idx];
                        }
                    }
                }
                *pred_val = pred;
            }
            for (ext_series, val) in extended.iter_mut().zip(new_vals) {
                ext_series.push(val);
            }
        }

        // Extract only the forecasted values
        let mut result = Vec::with_capacity(k);
        for series in &extended {
            result.push(series[p..].to_vec());
        }

        Ok(result)
    }

    /// Return the lag coefficient matrices.
    ///
    /// Returns `p` matrices, each `k × k`. Element `[lag][i][j]` is the
    /// effect of variable `j` at lag `lag+1` on variable `i`.
    pub fn coefficients(&self) -> &[Vec<Vec<T>>] {
        &self.coefficients
    }

    /// Compute in-sample residuals.
    ///
    /// Returns `k` vectors, each of length `n - p`, where `n` is the length
    /// of the original data.
    ///
    /// # Errors
    ///
    /// Returns an error if the model has not been fitted, or if `data`
    /// dimensions do not match.
    pub fn residuals(&self, data: &[Vec<T>]) -> Result<Vec<Vec<T>>> {
        if !self.fitted {
            return Err(StatsError::InvalidParameter {
                name: "model",
                reason: "model must be fitted before computing residuals",
            });
        }

        let k = self.k;
        let p = self.p;

        if data.len() != k {
            return Err(StatsError::LengthMismatch {
                expected: k,
                got: data.len(),
            });
        }

        let n = data[0].len();
        for series in data.iter().skip(1) {
            if series.len() != n {
                return Err(StatsError::LengthMismatch {
                    expected: n,
                    got: series.len(),
                });
            }
        }

        if n <= p {
            return Err(StatsError::InsufficientData {
                need: p + 1,
                got: n,
            });
        }

        let m = n - p;
        let mut resids = vec![vec![T::zero(); m]; k];

        for (row, t) in (p..n).enumerate() {
            for eq in 0..k {
                let mut pred = self.intercept[eq];
                for lag in 0..p {
                    for (var, d) in data.iter().enumerate() {
                        pred += self.coefficients[lag][eq][var] * d[t - 1 - lag];
                    }
                }
                resids[eq][row] = data[eq][t] - pred;
            }
        }

        Ok(resids)
    }

    /// Perform a Granger causality test.
    ///
    /// Tests whether lags of variable `cause` improve the prediction of
    /// variable `effect` beyond what the other variables already provide.
    ///
    /// Uses an F-test comparing the unrestricted model (all variables) to a
    /// restricted model (excluding `cause` lags from the `effect` equation).
    ///
    /// # Errors
    ///
    /// Returns an error if the model is not fitted, indices are out of range,
    /// or a linear system is singular.
    #[allow(clippy::too_many_lines)]
    pub fn granger_causality(
        &self,
        data: &[Vec<T>],
        cause: usize,
        effect: usize,
    ) -> Result<GrangerResult<T>> {
        if !self.fitted {
            return Err(StatsError::InvalidParameter {
                name: "model",
                reason: "model must be fitted before Granger causality test",
            });
        }

        let k = self.k;
        let p = self.p;

        if cause >= k || effect >= k {
            return Err(StatsError::InvalidParameter {
                name: "cause/effect",
                reason: "index out of range",
            });
        }

        if data.len() != k {
            return Err(StatsError::LengthMismatch {
                expected: k,
                got: data.len(),
            });
        }

        let n = data[0].len();
        for series in data.iter().skip(1) {
            if series.len() != n {
                return Err(StatsError::LengthMismatch {
                    expected: n,
                    got: series.len(),
                });
            }
        }

        let m = n.saturating_sub(p); // usable observations

        if m < k * p + 2 {
            return Err(StatsError::InsufficientData {
                need: p + k * p + 2,
                got: n,
            });
        }

        // Unrestricted RSS: use the full model residuals for `effect`
        let full_resids = self.residuals(data)?;
        let mut rss_u = T::zero();
        for &e in &full_resids[effect] {
            rss_u += e * e;
        }

        // Restricted model: fit `effect` equation without `cause` lags
        let nr_restricted = 1 + (k - 1) * p; // intercept + (k-1)*p regressors

        let mut z_r = vec![T::zero(); m * nr_restricted];
        let mut y_r = vec![T::zero(); m];

        for (row, t) in (p..n).enumerate() {
            z_r[row * nr_restricted] = T::one();

            let mut col = 1;
            for lag in 0..p {
                for (var, d) in data.iter().enumerate() {
                    if var == cause {
                        continue;
                    }
                    z_r[row * nr_restricted + col] = d[t - 1 - lag];
                    col += 1;
                }
            }
            y_r[row] = data[effect][t];
        }

        // Solve restricted OLS
        let mut ztz_r = vec![T::zero(); nr_restricted * nr_restricted];
        for i in 0..nr_restricted {
            for j in i..nr_restricted {
                let mut s = T::zero();
                for r in 0..m {
                    s += z_r[r * nr_restricted + i] * z_r[r * nr_restricted + j];
                }
                ztz_r[i * nr_restricted + j] = s;
                ztz_r[j * nr_restricted + i] = s;
            }
        }

        let mut zty_r = vec![T::zero(); nr_restricted];
        for i in 0..nr_restricted {
            let mut s = T::zero();
            for r in 0..m {
                s += z_r[r * nr_restricted + i] * y_r[r];
            }
            zty_r[i] = s;
        }

        let beta_r = solve_linear_system(&ztz_r, &zty_r, nr_restricted)?;

        // Compute restricted RSS
        let mut rss_r = T::zero();
        for r in 0..m {
            let mut pred = T::zero();
            for i in 0..nr_restricted {
                pred += beta_r[i] * z_r[r * nr_restricted + i];
            }
            let e = y_r[r] - pred;
            rss_r += e * e;
        }

        // F-test: F = ((RSS_r - RSS_u) / p) / (RSS_u / (T - kp - 1))
        let df1 = p; // number of restricted parameters
        let df2 = m.saturating_sub(k * p + 1);

        if df2 == 0 {
            return Err(StatsError::InsufficientData {
                need: p + k * p + 2,
                got: n,
            });
        }

        let f_stat =
            ((rss_r - rss_u) / T::from_f64(df1 as f64)) / (rss_u / T::from_f64(df2 as f64));

        // Approximate p-value using the F-distribution CDF.
        // Use the regularized incomplete beta function:
        //   P(F <= x) = I_{x·d1/(x·d1+d2)}(d1/2, d2/2)
        // p-value = 1 - CDF(f_stat)
        let p_value = f_distribution_sf(f_stat, df1, df2);

        let significant = p_value < T::from_f64(0.05);

        Ok(GrangerResult {
            f_statistic: f_stat,
            p_value,
            significant,
        })
    }
}

// ── Linear algebra helpers ──────────────────────────────────────────────

/// Solve a linear system Ax = b using Gaussian elimination with partial pivoting.
fn solve_linear_system<T: Float>(a: &[T], b: &[T], n: usize) -> Result<Vec<T>> {
    let mut aug = vec![T::zero(); n * (n + 1)];
    for i in 0..n {
        for j in 0..n {
            aug[i * (n + 1) + j] = a[i * n + j];
        }
        aug[i * (n + 1) + n] = b[i];
    }

    // Forward elimination
    for col in 0..n {
        // Partial pivoting
        let mut max_row = col;
        let mut max_val = aug[col * (n + 1) + col].abs();
        for row in (col + 1)..n {
            let val = aug[row * (n + 1) + col].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }
        if max_val < T::from_f64(1e-15) {
            return Err(StatsError::SingularMatrix);
        }
        if max_row != col {
            for j in 0..=n {
                aug.swap(col * (n + 1) + j, max_row * (n + 1) + j);
            }
        }
        let pivot = aug[col * (n + 1) + col];
        for row in (col + 1)..n {
            let factor = aug[row * (n + 1) + col] / pivot;
            for j in col..=n {
                let pivot_val = aug[col * (n + 1) + j];
                aug[row * (n + 1) + j] -= factor * pivot_val;
            }
        }
    }

    // Back substitution
    let mut x = vec![T::zero(); n];
    for i in (0..n).rev() {
        let mut sum = aug[i * (n + 1) + n];
        for j in (i + 1)..n {
            sum -= aug[i * (n + 1) + j] * x[j];
        }
        x[i] = sum / aug[i * (n + 1) + i];
    }

    Ok(x)
}

/// Approximate survival function (1 - CDF) of the F-distribution.
///
/// Uses the regularized incomplete beta function relationship:
///   P(F > x) = 1 - I_{d1*x/(d1*x+d2)}(d1/2, d2/2)
///
/// The incomplete beta is computed via a continued fraction expansion.
fn f_distribution_sf<T: Float>(x: T, d1: usize, d2: usize) -> T {
    if x <= T::zero() {
        return T::one();
    }

    let d1f = T::from_f64(d1 as f64);
    let d2f = T::from_f64(d2 as f64);
    let a = d1f / T::from_f64(2.0);
    let b = d2f / T::from_f64(2.0);
    let z = d1f * x / (d1f * x + d2f);

    // p-value = 1 - I_z(a, b)
    let ibeta = regularized_incomplete_beta(z, a, b);
    T::one() - ibeta
}

/// Regularized incomplete beta function I_x(a, b) via continued fraction
/// (Lentz's method).
fn regularized_incomplete_beta<T: Float>(x: T, a: T, b: T) -> T {
    let zero = T::zero();
    let one = T::one();
    let two = T::from_f64(2.0);
    let eps = T::from_f64(1e-14);
    let tiny = T::from_f64(1e-30);

    if x <= zero {
        return zero;
    }
    if x >= one {
        return one;
    }

    // Use the symmetry relation if x > (a+1)/(a+b+2)
    let threshold = (a + one) / (a + b + two);
    if x > threshold {
        return one - regularized_incomplete_beta(one - x, b, a);
    }

    // ln(B(a,b)) = ln(Gamma(a)) + ln(Gamma(b)) - ln(Gamma(a+b))
    let ln_beta = ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b);

    // Front factor: x^a * (1-x)^b / (a * B(a,b))
    let front = (a * x.ln() + b * (one - x).ln() - ln_beta).exp() / a;

    // Continued fraction (Lentz's method)
    let mut c = one;
    let mut d = one - (a + b) * x / (a + one);
    if d.abs() < tiny {
        d = tiny;
    }
    d = one / d;
    let mut h = d;

    let max_iter: i32 = 200;
    for m_val in 1..=max_iter {
        let m = T::from_f64(f64::from(m_val));

        // Even step: d_{2m}
        let num_even = m * (b - m) * x / ((a + two * m - one) * (a + two * m));
        d = one + num_even * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = one + num_even / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = one / d;
        h *= d * c;

        // Odd step: d_{2m+1}
        let num_odd = -((a + m) * (a + b + m) * x) / ((a + two * m) * (a + two * m + one));
        d = one + num_odd * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = one + num_odd / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = one / d;
        let delta = d * c;
        h *= delta;

        if (delta - one).abs() < eps {
            break;
        }
    }

    front * h
}

/// Natural log of the Gamma function via Stirling's approximation
/// (Lanczos approximation with g=7).
fn ln_gamma<T: Float>(x: T) -> T {
    // Lanczos coefficients for g=7, n=9
    #[allow(clippy::excessive_precision)]
    let coefs: [f64; 9] = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1_259.139_216_722_403,
        771.323_428_777_653_08,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];

    let one = T::one();
    let half = T::from_f64(0.5);
    let g = T::from_f64(7.0);

    if x < half {
        // Reflection formula: Gamma(x) = pi / (sin(pi*x) * Gamma(1-x))
        let pi = T::from_f64(core::f64::consts::PI);
        let sin_val = (pi * x).sin();
        if sin_val.abs() < T::from_f64(1e-30) {
            return T::from_f64(f64::INFINITY);
        }
        return pi.ln() - sin_val.abs().ln() - ln_gamma(one - x);
    }

    let x = x - one;
    let mut sum = T::from_f64(coefs[0]);
    for (i, &c) in coefs.iter().enumerate().skip(1) {
        sum += T::from_f64(c) / (x + T::from_f64(i as f64));
    }

    let t = x + g + half;
    let ln_sqrt_2pi = T::from_f64(0.918_938_533_204_672_8);

    ln_sqrt_2pi + (x + half) * t.ln() - t + sum.ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate a simple bivariate VAR(1) process:
    ///   y1(t) = 0.5 * y1(t-1) + 0.3 * y2(t-1) + noise
    ///   y2(t) = 0.2 * y1(t-1) + 0.4 * y2(t-1) + noise
    fn generate_var1_data(n: usize) -> Vec<Vec<f64>> {
        let mut y1 = vec![0.0; n];
        let mut y2 = vec![0.0; n];
        for t in 1..n {
            // Deterministic pseudo-noise for reproducibility
            let noise1 = ((t * 17 + 3) % 11) as f64 / 55.0 - 0.1;
            let noise2 = ((t * 13 + 7) % 11) as f64 / 55.0 - 0.1;
            y1[t] = 0.5 * y1[t - 1] + 0.3 * y2[t - 1] + noise1;
            y2[t] = 0.2 * y1[t - 1] + 0.4 * y2[t - 1] + noise2;
        }
        vec![y1, y2]
    }

    #[test]
    fn test_var_bivariate() {
        let data = generate_var1_data(200);
        let mut model = VarModel::new(1).unwrap();
        model.fit(&data).unwrap();

        // Check we got 1 coefficient matrix of size 2×2
        assert_eq!(model.coefficients().len(), 1);
        assert_eq!(model.coefficients()[0].len(), 2);
        assert_eq!(model.coefficients()[0][0].len(), 2);

        // Forecast
        let fc = model.forecast(5).unwrap();
        assert_eq!(fc.len(), 2); // 2 variables
        assert_eq!(fc[0].len(), 5); // 5 steps
        assert_eq!(fc[1].len(), 5);

        // Forecasts should be finite
        for var in &fc {
            for &val in var {
                assert!(val.is_finite(), "forecast should be finite, got {val}");
            }
        }
    }

    #[test]
    fn test_var_coefficients_shape() {
        // k=3 variables, p=2 lags
        let n = 200;
        let mut y1 = vec![0.0_f64; n];
        let mut y2 = vec![0.0; n];
        let mut y3 = vec![0.0; n];
        for t in 1..n {
            let n1 = ((t * 17 + 3) % 11) as f64 / 55.0;
            let n2 = ((t * 13 + 7) % 11) as f64 / 55.0;
            let n3 = ((t * 11 + 5) % 13) as f64 / 65.0;
            y1[t] = 0.3 * y1[t.saturating_sub(1)] + 0.1 * y2[t.saturating_sub(1)] + n1;
            y2[t] = 0.2 * y1[t.saturating_sub(1)] + 0.4 * y3[t.saturating_sub(1)] + n2;
            y3[t] = 0.1 * y2[t.saturating_sub(1)] + 0.5 * y3[t.saturating_sub(1)] + n3;
        }
        let data = vec![y1, y2, y3];

        let mut model = VarModel::new(2).unwrap();
        model.fit(&data).unwrap();

        // p=2 coefficient matrices, each 3×3
        let coeffs = model.coefficients();
        assert_eq!(coeffs.len(), 2);
        for mat in coeffs {
            assert_eq!(mat.len(), 3);
            for row in mat {
                assert_eq!(row.len(), 3);
            }
        }
    }

    #[test]
    fn test_var_residuals() {
        let data = generate_var1_data(200);
        let mut model = VarModel::new(1).unwrap();
        model.fit(&data).unwrap();

        let resids = model.residuals(&data).unwrap();
        assert_eq!(resids.len(), 2); // 2 variables
        assert_eq!(resids[0].len(), 199); // n - p = 200 - 1

        // Residuals should be finite
        for var in &resids {
            for &val in var {
                assert!(val.is_finite(), "residual should be finite, got {val}");
            }
        }

        // Mean of residuals should be near zero (good fit)
        for (i, var) in resids.iter().enumerate() {
            let sum: f64 = var.iter().sum();
            let mean = sum / var.len() as f64;
            assert!(
                mean.abs() < 0.5,
                "mean residual for var {i} should be near zero, got {mean}"
            );
        }
    }

    #[test]
    fn test_var_granger_causality() {
        // Create data where y1 clearly Granger-causes y2 but not vice versa
        let n = 500;
        let mut y1 = vec![0.0_f64; n];
        let mut y2 = vec![0.0; n];
        for t in 1..n {
            let noise1 = ((t * 17 + 3) % 11) as f64 / 110.0 - 0.05;
            let noise2 = ((t * 13 + 7) % 11) as f64 / 110.0 - 0.05;
            // y1 is autoregressive only (no influence from y2)
            y1[t] = 0.7 * y1[t - 1] + noise1;
            // y2 depends strongly on y1 (Granger causality)
            y2[t] = 0.8 * y1[t - 1] + 0.1 * y2[t - 1] + noise2;
        }
        let data = vec![y1, y2];

        let mut model = VarModel::new(1).unwrap();
        model.fit(&data).unwrap();

        // y1 -> y2 should be significant
        let result = model.granger_causality(&data, 0, 1).unwrap();
        assert!(
            result.f_statistic > 1.0,
            "F-statistic should be large for causal relationship, got {}",
            result.f_statistic
        );
        assert!(result.f_statistic.is_finite());
        assert!(result.p_value.is_finite());
        // Strong causal link should be significant
        assert!(
            result.significant,
            "y1 -> y2 should be significant (F={}, p={})",
            result.f_statistic, result.p_value
        );
    }

    #[test]
    fn test_var_insufficient_data() {
        let data = vec![vec![1.0_f64, 2.0], vec![3.0, 4.0]];
        let mut model = VarModel::new(2).unwrap();
        assert!(model.fit(&data).is_err());
    }
}
