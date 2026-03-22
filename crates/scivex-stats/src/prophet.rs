//! Prophet-style additive decomposable time series forecasting.
//!
//! Implements the model `y(t) = g(t) + s(t) + h(t) + ε` where:
//! - `g(t)` is a piecewise-linear trend with changepoints
//! - `s(t)` is Fourier-based seasonality (yearly and/or weekly)
//! - `h(t)` is for holiday effects (currently not modeled)
//! - `ε` is noise
//!
//! The model is fitted via MAP estimation using ridge regression on a combined
//! design matrix of trend and seasonality features.

use scivex_core::Float;

use crate::error::{Result, StatsError};

// ── Configuration ─────────────────────────────────────────────────────

/// Configuration for Prophet-style forecasting.
///
/// # Examples
///
/// ```
/// # use scivex_stats::prophet::ProphetConfig;
/// let config = ProphetConfig::<f64>::new();
/// assert!(config.yearly_seasonality);
/// assert!(config.weekly_seasonality);
/// assert_eq!(config.n_changepoints, 25);
/// ```
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct ProphetConfig<T: Float> {
    /// Whether to include yearly seasonality (period = 365.25).
    pub yearly_seasonality: bool,
    /// Whether to include weekly seasonality (period = 7).
    pub weekly_seasonality: bool,
    /// Fourier order for yearly seasonality.
    pub yearly_fourier_order: usize,
    /// Fourier order for weekly seasonality.
    pub weekly_fourier_order: usize,
    /// Fraction of the data in which to place changepoints (default 0.8).
    pub changepoint_range: T,
    /// Number of potential changepoints (default 25).
    pub n_changepoints: usize,
    /// Regularization strength for changepoint magnitudes (default 0.05).
    pub changepoint_prior_scale: T,
    /// Regularization strength for seasonality coefficients (default 10.0).
    pub seasonality_prior_scale: T,
}

impl<T: Float> ProphetConfig<T> {
    /// Create a new configuration with sensible defaults.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_stats::prophet::ProphetConfig;
    /// let config = ProphetConfig::<f64>::new();
    /// assert_eq!(config.yearly_fourier_order, 10);
    /// assert_eq!(config.weekly_fourier_order, 3);
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self {
            yearly_seasonality: true,
            weekly_seasonality: true,
            yearly_fourier_order: 10,
            weekly_fourier_order: 3,
            changepoint_range: T::from_f64(0.8),
            n_changepoints: 25,
            changepoint_prior_scale: T::from_f64(0.05),
            seasonality_prior_scale: T::from_f64(10.0),
        }
    }
}

impl<T: Float> Default for ProphetConfig<T> {
    fn default() -> Self {
        Self::new()
    }
}

// ── Model ─────────────────────────────────────────────────────────────

/// A fitted Prophet-style time series model.
///
/// # Examples
///
/// ```
/// use scivex_stats::prophet::{Prophet, ProphetConfig};
///
/// let config = ProphetConfig::<f64>::new();
/// let mut model = Prophet::new(config);
///
/// // Simple linear trend: y = 2*t + 1
/// let t: Vec<f64> = (0..100).map(|i| i as f64).collect();
/// let y: Vec<f64> = t.iter().map(|&ti| 2.0 * ti + 1.0).collect();
///
/// model.fit(&t, &y).unwrap();
/// let forecast = model.predict(&[100.0, 101.0, 102.0]).unwrap();
/// // Predictions should be close to the true trend
/// assert!((forecast.yhat[0] - 201.0).abs() < 10.0);
/// ```
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct Prophet<T: Float> {
    config: ProphetConfig<T>,
    // Trend parameters
    k: T,                 // base growth rate
    m: T,                 // offset
    deltas: Vec<T>,       // changepoint rate adjustments
    changepoints: Vec<T>, // changepoint times (normalized)
    // Seasonality parameters
    seasonality_coeffs: Vec<T>, // Fourier coefficients
    // Scale factors for reversing normalization
    y_scale: T,
    y_offset: T,
    t_min: T,
    t_scale: T,
    // Seasonality layout
    n_yearly_terms: usize,
    n_weekly_terms: usize,
    fitted: bool,
}

/// Result of Prophet forecasting.
///
/// Contains the point predictions and individual components.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct ProphetForecast<T: Float> {
    /// Point predictions.
    pub yhat: Vec<T>,
    /// Trend component at each time point.
    pub trend: Vec<T>,
    /// Seasonality component at each time point.
    pub seasonality: Vec<T>,
}

impl<T: Float> Prophet<T> {
    /// Create a new (unfitted) Prophet model with the given configuration.
    pub fn new(config: ProphetConfig<T>) -> Self {
        Self {
            config,
            k: T::zero(),
            m: T::zero(),
            deltas: Vec::new(),
            changepoints: Vec::new(),
            seasonality_coeffs: Vec::new(),
            y_scale: T::one(),
            y_offset: T::zero(),
            t_min: T::zero(),
            t_scale: T::one(),
            n_yearly_terms: 0,
            n_weekly_terms: 0,
            fitted: false,
        }
    }

    /// Fit the model to training data.
    ///
    /// # Arguments
    ///
    /// - `t` — time points (must be sorted ascending, at least 2 values)
    /// - `y` — observed values corresponding to each time point
    ///
    /// # Errors
    ///
    /// Returns [`StatsError::EmptyInput`] if inputs are empty,
    /// [`StatsError::LengthMismatch`] if `t` and `y` differ in length,
    /// or [`StatsError::InsufficientData`] if fewer than 2 observations.
    #[allow(clippy::too_many_lines)]
    pub fn fit(&mut self, t: &[T], y: &[T]) -> Result<()> {
        let n = t.len();
        if n == 0 {
            return Err(StatsError::EmptyInput);
        }
        if t.len() != y.len() {
            return Err(StatsError::LengthMismatch {
                expected: t.len(),
                got: y.len(),
            });
        }
        if n < 2 {
            return Err(StatsError::InsufficientData { need: 2, got: n });
        }

        // ── Normalize time to [0, 1] ──────────────────────────────────
        let t_min = t[0];
        let t_max = t[n - 1];
        let t_scale = t_max - t_min;
        if t_scale.abs() < T::from_f64(1e-15) {
            return Err(StatsError::InvalidParameter {
                name: "t",
                reason: "all time values are identical",
            });
        }
        self.t_min = t_min;
        self.t_scale = t_scale;

        let t_norm: Vec<T> = t.iter().map(|&ti| (ti - t_min) / t_scale).collect();

        // ── Normalize y to zero mean ──────────────────────────────────
        let y_offset = y.iter().copied().fold(T::zero(), |a, b| a + b) / T::from_usize(n);
        let y_centered: Vec<T> = y.iter().map(|&yi| yi - y_offset).collect();
        let y_abs_max = y_centered
            .iter()
            .map(|v| v.abs())
            .fold(T::from_f64(1e-15), scivex_core::Float::max);
        let y_scale = y_abs_max;
        let y_norm: Vec<T> = y_centered.iter().map(|&yi| yi / y_scale).collect();
        self.y_offset = y_offset;
        self.y_scale = y_scale;

        // ── Set up changepoints ───────────────────────────────────────
        let n_cp = self.config.n_changepoints.min(n.saturating_sub(1));
        let cp_range = self.config.changepoint_range;
        let mut changepoints = Vec::with_capacity(n_cp);
        if n_cp > 0 {
            let cp_end = cp_range;
            for i in 1..=n_cp {
                let frac = T::from_usize(i) / T::from_usize(n_cp + 1);
                changepoints.push(frac * cp_end);
            }
        }
        self.changepoints = changepoints;

        // ── Build design matrix columns ───────────────────────────────
        // Trend columns: [t, A_1, A_2, ..., A_cp, 1 (intercept)]
        // Where A_j(t) = max(0, t - s_j), with s_j the changepoint time.
        // We treat: g(t) = k*t + m + sum_j delta_j * max(0, t - s_j)
        // Plus continuity adjustment is implicitly handled.
        let n_trend_cols = 2 + self.changepoints.len(); // t, changepoints, intercept

        // Seasonality columns
        let n_yearly = if self.config.yearly_seasonality {
            2 * self.config.yearly_fourier_order
        } else {
            0
        };
        let n_weekly = if self.config.weekly_seasonality {
            2 * self.config.weekly_fourier_order
        } else {
            0
        };
        self.n_yearly_terms = n_yearly;
        self.n_weekly_terms = n_weekly;
        let n_season_cols = n_yearly + n_weekly;
        let n_cols = n_trend_cols + n_season_cols;

        // Build the full design matrix in row-major order
        let mut x_data = vec![T::zero(); n * n_cols];

        for i in 0..n {
            let row_off = i * n_cols;
            let ti = t_norm[i];

            // Column 0: time
            x_data[row_off] = ti;

            // Columns 1..=n_cp: changepoint indicators (ramp)
            for (j, &sj) in self.changepoints.iter().enumerate() {
                if ti > sj {
                    x_data[row_off + 1 + j] = ti - sj;
                }
            }

            // Intercept column
            x_data[row_off + n_trend_cols - 1] = T::one();

            // Seasonality features (use original time scale for period matching)
            let t_orig = t[i];
            let mut col = n_trend_cols;
            if self.config.yearly_seasonality {
                let period = T::from_f64(365.25);
                for order in 1..=self.config.yearly_fourier_order {
                    let arg = T::from_f64(2.0) * T::pi() * T::from_usize(order) * t_orig / period;
                    x_data[row_off + col] = arg.cos();
                    x_data[row_off + col + 1] = arg.sin();
                    col += 2;
                }
            }
            if self.config.weekly_seasonality {
                let period = T::from_f64(7.0);
                for order in 1..=self.config.weekly_fourier_order {
                    let arg = T::from_f64(2.0) * T::pi() * T::from_usize(order) * t_orig / period;
                    x_data[row_off + col] = arg.cos();
                    x_data[row_off + col + 1] = arg.sin();
                    col += 2;
                }
            }
        }

        // ── Build regularization vector (per-column lambda) ───────────
        let mut lambda = vec![T::zero(); n_cols];
        // Changepoint columns get strong regularization
        let cp_lambda =
            T::one() / (self.config.changepoint_prior_scale * self.config.changepoint_prior_scale);
        for lam in lambda.iter_mut().skip(1).take(self.changepoints.len()) {
            *lam = cp_lambda;
        }
        // Seasonality columns get regularization
        let season_lambda =
            T::one() / (self.config.seasonality_prior_scale * self.config.seasonality_prior_scale);
        for lam in lambda
            .iter_mut()
            .skip(n_trend_cols)
            .take(n_cols - n_trend_cols)
        {
            *lam = season_lambda;
        }

        // ── Solve ridge regression: (X'X + diag(lambda)) beta = X'y ──
        let beta = solve_ridge_with_lambda(&x_data, n, n_cols, &y_norm, &lambda)?;

        // ── Extract parameters ────────────────────────────────────────
        self.k = beta[0];
        self.m = beta[n_trend_cols - 1];
        self.deltas = beta[1..=self.changepoints.len()].to_vec();
        self.seasonality_coeffs = beta[n_trend_cols..].to_vec();
        self.fitted = true;

        Ok(())
    }

    /// Generate forecasts for the given time points.
    ///
    /// # Errors
    ///
    /// Returns [`StatsError::InvalidParameter`] if the model has not been fitted.
    pub fn predict(&self, t_future: &[T]) -> Result<ProphetForecast<T>> {
        if !self.fitted {
            return Err(StatsError::InvalidParameter {
                name: "model",
                reason: "model has not been fitted; call fit() first",
            });
        }

        let n = t_future.len();
        let mut trend = Vec::with_capacity(n);
        let mut seasonality = Vec::with_capacity(n);
        let mut yhat = Vec::with_capacity(n);

        for &ti in t_future {
            // Normalized time
            let t_n = (ti - self.t_min) / self.t_scale;

            // Trend: k*t + sum_j delta_j * max(0, t - s_j) + m
            let mut g = self.k * t_n + self.m;
            for (j, &sj) in self.changepoints.iter().enumerate() {
                if t_n > sj {
                    g += self.deltas[j] * (t_n - sj);
                }
            }

            // Seasonality using original time scale
            let mut s = T::zero();
            let mut idx = 0;
            if self.config.yearly_seasonality {
                let period = T::from_f64(365.25);
                for order in 1..=self.config.yearly_fourier_order {
                    let arg = T::from_f64(2.0) * T::pi() * T::from_usize(order) * ti / period;
                    s += self.seasonality_coeffs[idx] * arg.cos();
                    s += self.seasonality_coeffs[idx + 1] * arg.sin();
                    idx += 2;
                }
            }
            if self.config.weekly_seasonality {
                let period = T::from_f64(7.0);
                for order in 1..=self.config.weekly_fourier_order {
                    let arg = T::from_f64(2.0) * T::pi() * T::from_usize(order) * ti / period;
                    s += self.seasonality_coeffs[idx] * arg.cos();
                    s += self.seasonality_coeffs[idx + 1] * arg.sin();
                    idx += 2;
                }
            }

            // Rescale back
            let trend_val = g * self.y_scale + self.y_offset;
            let season_val = s * self.y_scale;

            trend.push(trend_val);
            seasonality.push(season_val);
            yhat.push(trend_val + season_val);
        }

        Ok(ProphetForecast {
            yhat,
            trend,
            seasonality,
        })
    }
}

// ── Ridge regression solver ───────────────────────────────────────────

/// Solve `(X'X + diag(lambda)) beta = X'y` via Cholesky-like decomposition.
///
/// `x_data` is a row-major `n x p` matrix, `y` has length `n`,
/// `lambda` has length `p`.
fn solve_ridge_with_lambda<T: Float>(
    x_data: &[T],
    n: usize,
    p: usize,
    y: &[T],
    lambda: &[T],
) -> Result<Vec<T>> {
    // Compute X'X (p x p, symmetric)
    let mut xtx = vec![T::zero(); p * p];
    for i in 0..n {
        let row = &x_data[i * p..(i + 1) * p];
        for j in 0..p {
            for k in j..p {
                let val = row[j] * row[k];
                xtx[j * p + k] += val;
                if k != j {
                    xtx[k * p + j] += val;
                }
            }
        }
    }

    // Add regularization to diagonal
    for j in 0..p {
        xtx[j * p + j] += lambda[j];
    }

    // Compute X'y (p x 1)
    let mut xty = vec![T::zero(); p];
    for i in 0..n {
        let row = &x_data[i * p..(i + 1) * p];
        for j in 0..p {
            xty[j] += row[j] * y[i];
        }
    }

    // Solve via Cholesky decomposition: xtx = L L'
    // Then L z = xty, L' beta = z
    let l = cholesky_decompose(&xtx, p)?;
    let z = forward_substitute(&l, p, &xty);
    let beta = backward_substitute_transpose(&l, p, &z);

    Ok(beta)
}

/// Cholesky decomposition of a symmetric positive-definite matrix.
/// Returns the lower-triangular factor L such that A = L L^T.
fn cholesky_decompose<T: Float>(a: &[T], n: usize) -> Result<Vec<T>> {
    let mut l = vec![T::zero(); n * n];

    for j in 0..n {
        // Diagonal element
        let mut sum = a[j * n + j];
        for k in 0..j {
            sum -= l[j * n + k] * l[j * n + k];
        }
        if sum <= T::zero() {
            // Matrix is not positive definite; add a small nudge and retry
            // This can happen with ill-conditioned design matrices
            let nudge = T::from_f64(1e-10);
            let sum_nudged = sum + nudge;
            if sum_nudged <= T::zero() {
                return Err(StatsError::SingularMatrix);
            }
            l[j * n + j] = sum_nudged.sqrt();
        } else {
            l[j * n + j] = sum.sqrt();
        }

        let diag = l[j * n + j];
        // Off-diagonal elements
        for i in (j + 1)..n {
            let mut sum = a[i * n + j];
            for k in 0..j {
                sum -= l[i * n + k] * l[j * n + k];
            }
            l[i * n + j] = sum / diag;
        }
    }

    Ok(l)
}

/// Forward substitution: solve L z = b where L is lower-triangular.
fn forward_substitute<T: Float>(l: &[T], n: usize, b: &[T]) -> Vec<T> {
    let mut z = vec![T::zero(); n];
    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i {
            sum -= l[i * n + j] * z[j];
        }
        z[i] = sum / l[i * n + i];
    }
    z
}

/// Backward substitution: solve L^T beta = z where L is lower-triangular.
fn backward_substitute_transpose<T: Float>(l: &[T], n: usize, z: &[T]) -> Vec<T> {
    let mut beta = vec![T::zero(); n];
    for i in (0..n).rev() {
        let mut sum = z[i];
        for j in (i + 1)..n {
            sum -= l[j * n + i] * beta[j];
        }
        beta[i] = sum / l[i * n + i];
    }
    beta
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prophet_linear_trend() {
        // Pure linear trend: y = 3*t + 5
        let mut config = ProphetConfig::<f64>::new();
        config.yearly_seasonality = false;
        config.weekly_seasonality = false;

        let mut model = Prophet::new(config);
        let t: Vec<f64> = (0..200).map(f64::from).collect();
        let y: Vec<f64> = t.iter().map(|&ti| 3.0 * ti + 5.0).collect();

        model.fit(&t, &y).unwrap();
        let forecast = model.predict(&[200.0, 210.0, 220.0]).unwrap();

        // Check that trend is close to true values
        assert!(
            (forecast.yhat[0] - 605.0).abs() < 15.0,
            "Expected ~605, got {}",
            forecast.yhat[0]
        );
        assert!(
            (forecast.yhat[1] - 635.0).abs() < 15.0,
            "Expected ~635, got {}",
            forecast.yhat[1]
        );
        assert!(
            (forecast.yhat[2] - 665.0).abs() < 15.0,
            "Expected ~665, got {}",
            forecast.yhat[2]
        );

        // All seasonality should be ~zero
        for &s in &forecast.seasonality {
            assert!(s.abs() < 1.0, "Seasonality should be ~0, got {s}");
        }
    }

    #[test]
    fn test_prophet_seasonal() {
        // Data with known sinusoidal pattern on weekly period
        let mut config = ProphetConfig::<f64>::new();
        config.yearly_seasonality = false;
        config.weekly_seasonality = true;
        config.weekly_fourier_order = 3;
        config.n_changepoints = 0;
        config.seasonality_prior_scale = 100.0; // weak regularization

        let mut model = Prophet::new(config);
        let n = 200;
        let t: Vec<f64> = (0..n).map(f64::from).collect();
        let y: Vec<f64> = t
            .iter()
            .map(|&ti| 10.0 * (2.0 * std::f64::consts::PI * ti / 7.0).sin() + 50.0)
            .collect();

        model.fit(&t, &y).unwrap();

        // Check that seasonality is captured by predicting within known data
        let forecast = model.predict(&t[0..14]).unwrap();
        for (i, (ti, yhat_i)) in t[0..14].iter().zip(forecast.yhat.iter()).enumerate() {
            let expected = 10.0 * (2.0 * std::f64::consts::PI * ti / 7.0).sin() + 50.0;
            assert!(
                (yhat_i - expected).abs() < 5.0,
                "At t={ti}, expected ~{expected:.1}, got {yhat_i:.1} (i={i})",
            );
        }
    }

    #[test]
    fn test_prophet_forecast() {
        // Fit on linear data and predict future
        let mut config = ProphetConfig::<f64>::new();
        config.yearly_seasonality = false;
        config.weekly_seasonality = false;

        let mut model = Prophet::new(config);
        let t: Vec<f64> = (0..100).map(f64::from).collect();
        let y: Vec<f64> = t.iter().map(|&ti| 2.0 * ti + 10.0).collect();

        model.fit(&t, &y).unwrap();

        // Future predictions should continue the trend
        let future_t: Vec<f64> = (100..110).map(f64::from).collect();
        let forecast = model.predict(&future_t).unwrap();

        assert_eq!(forecast.yhat.len(), 10);
        assert_eq!(forecast.trend.len(), 10);
        assert_eq!(forecast.seasonality.len(), 10);

        // Should be monotonically increasing
        for i in 1..forecast.yhat.len() {
            assert!(
                forecast.yhat[i] > forecast.yhat[i - 1],
                "Predictions should be increasing"
            );
        }

        // First future prediction should be near 210
        assert!(
            (forecast.yhat[0] - 210.0).abs() < 20.0,
            "Expected ~210, got {}",
            forecast.yhat[0]
        );
    }

    #[test]
    fn test_prophet_changepoint() {
        // Data with a trend change at t=50
        let mut config = ProphetConfig::<f64>::new();
        config.yearly_seasonality = false;
        config.weekly_seasonality = false;
        config.n_changepoints = 25;
        config.changepoint_prior_scale = 0.5; // allow changepoints

        let mut model = Prophet::new(config);
        let t: Vec<f64> = (0..100).map(f64::from).collect();
        let y: Vec<f64> = t
            .iter()
            .map(|&ti| {
                if ti < 50.0 {
                    1.0 * ti
                } else {
                    50.0 + 5.0 * (ti - 50.0)
                }
            })
            .collect();

        model.fit(&t, &y).unwrap();
        let forecast = model.predict(&[90.0, 95.0, 100.0]).unwrap();

        // The model should capture the steeper trend after the changepoint
        // y(90) ≈ 50 + 5*40 = 250
        // y(95) ≈ 50 + 5*45 = 275
        assert!(
            forecast.yhat[1] > forecast.yhat[0],
            "Should have increasing trend"
        );
        assert!(
            (forecast.yhat[0] - 250.0).abs() < 40.0,
            "Expected ~250, got {}",
            forecast.yhat[0]
        );
    }

    #[test]
    fn test_prophet_empty_input() {
        let config = ProphetConfig::<f64>::new();
        let mut model = Prophet::new(config);

        let result = model.fit(&[], &[]);
        assert!(result.is_err());
        match result.unwrap_err() {
            StatsError::EmptyInput => {}
            other => panic!("Expected EmptyInput, got {other:?}"),
        }
    }
}
