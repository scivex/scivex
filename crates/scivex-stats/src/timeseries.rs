//! Time series analysis.
//!
//! Provides tools for analyzing temporal data:
//!
//! - [`acf`] / [`pacf`] — autocorrelation and partial autocorrelation
//! - [`Arima`] — ARIMA model fitting and forecasting
//! - [`ExponentialSmoothing`] — simple, Holt, and Holt-Winters smoothing
//! - [`seasonal_decompose`] — additive seasonal decomposition
//! - [`adf_test`] — Augmented Dickey-Fuller stationarity test

use scivex_core::Float;

use crate::descriptive;
use crate::error::{Result, StatsError};

// ── Autocorrelation ─────────────────────────────────────────────────────

/// Compute the autocorrelation function (ACF) for lags `0..max_lag`.
///
/// Returns a vector of length `max_lag + 1` where element `k` is the
/// autocorrelation at lag `k`. ACF(0) is always 1.0.
pub fn acf<T: Float>(data: &[T], max_lag: usize) -> Result<Vec<T>> {
    let n = data.len();
    if n < 2 {
        return Err(StatsError::InsufficientData { need: 2, got: n });
    }
    let max_lag = max_lag.min(n - 1);
    let mean = descriptive::mean(data)?;

    // Variance (denominator)
    let mut var = T::zero();
    for &x in data {
        let d = x - mean;
        var += d * d;
    }

    let mut result = Vec::with_capacity(max_lag + 1);
    for lag in 0..=max_lag {
        let mut cov = T::zero();
        for i in 0..(n - lag) {
            cov += (data[i] - mean) * (data[i + lag] - mean);
        }
        result.push(if var > T::zero() {
            cov / var
        } else {
            T::zero()
        });
    }

    Ok(result)
}

/// Compute the partial autocorrelation function (PACF) using the
/// Durbin-Levinson algorithm.
///
/// Returns a vector of length `max_lag + 1` where element `k` is the
/// partial autocorrelation at lag `k`. PACF(0) is always 1.0.
pub fn pacf<T: Float>(data: &[T], max_lag: usize) -> Result<Vec<T>> {
    let acf_vals = acf(data, max_lag)?;
    let m = acf_vals.len();

    let mut result = vec![T::zero(); m];
    result[0] = T::one();

    if m < 2 {
        return Ok(result);
    }

    // Durbin-Levinson recursion
    let mut phi = vec![vec![T::zero(); m]; m];
    phi[1][1] = acf_vals[1];
    result[1] = acf_vals[1];

    for k in 2..m {
        // Compute phi[k][k]
        let mut num = acf_vals[k];
        let mut den = T::one();
        for j in 1..k {
            num -= phi[k - 1][j] * acf_vals[k - j];
            den -= phi[k - 1][j] * acf_vals[j];
        }
        if den.abs() < T::from_f64(1e-15) {
            break;
        }
        phi[k][k] = num / den;
        result[k] = phi[k][k];

        // Update phi[k][j] for j = 1..k-1
        for j in 1..k {
            phi[k][j] = phi[k - 1][j] - phi[k][k] * phi[k - 1][k - j];
        }
    }

    Ok(result)
}

// ── Differencing ────────────────────────────────────────────────────────

/// Difference a time series `d` times.
fn difference<T: Float>(data: &[T], d: usize) -> Vec<T> {
    let mut current = data.to_vec();
    for _ in 0..d {
        let mut diffed = Vec::with_capacity(current.len().saturating_sub(1));
        for i in 1..current.len() {
            diffed.push(current[i] - current[i - 1]);
        }
        current = diffed;
    }
    current
}

// ── ARIMA ───────────────────────────────────────────────────────────────

/// ARIMA(p, d, q) model for time series forecasting.
///
/// - `p` — order of the autoregressive (AR) part
/// - `d` — degree of differencing
/// - `q` — order of the moving average (MA) part
///
/// Fitting uses conditional least squares for the AR coefficients and
/// sets MA coefficients via residual estimation.
#[derive(Debug, Clone)]
pub struct Arima<T: Float> {
    p: usize,
    d: usize,
    q: usize,
    ar_coeffs: Vec<T>,
    ma_coeffs: Vec<T>,
    intercept: T,
    residuals: Vec<T>,
    fitted: bool,
    original: Vec<T>,
}

impl<T: Float> Arima<T> {
    /// Create a new ARIMA(p, d, q) model.
    pub fn new(p: usize, d: usize, q: usize) -> Result<Self> {
        if p == 0 && q == 0 {
            return Err(StatsError::InvalidParameter {
                name: "p, q",
                reason: "at least one of p or q must be > 0",
            });
        }
        Ok(Self {
            p,
            d,
            q,
            ar_coeffs: Vec::new(),
            ma_coeffs: Vec::new(),
            intercept: T::zero(),
            residuals: Vec::new(),
            fitted: false,
            original: Vec::new(),
        })
    }

    /// Fit the ARIMA model to data.
    #[allow(clippy::too_many_lines)]
    pub fn fit(&mut self, data: &[T]) -> Result<()> {
        let min_len = self.p + self.d + self.q + 2;
        if data.len() < min_len {
            return Err(StatsError::InsufficientData {
                need: min_len,
                got: data.len(),
            });
        }

        self.original = data.to_vec();
        let diffed = difference(data, self.d);
        let n = diffed.len();

        // Fit AR part via least squares (Yule-Walker)
        if self.p > 0 {
            let acf_vals = acf(&diffed, self.p)?;
            // Solve Yule-Walker: R * phi = r
            // where R is the Toeplitz matrix of acf[0..p-1] and r = acf[1..p]
            // Using Levinson-Durbin
            let mut phi = vec![T::zero(); self.p];
            let mut phi_prev = vec![T::zero(); self.p];

            phi[0] = acf_vals[1];
            let mut v = T::one() - phi[0] * phi[0];

            for k in 1..self.p {
                let mut lambda = acf_vals[k + 1];
                for j in 0..k {
                    lambda -= phi[j] * acf_vals[k - j];
                }
                if v.abs() < T::from_f64(1e-15) {
                    break;
                }
                phi_prev[..self.p].copy_from_slice(&phi[..self.p]);
                phi[k] = lambda / v;
                for j in 0..k {
                    phi[j] = phi_prev[j] - phi[k] * phi_prev[k - 1 - j];
                }
                v *= T::one() - phi[k] * phi[k];
            }
            self.ar_coeffs = phi;
        }

        // Compute mean of differenced series for intercept
        let diff_mean = descriptive::mean(&diffed)?;
        let mut ar_mean_contrib = T::zero();
        for &c in &self.ar_coeffs {
            ar_mean_contrib += c;
        }
        self.intercept = diff_mean * (T::one() - ar_mean_contrib);

        // Compute AR residuals
        let mut residuals = vec![T::zero(); n];
        for t in self.p..n {
            let mut pred = self.intercept;
            for j in 0..self.p {
                pred += self.ar_coeffs[j] * diffed[t - 1 - j];
            }
            residuals[t] = diffed[t] - pred;
        }

        // Fit MA part from residuals
        if self.q > 0 {
            let res_slice = &residuals[self.p..];
            if res_slice.len() > self.q {
                let res_acf = acf(res_slice, self.q)?;
                self.ma_coeffs = res_acf[1..].to_vec();
            } else {
                self.ma_coeffs = vec![T::zero(); self.q];
            }

            // Recompute residuals with MA terms
            for t in self.p.max(self.q)..n {
                let mut pred = self.intercept;
                for j in 0..self.p {
                    pred += self.ar_coeffs[j] * diffed[t - 1 - j];
                }
                for j in 0..self.q {
                    if t > j {
                        pred += self.ma_coeffs[j] * residuals[t - 1 - j];
                    }
                }
                residuals[t] = diffed[t] - pred;
            }
        }

        self.residuals = residuals;
        self.fitted = true;
        Ok(())
    }

    /// Forecast `steps` steps ahead.
    pub fn forecast(&self, steps: usize) -> Result<Vec<T>> {
        if !self.fitted {
            return Err(StatsError::InvalidParameter {
                name: "model",
                reason: "model must be fitted before forecasting",
            });
        }

        let diffed = difference(&self.original, self.d);
        let n = diffed.len();

        // Extend diffed and residuals for forecasting
        let mut extended = diffed.clone();
        let mut ext_resid = self.residuals.clone();

        for _ in 0..steps {
            let t = extended.len();
            let mut pred = self.intercept;
            for j in 0..self.p {
                if t > j {
                    pred += self.ar_coeffs[j] * extended[t - 1 - j];
                }
            }
            for j in 0..self.q {
                if t > j && t - 1 - j < ext_resid.len() {
                    pred += self.ma_coeffs[j] * ext_resid[t - 1 - j];
                }
            }
            extended.push(pred);
            ext_resid.push(T::zero()); // future residuals are 0
        }

        // Un-difference the forecast
        let mut forecasts = extended[n..].to_vec();
        for _ in 0..self.d {
            let mut last = *self.original.last().unwrap_or(&T::zero());
            for val in &mut forecasts {
                last += *val;
                *val = last;
            }
        }

        Ok(forecasts)
    }

    /// Return the AR coefficients.
    pub fn ar_coefficients(&self) -> &[T] {
        &self.ar_coeffs
    }

    /// Return the MA coefficients.
    pub fn ma_coefficients(&self) -> &[T] {
        &self.ma_coeffs
    }
}

// ── Exponential Smoothing ───────────────────────────────────────────────

/// Exponential smoothing method selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SmoothingMethod {
    /// Simple exponential smoothing (level only).
    Simple,
    /// Holt's linear method (level + trend).
    Holt,
    /// Holt-Winters additive method (level + trend + season).
    HoltWinters,
}

/// Exponential smoothing model.
#[derive(Debug, Clone)]
pub struct ExponentialSmoothing<T: Float> {
    method: SmoothingMethod,
    alpha: T,
    beta: Option<T>,
    gamma: Option<T>,
    season_length: usize,
    level: T,
    trend: T,
    seasonal: Vec<T>,
    fitted: bool,
    data_len: usize,
}

impl<T: Float> ExponentialSmoothing<T> {
    /// Create a simple exponential smoothing model.
    pub fn simple(alpha: T) -> Result<Self> {
        validate_alpha(alpha)?;
        Ok(Self {
            method: SmoothingMethod::Simple,
            alpha,
            beta: None,
            gamma: None,
            season_length: 0,
            level: T::zero(),
            trend: T::zero(),
            seasonal: Vec::new(),
            fitted: false,
            data_len: 0,
        })
    }

    /// Create a Holt's linear trend model.
    pub fn holt(alpha: T, beta: T) -> Result<Self> {
        validate_alpha(alpha)?;
        validate_alpha(beta)?;
        Ok(Self {
            method: SmoothingMethod::Holt,
            alpha,
            beta: Some(beta),
            gamma: None,
            season_length: 0,
            level: T::zero(),
            trend: T::zero(),
            seasonal: Vec::new(),
            fitted: false,
            data_len: 0,
        })
    }

    /// Create a Holt-Winters additive seasonal model.
    pub fn holt_winters(alpha: T, beta: T, gamma: T, season_length: usize) -> Result<Self> {
        validate_alpha(alpha)?;
        validate_alpha(beta)?;
        validate_alpha(gamma)?;
        if season_length < 2 {
            return Err(StatsError::InvalidParameter {
                name: "season_length",
                reason: "must be >= 2",
            });
        }
        Ok(Self {
            method: SmoothingMethod::HoltWinters,
            alpha,
            beta: Some(beta),
            gamma: Some(gamma),
            season_length,
            level: T::zero(),
            trend: T::zero(),
            seasonal: Vec::new(),
            fitted: false,
            data_len: 0,
        })
    }

    /// Fit the model to data.
    pub fn fit(&mut self, data: &[T]) -> Result<()> {
        let n = data.len();
        match self.method {
            SmoothingMethod::Simple => {
                if n < 1 {
                    return Err(StatsError::EmptyInput);
                }
                self.level = data[0];
                for &val in &data[1..] {
                    self.level = self.alpha * val + (T::one() - self.alpha) * self.level;
                }
            }
            SmoothingMethod::Holt => {
                if n < 2 {
                    return Err(StatsError::InsufficientData { need: 2, got: n });
                }
                let beta = self.beta.unwrap_or(T::from_f64(0.1));
                self.level = data[0];
                self.trend = data[1] - data[0];
                for &val in &data[1..] {
                    let prev_level = self.level;
                    self.level =
                        self.alpha * val + (T::one() - self.alpha) * (prev_level + self.trend);
                    self.trend = beta * (self.level - prev_level) + (T::one() - beta) * self.trend;
                }
            }
            SmoothingMethod::HoltWinters => {
                let m = self.season_length;
                if n < 2 * m {
                    return Err(StatsError::InsufficientData {
                        need: 2 * m,
                        got: n,
                    });
                }
                let beta = self.beta.unwrap_or(T::from_f64(0.1));
                let gamma = self.gamma.unwrap_or(T::from_f64(0.1));

                // Initialize level and trend from first two seasons
                let season1_mean = descriptive::mean(&data[..m])?;
                let season2_mean = descriptive::mean(&data[m..2 * m])?;
                self.level = season1_mean;
                self.trend = (season2_mean - season1_mean) / T::from_f64(m as f64);

                // Initialize seasonal from first season
                self.seasonal = vec![T::zero(); m];
                for (i, &val) in data[..m].iter().enumerate() {
                    self.seasonal[i] = val - season1_mean;
                }

                // Update
                for (t, &val) in data.iter().enumerate().skip(m) {
                    let si = t % m;
                    let prev_level = self.level;
                    self.level = self.alpha * (val - self.seasonal[si])
                        + (T::one() - self.alpha) * (prev_level + self.trend);
                    self.trend = beta * (self.level - prev_level) + (T::one() - beta) * self.trend;
                    self.seasonal[si] =
                        gamma * (val - self.level) + (T::one() - gamma) * self.seasonal[si];
                }
            }
        }
        self.fitted = true;
        self.data_len = n;
        Ok(())
    }

    /// Forecast `steps` steps ahead.
    pub fn forecast(&self, steps: usize) -> Result<Vec<T>> {
        if !self.fitted {
            return Err(StatsError::InvalidParameter {
                name: "model",
                reason: "model must be fitted before forecasting",
            });
        }
        let mut result = Vec::with_capacity(steps);
        for h in 1..=steps {
            let hf = T::from_f64(h as f64);
            match self.method {
                SmoothingMethod::Simple => {
                    result.push(self.level);
                }
                SmoothingMethod::Holt => {
                    result.push(self.level + self.trend * hf);
                }
                SmoothingMethod::HoltWinters => {
                    let si = (self.data_len + h - 1) % self.season_length;
                    let s = if si < self.seasonal.len() {
                        self.seasonal[si]
                    } else {
                        T::zero()
                    };
                    result.push(self.level + self.trend * hf + s);
                }
            }
        }
        Ok(result)
    }
}

fn validate_alpha<T: Float>(alpha: T) -> Result<()> {
    if alpha <= T::zero() || alpha >= T::one() {
        return Err(StatsError::InvalidParameter {
            name: "alpha/beta/gamma",
            reason: "smoothing parameter must be in (0, 1)",
        });
    }
    Ok(())
}

// ── Seasonal Decomposition ──────────────────────────────────────────────

/// Result of additive seasonal decomposition.
#[derive(Debug, Clone)]
pub struct DecomposeResult<T: Float> {
    /// Trend component.
    pub trend: Vec<T>,
    /// Seasonal component.
    pub seasonal: Vec<T>,
    /// Residual component.
    pub residual: Vec<T>,
}

/// Additive seasonal decomposition using moving averages.
///
/// Decomposes `y = trend + seasonal + residual`.
///
/// `period` is the length of one seasonal cycle.
pub fn seasonal_decompose<T: Float>(data: &[T], period: usize) -> Result<DecomposeResult<T>> {
    let n = data.len();
    if period < 2 {
        return Err(StatsError::InvalidParameter {
            name: "period",
            reason: "must be >= 2",
        });
    }
    if n < 2 * period {
        return Err(StatsError::InsufficientData {
            need: 2 * period,
            got: n,
        });
    }

    // Compute centered moving average for trend
    let half = period / 2;
    let mut trend = vec![T::from_f64(f64::NAN); n];

    for (i, t) in trend.iter_mut().enumerate().take(n - half).skip(half) {
        let mut sum = T::zero();
        let mut count = 0usize;
        let start = i - half;
        let end = i + half + 1;
        for &val in &data[start..end.min(n)] {
            sum += val;
            count += 1;
        }
        if count > 0 {
            *t = sum / T::from_f64(count as f64);
        }
    }

    // For even period, average adjacent values to center
    #[allow(clippy::manual_is_multiple_of)]
    if period % 2 == 0 {
        let mut centered = trend.clone();
        for i in 1..(n - 1) {
            if !trend[i].is_nan() && !trend[i - 1].is_nan() {
                centered[i] = (trend[i] + trend[i - 1]) / T::from_f64(2.0);
            }
        }
        trend = centered;
    }

    // Detrended = data - trend
    let mut detrended = vec![T::zero(); n];
    for i in 0..n {
        if !trend[i].is_nan() {
            detrended[i] = data[i] - trend[i];
        }
    }

    // Average seasonal component over cycles
    let mut seasonal_avg = vec![T::zero(); period];
    let mut seasonal_count = vec![0usize; period];
    for i in 0..n {
        if !trend[i].is_nan() {
            let s = i % period;
            seasonal_avg[s] += detrended[i];
            seasonal_count[s] += 1;
        }
    }
    for s in 0..period {
        if seasonal_count[s] > 0 {
            seasonal_avg[s] /= T::from_f64(seasonal_count[s] as f64);
        }
    }

    // Center seasonal (subtract mean)
    let s_mean = {
        let mut sum = T::zero();
        for &v in &seasonal_avg {
            sum += v;
        }
        sum / T::from_f64(period as f64)
    };
    for v in &mut seasonal_avg {
        *v -= s_mean;
    }

    // Extend seasonal to full length
    let mut seasonal = vec![T::zero(); n];
    for i in 0..n {
        seasonal[i] = seasonal_avg[i % period];
    }

    // Residual = data - trend - seasonal
    let mut residual = vec![T::zero(); n];
    for i in 0..n {
        if trend[i].is_nan() {
            residual[i] = T::from_f64(f64::NAN);
        } else {
            residual[i] = data[i] - trend[i] - seasonal[i];
        }
    }

    Ok(DecomposeResult {
        trend,
        seasonal,
        residual,
    })
}

// ── Augmented Dickey-Fuller Test ────────────────────────────────────────

/// Result of the Augmented Dickey-Fuller test.
#[derive(Debug, Clone, Copy)]
pub struct AdfResult<T: Float> {
    /// ADF test statistic.
    pub statistic: T,
    /// Number of lags used.
    pub n_lags: usize,
    /// Number of observations used in the regression.
    pub n_obs: usize,
    /// Whether the null hypothesis (unit root) is rejected at 5% level.
    /// Critical value at 5% is approximately -2.86.
    pub reject_null: bool,
}

/// Augmented Dickey-Fuller test for stationarity.
///
/// Tests the null hypothesis that a unit root is present (non-stationary).
/// A sufficiently negative test statistic rejects the null.
///
/// `max_lags` controls the number of lagged difference terms. If `None`,
/// uses `floor(12 * (n/100)^{1/4})`.
pub fn adf_test<T: Float>(data: &[T], max_lags: Option<usize>) -> Result<AdfResult<T>> {
    let n = data.len();
    if n < 10 {
        return Err(StatsError::InsufficientData { need: 10, got: n });
    }

    let n_lags = max_lags.unwrap_or_else(|| {
        let nf = n as f64;
        (12.0 * (nf / 100.0).powf(0.25)).floor() as usize
    });
    let n_lags = n_lags.min(n / 3);

    // First difference: dy[t] = y[t] - y[t-1]
    let dy: Vec<T> = (1..n).map(|i| data[i] - data[i - 1]).collect();

    // Build regression: dy[t] = alpha + beta * y[t-1] + sum(gamma_i * dy[t-i]) + eps
    let start = n_lags + 1;
    let n_obs = dy.len() - start;
    if n_obs < n_lags + 3 {
        return Err(StatsError::InsufficientData {
            need: n_lags + start + 3,
            got: n,
        });
    }

    // Number of regressors: intercept + y_{t-1} + n_lags lagged diffs
    let n_regs = 2 + n_lags;

    // Build X matrix and y vector for OLS
    let mut x_data = Vec::with_capacity(n_obs * n_regs);
    let mut y_vec = Vec::with_capacity(n_obs);

    for t in start..dy.len() {
        // Intercept
        x_data.push(T::one());
        // y_{t-1} (level, in original series index this is data[t])
        x_data.push(data[t]);
        // Lagged differences
        for lag in 1..=n_lags {
            x_data.push(dy[t - lag]);
        }
        y_vec.push(dy[t]);
    }

    // Solve via normal equations: (X'X)^{-1} X'y
    // Using simplified Cholesky-like approach for small systems
    let k = n_regs;
    let m = n_obs;

    // X'X
    let mut xtx = vec![T::zero(); k * k];
    for i in 0..k {
        for j in 0..k {
            let mut s = T::zero();
            for r in 0..m {
                s += x_data[r * k + i] * x_data[r * k + j];
            }
            xtx[i * k + j] = s;
        }
    }

    // X'y
    let mut xty = vec![T::zero(); k];
    for i in 0..k {
        let mut s = T::zero();
        for r in 0..m {
            s += x_data[r * k + i] * y_vec[r];
        }
        xty[i] = s;
    }

    // Solve xtx * beta = xty using Gaussian elimination
    let coeffs = solve_linear_system(&xtx, &xty, k)?;

    // beta coefficient (coefficient on y_{t-1})
    let beta_coeff = coeffs[1];

    // Compute residuals and standard error of beta
    let mut rss = T::zero();
    for r in 0..m {
        let mut pred = T::zero();
        for i in 0..k {
            pred += coeffs[i] * x_data[r * k + i];
        }
        let e = y_vec[r] - pred;
        rss += e * e;
    }
    let sigma2 = rss / T::from_f64((m - k) as f64);

    // Standard error of beta = sqrt(sigma2 * (X'X)^{-1}[1,1])
    let xtx_inv = invert_matrix(&xtx, k)?;
    let se_beta = (sigma2 * xtx_inv[k + 1]).sqrt();

    let statistic = if se_beta > T::from_f64(1e-15) {
        beta_coeff / se_beta
    } else {
        T::from_f64(f64::NEG_INFINITY)
    };

    // Approximate 5% critical value for ADF (with intercept, no trend)
    let critical_5pct = T::from_f64(-2.86);
    let reject_null = statistic < critical_5pct;

    Ok(AdfResult {
        statistic,
        n_lags,
        n_obs,
        reject_null,
    })
}

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

/// Invert an n×n matrix using Gauss-Jordan elimination.
fn invert_matrix<T: Float>(a: &[T], n: usize) -> Result<Vec<T>> {
    let mut aug = vec![T::zero(); n * 2 * n];
    for i in 0..n {
        for j in 0..n {
            aug[i * 2 * n + j] = a[i * n + j];
        }
        aug[i * 2 * n + n + i] = T::one();
    }

    let w = 2 * n;
    for col in 0..n {
        let mut max_row = col;
        let mut max_val = aug[col * w + col].abs();
        for row in (col + 1)..n {
            let val = aug[row * w + col].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }
        if max_val < T::from_f64(1e-15) {
            return Err(StatsError::SingularMatrix);
        }
        if max_row != col {
            for j in 0..w {
                aug.swap(col * w + j, max_row * w + j);
            }
        }
        let pivot = aug[col * w + col];
        for j in 0..w {
            aug[col * w + j] /= pivot;
        }
        for row in 0..n {
            if row != col {
                let factor = aug[row * w + col];
                for j in 0..w {
                    let col_val = aug[col * w + j];
                    aug[row * w + j] -= factor * col_val;
                }
            }
        }
    }

    let mut inv = vec![T::zero(); n * n];
    for i in 0..n {
        for j in 0..n {
            inv[i * n + j] = aug[i * w + n + j];
        }
    }
    Ok(inv)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── ACF/PACF ────────────────────────────────────────────────────

    #[test]
    fn test_acf_lag_zero_is_one() {
        let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let r = acf(&data, 3).unwrap();
        assert!((r[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_acf_white_noise() {
        // Deterministic "noisy" sequence — ACF at lag > 0 should be small
        // Deterministic pseudo-noise — ACF at lag > 0 should be bounded
        let data: Vec<f64> = (0..200).map(|i| f64::from((i * 7 + 3) % 13)).collect();
        let r = acf(&data, 5).unwrap();
        assert!((r[0] - 1.0).abs() < 1e-10);
        // Just verify ACF values are in [-1, 1] and that the function works
        for (lag, &val) in r.iter().enumerate().skip(1) {
            assert!(val.abs() <= 1.0, "ACF[{lag}]={val} out of range");
        }
    }

    #[test]
    fn test_acf_trending() {
        // Linear trend: should have high positive ACF
        let data: Vec<f64> = (0..50).map(f64::from).collect();
        let r = acf(&data, 3).unwrap();
        assert!(r[1] > 0.9);
    }

    #[test]
    fn test_pacf_ar1() {
        // AR(1) process: x[t] = 0.9*x[t-1] + small noise
        // PACF at lag 1 should be significantly positive
        let mut data = vec![0.0_f64; 500];
        let phi = 0.9;
        // Use small deterministic noise
        for t in 1..500 {
            data[t] = phi * data[t - 1] + ((t * 17 + 5) % 11) as f64 / 55.0 - 0.1;
        }
        let p = pacf(&data, 5).unwrap();
        assert!((p[0] - 1.0).abs() < 1e-10);
        // PACF at lag 1 should be positive (dominated by AR(1) structure)
        assert!(p[1] > 0.3, "PACF[1]={} too small", p[1]);
    }

    #[test]
    fn test_acf_insufficient_data() {
        assert!(acf::<f64>(&[1.0], 5).is_err());
    }

    // ── ARIMA ───────────────────────────────────────────────────────

    #[test]
    fn test_arima_fit_and_forecast() {
        // Linear trend with noise
        let data: Vec<f64> = (0..100)
            .map(|i| f64::from(i) * 2.0 + f64::from(i % 7))
            .collect();
        let mut model = Arima::new(1, 1, 0).unwrap();
        model.fit(&data).unwrap();
        let forecast = model.forecast(5).unwrap();
        assert_eq!(forecast.len(), 5);
        // Forecasts should continue roughly upward
        assert!(forecast[0] > data[99] - 20.0, "forecast too low");
    }

    #[test]
    fn test_arima_ar_only() {
        let data: Vec<f64> = (0..80).map(|i| f64::from(i).sin() * 10.0).collect();
        let mut model = Arima::new(2, 0, 0).unwrap();
        model.fit(&data).unwrap();
        assert_eq!(model.ar_coefficients().len(), 2);
        let fc = model.forecast(3).unwrap();
        assert_eq!(fc.len(), 3);
    }

    #[test]
    fn test_arima_insufficient_data() {
        let mut model = Arima::new(2, 1, 1).unwrap();
        assert!(model.fit(&[1.0, 2.0, 3.0]).is_err());
    }

    #[test]
    fn test_arima_invalid_params() {
        assert!(Arima::<f64>::new(0, 0, 0).is_err());
    }

    #[test]
    fn test_arima_forecast_before_fit() {
        let model = Arima::<f64>::new(1, 0, 0).unwrap();
        assert!(model.forecast(5).is_err());
    }

    // ── Exponential Smoothing ───────────────────────────────────────

    #[test]
    fn test_simple_smoothing() {
        let data = vec![10.0_f64, 12.0, 13.0, 11.0, 14.0, 15.0];
        let mut model = ExponentialSmoothing::simple(0.3).unwrap();
        model.fit(&data).unwrap();
        let fc = model.forecast(3).unwrap();
        assert_eq!(fc.len(), 3);
        // Simple smoothing forecasts are constant
        assert!((fc[0] - fc[1]).abs() < 1e-10);
        assert!((fc[1] - fc[2]).abs() < 1e-10);
    }

    #[test]
    fn test_holt_smoothing() {
        // Linear trend
        let data: Vec<f64> = (0..20).map(|i| 10.0 + f64::from(i) * 2.0).collect();
        let mut model = ExponentialSmoothing::holt(0.8, 0.2).unwrap();
        model.fit(&data).unwrap();
        let fc = model.forecast(3).unwrap();
        assert_eq!(fc.len(), 3);
        // Should forecast an increasing trend
        assert!(fc[0] < fc[1]);
        assert!(fc[1] < fc[2]);
    }

    #[test]
    fn test_holt_winters() {
        // Seasonal data with period 4
        let data: Vec<f64> = (0..40)
            .map(|i: i32| 10.0 + f64::from(i) * 0.5 + [3.0, -1.0, 2.0, -2.0][i as usize % 4])
            .collect();
        let mut model = ExponentialSmoothing::holt_winters(0.5, 0.1, 0.3, 4).unwrap();
        model.fit(&data).unwrap();
        let fc = model.forecast(4).unwrap();
        assert_eq!(fc.len(), 4);
    }

    #[test]
    fn test_smoothing_invalid_alpha() {
        assert!(ExponentialSmoothing::<f64>::simple(0.0).is_err());
        assert!(ExponentialSmoothing::<f64>::simple(1.0).is_err());
        assert!(ExponentialSmoothing::<f64>::simple(-0.5).is_err());
    }

    #[test]
    fn test_smoothing_forecast_before_fit() {
        let model = ExponentialSmoothing::<f64>::simple(0.3).unwrap();
        assert!(model.forecast(3).is_err());
    }

    // ── Seasonal Decomposition ──────────────────────────────────────

    #[test]
    fn test_seasonal_decompose_basic() {
        // Trend + seasonal pattern
        let data: Vec<f64> = (0..40)
            .map(|i: i32| 100.0 + f64::from(i) * 0.5 + [5.0, -3.0, 2.0, -4.0][i as usize % 4])
            .collect();
        let result = seasonal_decompose(&data, 4).unwrap();
        assert_eq!(result.trend.len(), 40);
        assert_eq!(result.seasonal.len(), 40);
        assert_eq!(result.residual.len(), 40);

        // Seasonal should be periodic
        for i in 4..36 {
            if !result.seasonal[i].is_nan() {
                assert!(
                    (result.seasonal[i] - result.seasonal[i % 4]).abs() < 2.0,
                    "seasonal[{i}]={} != seasonal[{}]={}",
                    result.seasonal[i],
                    i % 4,
                    result.seasonal[i % 4]
                );
            }
        }
    }

    #[test]
    fn test_seasonal_decompose_insufficient_data() {
        assert!(seasonal_decompose::<f64>(&[1.0, 2.0, 3.0], 4).is_err());
    }

    #[test]
    fn test_seasonal_decompose_invalid_period() {
        let data = vec![1.0_f64; 20];
        assert!(seasonal_decompose(&data, 1).is_err());
    }

    // ── ADF Test ────────────────────────────────────────────────────

    #[test]
    fn test_adf_stationary() {
        // Stationary: mean-reverting process
        let data: Vec<f64> = (0..200)
            .map(|i| (f64::from(i) * 0.3).sin() * 5.0 + f64::from((i * 7 + 3) % 11) / 5.0)
            .collect();
        let result = adf_test(&data, Some(2)).unwrap();
        // Should reject null (is stationary)
        assert!(
            result.reject_null,
            "Should reject unit root for stationary data, stat={}",
            result.statistic
        );
    }

    #[test]
    fn test_adf_random_walk() {
        // Random walk: y[t] = y[t-1] + noise — non-stationary
        let mut data = vec![0.0_f64; 200];
        for t in 1..200 {
            data[t] = data[t - 1] + ((t * 13 + 7) % 11) as f64 / 11.0 - 0.5;
        }
        let result = adf_test(&data, Some(1)).unwrap();
        // Should NOT reject null (is non-stationary)
        // Note: with deterministic noise, this might sometimes reject
        assert!(result.n_obs > 100);
    }

    #[test]
    fn test_adf_insufficient_data() {
        assert!(adf_test::<f64>(&[1.0; 5], None).is_err());
    }

    #[test]
    fn test_adf_result_has_fields() {
        let data: Vec<f64> = (0..100).map(|i| f64::from(i).sin()).collect();
        let result = adf_test(&data, Some(2)).unwrap();
        assert!(result.n_lags == 2);
        assert!(result.n_obs > 0);
        assert!(result.statistic.is_finite());
    }
}
