//! GARCH(p,q) volatility models.

use scivex_core::Float;

use crate::error::{Result, StatsError};

/// GARCH(p,q) model for conditional volatility estimation.
///
/// The conditional variance `h_t` follows:
/// ```text
/// h_t = omega + sum_{i=1}^{q} alpha_i * r_{t-i}^2 + sum_{j=1}^{p} beta_j * h_{t-j}
/// ```
///
/// # Examples
///
/// ```
/// # use scivex_stats::garch::Garch;
/// let model = Garch::<f64>::new(1, 1).unwrap();
/// assert_eq!(model.p, 1);
/// assert_eq!(model.q, 1);
/// ```
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct Garch<T: Float> {
    /// GARCH order (number of lagged variance terms).
    pub p: usize,
    /// ARCH order (number of lagged squared-return terms).
    pub q: usize,
    /// Intercept (omega).
    pub omega: T,
    /// ARCH coefficients (alpha).
    pub alpha: Vec<T>,
    /// GARCH coefficients (beta).
    pub beta: Vec<T>,
    /// Whether the model has been fitted.
    fitted: bool,
}

impl<T: Float> Garch<T> {
    /// Create a new GARCH(p,q) model.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_stats::garch::Garch;
    /// let model = Garch::<f64>::new(1, 1).unwrap();
    /// assert_eq!(model.alpha.len(), 1);
    /// assert_eq!(model.beta.len(), 1);
    /// ```
    pub fn new(p: usize, q: usize) -> Result<Self> {
        if p == 0 && q == 0 {
            return Err(StatsError::InvalidParameter {
                name: "p,q",
                reason: "at least one of p or q must be positive",
            });
        }
        Ok(Self {
            p,
            q,
            omega: T::from_f64(0.0),
            alpha: vec![T::from_f64(0.0); q],
            beta: vec![T::from_f64(0.0); p],
            fitted: false,
        })
    }

    /// Fit the model to a series of returns using quasi-maximum likelihood.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_stats::garch::Garch;
    /// let returns: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin() * 0.02).collect();
    /// let mut model = Garch::<f64>::new(1, 1).unwrap();
    /// model.fit(&returns).unwrap();
    /// assert!(model.omega > 0.0);
    /// ```
    #[allow(clippy::too_many_lines)]
    pub fn fit(&mut self, returns: &[T]) -> Result<()> {
        let min_obs = self.p.max(self.q) + 10;
        let n = returns.len();
        if n < min_obs {
            return Err(StatsError::InsufficientData {
                need: min_obs,
                got: n,
            });
        }

        let r2: Vec<f64> = returns.iter().map(|r| r.to_f64() * r.to_f64()).collect();
        let sample_var: f64 = r2.iter().sum::<f64>() / n as f64;
        let max_order = self.p.max(self.q);

        let neg_ll = |omega: f64, alpha: &[f64], beta: &[f64]| -> f64 {
            let mut h = vec![sample_var; n];
            for t in max_order..n {
                let mut ht = omega;
                for (i, &a) in alpha.iter().enumerate() {
                    if t > i {
                        ht += a * r2[t - 1 - i];
                    }
                }
                for (j, &b) in beta.iter().enumerate() {
                    if t > j {
                        ht += b * h[t - 1 - j];
                    }
                }
                h[t] = ht.max(1e-12);
            }
            let mut ll = 0.0;
            for t in max_order..n {
                ll += h[t].ln() + r2[t] / h[t];
            }
            ll
        };

        let mut best_ll = f64::INFINITY;
        let mut best_omega = sample_var * 0.1;
        let mut best_alpha = vec![0.1; self.q];
        let mut best_beta = vec![0.8; self.p];

        let steps = 10;
        let step_size = 0.9 / f64::from(steps);

        if self.p <= 1 && self.q <= 1 {
            for ai in 1..steps {
                let a = f64::from(ai) * step_size;
                for bi in 0..steps {
                    let b = f64::from(bi) * step_size;
                    if a + b >= 0.999 {
                        continue;
                    }
                    let om = sample_var * (1.0 - a - b);
                    if om <= 0.0 {
                        continue;
                    }
                    let alpha_v = if self.q > 0 { vec![a] } else { vec![] };
                    let beta_v = if self.p > 0 { vec![b] } else { vec![] };
                    let ll = neg_ll(om, &alpha_v, &beta_v);
                    if ll < best_ll {
                        best_ll = ll;
                        best_omega = om;
                        best_alpha = alpha_v;
                        best_beta = beta_v;
                    }
                }
            }
        } else {
            let a_val = 0.05;
            let b_val = 0.85 / self.p.max(1) as f64;
            let sum_ab = a_val * self.q as f64 + b_val * self.p as f64;
            if sum_ab < 0.999 {
                best_omega = sample_var * (1.0 - sum_ab);
                best_alpha = vec![a_val; self.q];
                best_beta = vec![b_val; self.p];
            }
        }

        // Refine with small perturbations.
        let mut delta = 0.02_f64;
        for _ in 0..50 {
            let mut improved = false;
            for param_idx in 0..(1 + self.q + self.p) {
                for sign in &[-1.0, 1.0] {
                    let mut trial_omega = best_omega;
                    let mut trial_alpha = best_alpha.clone();
                    let mut trial_beta = best_beta.clone();

                    if param_idx == 0 {
                        trial_omega += sign * delta * sample_var;
                    } else if param_idx <= self.q {
                        trial_alpha[param_idx - 1] += sign * delta;
                    } else {
                        trial_beta[param_idx - 1 - self.q] += sign * delta;
                    }

                    if trial_omega <= 0.0 {
                        continue;
                    }
                    let sum_ab: f64 =
                        trial_alpha.iter().sum::<f64>() + trial_beta.iter().sum::<f64>();
                    if sum_ab >= 0.999 {
                        continue;
                    }
                    if trial_alpha.iter().any(|&a| a < 0.0) || trial_beta.iter().any(|&b| b < 0.0) {
                        continue;
                    }

                    let ll = neg_ll(trial_omega, &trial_alpha, &trial_beta);
                    if ll < best_ll {
                        best_ll = ll;
                        best_omega = trial_omega;
                        best_alpha = trial_alpha;
                        best_beta = trial_beta;
                        improved = true;
                        break;
                    }
                }
                if improved {
                    break;
                }
            }
            if !improved {
                delta *= 0.5;
                if delta < 1e-8 {
                    break;
                }
            }
        }

        self.omega = T::from_f64(best_omega);
        self.alpha = best_alpha.iter().map(|&a| T::from_f64(a)).collect();
        self.beta = best_beta.iter().map(|&b| T::from_f64(b)).collect();
        self.fitted = true;
        Ok(())
    }

    /// Compute the conditional volatility (standard deviation) series.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_stats::garch::Garch;
    /// let returns: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin() * 0.02).collect();
    /// let mut model = Garch::<f64>::new(1, 1).unwrap();
    /// model.fit(&returns).unwrap();
    /// let vol = model.conditional_volatility(&returns).unwrap();
    /// assert_eq!(vol.len(), 100);
    /// ```
    pub fn conditional_volatility(&self, returns: &[T]) -> Result<Vec<T>> {
        if !self.fitted {
            return Err(StatsError::InvalidParameter {
                name: "model",
                reason: "model must be fitted before computing volatility",
            });
        }
        let n = returns.len();
        let max_order = self.p.max(self.q);
        if n < max_order {
            return Err(StatsError::InsufficientData {
                need: max_order,
                got: n,
            });
        }

        let r2: Vec<T> = returns.iter().map(|&r| r * r).collect();
        let sample_var: T = r2.iter().copied().sum::<T>() / T::from_f64(n as f64);

        let mut h = vec![sample_var; n];
        for t in max_order..n {
            let mut ht = self.omega;
            for (i, &a) in self.alpha.iter().enumerate() {
                if t > i {
                    ht += a * r2[t - 1 - i];
                }
            }
            for (j, &b) in self.beta.iter().enumerate() {
                if t > j {
                    ht += b * h[t - 1 - j];
                }
            }
            let eps = T::from_f64(1e-12);
            h[t] = ht.max(eps);
        }

        Ok(h.iter().map(|&v| v.sqrt()).collect())
    }

    /// Forecast the conditional variance for `steps` periods ahead.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_stats::garch::Garch;
    /// let returns: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin() * 0.02).collect();
    /// let mut model = Garch::<f64>::new(1, 1).unwrap();
    /// model.fit(&returns).unwrap();
    /// let forecast = model.forecast_variance(5).unwrap();
    /// assert_eq!(forecast.len(), 5);
    /// ```
    pub fn forecast_variance(&self, steps: usize) -> Result<Vec<T>> {
        if !self.fitted {
            return Err(StatsError::InvalidParameter {
                name: "model",
                reason: "model must be fitted before forecasting",
            });
        }

        let sum_alpha: T = self.alpha.iter().copied().sum();
        let sum_beta: T = self.beta.iter().copied().sum();
        let one = T::from_f64(1.0);
        let persistence = sum_alpha + sum_beta;

        let uv = if persistence < one {
            self.omega / (one - persistence)
        } else {
            self.omega
        };

        let mut forecast = Vec::with_capacity(steps);
        let mut h = uv;
        for _ in 0..steps {
            h = self.omega + persistence * h;
            forecast.push(h);
        }

        Ok(forecast)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use scivex_core::random::Rng;

    fn generate_garch_returns(n: usize, seed: u64) -> Vec<f64> {
        let mut rng = Rng::new(seed);
        let mut returns = Vec::with_capacity(n);
        let mut h = 0.01_f64;
        let omega = 0.00001;
        let alpha = 0.1;
        let beta = 0.85;
        for _ in 0..n {
            let z = rng.next_normal_f64();
            let r = h.sqrt() * z;
            returns.push(r);
            h = omega + alpha * r * r + beta * h;
        }
        returns
    }

    #[test]
    fn test_garch11_fit_positive_variance() {
        let returns = generate_garch_returns(500, 42);
        let mut model = Garch::<f64>::new(1, 1).unwrap();
        model.fit(&returns).unwrap();
        assert!(model.omega > 0.0);
        assert!(model.alpha[0] >= 0.0);
        assert!(model.beta[0] >= 0.0);
        assert!(model.alpha[0] + model.beta[0] < 1.0);
    }

    #[test]
    fn test_conditional_volatility_length() {
        let returns = generate_garch_returns(200, 123);
        let mut model = Garch::<f64>::new(1, 1).unwrap();
        model.fit(&returns).unwrap();
        let vol = model.conditional_volatility(&returns).unwrap();
        assert_eq!(vol.len(), returns.len());
        for &v in &vol {
            assert!(v > 0.0);
        }
    }

    #[test]
    fn test_forecast_variance() {
        let returns = generate_garch_returns(200, 99);
        let mut model = Garch::<f64>::new(1, 1).unwrap();
        model.fit(&returns).unwrap();
        let forecast = model.forecast_variance(10).unwrap();
        assert_eq!(forecast.len(), 10);
        for &v in &forecast {
            assert!(v > 0.0);
        }
    }

    #[test]
    fn test_unfitted_model_errors() {
        let model = Garch::<f64>::new(1, 1).unwrap();
        let returns = vec![0.01, -0.02, 0.03, -0.01, 0.02];
        assert!(model.conditional_volatility(&returns).is_err());
        assert!(model.forecast_variance(5).is_err());
    }
}
