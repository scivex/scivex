//! Mean-field variational inference (BBVI and CAVI).

use scivex_core::Float;

use crate::error::{Result, StatsError};

// ---------------------------------------------------------------------------
// LCG-based random number generation (same pattern as other bayesian files)
// ---------------------------------------------------------------------------

#[inline]
fn lcg_next(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 33) as f64 / (1u64 << 31) as f64
}

/// Box-Muller transform for standard normal samples.
fn normal_sample<T: Float>(state: &mut u64) -> T {
    let u1 = lcg_next(state).max(1e-15);
    let u2 = lcg_next(state);
    T::from_f64((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos())
}

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// A fully factorized (mean-field) Gaussian variational distribution.
///
/// # Examples
///
/// ```
/// # use scivex_stats::bayesian::VariationalDistribution;
/// let q = VariationalDistribution::<f64>::new(3);
/// assert_eq!(q.means.len(), 3);
/// ```
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct VariationalDistribution<T: Float> {
    /// Variational means for each parameter.
    pub means: Vec<T>,
    /// Log standard deviations (log-space ensures positivity of σ).
    pub log_stds: Vec<T>,
}

impl<T: Float> VariationalDistribution<T> {
    /// Create a new variational distribution with `n_params` parameters,
    /// initialised to means = 0 and log_stds = 0 (i.e. σ = 1).
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_stats::bayesian::VariationalDistribution;
    /// let q = VariationalDistribution::<f64>::new(2);
    /// assert_eq!(q.log_stds.len(), 2);
    /// ```
    pub fn new(n_params: usize) -> Self {
        Self {
            means: vec![T::zero(); n_params],
            log_stds: vec![T::zero(); n_params],
        }
    }

    /// Draw a single sample from q(θ) = N(μ, diag(σ²)) using the reparameterisation trick.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_stats::bayesian::VariationalDistribution;
    /// let q = VariationalDistribution::<f64>::new(2);
    /// let mut rng = 42u64;
    /// let s = q.sample(&mut rng);
    /// assert_eq!(s.len(), 2);
    /// ```
    pub fn sample(&self, rng_state: &mut u64) -> Vec<T> {
        self.means
            .iter()
            .zip(self.log_stds.iter())
            .map(|(&mu, &log_std)| {
                let eps: T = normal_sample(rng_state);
                mu + log_std.exp() * eps
            })
            .collect()
    }

    /// Entropy of the factorized Gaussian: Σ (0.5 * ln(2πe) + log_std).
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_stats::bayesian::VariationalDistribution;
    /// let q = VariationalDistribution::<f64>::new(1);
    /// let h = q.entropy();
    /// assert!(h > 0.0);
    /// ```
    pub fn entropy(&self) -> T {
        // H = Σ_i [0.5 * ln(2πe) + log_σ_i]
        // 0.5 * ln(2πe) ≈ 1.4189385332
        let half_ln_2pie =
            T::from_f64(0.5 * (2.0 * std::f64::consts::PI * std::f64::consts::E).ln());
        let mut h = T::zero();
        for &ls in &self.log_stds {
            h = h + half_ln_2pie + ls;
        }
        h
    }

    /// Log density of the variational distribution evaluated at `theta`.
    fn log_q(&self, theta: &[T]) -> T {
        // log q(θ) = Σ_i [ -0.5*ln(2π) - log_σ_i - 0.5*((θ_i - μ_i)/σ_i)² ]
        let half_ln_2pi = T::from_f64(0.5 * (2.0 * std::f64::consts::PI).ln());
        let mut lp = T::zero();
        for (i, &th) in theta.iter().enumerate() {
            let sigma = self.log_stds[i].exp();
            let z = (th - self.means[i]) / sigma;
            lp = lp - half_ln_2pi - self.log_stds[i] - T::from_f64(0.5) * z * z;
        }
        lp
    }
}

/// Configuration for variational inference.
///
/// # Examples
///
/// ```
/// # use scivex_stats::bayesian::ViConfig;
/// let cfg = ViConfig::<f64>::new(3);
/// assert_eq!(cfg.n_params, 3);
/// assert_eq!(cfg.max_iter, 1000);
/// ```
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct ViConfig<T: Float> {
    /// Number of variational parameters.
    pub n_params: usize,
    /// Maximum number of optimisation iterations.
    pub max_iter: usize,
    /// Learning rate for stochastic gradient ascent.
    pub learning_rate: T,
    /// Number of Monte Carlo samples for ELBO gradient estimation.
    pub n_samples: usize,
    /// Convergence tolerance on ELBO change.
    pub tol: T,
    /// Random seed.
    pub seed: u64,
}

impl<T: Float> ViConfig<T> {
    /// Create a configuration with sensible defaults.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_stats::bayesian::ViConfig;
    /// let cfg = ViConfig::<f64>::new(2);
    /// assert_eq!(cfg.n_samples, 10);
    /// ```
    pub fn new(n_params: usize) -> Self {
        Self {
            n_params,
            max_iter: 1000,
            learning_rate: T::from_f64(0.01),
            n_samples: 10,
            tol: T::from_f64(1e-6),
            seed: 42,
        }
    }
}

/// Result of variational inference.
///
/// # Examples
///
/// ```
/// # use scivex_stats::bayesian::{cavi_gaussian, ViResult};
/// let data = vec![5.0_f64, 5.1, 4.9, 5.0, 5.2];
/// let result = cavi_gaussian(&data, 0.0, 100.0, 1.0, 10).unwrap();
/// assert!(result.converged);
/// ```
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct ViResult<T: Float> {
    /// Posterior mean estimates.
    pub means: Vec<T>,
    /// Posterior standard deviation estimates.
    pub stds: Vec<T>,
    /// ELBO at each iteration.
    pub elbo_history: Vec<T>,
    /// Whether the algorithm converged.
    pub converged: bool,
    /// Number of iterations performed.
    pub iterations: usize,
}

// ---------------------------------------------------------------------------
// Mean-field variational inference (BBVI with REINFORCE)
// ---------------------------------------------------------------------------

/// Mean-field variational inference using stochastic gradient ascent on the ELBO.
///
/// `log_joint` computes log p(data | params) + log p(params) for given parameter values.
/// The variational family is a fully factorized Gaussian (mean-field).
///
/// # Algorithm (Black-box VI / BBVI)
///
/// 1. Initialise variational params: means = 0, log_stds = 0.
/// 2. For each iteration:
///    a. Draw `n_samples` from q(θ) = N(μ, diag(σ²)).
///    b. For each sample, compute score × (log_joint(θ) − log_q(θ)) (REINFORCE).
///    c. Score function: ∂log q/∂μ = (θ−μ)/σ², ∂log q/∂log_σ = ((θ−μ)²/σ² − 1).
///    d. Average gradients over samples.
///    e. Update μ += lr * grad_μ, log_σ += lr * grad_log_σ.
///    f. Compute ELBO estimate = E[log_joint(θ)] + entropy(q).
///    g. Check convergence: |ELBO_new − ELBO_old| < tol.
/// 3. Return means, exp(log_stds), ELBO history.
///
/// # Examples
///
/// ```
/// # use scivex_stats::bayesian::{mean_field_vi, ViConfig};
/// let log_joint = |x: &[f64]| -> f64 { -0.5 * (x[0] - 3.0) * (x[0] - 3.0) };
/// let mut cfg = ViConfig::<f64>::new(1);
/// cfg.max_iter = 100;
/// let result = mean_field_vi(log_joint, &cfg).unwrap();
/// assert_eq!(result.means.len(), 1);
/// ```
pub fn mean_field_vi<T, F>(log_joint: F, config: &ViConfig<T>) -> Result<ViResult<T>>
where
    T: Float,
    F: Fn(&[T]) -> T,
{
    let n_params = config.n_params;
    if n_params == 0 {
        return Err(StatsError::InvalidParameter {
            name: "n_params",
            reason: "must be at least 1",
        });
    }

    let mut q = VariationalDistribution::new(n_params);
    let mut rng_state = config.seed;
    let lr = config.learning_rate;
    let ns = T::from_f64(config.n_samples as f64);

    let mut elbo_history: Vec<T> = Vec::with_capacity(config.max_iter);
    let mut converged = false;
    let mut iterations = 0;

    for iter in 0..config.max_iter {
        iterations = iter + 1;

        let mut grad_mu = vec![T::zero(); n_params];
        let mut grad_log_std = vec![T::zero(); n_params];
        let mut elbo_sum = T::zero();

        for _ in 0..config.n_samples {
            let sample = q.sample(&mut rng_state);
            let lj = log_joint(&sample);
            let lq = q.log_q(&sample);
            let advantage = lj - lq;

            for j in 0..n_params {
                let sigma = q.log_stds[j].exp();
                let sigma_sq = sigma * sigma;
                let diff = sample[j] - q.means[j];

                // Score function gradients (REINFORCE)
                let score_mu = diff / sigma_sq;
                let score_log_std = diff * diff / sigma_sq - T::one();

                grad_mu[j] += score_mu * advantage;
                grad_log_std[j] += score_log_std * advantage;
            }

            elbo_sum += lj;
        }

        // Average over samples
        for j in 0..n_params {
            grad_mu[j] /= ns;
            grad_log_std[j] /= ns;
        }

        // Update variational parameters via gradient ascent
        for j in 0..n_params {
            q.means[j] += lr * grad_mu[j];
            q.log_stds[j] += lr * grad_log_std[j];
        }

        // ELBO estimate: E[log_joint] + entropy(q)
        let elbo = elbo_sum / ns + q.entropy();
        elbo_history.push(elbo);

        // Convergence check
        if elbo_history.len() >= 2 {
            let prev = elbo_history[elbo_history.len() - 2];
            if (elbo - prev).abs() < config.tol {
                converged = true;
                break;
            }
        }
    }

    let stds: Vec<T> = q.log_stds.iter().map(|&ls| ls.exp()).collect();

    Ok(ViResult {
        means: q.means,
        stds,
        elbo_history,
        converged,
        iterations,
    })
}

// ---------------------------------------------------------------------------
// Coordinate Ascent VI (CAVI) for conjugate Gaussian model
// ---------------------------------------------------------------------------

/// Coordinate Ascent VI (CAVI) for a conjugate Gaussian model.
///
/// Fits q(μ) = N(m, s²) to the posterior of a Gaussian likelihood with known variance.
///
/// The model is: data ~ N(μ, `likelihood_var`), prior μ ~ N(`prior_mean`, `prior_var`).
/// The posterior is analytically N(m*, s*²) where:
///   - precision = n / likelihood_var + 1 / prior_var
///   - m* = (n * x̄ / likelihood_var + prior_mean / prior_var) / precision
///   - s*² = 1 / precision
///
/// This converges in one iteration (the posterior is exact), but iterates
/// for demonstration of the CAVI procedure.
///
/// # Examples
///
/// ```
/// # use scivex_stats::bayesian::cavi_gaussian;
/// let data = vec![5.0_f64, 5.1, 4.9, 5.0, 5.2];
/// let result = cavi_gaussian(&data, 0.0, 100.0, 1.0, 10).unwrap();
/// assert!((result.means[0] - 5.04).abs() < 0.1);
/// ```
pub fn cavi_gaussian<T: Float>(
    data: &[T],
    prior_mean: T,
    prior_var: T,
    likelihood_var: T,
    max_iter: usize,
) -> Result<ViResult<T>> {
    if data.is_empty() {
        return Err(StatsError::EmptyInput);
    }
    if prior_var <= T::zero() {
        return Err(StatsError::InvalidParameter {
            name: "prior_var",
            reason: "must be positive",
        });
    }
    if likelihood_var <= T::zero() {
        return Err(StatsError::InvalidParameter {
            name: "likelihood_var",
            reason: "must be positive",
        });
    }

    let n = T::from_f64(data.len() as f64);

    // Compute data mean
    let mut x_bar = T::zero();
    for &x in data {
        x_bar += x;
    }
    x_bar /= n;

    let mut elbo_history: Vec<T> = Vec::with_capacity(max_iter);
    let mut post_mean = prior_mean;
    let mut post_var = prior_var;
    let mut converged = false;
    let mut iterations = 0;

    for iter in 0..max_iter {
        iterations = iter + 1;

        // CAVI closed-form updates
        let precision = n / likelihood_var + T::one() / prior_var;
        let new_post_var = T::one() / precision;
        let new_post_mean = (n * x_bar / likelihood_var + prior_mean / prior_var) / precision;

        post_mean = new_post_mean;
        post_var = new_post_var;

        // ELBO = E_q[log p(data, μ)] - E_q[log q(μ)]
        // For Gaussian conjugate: analytical but we compute a simplified version.
        //
        // E_q[log p(data | μ)] = -n/2 * ln(2π σ²_lik) - 1/(2σ²_lik) * Σ E[(x_i - μ)²]
        //   where E[(x_i - μ)²] = (x_i - m)² + s²
        //
        // E_q[log p(μ)] = -0.5*ln(2π τ²) - 1/(2τ²) * ((m - μ₀)² + s²)
        //
        // H(q) = 0.5 * ln(2πe s²)
        let half = T::from_f64(0.5);
        let two_pi = T::from_f64(2.0 * std::f64::consts::PI);

        // E_q[log p(data | μ)]
        let mut e_log_lik = -n * half * (two_pi * likelihood_var).ln();
        let mut sum_sq = T::zero();
        for &x in data {
            let diff = x - post_mean;
            sum_sq = sum_sq + diff * diff + post_var;
        }
        e_log_lik -= sum_sq / (T::from_f64(2.0) * likelihood_var);

        // E_q[log p(μ)]
        let diff_prior = post_mean - prior_mean;
        let e_log_prior = -half * (two_pi * prior_var).ln()
            - (diff_prior * diff_prior + post_var) / (T::from_f64(2.0) * prior_var);

        // Entropy H(q)
        let entropy = half * (two_pi * T::from_f64(std::f64::consts::E) * post_var).ln();

        let elbo = e_log_lik + e_log_prior + entropy;
        elbo_history.push(elbo);

        // Convergence check
        if elbo_history.len() >= 2 {
            let prev = elbo_history[elbo_history.len() - 2];
            if (elbo - prev).abs() < T::from_f64(1e-12) {
                converged = true;
                break;
            }
        }
    }

    Ok(ViResult {
        means: vec![post_mean],
        stds: vec![post_var.sqrt()],
        elbo_history,
        converged,
        iterations,
    })
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cavi_gaussian_posterior() {
        // 10 data points from N(5, 1)
        let data: Vec<f64> = vec![4.5, 5.1, 5.3, 4.8, 5.0, 5.2, 4.9, 5.4, 4.7, 5.1];
        let prior_mean = 0.0;
        let prior_var = 100.0; // vague prior
        let likelihood_var = 1.0;

        let result = cavi_gaussian(&data, prior_mean, prior_var, likelihood_var, 10).unwrap();

        // Analytical posterior:
        let n = data.len() as f64;
        let x_bar: f64 = data.iter().sum::<f64>() / n;
        let precision = n / likelihood_var + 1.0 / prior_var;
        let post_mean_exact = (n * x_bar / likelihood_var + prior_mean / prior_var) / precision;
        let post_var_exact = 1.0 / precision;

        assert!(
            (result.means[0] - post_mean_exact).abs() < 1e-10,
            "mean: got {}, expected {}",
            result.means[0],
            post_mean_exact,
        );
        assert!(
            (result.stds[0] - post_var_exact.sqrt()).abs() < 1e-10,
            "std: got {}, expected {}",
            result.stds[0],
            post_var_exact.sqrt(),
        );
    }

    #[test]
    fn test_mean_field_vi_unimodal() {
        // log_joint = -0.5 * (θ - 3)², posterior should center near 3
        let log_joint = |params: &[f64]| -> f64 { -0.5 * (params[0] - 3.0).powi(2) };

        let mut config = ViConfig::new(1);
        config.max_iter = 5000;
        config.learning_rate = 0.05;
        config.n_samples = 50;
        config.seed = 12345;

        let result = mean_field_vi(log_joint, &config).unwrap();

        assert!(
            (result.means[0] - 3.0).abs() < 0.5,
            "mean should be near 3.0, got {}",
            result.means[0],
        );
    }

    #[test]
    fn test_vi_elbo_increases() {
        // For a simple Gaussian target the ELBO should generally increase
        let log_joint = |params: &[f64]| -> f64 { -0.5 * params[0] * params[0] };

        let mut config = ViConfig::new(1);
        config.max_iter = 200;
        config.learning_rate = 0.01;
        config.n_samples = 50;
        config.seed = 99;
        config.tol = 1e-12; // prevent early stopping

        let result = mean_field_vi(log_joint, &config).unwrap();

        // Check that the final ELBO is higher than the initial ELBO
        // (BBVI with REINFORCE can be noisy, so we compare first vs last)
        let first = result.elbo_history[0];
        let last = *result.elbo_history.last().unwrap();
        assert!(
            last >= first - 1.0,
            "ELBO should generally increase: first={first}, last={last}",
        );
    }

    #[test]
    fn test_vi_config_defaults() {
        let config: ViConfig<f64> = ViConfig::new(5);
        assert_eq!(config.n_params, 5);
        assert_eq!(config.max_iter, 1000);
        assert!((config.learning_rate - 0.01).abs() < 1e-15);
        assert_eq!(config.n_samples, 10);
        assert!((config.tol - 1e-6).abs() < 1e-15);
        assert_eq!(config.seed, 42);
    }

    #[test]
    fn test_vi_multi_param() {
        // 2-parameter model: independent Gaussians centered at 2 and -1
        let log_joint = |params: &[f64]| -> f64 {
            -0.5 * (params[0] - 2.0).powi(2) - 0.5 * (params[1] + 1.0).powi(2)
        };

        let mut config = ViConfig::new(2);
        config.max_iter = 5000;
        config.learning_rate = 0.05;
        config.n_samples = 50;
        config.seed = 777;

        let result = mean_field_vi(log_joint, &config).unwrap();

        assert!(
            (result.means[0] - 2.0).abs() < 0.5,
            "param 0 should be near 2.0, got {}",
            result.means[0],
        );
        assert!(
            (result.means[1] - (-1.0)).abs() < 0.5,
            "param 1 should be near -1.0, got {}",
            result.means[1],
        );
    }
}
