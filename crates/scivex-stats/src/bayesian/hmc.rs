//! Hamiltonian Monte Carlo sampler.

use scivex_core::Float;
use scivex_core::random::Rng;

use crate::error::{Result, StatsError};

use super::{McmcConfig, McmcResult};

/// Hamiltonian Monte Carlo (HMC) sampler.
///
/// # Examples
///
/// ```
/// # use scivex_stats::bayesian::{HamiltonianMC, McmcConfig};
/// let hmc = HamiltonianMC::new(0.1_f64, 10);
/// let log_prob = |x: &[f64]| -> f64 { -0.5 * x[0] * x[0] };
/// let grad = |x: &[f64]| -> Vec<f64> { vec![-x[0]] };
/// let cfg = McmcConfig::new(100, 50, 42, 1);
/// let result = hmc.sample(log_prob, grad, &[0.0], &cfg).unwrap();
/// assert_eq!(result.samples[0].len(), 100);
/// ```
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct HamiltonianMC<T: Float> {
    /// Leapfrog step size.
    step_size: T,
    /// Number of leapfrog steps per proposal.
    n_leapfrog: usize,
}

impl<T: Float> HamiltonianMC<T> {
    /// Create a new HMC sampler.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_stats::bayesian::HamiltonianMC;
    /// let hmc = HamiltonianMC::new(0.05_f64, 20);
    /// ```
    pub fn new(step_size: T, n_leapfrog: usize) -> Self {
        Self {
            step_size,
            n_leapfrog,
        }
    }

    /// Draw samples from the target distribution.
    ///
    /// - `log_prob`: function returning the (unnormalised) log-probability.
    /// - `grad`: function returning the gradient of `log_prob` at a point.
    /// - `initial`: starting parameter values.
    /// - `config`: MCMC configuration.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_stats::bayesian::{HamiltonianMC, McmcConfig};
    /// let hmc = HamiltonianMC::new(0.1_f64, 10);
    /// let log_prob = |x: &[f64]| -> f64 { -0.5 * x[0] * x[0] };
    /// let grad = |x: &[f64]| -> Vec<f64> { vec![-x[0]] };
    /// let cfg = McmcConfig::new(100, 50, 42, 1);
    /// let result = hmc.sample(log_prob, grad, &[0.0], &cfg).unwrap();
    /// assert!(result.acceptance_rate[0] > 0.0);
    /// ```
    #[allow(clippy::too_many_lines)]
    pub fn sample<F, G>(
        &self,
        log_prob: F,
        grad: G,
        initial: &[T],
        config: &McmcConfig<T>,
    ) -> Result<McmcResult<T>>
    where
        F: Fn(&[T]) -> T,
        G: Fn(&[T]) -> Vec<T>,
    {
        let n_params = initial.len();
        if n_params == 0 {
            return Err(StatsError::EmptyInput);
        }

        let total = config.n_warmup + config.n_samples * config.thin;
        let mut rng = Rng::new(config.seed);

        let mut current = initial.to_vec();
        let mut current_lp = log_prob(&current);

        let mut samples: Vec<Vec<T>> = Vec::with_capacity(config.n_samples);
        let mut log_probs: Vec<T> = Vec::with_capacity(config.n_samples);
        let mut accepted = 0u64;
        let mut total_proposals = 0u64;

        let half = T::from_f64(0.5);
        let zero = T::from_f64(0.0);

        for step in 0..total {
            // Sample momentum.
            let mut momentum: Vec<T> = (0..n_params)
                .map(|_| T::from_f64(rng.next_normal_f64()))
                .collect();

            let current_ke: T = momentum.iter().map(|&p| half * p * p).sum();

            // Leapfrog integration.
            let mut q = current.clone();
            let mut g = grad(&q);

            // Half step for momentum.
            for j in 0..n_params {
                momentum[j] += half * self.step_size * g[j];
            }

            // Full steps.
            for l in 0..self.n_leapfrog {
                for j in 0..n_params {
                    q[j] += self.step_size * momentum[j];
                }
                g = grad(&q);
                if l < self.n_leapfrog - 1 {
                    for j in 0..n_params {
                        momentum[j] += self.step_size * g[j];
                    }
                }
            }

            // Half step for momentum at end.
            for j in 0..n_params {
                momentum[j] += half * self.step_size * g[j];
            }

            let proposed_lp = log_prob(&q);
            let proposed_ke: T = momentum.iter().map(|&p| half * p * p).sum();

            // Accept/reject based on Hamiltonian.
            let log_alpha = proposed_lp - current_lp + current_ke - proposed_ke;
            let u = T::from_f64(rng.next_f64());

            total_proposals += 1;
            if log_alpha >= zero || u.ln() < log_alpha {
                current = q;
                current_lp = proposed_lp;
                accepted += 1;
            }

            #[allow(clippy::manual_is_multiple_of)]
            if step >= config.n_warmup && (step - config.n_warmup) % config.thin == 0 {
                samples.push(current.clone());
                log_probs.push(current_lp);
            }
        }

        let acceptance_rate = if total_proposals > 0 {
            T::from_f64(accepted as f64 / total_proposals as f64)
        } else {
            T::from_f64(0.0)
        };

        Ok(McmcResult {
            samples: vec![samples],
            acceptance_rate: vec![acceptance_rate],
            log_probs: vec![log_probs],
        })
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn test_hmc_2d_gaussian() {
        // Sample from bivariate N([2, -1], I).
        let log_prob = |x: &[f64]| -> f64 { -0.5 * ((x[0] - 2.0).powi(2) + (x[1] + 1.0).powi(2)) };
        let grad_fn = |x: &[f64]| -> Vec<f64> { vec![-(x[0] - 2.0), -(x[1] + 1.0)] };

        let hmc = HamiltonianMC::new(0.1, 20);
        let config = McmcConfig::new(3000, 1000, 42, 1);
        let result = hmc.sample(log_prob, grad_fn, &[0.0, 0.0], &config).unwrap();

        let samples = &result.samples[0];
        let mean0: f64 = samples.iter().map(|s| s[0]).sum::<f64>() / samples.len() as f64;
        let mean1: f64 = samples.iter().map(|s| s[1]).sum::<f64>() / samples.len() as f64;

        assert!(
            (mean0 - 2.0).abs() < 0.3,
            "mean[0] = {mean0}, expected ~2.0"
        );
        assert!(
            (mean1 - (-1.0)).abs() < 0.3,
            "mean[1] = {mean1}, expected ~-1.0"
        );
    }
}
