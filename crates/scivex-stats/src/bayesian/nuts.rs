//! No-U-Turn Sampler (NUTS) — an extension of HMC that automatically tunes
//! the trajectory length using recursive tree doubling.

use scivex_core::Float;
use scivex_core::random::Rng;

use crate::error::{Result, StatsError};

use super::{McmcConfig, McmcResult};

/// No-U-Turn Sampler (NUTS) (Hoffman & Gelman, 2014).
///
/// NUTS extends Hamiltonian Monte Carlo by automatically selecting the number
/// of leapfrog steps, eliminating the need to hand-tune trajectory length.
/// It builds a balanced binary tree of leapfrog states and stops when the
/// trajectory makes a U-turn. During warmup, dual averaging is used to adapt
/// the step size to achieve a target acceptance probability.
///
/// # Examples
///
/// ```
/// # use scivex_stats::bayesian::{Nuts, McmcConfig};
/// let nuts = Nuts::new(0.1_f64, 10, 0.8);
/// let log_prob = |x: &[f64]| -> f64 { -0.5 * x[0] * x[0] };
/// let grad = |x: &[f64]| -> Vec<f64> { vec![-x[0]] };
/// let cfg = McmcConfig::new(100, 50, 42, 1);
/// let result = nuts.sample(log_prob, grad, &[0.0], &cfg).unwrap();
/// assert_eq!(result.samples[0].len(), 100);
/// ```
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct Nuts<T: Float> {
    /// Initial leapfrog step size (adapted during warmup).
    initial_step_size: T,
    /// Maximum tree depth for trajectory doubling.
    max_tree_depth: usize,
    /// Target acceptance probability for dual averaging.
    target_accept: T,
}

impl<T: Float> Nuts<T> {
    /// Create a new NUTS sampler.
    ///
    /// # Arguments
    ///
    /// * `initial_step_size` — starting leapfrog step size (adapted during warmup).
    /// * `max_tree_depth` — maximum number of tree doublings (default is 10).
    /// * `target_accept` — target acceptance probability for dual averaging (typically 0.8).
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_stats::bayesian::Nuts;
    /// let nuts = Nuts::new(0.1_f64, 10, 0.8);
    /// ```
    pub fn new(initial_step_size: T, max_tree_depth: usize, target_accept: T) -> Self {
        Self {
            initial_step_size,
            max_tree_depth,
            target_accept,
        }
    }

    /// Draw samples from the target distribution using NUTS.
    ///
    /// - `log_prob`: function returning the (unnormalised) log-probability.
    /// - `grad`: function returning the gradient of `log_prob` at a point.
    /// - `initial`: starting parameter values.
    /// - `config`: MCMC configuration.
    ///
    /// During warmup, the step size is adapted via dual averaging to achieve
    /// `target_accept`. After warmup, the adapted step size is fixed.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_stats::bayesian::{Nuts, McmcConfig};
    /// let nuts = Nuts::new(0.1_f64, 10, 0.8);
    /// let log_prob = |x: &[f64]| -> f64 { -0.5 * x[0] * x[0] };
    /// let grad = |x: &[f64]| -> Vec<f64> { vec![-x[0]] };
    /// let cfg = McmcConfig::new(200, 100, 42, 1);
    /// let result = nuts.sample(log_prob, grad, &[0.0], &cfg).unwrap();
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
        let mut sum_accept = T::from_f64(0.0);
        let mut total_proposals = 0u64;

        let half = T::from_f64(0.5);
        let zero = T::from_f64(0.0);
        let one = T::from_f64(1.0);

        // Dual averaging state.
        let gamma = T::from_f64(0.05);
        let t0 = T::from_f64(10.0);
        let kappa = T::from_f64(0.75);
        let mu = (T::from_f64(10.0) * self.initial_step_size).ln();
        let mut step_size = self.initial_step_size;
        let mut log_epsilon_bar = T::from_f64(0.0);
        let mut h_bar = T::from_f64(0.0);

        for step in 0..total {
            // Sample momentum ~ N(0, I).
            let momentum: Vec<T> = (0..n_params)
                .map(|_| T::from_f64(rng.next_normal_f64()))
                .collect();

            let kinetic: T = momentum.iter().map(|&p| half * p * p).sum();
            let joint0 = current_lp - kinetic;

            // Slice variable: log_u = joint0 + ln(U), U ~ Uniform(0,1).
            let log_u = joint0 + T::from_f64(rng.next_f64()).ln();

            // Initialize tree.
            let mut q_minus = current.clone();
            let mut r_minus = momentum.clone();
            let mut q_plus = current.clone();
            let mut r_plus = momentum.clone();
            let mut candidate = current.clone();
            let mut candidate_lp = current_lp;
            let mut depth: usize = 0;
            let mut n_valid = 1usize;
            let mut keep_going = true;
            let mut tree_alpha = zero;
            let mut tree_n_alpha = 0usize;

            while keep_going && depth < self.max_tree_depth {
                // Choose direction: forward (+1) or backward (-1).
                let direction: i32 = if rng.next_f64() < 0.5 { -1 } else { 1 };

                // Build a tree of depth `depth` in `direction`.
                // We extend from the appropriate end of the trajectory.
                let (from_q, from_r) = if direction == -1 {
                    (q_minus.clone(), r_minus.clone())
                } else {
                    (q_plus.clone(), r_plus.clone())
                };

                // Build tree iteratively: 2^depth leapfrog steps.
                let n_steps = 1usize << depth;
                let mut sub_q = from_q;
                let mut sub_r = from_r;
                let mut sub_candidate = sub_q.clone();
                let mut sub_candidate_lp = log_prob(&sub_candidate);
                let mut sub_n_valid = 0usize;
                let mut sub_ok = true;
                let mut sub_alpha = zero;
                let mut sub_n_alpha = 0usize;

                // Track the endpoints of this sub-tree.
                let mut sub_q_minus = sub_q.clone();
                let mut sub_r_minus = sub_r.clone();
                let mut sub_q_plus = sub_q.clone();
                let mut sub_r_plus = sub_r.clone();

                for leaf in 0..n_steps {
                    // One leapfrog step in `direction`.
                    let g = grad(&sub_q);
                    for j in 0..n_params {
                        sub_r[j] += half * step_size * T::from_f64(f64::from(direction)) * g[j];
                    }
                    for j in 0..n_params {
                        sub_q[j] += step_size * T::from_f64(f64::from(direction)) * sub_r[j];
                    }
                    let g2 = grad(&sub_q);
                    for j in 0..n_params {
                        sub_r[j] += half * step_size * T::from_f64(f64::from(direction)) * g2[j];
                    }

                    let lp_new = log_prob(&sub_q);
                    let ke_new: T = sub_r.iter().map(|&p| half * p * p).sum();
                    let joint_new = lp_new - ke_new;

                    // Acceptance statistic for dual averaging.
                    let log_alpha_leaf = joint_new - joint0;
                    let alpha_leaf = if log_alpha_leaf > zero {
                        one
                    } else {
                        log_alpha_leaf.exp()
                    };
                    sub_alpha += alpha_leaf;
                    sub_n_alpha += 1;

                    // Check if this state is in the slice.
                    if log_u <= joint_new {
                        sub_n_valid += 1;
                        // Multinomial sampling: accept with probability 1/sub_n_valid.
                        let accept_prob = T::from_f64(1.0 / sub_n_valid as f64);
                        if T::from_f64(rng.next_f64()) < accept_prob {
                            sub_candidate.clone_from(&sub_q);
                            sub_candidate_lp = lp_new;
                        }
                    }

                    // Divergence check: if energy error is too large, stop.
                    let max_energy_error = T::from_f64(1000.0);
                    if (joint_new - joint0).abs() > max_energy_error {
                        sub_ok = false;
                        break;
                    }

                    // Track sub-tree endpoints.
                    if leaf == 0 {
                        sub_q_minus.clone_from(&sub_q);
                        sub_r_minus.clone_from(&sub_r);
                    }
                    sub_q_plus.clone_from(&sub_q);
                    sub_r_plus.clone_from(&sub_r);
                }

                // If direction is backward, swap the endpoint meanings:
                // the "first" state we computed is actually the new far-backward end.
                if direction == -1 {
                    core::mem::swap(&mut sub_q_minus, &mut sub_q_plus);
                    core::mem::swap(&mut sub_r_minus, &mut sub_r_plus);
                }

                // U-turn check on the sub-tree itself.
                if sub_ok && n_steps > 1 {
                    let mut dot_fwd = zero;
                    let mut dot_bwd = zero;
                    for j in 0..n_params {
                        let dq = sub_q_plus[j] - sub_q_minus[j];
                        dot_fwd += dq * sub_r_plus[j];
                        dot_bwd += dq * sub_r_minus[j];
                    }
                    if dot_fwd < zero || dot_bwd < zero {
                        sub_ok = false;
                    }
                }

                // Merge this sub-tree into the main tree.
                tree_alpha += sub_alpha;
                tree_n_alpha += sub_n_alpha;

                if sub_ok && sub_n_valid > 0 {
                    // Accept candidate from sub-tree with probability sub_n_valid / (n_valid + sub_n_valid).
                    let accept_prob =
                        T::from_f64(sub_n_valid as f64 / (n_valid + sub_n_valid) as f64);
                    if T::from_f64(rng.next_f64()) < accept_prob {
                        candidate = sub_candidate;
                        candidate_lp = sub_candidate_lp;
                    }
                    n_valid += sub_n_valid;
                }

                // Update tree endpoints.
                if direction == -1 {
                    q_minus = sub_q_minus;
                    r_minus = sub_r_minus;
                } else {
                    q_plus = sub_q_plus;
                    r_plus = sub_r_plus;
                }

                // U-turn check on the full tree.
                if sub_ok {
                    let mut dot_fwd = zero;
                    let mut dot_bwd = zero;
                    for j in 0..n_params {
                        let dq = q_plus[j] - q_minus[j];
                        dot_fwd += dq * r_plus[j];
                        dot_bwd += dq * r_minus[j];
                    }
                    if dot_fwd < zero || dot_bwd < zero {
                        keep_going = false;
                    }
                } else {
                    keep_going = false;
                }

                depth += 1;
            }

            // Accept the candidate.
            current = candidate;
            current_lp = candidate_lp;

            // Dual averaging: adapt step size during warmup.
            let alpha_mean = if tree_n_alpha > 0 {
                tree_alpha / T::from_f64(tree_n_alpha as f64)
            } else {
                zero
            };
            sum_accept += alpha_mean;
            total_proposals += 1;

            if step < config.n_warmup {
                let m = T::from_f64((step + 1) as f64);
                let inv_m_t0 = one / (m + t0);
                h_bar = (one - inv_m_t0) * h_bar + (self.target_accept - alpha_mean) * inv_m_t0;
                let log_epsilon = mu - m.sqrt() / gamma * h_bar;
                let m_neg_kappa = (-(kappa * m.ln())).exp();
                log_epsilon_bar = m_neg_kappa * log_epsilon + (one - m_neg_kappa) * log_epsilon_bar;
                step_size = log_epsilon.exp();

                // Clamp step size to avoid numerical issues.
                let min_step = T::from_f64(1e-10);
                let max_step = T::from_f64(1e2);
                if step_size < min_step {
                    step_size = min_step;
                }
                if step_size > max_step {
                    step_size = max_step;
                }
            }

            if step == config.n_warmup.saturating_sub(1) && config.n_warmup > 0 {
                // Fix step size at end of warmup.
                step_size = log_epsilon_bar.exp();
                let min_step = T::from_f64(1e-10);
                if step_size < min_step {
                    step_size = min_step;
                }
            }

            #[allow(clippy::manual_is_multiple_of)]
            if step >= config.n_warmup && (step - config.n_warmup) % config.thin == 0 {
                samples.push(current.clone());
                log_probs.push(current_lp);
            }
        }

        let acceptance_rate = if total_proposals > 0 {
            sum_accept / T::from_f64(total_proposals as f64)
        } else {
            zero
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
    fn test_nuts_1d_gaussian() {
        // Sample from N(0, 1).
        let log_prob = |x: &[f64]| -> f64 { -0.5 * x[0] * x[0] };
        let grad_fn = |x: &[f64]| -> Vec<f64> { vec![-x[0]] };

        let nuts = Nuts::new(0.1, 10, 0.8);
        let config = McmcConfig::new(3000, 1000, 42, 1);
        let result = nuts.sample(log_prob, grad_fn, &[0.0], &config).unwrap();

        let samples = &result.samples[0];
        assert_eq!(samples.len(), 3000);

        let mean: f64 = samples.iter().map(|s| s[0]).sum::<f64>() / samples.len() as f64;
        let variance: f64 =
            samples.iter().map(|s| (s[0] - mean).powi(2)).sum::<f64>() / samples.len() as f64;

        assert!(mean.abs() < 0.3, "mean = {mean}, expected ~0.0");
        assert!(
            (variance - 1.0).abs() < 0.5,
            "variance = {variance}, expected ~1.0"
        );
    }

    #[test]
    fn test_nuts_2d_gaussian() {
        // Sample from bivariate N([2, -1], I).
        let log_prob = |x: &[f64]| -> f64 { -0.5 * ((x[0] - 2.0).powi(2) + (x[1] + 1.0).powi(2)) };
        let grad_fn = |x: &[f64]| -> Vec<f64> { vec![-(x[0] - 2.0), -(x[1] + 1.0)] };

        let nuts = Nuts::new(0.1, 10, 0.8);
        let config = McmcConfig::new(3000, 1000, 42, 1);
        let result = nuts
            .sample(log_prob, grad_fn, &[0.0, 0.0], &config)
            .unwrap();

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

    #[test]
    fn test_nuts_adapts_step_size() {
        // Use a deliberately poor initial step size and verify it changes.
        let log_prob = |x: &[f64]| -> f64 { -0.5 * x[0] * x[0] };
        let grad_fn = |x: &[f64]| -> Vec<f64> { vec![-x[0]] };

        let initial_step = 1.0_f64;
        let nuts = Nuts::new(initial_step, 10, 0.8);
        let config = McmcConfig::new(100, 500, 42, 1);
        let result = nuts.sample(log_prob, grad_fn, &[0.0], &config).unwrap();

        // The sampler should still produce valid samples.
        assert_eq!(result.samples[0].len(), 100);
        // Acceptance rate should be reasonable after adaptation.
        assert!(
            result.acceptance_rate[0] > 0.1,
            "acceptance_rate = {}, expected > 0.1",
            result.acceptance_rate[0]
        );
    }

    #[test]
    fn test_nuts_empty_initial() {
        let log_prob = |_x: &[f64]| -> f64 { 0.0 };
        let grad_fn = |_x: &[f64]| -> Vec<f64> { vec![] };

        let nuts = Nuts::new(0.1_f64, 10, 0.8);
        let config = McmcConfig::new(100, 50, 42, 1);
        let result = nuts.sample(log_prob, grad_fn, &[], &config);
        assert!(result.is_err());
        assert!(matches!(result, Err(StatsError::EmptyInput)));
    }

    #[test]
    fn test_nuts_vs_hmc() {
        use crate::bayesian::HamiltonianMC;

        // Both samplers should recover similar means from N(3, 1).
        let log_prob = |x: &[f64]| -> f64 { -0.5 * (x[0] - 3.0).powi(2) };
        let grad_fn = |x: &[f64]| -> Vec<f64> { vec![-(x[0] - 3.0)] };

        let nuts = Nuts::new(0.1, 10, 0.8);
        let config_nuts = McmcConfig::new(3000, 1000, 42, 1);
        let result_nuts = nuts
            .sample(log_prob, grad_fn, &[0.0], &config_nuts)
            .unwrap();

        let hmc = HamiltonianMC::new(0.1, 20);
        let config_hmc = McmcConfig::new(3000, 1000, 99, 1);
        let result_hmc = hmc.sample(log_prob, grad_fn, &[0.0], &config_hmc).unwrap();

        let mean_nuts: f64 = result_nuts.samples[0].iter().map(|s| s[0]).sum::<f64>()
            / result_nuts.samples[0].len() as f64;
        let mean_hmc: f64 = result_hmc.samples[0].iter().map(|s| s[0]).sum::<f64>()
            / result_hmc.samples[0].len() as f64;

        assert!(
            (mean_nuts - 3.0).abs() < 0.3,
            "NUTS mean = {mean_nuts}, expected ~3.0"
        );
        assert!(
            (mean_hmc - 3.0).abs() < 0.3,
            "HMC mean = {mean_hmc}, expected ~3.0"
        );
        assert!(
            (mean_nuts - mean_hmc).abs() < 0.5,
            "NUTS mean ({mean_nuts}) and HMC mean ({mean_hmc}) differ too much"
        );
    }
}
