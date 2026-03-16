//! Metropolis-Hastings sampler with Gaussian random-walk proposal.

use scivex_core::Float;
use scivex_core::random::Rng;

use crate::error::{Result, StatsError};

use super::{McmcConfig, McmcResult};

/// Random-walk Metropolis-Hastings sampler.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct MetropolisHastings<T: Float> {
    /// Proposal standard deviation for each parameter.
    proposal_scale: Vec<T>,
}

impl<T: Float> MetropolisHastings<T> {
    /// Create a new MH sampler with the given proposal scales.
    pub fn new(proposal_scale: Vec<T>) -> Self {
        Self { proposal_scale }
    }

    /// Draw samples from the target distribution specified by `log_prob`.
    ///
    /// - `log_prob`: function returning the (unnormalised) log-probability.
    /// - `initial`: starting parameter values.
    /// - `config`: MCMC configuration.
    pub fn sample<F>(
        &self,
        log_prob: F,
        initial: &[T],
        config: &McmcConfig<T>,
    ) -> Result<McmcResult<T>>
    where
        F: Fn(&[T]) -> T,
    {
        let n_params = initial.len();
        if n_params == 0 {
            return Err(StatsError::EmptyInput);
        }
        if self.proposal_scale.len() != n_params {
            return Err(StatsError::LengthMismatch {
                expected: n_params,
                got: self.proposal_scale.len(),
            });
        }

        let total = config.n_warmup + config.n_samples * config.thin;
        let mut rng = Rng::new(config.seed);

        let mut current = initial.to_vec();
        let mut current_lp = log_prob(&current);

        let mut samples: Vec<Vec<T>> = Vec::with_capacity(config.n_samples);
        let mut log_probs: Vec<T> = Vec::with_capacity(config.n_samples);
        let mut accepted = 0u64;
        let mut total_proposals = 0u64;

        for step in 0..total {
            // Propose.
            let mut proposal = current.clone();
            for (j, prop) in proposal.iter_mut().enumerate().take(n_params) {
                let noise = T::from_f64(rng.next_normal_f64()) * self.proposal_scale[j];
                *prop += noise;
            }

            let proposal_lp = log_prob(&proposal);
            let log_alpha = proposal_lp - current_lp;
            let u = T::from_f64(rng.next_f64());

            total_proposals += 1;
            if log_alpha >= T::from_f64(0.0) || u.ln() < log_alpha {
                current = proposal;
                current_lp = proposal_lp;
                accepted += 1;
            }

            // After warmup, record samples at thinning intervals.
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
    fn test_mh_1d_gaussian() {
        // Sample from N(3, 1).
        let log_prob = |x: &[f64]| -> f64 { -0.5 * (x[0] - 3.0) * (x[0] - 3.0) };

        let mh = MetropolisHastings::new(vec![1.0]);
        let config = McmcConfig::new(5000, 1000, 42, 1);
        let result = mh.sample(log_prob, &[0.0], &config).unwrap();

        let samples = &result.samples[0];
        let mean: f64 = samples.iter().map(|s| s[0]).sum::<f64>() / samples.len() as f64;
        assert!((mean - 3.0).abs() < 0.3, "mean = {mean}, expected ~3.0");
    }

    #[test]
    fn test_mh_acceptance_rate() {
        let log_prob = |x: &[f64]| -> f64 { -0.5 * x[0] * x[0] };

        let mh = MetropolisHastings::new(vec![1.0]);
        let config = McmcConfig::new(2000, 500, 99, 1);
        let result = mh.sample(log_prob, &[0.0], &config).unwrap();

        let rate = result.acceptance_rate[0];
        assert!(
            rate > 0.1 && rate < 0.9,
            "acceptance rate = {}",
            rate.to_f64()
        );
    }
}
