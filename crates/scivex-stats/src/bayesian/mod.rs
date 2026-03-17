//! Bayesian inference: MCMC samplers and convergence diagnostics.

pub mod diagnostics;
pub mod hmc;
pub mod metropolis;
pub mod variational;

pub use diagnostics::{TraceSummary, effective_sample_size, rhat, trace_summary};
pub use hmc::HamiltonianMC;
pub use metropolis::MetropolisHastings;
pub use variational::{VariationalDistribution, ViConfig, ViResult, cavi_gaussian, mean_field_vi};

use scivex_core::Float;

/// Configuration for an MCMC sampler.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct McmcConfig<T: Float> {
    /// Number of samples to draw (after warmup).
    pub n_samples: usize,
    /// Number of warmup (burn-in) samples to discard.
    pub n_warmup: usize,
    /// Random seed.
    pub seed: u64,
    /// Thinning interval (keep every `thin`-th sample).
    pub thin: usize,
    /// Phantom to use T in the type.
    _marker: core::marker::PhantomData<T>,
}

impl<T: Float> McmcConfig<T> {
    /// Create a new MCMC configuration.
    pub fn new(n_samples: usize, n_warmup: usize, seed: u64, thin: usize) -> Self {
        Self {
            n_samples,
            n_warmup,
            seed,
            thin: thin.max(1),
            _marker: core::marker::PhantomData,
        }
    }
}

/// Result of an MCMC run.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct McmcResult<T: Float> {
    /// Samples: `[chain][sample][param]`.
    pub samples: Vec<Vec<Vec<T>>>,
    /// Acceptance rate per chain.
    pub acceptance_rate: Vec<T>,
    /// Log-probability at each sample: `[chain][sample]`.
    pub log_probs: Vec<Vec<T>>,
}
