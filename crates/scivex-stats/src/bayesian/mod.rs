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
///
/// # Examples
///
/// ```
/// # use scivex_stats::bayesian::McmcConfig;
/// let cfg = McmcConfig::<f64>::new(1000, 500, 42, 1);
/// assert_eq!(cfg.n_samples, 1000);
/// assert_eq!(cfg.n_warmup, 500);
/// ```
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
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_stats::bayesian::McmcConfig;
    /// let cfg = McmcConfig::<f64>::new(2000, 1000, 42, 2);
    /// assert_eq!(cfg.thin, 2);
    /// ```
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
///
/// # Examples
///
/// ```
/// # use scivex_stats::bayesian::{MetropolisHastings, McmcConfig};
/// let mh = MetropolisHastings::new(vec![1.0_f64]);
/// let log_prob = |x: &[f64]| -> f64 { -0.5 * x[0] * x[0] };
/// let cfg = McmcConfig::new(100, 50, 42, 1);
/// let result = mh.sample(log_prob, &[0.0], &cfg).unwrap();
/// assert!(!result.samples.is_empty());
/// ```
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
