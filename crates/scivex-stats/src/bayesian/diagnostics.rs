//! MCMC convergence diagnostics: R-hat, effective sample size, trace summary.

use scivex_core::Float;

use super::McmcResult;

/// Compute the split-chain Gelman-Rubin R-hat diagnostic.
///
/// `chains` has shape `[n_chains][n_samples][n_params]`.
/// Returns one R-hat value per parameter.
///
/// # Examples
///
/// ```
/// # use scivex_stats::bayesian::rhat;
/// let chain1 = vec![vec![1.0_f64], vec![1.1], vec![0.9], vec![1.05]];
/// let chain2 = vec![vec![1.0_f64], vec![0.95], vec![1.1], vec![1.0]];
/// let r = rhat(&[chain1, chain2]);
/// assert_eq!(r.len(), 1);
/// ```
pub fn rhat<T: Float>(chains: &[Vec<Vec<T>>]) -> Vec<T> {
    if chains.is_empty() || chains[0].is_empty() || chains[0][0].is_empty() {
        return vec![];
    }

    let n_params = chains[0][0].len();
    let one = T::from_f64(1.0);

    // Split each chain in half to get more chains.
    let mut split_chains: Vec<Vec<Vec<T>>> = Vec::new();
    for chain in chains {
        let mid = chain.len() / 2;
        if mid == 0 {
            split_chains.push(chain.clone());
            continue;
        }
        split_chains.push(chain[..mid].to_vec());
        split_chains.push(chain[mid..].to_vec());
    }

    let m = split_chains.len(); // number of chains
    let n = split_chains[0].len(); // samples per chain

    if n < 2 || m < 2 {
        return vec![one; n_params];
    }

    let nf = T::from_f64(n as f64);
    let mf = T::from_f64(m as f64);

    let mut result = Vec::with_capacity(n_params);

    for p in 0..n_params {
        let mut chain_means: Vec<T> = Vec::with_capacity(m);
        let mut chain_vars: Vec<T> = Vec::with_capacity(m);

        for chain in &split_chains {
            let mean: T = chain.iter().map(|s| s[p]).sum::<T>() / nf;
            chain_means.push(mean);
            let var: T = chain
                .iter()
                .map(|s| (s[p] - mean) * (s[p] - mean))
                .sum::<T>()
                / (nf - one);
            chain_vars.push(var);
        }

        let grand_mean: T = chain_means.iter().copied().sum::<T>() / mf;

        let b: T = chain_means
            .iter()
            .map(|&m_val| (m_val - grand_mean) * (m_val - grand_mean))
            .sum::<T>()
            * nf
            / (mf - one);

        let w: T = chain_vars.iter().copied().sum::<T>() / mf;
        let var_hat = (one - one / nf) * w + b / nf;

        let r = if w > T::from_f64(0.0) {
            (var_hat / w).sqrt()
        } else {
            one
        };
        result.push(r);
    }

    result
}

/// Compute the effective sample size for each parameter across chains.
///
/// Uses a simple autocorrelation-based estimate.
///
/// # Examples
///
/// ```
/// # use scivex_stats::bayesian::effective_sample_size;
/// let chain = vec![vec![1.0_f64], vec![1.1], vec![0.9], vec![1.05], vec![1.0]];
/// let ess = effective_sample_size(&[chain]);
/// assert_eq!(ess.len(), 1);
/// assert!(ess[0] > 0.0);
/// ```
pub fn effective_sample_size<T: Float>(chains: &[Vec<Vec<T>>]) -> Vec<T> {
    if chains.is_empty() || chains[0].is_empty() || chains[0][0].is_empty() {
        return vec![];
    }

    let n_params = chains[0][0].len();
    let mut result = Vec::with_capacity(n_params);

    for p in 0..n_params {
        let mut pooled: Vec<T> = Vec::new();
        for chain in chains {
            for sample in chain {
                pooled.push(sample[p]);
            }
        }

        let n = pooled.len();
        let nf = T::from_f64(n as f64);
        let mean: T = pooled.iter().copied().sum::<T>() / nf;
        let var: T = pooled.iter().map(|&x| (x - mean) * (x - mean)).sum::<T>() / nf;

        if var <= T::from_f64(0.0) || n < 4 {
            result.push(nf);
            continue;
        }

        let mut tau = T::from_f64(1.0);
        let max_lag = n / 2;
        for lag in 1..max_lag {
            let mut acf_val = T::from_f64(0.0);
            for i in 0..(n - lag) {
                acf_val += (pooled[i] - mean) * (pooled[i + lag] - mean);
            }
            acf_val /= nf * var;

            if acf_val < T::from_f64(0.0) {
                break;
            }
            tau += T::from_f64(2.0) * acf_val;
        }

        let ess = nf / tau;
        result.push(ess);
    }

    result
}

/// Summary statistics for an MCMC trace.
///
/// # Examples
///
/// ```
/// # use scivex_stats::bayesian::{MetropolisHastings, McmcConfig, trace_summary};
/// let mh = MetropolisHastings::new(vec![1.0_f64]);
/// let log_prob = |x: &[f64]| -> f64 { -0.5 * x[0] * x[0] };
/// let cfg = McmcConfig::new(200, 100, 42, 1);
/// let result = mh.sample(log_prob, &[0.0], &cfg).unwrap();
/// let summary = trace_summary(&result);
/// assert_eq!(summary.mean.len(), 1);
/// ```
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct TraceSummary<T: Float> {
    /// Posterior mean per parameter.
    pub mean: Vec<T>,
    /// Posterior standard deviation per parameter.
    pub std: Vec<T>,
    /// 2.5th percentile per parameter.
    pub q025: Vec<T>,
    /// 25th percentile per parameter.
    pub q25: Vec<T>,
    /// 50th percentile (median) per parameter.
    pub q50: Vec<T>,
    /// 75th percentile per parameter.
    pub q75: Vec<T>,
    /// 97.5th percentile per parameter.
    pub q975: Vec<T>,
    /// R-hat per parameter.
    pub rhat: Vec<T>,
    /// Effective sample size per parameter.
    pub n_eff: Vec<T>,
}

/// Compute a full trace summary from MCMC results.
///
/// # Examples
///
/// ```
/// # use scivex_stats::bayesian::{MetropolisHastings, McmcConfig, trace_summary};
/// let mh = MetropolisHastings::new(vec![1.0_f64]);
/// let log_prob = |x: &[f64]| -> f64 { -0.5 * x[0] * x[0] };
/// let cfg = McmcConfig::new(200, 100, 42, 1);
/// let result = mh.sample(log_prob, &[0.0], &cfg).unwrap();
/// let ts = trace_summary(&result);
/// assert!(ts.q025[0] <= ts.q975[0]);
/// ```
#[allow(clippy::too_many_lines)]
pub fn trace_summary<T: Float>(result: &McmcResult<T>) -> TraceSummary<T> {
    let chains = &result.samples;
    if chains.is_empty() || chains[0].is_empty() || chains[0][0].is_empty() {
        return TraceSummary {
            mean: vec![],
            std: vec![],
            q025: vec![],
            q25: vec![],
            q50: vec![],
            q75: vec![],
            q975: vec![],
            rhat: vec![],
            n_eff: vec![],
        };
    }

    let n_params = chains[0][0].len();
    let r_hat_vals = rhat(chains);
    let ess_vals = effective_sample_size(chains);

    let mut means = Vec::with_capacity(n_params);
    let mut stds = Vec::with_capacity(n_params);
    let mut q025s = Vec::with_capacity(n_params);
    let mut q25s = Vec::with_capacity(n_params);
    let mut q50s = Vec::with_capacity(n_params);
    let mut q75s = Vec::with_capacity(n_params);
    let mut q975s = Vec::with_capacity(n_params);

    for p in 0..n_params {
        let mut vals: Vec<T> = Vec::new();
        for chain in chains {
            for sample in chain {
                vals.push(sample[p]);
            }
        }
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = vals.len();
        let nf = T::from_f64(n as f64);
        let mean: T = vals.iter().copied().sum::<T>() / nf;
        means.push(mean);

        let var: T = vals.iter().map(|&x| (x - mean) * (x - mean)).sum::<T>() / nf;
        stds.push(var.sqrt());

        q025s.push(quantile_sorted(&vals, 0.025));
        q25s.push(quantile_sorted(&vals, 0.25));
        q50s.push(quantile_sorted(&vals, 0.5));
        q75s.push(quantile_sorted(&vals, 0.75));
        q975s.push(quantile_sorted(&vals, 0.975));
    }

    TraceSummary {
        mean: means,
        std: stds,
        q025: q025s,
        q25: q25s,
        q50: q50s,
        q75: q75s,
        q975: q975s,
        rhat: r_hat_vals,
        n_eff: ess_vals,
    }
}

/// Quantile from a pre-sorted slice using linear interpolation.
#[allow(clippy::cast_possible_truncation)]
fn quantile_sorted<T: Float>(sorted: &[T], p: f64) -> T {
    let n = sorted.len();
    if n == 0 {
        return T::from_f64(0.0);
    }
    if n == 1 {
        return sorted[0];
    }
    let idx = p * (n - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    let frac = T::from_f64(idx - lo as f64);
    if lo == hi || hi >= n {
        return sorted[lo.min(n - 1)];
    }
    sorted[lo] + frac * (sorted[hi] - sorted[lo])
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use crate::bayesian::{McmcConfig, MetropolisHastings};

    #[test]
    fn test_rhat_converged() {
        let log_prob = |x: &[f64]| -> f64 { -0.5 * x[0] * x[0] };
        let mh = MetropolisHastings::new(vec![1.0]);

        let config1 = McmcConfig::new(2000, 500, 42, 1);
        let config2 = McmcConfig::new(2000, 500, 123, 1);

        let r1 = mh.sample(&log_prob, &[0.0], &config1).unwrap();
        let r2 = mh.sample(&log_prob, &[0.0], &config2).unwrap();

        let chains = vec![r1.samples[0].clone(), r2.samples[0].clone()];
        let r = rhat(&chains);
        assert!(r[0] < 1.1, "R-hat = {}, expected < 1.1", r[0].to_f64());
    }

    #[test]
    fn test_ess_less_than_total() {
        let log_prob = |x: &[f64]| -> f64 { -0.5 * x[0] * x[0] };
        let mh = MetropolisHastings::new(vec![1.0]);
        let config = McmcConfig::new(2000, 500, 42, 1);
        let result = mh.sample(log_prob, &[0.0], &config).unwrap();

        let ess = effective_sample_size(&result.samples);
        let total = result.samples[0].len() as f64;
        assert!(
            ess[0].to_f64() < total,
            "ESS = {}, total = {total}",
            ess[0].to_f64()
        );
    }

    #[test]
    fn test_trace_summary_quantiles_ordered() {
        let log_prob = |x: &[f64]| -> f64 { -0.5 * x[0] * x[0] };
        let mh = MetropolisHastings::new(vec![1.0]);
        let config = McmcConfig::new(2000, 500, 42, 1);
        let result = mh.sample(log_prob, &[0.0], &config).unwrap();

        let summary = trace_summary(&result);
        assert!(summary.q025[0] <= summary.q25[0]);
        assert!(summary.q25[0] <= summary.q50[0]);
        assert!(summary.q50[0] <= summary.q75[0]);
        assert!(summary.q75[0] <= summary.q975[0]);
    }
}
