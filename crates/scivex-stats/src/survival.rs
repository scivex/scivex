//! Survival analysis: Kaplan-Meier estimator, log-rank test, Cox PH regression.

use scivex_core::Float;
use scivex_core::Tensor;

use crate::distributions::{ChiSquared, Distribution, Normal};
use crate::error::{Result, StatsError};

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// A single survival observation.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone, Copy)]
pub struct SurvivalRecord<T: Float> {
    /// Time to event or censoring.
    pub time: T,
    /// `true` if the event occurred, `false` if censored.
    pub event: bool,
}

/// Result of a Kaplan-Meier estimate.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct KaplanMeierEstimate<T: Float> {
    /// Unique event times (sorted).
    pub times: Vec<T>,
    /// Survival probability at each event time.
    pub survival_prob: Vec<T>,
    /// Number at risk just before each event time.
    pub at_risk: Vec<usize>,
    /// Number of events at each event time.
    pub events: Vec<usize>,
    /// Lower 95 % confidence interval (Greenwood).
    pub ci_lower: Vec<T>,
    /// Upper 95 % confidence interval (Greenwood).
    pub ci_upper: Vec<T>,
}

/// Result of a log-rank test.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone, Copy)]
pub struct LogRankResult<T: Float> {
    /// Chi-squared test statistic.
    pub statistic: T,
    /// Two-sided p-value from chi-squared(1).
    pub p_value: T,
}

/// Result of Cox proportional-hazards regression.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct CoxPHResult<T: Float> {
    /// Estimated regression coefficients.
    pub coefficients: Vec<T>,
    /// Standard errors of coefficients.
    pub std_errors: Vec<T>,
    /// Hazard ratios `exp(beta)`.
    pub hazard_ratios: Vec<T>,
    /// Two-sided p-values (Wald test).
    pub p_values: Vec<T>,
    /// Log partial likelihood at convergence.
    pub log_likelihood: T,
    /// Concordance index (Harrell's C).
    pub concordance: T,
}

// ---------------------------------------------------------------------------
// Kaplan-Meier
// ---------------------------------------------------------------------------

/// Compute the Kaplan-Meier (product-limit) survival estimate with 95 %
/// Greenwood confidence intervals.
#[allow(clippy::too_many_lines)]
pub fn kaplan_meier<T: Float>(records: &[SurvivalRecord<T>]) -> Result<KaplanMeierEstimate<T>> {
    if records.is_empty() {
        return Err(StatsError::EmptyInput);
    }

    let mut sorted: Vec<SurvivalRecord<T>> = records.to_vec();
    sorted.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());

    let mut unique_times: Vec<T> = Vec::new();
    let mut event_counts: Vec<usize> = Vec::new();
    let mut censor_counts: Vec<usize> = Vec::new();

    let mut i = 0;
    while i < sorted.len() {
        let t = sorted[i].time;
        let mut d = 0usize;
        let mut c = 0usize;
        while i < sorted.len() && sorted[i].time == t {
            if sorted[i].event {
                d += 1;
            } else {
                c += 1;
            }
            i += 1;
        }
        unique_times.push(t);
        event_counts.push(d);
        censor_counts.push(c);
    }

    let n = records.len();
    let one = T::from_f64(1.0);
    let zero = T::from_f64(0.0);

    let mut times_out: Vec<T> = Vec::new();
    let mut surv: Vec<T> = Vec::new();
    let mut at_risk_out: Vec<usize> = Vec::new();
    let mut events_out: Vec<usize> = Vec::new();

    let mut s = one;
    let mut greenwood_sum = zero;
    let mut n_at_risk = n;

    let z_alpha = T::from_f64(1.96);

    let mut ci_lower_out: Vec<T> = Vec::new();
    let mut ci_upper_out: Vec<T> = Vec::new();

    for (idx, &t) in unique_times.iter().enumerate() {
        let d = event_counts[idx];
        if d == 0 {
            n_at_risk -= censor_counts[idx];
            continue;
        }

        let ni = n_at_risk;
        let di = d;

        let ni_f = T::from_f64(ni as f64);
        let di_f = T::from_f64(di as f64);

        s *= one - di_f / ni_f;

        let denom = ni_f * (ni_f - di_f);
        if denom > zero {
            greenwood_sum += di_f / denom;
        }

        let se = s * greenwood_sum.sqrt();
        let lo = (s - z_alpha * se).max(zero);
        let hi = (s + z_alpha * se).min(one);

        times_out.push(t);
        surv.push(s);
        at_risk_out.push(ni);
        events_out.push(di);
        ci_lower_out.push(lo);
        ci_upper_out.push(hi);

        n_at_risk -= di + censor_counts[idx];
    }

    Ok(KaplanMeierEstimate {
        times: times_out,
        survival_prob: surv,
        at_risk: at_risk_out,
        events: events_out,
        ci_lower: ci_lower_out,
        ci_upper: ci_upper_out,
    })
}

/// Return the median survival time from a Kaplan-Meier estimate, i.e. the
/// first time at which S(t) <= 0.5.  Returns `None` if the curve never
/// reaches 0.5 (heavy censoring).
pub fn median_survival_time<T: Float>(km: &KaplanMeierEstimate<T>) -> Option<T> {
    let half = T::from_f64(0.5);
    for (i, &s) in km.survival_prob.iter().enumerate() {
        if s <= half {
            return Some(km.times[i]);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Log-rank test
// ---------------------------------------------------------------------------

struct TaggedRecord<U: Float> {
    time: U,
    event: bool,
    group: u8,
}

/// Two-sample log-rank test comparing survival distributions of two groups.
///
/// Returns a chi-squared(1) test statistic and p-value.
#[allow(clippy::too_many_lines)]
pub fn log_rank_test<T: Float>(
    group1: &[SurvivalRecord<T>],
    group2: &[SurvivalRecord<T>],
) -> Result<LogRankResult<T>> {
    if group1.is_empty() || group2.is_empty() {
        return Err(StatsError::EmptyInput);
    }

    let mut all: Vec<TaggedRecord<T>> = Vec::with_capacity(group1.len() + group2.len());
    for r in group1 {
        all.push(TaggedRecord {
            time: r.time,
            event: r.event,
            group: 0,
        });
    }
    for r in group2 {
        all.push(TaggedRecord {
            time: r.time,
            event: r.event,
            group: 1,
        });
    }
    all.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());

    let mut n1 = group1.len() as f64;
    let mut n2 = group2.len() as f64;

    let mut obs_minus_exp = 0.0_f64;
    let mut var_sum = 0.0_f64;

    let mut i = 0;
    while i < all.len() {
        let t = all[i].time;
        let mut d1 = 0.0_f64;
        let mut d2 = 0.0_f64;
        let mut c1 = 0.0_f64;
        let mut c2 = 0.0_f64;
        while i < all.len() && all[i].time == t {
            if all[i].event {
                if all[i].group == 0 {
                    d1 += 1.0;
                } else {
                    d2 += 1.0;
                }
            } else if all[i].group == 0 {
                c1 += 1.0;
            } else {
                c2 += 1.0;
            }
            i += 1;
        }

        let d = d1 + d2;
        let n = n1 + n2;

        if d > 0.0 && n > 0.0 {
            let e1 = d * n1 / n;
            obs_minus_exp += d1 - e1;

            if n > 1.0 {
                let v = d * n1 * n2 * (n - d) / (n * n * (n - 1.0));
                var_sum += v;
            }
        }

        n1 -= d1 + c1;
        n2 -= d2 + c2;
    }

    let zero = T::from_f64(0.0);
    if var_sum <= 0.0 {
        return Ok(LogRankResult {
            statistic: zero,
            p_value: T::from_f64(1.0),
        });
    }

    let chi2 = obs_minus_exp * obs_minus_exp / var_sum;
    let chi2_dist = ChiSquared::new(T::from_f64(1.0))?;
    let p = T::from_f64(1.0) - chi2_dist.cdf(T::from_f64(chi2));

    Ok(LogRankResult {
        statistic: T::from_f64(chi2),
        p_value: p,
    })
}

// ---------------------------------------------------------------------------
// Cox Proportional Hazards
// ---------------------------------------------------------------------------

/// Fit a Cox proportional-hazards model via Newton-Raphson on the partial
/// log-likelihood (Breslow approximation for ties).
///
/// - `times`: event/censoring times of length `n`.
/// - `events`: `true` if the event occurred, length `n`.
/// - `covariates`: `[n x p]` tensor of covariates.
#[allow(clippy::too_many_lines)]
#[allow(clippy::similar_names)]
pub fn cox_ph<T: Float>(
    times: &[T],
    events: &[bool],
    covariates: &Tensor<T>,
) -> Result<CoxPHResult<T>> {
    let n = times.len();
    if covariates.ndim() != 2 {
        return Err(StatsError::InvalidParameter {
            name: "covariates",
            reason: "must be a 2-D tensor",
        });
    }
    let n_cov = covariates.shape()[0];
    let p = covariates.shape()[1];
    if n != n_cov || n != events.len() {
        return Err(StatsError::LengthMismatch {
            expected: n,
            got: n_cov,
        });
    }
    if n < p + 1 {
        return Err(StatsError::InsufficientData {
            need: p + 1,
            got: n,
        });
    }

    let cov_slice = covariates.as_slice();
    let zero = T::from_f64(0.0);
    let one = T::from_f64(1.0);

    // Sort by descending time for risk-set computation.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| times[b].partial_cmp(&times[a]).unwrap());

    let max_iter = 200;
    let tol = T::from_f64(1e-9);
    let ridge = T::from_f64(1e-6);

    let mut beta = vec![zero; p];
    let mut converged = false;

    let mut prev_ll = T::from_f64(f64::NEG_INFINITY);

    for iter in 0..max_iter {
        let (grad, hess, ll) = cox_grad_hess(&order, events, cov_slice, &beta, p);

        // Newton step: solve (-H + ridge*I) d = g, then beta += d.
        let mut neg_hess: Vec<T> = hess.iter().map(|&h| -h).collect();
        for j in 0..p {
            neg_hess[j * p + j] += ridge;
        }
        let hess_t = Tensor::from_vec(neg_hess, vec![p, p])?;
        let grad_t = Tensor::from_slice(&grad, vec![p])?;
        let delta_t = hess_t
            .lstsq(&grad_t)
            .map_err(|_| StatsError::SingularMatrix)?;
        let delta = delta_t.as_slice().to_vec();

        // Halving line search: try full step, then halve until
        // log-likelihood improves or step is small enough.
        let mut step_size = one;
        let half = T::from_f64(0.5);
        let mut best_beta = beta.clone();
        let mut best_ll = ll;
        let mut found = false;

        for _ in 0..20 {
            let mut trial_beta = beta.clone();
            for j in 0..p {
                trial_beta[j] += delta[j] * step_size;
            }
            let (_, _, trial_ll) = cox_grad_hess(&order, events, cov_slice, &trial_beta, p);
            if trial_ll >= best_ll || trial_ll >= ll {
                best_beta = trial_beta;
                best_ll = trial_ll;
                found = true;
                break;
            }
            step_size *= half;
        }

        if !found {
            // Accept anyway with smallest step.
            for j in 0..p {
                best_beta[j] = beta[j] + delta[j] * step_size;
            }
        }

        let mut max_change = zero;
        for j in 0..p {
            let change = (best_beta[j] - beta[j]).abs();
            if change > max_change {
                max_change = change;
            }
        }
        beta = best_beta;

        // Check convergence on both parameter change and log-likelihood change.
        let ll_change = (best_ll - prev_ll).abs();
        prev_ll = best_ll;

        if max_change < tol || (iter > 0 && ll_change < tol) {
            converged = true;
            break;
        }

        if iter == max_iter - 1 {
            return Err(StatsError::ConvergenceFailure {
                iterations: max_iter,
            });
        }
    }

    if !converged {
        return Err(StatsError::ConvergenceFailure {
            iterations: max_iter,
        });
    }

    // Re-compute Hessian at the final beta for standard errors.
    let (_, hess_final, log_lik) = cox_grad_hess(&order, events, cov_slice, &beta, p);

    let mut neg_hess2: Vec<T> = hess_final.iter().map(|&h| -h).collect();
    // Add small ridge for invertibility.
    for j in 0..p {
        neg_hess2[j * p + j] += ridge;
    }
    let info_t = Tensor::from_vec(neg_hess2, vec![p, p])?;
    let info_inv = info_t.inv().map_err(|_| StatsError::SingularMatrix)?;
    let info_inv_s = info_inv.as_slice();

    let mut std_errors = Vec::with_capacity(p);
    let mut hazard_ratios = Vec::with_capacity(p);
    let mut p_values = Vec::with_capacity(p);

    let normal = Normal::standard();

    for j in 0..p {
        let var_j = info_inv_s[j * p + j];
        let se = if var_j > zero { var_j.sqrt() } else { zero };
        std_errors.push(se);
        hazard_ratios.push(beta[j].exp());

        let z = if se > zero { beta[j] / se } else { zero };
        let pv = T::from_f64(2.0) * (one - normal.cdf(z.abs()));
        p_values.push(pv);
    }

    let concordance = compute_concordance(times, events, &beta, cov_slice, p);

    Ok(CoxPHResult {
        coefficients: beta,
        std_errors,
        hazard_ratios,
        p_values,
        log_likelihood: log_lik,
        concordance,
    })
}

/// Compute gradient, Hessian, and log partial likelihood for Cox PH.
fn cox_grad_hess<T: Float>(
    order: &[usize],
    events: &[bool],
    cov_slice: &[T],
    beta: &[T],
    p: usize,
) -> (Vec<T>, Vec<T>, T) {
    let zero = T::from_f64(0.0);
    let mut grad = vec![zero; p];
    let mut hess = vec![zero; p * p];
    let mut s0 = zero;
    let mut s1 = vec![zero; p];
    let mut s2 = vec![zero; p * p];
    let mut log_lik = zero;

    for &idx in order {
        let xi = &cov_slice[idx * p..(idx + 1) * p];
        let eta: T = xi.iter().zip(beta.iter()).map(|(&x, &b)| x * b).sum();
        let w = eta.exp();

        s0 += w;
        for j in 0..p {
            s1[j] += xi[j] * w;
            for k in 0..p {
                s2[j * p + k] += xi[j] * xi[k] * w;
            }
        }

        if events[idx] {
            log_lik += eta - s0.ln();
            for j in 0..p {
                grad[j] += xi[j] - s1[j] / s0;
                for k in 0..p {
                    hess[j * p + k] -= s2[j * p + k] / s0 - s1[j] * s1[k] / (s0 * s0);
                }
            }
        }
    }

    (grad, hess, log_lik)
}

/// Harrell's concordance index.
fn compute_concordance<T: Float>(
    times: &[T],
    events: &[bool],
    beta: &[T],
    cov_slice: &[T],
    p: usize,
) -> T {
    let n = times.len();
    let mut eta: Vec<T> = Vec::with_capacity(n);
    for i in 0..n {
        let xi = &cov_slice[i * p..(i + 1) * p];
        let val: T = xi.iter().zip(beta.iter()).map(|(&x, &b)| x * b).sum();
        eta.push(val);
    }

    let mut concordant = 0u64;
    let mut discordant = 0u64;

    for i in 0..n {
        if !events[i] {
            continue;
        }
        for j in 0..n {
            if i == j {
                continue;
            }
            if times[j] > times[i] {
                if eta[j] < eta[i] {
                    concordant += 1;
                } else if eta[j] > eta[i] {
                    discordant += 1;
                }
            }
        }
    }

    let total = concordant + discordant;
    if total == 0 {
        return T::from_f64(0.5);
    }
    T::from_f64(concordant as f64 / total as f64)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    fn rec(time: f64, event: bool) -> SurvivalRecord<f64> {
        SurvivalRecord { time, event }
    }

    #[test]
    fn test_km_no_censoring() {
        let records: Vec<SurvivalRecord<f64>> = vec![
            rec(1.0, true),
            rec(2.0, true),
            rec(3.0, true),
            rec(4.0, true),
            rec(5.0, true),
        ];
        let km = kaplan_meier(&records).unwrap();
        assert_eq!(km.times.len(), 5);
        assert!((km.survival_prob[0] - 0.8).abs() < 1e-10);
        assert!((km.survival_prob[4] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_km_with_censoring() {
        let records = vec![
            rec(1.0, true),
            rec(2.0, false),
            rec(3.0, true),
            rec(4.0, false),
            rec(5.0, true),
        ];
        let km = kaplan_meier(&records).unwrap();
        assert!(km.survival_prob[0] <= 1.0);
        for w in km.survival_prob.windows(2) {
            assert!(w[1] <= w[0] + 1e-15);
        }
    }

    #[test]
    fn test_median_survival() {
        let records: Vec<SurvivalRecord<f64>> = (1..=10).map(|i| rec(f64::from(i), true)).collect();
        let km = kaplan_meier(&records).unwrap();
        let med = median_survival_time(&km);
        assert!(med.is_some());
        let m = med.unwrap();
        assert!((5.0..=6.0).contains(&m));
    }

    #[test]
    fn test_log_rank_different_groups() {
        let g1: Vec<SurvivalRecord<f64>> = (1..=10).map(|i| rec(f64::from(i), true)).collect();
        let g2: Vec<SurvivalRecord<f64>> = (20..=29).map(|i| rec(f64::from(i), true)).collect();
        let result = log_rank_test(&g1, &g2).unwrap();
        assert!(result.p_value < 0.05, "p = {}", result.p_value.to_f64());
    }

    #[test]
    fn test_log_rank_same_group() {
        let g1: Vec<SurvivalRecord<f64>> = vec![
            rec(1.0, true),
            rec(3.0, true),
            rec(5.0, true),
            rec(7.0, true),
            rec(9.0, true),
        ];
        let g2: Vec<SurvivalRecord<f64>> = vec![
            rec(2.0, true),
            rec(4.0, true),
            rec(6.0, true),
            rec(8.0, true),
            rec(10.0, true),
        ];
        let result = log_rank_test(&g1, &g2).unwrap();
        assert!(result.p_value > 0.05, "p = {}", result.p_value.to_f64());
    }

    #[test]
    fn test_cox_ph_basic() {
        let times: Vec<f64> = vec![10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let events: Vec<bool> = vec![true; 10];
        let cov_data: Vec<f64> = (1..=10).map(f64::from).collect();
        let cov = Tensor::from_vec(cov_data, vec![10, 1]).unwrap();
        let result = cox_ph(&times, &events, &cov).unwrap();
        assert_eq!(result.coefficients.len(), 1);
        assert!(result.coefficients[0] > 0.0);
        assert!(result.hazard_ratios[0] > 1.0);
    }

    #[test]
    fn test_cox_ph_concordance() {
        let times: Vec<f64> = vec![10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let events: Vec<bool> = vec![true; 10];
        let cov_data: Vec<f64> = (1..=10).map(f64::from).collect();
        let cov = Tensor::from_vec(cov_data, vec![10, 1]).unwrap();
        let result = cox_ph(&times, &events, &cov).unwrap();
        assert!(
            result.concordance > 0.5,
            "concordance = {}",
            result.concordance.to_f64()
        );
    }
}
