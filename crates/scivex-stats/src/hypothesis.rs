//! Hypothesis tests: t-tests, chi-square, Kolmogorov–Smirnov, ANOVA, Mann–Whitney.

use scivex_core::Float;

use crate::descriptive::{mean, variance};
use crate::distributions::{ChiSquared, Distribution, Normal, StudentT};
use crate::error::{Result, StatsError};
use crate::special::regularized_beta;

/// Result of a hypothesis test.
#[derive(Debug, Clone)]
pub struct TestResult<T: Float> {
    /// The test statistic.
    pub statistic: T,
    /// The p-value.
    pub p_value: T,
    /// Degrees of freedom (if applicable).
    pub df: Option<T>,
}

/// One-sample t-test: test whether the population mean equals `mu_0`.
pub fn t_test_one_sample<T: Float>(data: &[T], mu_0: T) -> Result<TestResult<T>> {
    let n = data.len();
    if n < 2 {
        return Err(StatsError::InsufficientData { need: 2, got: n });
    }
    let nf = T::from_f64(n as f64);
    let m = mean(data)?;
    let v = variance(data)?;
    let se = (v / nf).sqrt();

    if se == T::from_f64(0.0) {
        return Ok(TestResult {
            statistic: T::from_f64(0.0),
            p_value: T::from_f64(1.0),
            df: Some(nf - T::from_f64(1.0)),
        });
    }

    let t = (m - mu_0) / se;
    let df = nf - T::from_f64(1.0);
    let dist = StudentT::new(df)?;
    // Two-tailed p-value
    let p = T::from_f64(2.0) * (T::from_f64(1.0) - dist.cdf(t.abs()));

    Ok(TestResult {
        statistic: t,
        p_value: p,
        df: Some(df),
    })
}

/// Welch's two-sample t-test (unequal variances).
pub fn t_test_two_sample<T: Float>(x: &[T], y: &[T]) -> Result<TestResult<T>> {
    let nx = x.len();
    let ny = y.len();
    if nx < 2 {
        return Err(StatsError::InsufficientData { need: 2, got: nx });
    }
    if ny < 2 {
        return Err(StatsError::InsufficientData { need: 2, got: ny });
    }
    let nxf = T::from_f64(nx as f64);
    let nyf = T::from_f64(ny as f64);
    let mx = mean(x)?;
    let my = mean(y)?;
    let vx = variance(x)?;
    let vy = variance(y)?;
    let one = T::from_f64(1.0);

    let se = (vx / nxf + vy / nyf).sqrt();
    if se == T::from_f64(0.0) {
        return Ok(TestResult {
            statistic: T::from_f64(0.0),
            p_value: T::from_f64(1.0),
            df: Some(nxf + nyf - T::from_f64(2.0)),
        });
    }

    let t = (mx - my) / se;

    // Welch-Satterthwaite degrees of freedom
    let sx_n = vx / nxf;
    let sy_n = vy / nyf;
    let num = (sx_n + sy_n) * (sx_n + sy_n);
    let denom = sx_n * sx_n / (nxf - one) + sy_n * sy_n / (nyf - one);
    let df = num / denom;

    let dist = StudentT::new(df)?;
    let p = T::from_f64(2.0) * (one - dist.cdf(t.abs()));

    Ok(TestResult {
        statistic: t,
        p_value: p,
        df: Some(df),
    })
}

/// Chi-square goodness-of-fit test.
///
/// `observed` and `expected` must have the same length and all `expected > 0`.
pub fn chi_square_test<T: Float>(observed: &[T], expected: &[T]) -> Result<TestResult<T>> {
    let n = observed.len();
    if n != expected.len() {
        return Err(StatsError::LengthMismatch {
            expected: n,
            got: expected.len(),
        });
    }
    if n < 2 {
        return Err(StatsError::InsufficientData { need: 2, got: n });
    }

    let zero = T::from_f64(0.0);
    let one = T::from_f64(1.0);
    let mut chi2 = zero;
    for (&o, &e) in observed.iter().zip(expected.iter()) {
        if e <= zero {
            return Err(StatsError::InvalidParameter {
                name: "expected",
                reason: "all values must be positive",
            });
        }
        let diff = o - e;
        chi2 += diff * diff / e;
    }

    let df = T::from_f64((n - 1) as f64);
    let dist = ChiSquared::new(df)?;
    let p = one - dist.cdf(chi2);

    Ok(TestResult {
        statistic: chi2,
        p_value: p,
        df: Some(df),
    })
}

/// Two-sample Kolmogorov–Smirnov test.
///
/// Tests whether two samples come from the same distribution.
pub fn ks_test_two_sample<T: Float>(x: &[T], y: &[T]) -> Result<TestResult<T>> {
    if x.is_empty() || y.is_empty() {
        return Err(StatsError::EmptyInput);
    }
    let mut xs = x.to_vec();
    let mut ys = y.to_vec();
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    ys.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));

    let nx = T::from_f64(x.len() as f64);
    let ny = T::from_f64(y.len() as f64);
    let one = T::from_f64(1.0);
    let zero = T::from_f64(0.0);

    // Merge and compute max |F1 - F2|
    let mut ix = 0usize;
    let mut iy = 0usize;
    let mut d_max = zero;

    while ix < xs.len() && iy < ys.len() {
        let ex = T::from_f64((ix + 1) as f64) / nx;
        let ey = T::from_f64((iy + 1) as f64) / ny;

        if xs[ix] <= ys[iy] {
            let d = (ex - T::from_f64(iy as f64) / ny).abs();
            d_max = d_max.max(d);
            ix += 1;
        } else {
            let d = (T::from_f64(ix as f64) / nx - ey).abs();
            d_max = d_max.max(d);
            iy += 1;
        }
    }
    // Handle remaining
    while ix < xs.len() {
        let ex = T::from_f64((ix + 1) as f64) / nx;
        let d = (ex - one).abs();
        d_max = d_max.max(d);
        ix += 1;
    }
    while iy < ys.len() {
        let ey = T::from_f64((iy + 1) as f64) / ny;
        let d = (one - ey).abs();
        d_max = d_max.max(d);
        iy += 1;
    }

    // Asymptotic p-value using Kolmogorov distribution approximation
    let en = (nx * ny / (nx + ny)).sqrt();
    let lambda = (en + T::from_f64(0.12) + T::from_f64(0.11) / en) * d_max;

    // Kolmogorov survival function approximation (first few terms)
    let two = T::from_f64(2.0);
    let mut p = zero;
    for k in 1..=100 {
        let kf = T::from_f64(f64::from(k));
        let sign = if k % 2 == 0 { -T::from_f64(1.0) } else { one };
        let term = sign * (-two * lambda * lambda * kf * kf).exp();
        p += term;
        if term.abs() < T::from_f64(1e-15) {
            break;
        }
    }
    let p_value = (two * p).max(zero).min(one);

    Ok(TestResult {
        statistic: d_max,
        p_value,
        df: None,
    })
}

/// One-way ANOVA (analysis of variance).
///
/// Tests whether the means of multiple groups are equal.
pub fn anova_oneway<T: Float>(groups: &[&[T]]) -> Result<TestResult<T>> {
    let k = groups.len();
    if k < 2 {
        return Err(StatsError::InsufficientData { need: 2, got: k });
    }
    let one = T::from_f64(1.0);
    let zero = T::from_f64(0.0);

    // Overall mean
    let total_n: usize = groups.iter().map(|g| g.len()).sum();
    if total_n <= k {
        return Err(StatsError::InsufficientData {
            need: k + 1,
            got: total_n,
        });
    }
    let grand_sum: T = groups.iter().flat_map(|g| g.iter().copied()).sum();
    let grand_mean = grand_sum / T::from_f64(total_n as f64);

    // Sum of squares between and within
    let mut ss_between = zero;
    let mut ss_within = zero;
    for group in groups {
        let ni = T::from_f64(group.len() as f64);
        let gi_mean = mean(group)?;
        ss_between += ni * (gi_mean - grand_mean) * (gi_mean - grand_mean);
        for &x in *group {
            ss_within += (x - gi_mean) * (x - gi_mean);
        }
    }

    let df_between = T::from_f64((k - 1) as f64);
    let df_within = T::from_f64((total_n - k) as f64);
    let ms_between = ss_between / df_between;
    let ms_within = ss_within / df_within;

    if ms_within == zero {
        return Ok(TestResult {
            statistic: T::infinity(),
            p_value: zero,
            df: Some(df_between),
        });
    }

    let f_stat = ms_between / ms_within;

    // p-value from F-distribution via regularized beta
    // P(F > f) = 1 - I_{d1*f/(d1*f+d2)}(d1/2, d2/2)  where d1=df_between, d2=df_within
    let half = T::from_f64(0.5);
    let x = df_between * f_stat / (df_between * f_stat + df_within);
    let ib = regularized_beta(x, df_between * half, df_within * half)?;
    let p_value = one - ib;

    Ok(TestResult {
        statistic: f_stat,
        p_value,
        df: Some(df_between),
    })
}

/// Mann–Whitney U test (non-parametric two-sample test).
///
/// Uses the normal approximation for the p-value.
pub fn mann_whitney_u<T: Float>(x: &[T], y: &[T]) -> Result<TestResult<T>> {
    let nx = x.len();
    let ny = y.len();
    if nx == 0 || ny == 0 {
        return Err(StatsError::EmptyInput);
    }

    // Combine and rank
    let mut combined: Vec<(T, bool)> = x
        .iter()
        .map(|&v| (v, true))
        .chain(y.iter().map(|&v| (v, false)))
        .collect();
    combined.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal));

    // Assign ranks with average tie-breaking
    let n_total = combined.len();
    let mut ranks = vec![0.0_f64; n_total];
    let mut i = 0;
    while i < n_total {
        let mut j = i;
        while j < n_total - 1
            && combined[j + 1]
                .0
                .partial_cmp(&combined[j].0)
                .unwrap_or(core::cmp::Ordering::Equal)
                == core::cmp::Ordering::Equal
        {
            j += 1;
        }
        let avg = (i + j) as f64 / 2.0 + 1.0;
        for r in &mut ranks[i..=j] {
            *r = avg;
        }
        i = j + 1;
    }

    // U statistic for x
    let r1: f64 = combined
        .iter()
        .zip(ranks.iter())
        .filter(|(c, _)| c.1)
        .map(|(_, &r)| r)
        .sum();
    let nxf = nx as f64;
    let nyf = ny as f64;
    let u1 = r1 - nxf * (nxf + 1.0) / 2.0;

    // Normal approximation
    let mu_u = nxf * nyf / 2.0;
    let sigma_u = (nxf * nyf * (nxf + nyf + 1.0) / 12.0).sqrt();

    let z = if sigma_u > f64::EPSILON {
        (u1 - mu_u) / sigma_u
    } else {
        0.0
    };

    let norm = Normal::<f64>::standard();
    let p_value = 2.0 * (1.0 - norm.cdf(z.abs()));

    Ok(TestResult {
        statistic: T::from_f64(u1),
        p_value: T::from_f64(p_value),
        df: None,
    })
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn test_t_test_one_sample_significant() {
        // Mean is clearly not 0
        let data = [5.0, 5.1, 4.9, 5.2, 4.8, 5.0, 5.1, 4.9];
        let result = t_test_one_sample(&data, 0.0).unwrap();
        assert!(result.p_value < 0.001);
    }

    #[test]
    fn test_t_test_one_sample_non_significant() {
        let data = [5.0, 5.1, 4.9, 5.2, 4.8, 5.0, 5.1, 4.9];
        let result = t_test_one_sample(&data, 5.0).unwrap();
        assert!(result.p_value > 0.05);
    }

    #[test]
    fn test_t_test_two_sample() {
        let x = [10.0, 10.5, 9.8, 10.2, 10.1];
        let y = [5.0, 5.2, 4.9, 5.1, 5.3];
        let result = t_test_two_sample(&x, &y).unwrap();
        assert!(result.p_value < 0.001);
    }

    #[test]
    fn test_chi_square() {
        let observed = [50.0, 30.0, 20.0];
        let expected = [33.33, 33.33, 33.34];
        let result = chi_square_test(&observed, &expected).unwrap();
        assert!(result.statistic > 0.0);
        assert!(result.p_value < 0.05);
    }

    #[test]
    fn test_ks_same_distribution() {
        // Two samples from the same range should have high p-value
        let x: Vec<f64> = (0..100).map(|i| f64::from(i) / 100.0).collect();
        let y: Vec<f64> = (0..100).map(|i| (f64::from(i) + 0.5) / 100.0).collect();
        let result = ks_test_two_sample(&x, &y).unwrap();
        assert!(result.p_value > 0.05);
    }

    #[test]
    fn test_anova_different_groups() {
        let g1 = [1.0, 2.0, 3.0, 2.0, 1.0];
        let g2 = [10.0, 11.0, 12.0, 11.0, 10.0];
        let g3 = [20.0, 21.0, 22.0, 21.0, 20.0];
        let result = anova_oneway(&[&g1, &g2, &g3]).unwrap();
        assert!(result.p_value < 0.001);
    }
}
