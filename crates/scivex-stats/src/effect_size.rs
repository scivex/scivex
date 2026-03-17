//! Effect size measures for quantifying the magnitude of statistical effects.
//!
//! Provides Cohen's d, Glass's Δ, Hedges' g, η², ω², Cramér's V,
//! point-biserial correlation, and Cohen's w.

use scivex_core::Float;

use crate::descriptive;
use crate::error::{Result, StatsError};

/// Qualitative interpretation of an effect size magnitude.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffectSizeInterpretation {
    /// |d| < 0.2
    Negligible,
    /// 0.2 ≤ |d| < 0.5
    Small,
    /// 0.5 ≤ |d| < 0.8
    Medium,
    /// |d| ≥ 0.8
    Large,
}

/// Interpret a Cohen's d value using conventional thresholds.
pub fn interpret_cohens_d<T: Float>(d: T) -> EffectSizeInterpretation {
    let abs_d = d.abs();
    if abs_d < T::from_f64(0.2) {
        EffectSizeInterpretation::Negligible
    } else if abs_d < T::from_f64(0.5) {
        EffectSizeInterpretation::Small
    } else if abs_d < T::from_f64(0.8) {
        EffectSizeInterpretation::Medium
    } else {
        EffectSizeInterpretation::Large
    }
}

/// Compute Cohen's d — the standardised mean difference using a pooled
/// standard deviation.
///
/// `d = (mean1 - mean2) / pooled_std`
///
/// where `pooled_std = sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))`.
///
/// Both groups must contain at least 2 elements.
pub fn cohens_d<T: Float>(group1: &[T], group2: &[T]) -> Result<T> {
    let n1 = group1.len();
    let n2 = group2.len();
    if n1 < 2 {
        return Err(StatsError::InsufficientData { need: 2, got: n1 });
    }
    if n2 < 2 {
        return Err(StatsError::InsufficientData { need: 2, got: n2 });
    }

    let m1 = descriptive::mean(group1)?;
    let m2 = descriptive::mean(group2)?;
    let var1 = descriptive::variance(group1)?; // ddof=1
    let var2 = descriptive::variance(group2)?;

    let n1f = T::from_f64(n1 as f64);
    let n2f = T::from_f64(n2 as f64);
    let one = T::from_f64(1.0);
    let two = T::from_f64(2.0);

    let pooled_var = ((n1f - one) * var1 + (n2f - one) * var2) / (n1f + n2f - two);
    let pooled_std = pooled_var.sqrt();

    if pooled_std == T::from_f64(0.0) {
        return Ok(T::from_f64(0.0));
    }

    Ok((m1 - m2) / pooled_std)
}

/// Compute Glass's Δ — the standardised mean difference using the control
/// group's standard deviation as the denominator.
///
/// `Δ = (mean_treatment - mean_control) / std_control`
///
/// Both groups must contain at least 2 elements.
pub fn glass_delta<T: Float>(treatment: &[T], control: &[T]) -> Result<T> {
    let nt = treatment.len();
    let nc = control.len();
    if nt < 2 {
        return Err(StatsError::InsufficientData { need: 2, got: nt });
    }
    if nc < 2 {
        return Err(StatsError::InsufficientData { need: 2, got: nc });
    }

    let m_t = descriptive::mean(treatment)?;
    let m_c = descriptive::mean(control)?;
    let s_c = descriptive::std_dev(control)?;

    if s_c == T::from_f64(0.0) {
        return Ok(T::from_f64(0.0));
    }

    Ok((m_t - m_c) / s_c)
}

/// Compute Hedges' g — Cohen's d with a small-sample bias correction factor.
///
/// `g = d * (1 - 3 / (4*(n1+n2) - 9))`
///
/// Both groups must contain at least 2 elements.
pub fn hedges_g<T: Float>(group1: &[T], group2: &[T]) -> Result<T> {
    let d = cohens_d(group1, group2)?;
    let n = (group1.len() + group2.len()) as f64;
    let correction = T::from_f64(1.0 - 3.0 / (4.0 * n - 9.0));
    Ok(d * correction)
}

/// Compute η² (eta squared) — the proportion of total variance explained by
/// group membership.
///
/// `η² = SS_between / SS_total`
///
/// Requires at least 2 groups, each with at least 1 element.
pub fn eta_squared<T: Float>(groups: &[&[T]]) -> Result<T> {
    if groups.len() < 2 {
        return Err(StatsError::InsufficientData {
            need: 2,
            got: groups.len(),
        });
    }

    // Grand mean
    let total_n: usize = groups.iter().map(|g| g.len()).sum();
    if total_n == 0 {
        return Err(StatsError::EmptyInput);
    }
    let grand_sum: T = groups.iter().flat_map(|g| g.iter().copied()).sum();
    let grand_mean = grand_sum / T::from_f64(total_n as f64);

    let mut ss_between = T::from_f64(0.0);
    for &group in groups {
        if group.is_empty() {
            return Err(StatsError::EmptyInput);
        }
        let m_k = descriptive::mean(group)?;
        let n_k = T::from_f64(group.len() as f64);
        let diff = m_k - grand_mean;
        ss_between += n_k * diff * diff;
    }

    let ss_total: T = groups
        .iter()
        .flat_map(|g| g.iter().copied())
        .map(|x| {
            let diff = x - grand_mean;
            diff * diff
        })
        .sum();

    if ss_total == T::from_f64(0.0) {
        return Ok(T::from_f64(0.0));
    }

    Ok(ss_between / ss_total)
}

/// Compute ω² (omega squared) — a less biased alternative to η².
///
/// `ω² = (SS_between - (k-1)*MS_within) / (SS_total + MS_within)`
///
/// Requires at least 2 groups, each with at least 1 element.
pub fn omega_squared<T: Float>(groups: &[&[T]]) -> Result<T> {
    let k = groups.len();
    if k < 2 {
        return Err(StatsError::InsufficientData { need: 2, got: k });
    }

    let total_n: usize = groups.iter().map(|g| g.len()).sum();
    if total_n == 0 {
        return Err(StatsError::EmptyInput);
    }
    let grand_sum: T = groups.iter().flat_map(|g| g.iter().copied()).sum();
    let grand_mean = grand_sum / T::from_f64(total_n as f64);

    let mut ss_between = T::from_f64(0.0);
    let mut ss_within = T::from_f64(0.0);

    for &group in groups {
        if group.is_empty() {
            return Err(StatsError::EmptyInput);
        }
        let m_k = descriptive::mean(group)?;
        let n_k = T::from_f64(group.len() as f64);
        let diff = m_k - grand_mean;
        ss_between += n_k * diff * diff;

        for &x in group {
            let d = x - m_k;
            ss_within += d * d;
        }
    }

    let ss_total = ss_between + ss_within;
    let df_within = total_n - k;
    if df_within == 0 {
        return Ok(T::from_f64(0.0));
    }
    let ms_within = ss_within / T::from_f64(df_within as f64);
    let k_minus_1 = T::from_f64((k - 1) as f64);

    let denom = ss_total + ms_within;
    if denom == T::from_f64(0.0) {
        return Ok(T::from_f64(0.0));
    }

    Ok((ss_between - k_minus_1 * ms_within) / denom)
}

/// Compute Cramér's V from a contingency table.
///
/// `V = sqrt(χ² / (n * min(r-1, c-1)))`
///
/// `observed` is a flat row-major contingency table with `rows` rows and `cols`
/// columns. Each inner `Vec` represents one row.
pub fn cramers_v<T: Float>(observed: &[Vec<T>], rows: usize, cols: usize) -> Result<T> {
    if observed.len() != rows {
        return Err(StatsError::LengthMismatch {
            expected: rows,
            got: observed.len(),
        });
    }
    for row in observed {
        if row.len() != cols {
            return Err(StatsError::LengthMismatch {
                expected: cols,
                got: row.len(),
            });
        }
    }
    if rows < 2 || cols < 2 {
        return Err(StatsError::InsufficientData {
            need: 2,
            got: rows.min(cols),
        });
    }

    let zero = T::from_f64(0.0);

    // Row totals, column totals, grand total
    let mut row_totals = vec![zero; rows];
    let mut col_totals = vec![zero; cols];
    let mut grand_total = zero;

    for (i, row) in observed.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            row_totals[i] += val;
            col_totals[j] += val;
            grand_total += val;
        }
    }

    if grand_total == zero {
        return Err(StatsError::EmptyInput);
    }

    // χ² = Σ (O - E)² / E
    let mut chi2 = zero;
    for (i, row) in observed.iter().enumerate() {
        for (j, &o) in row.iter().enumerate() {
            let e = row_totals[i] * col_totals[j] / grand_total;
            if e == zero {
                continue;
            }
            let diff = o - e;
            chi2 += diff * diff / e;
        }
    }

    let min_dim = (rows - 1).min(cols - 1);
    let denom = grand_total * T::from_f64(min_dim as f64);

    if denom == zero {
        return Ok(zero);
    }

    Ok((chi2 / denom).sqrt())
}

/// Compute the point-biserial correlation between a binary variable and a
/// continuous variable.
///
/// `r_pb = (M1 - M0) / S * sqrt(n0 * n1 / n²)`
///
/// where `S` is the standard deviation of the full continuous sample.
pub fn point_biserial<T: Float>(binary: &[bool], continuous: &[T]) -> Result<T> {
    let n = binary.len();
    if n != continuous.len() {
        return Err(StatsError::LengthMismatch {
            expected: n,
            got: continuous.len(),
        });
    }
    if n < 2 {
        return Err(StatsError::InsufficientData { need: 2, got: n });
    }

    let zero = T::from_f64(0.0);

    let mut sum0 = zero;
    let mut sum1 = zero;
    let mut n0: usize = 0;
    let mut n1: usize = 0;

    for (i, &b) in binary.iter().enumerate() {
        if b {
            sum1 += continuous[i];
            n1 += 1;
        } else {
            sum0 += continuous[i];
            n0 += 1;
        }
    }

    if n0 == 0 || n1 == 0 {
        return Err(StatsError::InsufficientData { need: 1, got: 0 });
    }

    let m0 = sum0 / T::from_f64(n0 as f64);
    let m1 = sum1 / T::from_f64(n1 as f64);

    // Population std of the full sample (ddof=0)
    let s = descriptive::std_dev_with_ddof(continuous, 0)?;

    if s == zero {
        return Ok(zero);
    }

    let nf = T::from_f64(n as f64);
    let n0f = T::from_f64(n0 as f64);
    let n1f = T::from_f64(n1 as f64);

    Ok((m1 - m0) / s * (n0f * n1f / (nf * nf)).sqrt())
}

/// Compute Cohen's w for goodness-of-fit.
///
/// `w = sqrt(Σ (O_i - E_i)² / E_i  /  n)`
///
/// where `n = Σ O_i`.  Equivalently, `w = sqrt(χ² / n)`.
pub fn cohens_w<T: Float>(observed: &[T], expected: &[T]) -> Result<T> {
    if observed.len() != expected.len() {
        return Err(StatsError::LengthMismatch {
            expected: expected.len(),
            got: observed.len(),
        });
    }
    if observed.is_empty() {
        return Err(StatsError::EmptyInput);
    }

    let zero = T::from_f64(0.0);
    let mut chi2 = zero;
    let mut n = zero;

    for (&o, &e) in observed.iter().zip(expected.iter()) {
        if e == zero {
            continue;
        }
        let diff = o - e;
        chi2 += diff * diff / e;
        n += o;
    }

    if n == zero {
        return Ok(zero);
    }

    Ok((chi2 / n).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-6;

    #[test]
    fn test_cohens_d_identical_groups() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let b = [1.0, 2.0, 3.0, 4.0, 5.0];
        let d = cohens_d(&a, &b).unwrap();
        assert!(d.abs() < TOL, "d = {d}");
    }

    #[test]
    fn test_cohens_d_known_value() {
        // Two groups with known separation and equal spread
        let g1 = [-0.5_f64, 0.5, -0.5, 0.5]; // mean=0, var=1/3
        let g2 = [0.5, 1.5, 0.5, 1.5]; // mean=1, var=1/3
        let d = cohens_d(&g1, &g2).unwrap();
        // pooled_std = sqrt(var) = sqrt(1/3)
        // d = -1 / sqrt(1/3) = -sqrt(3) ≈ -1.7320508
        assert!((d + 3.0_f64.sqrt()).abs() < TOL, "d = {d}");
    }

    #[test]
    fn test_hedges_g_small_sample_correction() {
        let g1 = [1.0, 2.0, 3.0];
        let g2 = [4.0, 5.0, 6.0];
        let d = cohens_d(&g1, &g2).unwrap();
        let g = hedges_g(&g1, &g2).unwrap();
        // Hedges' g should have smaller absolute value than Cohen's d
        assert!(g.abs() < d.abs(), "g={g}, d={d}");
    }

    #[test]
    fn test_eta_squared_known() {
        // Three groups: [1,1,1], [2,2,2], [3,3,3]
        // Grand mean = 2, SS_between = 3*(1-2)^2 + 3*(2-2)^2 + 3*(3-2)^2 = 6
        // SS_total = 6 (same, since SS_within = 0)
        // η² = 1.0
        let g1: &[f64] = &[1.0, 1.0, 1.0];
        let g2: &[f64] = &[2.0, 2.0, 2.0];
        let g3: &[f64] = &[3.0, 3.0, 3.0];
        let eta2 = eta_squared(&[g1, g2, g3]).unwrap();
        assert!((eta2 - 1.0).abs() < TOL, "eta2 = {eta2}");

        // Add within-group variance
        let g1: &[f64] = &[1.0, 2.0, 3.0]; // mean=2
        let g2: &[f64] = &[4.0, 5.0, 6.0]; // mean=5
        let g3: &[f64] = &[7.0, 8.0, 9.0]; // mean=8
        // grand mean = 5
        // SS_between = 3*9 + 0 + 3*9 = 54
        // SS_total = 16+9+4+1+0+1+4+9+16 = 60
        let eta2 = eta_squared(&[g1, g2, g3]).unwrap();
        assert!((eta2 - 54.0 / 60.0).abs() < TOL, "eta2 = {eta2}");
    }

    #[test]
    fn test_cramers_v_independent() {
        // Uniform 2×2 table — perfectly independent
        let table = vec![vec![25.0_f64, 25.0], vec![25.0, 25.0]];
        let v = cramers_v(&table, 2, 2).unwrap();
        assert!(v.abs() < TOL, "V = {v}");
    }

    #[test]
    fn test_cramers_v_perfect() {
        // Diagonal 2×2 — perfect association
        let table = vec![vec![50.0_f64, 0.0], vec![0.0, 50.0]];
        let v = cramers_v(&table, 2, 2).unwrap();
        assert!((v - 1.0).abs() < TOL, "V = {v}");
    }

    #[test]
    fn test_point_biserial_known() {
        // Binary: F F F T T T
        // Cont:   1 2 3 4 5 6
        // M0 = 2, M1 = 5, S_pop = std of 1..6
        let binary = [false, false, false, true, true, true];
        let cont = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let r = point_biserial(&binary, &cont).unwrap();
        // Manual: M1-M0=3, S=sqrt(35/12)≈1.70783, n0*n1/n²=9/36=0.25
        // r = 3/1.70783 * 0.5 ≈ 0.87831
        assert!((r - 0.87831).abs() < 1e-4, "r = {r}");
    }

    #[test]
    fn test_effect_size_interpretation() {
        assert_eq!(
            interpret_cohens_d(0.1_f64),
            EffectSizeInterpretation::Negligible
        );
        assert_eq!(interpret_cohens_d(0.3_f64), EffectSizeInterpretation::Small);
        assert_eq!(
            interpret_cohens_d(0.6_f64),
            EffectSizeInterpretation::Medium
        );
        assert_eq!(interpret_cohens_d(1.0_f64), EffectSizeInterpretation::Large);
        // Negative values
        assert_eq!(
            interpret_cohens_d(-0.9_f64),
            EffectSizeInterpretation::Large
        );
    }
}
