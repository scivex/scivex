//! Confidence interval construction.
//!
//! Provides functions to compute confidence intervals for means and proportions.

use scivex_core::Float;

use crate::descriptive;
use crate::distributions::{Distribution, Normal, StudentT};
use crate::error::{Result, StatsError};

/// A confidence interval with lower and upper bounds.
#[derive(Debug, Clone, Copy)]
pub struct ConfidenceInterval<T: Float> {
    /// Lower bound of the interval.
    pub lower: T,
    /// Upper bound of the interval.
    pub upper: T,
    /// Point estimate (center).
    pub estimate: T,
    /// Confidence level (e.g. 0.95).
    pub confidence: T,
}

/// Compute a confidence interval for the mean of a sample using the t-distribution.
///
/// Uses the Student's t-distribution with `n-1` degrees of freedom.
///
/// # Arguments
/// * `data` — sample data
/// * `confidence` — confidence level (e.g. 0.95 for 95% CI)
pub fn ci_mean<T: Float>(data: &[T], confidence: T) -> Result<ConfidenceInterval<T>> {
    if data.len() < 2 {
        return Err(StatsError::InsufficientData {
            need: 2,
            got: data.len(),
        });
    }
    if confidence <= T::zero() || confidence >= T::one() {
        return Err(StatsError::InvalidParameter {
            name: "confidence",
            reason: "must be in (0, 1)",
        });
    }

    let n = data.len();
    let mean = descriptive::mean(data)?;
    let se = descriptive::std_dev(data)? / T::from_f64((n as f64).sqrt());
    let df = T::from_f64((n - 1) as f64);

    let t_dist = StudentT::new(df)?;
    let alpha = (T::one() - confidence) / T::from_f64(2.0);
    let t_crit = t_dist.ppf(T::one() - alpha)?;

    Ok(ConfidenceInterval {
        lower: mean - t_crit * se,
        upper: mean + t_crit * se,
        estimate: mean,
        confidence,
    })
}

/// Compute a confidence interval for a proportion using the normal approximation.
///
/// # Arguments
/// * `successes` — number of successes
/// * `total` — total number of trials
/// * `confidence` — confidence level (e.g. 0.95)
pub fn ci_proportion<T: Float>(
    successes: usize,
    total: usize,
    confidence: T,
) -> Result<ConfidenceInterval<T>> {
    if total == 0 {
        return Err(StatsError::EmptyInput);
    }
    if successes > total {
        return Err(StatsError::InvalidParameter {
            name: "successes",
            reason: "cannot exceed total",
        });
    }
    if confidence <= T::zero() || confidence >= T::one() {
        return Err(StatsError::InvalidParameter {
            name: "confidence",
            reason: "must be in (0, 1)",
        });
    }

    let p_hat = T::from_f64(successes as f64 / total as f64);
    let n = T::from_f64(total as f64);
    let se = (p_hat * (T::one() - p_hat) / n).sqrt();

    let normal = Normal::new(T::zero(), T::one())?;
    let alpha = (T::one() - confidence) / T::from_f64(2.0);
    let z_crit = normal.ppf(T::one() - alpha)?;

    let lower = (p_hat - z_crit * se).max(T::zero());
    let upper = (p_hat + z_crit * se).min(T::one());

    Ok(ConfidenceInterval {
        lower,
        upper,
        estimate: p_hat,
        confidence,
    })
}

/// Compute a z-based confidence interval for the mean (known population std dev).
///
/// # Arguments
/// * `data` — sample data
/// * `sigma` — known population standard deviation
/// * `confidence` — confidence level (e.g. 0.95)
pub fn ci_mean_z<T: Float>(data: &[T], sigma: T, confidence: T) -> Result<ConfidenceInterval<T>> {
    if data.is_empty() {
        return Err(StatsError::EmptyInput);
    }
    if sigma <= T::zero() {
        return Err(StatsError::InvalidParameter {
            name: "sigma",
            reason: "must be positive",
        });
    }
    if confidence <= T::zero() || confidence >= T::one() {
        return Err(StatsError::InvalidParameter {
            name: "confidence",
            reason: "must be in (0, 1)",
        });
    }

    let n = data.len();
    let mean = descriptive::mean(data)?;
    let se = sigma / T::from_f64((n as f64).sqrt());

    let normal = Normal::new(T::zero(), T::one())?;
    let alpha = (T::one() - confidence) / T::from_f64(2.0);
    let z_crit = normal.ppf(T::one() - alpha)?;

    Ok(ConfidenceInterval {
        lower: mean - z_crit * se,
        upper: mean + z_crit * se,
        estimate: mean,
        confidence,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ci_mean_basic() {
        let data: Vec<f64> = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let ci = ci_mean(&data, 0.95).unwrap();
        assert!((ci.estimate - 6.0).abs() < 1e-10);
        assert!(ci.lower < 6.0);
        assert!(ci.upper > 6.0);
        assert!(ci.lower < ci.upper);
    }

    #[test]
    fn test_ci_mean_contains_true_mean() {
        // Generate data from N(100, 10)
        let data: Vec<f64> = (0..100)
            .map(|i| 100.0 + (f64::from(i) - 50.0) * 0.2)
            .collect();
        let ci = ci_mean(&data, 0.99).unwrap();
        // True mean is 100
        assert!(ci.lower < 100.0 && ci.upper > 100.0);
    }

    #[test]
    fn test_ci_mean_insufficient_data() {
        assert!(ci_mean::<f64>(&[1.0], 0.95).is_err());
    }

    #[test]
    fn test_ci_proportion_basic() {
        let ci = ci_proportion::<f64>(50, 100, 0.95).unwrap();
        assert!((ci.estimate - 0.5).abs() < 1e-10);
        assert!(ci.lower < 0.5);
        assert!(ci.upper > 0.5);
        assert!(ci.lower >= 0.0);
        assert!(ci.upper <= 1.0);
    }

    #[test]
    fn test_ci_proportion_edge() {
        let ci = ci_proportion::<f64>(0, 100, 0.95).unwrap();
        assert!((ci.estimate).abs() < 1e-10);
        assert!((ci.lower).abs() < 1e-10); // clamped to 0

        let ci2 = ci_proportion::<f64>(100, 100, 0.95).unwrap();
        assert!((ci2.estimate - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_ci_mean_z() {
        let data: Vec<f64> = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let ci = ci_mean_z(&data, 3.0, 0.95).unwrap();
        assert!((ci.estimate - 6.0).abs() < 1e-10);
        assert!(ci.lower < 6.0);
        assert!(ci.upper > 6.0);
    }

    #[test]
    fn test_ci_wider_at_higher_confidence() {
        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ci90 = ci_mean(&data, 0.90).unwrap();
        let ci99 = ci_mean(&data, 0.99).unwrap();
        let width90 = ci90.upper - ci90.lower;
        let width99 = ci99.upper - ci99.lower;
        assert!(width99 > width90, "99% CI should be wider than 90% CI");
    }

    #[test]
    fn test_ci_invalid_confidence() {
        let data: Vec<f64> = vec![1.0, 2.0, 3.0];
        assert!(ci_mean(&data, 0.0).is_err());
        assert!(ci_mean(&data, 1.0).is_err());
        assert!(ci_mean(&data, -0.5).is_err());
    }
}
