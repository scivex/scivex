//! Automated time series feature extraction for ML feature engineering.
//!
//! Provides rolling-window statistics extraction similar to tsfresh,
//! computing a configurable set of summary features over sliding windows
//! of a time series.

use scivex_core::Float;

use crate::descriptive;
use crate::error::{Result, StatsError};

/// Which features to extract from each window.
#[derive(Debug, Clone)]
pub enum TsFeature {
    /// Arithmetic mean.
    Mean,
    /// Sample variance (ddof = 1).
    Variance,
    /// Sample standard deviation (ddof = 1).
    StdDev,
    /// Minimum value.
    Min,
    /// Maximum value.
    Max,
    /// Median value.
    Median,
    /// Adjusted Fisher–Pearson skewness.
    Skewness,
    /// Excess kurtosis.
    Kurtosis,
    /// Range (max − min).
    Range,
    /// Arbitrary quantile in [0, 1].
    Quantile(f64),
    /// Sum of elements.
    Sum,
    /// Sum of squared elements.
    AbsEnergy,
    /// Mean of |x[i+1] − x[i]|.
    MeanAbsChange,
    /// Longest consecutive run of values above the window mean.
    LongestStrikeAboveMean,
    /// Autocorrelation at a given lag.
    AutoCorrelation(usize),
    /// Count of elements above the window mean.
    CountAboveMean,
    /// Count of elements below the window mean.
    CountBelowMean,
    /// Slope of a simple linear regression fit.
    LinearTrendSlope,
}

impl TsFeature {
    /// Human-readable name for this feature.
    fn name(&self) -> String {
        match self {
            Self::Mean => "mean".to_string(),
            Self::Variance => "variance".to_string(),
            Self::StdDev => "std_dev".to_string(),
            Self::Min => "min".to_string(),
            Self::Max => "max".to_string(),
            Self::Median => "median".to_string(),
            Self::Skewness => "skewness".to_string(),
            Self::Kurtosis => "kurtosis".to_string(),
            Self::Range => "range".to_string(),
            Self::Quantile(q) => format!("quantile_{q}"),
            Self::Sum => "sum".to_string(),
            Self::AbsEnergy => "abs_energy".to_string(),
            Self::MeanAbsChange => "mean_abs_change".to_string(),
            Self::LongestStrikeAboveMean => "longest_strike_above_mean".to_string(),
            Self::AutoCorrelation(lag) => format!("autocorrelation_lag_{lag}"),
            Self::CountAboveMean => "count_above_mean".to_string(),
            Self::CountBelowMean => "count_below_mean".to_string(),
            Self::LinearTrendSlope => "linear_trend_slope".to_string(),
        }
    }
}

/// Result of time series feature extraction.
#[derive(Debug, Clone)]
pub struct TsFeatureResult<T: Float> {
    /// Names of the extracted features, in order.
    pub feature_names: Vec<String>,
    /// Extracted feature values: `features[window_idx][feature_idx]`.
    pub features: Vec<Vec<T>>,
}

/// Extract features from a time series using rolling windows.
///
/// # Arguments
///
/// * `data` — input time series
/// * `window_size` — size of each rolling window
/// * `step_size` — step between consecutive windows (1 = fully overlapping)
/// * `features` — which features to compute
///
/// # Errors
///
/// Returns [`StatsError::EmptyInput`] if `data` is empty, or
/// [`StatsError::InvalidParameter`] if `window_size` or `step_size` is zero
/// or `window_size` exceeds the data length.
///
/// # Examples
///
/// ```
/// use scivex_stats::ts_features::{extract_features, TsFeature};
///
/// let data: Vec<f64> = (0..20).map(|i| i as f64).collect();
/// let features = vec![TsFeature::Mean, TsFeature::Min, TsFeature::Max];
/// let result = extract_features(&data, 5, 5, &features).unwrap();
///
/// assert_eq!(result.feature_names, vec!["mean", "min", "max"]);
/// assert_eq!(result.features.len(), 4); // (20 - 5) / 5 + 1 = 4 windows
///
/// // First window [0,1,2,3,4]: mean=2.0, min=0.0, max=4.0
/// assert!((result.features[0][0] - 2.0).abs() < 1e-10);
/// assert!((result.features[0][1] - 0.0).abs() < 1e-10);
/// assert!((result.features[0][2] - 4.0).abs() < 1e-10);
/// ```
pub fn extract_features<T: Float>(
    data: &[T],
    window_size: usize,
    step_size: usize,
    features: &[TsFeature],
) -> Result<TsFeatureResult<T>> {
    if data.is_empty() {
        return Err(StatsError::EmptyInput);
    }
    if window_size == 0 {
        return Err(StatsError::InvalidParameter {
            name: "window_size",
            reason: "must be greater than zero",
        });
    }
    if step_size == 0 {
        return Err(StatsError::InvalidParameter {
            name: "step_size",
            reason: "must be greater than zero",
        });
    }
    if window_size > data.len() {
        return Err(StatsError::InsufficientData {
            need: window_size,
            got: data.len(),
        });
    }
    if features.is_empty() {
        return Err(StatsError::InvalidParameter {
            name: "features",
            reason: "must specify at least one feature",
        });
    }

    let feature_names: Vec<String> = features.iter().map(TsFeature::name).collect();
    let num_windows = (data.len() - window_size) / step_size + 1;
    let mut all_features = Vec::with_capacity(num_windows);

    for w in 0..num_windows {
        let start = w * step_size;
        let window = &data[start..start + window_size];
        let mut row = Vec::with_capacity(features.len());

        for feat in features {
            let val = compute_feature(window, feat)?;
            row.push(val);
        }
        all_features.push(row);
    }

    Ok(TsFeatureResult {
        feature_names,
        features: all_features,
    })
}

/// Extract a comprehensive default set of features with step size 1.
///
/// The default set includes: Mean, Variance, StdDev, Min, Max, Median,
/// Skewness, Kurtosis, Range, Sum, AbsEnergy, MeanAbsChange, and
/// LinearTrendSlope.
///
/// # Errors
///
/// Returns an error if `data` is empty or `window_size` is invalid.
pub fn extract_default_features<T: Float>(
    data: &[T],
    window_size: usize,
) -> Result<TsFeatureResult<T>> {
    let default_features = vec![
        TsFeature::Mean,
        TsFeature::Variance,
        TsFeature::StdDev,
        TsFeature::Min,
        TsFeature::Max,
        TsFeature::Median,
        TsFeature::Skewness,
        TsFeature::Kurtosis,
        TsFeature::Range,
        TsFeature::Sum,
        TsFeature::AbsEnergy,
        TsFeature::MeanAbsChange,
        TsFeature::LinearTrendSlope,
    ];
    extract_features(data, window_size, 1, &default_features)
}

/// Compute a single feature for a given window.
#[allow(clippy::too_many_lines)]
fn compute_feature<T: Float>(window: &[T], feature: &TsFeature) -> Result<T> {
    match feature {
        TsFeature::Mean => descriptive::mean(window),

        TsFeature::Variance => {
            if window.len() < 2 {
                return Ok(T::from_f64(0.0));
            }
            descriptive::variance(window)
        }

        TsFeature::StdDev => {
            if window.len() < 2 {
                return Ok(T::from_f64(0.0));
            }
            descriptive::std_dev(window)
        }

        TsFeature::Min => {
            let mut min = window[0];
            for &v in &window[1..] {
                if v < min {
                    min = v;
                }
            }
            Ok(min)
        }

        TsFeature::Max => {
            let mut max = window[0];
            for &v in &window[1..] {
                if v > max {
                    max = v;
                }
            }
            Ok(max)
        }

        TsFeature::Median => descriptive::median(window),

        TsFeature::Skewness => {
            if window.len() < 3 {
                return Ok(T::from_f64(0.0));
            }
            descriptive::skewness(window)
        }

        TsFeature::Kurtosis => {
            if window.len() < 4 {
                return Ok(T::from_f64(0.0));
            }
            descriptive::kurtosis(window)
        }

        TsFeature::Range => {
            let mut min = window[0];
            let mut max = window[0];
            for &v in &window[1..] {
                if v < min {
                    min = v;
                }
                if v > max {
                    max = v;
                }
            }
            Ok(max - min)
        }

        TsFeature::Quantile(q) => descriptive::quantile(window, T::from_f64(*q)),

        TsFeature::Sum => {
            let s: T = window.iter().copied().sum();
            Ok(s)
        }

        TsFeature::AbsEnergy => {
            let e: T = window.iter().map(|&x| x * x).sum();
            Ok(e)
        }

        TsFeature::MeanAbsChange => {
            if window.len() < 2 {
                return Ok(T::from_f64(0.0));
            }
            let mut total = T::from_f64(0.0);
            for i in 1..window.len() {
                total += (window[i] - window[i - 1]).abs();
            }
            Ok(total / T::from_f64((window.len() - 1) as f64))
        }

        TsFeature::LongestStrikeAboveMean => {
            let m = descriptive::mean(window)?;
            let mut longest = 0usize;
            let mut current = 0usize;
            for &v in window {
                if v > m {
                    current += 1;
                    if current > longest {
                        longest = current;
                    }
                } else {
                    current = 0;
                }
            }
            Ok(T::from_f64(longest as f64))
        }

        TsFeature::AutoCorrelation(lag) => {
            let lag = *lag;
            if lag == 0 || lag >= window.len() {
                return Ok(T::from_f64(0.0));
            }
            let m = descriptive::mean(window)?;
            let mut num = T::from_f64(0.0);
            let mut den = T::from_f64(0.0);
            for &v in window {
                den += (v - m) * (v - m);
            }
            if den == T::from_f64(0.0) {
                return Ok(T::from_f64(0.0));
            }
            for i in 0..window.len() - lag {
                num += (window[i] - m) * (window[i + lag] - m);
            }
            Ok(num / den)
        }

        TsFeature::CountAboveMean => {
            let m = descriptive::mean(window)?;
            let count = window.iter().filter(|&&v| v > m).count();
            Ok(T::from_f64(count as f64))
        }

        TsFeature::CountBelowMean => {
            let m = descriptive::mean(window)?;
            let count = window.iter().filter(|&&v| v < m).count();
            Ok(T::from_f64(count as f64))
        }

        TsFeature::LinearTrendSlope => {
            let n = window.len();
            if n < 2 {
                return Ok(T::from_f64(0.0));
            }
            // Fit y = a + b*x via least squares where x = 0, 1, ..., n-1
            let nf = T::from_f64(n as f64);
            // sum_x = n*(n-1)/2, sum_x2 = n*(n-1)*(2n-1)/6
            let sum_x = T::from_f64((n * (n - 1)) as f64 / 2.0);
            let sum_x2 = T::from_f64((n * (n - 1) * (2 * n - 1)) as f64 / 6.0);
            let sum_y: T = window.iter().copied().sum();
            let mut sum_xy = T::from_f64(0.0);
            for (i, &y) in window.iter().enumerate() {
                sum_xy += T::from_f64(i as f64) * y;
            }
            let denom = nf * sum_x2 - sum_x * sum_x;
            if denom == T::from_f64(0.0) {
                return Ok(T::from_f64(0.0));
            }
            let slope = (nf * sum_xy - sum_x * sum_y) / denom;
            Ok(slope)
        }
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
    fn test_extract_constant_series() {
        let data = vec![5.0_f64; 20];
        let features = vec![
            TsFeature::Mean,
            TsFeature::Variance,
            TsFeature::LinearTrendSlope,
        ];
        let result = extract_features(&data, 10, 10, &features).unwrap();
        assert_eq!(result.features.len(), 2); // (20-10)/10 + 1 = 2

        for row in &result.features {
            assert!((row[0] - 5.0).abs() < 1e-10, "mean should be 5.0");
            assert!((row[1] - 0.0).abs() < 1e-10, "variance should be 0.0");
            assert!((row[2] - 0.0).abs() < 1e-10, "slope should be 0.0");
        }
    }

    #[test]
    fn test_extract_linear_trend() {
        // Linearly increasing: 0, 1, 2, ..., 19
        let data: Vec<f64> = (0..20).map(f64::from).collect();
        let features = vec![
            TsFeature::Mean,
            TsFeature::LinearTrendSlope,
            TsFeature::Min,
            TsFeature::Max,
        ];
        let result = extract_features(&data, 10, 10, &features).unwrap();

        // First window [0..10]: mean = 4.5, slope = 1.0, min = 0, max = 9
        let row = &result.features[0];
        assert!((row[0] - 4.5).abs() < 1e-10, "mean of 0..9 should be 4.5");
        assert!((row[1] - 1.0).abs() < 1e-10, "slope should be 1.0");
        assert!((row[2] - 0.0).abs() < 1e-10, "min should be 0.0");
        assert!((row[3] - 9.0).abs() < 1e-10, "max should be 9.0");
    }

    #[test]
    fn test_extract_features_window_count() {
        let data: Vec<f64> = (0..100).map(f64::from).collect();
        let window_size = 10;
        let step_size = 3;
        let features = vec![TsFeature::Mean];
        let result = extract_features(&data, window_size, step_size, &features).unwrap();

        let expected_windows = (100 - window_size) / step_size + 1;
        assert_eq!(result.features.len(), expected_windows);
    }

    #[test]
    fn test_extract_default_features() {
        let data: Vec<f64> = (0..50).map(|i| f64::from(i).sin()).collect();
        let result = extract_default_features(&data, 10).unwrap();

        // Default set has 13 features
        assert_eq!(result.feature_names.len(), 13);
        assert_eq!(result.features[0].len(), 13);

        // Verify expected feature names
        assert_eq!(result.feature_names[0], "mean");
        assert_eq!(result.feature_names[1], "variance");
        assert_eq!(result.feature_names[12], "linear_trend_slope");

        // Number of windows: (50 - 10) / 1 + 1 = 41
        assert_eq!(result.features.len(), 41);
    }

    #[test]
    fn test_extract_empty_input() {
        let data: Vec<f64> = vec![];
        let features = vec![TsFeature::Mean];
        let result = extract_features(&data, 5, 1, &features);
        assert!(result.is_err());
        match result.unwrap_err() {
            StatsError::EmptyInput => {}
            other => panic!("expected EmptyInput, got {other:?}"),
        }
    }
}
