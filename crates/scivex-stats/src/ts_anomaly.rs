//! Time series anomaly detection.
//!
//! Provides multiple methods for detecting anomalies in temporal data:
//!
//! - [`zscore_anomaly`] — sliding-window z-score detection
//! - [`seasonal_anomaly`] — detection via seasonal decomposition residuals
//! - [`isolation_forest_anomaly`] — isolation forest adapted for time series
//! - [`ewma_anomaly`] — exponentially weighted moving average detection

use scivex_core::Float;
use scivex_core::random::Rng;

use crate::descriptive;
use crate::error::{Result, StatsError};

// ── Result type ───────────────────────────────────────────────────────

/// Result of anomaly detection on a time series.
#[derive(Debug, Clone)]
pub struct AnomalyResult<T: Float> {
    /// Indices of data points detected as anomalies.
    pub anomaly_indices: Vec<usize>,
    /// Anomaly score for each data point (higher = more anomalous).
    pub scores: Vec<T>,
    /// Threshold used for detection.
    pub threshold: T,
}

// ── Z-Score anomaly detection ─────────────────────────────────────────

/// Detect anomalies using z-score with a sliding window.
///
/// For each data point, the mean and standard deviation are computed over a
/// centered window of size `window_size`. Points where `|z-score| > threshold`
/// are flagged as anomalies.
///
/// # Errors
///
/// - [`StatsError::EmptyInput`] if `data` is empty.
/// - [`StatsError::InvalidParameter`] if `window_size < 3` or `threshold <= 0`.
///
/// # Examples
///
/// ```
/// # use scivex_stats::ts_anomaly::zscore_anomaly;
/// let mut data: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
/// data[50] = 10.0; // inject spike
/// let result = zscore_anomaly(&data, 11, 3.0).unwrap();
/// assert!(result.anomaly_indices.contains(&50));
/// ```
pub fn zscore_anomaly<T: Float>(
    data: &[T],
    window_size: usize,
    threshold: T,
) -> Result<AnomalyResult<T>> {
    if data.is_empty() {
        return Err(StatsError::EmptyInput);
    }
    if window_size < 3 {
        return Err(StatsError::InvalidParameter {
            name: "window_size",
            reason: "must be >= 3",
        });
    }
    if threshold <= T::zero() {
        return Err(StatsError::InvalidParameter {
            name: "threshold",
            reason: "must be > 0",
        });
    }

    let n = data.len();
    let half = window_size / 2;
    let mut scores = Vec::with_capacity(n);
    let mut anomaly_indices = Vec::new();

    for i in 0..n {
        let start = i.saturating_sub(half);
        let end = (i + half + 1).min(n);
        let window = &data[start..end];

        let m = descriptive::mean(window)?;
        let s = descriptive::std_dev(window).unwrap_or(T::zero());

        let z = if s > T::zero() {
            (data[i] - m) / s
        } else {
            T::zero()
        };

        let abs_z = if z < T::zero() { T::zero() - z } else { z };
        scores.push(abs_z);

        if abs_z > threshold {
            anomaly_indices.push(i);
        }
    }

    Ok(AnomalyResult {
        anomaly_indices,
        scores,
        threshold,
    })
}

// ── Seasonal anomaly detection ────────────────────────────────────────

/// Detect anomalies using seasonal decomposition residuals.
///
/// Performs additive seasonal decomposition via
/// [`crate::timeseries::seasonal_decompose`], then flags points where
/// `|residual| > n_sigma * std(residuals)`.
///
/// # Errors
///
/// - [`StatsError::EmptyInput`] if `data` is empty.
/// - [`StatsError::InvalidParameter`] if `period < 2` or `n_sigma <= 0`.
/// - [`StatsError::InsufficientData`] if `data.len() < 2 * period`.
pub fn seasonal_anomaly<T: Float>(
    data: &[T],
    period: usize,
    n_sigma: T,
) -> Result<AnomalyResult<T>> {
    if data.is_empty() {
        return Err(StatsError::EmptyInput);
    }
    if n_sigma <= T::zero() {
        return Err(StatsError::InvalidParameter {
            name: "n_sigma",
            reason: "must be > 0",
        });
    }

    let decomp = crate::timeseries::seasonal_decompose(data, period)?;
    let residuals = &decomp.residual;

    // Filter out NaN residuals for std computation
    let finite_residuals: Vec<T> = residuals
        .iter()
        .copied()
        .filter(|r| {
            let v = r.to_f64();
            v.is_finite()
        })
        .collect();

    if finite_residuals.is_empty() {
        return Err(StatsError::InsufficientData { need: 1, got: 0 });
    }

    let residual_std = descriptive::std_dev(&finite_residuals)?;
    let thresh = n_sigma * residual_std;

    let n = data.len();
    let mut scores = Vec::with_capacity(n);
    let mut anomaly_indices = Vec::new();

    for (i, &r) in residuals.iter().enumerate() {
        let r_f64 = r.to_f64();
        if !r_f64.is_finite() {
            // Trend edges have NaN residuals; treat as non-anomalous
            scores.push(T::zero());
            continue;
        }
        let abs_r = if r < T::zero() { T::zero() - r } else { r };
        scores.push(abs_r);

        if abs_r > thresh {
            anomaly_indices.push(i);
        }
    }

    Ok(AnomalyResult {
        anomaly_indices,
        scores,
        threshold: thresh,
    })
}

// ── Isolation Forest anomaly detection ────────────────────────────────

/// Internal node structure for isolation trees.
enum IsoNode {
    Internal {
        feature: usize,
        split: f64,
        left: Box<IsoNode>,
        right: Box<IsoNode>,
    },
    Leaf {
        size: usize,
    },
}

/// Average path length of an unsuccessful search in a BST with `n` elements.
fn avg_path_length(n: usize) -> f64 {
    if n <= 1 {
        return 0.0;
    }
    let n_f = n as f64;
    // c(n) = 2 * H(n-1) - 2*(n-1)/n, where H is the harmonic number
    // H(k) ≈ ln(k) + 0.5772156649 (Euler-Mascheroni constant)
    let h = (n_f - 1.0).ln() + 0.577_215_664_9;
    2.0 * h - 2.0 * (n_f - 1.0) / n_f
}

/// Build a single isolation tree from feature vectors.
fn build_iso_tree(
    rng: &mut Rng,
    data: &[Vec<f64>],
    indices: &[usize],
    depth: usize,
    max_depth: usize,
    n_features: usize,
) -> IsoNode {
    if indices.len() <= 1 || depth >= max_depth {
        return IsoNode::Leaf {
            size: indices.len(),
        };
    }

    // Pick a random feature
    let feature = (rng.next_u64() as usize) % n_features;

    // Find min/max of this feature among selected indices
    let mut min_val = f64::INFINITY;
    let mut max_val = f64::NEG_INFINITY;
    for &idx in indices {
        let v = data[idx][feature];
        if v < min_val {
            min_val = v;
        }
        if v > max_val {
            max_val = v;
        }
    }

    if (max_val - min_val).abs() < 1e-15 {
        return IsoNode::Leaf {
            size: indices.len(),
        };
    }

    // Random split value between min and max
    let split = min_val + rng.next_f64() * (max_val - min_val);

    let mut left_idx = Vec::new();
    let mut right_idx = Vec::new();
    for &idx in indices {
        if data[idx][feature] < split {
            left_idx.push(idx);
        } else {
            right_idx.push(idx);
        }
    }

    // Avoid degenerate splits
    if left_idx.is_empty() || right_idx.is_empty() {
        return IsoNode::Leaf {
            size: indices.len(),
        };
    }

    IsoNode::Internal {
        feature,
        split,
        left: Box::new(build_iso_tree(
            rng,
            data,
            &left_idx,
            depth + 1,
            max_depth,
            n_features,
        )),
        right: Box::new(build_iso_tree(
            rng,
            data,
            &right_idx,
            depth + 1,
            max_depth,
            n_features,
        )),
    }
}

/// Compute the path length for a sample in an isolation tree.
fn path_length(node: &IsoNode, sample: &[f64], depth: usize) -> f64 {
    match node {
        IsoNode::Leaf { size } => depth as f64 + avg_path_length(*size),
        IsoNode::Internal {
            feature,
            split,
            left,
            right,
        } => {
            if sample[*feature] < *split {
                path_length(left, sample, depth + 1)
            } else {
                path_length(right, sample, depth + 1)
            }
        }
    }
}

/// Detect anomalies using Isolation Forest adapted for time series.
///
/// Sliding windows are used to extract feature vectors `[mean, std, range, slope]`
/// from the data, and an isolation forest is run on these vectors.
/// Points in the top `contamination` fraction by anomaly score are flagged.
///
/// # Errors
///
/// - [`StatsError::EmptyInput`] if `data` is empty.
/// - [`StatsError::InvalidParameter`] if `window_size < 2`, `n_trees == 0`,
///   or `contamination` is not in `(0, 0.5]`.
/// - [`StatsError::InsufficientData`] if `data.len() < window_size`.
#[allow(clippy::too_many_lines)]
pub fn isolation_forest_anomaly<T: Float>(
    data: &[T],
    window_size: usize,
    n_trees: usize,
    contamination: T,
    seed: u64,
) -> Result<AnomalyResult<T>> {
    if data.is_empty() {
        return Err(StatsError::EmptyInput);
    }
    if window_size < 2 {
        return Err(StatsError::InvalidParameter {
            name: "window_size",
            reason: "must be >= 2",
        });
    }
    if n_trees == 0 {
        return Err(StatsError::InvalidParameter {
            name: "n_trees",
            reason: "must be >= 1",
        });
    }
    let cont_f64 = contamination.to_f64();
    if cont_f64 <= 0.0 || cont_f64 > 0.5 {
        return Err(StatsError::InvalidParameter {
            name: "contamination",
            reason: "must be in (0.0, 0.5]",
        });
    }
    let n = data.len();
    if n < window_size {
        return Err(StatsError::InsufficientData {
            need: window_size,
            got: n,
        });
    }

    let n_windows = n - window_size + 1;
    let n_features = 4; // mean, std, range, slope

    // Extract window features
    let mut features: Vec<Vec<f64>> = Vec::with_capacity(n_windows);
    for i in 0..n_windows {
        let window = &data[i..i + window_size];
        let m = descriptive::mean(window)?.to_f64();
        let s = descriptive::std_dev(window).unwrap_or(T::zero()).to_f64();

        let mut min_v = window[0].to_f64();
        let mut max_v = window[0].to_f64();
        for &v in &window[1..] {
            let vf = v.to_f64();
            if vf < min_v {
                min_v = vf;
            }
            if vf > max_v {
                max_v = vf;
            }
        }
        let range = max_v - min_v;

        // Simple slope: (last - first) / (window_size - 1)
        let slope =
            (window[window_size - 1].to_f64() - window[0].to_f64()) / (window_size - 1) as f64;

        features.push(vec![m, s, range, slope]);
    }

    // Build isolation trees
    let max_depth = ((n_windows as f64).log2().ceil() as usize).max(1);
    let mut rng = Rng::new(seed);

    let mut trees = Vec::with_capacity(n_trees);
    for _ in 0..n_trees {
        // Subsample: use min(256, n_windows) samples per tree
        let sample_size = n_windows.min(256);
        let mut sample_indices: Vec<usize> = Vec::with_capacity(sample_size);
        for _ in 0..sample_size {
            let idx = (rng.next_u64() as usize) % n_windows;
            sample_indices.push(idx);
        }
        trees.push(build_iso_tree(
            &mut rng,
            &features,
            &sample_indices,
            0,
            max_depth,
            n_features,
        ));
    }

    // Compute anomaly scores for all windows
    let c_n = avg_path_length(n_windows);
    let mut window_scores: Vec<f64> = Vec::with_capacity(n_windows);
    for feat in &features {
        let avg_h: f64 = trees
            .iter()
            .map(|tree| path_length(tree, feat, 0))
            .sum::<f64>()
            / n_trees as f64;

        // s(x) = 2^(-E[h(x)] / c(n))
        let score = if c_n > 0.0 {
            f64::powf(2.0, -avg_h / c_n)
        } else {
            0.5
        };
        window_scores.push(score);
    }

    // Map window scores back to individual data points.
    // Each point may belong to multiple windows; take the max score.
    let mut point_scores = vec![0.0_f64; n];
    for (w, &ws) in window_scores.iter().enumerate() {
        for ps in &mut point_scores[w..w + window_size] {
            if ws > *ps {
                *ps = ws;
            }
        }
    }

    // Determine threshold from contamination rate
    let mut sorted_scores = point_scores.clone();
    sorted_scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    let cutoff_idx = ((1.0 - cont_f64) * n as f64) as usize;
    let cutoff_idx = cutoff_idx.min(n - 1);
    let threshold_f64 = sorted_scores[cutoff_idx];

    let mut anomaly_indices = Vec::new();
    let scores: Vec<T> = point_scores
        .iter()
        .enumerate()
        .map(|(i, &s)| {
            if s >= threshold_f64 {
                anomaly_indices.push(i);
            }
            T::from_f64(s)
        })
        .collect();

    Ok(AnomalyResult {
        anomaly_indices,
        scores,
        threshold: T::from_f64(threshold_f64),
    })
}

// ── EWMA anomaly detection ───────────────────────────────────────────

/// Detect anomalies using exponentially weighted moving average (EWMA).
///
/// Computes an EWMA prediction and flags points that deviate more than
/// `n_sigma` standard deviations from the prediction, where the variance
/// is also exponentially weighted.
///
/// # Errors
///
/// - [`StatsError::EmptyInput`] if `data` is empty.
/// - [`StatsError::InvalidParameter`] if `alpha` is not in `(0, 1)` or
///   `n_sigma <= 0`.
pub fn ewma_anomaly<T: Float>(data: &[T], alpha: T, n_sigma: T) -> Result<AnomalyResult<T>> {
    if data.is_empty() {
        return Err(StatsError::EmptyInput);
    }
    let alpha_f64 = alpha.to_f64();
    if alpha_f64 <= 0.0 || alpha_f64 >= 1.0 {
        return Err(StatsError::InvalidParameter {
            name: "alpha",
            reason: "must be in (0.0, 1.0)",
        });
    }
    if n_sigma <= T::zero() {
        return Err(StatsError::InvalidParameter {
            name: "n_sigma",
            reason: "must be > 0",
        });
    }

    let n = data.len();
    let one = T::one();
    let one_minus_alpha = one - alpha;

    let mut scores = Vec::with_capacity(n);
    let mut anomaly_indices = Vec::new();

    // Initialize EWMA and variance
    let mut ewma = data[0];
    let mut ewma_var = T::zero();

    // First point: no prediction available, score is 0
    scores.push(T::zero());

    for (i, &val) in data.iter().enumerate().skip(1) {
        // Prediction error: deviation from previous EWMA
        let error = val - ewma;
        let abs_error = if error < T::zero() {
            T::zero() - error
        } else {
            error
        };

        // Compute score using the *previous* variance (before this point
        // inflates it), so that a sudden shift is not masked by its own
        // contribution to the running variance.
        let std = ewma_var.sqrt();
        let score = if std > T::zero() {
            abs_error / std
        } else if abs_error > T::zero() {
            // Variance is zero but there is a deviation — maximally anomalous.
            T::from_f64(1e12)
        } else {
            T::zero()
        };

        scores.push(score);

        if score > n_sigma {
            anomaly_indices.push(i);
        }

        // Update EWMA variance *after* scoring
        ewma_var = alpha * error * error + one_minus_alpha * ewma_var;

        // Update EWMA
        ewma = alpha * val + one_minus_alpha * ewma;
    }

    Ok(AnomalyResult {
        anomaly_indices,
        scores,
        threshold: n_sigma,
    })
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zscore_anomaly_spike() {
        // Smooth sine wave with a spike
        let mut data: Vec<f64> = (0..200).map(|i| (f64::from(i) * 0.1).sin()).collect();
        data[100] = 20.0; // obvious spike

        let result = zscore_anomaly(&data, 21, 3.0).unwrap();
        assert!(
            result.anomaly_indices.contains(&100),
            "spike at index 100 should be detected, got indices: {:?}",
            result.anomaly_indices
        );
        assert_eq!(result.scores.len(), data.len());
        assert!((result.threshold - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_seasonal_anomaly() {
        // Seasonal data with period 10
        let mut data: Vec<f64> = (0..100)
            .map(|i| {
                let seasonal = (f64::from(i % 10) * 0.628).sin(); // 2π/10 ≈ 0.628
                let trend = f64::from(i) * 0.01;
                trend + seasonal
            })
            .collect();
        // Inject anomaly: large deviation in the middle
        data[55] += 10.0;

        let result = seasonal_anomaly(&data, 10, 3.0).unwrap();
        assert!(
            result.anomaly_indices.contains(&55),
            "anomaly at index 55 should be detected, got indices: {:?}",
            result.anomaly_indices
        );
    }

    #[test]
    fn test_isolation_forest_anomaly() {
        // Normal data with a few obvious outliers
        let mut data: Vec<f64> = (0..200).map(|i| (f64::from(i) * 0.05).sin()).collect();
        // Insert extreme outliers
        data[50] = 50.0;
        data[150] = -50.0;

        let result = isolation_forest_anomaly(&data, 5, 100, 0.05, 42).unwrap();
        // With contamination=0.05, we expect ~10 flagged points.
        // The extreme outliers at 50 and 150 should be among them.
        assert!(
            result.anomaly_indices.contains(&50),
            "outlier at 50 should be detected, got: {:?}",
            result.anomaly_indices
        );
        assert!(
            result.anomaly_indices.contains(&150),
            "outlier at 150 should be detected, got: {:?}",
            result.anomaly_indices
        );
        assert_eq!(result.scores.len(), data.len());
    }

    #[test]
    fn test_ewma_anomaly_level_shift() {
        // Constant signal with a sudden level shift
        let mut data: Vec<f64> = vec![1.0; 100];
        // Shift level at index 50
        for v in data.iter_mut().skip(50) {
            *v = 10.0;
        }

        let result = ewma_anomaly(&data, 0.3, 2.0).unwrap();
        // The point at index 50 (and possibly a few after) should be flagged
        assert!(
            result.anomaly_indices.contains(&50),
            "level shift at index 50 should be detected, got: {:?}",
            result.anomaly_indices
        );
        assert_eq!(result.scores.len(), data.len());
    }

    #[test]
    fn test_anomaly_empty_input() {
        let empty: &[f64] = &[];
        assert!(zscore_anomaly(empty, 5, 3.0).is_err());
        assert!(seasonal_anomaly(empty, 4, 3.0).is_err());
        assert!(isolation_forest_anomaly(empty, 5, 10, 0.1, 0).is_err());
        assert!(ewma_anomaly(empty, 0.3, 3.0).is_err());
    }
}
