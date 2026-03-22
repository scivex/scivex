//! Univariate scoring functions for feature selection.

use scivex_core::{Float, Tensor};

use crate::error::{MlError, Result};

/// Compute chi-squared statistics between each feature and the target.
///
/// For each feature column, a contingency table is built between binned feature
/// values and target classes, and the chi-squared statistic is computed.
///
/// `x` must have shape `[n_samples, n_features]` with **non-negative** values.
/// `y` must have shape `[n_samples]` with discrete class labels.
///
/// Returns a `Vec<T>` of length `n_features` with the chi-squared score for
/// each feature.
///
/// # Examples
///
/// ```
/// # use scivex_core::prelude::*;
/// # use scivex_ml::feature_selection::chi2;
/// let x = Tensor::from_vec(vec![1.0_f64, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0], vec![4, 2]).unwrap();
/// let y = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0], vec![4]).unwrap();
/// let scores = chi2(&x, &y).unwrap();
/// assert_eq!(scores.len(), 2);
/// ```
pub fn chi2<T: Float>(x: &Tensor<T>, y: &Tensor<T>) -> Result<Vec<T>> {
    let (n, p) = matrix_shape(x)?;
    check_y(y, n)?;

    let x_data = x.as_slice();
    let y_data = y.as_slice();

    // Discover unique classes in y
    let classes = unique_sorted(y_data);
    if classes.is_empty() {
        return Err(MlError::EmptyInput);
    }
    let n_classes = classes.len();

    // For each feature, compute the chi2 statistic using the observed and
    // expected frequency approach. We sum feature values per class to form
    // an (n_classes x 1) observed table, treating each feature independently.
    //
    // The chi2 statistic between feature j and target y is computed as:
    //   chi2_j = n * (sum_c (observed_cj^2 / (row_sum_c * col_sum_j)) - 1)
    // which is equivalent to the standard chi2 on the contingency table.

    let mut scores = Vec::with_capacity(p);

    for j in 0..p {
        // observed_c = sum of feature j values for samples in class c
        let mut observed = vec![T::zero(); n_classes];
        let mut class_counts = vec![0usize; n_classes];

        for i in 0..n {
            let val = x_data[i * p + j];
            if val < T::zero() {
                return Err(MlError::InvalidParameter {
                    name: "x",
                    reason: "chi2 requires non-negative feature values",
                });
            }
            let ci = class_index(&classes, y_data[i]);
            observed[ci] += val;
            class_counts[ci] += 1;
        }

        // Column total
        let col_total: T = observed.iter().copied().fold(T::zero(), |a, b| a + b);

        if col_total == T::zero() {
            scores.push(T::zero());
            continue;
        }

        // Row totals (count of samples per class as T)
        let row_totals: Vec<T> = class_counts.iter().map(|&c| T::from_usize(c)).collect();

        let n_t = T::from_usize(n);

        // chi2 = sum_c (observed_c - expected_c)^2 / expected_c
        // where expected_c = row_total_c * col_total / n
        let mut chi2_stat = T::zero();
        for c in 0..n_classes {
            let expected = row_totals[c] * col_total / n_t;
            if expected > T::zero() {
                let diff = observed[c] - expected;
                chi2_stat += diff * diff / expected;
            }
        }

        scores.push(chi2_stat);
    }

    Ok(scores)
}

/// Compute one-way ANOVA F-values between each feature and the target.
///
/// For each feature, the between-group variance is compared with the
/// within-group variance, where groups are defined by unique values in `y`.
///
/// `x` must have shape `[n_samples, n_features]`.
/// `y` must have shape `[n_samples]` with discrete class labels.
///
/// Returns a `Vec<T>` of length `n_features` with the F-statistic for each
/// feature.
///
/// # Examples
///
/// ```
/// # use scivex_core::prelude::*;
/// # use scivex_ml::feature_selection::f_classif;
/// let x = Tensor::from_vec(
///     vec![1.0_f64, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
///     vec![4, 2],
/// ).unwrap();
/// let y = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0], vec![4]).unwrap();
/// let scores = f_classif(&x, &y).unwrap();
/// assert_eq!(scores.len(), 2);
/// ```
pub fn f_classif<T: Float>(x: &Tensor<T>, y: &Tensor<T>) -> Result<Vec<T>> {
    let (n, p) = matrix_shape(x)?;
    check_y(y, n)?;

    let x_data = x.as_slice();
    let y_data = y.as_slice();

    let classes = unique_sorted(y_data);
    let n_classes = classes.len();
    if n_classes < 2 {
        return Err(MlError::InvalidParameter {
            name: "y",
            reason: "f_classif requires at least 2 classes",
        });
    }

    let n_t = T::from_usize(n);
    let k_t = T::from_usize(n_classes);

    let mut scores = Vec::with_capacity(p);

    for j in 0..p {
        // Grand mean for feature j
        let mut grand_sum = T::zero();
        for i in 0..n {
            grand_sum += x_data[i * p + j];
        }
        let grand_mean = grand_sum / n_t;

        // Per-class sums and counts
        let mut class_sums = vec![T::zero(); n_classes];
        let mut class_counts = vec![0usize; n_classes];
        for i in 0..n {
            let ci = class_index(&classes, y_data[i]);
            class_sums[ci] += x_data[i * p + j];
            class_counts[ci] += 1;
        }

        // Between-group sum of squares
        let mut ss_between = T::zero();
        for c in 0..n_classes {
            if class_counts[c] == 0 {
                continue;
            }
            let class_mean = class_sums[c] / T::from_usize(class_counts[c]);
            let diff = class_mean - grand_mean;
            ss_between += T::from_usize(class_counts[c]) * diff * diff;
        }

        // Within-group sum of squares
        let mut ss_within = T::zero();
        for i in 0..n {
            let ci = class_index(&classes, y_data[i]);
            let class_mean = class_sums[ci] / T::from_usize(class_counts[ci]);
            let diff = x_data[i * p + j] - class_mean;
            ss_within += diff * diff;
        }

        // F = (SS_between / (k-1)) / (SS_within / (n-k))
        let df_between = k_t - T::one();
        let df_within = n_t - k_t;

        let f_stat = if ss_within > T::zero() && df_within > T::zero() {
            (ss_between / df_between) / (ss_within / df_within)
        } else {
            T::zero()
        };

        scores.push(f_stat);
    }

    Ok(scores)
}

// ── helpers ──

fn matrix_shape<T: Float>(x: &Tensor<T>) -> Result<(usize, usize)> {
    let s = x.shape();
    if s.len() != 2 {
        return Err(MlError::DimensionMismatch {
            expected: 2,
            got: s.len(),
        });
    }
    if s[0] == 0 {
        return Err(MlError::EmptyInput);
    }
    Ok((s[0], s[1]))
}

fn check_y<T: Float>(y: &Tensor<T>, n: usize) -> Result<()> {
    if y.ndim() != 1 || y.shape()[0] != n {
        return Err(MlError::DimensionMismatch {
            expected: n,
            got: y.shape()[0],
        });
    }
    Ok(())
}

/// Return sorted unique values from a slice (using to_f64 for comparison).
fn unique_sorted<T: Float>(data: &[T]) -> Vec<T> {
    let mut vals: Vec<f64> = data.iter().map(|v| v.to_f64()).collect();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    vals.dedup();
    vals.into_iter().map(T::from_f64).collect()
}

/// Find the index of `val` in `classes` (using to_f64 comparison).
fn class_index<T: Float>(classes: &[T], val: T) -> usize {
    let v = val.to_f64();
    classes
        .iter()
        .position(|c| (c.to_f64() - v).abs() < f64::EPSILON)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chi2_basic() {
        // Feature 0 perfectly separates classes, feature 1 is constant
        let x =
            Tensor::from_vec(vec![1.0_f64, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0], vec![4, 2]).unwrap();
        let y = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0], vec![4]).unwrap();
        let scores = chi2(&x, &y).unwrap();
        assert_eq!(scores.len(), 2);
        // Feature 0 should have a higher chi2 than feature 1
        assert!(scores[0] > scores[1], "chi2 scores: {scores:?}");
    }

    #[test]
    fn test_chi2_negative_values_error() {
        let x = Tensor::from_vec(vec![-1.0_f64, 2.0], vec![1, 2]).unwrap();
        let y = Tensor::from_vec(vec![0.0], vec![1]).unwrap();
        assert!(chi2(&x, &y).is_err());
    }

    #[test]
    fn test_f_classif_basic() {
        // Feature 0: class 0 = [1, 2], class 1 = [10, 11] => high F
        // Feature 1: class 0 = [5, 5], class 1 = [5, 5]   => F = 0
        let x = Tensor::from_vec(
            vec![1.0_f64, 5.0, 2.0, 5.0, 10.0, 5.0, 11.0, 5.0],
            vec![4, 2],
        )
        .unwrap();
        let y = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0], vec![4]).unwrap();
        let scores = f_classif(&x, &y).unwrap();
        assert_eq!(scores.len(), 2);
        assert!(scores[0] > 1.0, "F scores: {scores:?}");
        assert!(
            scores[1] < 1e-10,
            "F score for constant feature should be ~0: {}",
            scores[1]
        );
    }

    #[test]
    fn test_f_classif_needs_two_classes() {
        let x = Tensor::from_vec(vec![1.0_f64, 2.0], vec![2, 1]).unwrap();
        let y = Tensor::from_vec(vec![0.0, 0.0], vec![2]).unwrap();
        assert!(f_classif(&x, &y).is_err());
    }
}
