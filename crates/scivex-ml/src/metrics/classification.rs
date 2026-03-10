use scivex_core::Float;

use crate::error::{MlError, Result};

/// Fraction of correct predictions.
pub fn accuracy<T: Float>(y_true: &[T], y_pred: &[T]) -> Result<T> {
    if y_true.is_empty() {
        return Err(MlError::EmptyInput);
    }
    if y_true.len() != y_pred.len() {
        return Err(MlError::DimensionMismatch {
            expected: y_true.len(),
            got: y_pred.len(),
        });
    }
    let correct = y_true
        .iter()
        .copied()
        .zip(y_pred.iter().copied())
        .filter(|(a, b)| (*a - *b).abs() < T::epsilon())
        .count();
    Ok(T::from_usize(correct) / T::from_usize(y_true.len()))
}

/// Binary precision (positive class is the maximum value in `y_true`).
pub fn precision<T: Float>(y_true: &[T], y_pred: &[T]) -> Result<T> {
    if y_true.is_empty() {
        return Err(MlError::EmptyInput);
    }
    if y_true.len() != y_pred.len() {
        return Err(MlError::DimensionMismatch {
            expected: y_true.len(),
            got: y_pred.len(),
        });
    }
    let pos = positive_class(y_true);
    let (mut tp, mut fp) = (0usize, 0usize);
    for (t, p) in y_true.iter().zip(y_pred) {
        let pred_pos = (*p - pos).abs() < T::epsilon();
        let true_pos = (*t - pos).abs() < T::epsilon();
        if pred_pos {
            if true_pos {
                tp += 1;
            } else {
                fp += 1;
            }
        }
    }
    if tp + fp == 0 {
        return Ok(T::zero());
    }
    Ok(T::from_usize(tp) / T::from_usize(tp + fp))
}

/// Binary recall.
pub fn recall<T: Float>(y_true: &[T], y_pred: &[T]) -> Result<T> {
    if y_true.is_empty() {
        return Err(MlError::EmptyInput);
    }
    if y_true.len() != y_pred.len() {
        return Err(MlError::DimensionMismatch {
            expected: y_true.len(),
            got: y_pred.len(),
        });
    }
    let pos = positive_class(y_true);
    let (mut tp, mut fn_count) = (0usize, 0usize);
    for (t, p) in y_true.iter().zip(y_pred) {
        let true_pos = (*t - pos).abs() < T::epsilon();
        if true_pos {
            let pred_pos = (*p - pos).abs() < T::epsilon();
            if pred_pos {
                tp += 1;
            } else {
                fn_count += 1;
            }
        }
    }
    if tp + fn_count == 0 {
        return Ok(T::zero());
    }
    Ok(T::from_usize(tp) / T::from_usize(tp + fn_count))
}

/// F1 score: harmonic mean of precision and recall.
pub fn f1_score<T: Float>(y_true: &[T], y_pred: &[T]) -> Result<T> {
    let p = precision(y_true, y_pred)?;
    let r = recall(y_true, y_pred)?;
    let denom = p + r;
    if denom < T::epsilon() {
        return Ok(T::zero());
    }
    let two = T::from_usize(2);
    Ok(two * p * r / denom)
}

/// Compute an n×n confusion matrix.
///
/// `n_classes` is the number of distinct classes (labels assumed to be 0..n_classes-1).
pub fn confusion_matrix<T: Float>(
    y_true: &[T],
    y_pred: &[T],
    n_classes: usize,
) -> Result<Vec<Vec<usize>>> {
    if y_true.is_empty() {
        return Err(MlError::EmptyInput);
    }
    if y_true.len() != y_pred.len() {
        return Err(MlError::DimensionMismatch {
            expected: y_true.len(),
            got: y_pred.len(),
        });
    }
    let mut matrix = vec![vec![0usize; n_classes]; n_classes];
    for (t, p) in y_true.iter().zip(y_pred) {
        let ti = t.round();
        let pi = p.round();
        // Convert to usize indices
        let row = to_index(ti, n_classes);
        let col = to_index(pi, n_classes);
        matrix[row][col] += 1;
    }
    Ok(matrix)
}

/// Identify the positive class as the maximum label value.
fn positive_class<T: Float>(y: &[T]) -> T {
    y.iter().copied().fold(T::neg_infinity(), T::max)
}

/// Safely convert a float label to a usize index, clamping to [0, n-1].
fn to_index<T: Float>(val: T, n: usize) -> usize {
    // Round, clamp, cast
    let v = val.round();
    let zero = T::zero();
    let max_t = T::from_usize(n - 1);
    let clamped = v.max(zero).min(max_t);
    // Convert to usize via from_usize inverse: multiply by 1, compare increments
    let mut idx = 0usize;
    while idx < n - 1 {
        if clamped - T::from_usize(idx) < T::from_usize(1) / T::from_usize(2) {
            return idx;
        }
        idx += 1;
    }
    idx
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn test_accuracy_perfect() {
        let y = [1.0_f64, 0.0, 1.0, 1.0];
        assert!((accuracy(&y, &y).unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_accuracy_half() {
        let y_true = [1.0_f64, 0.0, 1.0, 0.0];
        let y_pred = [1.0_f64, 1.0, 0.0, 0.0];
        assert!((accuracy(&y_true, &y_pred).unwrap() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_precision_recall_f1() {
        // TP=2, FP=1, FN=1
        let y_true = [1.0_f64, 1.0, 0.0, 1.0];
        let y_pred = [1.0, 0.0, 1.0, 1.0];
        let p = precision(&y_true, &y_pred).unwrap();
        let r = recall(&y_true, &y_pred).unwrap();
        // precision = 2/3
        assert!((p - 2.0 / 3.0).abs() < 1e-12);
        // recall = 2/3
        assert!((r - 2.0 / 3.0).abs() < 1e-12);
        let f1 = f1_score(&y_true, &y_pred).unwrap();
        assert!((f1 - 2.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_confusion_matrix_2x2() {
        let y_true = [0.0_f64, 0.0, 1.0, 1.0];
        let y_pred = [0.0, 1.0, 0.0, 1.0];
        let cm = confusion_matrix(&y_true, &y_pred, 2).unwrap();
        assert_eq!(cm, vec![vec![1, 1], vec![1, 1]]);
    }

    #[test]
    fn test_empty_input() {
        let empty: [f64; 0] = [];
        assert!(accuracy(&empty, &empty).is_err());
    }
}
