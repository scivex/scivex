//! Peak detection and prominence calculation.

use scivex_core::{Float, Tensor};

use crate::error::{Result, SignalError};

/// Find local maxima in a 1-D signal.
///
/// A sample `x[i]` is a peak if `x[i] > x[i-1]` and `x[i] > x[i+1]` (strict),
/// it is >= `min_height` (if `Some`), and at least `min_distance` apart from the
/// previously accepted peak.
///
/// Returns the indices of the detected peaks.
pub fn find_peaks<T: Float>(
    x: &Tensor<T>,
    min_height: Option<T>,
    min_distance: Option<usize>,
) -> Result<Vec<usize>> {
    if x.ndim() != 1 {
        return Err(SignalError::InvalidParameter {
            name: "x",
            reason: "find_peaks expects a 1-D tensor",
        });
    }
    let n = x.numel();
    if n < 3 {
        return Ok(vec![]);
    }
    let xs = x.as_slice();

    // Find all local maxima.
    let mut candidates: Vec<usize> = Vec::new();
    for i in 1..n - 1 {
        if xs[i] > xs[i - 1] && xs[i] > xs[i + 1] {
            #[allow(clippy::collapsible_if)]
            if let Some(h) = min_height {
                if xs[i] < h {
                    continue;
                }
            }
            candidates.push(i);
        }
    }

    // Filter by minimum distance (greedy: keep tallest first could be done,
    // but simple left-to-right filtering matches scipy's basic behavior).
    #[allow(clippy::collapsible_if)]
    if let Some(dist) = min_distance {
        if dist > 0 {
            let mut filtered = Vec::new();
            let mut last_peak: Option<usize> = None;
            for &idx in &candidates {
                #[allow(clippy::collapsible_if)]
                if let Some(last) = last_peak {
                    if idx - last < dist {
                        continue;
                    }
                }
                filtered.push(idx);
                last_peak = Some(idx);
            }
            return Ok(filtered);
        }
    }

    Ok(candidates)
}

/// Compute the prominence of each peak.
///
/// Prominence is the height of a peak relative to the highest contour line
/// that does not contain a higher peak. In simpler terms, for each peak we
/// walk left and right until we find a higher peak (or the signal boundary),
/// and the prominence is `peak_height - max(min_left, min_right)`.
pub fn peak_prominences<T: Float>(x: &Tensor<T>, peaks: &[usize]) -> Result<Vec<T>> {
    if x.ndim() != 1 {
        return Err(SignalError::InvalidParameter {
            name: "x",
            reason: "peak_prominences expects a 1-D tensor",
        });
    }
    let xs = x.as_slice();
    let n = xs.len();

    let mut proms = Vec::with_capacity(peaks.len());

    for &peak in peaks {
        if peak >= n {
            return Err(SignalError::InvalidParameter {
                name: "peaks",
                reason: "peak index out of bounds",
            });
        }
        let peak_val = xs[peak];

        // Walk left to find the minimum before a higher peak or boundary.
        let mut left_min = peak_val;
        for &val in xs[..peak].iter().rev() {
            if val > peak_val {
                break;
            }
            if val < left_min {
                left_min = val;
            }
        }

        // Walk right.
        let mut right_min = peak_val;
        for &val in &xs[peak + 1..] {
            if val > peak_val {
                break;
            }
            if val < right_min {
                right_min = val;
            }
        }

        // Prominence = peak height - max(left_min, right_min).
        let reference = if left_min > right_min {
            left_min
        } else {
            right_min
        };
        proms.push(peak_val - reference);
    }

    Ok(proms)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_peaks_simple() {
        let data = vec![0.0, 0.0, 1.0, 0.0, 0.0];
        let x = Tensor::from_vec(data, vec![5]).unwrap();
        let peaks = find_peaks(&x, None, None).unwrap();
        assert_eq!(peaks, vec![2]);
    }

    #[test]
    fn test_find_peaks_multiple() {
        let data = vec![0.0, 1.0, 0.0, 2.0, 0.0, 1.5, 0.0];
        let x = Tensor::from_vec(data, vec![7]).unwrap();
        let peaks = find_peaks(&x, None, None).unwrap();
        assert_eq!(peaks, vec![1, 3, 5]);
    }

    #[test]
    fn test_find_peaks_min_height() {
        let data = vec![0.0, 1.0, 0.0, 2.0, 0.0, 0.5, 0.0];
        let x = Tensor::from_vec(data, vec![7]).unwrap();
        let peaks = find_peaks(&x, Some(1.5), None).unwrap();
        assert_eq!(peaks, vec![3]);
    }

    #[test]
    fn test_find_peaks_min_distance() {
        let data = vec![0.0, 1.0, 0.0, 2.0, 0.0, 1.5, 0.0];
        let x = Tensor::from_vec(data, vec![7]).unwrap();
        let peaks = find_peaks(&x, None, Some(3)).unwrap();
        assert_eq!(peaks, vec![1, 5]);
    }

    #[test]
    fn test_find_peaks_short_signal() {
        let x = Tensor::from_vec(vec![1.0, 2.0], vec![2]).unwrap();
        let peaks = find_peaks(&x, None, None).unwrap();
        assert!(peaks.is_empty());
    }

    #[test]
    fn test_prominence_single_peak() {
        let data = vec![0.0, 0.0, 1.0, 0.0, 0.0];
        let x = Tensor::from_vec(data, vec![5]).unwrap();
        let proms = peak_prominences(&x, &[2]).unwrap();
        assert_eq!(proms.len(), 1);
        assert!((proms[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_prominence_two_peaks() {
        let data = vec![0.0, 2.0, 0.5, 1.0, 0.0];
        let x = Tensor::from_vec(data, vec![5]).unwrap();
        let proms = peak_prominences(&x, &[1, 3]).unwrap();
        assert!((proms[0] - 2.0).abs() < 1e-10); // 2.0 - 0.0
        assert!((proms[1] - 0.5).abs() < 1e-10); // 1.0 - 0.5
    }
}
