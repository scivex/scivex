//! Window functions: rolling, expanding, and exponentially weighted.

use scivex_core::Float;

use super::Series;
use crate::error::{FrameError, Result};

/// Configuration for rolling window operations.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct RollingWindow {
    /// Number of elements in each window.
    pub window_size: usize,
    /// Minimum number of non-null values required to produce a result.
    pub min_periods: usize,
    /// Whether to center the window on the current element.
    pub center: bool,
}

impl RollingWindow {
    /// Create a new rolling window with the given size.
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            min_periods: window_size,
            center: false,
        }
    }

    /// Set the minimum number of non-null observations required.
    pub fn min_periods(mut self, n: usize) -> Self {
        self.min_periods = n;
        self
    }

    /// Whether to center the window label.
    pub fn center(mut self, c: bool) -> Self {
        self.center = c;
        self
    }
}

// ---------------------------------------------------------------------------
// Rolling window operations
// ---------------------------------------------------------------------------

impl<T: Float> Series<T> {
    /// Compute the window bounds for element `i`.
    fn window_bounds(&self, i: usize, w: &RollingWindow) -> (usize, usize) {
        let n = self.len();
        if w.center {
            let half = w.window_size / 2;
            let start = i.saturating_sub(half);
            let end = (i + w.window_size - half).min(n);
            (start, end)
        } else {
            let start = (i + 1).saturating_sub(w.window_size);
            (start, i + 1)
        }
    }

    /// Collect non-null values in a window.
    fn window_values(&self, start: usize, end: usize) -> Vec<T> {
        (start..end)
            .filter(|&j| !self.is_null_at(j))
            .map(|j| self.data[j])
            .collect()
    }

    /// Rolling mean.
    pub fn rolling_mean(&self, w: &RollingWindow) -> Result<Series<T>> {
        if w.window_size == 0 {
            return Err(FrameError::InvalidArgument {
                reason: "window_size must be > 0",
            });
        }
        let n = self.len();
        let mut data = vec![T::zero(); n];
        let mut null_mask = vec![false; n];

        for i in 0..n {
            let (start, end) = self.window_bounds(i, w);
            let vals = self.window_values(start, end);
            if vals.len() >= w.min_periods {
                let sum: T = vals.iter().copied().sum();
                data[i] = sum / T::from_usize(vals.len());
            } else {
                null_mask[i] = true;
            }
        }

        let has_nulls = null_mask.iter().any(|&v| v);
        Ok(Series {
            name: self.name.clone(),
            data,
            null_mask: if has_nulls { Some(null_mask) } else { None },
        })
    }

    /// Rolling sum.
    pub fn rolling_sum(&self, w: &RollingWindow) -> Result<Series<T>> {
        if w.window_size == 0 {
            return Err(FrameError::InvalidArgument {
                reason: "window_size must be > 0",
            });
        }
        let n = self.len();
        let mut data = vec![T::zero(); n];
        let mut null_mask = vec![false; n];

        for i in 0..n {
            let (start, end) = self.window_bounds(i, w);
            let vals = self.window_values(start, end);
            if vals.len() >= w.min_periods {
                data[i] = vals.iter().copied().sum();
            } else {
                null_mask[i] = true;
            }
        }

        let has_nulls = null_mask.iter().any(|&v| v);
        Ok(Series {
            name: self.name.clone(),
            data,
            null_mask: if has_nulls { Some(null_mask) } else { None },
        })
    }

    /// Rolling minimum.
    pub fn rolling_min(&self, w: &RollingWindow) -> Result<Series<T>> {
        if w.window_size == 0 {
            return Err(FrameError::InvalidArgument {
                reason: "window_size must be > 0",
            });
        }
        let n = self.len();
        let mut data = vec![T::zero(); n];
        let mut null_mask = vec![false; n];

        for i in 0..n {
            let (start, end) = self.window_bounds(i, w);
            let vals = self.window_values(start, end);
            if vals.len() >= w.min_periods {
                data[i] = vals
                    .iter()
                    .copied()
                    .reduce(T::min)
                    .expect("non-empty window");
            } else {
                null_mask[i] = true;
            }
        }

        let has_nulls = null_mask.iter().any(|&v| v);
        Ok(Series {
            name: self.name.clone(),
            data,
            null_mask: if has_nulls { Some(null_mask) } else { None },
        })
    }

    /// Rolling maximum.
    pub fn rolling_max(&self, w: &RollingWindow) -> Result<Series<T>> {
        if w.window_size == 0 {
            return Err(FrameError::InvalidArgument {
                reason: "window_size must be > 0",
            });
        }
        let n = self.len();
        let mut data = vec![T::zero(); n];
        let mut null_mask = vec![false; n];

        for i in 0..n {
            let (start, end) = self.window_bounds(i, w);
            let vals = self.window_values(start, end);
            if vals.len() >= w.min_periods {
                data[i] = vals
                    .iter()
                    .copied()
                    .reduce(T::max)
                    .expect("non-empty window");
            } else {
                null_mask[i] = true;
            }
        }

        let has_nulls = null_mask.iter().any(|&v| v);
        Ok(Series {
            name: self.name.clone(),
            data,
            null_mask: if has_nulls { Some(null_mask) } else { None },
        })
    }

    /// Rolling standard deviation (population).
    pub fn rolling_std(&self, w: &RollingWindow) -> Result<Series<T>> {
        if w.window_size == 0 {
            return Err(FrameError::InvalidArgument {
                reason: "window_size must be > 0",
            });
        }
        let n = self.len();
        let mut data = vec![T::zero(); n];
        let mut null_mask = vec![false; n];

        for i in 0..n {
            let (start, end) = self.window_bounds(i, w);
            let vals = self.window_values(start, end);
            if vals.len() >= w.min_periods {
                let count = T::from_usize(vals.len());
                let mean = vals.iter().copied().sum::<T>() / count;
                let var = vals.iter().map(|&v| (v - mean) * (v - mean)).sum::<T>() / count;
                data[i] = var.sqrt();
            } else {
                null_mask[i] = true;
            }
        }

        let has_nulls = null_mask.iter().any(|&v| v);
        Ok(Series {
            name: self.name.clone(),
            data,
            null_mask: if has_nulls { Some(null_mask) } else { None },
        })
    }

    // -----------------------------------------------------------------------
    // Expanding
    // -----------------------------------------------------------------------

    /// Expanding (cumulative) sum.
    pub fn expanding_sum(&self) -> Series<T> {
        let mut data = Vec::with_capacity(self.len());
        let mut acc = T::zero();
        for i in 0..self.len() {
            if !self.is_null_at(i) {
                acc += self.data[i];
            }
            data.push(acc);
        }
        Series {
            name: self.name.clone(),
            data,
            null_mask: self.null_mask.clone(),
        }
    }

    /// Expanding (cumulative) mean.
    pub fn expanding_mean(&self) -> Series<T> {
        let mut data = Vec::with_capacity(self.len());
        let mut sum = T::zero();
        let mut count = 0usize;
        for i in 0..self.len() {
            if !self.is_null_at(i) {
                sum += self.data[i];
                count += 1;
            }
            if count > 0 {
                data.push(sum / T::from_usize(count));
            } else {
                data.push(T::zero());
            }
        }
        Series {
            name: self.name.clone(),
            data,
            null_mask: self.null_mask.clone(),
        }
    }

    /// Expanding (cumulative) minimum.
    pub fn expanding_min(&self) -> Series<T> {
        let mut data = Vec::with_capacity(self.len());
        let mut current_min = T::infinity();
        for i in 0..self.len() {
            if !self.is_null_at(i) {
                current_min = current_min.min(self.data[i]);
            }
            data.push(current_min);
        }
        Series {
            name: self.name.clone(),
            data,
            null_mask: self.null_mask.clone(),
        }
    }

    /// Expanding (cumulative) maximum.
    pub fn expanding_max(&self) -> Series<T> {
        let mut data = Vec::with_capacity(self.len());
        let mut current_max = T::neg_infinity();
        for i in 0..self.len() {
            if !self.is_null_at(i) {
                current_max = current_max.max(self.data[i]);
            }
            data.push(current_max);
        }
        Series {
            name: self.name.clone(),
            data,
            null_mask: self.null_mask.clone(),
        }
    }

    // -----------------------------------------------------------------------
    // Exponentially weighted
    // -----------------------------------------------------------------------

    /// Exponentially weighted moving average.
    ///
    /// `alpha` must be in `(0, 1]`. Each value is weighted by `alpha * (1-alpha)^i`
    /// where `i` is the distance from the current position.
    pub fn ewm_mean(&self, alpha: T) -> Result<Series<T>> {
        if alpha <= T::zero() || alpha > T::one() {
            return Err(FrameError::InvalidArgument {
                reason: "alpha must be in (0, 1]",
            });
        }

        let n = self.len();
        let mut data = vec![T::zero(); n];
        let mut null_mask = vec![true; n];

        let one_minus_alpha = T::one() - alpha;
        let mut ewm = T::zero();
        let mut weight_sum = T::zero();
        let mut started = false;

        for i in 0..n {
            if self.is_null_at(i) {
                if started {
                    data[i] = ewm / weight_sum;
                    null_mask[i] = false;
                }
                // Decay the weights even on null
                weight_sum *= one_minus_alpha;
                ewm *= one_minus_alpha;
                continue;
            }

            if started {
                weight_sum = weight_sum * one_minus_alpha + T::one();
                ewm = ewm * one_minus_alpha + self.data[i];
            } else {
                ewm = self.data[i];
                weight_sum = T::one();
                started = true;
            }
            data[i] = ewm / weight_sum;
            null_mask[i] = false;
        }

        let has_nulls = null_mask.iter().any(|&v| v);
        Ok(Series {
            name: self.name.clone(),
            data,
            null_mask: if has_nulls { Some(null_mask) } else { None },
        })
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn test_rolling_mean_basic() {
        let s = Series::new("x", vec![1.0_f64, 2.0, 3.0, 4.0, 5.0]);
        let w = RollingWindow::new(3);
        let result = s.rolling_mean(&w).unwrap();
        // First two values don't have enough data (min_periods=3)
        assert!(result.is_null_at(0));
        assert!(result.is_null_at(1));
        assert!((result.get(2).unwrap() - 2.0).abs() < 1e-10);
        assert!((result.get(3).unwrap() - 3.0).abs() < 1e-10);
        assert!((result.get(4).unwrap() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_rolling_with_nulls() {
        let s = Series::with_nulls(
            "x",
            vec![1.0_f64, 0.0, 3.0, 4.0],
            vec![false, true, false, false],
        )
        .unwrap();
        let w = RollingWindow::new(3).min_periods(2);
        let result = s.rolling_mean(&w).unwrap();
        // Window [0..1]: only 1 non-null → null (min_periods=2)
        assert!(result.is_null_at(0));
        // Window [0..2]: 1 non-null → null
        assert!(result.is_null_at(1));
        // Window [0..3]: 1.0 and 3.0 → 2.0
        assert!((result.get(2).unwrap() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_rolling_min_periods() {
        let s = Series::new("x", vec![1.0_f64, 2.0, 3.0]);
        let w = RollingWindow::new(3).min_periods(1);
        let result = s.rolling_mean(&w).unwrap();
        // Even the first element should produce a result
        assert!(!result.is_null_at(0));
        assert_eq!(result.get(0), Some(1.0));
        assert!((result.get(1).unwrap() - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_rolling_centered() {
        let s = Series::new("x", vec![1.0_f64, 2.0, 3.0, 4.0, 5.0]);
        let w = RollingWindow::new(3).center(true);
        let result = s.rolling_mean(&w).unwrap();
        // Centered: window for i=1 is [0,1,2] → mean=2.0
        assert!((result.get(1).unwrap() - 2.0).abs() < 1e-10);
        // i=2: [1,2,3] → 3.0
        assert!((result.get(2).unwrap() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_expanding_sum() {
        let s = Series::new("x", vec![1.0_f64, 2.0, 3.0, 4.0]);
        let result = s.expanding_sum();
        assert_eq!(result.as_slice(), &[1.0, 3.0, 6.0, 10.0]);
    }

    #[test]
    fn test_expanding_mean() {
        let s = Series::new("x", vec![2.0_f64, 4.0, 6.0]);
        let result = s.expanding_mean();
        assert_eq!(result.get(0), Some(2.0));
        assert_eq!(result.get(1), Some(3.0));
        assert_eq!(result.get(2), Some(4.0));
    }

    #[test]
    fn test_ewm_mean() {
        let s = Series::new("x", vec![1.0_f64, 2.0, 3.0, 4.0]);
        let result = s.ewm_mean(0.5).unwrap();
        // First value = 1.0
        assert_eq!(result.get(0), Some(1.0));
        // Second: (0.5*1 + 2) / (0.5 + 1) = 2.5/1.5 ≈ 1.667
        assert!((result.get(1).unwrap() - 5.0 / 3.0).abs() < 1e-10);
        assert!(!result.is_null_at(3));
    }

    // -- Edge-case tests -------------------------------------------------------

    #[test]
    fn test_rolling_window_size_greater_than_data() {
        let s = Series::new("x", vec![1.0_f64, 2.0]);
        let w = RollingWindow::new(5);
        let result = s.rolling_mean(&w).unwrap();
        // min_periods=5 but data has only 2 elements → all null
        assert!(result.is_null_at(0));
        assert!(result.is_null_at(1));
    }

    #[test]
    fn test_rolling_window_size_1() {
        let s = Series::new("x", vec![1.0_f64, 2.0, 3.0]);
        let w = RollingWindow::new(1);
        let result = s.rolling_mean(&w).unwrap();
        // Window of 1 = identity
        assert_eq!(result.get(0), Some(1.0));
        assert_eq!(result.get(1), Some(2.0));
        assert_eq!(result.get(2), Some(3.0));
    }

    #[test]
    fn test_rolling_window_size_0_error() {
        let s = Series::new("x", vec![1.0_f64, 2.0]);
        let w = RollingWindow::new(0);
        assert!(s.rolling_mean(&w).is_err());
        assert!(s.rolling_sum(&w).is_err());
        assert!(s.rolling_min(&w).is_err());
        assert!(s.rolling_max(&w).is_err());
        assert!(s.rolling_std(&w).is_err());
    }

    #[test]
    fn test_rolling_sum_basic() {
        let s = Series::new("x", vec![1.0_f64, 2.0, 3.0, 4.0]);
        let w = RollingWindow::new(2);
        let result = s.rolling_sum(&w).unwrap();
        assert!(result.is_null_at(0));
        assert_eq!(result.get(1), Some(3.0));
        assert_eq!(result.get(2), Some(5.0));
        assert_eq!(result.get(3), Some(7.0));
    }

    #[test]
    fn test_rolling_min_max() {
        let s = Series::new("x", vec![3.0_f64, 1.0, 4.0, 1.0, 5.0]);
        let w = RollingWindow::new(3);
        let rmin = s.rolling_min(&w).unwrap();
        let rmax = s.rolling_max(&w).unwrap();
        // index 2: window [3,1,4] → min=1, max=4
        assert_eq!(rmin.get(2), Some(1.0));
        assert_eq!(rmax.get(2), Some(4.0));
    }

    #[test]
    fn test_ewm_mean_alpha_1() {
        let s = Series::new("x", vec![1.0_f64, 2.0, 3.0]);
        let result = s.ewm_mean(1.0).unwrap();
        // alpha=1 means no memory, each value is just itself
        assert_eq!(result.get(0), Some(1.0));
        assert_eq!(result.get(1), Some(2.0));
        assert_eq!(result.get(2), Some(3.0));
    }

    #[test]
    fn test_ewm_mean_invalid_alpha() {
        let s = Series::new("x", vec![1.0_f64]);
        assert!(s.ewm_mean(0.0).is_err());
        assert!(s.ewm_mean(-0.5).is_err());
    }

    #[test]
    fn test_expanding_min_max() {
        let s = Series::new("x", vec![3.0_f64, 1.0, 4.0, 1.0, 5.0]);
        let emin = s.expanding_min();
        let emax = s.expanding_max();
        assert_eq!(emin.as_slice(), &[3.0, 1.0, 1.0, 1.0, 1.0]);
        assert_eq!(emax.as_slice(), &[3.0, 3.0, 4.0, 4.0, 5.0]);
    }

    #[test]
    fn test_rolling_std_basic() {
        let s = Series::new("x", vec![1.0_f64, 2.0, 3.0, 4.0, 5.0]);
        let w = RollingWindow::new(3);
        let result = s.rolling_std(&w).unwrap();
        assert!(result.is_null_at(0));
        assert!(result.is_null_at(1));
        // Window [1,2,3]: mean=2, var=(1+0+1)/3=2/3, std=sqrt(2/3)
        let std_val = result.get(2).unwrap();
        let expected = (2.0_f64 / 3.0).sqrt();
        assert!((std_val - expected).abs() < 1e-10);
    }

    #[test]
    fn test_expanding_sum_empty() {
        let s: Series<f64> = Series::new("x", vec![]);
        let result = s.expanding_sum();
        assert!(result.is_empty());
    }

    #[test]
    fn test_expanding_mean_empty() {
        let s: Series<f64> = Series::new("x", vec![]);
        let result = s.expanding_mean();
        assert!(result.is_empty());
    }

    #[test]
    fn test_ewm_mean_with_nulls() {
        let s = Series::with_nulls("x", vec![1.0_f64, 0.0, 3.0], vec![false, true, false]).unwrap();
        let result = s.ewm_mean(0.5).unwrap();
        assert_eq!(result.get(0), Some(1.0));
        // Index 1 is null in source but ewm should still produce a value (carry forward)
        assert!(!result.is_null_at(1));
        assert!(!result.is_null_at(2));
    }

    #[test]
    fn test_rolling_sum_window_larger_than_data() {
        let s = Series::new("x", vec![1.0_f64, 2.0]);
        let w = RollingWindow::new(5);
        let result = s.rolling_sum(&w).unwrap();
        assert!(result.is_null_at(0));
        assert!(result.is_null_at(1));
    }

    #[test]
    fn test_rolling_min_max_window_1() {
        let s = Series::new("x", vec![3.0_f64, 1.0, 4.0]);
        let w = RollingWindow::new(1);
        let rmin = s.rolling_min(&w).unwrap();
        let rmax = s.rolling_max(&w).unwrap();
        assert_eq!(rmin.as_slice(), &[3.0, 1.0, 4.0]);
        assert_eq!(rmax.as_slice(), &[3.0, 1.0, 4.0]);
    }

    #[test]
    fn test_rolling_sum_window_1() {
        let s = Series::new("x", vec![10.0_f64, 20.0, 30.0]);
        let w = RollingWindow::new(1);
        let result = s.rolling_sum(&w).unwrap();
        assert_eq!(result.as_slice(), &[10.0, 20.0, 30.0]);
    }
}
