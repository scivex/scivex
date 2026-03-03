//! Window functions: rolling, expanding, and exponentially weighted.

use scivex_core::Float;

use super::Series;
use crate::error::{FrameError, Result};

/// Configuration for rolling window operations.
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
                data[i] = vals.iter().copied().reduce(T::min).unwrap();
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
                data[i] = vals.iter().copied().reduce(T::max).unwrap();
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
}
