//! Welford's online algorithms for streaming mean and variance.

use scivex_core::Float;

/// Online mean accumulator using Welford's algorithm.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct StreamingMean<T: Float> {
    mean: T,
    count: usize,
}

impl<T: Float> StreamingMean<T> {
    /// Create a new `StreamingMean` accumulator.
    #[must_use]
    pub fn new() -> Self {
        Self {
            mean: T::zero(),
            count: 0,
        }
    }

    /// Incorporate a single observation.
    pub fn update(&mut self, value: T) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / T::from_usize(self.count);
    }

    /// Incorporate a batch of observations.
    pub fn update_batch(&mut self, values: &[T]) {
        for &v in values {
            self.update(v);
        }
    }

    /// Current running mean, or zero if no observations.
    #[must_use]
    pub fn mean(&self) -> T {
        self.mean
    }

    /// Number of observations seen.
    #[must_use]
    pub fn count(&self) -> usize {
        self.count
    }

    /// Reset to the initial state.
    pub fn reset(&mut self) {
        self.mean = T::zero();
        self.count = 0;
    }
}

impl<T: Float> Default for StreamingMean<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Online mean and variance accumulator using Welford's algorithm.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct StreamingVariance<T: Float> {
    mean: T,
    m2: T,
    count: usize,
}

impl<T: Float> StreamingVariance<T> {
    /// Create a new `StreamingVariance` accumulator.
    #[must_use]
    pub fn new() -> Self {
        Self {
            mean: T::zero(),
            m2: T::zero(),
            count: 0,
        }
    }

    /// Incorporate a single observation.
    pub fn update(&mut self, value: T) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / T::from_usize(self.count);
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
    }

    /// Incorporate a batch of observations.
    pub fn update_batch(&mut self, values: &[T]) {
        for &v in values {
            self.update(v);
        }
    }

    /// Current running mean, or zero if no observations.
    #[must_use]
    pub fn mean(&self) -> T {
        self.mean
    }

    /// Population variance (biased, divides by N).
    ///
    /// Returns zero if fewer than one observation.
    #[must_use]
    pub fn variance(&self) -> T {
        if self.count < 1 {
            return T::zero();
        }
        self.m2 / T::from_usize(self.count)
    }

    /// Sample variance (unbiased, divides by N-1).
    ///
    /// Returns zero if fewer than two observations.
    #[must_use]
    pub fn sample_variance(&self) -> T {
        if self.count < 2 {
            return T::zero();
        }
        self.m2 / T::from_usize(self.count - 1)
    }

    /// Population standard deviation.
    #[must_use]
    pub fn std_dev(&self) -> T {
        self.variance().sqrt()
    }

    /// Number of observations seen.
    #[must_use]
    pub fn count(&self) -> usize {
        self.count
    }

    /// Reset to the initial state.
    pub fn reset(&mut self) {
        self.mean = T::zero();
        self.m2 = T::zero();
        self.count = 0;
    }
}

impl<T: Float> Default for StreamingVariance<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_mean_basic() {
        let mut sm = StreamingMean::<f64>::new();
        for &v in &[1.0, 2.0, 3.0, 4.0, 5.0] {
            sm.update(v);
        }
        assert!((sm.mean() - 3.0).abs() < 1e-12);
        assert_eq!(sm.count(), 5);
    }

    #[test]
    fn test_streaming_mean_batch() {
        let mut sm_single = StreamingMean::<f64>::new();
        for &v in &[1.0, 2.0, 3.0, 4.0, 5.0] {
            sm_single.update(v);
        }

        let mut sm_batch = StreamingMean::<f64>::new();
        sm_batch.update_batch(&[1.0, 2.0, 3.0, 4.0, 5.0]);

        assert!((sm_single.mean() - sm_batch.mean()).abs() < 1e-12);
        assert_eq!(sm_single.count(), sm_batch.count());
    }

    #[test]
    fn test_streaming_variance_basic() {
        // Values: [2, 4, 4, 4, 5, 5, 7, 9]
        // Mean = 5, population variance = 4.0
        let mut sv = StreamingVariance::<f64>::new();
        for &v in &[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
            sv.update(v);
        }
        assert!((sv.mean() - 5.0).abs() < 1e-12);
        assert!((sv.variance() - 4.0).abs() < 1e-12);
    }

    #[test]
    fn test_streaming_variance_sample() {
        // Values: [2, 4, 4, 4, 5, 5, 7, 9]
        // sample_variance = 32/7 ~ 4.571428...
        let mut sv = StreamingVariance::<f64>::new();
        sv.update_batch(&[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
        let expected = 32.0 / 7.0;
        assert!((sv.sample_variance() - expected).abs() < 1e-12);
    }

    #[test]
    fn test_streaming_std_dev() {
        let mut sv = StreamingVariance::<f64>::new();
        sv.update_batch(&[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
        assert!((sv.std_dev() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_streaming_reset() {
        let mut sv = StreamingVariance::<f64>::new();
        sv.update_batch(&[1.0, 2.0, 3.0]);
        sv.reset();
        assert_eq!(sv.count(), 0);
        assert!((sv.mean() - 0.0).abs() < 1e-12);
        assert!((sv.variance() - 0.0).abs() < 1e-12);
    }
}
