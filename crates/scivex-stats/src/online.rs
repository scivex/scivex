//! Online (streaming) statistics using numerically stable algorithms.
//!
//! Provides incremental computation of mean, variance, min, max, skewness,
//! and kurtosis without storing all data points. Uses Welford's algorithm
//! for numerically stable variance computation.

use scivex_core::Float;

/// Online statistics accumulator using Welford's algorithm.
///
/// Computes running mean, variance, standard deviation, min, max, skewness,
/// and kurtosis in a single pass with O(1) memory.
///
/// # Examples
///
/// ```
/// # use scivex_stats::online::OnlineStats;
/// let mut stats = OnlineStats::<f64>::new();
/// for &x in &[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
///     stats.push(x);
/// }
/// assert_eq!(stats.count(), 8);
/// assert!((stats.mean() - 5.0).abs() < 1e-10);
/// assert!((stats.variance() - 4.0).abs() < 1e-10);
/// ```
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct OnlineStats<T: Float> {
    count: usize,
    mean: T,
    m2: T,
    m3: T,
    m4: T,
    min: T,
    max: T,
}

impl<T: Float> Default for OnlineStats<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float> OnlineStats<T> {
    /// Create a new empty accumulator.
    pub fn new() -> Self {
        Self {
            count: 0,
            mean: T::zero(),
            m2: T::zero(),
            m3: T::zero(),
            m4: T::zero(),
            min: T::infinity(),
            max: T::neg_infinity(),
        }
    }

    /// Add a single observation.
    pub fn push(&mut self, x: T) {
        let n1 = self.count;
        self.count += 1;
        let n = T::from_usize(self.count);
        let delta = x - self.mean;
        let delta_n = delta / n;
        let delta_n2 = delta_n * delta_n;
        let term1 = delta * delta_n * T::from_usize(n1);

        // Update central moments (Terriberry's extension of Welford).
        self.m4 += term1 * delta_n2 * (n * n - T::from_usize(3) * n + T::from_usize(3))
            + T::from_usize(6) * delta_n2 * self.m2
            - T::from_usize(4) * delta_n * self.m3;
        self.m3 += term1 * delta_n * (n - T::from_usize(2)) - T::from_usize(3) * delta_n * self.m2;
        self.m2 += term1;
        self.mean += delta_n;

        if x < self.min {
            self.min = x;
        }
        if x > self.max {
            self.max = x;
        }
    }

    /// Add a batch of observations.
    pub fn push_slice(&mut self, data: &[T]) {
        for &x in data {
            self.push(x);
        }
    }

    /// Number of observations seen.
    pub fn count(&self) -> usize {
        self.count
    }

    /// Current mean (returns 0 if empty).
    pub fn mean(&self) -> T {
        self.mean
    }

    /// Population variance (biased, ddof=0).
    pub fn variance(&self) -> T {
        if self.count < 2 {
            return T::zero();
        }
        self.m2 / T::from_usize(self.count)
    }

    /// Sample variance (unbiased, ddof=1).
    pub fn sample_variance(&self) -> T {
        if self.count < 2 {
            return T::zero();
        }
        self.m2 / T::from_usize(self.count - 1)
    }

    /// Population standard deviation.
    pub fn std_dev(&self) -> T {
        self.variance().sqrt()
    }

    /// Sample standard deviation.
    pub fn sample_std_dev(&self) -> T {
        self.sample_variance().sqrt()
    }

    /// Minimum value seen (infinity if empty).
    pub fn min(&self) -> T {
        self.min
    }

    /// Maximum value seen (neg-infinity if empty).
    pub fn max(&self) -> T {
        self.max
    }

    /// Sample skewness (Fisher's definition).
    pub fn skewness(&self) -> T {
        if self.count < 3 || self.m2 == T::zero() {
            return T::zero();
        }
        let n = T::from_usize(self.count);
        n.sqrt() * self.m3 / (self.m2 * self.m2.sqrt())
    }

    /// Excess kurtosis (Fisher's definition, normal = 0).
    pub fn kurtosis(&self) -> T {
        if self.count < 4 || self.m2 == T::zero() {
            return T::zero();
        }
        let n = T::from_usize(self.count);
        n * self.m4 / (self.m2 * self.m2) - T::from_usize(3)
    }

    /// Merge another `OnlineStats` into this one (parallel-friendly).
    ///
    /// Uses Chan's parallel algorithm for combining partial results.
    pub fn merge(&mut self, other: &Self) {
        if other.count == 0 {
            return;
        }
        if self.count == 0 {
            *self = other.clone();
            return;
        }

        let na = T::from_usize(self.count);
        let nb = T::from_usize(other.count);
        let n = na + nb;
        let delta = other.mean - self.mean;
        let delta2 = delta * delta;
        let delta3 = delta2 * delta;
        let delta4 = delta2 * delta2;

        let new_m2 = self.m2 + other.m2 + delta2 * na * nb / n;
        let new_m3 = self.m3
            + other.m3
            + delta3 * na * nb * (na - nb) / (n * n)
            + T::from_usize(3) * delta * (na * other.m2 - nb * self.m2) / n;
        let new_m4 = self.m4
            + other.m4
            + delta4 * na * nb * (na * na - na * nb + nb * nb) / (n * n * n)
            + T::from_usize(6) * delta2 * (na * na * other.m2 + nb * nb * self.m2) / (n * n)
            + T::from_usize(4) * delta * (na * other.m3 - nb * self.m3) / n;

        self.mean = (na * self.mean + nb * other.mean) / n;
        self.m2 = new_m2;
        self.m3 = new_m3;
        self.m4 = new_m4;
        self.count += other.count;

        if other.min < self.min {
            self.min = other.min;
        }
        if other.max > self.max {
            self.max = other.max;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;

    #[test]
    fn test_online_mean_variance() {
        let data = [2.0_f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let mut stats = OnlineStats::new();
        stats.push_slice(&data);

        assert_eq!(stats.count(), 8);
        assert!((stats.mean() - 5.0).abs() < TOL);
        assert!((stats.variance() - 4.0).abs() < TOL);
        assert!((stats.sample_variance() - 32.0 / 7.0).abs() < TOL);
    }

    #[test]
    fn test_online_min_max() {
        let mut stats = OnlineStats::<f64>::new();
        stats.push_slice(&[3.0, 1.0, 4.0, 1.0, 5.0, 9.0]);
        assert!((stats.min() - 1.0).abs() < TOL);
        assert!((stats.max() - 9.0).abs() < TOL);
    }

    #[test]
    fn test_online_single_element() {
        let mut stats = OnlineStats::<f64>::new();
        stats.push(42.0);
        assert_eq!(stats.count(), 1);
        assert!((stats.mean() - 42.0).abs() < TOL);
        assert!((stats.variance() - 0.0).abs() < TOL);
    }

    #[test]
    fn test_online_merge() {
        let data = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut full = OnlineStats::new();
        full.push_slice(&data);

        let mut a = OnlineStats::new();
        a.push_slice(&data[..4]);
        let mut b = OnlineStats::new();
        b.push_slice(&data[4..]);
        a.merge(&b);

        assert_eq!(a.count(), full.count());
        assert!((a.mean() - full.mean()).abs() < TOL);
        assert!((a.variance() - full.variance()).abs() < TOL);
        assert!((a.min() - full.min()).abs() < TOL);
        assert!((a.max() - full.max()).abs() < TOL);
    }

    #[test]
    fn test_online_empty() {
        let stats = OnlineStats::<f64>::new();
        assert_eq!(stats.count(), 0);
        assert!((stats.mean() - 0.0).abs() < TOL);
        assert!((stats.variance() - 0.0).abs() < TOL);
    }

    #[test]
    fn test_online_std_dev() {
        let mut stats = OnlineStats::<f64>::new();
        stats.push_slice(&[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
        assert!((stats.std_dev() - 2.0).abs() < TOL);
    }

    #[test]
    fn test_online_f32() {
        let mut stats = OnlineStats::<f32>::new();
        stats.push_slice(&[1.0_f32, 2.0, 3.0]);
        assert!((stats.mean() - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_online_skewness_symmetric() {
        // Symmetric data should have ~0 skewness
        let mut stats = OnlineStats::<f64>::new();
        stats.push_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!(stats.skewness().abs() < 1e-10);
    }

    #[test]
    fn test_online_kurtosis_uniform() {
        // Uniform distribution has excess kurtosis of -1.2
        let mut stats = OnlineStats::<f64>::new();
        let data: Vec<f64> = (0..1000).map(|i| i as f64 / 999.0).collect();
        stats.push_slice(&data);
        assert!((stats.kurtosis() - (-1.2)).abs() < 0.01);
    }
}
