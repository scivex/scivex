use scivex_core::Float;
use scivex_core::random::Rng;

use crate::error::{Result, StatsError};

use super::Distribution;

/// Continuous uniform distribution on [a, b].
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone, Copy)]
pub struct Uniform<T: Float> {
    a: T,
    b: T,
}

impl<T: Float> Uniform<T> {
    /// Create a new uniform distribution on [a, b].
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_stats::distributions::{Uniform, Distribution};
    /// let u = Uniform::new(0.0_f64, 1.0).unwrap();
    /// assert!((u.cdf(0.5) - 0.5).abs() < 1e-10);
    /// assert!((u.mean() - 0.5).abs() < 1e-10);
    /// ```
    pub fn new(a: T, b: T) -> Result<Self> {
        if a >= b {
            return Err(StatsError::InvalidParameter {
                name: "a, b",
                reason: "requires a < b",
            });
        }
        Ok(Self { a, b })
    }
}

impl<T: Float> Distribution<T> for Uniform<T> {
    fn pdf(&self, x: T) -> T {
        if x >= self.a && x <= self.b {
            T::from_f64(1.0) / (self.b - self.a)
        } else {
            T::from_f64(0.0)
        }
    }

    fn cdf(&self, x: T) -> T {
        let zero = T::from_f64(0.0);
        let one = T::from_f64(1.0);
        if x < self.a {
            zero
        } else if x > self.b {
            one
        } else {
            (x - self.a) / (self.b - self.a)
        }
    }

    fn ppf(&self, p: T) -> Result<T> {
        let zero = T::from_f64(0.0);
        let one = T::from_f64(1.0);
        if p < zero || p > one {
            return Err(StatsError::InvalidParameter {
                name: "p",
                reason: "must be in [0, 1]",
            });
        }
        Ok(self.a + p * (self.b - self.a))
    }

    fn sample(&self, rng: &mut Rng) -> T {
        self.a + T::from_f64(rng.next_f64()) * (self.b - self.a)
    }

    fn mean(&self) -> T {
        (self.a + self.b) / T::from_f64(2.0)
    }

    fn variance(&self) -> T {
        let range = self.b - self.a;
        range * range / T::from_f64(12.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_invalid() {
        assert!(Uniform::<f64>::new(5.0, 2.0).is_err());
        assert!(Uniform::<f64>::new(1.0, 1.0).is_err());
    }

    #[test]
    fn test_uniform_cdf_pdf() {
        let u = Uniform::<f64>::new(0.0, 1.0).unwrap();
        assert!((u.pdf(0.5) - 1.0).abs() < 1e-10);
        assert!((u.cdf(0.5) - 0.5).abs() < 1e-10);
        assert!((u.cdf(0.0)).abs() < 1e-10);
        assert!((u.cdf(1.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_uniform_ppf_roundtrip() {
        let u = Uniform::<f64>::new(2.0, 8.0).unwrap();
        let x = 4.5;
        let p = u.cdf(x);
        let x2 = u.ppf(p).unwrap();
        assert!((x - x2).abs() < 1e-10);
    }

    #[test]
    fn test_uniform_mean_var() {
        let u = Uniform::<f64>::new(0.0, 12.0).unwrap();
        assert!((u.mean() - 6.0).abs() < 1e-10);
        assert!((u.variance() - 12.0).abs() < 1e-10);
    }
}
