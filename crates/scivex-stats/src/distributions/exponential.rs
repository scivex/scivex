use scivex_core::Float;
use scivex_core::random::Rng;

use crate::error::{Result, StatsError};

use super::Distribution;

/// Exponential distribution with rate parameter lambda.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone, Copy)]
pub struct Exponential<T: Float> {
    lambda: T,
}

impl<T: Float> Exponential<T> {
    /// Create a new exponential distribution with rate `lambda > 0`.
    pub fn new(lambda: T) -> Result<Self> {
        if lambda <= T::from_f64(0.0) {
            return Err(StatsError::InvalidParameter {
                name: "lambda",
                reason: "must be positive",
            });
        }
        Ok(Self { lambda })
    }
}

impl<T: Float> Distribution<T> for Exponential<T> {
    fn pdf(&self, x: T) -> T {
        if x < T::from_f64(0.0) {
            T::from_f64(0.0)
        } else {
            self.lambda * (-self.lambda * x).exp()
        }
    }

    fn cdf(&self, x: T) -> T {
        if x < T::from_f64(0.0) {
            T::from_f64(0.0)
        } else {
            T::from_f64(1.0) - (-self.lambda * x).exp()
        }
    }

    fn ppf(&self, p: T) -> Result<T> {
        let zero = T::from_f64(0.0);
        let one = T::from_f64(1.0);
        if p < zero || p >= one {
            return Err(StatsError::InvalidParameter {
                name: "p",
                reason: "must be in [0, 1)",
            });
        }
        Ok(-(one - p).ln() / self.lambda)
    }

    fn sample(&self, rng: &mut Rng) -> T {
        // Inverse transform: -ln(1 - U) / lambda
        let u = T::from_f64(rng.next_f64());
        let one = T::from_f64(1.0);
        -(one - u).ln() / self.lambda
    }

    fn mean(&self) -> T {
        T::from_f64(1.0) / self.lambda
    }

    fn variance(&self) -> T {
        T::from_f64(1.0) / (self.lambda * self.lambda)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exponential_invalid() {
        assert!(Exponential::<f64>::new(0.0).is_err());
        assert!(Exponential::<f64>::new(-1.0).is_err());
    }

    #[test]
    fn test_exponential_cdf() {
        let e = Exponential::<f64>::new(1.0).unwrap();
        // CDF(1) = 1 - e^{-1} ≈ 0.6321
        assert!((e.cdf(1.0) - 0.632_120_558_828_558).abs() < 1e-6);
        assert!((e.cdf(0.0)).abs() < 1e-10);
    }

    #[test]
    fn test_exponential_ppf_roundtrip() {
        let e = Exponential::<f64>::new(2.0).unwrap();
        let x = 1.5;
        let p = e.cdf(x);
        let x2 = e.ppf(p).unwrap();
        assert!((x - x2).abs() < 1e-10);
    }

    #[test]
    fn test_exponential_mean_var() {
        let e = Exponential::<f64>::new(0.5).unwrap();
        assert!((e.mean() - 2.0).abs() < 1e-10);
        assert!((e.variance() - 4.0).abs() < 1e-10);
    }
}
