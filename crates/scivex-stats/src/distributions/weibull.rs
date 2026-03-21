use scivex_core::Float;
use scivex_core::random::Rng;

use crate::error::{Result, StatsError};
use crate::special::ln_gamma;

use super::Distribution;

/// Weibull distribution with shape parameter `k` and scale parameter `lambda`.
///
/// PDF: `(k / lambda) * (x / lambda)^(k-1) * exp(-(x/lambda)^k)` for `x >= 0`.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone, Copy)]
pub struct Weibull<T: Float> {
    k: T,
    lambda: T,
}

impl<T: Float> Weibull<T> {
    /// Create a new Weibull distribution with shape `k > 0` and scale `lambda > 0`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_stats::distributions::{Weibull, Distribution};
    /// let w = Weibull::new(1.0_f64, 2.0).unwrap();
    /// assert!(w.pdf(1.0) > 0.0);
    /// ```
    pub fn new(k: T, lambda: T) -> Result<Self> {
        if k <= T::zero() {
            return Err(StatsError::InvalidParameter {
                name: "k",
                reason: "shape must be positive",
            });
        }
        if lambda <= T::zero() {
            return Err(StatsError::InvalidParameter {
                name: "lambda",
                reason: "scale must be positive",
            });
        }
        Ok(Self { k, lambda })
    }
}

impl<T: Float> Distribution<T> for Weibull<T> {
    fn pdf(&self, x: T) -> T {
        if x < T::zero() {
            return T::zero();
        }
        if x == T::zero() {
            return if self.k < T::one() {
                T::from_f64(f64::INFINITY)
            } else if self.k == T::one() {
                self.k / self.lambda
            } else {
                T::zero()
            };
        }
        let ratio = x / self.lambda;
        (self.k / self.lambda) * ratio.powf(self.k - T::one()) * (-(ratio.powf(self.k))).exp()
    }

    fn cdf(&self, x: T) -> T {
        if x < T::zero() {
            return T::zero();
        }
        T::one() - (-(x / self.lambda).powf(self.k)).exp()
    }

    fn ppf(&self, p: T) -> Result<T> {
        if p < T::zero() || p >= T::one() {
            return Err(StatsError::InvalidParameter {
                name: "p",
                reason: "must be in [0, 1)",
            });
        }
        // Inverse CDF: lambda * (-ln(1-p))^(1/k)
        Ok(self.lambda * (-(T::one() - p).ln()).powf(T::one() / self.k))
    }

    fn sample(&self, rng: &mut Rng) -> T {
        let u = T::from_f64(rng.next_f64());
        self.lambda * (-(T::one() - u).ln()).powf(T::one() / self.k)
    }

    fn mean(&self) -> T {
        // lambda * Gamma(1 + 1/k)
        let g = ln_gamma(T::one() + T::one() / self.k);
        self.lambda * g.exp()
    }

    fn variance(&self) -> T {
        // lambda^2 * [Gamma(1 + 2/k) - Gamma(1 + 1/k)^2]
        let g1 = ln_gamma(T::one() + T::one() / self.k).exp();
        let g2 = ln_gamma(T::one() + T::from_f64(2.0) / self.k).exp();
        self.lambda * self.lambda * (g2 - g1 * g1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weibull_invalid() {
        assert!(Weibull::<f64>::new(0.0, 1.0).is_err());
        assert!(Weibull::<f64>::new(1.0, 0.0).is_err());
        assert!(Weibull::<f64>::new(-1.0, 1.0).is_err());
    }

    #[test]
    fn test_weibull_cdf() {
        // k=1, lambda=1 => Exponential(1): CDF(1) = 1 - e^{-1}
        let d = Weibull::<f64>::new(1.0, 1.0).unwrap();
        assert!((d.cdf(1.0) - (1.0 - (-1.0_f64).exp())).abs() < 1e-10);
        assert!((d.cdf(0.0)).abs() < 1e-10);
    }

    #[test]
    fn test_weibull_ppf_roundtrip() {
        let d = Weibull::<f64>::new(2.0, 3.0).unwrap();
        let x = 2.5;
        let p = d.cdf(x);
        let x2 = d.ppf(p).unwrap();
        assert!((x - x2).abs() < 1e-10);
    }

    #[test]
    fn test_weibull_exponential_special_case() {
        // Weibull(k=1, lambda) = Exponential(1/lambda)
        let d = Weibull::<f64>::new(1.0, 2.0).unwrap();
        assert!((d.mean() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_weibull_mean_var() {
        let d = Weibull::<f64>::new(2.0, 1.0).unwrap();
        // Mean = Gamma(1.5) = sqrt(pi)/2 ≈ 0.8862
        let expected_mean = core::f64::consts::PI.sqrt() / 2.0;
        assert!((d.mean() - expected_mean).abs() < 1e-4);
    }
}
