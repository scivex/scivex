use scivex_core::Float;
use scivex_core::random::Rng;

use crate::error::{Result, StatsError};

use super::Distribution;

/// Laplace (double exponential) distribution with location `mu` and scale `b`.
///
/// PDF: `1/(2b) * exp(-|x - mu| / b)`
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone, Copy)]
pub struct Laplace<T: Float> {
    mu: T,
    b: T,
}

impl<T: Float> Laplace<T> {
    /// Create a new Laplace distribution with location `mu` and scale `b > 0`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_stats::distributions::{Laplace, Distribution};
    /// let l = Laplace::new(0.0_f64, 1.0).unwrap();
    /// assert!((l.mean() - 0.0).abs() < 1e-10);
    /// assert!((l.variance() - 2.0).abs() < 1e-10);
    /// ```
    pub fn new(mu: T, b: T) -> Result<Self> {
        if b <= T::zero() {
            return Err(StatsError::InvalidParameter {
                name: "b",
                reason: "scale must be positive",
            });
        }
        Ok(Self { mu, b })
    }
}

impl<T: Float> Distribution<T> for Laplace<T> {
    fn pdf(&self, x: T) -> T {
        let half = T::from_f64(0.5);
        half / self.b * (-((x - self.mu).abs()) / self.b).exp()
    }

    fn cdf(&self, x: T) -> T {
        let half = T::from_f64(0.5);
        if x < self.mu {
            half * ((x - self.mu) / self.b).exp()
        } else {
            T::one() - half * (-(x - self.mu) / self.b).exp()
        }
    }

    fn ppf(&self, p: T) -> Result<T> {
        if p <= T::zero() || p >= T::one() {
            return Err(StatsError::InvalidParameter {
                name: "p",
                reason: "must be in (0, 1)",
            });
        }
        let half = T::from_f64(0.5);
        if p < half {
            Ok(self.mu + self.b * (T::from_f64(2.0) * p).ln())
        } else {
            Ok(self.mu - self.b * (T::from_f64(2.0) * (T::one() - p)).ln())
        }
    }

    fn sample(&self, rng: &mut Rng) -> T {
        let u = T::from_f64(rng.next_f64()) - T::from_f64(0.5);
        self.mu - self.b * u.abs().ln() * if u < T::zero() { -T::one() } else { T::one() }
    }

    fn mean(&self) -> T {
        self.mu
    }

    fn variance(&self) -> T {
        T::from_f64(2.0) * self.b * self.b
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_laplace_invalid() {
        assert!(Laplace::<f64>::new(0.0, 0.0).is_err());
        assert!(Laplace::<f64>::new(0.0, -1.0).is_err());
    }

    #[test]
    fn test_laplace_cdf() {
        let d = Laplace::<f64>::new(0.0, 1.0).unwrap();
        assert!((d.cdf(0.0) - 0.5).abs() < 1e-10);
        // CDF(1) = 1 - 0.5*exp(-1) ≈ 0.8161
        let expected = 1.0 - 0.5 * (-1.0_f64).exp();
        assert!((d.cdf(1.0) - expected).abs() < 1e-10);
    }

    #[test]
    fn test_laplace_ppf_roundtrip() {
        let d = Laplace::<f64>::new(1.0, 2.0).unwrap();
        for &x in &[-3.0, 0.0, 1.0, 5.0] {
            let p = d.cdf(x);
            let x2 = d.ppf(p).unwrap();
            assert!((x - x2).abs() < 1e-10, "roundtrip failed for x={x}");
        }
    }

    #[test]
    fn test_laplace_mean_var() {
        let d = Laplace::<f64>::new(3.0, 2.0).unwrap();
        assert!((d.mean() - 3.0).abs() < 1e-10);
        assert!((d.variance() - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_laplace_symmetry() {
        let d = Laplace::<f64>::new(0.0, 1.0).unwrap();
        assert!((d.pdf(1.0) - d.pdf(-1.0)).abs() < 1e-10);
        assert!((d.cdf(-1.0) + d.cdf(1.0) - 1.0).abs() < 1e-10);
    }
}
