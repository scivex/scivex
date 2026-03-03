use scivex_core::Float;
use scivex_core::random::Rng;

use crate::error::{Result, StatsError};

use super::Distribution;
use super::gamma::Gamma;

/// Chi-squared distribution with `df` degrees of freedom.
///
/// Internally delegates to `Gamma(df/2, 0.5)`.
#[derive(Debug, Clone, Copy)]
pub struct ChiSquared<T: Float> {
    df: T,
    inner: Gamma<T>,
}

impl<T: Float> ChiSquared<T> {
    /// Create a new chi-squared distribution with `df > 0` degrees of freedom.
    pub fn new(df: T) -> Result<Self> {
        if df <= T::from_f64(0.0) {
            return Err(StatsError::InvalidParameter {
                name: "df",
                reason: "must be positive",
            });
        }
        let half = T::from_f64(0.5);
        let inner = Gamma::new(df * half, half)?;
        Ok(Self { df, inner })
    }
}

impl<T: Float> Distribution<T> for ChiSquared<T> {
    fn pdf(&self, x: T) -> T {
        self.inner.pdf(x)
    }

    fn cdf(&self, x: T) -> T {
        self.inner.cdf(x)
    }

    fn ppf(&self, p: T) -> Result<T> {
        self.inner.ppf(p)
    }

    fn sample(&self, rng: &mut Rng) -> T {
        self.inner.sample(rng)
    }

    fn mean(&self) -> T {
        self.df
    }

    fn variance(&self) -> T {
        T::from_f64(2.0) * self.df
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chi_squared_invalid() {
        assert!(ChiSquared::<f64>::new(0.0).is_err());
        assert!(ChiSquared::<f64>::new(-1.0).is_err());
    }

    #[test]
    fn test_chi_squared_mean_var() {
        let c = ChiSquared::<f64>::new(5.0).unwrap();
        assert!((c.mean() - 5.0).abs() < 1e-10);
        assert!((c.variance() - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_chi_squared_cdf() {
        // P(X <= 3.841) ≈ 0.95 for df=1
        let c = ChiSquared::<f64>::new(1.0).unwrap();
        let p = c.cdf(3.841);
        assert!((p - 0.95).abs() < 0.01);
    }
}
