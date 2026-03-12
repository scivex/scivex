use scivex_core::Float;
use scivex_core::random::Rng;

use crate::error::{Result, StatsError};

use super::{Distribution, Normal};

/// Log-normal distribution parameterized by `mu` and `sigma` of the underlying
/// normal distribution. If `X ~ Normal(mu, sigma)`, then `exp(X) ~ LogNormal(mu, sigma)`.
#[derive(Debug, Clone, Copy)]
pub struct LogNormal<T: Float> {
    mu: T,
    sigma: T,
    normal: Normal<T>,
}

impl<T: Float> LogNormal<T> {
    /// Create a new log-normal distribution with parameters `mu` and `sigma > 0`.
    pub fn new(mu: T, sigma: T) -> Result<Self> {
        if sigma <= T::zero() {
            return Err(StatsError::InvalidParameter {
                name: "sigma",
                reason: "must be positive",
            });
        }
        let normal = Normal::new(mu, sigma)?;
        Ok(Self { mu, sigma, normal })
    }
}

impl<T: Float> Distribution<T> for LogNormal<T> {
    fn pdf(&self, x: T) -> T {
        if x <= T::zero() {
            return T::zero();
        }
        let ln_x = x.ln();
        let z = (ln_x - self.mu) / self.sigma;
        let coeff = T::one() / (x * self.sigma * T::from_f64(core::f64::consts::TAU).sqrt());
        coeff * (T::from_f64(-0.5) * z * z).exp()
    }

    fn cdf(&self, x: T) -> T {
        if x <= T::zero() {
            return T::zero();
        }
        self.normal.cdf(x.ln())
    }

    fn ppf(&self, p: T) -> Result<T> {
        let z = self.normal.ppf(p)?;
        Ok(z.exp())
    }

    fn sample(&self, rng: &mut Rng) -> T {
        self.normal.sample(rng).exp()
    }

    fn mean(&self) -> T {
        (self.mu + self.sigma * self.sigma / T::from_f64(2.0)).exp()
    }

    fn variance(&self) -> T {
        let s2 = self.sigma * self.sigma;
        ((T::from_f64(2.0) * self.mu + s2).exp()) * (s2.exp() - T::one())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lognormal_invalid() {
        assert!(LogNormal::<f64>::new(0.0, 0.0).is_err());
        assert!(LogNormal::<f64>::new(0.0, -1.0).is_err());
    }

    #[test]
    fn test_lognormal_cdf() {
        let d = LogNormal::<f64>::new(0.0, 1.0).unwrap();
        // CDF(1) = CDF_normal(ln(1)) = CDF_normal(0) = 0.5
        let c = d.cdf(1.0);
        assert!((c - 0.5).abs() < 1e-8, "cdf(1.0)={c}");
        assert!((d.cdf(0.0)).abs() < 1e-10);
    }

    #[test]
    fn test_lognormal_ppf_roundtrip() {
        let d = LogNormal::<f64>::new(0.0, 0.5).unwrap();
        let x = 2.0;
        let p = d.cdf(x);
        let x2 = d.ppf(p).unwrap();
        assert!((x - x2).abs() < 1e-6);
    }

    #[test]
    fn test_lognormal_mean_var() {
        let d = LogNormal::<f64>::new(0.0, 1.0).unwrap();
        let expected_mean = (0.5_f64).exp();
        let expected_var = (1.0_f64.exp() - 1.0) * 1.0_f64.exp();
        assert!((d.mean() - expected_mean).abs() < 1e-10);
        assert!((d.variance() - expected_var).abs() < 1e-10);
    }

    #[test]
    fn test_lognormal_pdf_at_zero() {
        let d = LogNormal::<f64>::new(0.0, 1.0).unwrap();
        assert!((d.pdf(0.0)).abs() < 1e-10);
        assert!((d.pdf(-1.0)).abs() < 1e-10);
    }
}
