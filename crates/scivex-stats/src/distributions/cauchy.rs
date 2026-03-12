use scivex_core::Float;
use scivex_core::random::Rng;

use crate::error::{Result, StatsError};

use super::Distribution;

/// Compute atan(x) using the identity: atan(x) = x * sum_{n=0..} (-1)^n * x^{2n} / (2n+1)
/// with range reduction for convergence.
fn atan_approx<T: Float>(x: T) -> T {
    // Use the relation: convert to f64 and back (Float trait guarantees f64 round-trip)
    let xf = {
        let s = format!("{x:?}");
        s.parse::<f64>().unwrap_or(0.0)
    };
    T::from_f64(xf.atan())
}

/// Cauchy distribution (Lorentz distribution) with location `x0` and scale `gamma`.
///
/// The Cauchy distribution has no finite mean or variance.
#[derive(Debug, Clone, Copy)]
pub struct Cauchy<T: Float> {
    x0: T,
    gamma: T,
}

impl<T: Float> Cauchy<T> {
    /// Create a new Cauchy distribution with location `x0` and scale `gamma > 0`.
    pub fn new(x0: T, gamma: T) -> Result<Self> {
        if gamma <= T::zero() {
            return Err(StatsError::InvalidParameter {
                name: "gamma",
                reason: "scale must be positive",
            });
        }
        Ok(Self { x0, gamma })
    }
}

impl<T: Float> Distribution<T> for Cauchy<T> {
    fn pdf(&self, x: T) -> T {
        let z = (x - self.x0) / self.gamma;
        T::one() / (T::from_f64(core::f64::consts::PI) * self.gamma * (T::one() + z * z))
    }

    fn cdf(&self, x: T) -> T {
        let z = (x - self.x0) / self.gamma;
        T::from_f64(0.5) + atan_approx(z) / T::from_f64(core::f64::consts::PI)
    }

    fn ppf(&self, p: T) -> Result<T> {
        if p <= T::zero() || p >= T::one() {
            return Err(StatsError::InvalidParameter {
                name: "p",
                reason: "must be in (0, 1)",
            });
        }
        // Inverse CDF: x0 + gamma * tan(pi * (p - 0.5))
        let pi = T::from_f64(core::f64::consts::PI);
        Ok(self.x0 + self.gamma * (pi * (p - T::from_f64(0.5))).tan())
    }

    fn sample(&self, rng: &mut Rng) -> T {
        let u = T::from_f64(rng.next_f64());
        let pi = T::from_f64(core::f64::consts::PI);
        self.x0 + self.gamma * (pi * (u - T::from_f64(0.5))).tan()
    }

    fn mean(&self) -> T {
        // Cauchy has no finite mean; return NaN
        T::from_f64(f64::NAN)
    }

    fn variance(&self) -> T {
        // Cauchy has no finite variance; return NaN
        T::from_f64(f64::NAN)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cauchy_invalid() {
        assert!(Cauchy::<f64>::new(0.0, 0.0).is_err());
        assert!(Cauchy::<f64>::new(0.0, -1.0).is_err());
    }

    #[test]
    fn test_cauchy_cdf() {
        let d = Cauchy::<f64>::new(0.0, 1.0).unwrap();
        assert!((d.cdf(0.0) - 0.5).abs() < 1e-10);
        // CDF(1) = 0.5 + atan(1)/pi = 0.5 + 0.25 = 0.75
        assert!((d.cdf(1.0) - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_cauchy_ppf_roundtrip() {
        let d = Cauchy::<f64>::new(2.0, 3.0).unwrap();
        let x = 5.0;
        let p = d.cdf(x);
        let x2 = d.ppf(p).unwrap();
        assert!((x - x2).abs() < 1e-10);
    }

    #[test]
    fn test_cauchy_pdf_at_peak() {
        let d = Cauchy::<f64>::new(0.0, 1.0).unwrap();
        let expected = 1.0 / core::f64::consts::PI;
        assert!((d.pdf(0.0) - expected).abs() < 1e-10);
    }

    #[test]
    fn test_cauchy_mean_variance_nan() {
        let d = Cauchy::<f64>::new(0.0, 1.0).unwrap();
        assert!(d.mean().is_nan());
        assert!(d.variance().is_nan());
    }
}
