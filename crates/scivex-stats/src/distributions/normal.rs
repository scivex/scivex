use scivex_core::Float;
use scivex_core::random::Rng;

use crate::error::{Result, StatsError};
use crate::special::erf;

use super::Distribution;

/// Normal (Gaussian) distribution with mean mu and standard deviation sigma.
#[derive(Debug, Clone, Copy)]
pub struct Normal<T: Float> {
    mu: T,
    sigma: T,
}

impl<T: Float> Normal<T> {
    /// Create a new normal distribution with mean `mu` and standard deviation `sigma > 0`.
    pub fn new(mu: T, sigma: T) -> Result<Self> {
        if sigma <= T::from_f64(0.0) {
            return Err(StatsError::InvalidParameter {
                name: "sigma",
                reason: "must be positive",
            });
        }
        Ok(Self { mu, sigma })
    }

    /// Standard normal distribution N(0, 1).
    pub fn standard() -> Self {
        Self {
            mu: T::from_f64(0.0),
            sigma: T::from_f64(1.0),
        }
    }
}

impl<T: Float> Distribution<T> for Normal<T> {
    fn pdf(&self, x: T) -> T {
        let two = T::from_f64(2.0);
        let sqrt_2pi = (two * T::pi()).sqrt();
        let z = (x - self.mu) / self.sigma;
        (-(z * z) / two).exp() / (self.sigma * sqrt_2pi)
    }

    fn cdf(&self, x: T) -> T {
        let half = T::from_f64(0.5);
        let one = T::from_f64(1.0);
        let sqrt2 = T::from_f64(2.0).sqrt();
        let z = (x - self.mu) / self.sigma;
        half * (one + erf(z / sqrt2))
    }

    fn ppf(&self, p: T) -> Result<T> {
        let zero = T::from_f64(0.0);
        let one = T::from_f64(1.0);
        if p <= zero || p >= one {
            return Err(StatsError::InvalidParameter {
                name: "p",
                reason: "must be in (0, 1)",
            });
        }
        // Acklam's rational approximation for the standard normal quantile
        let z = standard_normal_ppf(p);
        Ok(self.mu + self.sigma * z)
    }

    fn sample(&self, rng: &mut Rng) -> T {
        self.mu + self.sigma * T::from_f64(rng.next_normal_f64())
    }

    fn mean(&self) -> T {
        self.mu
    }

    fn variance(&self) -> T {
        self.sigma * self.sigma
    }
}

/// Acklam's rational approximation for the standard normal quantile function.
fn standard_normal_ppf<T: Float>(p: T) -> T {
    let half = T::from_f64(0.5);

    // Coefficients for the rational approximation
    let a = [
        T::from_f64(-3.969_683_028_665_376e1),
        T::from_f64(2.209_460_984_245_205e2),
        T::from_f64(-2.759_285_104_469_687e2),
        T::from_f64(1.383_577_518_672_69e2),
        T::from_f64(-3.066_479_806_614_716e1),
        T::from_f64(2.506_628_277_459_239),
    ];
    let b = [
        T::from_f64(-5.447_609_879_822_406e1),
        T::from_f64(1.615_858_368_580_409e2),
        T::from_f64(-1.556_989_798_598_866e2),
        T::from_f64(6.680_131_188_771_972e1),
        T::from_f64(-1.328_068_155_288_572e1),
    ];
    let c = [
        T::from_f64(-7.784_894_002_430_293e-3),
        T::from_f64(-3.223_964_580_411_365e-1),
        T::from_f64(-2.400_758_277_161_838),
        T::from_f64(-2.549_732_539_343_734),
        T::from_f64(4.374_664_141_464_968),
        T::from_f64(2.938_163_982_698_783),
    ];
    let d = [
        T::from_f64(7.784_695_709_041_462e-3),
        T::from_f64(3.224_671_290_700_398e-1),
        T::from_f64(2.445_134_137_142_996),
        T::from_f64(3.754_408_661_907_416),
    ];

    let p_low = T::from_f64(0.02425);
    let p_high = T::from_f64(1.0) - p_low;
    let one = T::from_f64(1.0);

    if p < p_low {
        // Rational approximation for lower region
        let q = (-T::from_f64(2.0) * p.ln()).sqrt();
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + one)
    } else if p <= p_high {
        // Rational approximation for central region
        let q = p - half;
        let r = q * q;
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + one)
    } else {
        // Rational approximation for upper region
        let q = (-T::from_f64(2.0) * (one - p).ln()).sqrt();
        -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + one)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normal_invalid() {
        assert!(Normal::<f64>::new(0.0, 0.0).is_err());
        assert!(Normal::<f64>::new(0.0, -1.0).is_err());
    }

    #[test]
    fn test_normal_standard_cdf() {
        let n = Normal::<f64>::standard();
        // CDF(0) = 0.5
        assert!((n.cdf(0.0) - 0.5).abs() < 1e-6);
        // CDF(1.96) ≈ 0.975
        assert!((n.cdf(1.96) - 0.975_002_104_859_278).abs() < 1e-3);
    }

    #[test]
    fn test_normal_pdf_at_mean() {
        let n = Normal::<f64>::new(0.0, 1.0).unwrap();
        let expected = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
        assert!((n.pdf(0.0) - expected).abs() < 1e-10);
    }

    #[test]
    fn test_normal_ppf_roundtrip() {
        let n = Normal::<f64>::new(5.0, 2.0).unwrap();
        let x = 6.5;
        let p = n.cdf(x);
        let x2 = n.ppf(p).unwrap();
        assert!((x - x2).abs() < 1e-4);
    }

    #[test]
    fn test_normal_mean_var() {
        let n = Normal::<f64>::new(3.0, 2.0).unwrap();
        assert!((n.mean() - 3.0).abs() < 1e-10);
        assert!((n.variance() - 4.0).abs() < 1e-10);
    }
}
