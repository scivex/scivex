use scivex_core::Float;
use scivex_core::random::Rng;

use crate::error::{Result, StatsError};

use super::Distribution;

/// Pareto distribution (Type I) with shape `alpha` and scale `x_m`.
///
/// PDF: `alpha * x_m^alpha / x^(alpha+1)` for `x >= x_m`.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone, Copy)]
pub struct Pareto<T: Float> {
    alpha: T,
    x_m: T,
}

impl<T: Float> Pareto<T> {
    /// Create a new Pareto distribution with shape `alpha > 0` and scale `x_m > 0`.
    pub fn new(alpha: T, x_m: T) -> Result<Self> {
        if alpha <= T::zero() {
            return Err(StatsError::InvalidParameter {
                name: "alpha",
                reason: "shape must be positive",
            });
        }
        if x_m <= T::zero() {
            return Err(StatsError::InvalidParameter {
                name: "x_m",
                reason: "scale must be positive",
            });
        }
        Ok(Self { alpha, x_m })
    }
}

impl<T: Float> Distribution<T> for Pareto<T> {
    fn pdf(&self, x: T) -> T {
        if x < self.x_m {
            return T::zero();
        }
        self.alpha * self.x_m.powf(self.alpha) / x.powf(self.alpha + T::one())
    }

    fn cdf(&self, x: T) -> T {
        if x < self.x_m {
            return T::zero();
        }
        T::one() - (self.x_m / x).powf(self.alpha)
    }

    fn ppf(&self, p: T) -> Result<T> {
        if p < T::zero() || p >= T::one() {
            return Err(StatsError::InvalidParameter {
                name: "p",
                reason: "must be in [0, 1)",
            });
        }
        // Inverse CDF: x_m / (1-p)^(1/alpha)
        Ok(self.x_m / (T::one() - p).powf(T::one() / self.alpha))
    }

    fn sample(&self, rng: &mut Rng) -> T {
        let u = T::from_f64(rng.next_f64());
        self.x_m / (T::one() - u).powf(T::one() / self.alpha)
    }

    fn mean(&self) -> T {
        if self.alpha <= T::one() {
            T::from_f64(f64::INFINITY)
        } else {
            self.alpha * self.x_m / (self.alpha - T::one())
        }
    }

    fn variance(&self) -> T {
        let two = T::from_f64(2.0);
        if self.alpha <= two {
            T::from_f64(f64::INFINITY)
        } else {
            let a = self.alpha;
            let xm = self.x_m;
            (xm * xm * a) / ((a - T::one()) * (a - T::one()) * (a - two))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pareto_invalid() {
        assert!(Pareto::<f64>::new(0.0, 1.0).is_err());
        assert!(Pareto::<f64>::new(1.0, 0.0).is_err());
    }

    #[test]
    fn test_pareto_cdf() {
        let d = Pareto::<f64>::new(1.0, 1.0).unwrap();
        assert!((d.cdf(1.0)).abs() < 1e-10);
        assert!((d.cdf(2.0) - 0.5).abs() < 1e-10);
        assert!((d.cdf(0.5)).abs() < 1e-10); // below x_m
    }

    #[test]
    fn test_pareto_ppf_roundtrip() {
        let d = Pareto::<f64>::new(2.0, 1.0).unwrap();
        let x = 3.0;
        let p = d.cdf(x);
        let x2 = d.ppf(p).unwrap();
        assert!((x - x2).abs() < 1e-10);
    }

    #[test]
    fn test_pareto_mean_var() {
        let d = Pareto::<f64>::new(3.0, 1.0).unwrap();
        // Mean = 3/(3-1) = 1.5
        assert!((d.mean() - 1.5).abs() < 1e-10);
        // Var = 3 / (4 * 1) = 0.75
        let expected_var = 3.0 / (4.0 * 1.0);
        assert!((d.variance() - expected_var).abs() < 1e-10);
    }

    #[test]
    fn test_pareto_infinite_mean() {
        let d = Pareto::<f64>::new(1.0, 1.0).unwrap();
        assert!(d.mean().is_infinite());
    }
}
