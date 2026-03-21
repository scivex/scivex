use scivex_core::Float;
use scivex_core::random::Rng;

use crate::error::{Result, StatsError};
use crate::special::{ln_gamma, regularized_gamma_p};

use super::{Distribution, ppf_bisection};

/// Gamma distribution with shape `alpha` and rate `beta`.
///
/// PDF: f(x) = beta^alpha * x^{alpha-1} * e^{-beta*x} / Gamma(alpha)
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone, Copy)]
pub struct Gamma<T: Float> {
    alpha: T,
    beta: T,
}

impl<T: Float> Gamma<T> {
    /// Create a new Gamma distribution with shape `alpha > 0` and rate `beta > 0`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_stats::distributions::{Gamma, Distribution};
    /// let g = Gamma::new(3.0_f64, 2.0).unwrap();
    /// assert!((g.mean() - 1.5).abs() < 1e-10); // alpha/beta
    /// ```
    pub fn new(alpha: T, beta: T) -> Result<Self> {
        let zero = T::from_f64(0.0);
        if alpha <= zero {
            return Err(StatsError::InvalidParameter {
                name: "alpha",
                reason: "must be positive",
            });
        }
        if beta <= zero {
            return Err(StatsError::InvalidParameter {
                name: "beta",
                reason: "must be positive",
            });
        }
        Ok(Self { alpha, beta })
    }

    /// Access the shape parameter.
    pub fn alpha(&self) -> T {
        self.alpha
    }

    /// Access the rate parameter.
    pub fn beta(&self) -> T {
        self.beta
    }
}

impl<T: Float> Distribution<T> for Gamma<T> {
    fn pdf(&self, x: T) -> T {
        let zero = T::from_f64(0.0);
        let one = T::from_f64(1.0);
        if x < zero {
            return zero;
        }
        if x == zero {
            if self.alpha < one {
                return T::infinity();
            } else if self.alpha == one {
                return self.beta;
            }
            return zero;
        }
        let log_pdf = self.alpha * self.beta.ln() + (self.alpha - one) * x.ln()
            - self.beta * x
            - ln_gamma(self.alpha);
        log_pdf.exp()
    }

    fn cdf(&self, x: T) -> T {
        let zero = T::from_f64(0.0);
        if x <= zero {
            return zero;
        }
        // CDF = P(alpha, beta * x) where P is the lower regularized incomplete gamma
        regularized_gamma_p(self.alpha, self.beta * x).unwrap_or(zero)
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
        if p == zero {
            return Ok(zero);
        }
        // Use bisection with a reasonable upper bound
        let hi = self.mean() + T::from_f64(20.0) * self.variance().sqrt();
        Ok(ppf_bisection(|x| self.cdf(x), p, zero, hi))
    }

    fn sample(&self, rng: &mut Rng) -> T {
        // Marsaglia & Tsang's method for alpha >= 1
        let one = T::from_f64(1.0);

        if self.alpha < one {
            // For alpha < 1: sample Gamma(alpha+1) and multiply by U^{1/alpha}
            let g1 = Gamma::new(self.alpha + one, one).expect("alpha+1 always valid");
            let x = g1.sample(rng);
            let u = T::from_f64(rng.next_f64());
            return x * u.powf(one / self.alpha) / self.beta;
        }

        let d = self.alpha - T::from_f64(1.0 / 3.0);
        let c = one / (T::from_f64(9.0) * d).sqrt();

        loop {
            let x = T::from_f64(rng.next_normal_f64());
            let v = one + c * x;
            if v <= T::from_f64(0.0) {
                continue;
            }
            let v = v * v * v;
            let u = T::from_f64(rng.next_f64());
            let x2 = x * x;

            if u < one - T::from_f64(0.0331) * x2 * x2 {
                return d * v / self.beta;
            }
            if u.ln() < T::from_f64(0.5) * x2 + d * (one - v + v.ln()) {
                return d * v / self.beta;
            }
        }
    }

    fn mean(&self) -> T {
        self.alpha / self.beta
    }

    fn variance(&self) -> T {
        self.alpha / (self.beta * self.beta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gamma_invalid() {
        assert!(Gamma::<f64>::new(0.0, 1.0).is_err());
        assert!(Gamma::<f64>::new(1.0, 0.0).is_err());
    }

    #[test]
    fn test_gamma_cdf_exponential_case() {
        // Gamma(1, lambda) = Exponential(lambda)
        let g = Gamma::<f64>::new(1.0, 1.0).unwrap();
        let expected = 1.0 - (-1.0_f64).exp(); // ≈ 0.6321
        assert!((g.cdf(1.0) - expected).abs() < 1e-6);
    }

    #[test]
    fn test_gamma_mean_var() {
        let g = Gamma::<f64>::new(3.0, 2.0).unwrap();
        assert!((g.mean() - 1.5).abs() < 1e-10);
        assert!((g.variance() - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_gamma_ppf_roundtrip() {
        let g = Gamma::<f64>::new(2.0, 1.0).unwrap();
        let x = 3.0;
        let p = g.cdf(x);
        let x2 = g.ppf(p).unwrap();
        assert!((x - x2).abs() < 1e-4);
    }

    #[test]
    fn test_gamma_sampling_stats() {
        let g = Gamma::<f64>::new(5.0, 2.0).unwrap();
        let mut rng = Rng::new(42);
        let samples = g.sample_n(&mut rng, 50_000);
        let m: f64 = samples.iter().sum::<f64>() / samples.len() as f64;
        assert!((m - g.mean()).abs() < 0.05);
    }
}
