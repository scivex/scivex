use scivex_core::Float;
use scivex_core::random::Rng;

use crate::error::{Result, StatsError};
use crate::special::{ln_beta, regularized_beta};

use super::{Distribution, ppf_bisection};

/// Beta distribution with shape parameters alpha and beta.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone, Copy)]
pub struct Beta<T: Float> {
    alpha: T,
    beta_param: T,
}

impl<T: Float> Beta<T> {
    /// Create a new Beta distribution with `alpha > 0` and `beta > 0`.
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
        Ok(Self {
            alpha,
            beta_param: beta,
        })
    }
}

impl<T: Float> Distribution<T> for Beta<T> {
    fn pdf(&self, x: T) -> T {
        let zero = T::from_f64(0.0);
        let one = T::from_f64(1.0);
        if x < zero || x > one {
            return zero;
        }
        let a = self.alpha;
        let b = self.beta_param;
        let log_pdf = (a - one) * x.ln() + (b - one) * (one - x).ln() - ln_beta(a, b);
        log_pdf.exp()
    }

    fn cdf(&self, x: T) -> T {
        let zero = T::from_f64(0.0);
        let one = T::from_f64(1.0);
        if x <= zero {
            return zero;
        }
        if x >= one {
            return one;
        }
        regularized_beta(x, self.alpha, self.beta_param).unwrap_or(zero)
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
        Ok(ppf_bisection(|x| self.cdf(x), p, zero, one))
    }

    fn sample(&self, rng: &mut Rng) -> T {
        // Sample using gamma ratio: X = G1/(G1+G2) where G1 ~ Gamma(alpha,1), G2 ~ Gamma(beta,1)
        let one = T::from_f64(1.0);
        let g1 =
            super::gamma::Gamma::new(self.alpha, one).expect("alpha validated at construction");
        let g2 =
            super::gamma::Gamma::new(self.beta_param, one).expect("beta validated at construction");
        let x1 = g1.sample(rng);
        let x2 = g2.sample(rng);
        x1 / (x1 + x2)
    }

    fn mean(&self) -> T {
        self.alpha / (self.alpha + self.beta_param)
    }

    fn variance(&self) -> T {
        let a = self.alpha;
        let b = self.beta_param;
        let one = T::from_f64(1.0);
        (a * b) / ((a + b) * (a + b) * (a + b + one))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_beta_invalid() {
        assert!(Beta::<f64>::new(0.0, 1.0).is_err());
        assert!(Beta::<f64>::new(1.0, 0.0).is_err());
    }

    #[test]
    fn test_beta_uniform_case() {
        // Beta(1, 1) = Uniform(0, 1)
        let b = Beta::<f64>::new(1.0, 1.0).unwrap();
        assert!((b.cdf(0.5) - 0.5).abs() < 1e-6);
        assert!((b.pdf(0.5) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_beta_mean_var() {
        let b = Beta::<f64>::new(2.0, 5.0).unwrap();
        assert!((b.mean() - 2.0 / 7.0).abs() < 1e-10);
        let expected_var = 10.0 / (49.0 * 8.0);
        assert!((b.variance() - expected_var).abs() < 1e-10);
    }

    #[test]
    fn test_beta_ppf_roundtrip() {
        let b = Beta::<f64>::new(2.0, 3.0).unwrap();
        let x = 0.4;
        let p = b.cdf(x);
        let x2 = b.ppf(p).unwrap();
        assert!((x - x2).abs() < 1e-4);
    }
}
