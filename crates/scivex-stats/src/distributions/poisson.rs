use scivex_core::Float;
use scivex_core::random::Rng;

use crate::error::{Result, StatsError};
use crate::special::{ln_gamma, regularized_gamma_q};

use super::{Distribution, ppf_bisection};

/// Poisson distribution with rate parameter lambda.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone, Copy)]
pub struct Poisson<T: Float> {
    lambda: T,
}

impl<T: Float> Poisson<T> {
    /// Create a new Poisson distribution with `lambda > 0`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_stats::distributions::{Poisson, Distribution};
    /// let p = Poisson::new(5.0_f64).unwrap();
    /// assert!((p.mean() - 5.0).abs() < 1e-10);
    /// ```
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

impl<T: Float> Distribution<T> for Poisson<T> {
    fn pdf(&self, x: T) -> T {
        let zero = T::from_f64(0.0);
        if x < zero {
            return zero;
        }
        let k = x.floor();
        if (k - x).abs() > T::epsilon() {
            return zero; // not an integer
        }
        let one = T::from_f64(1.0);
        // PMF = lambda^k * e^{-lambda} / k!
        // log PMF = k*ln(lambda) - lambda - ln_gamma(k+1)
        let log_pmf = k * self.lambda.ln() - self.lambda - ln_gamma(k + one);
        log_pmf.exp()
    }

    fn cdf(&self, x: T) -> T {
        let zero = T::from_f64(0.0);
        if x < zero {
            return zero;
        }
        let k = x.floor();
        let one = T::from_f64(1.0);
        // CDF = Q(k+1, lambda) = 1 - P(k+1, lambda) = regularized_gamma_q(k+1, lambda)
        regularized_gamma_q(k + one, self.lambda).unwrap_or(zero)
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
        // Search for smallest k such that CDF(k) >= p
        let mut k = zero;
        loop {
            if self.cdf(k) >= p {
                return Ok(k);
            }
            k += one;
            if k > T::from_f64(10000.0) {
                // Fallback: bisection on continuous approximation
                let hi = self.lambda + T::from_f64(20.0) * self.lambda.sqrt();
                let val = ppf_bisection(|x| self.cdf(x), p, zero, hi);
                return Ok(val.floor());
            }
        }
    }

    fn sample(&self, rng: &mut Rng) -> T {
        // Knuth algorithm for small lambda, rejection for large
        let one = T::from_f64(1.0);
        let limit = T::from_f64(30.0);

        if self.lambda < limit {
            // Knuth: count events until product falls below threshold
            let l = (-self.lambda).exp();
            let mut k = T::from_f64(0.0);
            let mut p = one;
            loop {
                p *= T::from_f64(rng.next_f64());
                if p <= l {
                    return k;
                }
                k += one;
            }
        } else {
            // Normal approximation for large lambda, rounded
            let z = T::from_f64(rng.next_normal_f64());
            let x = self.lambda + z * self.lambda.sqrt();
            T::from_f64(0.0).max(x.round())
        }
    }

    fn mean(&self) -> T {
        self.lambda
    }

    fn variance(&self) -> T {
        self.lambda
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_poisson_invalid() {
        assert!(Poisson::<f64>::new(0.0).is_err());
        assert!(Poisson::<f64>::new(-1.0).is_err());
    }

    #[test]
    fn test_poisson_pmf() {
        let p = Poisson::<f64>::new(3.0).unwrap();
        // P(X=0) = e^{-3}
        assert!((p.pdf(0.0) - (-3.0_f64).exp()).abs() < 1e-10);
        // P(X=3) = 3^3 * e^{-3} / 6
        let expected = 27.0 * (-3.0_f64).exp() / 6.0;
        assert!((p.pdf(3.0) - expected).abs() < 1e-8);
    }

    #[test]
    fn test_poisson_cdf() {
        let p = Poisson::<f64>::new(1.0).unwrap();
        // CDF(0) = e^{-1} ≈ 0.3679
        assert!((p.cdf(0.0) - (-1.0_f64).exp()).abs() < 1e-4);
    }

    #[test]
    fn test_poisson_mean_var() {
        let p = Poisson::<f64>::new(5.0).unwrap();
        assert!((p.mean() - 5.0).abs() < 1e-10);
        assert!((p.variance() - 5.0).abs() < 1e-10);
    }
}
