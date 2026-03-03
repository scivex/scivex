use scivex_core::Float;
use scivex_core::random::Rng;

use crate::error::{Result, StatsError};
use crate::special::{ln_gamma, regularized_beta};

use super::{Distribution, ppf_bisection};

/// Binomial distribution with `n` trials and success probability `p`.
#[derive(Debug, Clone, Copy)]
pub struct Binomial<T: Float> {
    n: usize,
    p: T,
}

impl<T: Float> Binomial<T> {
    /// Create a new binomial distribution with `n >= 1` trials and `p` in [0, 1].
    pub fn new(n: usize, p: T) -> Result<Self> {
        let zero = T::from_f64(0.0);
        let one = T::from_f64(1.0);
        if n == 0 {
            return Err(StatsError::InvalidParameter {
                name: "n",
                reason: "must be at least 1",
            });
        }
        if p < zero || p > one {
            return Err(StatsError::InvalidParameter {
                name: "p",
                reason: "must be in [0, 1]",
            });
        }
        Ok(Self { n, p })
    }
}

impl<T: Float> Distribution<T> for Binomial<T> {
    fn pdf(&self, x: T) -> T {
        let zero = T::from_f64(0.0);
        let one = T::from_f64(1.0);
        let k = x.floor();
        if (k - x).abs() > T::epsilon() || k < zero || k > T::from_f64(self.n as f64) {
            return zero;
        }
        let nf = T::from_f64(self.n as f64);
        // log PMF = ln_gamma(n+1) - ln_gamma(k+1) - ln_gamma(n-k+1) + k*ln(p) + (n-k)*ln(1-p)
        let log_binom = ln_gamma(nf + one) - ln_gamma(k + one) - ln_gamma(nf - k + one);
        let log_pmf = log_binom + k * self.p.ln() + (nf - k) * (one - self.p).ln();
        log_pmf.exp()
    }

    fn cdf(&self, x: T) -> T {
        let zero = T::from_f64(0.0);
        let one = T::from_f64(1.0);
        if x < zero {
            return zero;
        }
        let nf = T::from_f64(self.n as f64);
        if x >= nf {
            return one;
        }
        let k = x.floor();
        // CDF = I_{1-p}(n-k, k+1)
        regularized_beta(one - self.p, nf - k, k + one).unwrap_or(zero)
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
        // Linear search for small n, bisection for large
        let nf = T::from_f64(self.n as f64);
        if self.n <= 1000 {
            let mut k = zero;
            loop {
                if self.cdf(k) >= p {
                    return Ok(k);
                }
                k += one;
                if k > nf {
                    return Ok(nf);
                }
            }
        } else {
            let val = ppf_bisection(|x| self.cdf(x), p, zero, nf);
            Ok(val.floor())
        }
    }

    fn sample(&self, rng: &mut Rng) -> T {
        // Direct simulation for small n, normal approximation for large
        let one = T::from_f64(1.0);
        if self.n <= 50 {
            let mut count = T::from_f64(0.0);
            for _ in 0..self.n {
                if T::from_f64(rng.next_f64()) < self.p {
                    count += one;
                }
            }
            count
        } else {
            // Normal approximation
            let nf = T::from_f64(self.n as f64);
            let mu = nf * self.p;
            let sigma = (nf * self.p * (one - self.p)).sqrt();
            let z = T::from_f64(rng.next_normal_f64());
            let x = (mu + sigma * z).round();
            x.max(T::from_f64(0.0)).min(nf)
        }
    }

    fn mean(&self) -> T {
        T::from_f64(self.n as f64) * self.p
    }

    fn variance(&self) -> T {
        let one = T::from_f64(1.0);
        T::from_f64(self.n as f64) * self.p * (one - self.p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binomial_invalid() {
        assert!(Binomial::<f64>::new(0, 0.5).is_err());
        assert!(Binomial::<f64>::new(10, -0.1).is_err());
        assert!(Binomial::<f64>::new(10, 1.1).is_err());
    }

    #[test]
    fn test_binomial_pmf() {
        let b = Binomial::<f64>::new(10, 0.5).unwrap();
        // P(X=5) = C(10,5) * 0.5^10 = 252/1024 ≈ 0.2461
        assert!((b.pdf(5.0) - 0.246_093_75).abs() < 1e-6);
    }

    #[test]
    fn test_binomial_mean_var() {
        let b = Binomial::<f64>::new(20, 0.3).unwrap();
        assert!((b.mean() - 6.0).abs() < 1e-10);
        assert!((b.variance() - 4.2).abs() < 1e-10);
    }

    #[test]
    fn test_binomial_cdf_boundary() {
        let b = Binomial::<f64>::new(5, 0.5).unwrap();
        assert!((b.cdf(5.0) - 1.0).abs() < 1e-6);
        assert!(b.cdf(-1.0) < 1e-10);
    }
}
