use scivex_core::Float;
use scivex_core::random::Rng;

use crate::error::{Result, StatsError};
use crate::special::ln_gamma;

use super::Distribution;

/// Negative binomial distribution: number of failures before `r` successes,
/// with success probability `p`.
///
/// PMF: `C(k+r-1, k) * p^r * (1-p)^k` for `k = 0, 1, 2, ...`
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone, Copy)]
pub struct NegativeBinomial<T: Float> {
    r: T,
    p: T,
}

impl<T: Float> NegativeBinomial<T> {
    /// Create a new negative binomial distribution with `r > 0` successes
    /// and success probability `p` in `(0, 1]`.
    pub fn new(r: T, p: T) -> Result<Self> {
        if r <= T::zero() {
            return Err(StatsError::InvalidParameter {
                name: "r",
                reason: "number of successes must be positive",
            });
        }
        if p <= T::zero() || p > T::one() {
            return Err(StatsError::InvalidParameter {
                name: "p",
                reason: "probability must be in (0, 1]",
            });
        }
        Ok(Self { r, p })
    }
}

impl<T: Float> Distribution<T> for NegativeBinomial<T> {
    fn pdf(&self, x: T) -> T {
        let k = x;
        if k < T::zero() || k != k.floor() {
            return T::zero();
        }
        // PMF = Gamma(k+r) / (Gamma(k+1) * Gamma(r)) * p^r * (1-p)^k
        let log_coeff = ln_gamma(k + self.r) - ln_gamma(k + T::one()) - ln_gamma(self.r);
        let log_pmf = log_coeff + self.r * self.p.ln() + k * (T::one() - self.p).ln();
        log_pmf.exp()
    }

    fn cdf(&self, x: T) -> T {
        if x < T::zero() {
            return T::zero();
        }
        let n = x.floor();
        let mut sum = T::zero();
        let mut k = T::zero();
        while k <= n {
            sum += self.pdf(k);
            k += T::one();
        }
        sum.min(T::one())
    }

    fn ppf(&self, p_val: T) -> Result<T> {
        if p_val < T::zero() || p_val > T::one() {
            return Err(StatsError::InvalidParameter {
                name: "p",
                reason: "must be in [0, 1]",
            });
        }
        let mut k = T::zero();
        let mut cum = T::zero();
        loop {
            cum += self.pdf(k);
            if cum >= p_val {
                return Ok(k);
            }
            k += T::one();
            // Safety limit
            if k > T::from_f64(10_000.0) {
                return Ok(k);
            }
        }
    }

    fn sample(&self, rng: &mut Rng) -> T {
        // Sample via Gamma-Poisson mixture:
        // If Y ~ Gamma(r, p/(1-p)), then X|Y ~ Poisson(Y), X ~ NegBin(r, p)
        let q = T::one() - self.p;
        let rate = self.p / q;

        // Sample from Gamma(r, rate) using Marsaglia & Tsang
        let gamma_sample = sample_gamma(self.r, rate, rng);

        // Sample from Poisson(gamma_sample)
        sample_poisson(gamma_sample, rng)
    }

    fn mean(&self) -> T {
        self.r * (T::one() - self.p) / self.p
    }

    fn variance(&self) -> T {
        self.r * (T::one() - self.p) / (self.p * self.p)
    }
}

/// Simple Gamma sampling via Marsaglia & Tsang for shape >= 1.
fn sample_gamma<T: Float>(shape: T, rate: T, rng: &mut Rng) -> T {
    let one = T::one();
    if shape < one {
        // Boost: Gamma(a) = Gamma(a+1) * U^(1/a)
        let u = T::from_f64(rng.next_f64());
        return sample_gamma(shape + one, rate, rng) * u.powf(one / shape);
    }
    let d = shape - T::from_f64(1.0 / 3.0);
    let c = T::one() / (T::from_f64(9.0) * d).sqrt();
    loop {
        let x = loop {
            let z = T::from_f64(rng.next_f64() * 2.0 - 1.0);
            let v = one + c * z;
            if v > T::zero() {
                break z;
            }
        };
        let v = (one + c * x) * (one + c * x) * (one + c * x);
        let u = T::from_f64(rng.next_f64());
        if u < one - T::from_f64(0.0331) * x * x * x * x {
            return d * v / rate;
        }
        if u.ln() < T::from_f64(0.5) * x * x + d * (one - v + v.ln()) {
            return d * v / rate;
        }
    }
}

/// Simple Poisson sampling via Knuth's algorithm.
fn sample_poisson<T: Float>(lambda: T, rng: &mut Rng) -> T {
    let l = (-lambda).exp();
    let mut k = T::zero();
    let mut p = T::one();
    loop {
        k += T::one();
        p *= T::from_f64(rng.next_f64());
        if p <= l {
            return k - T::one();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_negbin_invalid() {
        assert!(NegativeBinomial::<f64>::new(0.0, 0.5).is_err());
        assert!(NegativeBinomial::<f64>::new(1.0, 0.0).is_err());
        assert!(NegativeBinomial::<f64>::new(1.0, 1.5).is_err());
    }

    #[test]
    fn test_negbin_pmf() {
        // r=1, p=0.5 => Geometric. PMF(0) = 0.5, PMF(1) = 0.25
        let d = NegativeBinomial::<f64>::new(1.0, 0.5).unwrap();
        assert!((d.pdf(0.0) - 0.5).abs() < 1e-10);
        assert!((d.pdf(1.0) - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_negbin_cdf() {
        let d = NegativeBinomial::<f64>::new(1.0, 0.5).unwrap();
        // CDF(0) = 0.5, CDF(1) = 0.75
        assert!((d.cdf(0.0) - 0.5).abs() < 1e-10);
        assert!((d.cdf(1.0) - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_negbin_mean_var() {
        let d = NegativeBinomial::<f64>::new(5.0, 0.4).unwrap();
        // Mean = 5*0.6/0.4 = 7.5
        assert!((d.mean() - 7.5).abs() < 1e-10);
        // Var = 5*0.6/0.16 = 18.75
        assert!((d.variance() - 18.75).abs() < 1e-10);
    }

    #[test]
    fn test_negbin_ppf() {
        let d = NegativeBinomial::<f64>::new(1.0, 0.5).unwrap();
        assert!((d.ppf(0.4).unwrap()).abs() < 1e-10); // PPF(0.4) = 0
        assert!((d.ppf(0.6).unwrap() - 1.0).abs() < 1e-10); // PPF(0.6) = 1
    }
}
