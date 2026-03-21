use scivex_core::Float;
use scivex_core::random::Rng;

use crate::error::{Result, StatsError};
use crate::special::ln_gamma;

use super::Distribution;

/// Hypergeometric distribution: number of successes in `n` draws from a
/// population of size `big_n` containing `big_k` successes, without replacement.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone, Copy)]
pub struct Hypergeometric<T: Float> {
    big_n: usize,
    big_k: usize,
    n: usize,
    _marker: core::marker::PhantomData<T>,
}

impl<T: Float> Hypergeometric<T> {
    /// Create a new hypergeometric distribution.
    ///
    /// - `big_n` — population size
    /// - `big_k` — number of success states in population
    /// - `n` — number of draws
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_stats::distributions::{Hypergeometric, Distribution};
    /// let h = Hypergeometric::<f64>::new(20, 7, 5).unwrap();
    /// assert!((h.mean() - 1.75).abs() < 1e-10); // n*K/N
    /// ```
    pub fn new(big_n: usize, big_k: usize, n: usize) -> Result<Self> {
        if big_k > big_n {
            return Err(StatsError::InvalidParameter {
                name: "big_k",
                reason: "success states cannot exceed population size",
            });
        }
        if n > big_n {
            return Err(StatsError::InvalidParameter {
                name: "n",
                reason: "number of draws cannot exceed population size",
            });
        }
        Ok(Self {
            big_n,
            big_k,
            n,
            _marker: core::marker::PhantomData,
        })
    }
}

/// Log of binomial coefficient C(n, k) using ln_gamma.
fn ln_choose<T: Float>(n: usize, k: usize) -> T {
    if k > n {
        return T::from_f64(f64::NEG_INFINITY);
    }
    let nf = T::from_f64(n as f64 + 1.0);
    let kf = T::from_f64(k as f64 + 1.0);
    let nkf = T::from_f64((n - k) as f64 + 1.0);
    ln_gamma(nf) - ln_gamma(kf) - ln_gamma(nkf)
}

impl<T: Float> Distribution<T> for Hypergeometric<T> {
    fn pdf(&self, x: T) -> T {
        let k = x.floor();
        if k < T::zero() || (k - x).abs() > T::from_f64(1e-12) {
            return T::zero();
        }
        let ki = {
            let s = format!("{k:?}");
            s.parse::<f64>().unwrap_or(0.0) as usize
        };
        let max_k = self.big_k.min(self.n);
        let min_k = (self.n + self.big_k).saturating_sub(self.big_n);
        if ki < min_k || ki > max_k {
            return T::zero();
        }
        let log_pmf: T = ln_choose::<T>(self.big_k, ki)
            + ln_choose::<T>(self.big_n - self.big_k, self.n - ki)
            - ln_choose::<T>(self.big_n, self.n);
        log_pmf.exp()
    }

    fn cdf(&self, x: T) -> T {
        if x < T::zero() {
            return T::zero();
        }
        let n_max = x.floor().min(T::from_f64(self.n.min(self.big_k) as f64));
        let mut sum = T::zero();
        let mut k = T::zero();
        while k <= n_max {
            sum += self.pdf(k);
            k += T::one();
        }
        sum.min(T::one())
    }

    fn ppf(&self, p: T) -> Result<T> {
        if p < T::zero() || p > T::one() {
            return Err(StatsError::InvalidParameter {
                name: "p",
                reason: "must be in [0, 1]",
            });
        }
        let max_k = self.big_k.min(self.n);
        let mut cum = T::zero();
        for ki in 0..=max_k {
            cum += self.pdf(T::from_f64(ki as f64));
            if cum >= p {
                return Ok(T::from_f64(ki as f64));
            }
        }
        Ok(T::from_f64(max_k as f64))
    }

    fn sample(&self, rng: &mut Rng) -> T {
        // Direct simulation: draw n items without replacement
        let mut successes_remaining = self.big_k;
        let mut pop_remaining = self.big_n;
        let mut drawn_successes = 0usize;
        for _ in 0..self.n {
            let u = rng.next_f64();
            if u < successes_remaining as f64 / pop_remaining as f64 {
                drawn_successes += 1;
                successes_remaining -= 1;
            }
            pop_remaining -= 1;
        }
        T::from_f64(drawn_successes as f64)
    }

    fn mean(&self) -> T {
        T::from_f64(self.n as f64 * self.big_k as f64 / self.big_n as f64)
    }

    fn variance(&self) -> T {
        let n = self.n as f64;
        let bk = self.big_k as f64;
        let bn = self.big_n as f64;
        T::from_f64(n * bk / bn * (1.0 - bk / bn) * (bn - n) / (bn - 1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hypergeometric_invalid() {
        assert!(Hypergeometric::<f64>::new(10, 15, 5).is_err()); // K > N
        assert!(Hypergeometric::<f64>::new(10, 5, 15).is_err()); // n > N
    }

    #[test]
    fn test_hypergeometric_pmf() {
        // Classic: N=20, K=7, n=12. P(X=4) from scipy
        let d = Hypergeometric::<f64>::new(20, 7, 12).unwrap();
        let p4 = d.pdf(4.0);
        // P(X=4) = C(7,4)*C(13,8)/C(20,12)
        // C(7,4) = 35, C(13,8) = 1287, C(20,12) = 125970
        // 35*1287/125970 ≈ 0.35758
        assert!((p4 - 0.357_585_139_318_887).abs() < 1e-4, "p4={p4}");
    }

    #[test]
    fn test_hypergeometric_cdf_sums_to_one() {
        let d = Hypergeometric::<f64>::new(10, 4, 5).unwrap();
        let total = d.cdf(5.0);
        assert!((total - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_hypergeometric_mean_var() {
        let d = Hypergeometric::<f64>::new(20, 7, 12).unwrap();
        // Mean = 12*7/20 = 4.2
        assert!((d.mean() - 4.2).abs() < 1e-10);
    }

    #[test]
    fn test_hypergeometric_ppf() {
        let d = Hypergeometric::<f64>::new(20, 7, 12).unwrap();
        let x = d.ppf(0.5).unwrap();
        assert!((3.0..=5.0).contains(&x), "ppf(0.5)={x}");
    }
}
