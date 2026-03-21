use scivex_core::Float;
use scivex_core::random::Rng;

use crate::error::{Result, StatsError};

use super::Distribution;

/// Bernoulli distribution with success probability p.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone, Copy)]
pub struct Bernoulli<T: Float> {
    p: T,
}

impl<T: Float> Bernoulli<T> {
    /// Create a new Bernoulli distribution with `p` in [0, 1].
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_stats::distributions::{Bernoulli, Distribution};
    /// let b = Bernoulli::new(0.3_f64).unwrap();
    /// assert!((b.mean() - 0.3).abs() < 1e-10);
    /// ```
    pub fn new(p: T) -> Result<Self> {
        let zero = T::from_f64(0.0);
        let one = T::from_f64(1.0);
        if p < zero || p > one {
            return Err(StatsError::InvalidParameter {
                name: "p",
                reason: "must be in [0, 1]",
            });
        }
        Ok(Self { p })
    }
}

impl<T: Float> Distribution<T> for Bernoulli<T> {
    fn pdf(&self, x: T) -> T {
        let zero = T::from_f64(0.0);
        let one = T::from_f64(1.0);
        if (x - zero).abs() < T::epsilon() {
            one - self.p
        } else if (x - one).abs() < T::epsilon() {
            self.p
        } else {
            zero
        }
    }

    fn cdf(&self, x: T) -> T {
        let zero = T::from_f64(0.0);
        let one = T::from_f64(1.0);
        if x < zero {
            zero
        } else if x < one {
            one - self.p
        } else {
            one
        }
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
        if p <= one - self.p { Ok(zero) } else { Ok(one) }
    }

    fn sample(&self, rng: &mut Rng) -> T {
        if T::from_f64(rng.next_f64()) < self.p {
            T::from_f64(1.0)
        } else {
            T::from_f64(0.0)
        }
    }

    fn mean(&self) -> T {
        self.p
    }

    fn variance(&self) -> T {
        self.p * (T::from_f64(1.0) - self.p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bernoulli_invalid() {
        assert!(Bernoulli::<f64>::new(-0.1).is_err());
        assert!(Bernoulli::<f64>::new(1.1).is_err());
    }

    #[test]
    fn test_bernoulli_pmf() {
        let b = Bernoulli::<f64>::new(0.3).unwrap();
        assert!((b.pdf(0.0) - 0.7).abs() < 1e-10);
        assert!((b.pdf(1.0) - 0.3).abs() < 1e-10);
        assert!((b.pdf(0.5)).abs() < 1e-10);
    }

    #[test]
    fn test_bernoulli_mean_var() {
        let b = Bernoulli::<f64>::new(0.4).unwrap();
        assert!((b.mean() - 0.4).abs() < 1e-10);
        assert!((b.variance() - 0.24).abs() < 1e-10);
    }
}
