use scivex_core::Float;
use scivex_core::random::Rng;

use crate::error::{Result, StatsError};
use crate::special::{ln_gamma, regularized_beta};

use super::{Distribution, ppf_bisection};

/// Student's t-distribution with `df` degrees of freedom.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone, Copy)]
pub struct StudentT<T: Float> {
    df: T,
}

impl<T: Float> StudentT<T> {
    /// Create a new Student-t distribution with `df > 0` degrees of freedom.
    pub fn new(df: T) -> Result<Self> {
        if df <= T::from_f64(0.0) {
            return Err(StatsError::InvalidParameter {
                name: "df",
                reason: "must be positive",
            });
        }
        Ok(Self { df })
    }
}

impl<T: Float> Distribution<T> for StudentT<T> {
    fn pdf(&self, x: T) -> T {
        let one = T::from_f64(1.0);
        let half = T::from_f64(0.5);
        let v = self.df;

        let log_pdf = ln_gamma((v + one) * half)
            - ln_gamma(v * half)
            - half * (v * T::pi()).ln()
            - ((v + one) * half) * (one + x * x / v).ln();
        log_pdf.exp()
    }

    fn cdf(&self, x: T) -> T {
        let half = T::from_f64(0.5);
        let one = T::from_f64(1.0);
        let v = self.df;

        // Use the regularized incomplete beta function
        let t = v / (v + x * x);
        let ib = regularized_beta(t, v * half, half).unwrap_or(half);

        if x >= T::from_f64(0.0) {
            one - half * ib
        } else {
            half * ib
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
        // Bisection over a wide range
        let bound = T::from_f64(100.0);
        Ok(ppf_bisection(|x| self.cdf(x), p, -bound, bound))
    }

    fn sample(&self, rng: &mut Rng) -> T {
        // t = Z / sqrt(V/df) where Z ~ N(0,1), V ~ Chi2(df)
        let z = T::from_f64(rng.next_normal_f64());
        let chi2 = super::gamma::Gamma::new(self.df / T::from_f64(2.0), T::from_f64(0.5))
            .expect("df validated at construction");
        let v = chi2.sample(rng);
        z / (v / self.df).sqrt().max(T::from_f64(1e-30))
    }

    fn mean(&self) -> T {
        T::from_f64(0.0) // defined for df > 1, 0 otherwise
    }

    fn variance(&self) -> T {
        let two = T::from_f64(2.0);
        if self.df > two {
            self.df / (self.df - two)
        } else {
            T::infinity()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_student_t_invalid() {
        assert!(StudentT::<f64>::new(0.0).is_err());
    }

    #[test]
    fn test_student_t_cdf_symmetry() {
        let t = StudentT::<f64>::new(10.0).unwrap();
        // CDF(0) = 0.5
        assert!((t.cdf(0.0) - 0.5).abs() < 1e-6);
        // CDF(-x) = 1 - CDF(x)
        let x = 1.5;
        assert!((t.cdf(-x) + t.cdf(x) - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_student_t_mean_var() {
        let t = StudentT::<f64>::new(5.0).unwrap();
        assert!((t.mean()).abs() < 1e-10);
        assert!((t.variance() - 5.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_student_t_ppf_roundtrip() {
        let t = StudentT::<f64>::new(10.0).unwrap();
        let x = 2.0;
        let p = t.cdf(x);
        let x2 = t.ppf(p).unwrap();
        assert!((x - x2).abs() < 1e-3);
    }
}
