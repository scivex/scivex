//! Probability distributions.
//!
//! Each distribution implements the [`Distribution`] trait providing PDF, CDF,
//! quantile function (PPF), sampling, and theoretical moments.

mod bernoulli;
mod beta;
mod binomial;
mod chi_squared;
mod exponential;
mod gamma;
mod normal;
mod poisson;
mod student_t;
mod uniform;

pub use bernoulli::Bernoulli;
pub use beta::Beta;
pub use binomial::Binomial;
pub use chi_squared::ChiSquared;
pub use exponential::Exponential;
pub use gamma::Gamma;
pub use normal::Normal;
pub use poisson::Poisson;
pub use student_t::StudentT;
pub use uniform::Uniform;

use scivex_core::Float;
use scivex_core::random::Rng;

use crate::error::Result;

/// Common interface for probability distributions.
pub trait Distribution<T: Float> {
    /// Probability density (or mass) function evaluated at `x`.
    fn pdf(&self, x: T) -> T;

    /// Cumulative distribution function evaluated at `x`.
    fn cdf(&self, x: T) -> T;

    /// Percent point function (inverse CDF). Returns `x` such that `cdf(x) = p`.
    fn ppf(&self, p: T) -> Result<T>;

    /// Draw a single random sample.
    fn sample(&self, rng: &mut Rng) -> T;

    /// Draw `n` random samples.
    fn sample_n(&self, rng: &mut Rng, n: usize) -> Vec<T> {
        (0..n).map(|_| self.sample(rng)).collect()
    }

    /// Theoretical mean of the distribution.
    fn mean(&self) -> T;

    /// Theoretical variance of the distribution.
    fn variance(&self) -> T;
}

// ---------------------------------------------------------------------------
// PPF helpers
// ---------------------------------------------------------------------------

/// Newton–Raphson PPF solver given cdf and pdf closures.
#[allow(dead_code)]
pub(crate) fn ppf_newton<T: Float>(
    cdf_fn: impl Fn(T) -> T,
    pdf_fn: impl Fn(T) -> T,
    p: T,
    mut x: T,
    max_iter: usize,
) -> T {
    let eps = T::from_f64(1e-12);
    for _ in 0..max_iter {
        let fx = cdf_fn(x) - p;
        let dfx = pdf_fn(x);
        if dfx.abs() < T::from_f64(1e-30) {
            break;
        }
        let step = fx / dfx;
        x -= step;
        if step.abs() < eps {
            return x;
        }
    }
    x
}

/// Bisection PPF solver given a cdf closure.
pub(crate) fn ppf_bisection<T: Float>(cdf_fn: impl Fn(T) -> T, p: T, mut lo: T, mut hi: T) -> T {
    let eps = T::from_f64(1e-12);
    let two = T::from_f64(2.0);

    for _ in 0..200 {
        let mid = (lo + hi) / two;
        if (hi - lo).abs() < eps {
            return mid;
        }
        if cdf_fn(mid) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (lo + hi) / two
}
