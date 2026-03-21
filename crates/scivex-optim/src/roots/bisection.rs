//! Bisection method for root finding.

use scivex_core::Float;

use crate::error::{OptimError, Result};

use super::{RootOptions, RootResult};

/// Find a root of `f` on the interval `[a, b]` using the bisection method.
///
/// Requires `f(a)` and `f(b)` to have opposite signs. Convergence is
/// guaranteed and takes `O(log2((b-a)/xtol))` iterations.
///
/// # Examples
///
/// ```
/// # use scivex_optim::roots::{bisection, RootOptions};
/// // Root of x^2 - 2 on [0, 2] → √2
/// let result = bisection(|x: f64| x * x - 2.0, 0.0, 2.0, &RootOptions::default()).unwrap();
/// assert!((result.root - std::f64::consts::SQRT_2).abs() < 1e-8);
/// ```
pub fn bisection<T: Float, F: Fn(T) -> T>(
    f: F,
    a: T,
    b: T,
    options: &RootOptions<T>,
) -> Result<RootResult<T>> {
    let mut lo = a;
    let mut hi = b;
    let mut f_lo = f(lo);
    let f_hi = f(hi);

    // Check bracket validity
    if (f_lo > T::zero()) == (f_hi > T::zero()) {
        return Err(OptimError::BracketError);
    }

    let two = T::from_f64(2.0);
    let mut mid = lo;
    let mut f_mid = f_lo;

    for i in 0..options.max_iter {
        mid = (lo + hi) / two;
        f_mid = f(mid);

        if f_mid.abs() < options.ftol || (hi - lo) / two < options.xtol {
            return Ok(RootResult {
                root: mid,
                f_root: f_mid,
                iterations: i + 1,
                converged: true,
            });
        }

        if (f_mid > T::zero()) == (f_lo > T::zero()) {
            lo = mid;
            f_lo = f_mid;
        } else {
            hi = mid;
        }
    }

    Ok(RootResult {
        root: mid,
        f_root: f_mid,
        iterations: options.max_iter,
        converged: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bisection_sqrt2() {
        // x^2 - 2 = 0 on [1, 2] => root = sqrt(2)
        let result = bisection(|x: f64| x * x - 2.0, 1.0, 2.0, &RootOptions::default()).unwrap();
        assert!(result.converged);
        assert!((result.root - std::f64::consts::SQRT_2).abs() < 1e-10);
        assert!(result.f_root.abs() < 1e-10);
    }

    #[test]
    fn test_bisection_bracket_error() {
        // f(1) = -1, f(2) = 2, both positive side? No, f(1) = -1 < 0.
        // Let's try where signs match: f(x) = x^2 + 1 on [0, 1] => always positive.
        let result = bisection(|x: f64| x * x + 1.0, 0.0, 1.0, &RootOptions::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_bisection_f32() {
        let opts = RootOptions {
            xtol: 1e-6_f32,
            ftol: 1e-6_f32,
            max_iter: 100,
        };
        let result = bisection(|x: f32| x * x - 2.0, 1.0_f32, 2.0_f32, &opts).unwrap();
        assert!(result.converged);
        assert!((result.root - std::f32::consts::SQRT_2).abs() < 1e-5);
    }
}
