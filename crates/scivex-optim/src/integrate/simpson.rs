//! Composite Simpson's 1/3 rule.

use scivex_core::Float;

use crate::error::{OptimError, Result};

use super::QuadResult;

/// Approximate `∫_a^b f(x) dx` using the composite Simpson's 1/3 rule
/// with `n` subintervals.
///
/// Requires `n` to be even. Error is `O(h^4)` where `h = (b - a) / n`.
///
/// # Examples
///
/// ```
/// # use scivex_optim::integrate::simpson;
/// // ∫₀^π sin(x) dx = 2
/// let result = simpson(|x: f64| x.sin(), 0.0, std::f64::consts::PI, 100).unwrap();
/// assert!((result.value - 2.0).abs() < 1e-6);
/// ```
pub fn simpson<T: Float, F: Fn(T) -> T>(f: F, a: T, b: T, n: usize) -> Result<QuadResult<T>> {
    #[allow(clippy::manual_is_multiple_of)]
    if n == 0 || n % 2 != 0 {
        return Err(OptimError::InvalidParameter {
            name: "n",
            reason: "must be a positive even number",
        });
    }

    let n_t = T::from_usize(n);
    let h = (b - a) / n_t;
    let two = T::from_f64(2.0);
    let three = T::from_f64(3.0);
    let four = T::from_f64(4.0);

    let mut sum = f(a) + f(b);

    for i in 1..n {
        let x = a + T::from_usize(i) * h;
        let coeff = if i % 2 == 0 { two } else { four };
        sum += f(x) * coeff;
    }

    let value = sum * h / three;

    Ok(QuadResult {
        value,
        error_estimate: T::zero(),
        n_evals: n + 1,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simpson_sin() {
        // ∫_0^π sin(x) dx = 2
        let result = simpson(|x: f64| x.sin(), 0.0, std::f64::consts::PI, 1000).unwrap();
        assert!((result.value - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_simpson_cubic() {
        // ∫_0^1 x^3 dx = 1/4 — Simpson's is exact for polynomials up to degree 3
        let result = simpson(|x: f64| x * x * x, 0.0, 1.0, 4).unwrap();
        assert!((result.value - 0.25).abs() < 1e-14);
    }

    #[test]
    fn test_simpson_odd_n() {
        let result = simpson(|x: f64| x, 0.0, 1.0, 3);
        assert!(result.is_err());
    }
}
