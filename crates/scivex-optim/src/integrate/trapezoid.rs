//! Composite trapezoidal rule.

use scivex_core::Float;

use super::QuadResult;

/// Approximate `∫_a^b f(x) dx` using the composite trapezoidal rule with
/// `n` subintervals.
///
/// Error is `O(h^2)` where `h = (b - a) / n`.
///
/// # Examples
///
/// ```
/// # use scivex_optim::integrate::trapezoid;
/// // ∫₀¹ x² dx = 1/3
/// let result = trapezoid(|x: f64| x * x, 0.0, 1.0, 1000);
/// assert!((result.value - 1.0 / 3.0).abs() < 1e-5);
/// ```
pub fn trapezoid<T: Float, F: Fn(T) -> T>(f: F, a: T, b: T, n: usize) -> QuadResult<T> {
    assert!(n > 0, "trapezoid: n must be positive");

    let n_t = T::from_usize(n);
    let h = (b - a) / n_t;
    let two = T::from_f64(2.0);

    let mut sum = f(a) + f(b);
    for i in 1..n {
        let x = a + T::from_usize(i) * h;
        sum += f(x) * two;
    }

    let value = sum * h / two;

    QuadResult {
        value,
        error_estimate: T::zero(), // no built-in error estimate for fixed-n
        n_evals: n + 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trapezoid_sin() {
        // ∫_0^π sin(x) dx = 2
        let result = trapezoid(|x: f64| x.sin(), 0.0, std::f64::consts::PI, 1000);
        assert!((result.value - 2.0).abs() < 1e-5);
        assert_eq!(result.n_evals, 1001);
    }

    #[test]
    fn test_trapezoid_quadratic() {
        // ∫_0^1 x^2 dx = 1/3
        let result = trapezoid(|x: f64| x * x, 0.0, 1.0, 1000);
        assert!((result.value - 1.0 / 3.0).abs() < 1e-5);
    }
}
