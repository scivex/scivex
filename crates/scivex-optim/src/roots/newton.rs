//! Newton-Raphson method for root finding.

use scivex_core::Float;

use crate::error::{OptimError, Result};

use super::{RootOptions, RootResult};

/// Find a root of `f` using the Newton-Raphson method.
///
/// Requires both the function `f` and its derivative `f_prime`.
/// Starting from `x0`, iterates `x_{n+1} = x_n - f(x_n) / f'(x_n)`.
/// Converges quadratically near simple roots.
pub fn newton<T: Float, F: Fn(T) -> T, G: Fn(T) -> T>(
    f: F,
    f_prime: G,
    x0: T,
    options: &RootOptions<T>,
) -> Result<RootResult<T>> {
    let mut x = x0;

    for i in 0..options.max_iter {
        let fx = f(x);

        if fx.abs() < options.ftol {
            return Ok(RootResult {
                root: x,
                f_root: fx,
                iterations: i + 1,
                converged: true,
            });
        }

        let fpx = f_prime(x);
        if fpx.abs() < T::epsilon() {
            return Err(OptimError::NonFiniteValue {
                context: "newton: derivative is near zero",
            });
        }

        let x_new = x - fx / fpx;

        if !x_new.is_finite() {
            return Err(OptimError::NonFiniteValue {
                context: "newton: iterate became non-finite",
            });
        }

        if (x_new - x).abs() < options.xtol {
            let fx_new = f(x_new);
            return Ok(RootResult {
                root: x_new,
                f_root: fx_new,
                iterations: i + 1,
                converged: true,
            });
        }

        x = x_new;
    }

    let fx = f(x);
    Ok(RootResult {
        root: x,
        f_root: fx,
        iterations: options.max_iter,
        converged: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_newton_sqrt2() {
        // x^2 - 2 = 0, derivative = 2x
        let result = newton(
            |x: f64| x * x - 2.0,
            |x: f64| 2.0 * x,
            1.5,
            &RootOptions::default(),
        )
        .unwrap();
        assert!(result.converged);
        assert!((result.root - std::f64::consts::SQRT_2).abs() < 1e-12);
        assert!(result.iterations <= 10);
    }

    #[test]
    fn test_newton_cubic() {
        // x^3 - x - 2 = 0, derivative = 3x^2 - 1
        let result = newton(
            |x: f64| x * x * x - x - 2.0,
            |x: f64| 3.0 * x * x - 1.0,
            1.5,
            &RootOptions::default(),
        )
        .unwrap();
        assert!(result.converged);
        assert!(result.f_root.abs() < 1e-10);
    }
}
