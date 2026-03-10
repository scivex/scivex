//! Brent's method for root finding.
//!
//! Combines bisection, secant, and inverse quadratic interpolation for
//! guaranteed convergence with superlinear speed.

use scivex_core::Float;

use crate::error::{OptimError, Result};

use super::{RootOptions, RootResult};

/// Find a root of `f` on `[a, b]` using Brent's method.
///
/// Requires `f(a)` and `f(b)` to have opposite signs. Combines bisection,
/// secant, and inverse quadratic interpolation for robust superlinear
/// convergence.
pub fn brent_root<T: Float, F: Fn(T) -> T>(
    f: F,
    a: T,
    b: T,
    options: &RootOptions<T>,
) -> Result<RootResult<T>> {
    let mut a = a;
    let mut b = b;
    let mut fa = f(a);
    let mut fb = f(b);

    if (fa > T::zero()) == (fb > T::zero()) {
        return Err(OptimError::BracketError);
    }

    // Ensure |f(a)| >= |f(b)| so b is the best approximation
    if fa.abs() < fb.abs() {
        core::mem::swap(&mut a, &mut b);
        core::mem::swap(&mut fa, &mut fb);
    }

    let mut c = a;
    let mut fc = fa;
    let mut mflag = true;
    let mut d = b - a; // previous step

    let two = T::from_f64(2.0);
    let three = T::from_f64(3.0);

    for i in 0..options.max_iter {
        if fb.abs() < options.ftol || (b - a).abs() < options.xtol {
            return Ok(RootResult {
                root: b,
                f_root: fb,
                iterations: i + 1,
                converged: true,
            });
        }

        let mut s;
        if (fa - fc).abs() > T::epsilon() && (fb - fc).abs() > T::epsilon() {
            // Inverse quadratic interpolation
            s = a * fb * fc / ((fa - fb) * (fa - fc))
                + b * fa * fc / ((fb - fa) * (fb - fc))
                + c * fa * fb / ((fc - fa) * (fc - fb));
        } else {
            // Secant method
            s = b - fb * (b - a) / (fb - fa);
        }

        // Conditions for rejecting the interpolation and using bisection instead
        let mid = (a + b) / two;
        let cond1 = if a < b {
            s < (three * a + b) / T::from_f64(4.0) || s > b
        } else {
            s > (three * a + b) / T::from_f64(4.0) || s < b
        };
        let cond2 = mflag && (s - b).abs() >= (b - c).abs() / two;
        let cond3 = !mflag && (s - b).abs() >= (c - d).abs() / two;
        let cond4 = mflag && (b - c).abs() < options.xtol;
        let cond5 = !mflag && (c - d).abs() < options.xtol;

        if cond1 || cond2 || cond3 || cond4 || cond5 {
            s = mid;
            mflag = true;
        } else {
            mflag = false;
        }

        let fs = f(s);
        d = c;
        c = b;
        fc = fb;

        if (fa > T::zero()) == (fs > T::zero()) {
            a = s;
            fa = fs;
        } else {
            b = s;
            fb = fs;
        }

        // Keep |f(a)| >= |f(b)|
        if fa.abs() < fb.abs() {
            core::mem::swap(&mut a, &mut b);
            core::mem::swap(&mut fa, &mut fb);
        }
    }

    Ok(RootResult {
        root: b,
        f_root: fb,
        iterations: options.max_iter,
        converged: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_brent_cubic() {
        // x^3 - x - 2 = 0 on [1, 2]
        let result = brent_root(
            |x: f64| x * x * x - x - 2.0,
            1.0,
            2.0,
            &RootOptions::default(),
        )
        .unwrap();
        assert!(result.converged);
        assert!(result.f_root.abs() < 1e-10);
        // Known root is approximately 1.5214
        assert!((result.root - 1.521_379_706_804_567_6).abs() < 1e-10);
    }

    #[test]
    fn test_brent_sqrt2() {
        let result = brent_root(|x: f64| x * x - 2.0, 1.0, 2.0, &RootOptions::default()).unwrap();
        assert!(result.converged);
        assert!((result.root - std::f64::consts::SQRT_2).abs() < 1e-12);
    }

    #[test]
    fn test_brent_bracket_error() {
        let result = brent_root(|x: f64| x * x + 1.0, 0.0, 1.0, &RootOptions::default());
        assert!(result.is_err());
    }
}
