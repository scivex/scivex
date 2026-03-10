//! Brent's method for 1-D minimization.
//!
//! Combines golden-section search with parabolic interpolation for
//! superlinear convergence on smooth functions.

use scivex_core::Float;

use crate::error::Result;

use super::Minimize1dResult;

/// Find the minimum of `f` on `[a, b]` using Brent's method.
///
/// Combines golden-section search with parabolic interpolation.
/// Superlinear convergence on smooth unimodal functions.
pub fn brent_min<T: Float, F: Fn(T) -> T>(
    f: F,
    a: T,
    b: T,
    xtol: T,
    max_iter: usize,
) -> Result<Minimize1dResult<T>> {
    let golden = T::from_f64(0.381_966_011_250_105); // 1 - (sqrt(5)-1)/2
    let two = T::from_f64(2.0);
    let half = T::from_f64(0.5);

    let mut a = a;
    let mut b = b;
    if a > b {
        core::mem::swap(&mut a, &mut b);
    }

    // x is the point with the least function value found so far.
    // w is the point with the second least value.
    // v is the previous value of w.
    let mut x = a + golden * (b - a);
    let mut fx = f(x);
    let mut w = x;
    let mut fw = fx;
    let mut v = x;
    let mut fv = fx;

    let mut d = T::zero(); // step size
    let mut e = T::zero(); // previous step size

    for i in 0..max_iter {
        let midpoint = half * (a + b);
        let tol1 = xtol * x.abs() + T::from_f64(1e-20);
        let tol2 = two * tol1;

        // Convergence check
        if (x - midpoint).abs() <= tol2 - half * (b - a) {
            return Ok(Minimize1dResult {
                x_min: x,
                f_min: fx,
                iterations: i + 1,
                converged: true,
            });
        }

        let mut use_golden = true;

        // Try parabolic interpolation
        if e.abs() > tol1 {
            // Fit a parabola through x, w, v
            let r = (x - w) * (fx - fv);
            let q = (x - v) * (fx - fw);
            let mut p = (x - v) * q - (x - w) * r;
            let mut q = two * (q - r);

            if q > T::zero() {
                p = -p;
            } else {
                q = -q;
            }

            // Is the parabolic step acceptable?
            if p.abs() < (half * q * e).abs() && p > q * (a - x) && p < q * (b - x) {
                let step = p / q;
                let u = x + step;

                // Don't evaluate too close to endpoints
                if (u - a) < tol2 || (b - u) < tol2 {
                    d = if x < midpoint { tol1 } else { -tol1 };
                } else {
                    e = d;
                    d = step;
                }
                use_golden = false;
            }
        }

        if use_golden {
            // Golden-section step
            e = if x < midpoint { b - x } else { a - x };
            d = golden * e;
        }

        // Evaluate function at the new point
        let step = if d.abs() >= tol1 {
            d
        } else if d > T::zero() {
            tol1
        } else {
            -tol1
        };
        let u = x + step;
        let fu = f(u);

        // Update the bracket and best points
        if fu <= fx {
            if u < x {
                b = x;
            } else {
                a = x;
            }
            v = w;
            fv = fw;
            w = x;
            fw = fx;
            x = u;
            fx = fu;
        } else {
            if u < x {
                a = u;
            } else {
                b = u;
            }
            if fu <= fw || w == x {
                v = w;
                fv = fw;
                w = u;
                fw = fu;
            } else if fu <= fv || v == x || v == w {
                v = u;
                fv = fu;
            }
        }
    }

    Ok(Minimize1dResult {
        x_min: x,
        f_min: fx,
        iterations: max_iter,
        converged: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_brent_min_quadratic() {
        // (x - 3)^2 on [0, 5]
        let result = brent_min(|x: f64| (x - 3.0) * (x - 3.0), 0.0, 5.0, 1e-12, 200).unwrap();
        assert!(result.converged);
        assert!((result.x_min - 3.0).abs() < 1e-8);
    }

    #[test]
    fn test_brent_min_quartic() {
        // x^4 - 2x^2 + 1 = (x^2 - 1)^2 => minima at x = ±1
        // On [0, 3] the minimum is at x = 1
        let result =
            brent_min(|x: f64| x.powi(4) - 2.0 * x * x + 1.0, 0.0, 3.0, 1e-12, 200).unwrap();
        assert!(result.converged);
        assert!((result.x_min - 1.0).abs() < 1e-8);
        assert!(result.f_min.abs() < 1e-12);
    }
}
