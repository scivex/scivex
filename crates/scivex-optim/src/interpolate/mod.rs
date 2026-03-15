//! Interpolation algorithms (1-D and 2-D).
//!
//! Provides piecewise linear, cubic spline, B-spline, bilinear, and bicubic
//! interpolation. All interpolators are constructed once (precomputing
//! coefficients) and then evaluated many times.
//!
//! # Quick start
//!
//! ```ignore
//! use scivex_optim::interpolate::{Linear1d, Extrapolate};
//!
//! let xs = [0.0, 1.0, 2.0, 3.0];
//! let ys = [0.0, 1.0, 4.0, 9.0];
//! let interp = Linear1d::new(&xs, &ys, Extrapolate::Error).unwrap();
//! let y = interp.eval(1.5).unwrap(); // 2.5
//! ```

mod bicubic;
mod bilinear;
mod bspline;
mod cubic_spline;
mod linear;
pub(super) mod thomas;

pub use bicubic::Bicubic2d;
pub use bilinear::Bilinear2d;
pub use bspline::BSpline;
pub use cubic_spline::CubicSpline;
pub use linear::Linear1d;

use scivex_core::Float;

use crate::error::{OptimError, Result};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Method selector for 1-D convenience function [`interp1d`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interp1dMethod {
    /// Piecewise linear interpolation.
    Linear,
    /// Natural cubic spline interpolation.
    CubicSpline,
    /// Uniform B-spline interpolation (degree 3).
    BSpline,
}

/// Method selector for 2-D convenience function [`interp2d`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interp2dMethod {
    /// Bilinear interpolation on a rectilinear grid.
    Bilinear,
    /// Bicubic interpolation on a rectilinear grid.
    Bicubic,
}

/// Boundary condition for cubic spline construction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SplineBoundary<T> {
    /// Natural boundary: second derivative is zero at both endpoints.
    Natural,
    /// Clamped boundary: first derivative is prescribed at both endpoints.
    Clamped { left: T, right: T },
}

/// Controls behaviour when an evaluation point lies outside the data range.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Extrapolate {
    /// Return an error (default).
    #[default]
    Error,
    /// Clamp the query to the nearest boundary value.
    Clamp,
    /// Extend the nearest segment/polynomial beyond the boundary.
    Extend,
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Binary search for the interval index `i` such that `xs[i] <= x < xs[i+1]`.
///
/// Returns `Ok(i)` on success.
/// If `x` is outside `[xs[0], xs[n-1]]`, behaviour depends on `extrap`:
///
/// - `Extrapolate::Error` — returns `Err(InvalidParameter)`.
/// - `Extrapolate::Clamp` — clamps `x` to the boundary and returns the
///   boundary interval.
/// - `Extrapolate::Extend` — returns the first or last interval index.
#[inline]
pub(crate) fn find_interval<T: Float>(xs: &[T], x: T, extrap: Extrapolate) -> Result<(usize, T)> {
    debug_assert!(xs.len() >= 2);
    let n = xs.len();

    if x < xs[0] {
        return match extrap {
            Extrapolate::Error => Err(OptimError::InvalidParameter {
                name: "x",
                reason: "query point is below data range",
            }),
            Extrapolate::Clamp => Ok((0, xs[0])),
            Extrapolate::Extend => Ok((0, x)),
        };
    }

    if x > xs[n - 1] {
        return match extrap {
            Extrapolate::Error => Err(OptimError::InvalidParameter {
                name: "x",
                reason: "query point is above data range",
            }),
            Extrapolate::Clamp => Ok((n - 2, xs[n - 1])),
            Extrapolate::Extend => Ok((n - 2, x)),
        };
    }

    // Exact match on the last knot belongs to the last interval.
    if x == xs[n - 1] {
        return Ok((n - 2, x));
    }

    // Standard binary search
    let mut lo: usize = 0;
    let mut hi: usize = n - 1;
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if xs[mid] <= x {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    Ok((lo, x))
}

/// Validate that `xs` is strictly increasing and has at least `min_len` points.
pub(crate) fn validate_sorted<T: Float>(xs: &[T], min_len: usize) -> Result<()> {
    if xs.len() < min_len {
        return Err(OptimError::InvalidParameter {
            name: "xs",
            reason: "not enough data points",
        });
    }
    for i in 1..xs.len() {
        if xs[i] <= xs[i - 1] {
            return Err(OptimError::InvalidParameter {
                name: "xs",
                reason: "knots must be strictly increasing",
            });
        }
    }
    Ok(())
}

/// Validate that no values are NaN or infinite.
pub(crate) fn validate_finite<T: Float>(vals: &[T], name: &'static str) -> Result<()> {
    for &v in vals {
        if !v.is_finite() {
            return Err(OptimError::NonFiniteValue { context: name });
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Convenience functions
// ---------------------------------------------------------------------------

/// One-shot 1-D interpolation: build an interpolator and evaluate at `query`.
///
/// For repeated evaluation, prefer constructing the interpolator directly.
pub fn interp1d<T: Float>(
    xs: &[T],
    ys: &[T],
    query: &[T],
    method: Interp1dMethod,
) -> Result<Vec<T>> {
    match method {
        Interp1dMethod::Linear => {
            let interp = Linear1d::new(xs, ys, Extrapolate::Error)?;
            interp.eval_many(query)
        }
        Interp1dMethod::CubicSpline => {
            let interp = CubicSpline::new(xs, ys, SplineBoundary::Natural, Extrapolate::Error)?;
            interp.eval_many(query)
        }
        Interp1dMethod::BSpline => {
            let interp = BSpline::fit(xs, ys, 3, Extrapolate::Error)?;
            interp.eval_many(query)
        }
    }
}

/// One-shot 2-D interpolation: build an interpolator and evaluate at `query`.
///
/// `query` is a slice of `(x, y)` pairs.
pub fn interp2d<T: Float>(
    xs: Vec<T>,
    ys: Vec<T>,
    zs: Vec<Vec<T>>,
    query: &[(T, T)],
    method: Interp2dMethod,
) -> Result<Vec<T>> {
    match method {
        Interp2dMethod::Bilinear => {
            let interp = Bilinear2d::new(xs, ys, zs, Extrapolate::Error)?;
            interp.eval_many(query)
        }
        Interp2dMethod::Bicubic => {
            let interp = Bicubic2d::new(xs, ys, &zs, Extrapolate::Error)?;
            interp.eval_many(query)
        }
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn test_find_interval_basic() {
        let xs = [0.0, 1.0, 2.0, 3.0];
        let (i, x) = find_interval(&xs, 1.5, Extrapolate::Error).unwrap();
        assert_eq!(i, 1);
        assert!((x - 1.5).abs() < 1e-15);
    }

    #[test]
    fn test_find_interval_last_point() {
        let xs = [0.0, 1.0, 2.0, 3.0];
        let (i, x) = find_interval(&xs, 3.0, Extrapolate::Error).unwrap();
        assert_eq!(i, 2);
        assert!((x - 3.0).abs() < 1e-15);
    }

    #[test]
    fn test_find_interval_error_below() {
        let xs = [0.0, 1.0, 2.0];
        let res = find_interval(&xs, -0.1, Extrapolate::Error);
        assert!(res.is_err());
    }

    #[test]
    fn test_find_interval_clamp_above() {
        let xs = [0.0, 1.0, 2.0];
        let (i, x) = find_interval(&xs, 5.0, Extrapolate::Clamp).unwrap();
        assert_eq!(i, 1);
        assert!((x - 2.0).abs() < 1e-15);
    }

    #[test]
    fn test_find_interval_extend_below() {
        let xs = [0.0, 1.0, 2.0];
        let (i, x) = find_interval(&xs, -1.0, Extrapolate::Extend).unwrap();
        assert_eq!(i, 0);
        assert!((x - (-1.0)).abs() < 1e-15);
    }

    #[test]
    fn test_validate_sorted_ok() {
        assert!(validate_sorted(&[0.0, 1.0, 2.0], 2).is_ok());
    }

    #[test]
    fn test_validate_sorted_too_few() {
        assert!(validate_sorted(&[0.0_f64], 2).is_err());
    }

    #[test]
    fn test_validate_sorted_not_increasing() {
        assert!(validate_sorted(&[0.0, 2.0, 1.0], 2).is_err());
    }

    #[test]
    fn test_interp1d_linear() {
        let result = interp1d(
            &[0.0, 1.0, 2.0],
            &[0.0, 2.0, 4.0],
            &[0.5, 1.5],
            Interp1dMethod::Linear,
        )
        .unwrap();
        assert!((result[0] - 1.0).abs() < 1e-12);
        assert!((result[1] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_interp1d_cubic_spline() {
        let result = interp1d(
            &[0.0, 1.0, 2.0, 3.0],
            &[0.0, 1.0, 4.0, 9.0],
            &[1.0, 2.0],
            Interp1dMethod::CubicSpline,
        )
        .unwrap();
        assert!((result[0] - 1.0).abs() < 1e-10);
        assert!((result[1] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_interp1d_bspline() {
        let result = interp1d(
            &[0.0, 1.0, 2.0, 3.0, 4.0],
            &[0.0, 1.0, 4.0, 9.0, 16.0],
            &[2.0],
            Interp1dMethod::BSpline,
        )
        .unwrap();
        assert!((result[0] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_interp2d_bilinear() {
        let xs = vec![0.0, 1.0];
        let ys = vec![0.0, 1.0];
        let zs = vec![vec![0.0, 2.0], vec![1.0, 3.0]]; // z = x + 2y
        let result = interp2d(xs, ys, zs, &[(0.5, 0.5)], Interp2dMethod::Bilinear).unwrap();
        assert!((result[0] - 1.5).abs() < 1e-12);
    }

    #[test]
    fn test_interp2d_bicubic() {
        let xs = vec![0.0, 1.0, 2.0, 3.0];
        let ys = vec![0.0, 1.0, 2.0, 3.0];
        let zs: Vec<Vec<f64>> = (0..4)
            .map(|i| (0..4).map(|j| f64::from(i) + 2.0 * f64::from(j)).collect())
            .collect();
        let result = interp2d(xs, ys, zs, &[(1.5, 1.5)], Interp2dMethod::Bicubic).unwrap();
        assert!((result[0] - 4.5).abs() < 1e-10);
    }
}
