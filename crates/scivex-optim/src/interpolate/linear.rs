//! Piecewise linear interpolation.

use scivex_core::Float;

use crate::error::{OptimError, Result};

use super::{Extrapolate, find_interval, validate_finite, validate_sorted};

/// Piecewise linear 1-D interpolator.
///
/// Constructed from sorted `(x, y)` data. Each query is evaluated via binary
/// search + linear blend in O(log n).
#[derive(Debug, Clone)]
pub struct Linear1d<T: Float> {
    xs: Vec<T>,
    ys: Vec<T>,
    extrap: Extrapolate,
}

impl<T: Float> Linear1d<T> {
    /// Create a new linear interpolator.
    ///
    /// # Errors
    ///
    /// - `xs` and `ys` must have the same length (>= 2).
    /// - `xs` must be strictly increasing.
    /// - All values must be finite.
    pub fn new(xs: &[T], ys: &[T], extrap: Extrapolate) -> Result<Self> {
        if xs.len() != ys.len() {
            return Err(OptimError::InvalidParameter {
                name: "ys",
                reason: "xs and ys must have the same length",
            });
        }
        validate_sorted(xs, 2)?;
        validate_finite(xs, "xs")?;
        validate_finite(ys, "ys")?;

        Ok(Self {
            xs: xs.to_vec(),
            ys: ys.to_vec(),
            extrap,
        })
    }

    /// Evaluate the interpolant at a single point.
    pub fn eval(&self, x: T) -> Result<T> {
        let (i, xq) = find_interval(&self.xs, x, self.extrap)?;
        let dx = self.xs[i + 1] - self.xs[i];
        let t = (xq - self.xs[i]) / dx;
        Ok(self.ys[i] * (T::one() - t) + self.ys[i + 1] * t)
    }

    /// Evaluate the interpolant at many points.
    pub fn eval_many(&self, xs: &[T]) -> Result<Vec<T>> {
        xs.iter().map(|&x| self.eval(x)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_exact() {
        // y = 2x on [0,3]
        let xs = [0.0, 1.0, 2.0, 3.0];
        let ys = [0.0, 2.0, 4.0, 6.0];
        let interp = Linear1d::new(&xs, &ys, Extrapolate::Error).unwrap();

        for &x in &[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0] {
            let y = interp.eval(x).unwrap();
            assert!((y - 2.0 * x).abs() < 1e-12, "x={x}, y={y}");
        }
    }

    #[test]
    fn test_linear_midpoint() {
        let interp = Linear1d::new(&[0.0, 1.0, 2.0], &[0.0, 1.0, 4.0], Extrapolate::Error).unwrap();
        let y = interp.eval(1.5).unwrap();
        assert!((y - 2.5).abs() < 1e-12);
    }

    #[test]
    fn test_linear_eval_many() {
        let interp = Linear1d::new(&[0.0, 1.0, 2.0], &[0.0, 1.0, 4.0], Extrapolate::Error).unwrap();
        let result = interp.eval_many(&[0.0, 1.0, 2.0]).unwrap();
        assert!(result[0].abs() < 1e-12);
        assert!((result[1] - 1.0).abs() < 1e-12);
        assert!((result[2] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn test_linear_out_of_range_error() {
        let interp = Linear1d::new(&[0.0, 1.0, 2.0], &[0.0, 1.0, 4.0], Extrapolate::Error).unwrap();
        assert!(interp.eval(-0.5).is_err());
        assert!(interp.eval(2.5).is_err());
    }

    #[test]
    fn test_linear_clamp() {
        let interp = Linear1d::new(&[0.0, 1.0, 2.0], &[1.0, 3.0, 7.0], Extrapolate::Clamp).unwrap();
        assert!((interp.eval(-1.0).unwrap() - 1.0).abs() < 1e-12);
        assert!((interp.eval(5.0).unwrap() - 7.0).abs() < 1e-12);
    }

    #[test]
    fn test_linear_unsorted_error() {
        assert!(Linear1d::new(&[0.0, 2.0, 1.0], &[0.0, 1.0, 2.0], Extrapolate::Error).is_err());
    }

    #[test]
    fn test_linear_f32() {
        let interp = Linear1d::new(
            &[0.0_f32, 1.0, 2.0],
            &[0.0_f32, 1.0, 4.0],
            Extrapolate::Error,
        )
        .unwrap();
        let y = interp.eval(1.5_f32).unwrap();
        assert!((y - 2.5_f32).abs() < 1e-6);
    }
}
