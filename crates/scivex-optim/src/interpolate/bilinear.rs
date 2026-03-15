//! Bilinear interpolation on a 2-D rectilinear grid.

use scivex_core::Float;

use crate::error::{OptimError, Result};

use super::{Extrapolate, find_interval, validate_finite, validate_sorted};

/// Bilinear 2-D interpolator on a rectilinear grid.
///
/// Given grid axes `xs` (length M) and `ys` (length N) and values
/// `zs[i][j]` (M rows, N columns), interpolates via weighted average of
/// the four surrounding grid corners.
#[derive(Debug, Clone)]
pub struct Bilinear2d<T: Float> {
    xs: Vec<T>,
    ys: Vec<T>,
    zs: Vec<Vec<T>>,
    extrap: Extrapolate,
}

impl<T: Float> Bilinear2d<T> {
    /// Create a new bilinear interpolator.
    ///
    /// `zs[i][j]` is the value at `(xs[i], ys[j])`.
    ///
    /// # Errors
    ///
    /// - `xs` and `ys` must be strictly increasing with length >= 2.
    /// - `zs` must have shape `[xs.len()][ys.len()]`.
    /// - All values must be finite.
    pub fn new(xs: Vec<T>, ys: Vec<T>, zs: Vec<Vec<T>>, extrap: Extrapolate) -> Result<Self> {
        validate_sorted(&xs, 2)?;
        validate_sorted(&ys, 2)?;
        validate_finite(&xs, "xs")?;
        validate_finite(&ys, "ys")?;

        if zs.len() != xs.len() {
            return Err(OptimError::InvalidParameter {
                name: "zs",
                reason: "zs row count must match xs length",
            });
        }
        for row in &zs {
            if row.len() != ys.len() {
                return Err(OptimError::InvalidParameter {
                    name: "zs",
                    reason: "all zs rows must have length equal to ys length",
                });
            }
            validate_finite(row, "zs")?;
        }

        Ok(Self { xs, ys, zs, extrap })
    }

    /// Evaluate at a single `(x, y)` point.
    pub fn eval(&self, x: T, y: T) -> Result<T> {
        let (ix, xq) = find_interval(&self.xs, x, self.extrap)?;
        let (iy, yq) = find_interval(&self.ys, y, self.extrap)?;

        let x0 = self.xs[ix];
        let x1 = self.xs[ix + 1];
        let y0 = self.ys[iy];
        let y1 = self.ys[iy + 1];

        let tx = (xq - x0) / (x1 - x0);
        let ty = (yq - y0) / (y1 - y0);

        let z00 = self.zs[ix][iy];
        let z10 = self.zs[ix + 1][iy];
        let z01 = self.zs[ix][iy + 1];
        let z11 = self.zs[ix + 1][iy + 1];

        let one = T::one();
        let result = z00 * (one - tx) * (one - ty)
            + z10 * tx * (one - ty)
            + z01 * (one - tx) * ty
            + z11 * tx * ty;

        Ok(result)
    }

    /// Evaluate at many `(x, y)` points.
    pub fn eval_many(&self, points: &[(T, T)]) -> Result<Vec<T>> {
        points.iter().map(|&(x, y)| self.eval(x, y)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bilinear_exact_on_bilinear_fn() {
        // z = x + 2y
        let xs = vec![0.0, 1.0, 2.0];
        let ys = vec![0.0, 1.0, 2.0];
        let zs: Vec<Vec<f64>> = xs
            .iter()
            .map(|&x| ys.iter().map(|&y| x + 2.0 * y).collect())
            .collect();

        let interp = Bilinear2d::new(xs, ys, zs, Extrapolate::Error).unwrap();

        for &(x, y) in &[(0.5, 0.5), (1.5, 0.5), (0.5, 1.5), (1.0, 1.0)] {
            let z = interp.eval(x, y).unwrap();
            let expected = x + 2.0 * y;
            assert!(
                (z - expected).abs() < 1e-12,
                "at ({x},{y}): got {z}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_bilinear_cell_center() {
        let interp = Bilinear2d::new(
            vec![0.0, 1.0],
            vec![0.0, 1.0],
            vec![vec![0.0, 1.0], vec![1.0, 2.0]],
            Extrapolate::Error,
        )
        .unwrap();
        let z = interp.eval(0.5, 0.5).unwrap();
        assert!((z - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_bilinear_grid_point_exact() {
        let zs = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let interp = Bilinear2d::new(
            vec![0.0, 1.0, 2.0],
            vec![0.0, 1.0, 2.0],
            zs,
            Extrapolate::Error,
        )
        .unwrap();

        assert!((interp.eval(0.0, 0.0).unwrap() - 1.0).abs() < 1e-12);
        assert!((interp.eval(1.0, 1.0).unwrap() - 5.0).abs() < 1e-12);
        assert!((interp.eval(2.0, 2.0).unwrap() - 9.0).abs() < 1e-12);
    }

    #[test]
    fn test_bilinear_out_of_range_error() {
        let interp = Bilinear2d::new(
            vec![0.0, 1.0],
            vec![0.0, 1.0],
            vec![vec![0.0, 1.0], vec![1.0, 2.0]],
            Extrapolate::Error,
        )
        .unwrap();
        assert!(interp.eval(-0.1, 0.5).is_err());
        assert!(interp.eval(0.5, 1.1).is_err());
    }
}
