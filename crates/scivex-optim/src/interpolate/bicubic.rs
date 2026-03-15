//! Bicubic interpolation on a 2-D rectilinear grid.
//!
//! Uses Catmull-Rom finite differences for derivatives at grid points,
//! then builds a 16-coefficient polynomial per cell.

use scivex_core::Float;

use crate::error::{OptimError, Result};

use super::{Extrapolate, find_interval, validate_finite, validate_sorted};

/// Bicubic 2-D interpolator on a rectilinear grid.
///
/// Precomputes 16 polynomial coefficients per grid cell at construction time.
/// Each evaluation is O(log M + log N) for grid search plus O(1) polynomial
/// evaluation.
#[derive(Debug, Clone)]
pub struct Bicubic2d<T: Float> {
    xs: Vec<T>,
    ys: Vec<T>,
    /// Coefficients per cell: `cells[ix][iy]` is a 4x4 matrix in row-major order.
    cells: Vec<Vec<[T; 16]>>,
    extrap: Extrapolate,
}

impl<T: Float> Bicubic2d<T> {
    /// Create a new bicubic interpolator.
    ///
    /// `zs[i][j]` is the value at `(xs[i], ys[j])`.
    ///
    /// # Errors
    ///
    /// - `xs` and `ys` must be strictly increasing with length >= 2.
    /// - `zs` must have shape `[xs.len()][ys.len()]`.
    pub fn new(xs: Vec<T>, ys: Vec<T>, zs: &[Vec<T>], extrap: Extrapolate) -> Result<Self> {
        validate_sorted(&xs, 2)?;
        validate_sorted(&ys, 2)?;
        validate_finite(&xs, "xs")?;
        validate_finite(&ys, "ys")?;

        let nx = xs.len();
        let ny = ys.len();

        if zs.len() != nx {
            return Err(OptimError::InvalidParameter {
                name: "zs",
                reason: "zs row count must match xs length",
            });
        }
        for row in zs {
            if row.len() != ny {
                return Err(OptimError::InvalidParameter {
                    name: "zs",
                    reason: "all zs rows must have length equal to ys length",
                });
            }
            validate_finite(row, "zs")?;
        }

        // Compute derivatives using Catmull-Rom finite differences
        let dzdx = Self::compute_dx(&xs, zs, nx, ny);
        let dzdy = Self::compute_dy(&ys, zs, nx, ny);
        let d2zdxdy = Self::compute_dxy(&xs, &dzdy, nx, ny);

        // Precompute coefficients for each cell
        let mut cells = Vec::with_capacity(nx - 1);
        for ix in 0..nx - 1 {
            let mut row_cells = Vec::with_capacity(ny - 1);
            let hx = xs[ix + 1] - xs[ix];
            for iy in 0..ny - 1 {
                let hy = ys[iy + 1] - ys[iy];
                let coeffs = Self::compute_cell_coeffs(zs, &dzdx, &dzdy, &d2zdxdy, ix, iy, hx, hy);
                row_cells.push(coeffs);
            }
            cells.push(row_cells);
        }

        Ok(Self {
            xs,
            ys,
            cells,
            extrap,
        })
    }

    /// Evaluate at a single `(x, y)` point.
    pub fn eval(&self, x: T, y: T) -> Result<T> {
        let (ix, xq) = find_interval(&self.xs, x, self.extrap)?;
        let (iy, yq) = find_interval(&self.ys, y, self.extrap)?;

        let hx = self.xs[ix + 1] - self.xs[ix];
        let hy = self.ys[iy + 1] - self.ys[iy];

        let t = (xq - self.xs[ix]) / hx;
        let u = (yq - self.ys[iy]) / hy;

        let coeffs = &self.cells[ix][iy];

        let mut result = T::zero();
        let mut t_pow = T::one();
        for i in 0..4 {
            let mut u_pow = T::one();
            for j in 0..4 {
                result += coeffs[i * 4 + j] * t_pow * u_pow;
                u_pow *= u;
            }
            t_pow *= t;
        }

        Ok(result)
    }

    /// Evaluate at many points.
    pub fn eval_many(&self, points: &[(T, T)]) -> Result<Vec<T>> {
        points.iter().map(|&(x, y)| self.eval(x, y)).collect()
    }

    /// Compute dz/dx using Catmull-Rom (central differences in interior,
    /// one-sided at boundaries).
    fn compute_dx(xs: &[T], zs: &[Vec<T>], nx: usize, ny: usize) -> Vec<Vec<T>> {
        let mut dzdx = vec![vec![T::zero(); ny]; nx];
        for j in 0..ny {
            for i in 0..nx {
                dzdx[i][j] = if i == 0 {
                    (zs[1][j] - zs[0][j]) / (xs[1] - xs[0])
                } else if i == nx - 1 {
                    (zs[nx - 1][j] - zs[nx - 2][j]) / (xs[nx - 1] - xs[nx - 2])
                } else {
                    (zs[i + 1][j] - zs[i - 1][j]) / (xs[i + 1] - xs[i - 1])
                };
            }
        }
        dzdx
    }

    /// Compute dz/dy using Catmull-Rom.
    fn compute_dy(ys: &[T], zs: &[Vec<T>], nx: usize, ny: usize) -> Vec<Vec<T>> {
        let mut dzdy = vec![vec![T::zero(); ny]; nx];
        for i in 0..nx {
            for j in 0..ny {
                dzdy[i][j] = if j == 0 {
                    (zs[i][1] - zs[i][0]) / (ys[1] - ys[0])
                } else if j == ny - 1 {
                    (zs[i][ny - 1] - zs[i][ny - 2]) / (ys[ny - 1] - ys[ny - 2])
                } else {
                    (zs[i][j + 1] - zs[i][j - 1]) / (ys[j + 1] - ys[j - 1])
                };
            }
        }
        dzdy
    }

    /// Compute d2z/dxdy.
    fn compute_dxy(xs: &[T], dzdy: &[Vec<T>], nx: usize, ny: usize) -> Vec<Vec<T>> {
        let mut d2zdxdy = vec![vec![T::zero(); ny]; nx];
        for j in 0..ny {
            for i in 0..nx {
                d2zdxdy[i][j] = if i == 0 {
                    (dzdy[1][j] - dzdy[0][j]) / (xs[1] - xs[0])
                } else if i == nx - 1 {
                    (dzdy[nx - 1][j] - dzdy[nx - 2][j]) / (xs[nx - 1] - xs[nx - 2])
                } else {
                    (dzdy[i + 1][j] - dzdy[i - 1][j]) / (xs[i + 1] - xs[i - 1])
                };
            }
        }
        d2zdxdy
    }

    /// Compute the 16 bicubic coefficients for one cell using Hermite basis.
    #[allow(clippy::too_many_arguments)]
    fn compute_cell_coeffs(
        zs: &[Vec<T>],
        dzdx: &[Vec<T>],
        dzdy: &[Vec<T>],
        d2zdxdy: &[Vec<T>],
        ix: usize,
        iy: usize,
        hx: T,
        hy: T,
    ) -> [T; 16] {
        let f = [
            zs[ix][iy],
            zs[ix + 1][iy],
            zs[ix][iy + 1],
            zs[ix + 1][iy + 1],
        ];
        let fx = [
            dzdx[ix][iy] * hx,
            dzdx[ix + 1][iy] * hx,
            dzdx[ix][iy + 1] * hx,
            dzdx[ix + 1][iy + 1] * hx,
        ];
        let fy = [
            dzdy[ix][iy] * hy,
            dzdy[ix + 1][iy] * hy,
            dzdy[ix][iy + 1] * hy,
            dzdy[ix + 1][iy + 1] * hy,
        ];
        let fxy = [
            d2zdxdy[ix][iy] * hx * hy,
            d2zdxdy[ix + 1][iy] * hx * hy,
            d2zdxdy[ix][iy + 1] * hx * hy,
            d2zdxdy[ix + 1][iy + 1] * hx * hy,
        ];

        let x = [
            f[0], f[1], f[2], f[3], fx[0], fx[1], fx[2], fx[3], fy[0], fy[1], fy[2], fy[3], fxy[0],
            fxy[1], fxy[2], fxy[3],
        ];

        // The inverse of the bicubic coefficient matrix (standard result)
        #[rustfmt::skip]
        let inv: [[i8; 16]; 16] = [
            [ 1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [-3,  3,  0,  0, -2, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 2, -2,  0,  0,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0,  0,  0, -3,  3,  0,  0, -2, -1,  0,  0],
            [ 0,  0,  0,  0,  0,  0,  0,  0,  2, -2,  0,  0,  1,  1,  0,  0],
            [-3,  0,  3,  0,  0,  0,  0,  0, -2,  0, -1,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0, -3,  0,  3,  0,  0,  0,  0,  0, -2,  0, -1,  0],
            [ 9, -9, -9,  9,  6,  3, -6, -3,  6, -6,  3, -3,  4,  2,  2,  1],
            [-6,  6,  6, -6, -3, -3,  3,  3, -4,  4, -2,  2, -2, -2, -1, -1],
            [ 2,  0, -2,  0,  0,  0,  0,  0,  1,  0,  1,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  2,  0, -2,  0,  0,  0,  0,  0,  1,  0,  1,  0],
            [-6,  6,  6, -6, -4, -2,  4,  2, -3,  3, -3,  3, -2, -1, -2, -1],
            [ 4, -4, -4,  4,  2,  2, -2, -2,  2, -2,  2, -2,  1,  1,  1,  1],
        ];

        let mut alpha = [T::zero(); 16];
        for i in 0..16 {
            let mut sum = T::zero();
            for j in 0..16 {
                sum += T::from_f64(f64::from(inv[i][j])) * x[j];
            }
            alpha[i] = sum;
        }

        alpha
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bicubic_exact_on_bilinear_fn() {
        // z = x + 2y — a bilinear function should be reproduced exactly
        let xs = vec![0.0, 1.0, 2.0, 3.0];
        let ys = vec![0.0, 1.0, 2.0, 3.0];
        let zs: Vec<Vec<f64>> = xs
            .iter()
            .map(|&x| ys.iter().map(|&y| x + 2.0 * y).collect())
            .collect();

        let interp = Bicubic2d::new(xs, ys, &zs, Extrapolate::Error).unwrap();

        for &(x, y) in &[(0.5, 0.5), (1.5, 1.5), (2.5, 0.5)] {
            let z = interp.eval(x, y).unwrap();
            let expected = x + 2.0 * y;
            assert!(
                (z - expected).abs() < 1e-10,
                "at ({x},{y}): got {z}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_bicubic_smoother_than_bilinear() {
        // z = sin(x)*cos(y) with a dense grid
        let xs: Vec<f64> = (0..7).map(|i| f64::from(i) * 0.5).collect();
        let ys = xs.clone();
        let zs: Vec<Vec<f64>> = xs
            .iter()
            .map(|&x| ys.iter().map(|&y| x.sin() * y.cos()).collect())
            .collect();

        let bicubic = Bicubic2d::new(xs.clone(), ys.clone(), &zs, Extrapolate::Error).unwrap();
        let bilinear =
            super::super::Bilinear2d::new(xs.clone(), ys.clone(), zs, Extrapolate::Error).unwrap();

        let mut bicubic_err = 0.0_f64;
        let mut bilinear_err = 0.0_f64;

        for i in 0..xs.len() - 1 {
            for j in 0..ys.len() - 1 {
                let x = (xs[i] + xs[i + 1]) * 0.5;
                let y = (ys[j] + ys[j + 1]) * 0.5;
                let truth = x.sin() * y.cos();
                let bc = bicubic.eval(x, y).unwrap();
                let bl = bilinear.eval(x, y).unwrap();
                bicubic_err += (bc - truth).abs();
                bilinear_err += (bl - truth).abs();
            }
        }

        assert!(
            bicubic_err <= bilinear_err + 1e-10,
            "bicubic_err={bicubic_err}, bilinear_err={bilinear_err}"
        );
    }

    #[test]
    fn test_bicubic_grid_point_exact() {
        let zs = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let interp = Bicubic2d::new(
            vec![0.0, 1.0, 2.0],
            vec![0.0, 1.0, 2.0],
            &zs,
            Extrapolate::Error,
        )
        .unwrap();

        assert!((interp.eval(0.0, 0.0).unwrap() - 1.0).abs() < 1e-10);
        assert!((interp.eval(1.0, 1.0).unwrap() - 5.0).abs() < 1e-10);
        assert!((interp.eval(2.0, 2.0).unwrap() - 9.0).abs() < 1e-10);
    }
}
