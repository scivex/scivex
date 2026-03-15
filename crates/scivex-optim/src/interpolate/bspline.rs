//! Uniform B-spline interpolation via de Boor's algorithm.

use scivex_core::Float;

use crate::error::{OptimError, Result};

use super::{Extrapolate, validate_finite, validate_sorted};

/// B-spline interpolator.
///
/// Supports arbitrary degree with clamped (open) uniform knot vectors.
/// Construction fits control points to interpolate the data exactly.
/// Evaluation uses de Boor's algorithm.
#[derive(Debug, Clone)]
pub struct BSpline<T: Float> {
    knots: Vec<T>,
    control_points: Vec<T>,
    degree: usize,
    extrap: Extrapolate,
    x_min: T,
    x_max: T,
}

impl<T: Float> BSpline<T> {
    /// Fit a B-spline of the given `degree` that interpolates `(xs, ys)`.
    ///
    /// Uses a clamped uniform knot vector. Solves the interpolation
    /// conditions to find control points.
    ///
    /// # Errors
    ///
    /// - `degree` must be >= 1 and < `xs.len()`.
    /// - `xs` and `ys` must have the same length (>= `degree + 1`).
    /// - `xs` must be strictly increasing.
    pub fn fit(xs: &[T], ys: &[T], degree: usize, extrap: Extrapolate) -> Result<Self> {
        if xs.len() != ys.len() {
            return Err(OptimError::InvalidParameter {
                name: "ys",
                reason: "xs and ys must have the same length",
            });
        }
        if degree < 1 {
            return Err(OptimError::InvalidParameter {
                name: "degree",
                reason: "degree must be >= 1",
            });
        }
        if xs.len() < degree + 1 {
            return Err(OptimError::InvalidParameter {
                name: "xs",
                reason: "need at least degree+1 data points",
            });
        }
        validate_sorted(xs, 2)?;
        validate_finite(xs, "xs")?;
        validate_finite(ys, "ys")?;

        let n = xs.len();
        let x_min = xs[0];
        let x_max = xs[n - 1];

        let knots = Self::clamped_knots(xs, degree);
        let control_points = Self::solve_control_points(&knots, xs, ys, degree)?;

        Ok(Self {
            knots,
            control_points,
            degree,
            extrap,
            x_min,
            x_max,
        })
    }

    /// Create a B-spline directly from knots and control points.
    pub fn from_knots(
        knots: Vec<T>,
        control_points: Vec<T>,
        degree: usize,
        extrap: Extrapolate,
    ) -> Result<Self> {
        if knots.len() != control_points.len() + degree + 1 {
            return Err(OptimError::InvalidParameter {
                name: "knots",
                reason: "knots.len() must equal control_points.len() + degree + 1",
            });
        }
        if degree < 1 {
            return Err(OptimError::InvalidParameter {
                name: "degree",
                reason: "degree must be >= 1",
            });
        }

        let x_min = knots[degree];
        let x_max = knots[knots.len() - degree - 1];

        Ok(Self {
            knots,
            control_points,
            degree,
            extrap,
            x_min,
            x_max,
        })
    }

    /// Evaluate the B-spline at a single point using de Boor's algorithm.
    pub fn eval(&self, x: T) -> Result<T> {
        let xq = self.handle_extrap(x)?;
        Ok(self.de_boor(xq))
    }

    /// Evaluate at many points.
    pub fn eval_many(&self, xs: &[T]) -> Result<Vec<T>> {
        xs.iter().map(|&x| self.eval(x)).collect()
    }

    fn handle_extrap(&self, x: T) -> Result<T> {
        if x >= self.x_min && x <= self.x_max {
            return Ok(x);
        }
        match self.extrap {
            Extrapolate::Error => Err(OptimError::InvalidParameter {
                name: "x",
                reason: if x < self.x_min {
                    "query point is below data range"
                } else {
                    "query point is above data range"
                },
            }),
            Extrapolate::Clamp | Extrapolate::Extend => Ok(x.max(self.x_min).min(self.x_max)),
        }
    }

    fn clamped_knots(xs: &[T], degree: usize) -> Vec<T> {
        let n = xs.len();
        let num_knots = n + degree + 1;
        let mut knots = Vec::with_capacity(num_knots);

        for _ in 0..=degree {
            knots.push(xs[0]);
        }

        for i in 1..n - degree {
            let sum: T = xs[i..i + degree].iter().copied().sum();
            knots.push(sum / T::from_usize(degree));
        }

        for _ in 0..=degree {
            knots.push(xs[n - 1]);
        }

        knots
    }

    #[allow(clippy::needless_range_loop)]
    fn solve_control_points(knots: &[T], xs: &[T], ys: &[T], degree: usize) -> Result<Vec<T>> {
        let n = xs.len();

        let mut mat = vec![vec![T::zero(); n]; n];
        for i in 0..n {
            for j in 0..n {
                mat[i][j] = Self::basis_fn(knots, j, degree, xs[i]);
            }
        }

        // Gaussian elimination with partial pivoting
        let mut aug: Vec<Vec<T>> = mat
            .into_iter()
            .enumerate()
            .map(|(i, mut row)| {
                row.push(ys[i]);
                row
            })
            .collect();

        for col in 0..n {
            let mut max_row = col;
            let mut max_val = aug[col][col].abs();
            for row in col + 1..n {
                let v = aug[row][col].abs();
                if v > max_val {
                    max_val = v;
                    max_row = row;
                }
            }
            if max_val < T::epsilon() * T::from_f64(100.0) {
                return Err(OptimError::InvalidParameter {
                    name: "xs",
                    reason: "singular collocation matrix",
                });
            }
            aug.swap(col, max_row);

            let pivot = aug[col][col];
            for row in col + 1..n {
                let factor = aug[row][col] / pivot;
                for k in col..=n {
                    let val = aug[col][k];
                    aug[row][k] -= factor * val;
                }
            }
        }

        let mut cp = vec![T::zero(); n];
        for i in (0..n).rev() {
            let mut sum = aug[i][n];
            for j in i + 1..n {
                sum -= aug[i][j] * cp[j];
            }
            cp[i] = sum / aug[i][i];
        }

        Ok(cp)
    }

    /// Evaluate B-spline basis function B_{i,p}(x) using Cox-de Boor recursion.
    fn basis_fn(knots: &[T], i: usize, p: usize, x: T) -> T {
        if p == 0 {
            if x >= knots[i] && x < knots[i + 1] {
                return T::one();
            }
            // Right endpoint: if this is the last non-zero-width span and
            // x equals the right boundary, include it (closed on the right).
            if knots[i] < knots[i + 1] && x == knots[i + 1] {
                let is_last_real = (i + 2..knots.len()).all(|k| knots[k] == knots[i + 1]);
                if is_last_real {
                    return T::one();
                }
            }
            return T::zero();
        }

        let mut result = T::zero();

        let denom1 = knots[i + p] - knots[i];
        if denom1 > T::zero() {
            result += (x - knots[i]) / denom1 * Self::basis_fn(knots, i, p - 1, x);
        }

        let denom2 = knots[i + p + 1] - knots[i + 1];
        if denom2 > T::zero() {
            result += (knots[i + p + 1] - x) / denom2 * Self::basis_fn(knots, i + 1, p - 1, x);
        }

        result
    }

    /// De Boor's algorithm for efficient evaluation.
    fn de_boor(&self, x: T) -> T {
        let p = self.degree;
        let n = self.knots.len() - p - 1;

        let mut k = p;
        for i in p..n {
            if x >= self.knots[i] && x < self.knots[i + 1] {
                k = i;
                break;
            }
        }
        if x >= self.knots[n] {
            k = n - 1;
        }

        let mut d: Vec<T> = (0..=p)
            .map(|j| {
                let idx = k - p + j;
                if idx < self.control_points.len() {
                    self.control_points[idx]
                } else {
                    T::zero()
                }
            })
            .collect();

        for r in 1..=p {
            for j in (r..=p).rev() {
                let idx = k - p + j;
                let denom = self.knots[idx + p + 1 - r] - self.knots[idx];
                if denom > T::zero() {
                    let alpha = (x - self.knots[idx]) / denom;
                    d[j] = d[j - 1] * (T::one() - alpha) + d[j] * alpha;
                }
            }
        }

        d[p]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bspline_reproduces_data() {
        let xs = [0.0, 1.0, 2.0, 3.0, 4.0];
        let ys = [0.0, 1.0, 4.0, 9.0, 16.0];
        let spline = BSpline::fit(&xs, &ys, 3, Extrapolate::Error).unwrap();

        for (i, &x) in xs.iter().enumerate() {
            let y = spline.eval(x).unwrap();
            assert!(
                (y - ys[i]).abs() < 1e-6,
                "at x={x}: got {y}, expected {}",
                ys[i]
            );
        }
    }

    #[test]
    fn test_bspline_degree1_is_linear() {
        let spline = BSpline::fit(
            &[0.0, 1.0, 2.0, 3.0],
            &[0.0, 2.0, 4.0, 6.0],
            1,
            Extrapolate::Error,
        )
        .unwrap();
        let y = spline.eval(1.5).unwrap();
        assert!((y - 3.0).abs() < 1e-6, "got {y}");
    }

    #[test]
    fn test_bspline_degree3_smooth() {
        let xs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let ys: Vec<f64> = xs.iter().map(|&x| x.sin()).collect();
        let spline = BSpline::fit(&xs, &ys, 3, Extrapolate::Error).unwrap();

        for i in 0..40 {
            let x = 0.125 * f64::from(i);
            let _ = spline.eval(x).unwrap();
        }
    }

    #[test]
    fn test_bspline_invalid_degree() {
        assert!(BSpline::fit(&[0.0, 1.0, 2.0], &[0.0, 1.0, 4.0], 0, Extrapolate::Error).is_err());
        assert!(BSpline::fit(&[0.0, 1.0, 2.0], &[0.0, 1.0, 4.0], 3, Extrapolate::Error).is_err());
    }

    #[test]
    fn test_bspline_from_knots() {
        let knots = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let cps = vec![0.0, 1.0, 2.0, 3.0];
        let spline = BSpline::from_knots(knots, cps, 3, Extrapolate::Error).unwrap();
        let y = spline.eval(0.5).unwrap();
        assert!(y.is_finite());
    }
}
