//! Natural and clamped cubic spline interpolation.

use scivex_core::Float;

use crate::error::{OptimError, Result};

use super::thomas::thomas_solve;
use super::{Extrapolate, SplineBoundary, find_interval, validate_finite, validate_sorted};

/// Cubic spline 1-D interpolator.
///
/// Precomputes polynomial coefficients for each segment at construction time.
/// Each evaluation is O(log n) via binary search.
#[derive(Debug, Clone)]
pub struct CubicSpline<T: Float> {
    xs: Vec<T>,
    /// Coefficients `(a, b, c, d)` per segment: `S_i(x) = a + b*(x-x_i) + c*(x-x_i)^2 + d*(x-x_i)^3`
    coeffs: Vec<(T, T, T, T)>,
    extrap: Extrapolate,
}

impl<T: Float> CubicSpline<T> {
    /// Construct a cubic spline interpolator.
    ///
    /// # Errors
    ///
    /// - `xs` and `ys` must have the same length (>= 3).
    /// - `xs` must be strictly increasing.
    /// - All values must be finite.
    pub fn new(
        xs: &[T],
        ys: &[T],
        boundary: SplineBoundary<T>,
        extrap: Extrapolate,
    ) -> Result<Self> {
        if xs.len() != ys.len() {
            return Err(OptimError::InvalidParameter {
                name: "ys",
                reason: "xs and ys must have the same length",
            });
        }
        validate_sorted(xs, 3)?;
        validate_finite(xs, "xs")?;
        validate_finite(ys, "ys")?;

        let n = xs.len();

        // Step widths and divided differences
        let h: Vec<T> = (0..n - 1).map(|i| xs[i + 1] - xs[i]).collect();
        let delta: Vec<T> = (0..n - 1).map(|i| (ys[i + 1] - ys[i]) / h[i]).collect();

        let coeffs = match boundary {
            SplineBoundary::Natural => Self::solve_natural(ys, &h, &delta)?,
            SplineBoundary::Clamped { left, right } => {
                Self::solve_clamped(n, ys, &h, &delta, left, right)?
            }
        };

        Ok(Self {
            xs: xs.to_vec(),
            coeffs,
            extrap,
        })
    }

    fn solve_natural(ys: &[T], h: &[T], delta: &[T]) -> Result<Vec<(T, T, T, T)>> {
        let n = ys.len();
        let two = T::from_f64(2.0);
        let six = T::from_f64(6.0);

        if n == 3 {
            let three = T::from_f64(3.0);
            let m1 = three * (delta[1] - delta[0]) / (h[0] + h[1]);
            let m = vec![T::zero(), m1, T::zero()];
            return Ok(Self::build_coeffs(ys, h, &m, two, six));
        }

        let size = n - 2;
        let mut sub = Vec::with_capacity(size - 1);
        let mut diag = Vec::with_capacity(size);
        let mut sup = Vec::with_capacity(size - 1);
        let mut rhs = Vec::with_capacity(size);

        for i in 0..size {
            let row = i + 1;
            diag.push(two * (h[row - 1] + h[row]));
            rhs.push(six * (delta[row] - delta[row - 1]));
            if i > 0 {
                sub.push(h[row - 1]);
            }
            if i < size - 1 {
                sup.push(h[row]);
            }
        }

        let m_interior = thomas_solve(&sub, &diag, &sup, &rhs)?;

        let mut m = Vec::with_capacity(n);
        m.push(T::zero());
        m.extend_from_slice(&m_interior);
        m.push(T::zero());

        Ok(Self::build_coeffs(ys, h, &m, two, six))
    }

    fn solve_clamped(
        n: usize,
        ys: &[T],
        h: &[T],
        delta: &[T],
        left_deriv: T,
        right_deriv: T,
    ) -> Result<Vec<(T, T, T, T)>> {
        let two = T::from_f64(2.0);
        let six = T::from_f64(6.0);

        let mut sub = Vec::with_capacity(n - 1);
        let mut diag = Vec::with_capacity(n);
        let mut sup = Vec::with_capacity(n - 1);
        let mut rhs = Vec::with_capacity(n);

        // First row
        diag.push(two * h[0]);
        sup.push(h[0]);
        rhs.push(six * (delta[0] - left_deriv));

        // Interior rows
        for i in 1..n - 1 {
            sub.push(h[i - 1]);
            diag.push(two * (h[i - 1] + h[i]));
            sup.push(h[i]);
            rhs.push(six * (delta[i] - delta[i - 1]));
        }

        // Last row
        sub.push(h[n - 2]);
        diag.push(two * h[n - 2]);
        rhs.push(six * (right_deriv - delta[n - 2]));

        let m = thomas_solve(&sub, &diag, &sup, &rhs)?;

        Ok(Self::build_coeffs(ys, h, &m, two, six))
    }

    fn build_coeffs(ys: &[T], h: &[T], m: &[T], two: T, six: T) -> Vec<(T, T, T, T)> {
        let nm1 = h.len();
        let mut coeffs = Vec::with_capacity(nm1);
        for i in 0..nm1 {
            let a = ys[i];
            let b = (ys[i + 1] - ys[i]) / h[i] - h[i] * (two * m[i] + m[i + 1]) / six;
            let c = m[i] / two;
            let d = (m[i + 1] - m[i]) / (six * h[i]);
            coeffs.push((a, b, c, d));
        }
        coeffs
    }

    /// Evaluate the spline at a single point.
    pub fn eval(&self, x: T) -> Result<T> {
        let (i, xq) = find_interval(&self.xs, x, self.extrap)?;
        let dx = xq - self.xs[i];
        let (a, b, c, d) = self.coeffs[i];
        Ok(a + dx * (b + dx * (c + dx * d)))
    }

    /// Evaluate at many points.
    pub fn eval_many(&self, xs: &[T]) -> Result<Vec<T>> {
        xs.iter().map(|&x| self.eval(x)).collect()
    }

    /// Evaluate the first derivative at a single point.
    pub fn derivative(&self, x: T) -> Result<T> {
        let (i, xq) = find_interval(&self.xs, x, self.extrap)?;
        let dx = xq - self.xs[i];
        let (_, b, c, d) = self.coeffs[i];
        let two = T::from_f64(2.0);
        let three = T::from_f64(3.0);
        Ok(b + dx * (two * c + three * d * dx))
    }

    /// Evaluate the second derivative at a single point.
    pub fn second_derivative(&self, x: T) -> Result<T> {
        let (i, xq) = find_interval(&self.xs, x, self.extrap)?;
        let dx = xq - self.xs[i];
        let (_, _, c, d) = self.coeffs[i];
        let two = T::from_f64(2.0);
        let six = T::from_f64(6.0);
        Ok(two * c + six * d * dx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cubic_spline_reproduces_data() {
        let xs = [0.0, 1.0, 2.0, 3.0, 4.0];
        let ys = [0.0, 1.0, 4.0, 9.0, 16.0];
        let spline =
            CubicSpline::new(&xs, &ys, SplineBoundary::Natural, Extrapolate::Error).unwrap();
        for (i, &x) in xs.iter().enumerate() {
            let y = spline.eval(x).unwrap();
            assert!(
                (y - ys[i]).abs() < 1e-10,
                "at x={x}: got {y}, expected {}",
                ys[i]
            );
        }
    }

    #[test]
    fn test_cubic_spline_natural_boundary() {
        let spline = CubicSpline::new(
            &[0.0, 1.0, 2.0, 3.0, 4.0],
            &[0.0, 0.5, 2.0, 1.5, 0.0],
            SplineBoundary::Natural,
            Extrapolate::Error,
        )
        .unwrap();
        let sd_left = spline.second_derivative(0.0).unwrap();
        let sd_right = spline.second_derivative(4.0).unwrap();
        assert!(sd_left.abs() < 1e-10, "left 2nd deriv = {sd_left}");
        assert!(sd_right.abs() < 1e-10, "right 2nd deriv = {sd_right}");
    }

    #[test]
    fn test_cubic_spline_clamped() {
        let spline = CubicSpline::new(
            &[0.0, 1.0, 2.0, 3.0],
            &[0.0, 1.0, 4.0, 9.0],
            SplineBoundary::Clamped {
                left: 0.0,
                right: 6.0,
            },
            Extrapolate::Error,
        )
        .unwrap();

        let d_left = spline.derivative(0.0).unwrap();
        let d_right = spline.derivative(3.0).unwrap();
        assert!(d_left.abs() < 1e-8, "left derivative = {d_left}");
        assert!((d_right - 6.0).abs() < 1e-8, "right derivative = {d_right}");
    }

    #[test]
    fn test_cubic_spline_exact_for_cubic() {
        let xs = [0.0_f64, 1.0, 2.0, 3.0];
        let ys: Vec<f64> = xs.iter().map(|&x| x * x * x).collect();
        let spline = CubicSpline::new(
            &xs,
            &ys,
            SplineBoundary::Clamped {
                left: 0.0,
                right: 27.0,
            },
            Extrapolate::Error,
        )
        .unwrap();

        for x in [0.5, 1.0, 1.5, 2.0, 2.5] {
            let y = spline.eval(x).unwrap();
            let expected = x * x * x;
            assert!(
                (y - expected).abs() < 1e-8,
                "at x={x}: got {y}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_cubic_spline_derivative() {
        let xs = [0.0, 1.0, 2.0, 3.0, 4.0];
        let ys: Vec<f64> = xs.iter().map(|&x| x * x).collect();
        let spline =
            CubicSpline::new(&xs, &ys, SplineBoundary::Natural, Extrapolate::Error).unwrap();
        let d = spline.derivative(2.0).unwrap();
        assert!((d - 4.0).abs() < 0.5, "derivative at 2.0 = {d}");
    }

    #[test]
    fn test_cubic_spline_continuity() {
        let xs = [0.0, 1.0, 2.0, 3.0, 4.0];
        let ys = [0.0, 1.0, 0.0, 1.0, 0.0];
        let spline =
            CubicSpline::new(&xs, &ys, SplineBoundary::Natural, Extrapolate::Error).unwrap();

        let eps = 1e-8;
        for &x in &xs[1..xs.len() - 1] {
            let left = spline.eval(x - eps).unwrap();
            let right = spline.eval(x).unwrap();
            assert!(
                (left - right).abs() < 1e-5,
                "discontinuity at x={x}: left={left}, right={right}"
            );
        }
    }
}
