//! Thomas algorithm (tridiagonal matrix solver).
//!
//! Solves `A x = d` where `A` is tridiagonal in O(n) time and O(n) space.
//! This is an internal helper used by cubic spline construction.

use scivex_core::Float;

use crate::error::{OptimError, Result};

/// Solve a tridiagonal system `A x = d` using the Thomas algorithm.
///
/// The tridiagonal matrix is specified by:
/// - `a` — sub-diagonal (length n-1), `A[i][i-1]` for i = 1..n
/// - `b` — main diagonal  (length n),   `A[i][i]`
/// - `c` — super-diagonal (length n-1), `A[i][i+1]` for i = 0..n-1
/// - `d` — right-hand side (length n)
///
/// Returns the solution vector `x` of length n.
pub(super) fn thomas_solve<T: Float>(a: &[T], b: &[T], c: &[T], d: &[T]) -> Result<Vec<T>> {
    let n = b.len();
    if n == 0 {
        return Err(OptimError::InvalidParameter {
            name: "b",
            reason: "empty system",
        });
    }
    if a.len() != n - 1 || c.len() != n - 1 || d.len() != n {
        return Err(OptimError::InvalidParameter {
            name: "tridiagonal",
            reason: "inconsistent array lengths",
        });
    }

    // Forward sweep
    let mut c_star = Vec::with_capacity(n);
    let mut d_star = Vec::with_capacity(n);

    if b[0] == T::zero() {
        return Err(OptimError::InvalidParameter {
            name: "b",
            reason: "zero pivot encountered (singular system)",
        });
    }

    c_star.push(c[0] / b[0]);
    d_star.push(d[0] / b[0]);

    for i in 1..n {
        let ci_prev = if i - 1 < c.len() {
            c_star[i - 1]
        } else {
            T::zero()
        };
        let denom = b[i] - a[i - 1] * ci_prev;
        if denom.abs() < T::epsilon() * T::from_f64(100.0) {
            return Err(OptimError::InvalidParameter {
                name: "b",
                reason: "zero pivot encountered (singular system)",
            });
        }
        if i < n - 1 {
            c_star.push(c[i] / denom);
        }
        d_star.push((d[i] - a[i - 1] * d_star[i - 1]) / denom);
    }

    // Back substitution
    let mut x = vec![T::zero(); n];
    x[n - 1] = d_star[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = d_star[i] - c_star[i] * x[i + 1];
    }

    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thomas_3x3() {
        // System:
        // [2  1  0] [x0]   [1]
        // [1  3  1] [x1] = [2]
        // [0  1  2] [x2]   [3]
        let a = vec![1.0, 1.0];
        let b = vec![2.0, 3.0, 2.0];
        let c = vec![1.0, 1.0];
        let d = vec![1.0, 2.0, 3.0];

        let x = thomas_solve(&a, &b, &c, &d).unwrap();
        assert_eq!(x.len(), 3);

        // Verify A*x = d
        let r0 = 2.0 * x[0] + 1.0 * x[1];
        let r1 = 1.0 * x[0] + 3.0 * x[1] + 1.0 * x[2];
        let r2 = 1.0 * x[1] + 2.0 * x[2];
        assert!((r0 - 1.0).abs() < 1e-12);
        assert!((r1 - 2.0).abs() < 1e-12);
        assert!((r2 - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_thomas_diagonal() {
        // Pure diagonal system: each x_i = d_i / b_i
        let a: Vec<f64> = vec![0.0, 0.0];
        let b = vec![2.0, 3.0, 4.0];
        let c = vec![0.0, 0.0];
        let d = vec![4.0, 9.0, 16.0];

        let x = thomas_solve(&a, &b, &c, &d).unwrap();
        assert!((x[0] - 2.0).abs() < 1e-12);
        assert!((x[1] - 3.0).abs() < 1e-12);
        assert!((x[2] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn test_thomas_singular() {
        let a = vec![0.0];
        let b = vec![0.0, 1.0];
        let c = vec![1.0];
        let d = vec![1.0, 1.0];

        assert!(thomas_solve(&a, &b, &c, &d).is_err());
    }
}
