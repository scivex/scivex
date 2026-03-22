//! Iterative solvers for sparse linear systems `Ax = b`.
//!
//! Provides Conjugate Gradient (CG) for symmetric positive-definite systems,
//! BiCGSTAB for general non-symmetric systems, and a Jacobi (diagonal)
//! preconditioner for accelerating convergence.

use scivex_core::Float;
use scivex_core::linalg::CsrMatrix;
use scivex_core::tensor::Tensor;

use crate::error::{OptimError, Result};

// ======================================================================
// Result type
// ======================================================================

/// Result of an iterative sparse solve.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct SparseSolveResult<T: Float> {
    /// Solution vector.
    pub x: Vec<T>,
    /// Number of iterations used.
    pub iterations: usize,
    /// Final residual norm.
    pub residual_norm: T,
    /// Whether the solver converged within tolerance.
    pub converged: bool,
}

// ======================================================================
// Vector helper functions
// ======================================================================

/// Dot product of two slices.
#[inline]
fn vec_dot<T: Float>(a: &[T], b: &[T]) -> T {
    a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * bi).sum()
}

/// Euclidean norm of a slice.
#[inline]
fn vec_norm<T: Float>(v: &[T]) -> T {
    vec_dot(v, v).sqrt()
}

/// y = a*x + y  (axpy in-place).
#[inline]
fn vec_axpy<T: Float>(alpha: T, x: &[T], y: &mut [T]) {
    for (yi, &xi) in y.iter_mut().zip(x.iter()) {
        *yi += alpha * xi;
    }
}

/// Compute r = b - A*x using the sparse matvec.
fn compute_residual<T: Float>(a: &CsrMatrix<T>, x: &[T], b: &[T]) -> Result<Vec<T>> {
    let x_tensor = Tensor::from_vec(x.to_vec(), vec![x.len()])?;
    let ax = a.matvec(&x_tensor)?;
    let ax_slice = ax.as_slice();
    let r: Vec<T> = b
        .iter()
        .zip(ax_slice.iter())
        .map(|(&bi, &ai)| bi - ai)
        .collect();
    Ok(r)
}

/// Sparse matrix-vector multiply returning a Vec.
fn sparse_matvec<T: Float>(a: &CsrMatrix<T>, x: &[T]) -> Result<Vec<T>> {
    let x_tensor = Tensor::from_vec(x.to_vec(), vec![x.len()])?;
    let result = a.matvec(&x_tensor)?;
    Ok(result.as_slice().to_vec())
}

// ======================================================================
// Conjugate Gradient
// ======================================================================

/// Solve `Ax = b` using the Conjugate Gradient method.
///
/// `A` must be symmetric positive-definite. The method iterates until the
/// relative residual norm `||r|| / ||b||` drops below `tol`, or `max_iter`
/// iterations are reached.
///
/// # Arguments
///
/// * `a` — Sparse coefficient matrix (must be square and SPD).
/// * `b` — Right-hand side vector.
/// * `x0` — Optional initial guess; zeros if `None`.
/// * `max_iter` — Maximum number of iterations.
/// * `tol` — Convergence tolerance on relative residual.
///
/// # Errors
///
/// Returns [`OptimError::InvalidParameter`] if dimensions are inconsistent,
/// and [`OptimError::ConvergenceFailure`] if `max_iter` is exhausted without
/// converging (the partial result is still returned inside the error via
/// [`SparseSolveResult::converged`] = `false` in the `Ok` path).
pub fn conjugate_gradient<T: Float>(
    a: &CsrMatrix<T>,
    b: &[T],
    x0: Option<&[T]>,
    max_iter: usize,
    tol: T,
) -> Result<SparseSolveResult<T>> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(OptimError::InvalidParameter {
            name: "a",
            reason: "matrix must be square",
        });
    }
    if b.len() != n {
        return Err(OptimError::InvalidParameter {
            name: "b",
            reason: "length must match matrix dimension",
        });
    }
    if x0.is_some_and(|x0v| x0v.len() != n) {
        return Err(OptimError::InvalidParameter {
            name: "x0",
            reason: "length must match matrix dimension",
        });
    }

    let b_norm = vec_norm(b);

    // x = x0 or zeros
    let mut x = x0.map_or_else(|| vec![T::zero(); n], <[T]>::to_vec);

    // r = b - A*x
    let mut r = compute_residual(a, &x, b)?;
    // p = r
    let mut p = r.clone();
    // rr = r · r
    let mut rr = vec_dot(&r, &r);

    let threshold = tol * b_norm;

    for k in 0..max_iter {
        let r_norm = rr.sqrt();
        if r_norm < threshold || (b_norm == T::zero() && r_norm < tol) {
            return Ok(SparseSolveResult {
                x,
                iterations: k,
                residual_norm: r_norm,
                converged: true,
            });
        }

        // Ap = A * p
        let ap = sparse_matvec(a, &p)?;
        let p_ap = vec_dot(&p, &ap);

        if p_ap == T::zero() {
            // Breakdown
            return Ok(SparseSolveResult {
                x,
                iterations: k,
                residual_norm: r_norm,
                converged: false,
            });
        }

        let alpha = rr / p_ap;

        // x = x + alpha * p
        vec_axpy(alpha, &p, &mut x);

        // r_new = r - alpha * Ap
        vec_axpy(-alpha, &ap, &mut r);

        let rr_new = vec_dot(&r, &r);
        let beta = rr_new / rr;

        // p = r + beta * p
        for (pi, &ri) in p.iter_mut().zip(r.iter()) {
            *pi = ri + beta * *pi;
        }

        rr = rr_new;
    }

    let final_norm = rr.sqrt();
    Ok(SparseSolveResult {
        x,
        iterations: max_iter,
        residual_norm: final_norm,
        converged: final_norm < threshold,
    })
}

// ======================================================================
// BiCGSTAB
// ======================================================================

/// Solve `Ax = b` using BiCGSTAB (Bi-Conjugate Gradient Stabilized).
///
/// Works for general (non-symmetric) sparse systems. Iterates until the
/// relative residual drops below `tol` or `max_iter` iterations are reached.
///
/// # Arguments
///
/// * `a` — Sparse coefficient matrix (must be square).
/// * `b` — Right-hand side vector.
/// * `x0` — Optional initial guess; zeros if `None`.
/// * `max_iter` — Maximum number of iterations.
/// * `tol` — Convergence tolerance on relative residual.
#[allow(clippy::too_many_lines)]
pub fn bicgstab<T: Float>(
    a: &CsrMatrix<T>,
    b: &[T],
    x0: Option<&[T]>,
    max_iter: usize,
    tol: T,
) -> Result<SparseSolveResult<T>> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(OptimError::InvalidParameter {
            name: "a",
            reason: "matrix must be square",
        });
    }
    if b.len() != n {
        return Err(OptimError::InvalidParameter {
            name: "b",
            reason: "length must match matrix dimension",
        });
    }
    if x0.is_some_and(|x0v| x0v.len() != n) {
        return Err(OptimError::InvalidParameter {
            name: "x0",
            reason: "length must match matrix dimension",
        });
    }

    let b_norm = vec_norm(b);
    let threshold = tol * b_norm;

    // x = x0 or zeros
    let mut x = match x0 {
        Some(v) => v.to_vec(),
        None => vec![T::zero(); n],
    };

    // r = b - A*x
    let mut r = compute_residual(a, &x, b)?;
    // r_hat = r (shadow residual, kept constant)
    let r_hat = r.clone();

    let mut rho = T::one();
    let mut alpha = T::one();
    let mut omega = T::one();

    let mut v = vec![T::zero(); n];
    let mut p = vec![T::zero(); n];

    for k in 0..max_iter {
        let r_norm = vec_norm(&r);
        if r_norm < threshold || (b_norm == T::zero() && r_norm < tol) {
            return Ok(SparseSolveResult {
                x,
                iterations: k,
                residual_norm: r_norm,
                converged: true,
            });
        }

        let rho_new = vec_dot(&r_hat, &r);

        if rho_new == T::zero() {
            // Breakdown
            return Ok(SparseSolveResult {
                x,
                iterations: k,
                residual_norm: r_norm,
                converged: false,
            });
        }

        let beta = (rho_new / rho) * (alpha / omega);

        // p = r + beta * (p - omega * v)
        for ((pi, &ri), &vi) in p.iter_mut().zip(r.iter()).zip(v.iter()) {
            *pi = ri + beta * (*pi - omega * vi);
        }

        // v = A * p
        v = sparse_matvec(a, &p)?;

        let r_hat_v = vec_dot(&r_hat, &v);
        if r_hat_v == T::zero() {
            return Ok(SparseSolveResult {
                x,
                iterations: k,
                residual_norm: r_norm,
                converged: false,
            });
        }
        alpha = rho_new / r_hat_v;

        // s = r - alpha * v
        let mut s = r.clone();
        vec_axpy(-alpha, &v, &mut s);

        let s_norm = vec_norm(&s);
        if s_norm < threshold {
            // x = x + alpha * p
            vec_axpy(alpha, &p, &mut x);
            return Ok(SparseSolveResult {
                x,
                iterations: k + 1,
                residual_norm: s_norm,
                converged: true,
            });
        }

        // t = A * s
        let t = sparse_matvec(a, &s)?;

        let t_t = vec_dot(&t, &t);
        omega = if t_t == T::zero() {
            T::zero()
        } else {
            vec_dot(&t, &s) / t_t
        };

        // x = x + alpha * p + omega * s
        vec_axpy(alpha, &p, &mut x);
        vec_axpy(omega, &s, &mut x);

        // r = s - omega * t
        r = s;
        vec_axpy(-omega, &t, &mut r);

        rho = rho_new;

        if omega == T::zero() {
            // Breakdown
            let final_norm = vec_norm(&r);
            return Ok(SparseSolveResult {
                x,
                iterations: k + 1,
                residual_norm: final_norm,
                converged: false,
            });
        }
    }

    let final_norm = vec_norm(&r);
    Ok(SparseSolveResult {
        x,
        iterations: max_iter,
        residual_norm: final_norm,
        converged: final_norm < threshold,
    })
}

// ======================================================================
// Jacobi Preconditioner
// ======================================================================

/// Jacobi (diagonal) preconditioner for sparse iterative solvers.
///
/// Stores the reciprocal of each diagonal element of the matrix. When a
/// diagonal entry is zero, the corresponding inverse is set to one (identity
/// fallback) to avoid division by zero.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct JacobiPreconditioner<T: Float> {
    inv_diag: Vec<T>,
}

impl<T: Float> JacobiPreconditioner<T> {
    /// Create from a sparse matrix (extracts and inverts diagonal).
    ///
    /// Zero diagonal entries are treated as ones to prevent division by zero.
    pub fn new(a: &CsrMatrix<T>) -> Self {
        let n = a.nrows().min(a.ncols());
        let mut inv_diag = Vec::with_capacity(n);
        for i in 0..n {
            let d = a.get(i, i).copied().unwrap_or(T::zero());
            if d == T::zero() {
                inv_diag.push(T::one());
            } else {
                inv_diag.push(T::one() / d);
            }
        }
        Self { inv_diag }
    }

    /// Apply preconditioner: `z = M^{-1} r` (element-wise multiply by
    /// inverse diagonal).
    pub fn apply(&self, r: &[T]) -> Vec<T> {
        r.iter()
            .zip(self.inv_diag.iter())
            .map(|(&ri, &di)| ri * di)
            .collect()
    }
}

// ======================================================================
// Preconditioned Conjugate Gradient
// ======================================================================

/// Solve `Ax = b` using the Preconditioned Conjugate Gradient method.
///
/// Same algorithm as [`conjugate_gradient`] but applies the preconditioner
/// `M^{-1}` to the residual at each step, which can significantly improve
/// convergence for ill-conditioned systems.
pub fn preconditioned_cg<T: Float>(
    a: &CsrMatrix<T>,
    b: &[T],
    preconditioner: &JacobiPreconditioner<T>,
    x0: Option<&[T]>,
    max_iter: usize,
    tol: T,
) -> Result<SparseSolveResult<T>> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(OptimError::InvalidParameter {
            name: "a",
            reason: "matrix must be square",
        });
    }
    if b.len() != n {
        return Err(OptimError::InvalidParameter {
            name: "b",
            reason: "length must match matrix dimension",
        });
    }
    if x0.is_some_and(|x0v| x0v.len() != n) {
        return Err(OptimError::InvalidParameter {
            name: "x0",
            reason: "length must match matrix dimension",
        });
    }

    let b_norm = vec_norm(b);
    let threshold = tol * b_norm;

    let mut x = match x0 {
        Some(v) => v.to_vec(),
        None => vec![T::zero(); n],
    };

    // r = b - A*x
    let mut r = compute_residual(a, &x, b)?;
    // z = M^{-1} r
    let mut z = preconditioner.apply(&r);
    // p = z
    let mut p = z.clone();
    // rz = r · z
    let mut rz = vec_dot(&r, &z);

    for k in 0..max_iter {
        let r_norm = vec_norm(&r);
        if r_norm < threshold || (b_norm == T::zero() && r_norm < tol) {
            return Ok(SparseSolveResult {
                x,
                iterations: k,
                residual_norm: r_norm,
                converged: true,
            });
        }

        // Ap = A * p
        let ap = sparse_matvec(a, &p)?;
        let p_ap = vec_dot(&p, &ap);

        if p_ap == T::zero() {
            return Ok(SparseSolveResult {
                x,
                iterations: k,
                residual_norm: r_norm,
                converged: false,
            });
        }

        let alpha = rz / p_ap;

        // x = x + alpha * p
        vec_axpy(alpha, &p, &mut x);

        // r = r - alpha * Ap
        vec_axpy(-alpha, &ap, &mut r);

        // z = M^{-1} r
        z = preconditioner.apply(&r);

        let rz_new = vec_dot(&r, &z);
        let beta = rz_new / rz;

        // p = z + beta * p
        for (pi, &zi) in p.iter_mut().zip(z.iter()) {
            *pi = zi + beta * *pi;
        }

        rz = rz_new;
    }

    let final_norm = vec_norm(&r);
    Ok(SparseSolveResult {
        x,
        iterations: max_iter,
        residual_norm: final_norm,
        converged: final_norm < threshold,
    })
}

// ======================================================================
// Tests
// ======================================================================

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use scivex_core::linalg::CsrMatrix;

    /// Helper: build a 3x3 SPD matrix.
    /// A = [[4, 1, 0],
    ///      [1, 3, 1],
    ///      [0, 1, 4]]
    fn spd_3x3() -> CsrMatrix<f64> {
        CsrMatrix::from_triplets(
            3,
            3,
            vec![0, 0, 1, 1, 1, 2, 2],
            vec![0, 1, 0, 1, 2, 1, 2],
            vec![4.0, 1.0, 1.0, 3.0, 1.0, 1.0, 4.0],
        )
        .unwrap()
    }

    #[test]
    fn test_cg_simple_3x3() {
        let a = spd_3x3();
        // b = A * [1, 2, 3] = [4+2, 1+6+3, 2+12] = [6, 10, 14]
        let b = [6.0, 10.0, 14.0];
        let result = conjugate_gradient(&a, &b, None, 100, 1e-10).unwrap();
        assert!(result.converged);
        assert!((result.x[0] - 1.0).abs() < 1e-8);
        assert!((result.x[1] - 2.0).abs() < 1e-8);
        assert!((result.x[2] - 3.0).abs() < 1e-8);
    }

    #[test]
    fn test_cg_diagonal_system() {
        // Diagonal matrix — CG should converge in 1 iteration for each
        // distinct eigenvalue. With n distinct values it takes at most n.
        let n = 10;
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        let mut b = vec![0.0; n];
        let expected: Vec<f64> = (1..=n).map(|i| i as f64).collect();

        for i in 0..n {
            rows.push(i);
            cols.push(i);
            vals.push((i + 1) as f64); // diag = 1,2,...,n
            b[i] = (i + 1) as f64 * expected[i]; // b = diag * x
        }

        let a = CsrMatrix::from_triplets(n, n, rows, cols, vals).unwrap();
        let result = conjugate_gradient(&a, &b, None, 100, 1e-12).unwrap();
        assert!(result.converged);
        for (i, (xi, ei)) in result.x.iter().zip(expected.iter()).enumerate() {
            assert!((*xi - *ei).abs() < 1e-8, "x[{i}] = {xi}, expected {ei}",);
        }
    }

    #[test]
    fn test_bicgstab_nonsymmetric() {
        // Non-symmetric:
        // A = [[3, 1, 0],
        //      [0, 4, 2],
        //      [1, 0, 5]]
        let a = CsrMatrix::from_triplets(
            3,
            3,
            vec![0, 0, 1, 1, 2, 2],
            vec![0, 1, 1, 2, 0, 2],
            vec![3.0, 1.0, 4.0, 2.0, 1.0, 5.0],
        )
        .unwrap();
        // x = [1, 2, 3]: b = [3+2, 8+6, 1+15] = [5, 14, 16]
        let b = [5.0, 14.0, 16.0];
        let result = bicgstab(&a, &b, None, 100, 1e-10).unwrap();
        assert!(result.converged, "BiCGSTAB did not converge");
        assert!((result.x[0] - 1.0).abs() < 1e-6);
        assert!((result.x[1] - 2.0).abs() < 1e-6);
        assert!((result.x[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_bicgstab_on_spd() {
        // BiCGSTAB should also work on SPD systems.
        let a = spd_3x3();
        let b = [6.0, 10.0, 14.0];
        let result = bicgstab(&a, &b, None, 100, 1e-10).unwrap();
        assert!(result.converged);
        assert!((result.x[0] - 1.0).abs() < 1e-6);
        assert!((result.x[1] - 2.0).abs() < 1e-6);
        assert!((result.x[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_jacobi_preconditioner() {
        let a = spd_3x3();
        let prec = JacobiPreconditioner::new(&a);
        // Diagonal of A is [4, 3, 4], so inv_diag = [0.25, 1/3, 0.25]
        assert!((prec.inv_diag[0] - 0.25).abs() < 1e-15);
        assert!((prec.inv_diag[1] - 1.0 / 3.0).abs() < 1e-15);
        assert!((prec.inv_diag[2] - 0.25).abs() < 1e-15);

        let r = [4.0, 3.0, 8.0];
        let z = prec.apply(&r);
        assert!((z[0] - 1.0).abs() < 1e-15);
        assert!((z[1] - 1.0).abs() < 1e-15);
        assert!((z[2] - 2.0).abs() < 1e-15);
    }

    #[test]
    fn test_preconditioned_cg_converges() {
        // Use a larger ill-conditioned diagonal system where
        // preconditioning should help.
        let n = 20;
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        let mut b = vec![0.0; n];

        for (i, bi) in b.iter_mut().enumerate() {
            rows.push(i);
            cols.push(i);
            // Condition number ~ n^2
            let d = ((i + 1) * (i + 1)) as f64;
            vals.push(d);
            *bi = d; // x = [1, 1, ..., 1]
        }

        let a = CsrMatrix::from_triplets(n, n, rows, cols, vals).unwrap();
        let prec = JacobiPreconditioner::new(&a);

        let result_pcg = preconditioned_cg(&a, &b, &prec, None, 100, 1e-10).unwrap();
        let result_cg = conjugate_gradient(&a, &b, None, 100, 1e-10).unwrap();

        assert!(result_pcg.converged);
        assert!(result_cg.converged);
        // Preconditioned CG should converge in fewer (or equal) iterations.
        // For a diagonal system, Jacobi preconditioner makes it trivial (1 iter).
        assert!(
            result_pcg.iterations <= result_cg.iterations,
            "PCG iters {} > CG iters {}",
            result_pcg.iterations,
            result_cg.iterations
        );
    }

    #[test]
    fn test_cg_custom_initial_guess() {
        let a = spd_3x3();
        let b = [6.0, 10.0, 14.0];
        // Start close to the solution
        let x0 = [0.9, 2.1, 2.9];
        let result = conjugate_gradient(&a, &b, Some(&x0), 100, 1e-10).unwrap();
        assert!(result.converged);
        assert!((result.x[0] - 1.0).abs() < 1e-8);
        assert!((result.x[1] - 2.0).abs() < 1e-8);
        assert!((result.x[2] - 3.0).abs() < 1e-8);
    }

    #[test]
    fn test_convergence_failure_max_iter_too_small() {
        let a = spd_3x3();
        // Use a b that requires more than 0 iterations
        let b = [6.0, 10.0, 14.0];
        let result = conjugate_gradient(&a, &b, None, 0, 1e-10).unwrap();
        assert!(!result.converged);
        assert_eq!(result.iterations, 0);
    }

    #[test]
    fn test_dimension_mismatch_errors() {
        let a = spd_3x3();

        // b too short
        let b_short = [1.0, 2.0];
        let err = conjugate_gradient(&a, &b_short, None, 10, 1e-10);
        assert!(err.is_err());

        // x0 wrong length
        let b = [6.0, 10.0, 14.0];
        let x0_bad = [1.0, 2.0];
        let err = conjugate_gradient(&a, &b, Some(&x0_bad), 10, 1e-10);
        assert!(err.is_err());

        // Non-square matrix
        let rect = CsrMatrix::from_triplets(2, 3, vec![0, 1], vec![0, 1], vec![1.0, 2.0]).unwrap();
        let err = conjugate_gradient(&rect, &[1.0, 2.0], None, 10, 1e-10);
        assert!(err.is_err());

        // Same checks for bicgstab
        let err = bicgstab(&a, &b_short, None, 10, 1e-10);
        assert!(err.is_err());
    }

    #[test]
    fn test_bicgstab_tridiagonal() {
        // Tridiagonal (non-symmetric):
        // A = [[2, -1, 0, 0],
        //      [0,  2, -1, 0],
        //      [0,  0,  2, -1],
        //      [0,  0,  0,  2]]
        // Upper triangular — clearly non-symmetric.
        // x = [1, 1, 1, 1]
        // b = A*x = [1, 1, 1, 2]
        let a = CsrMatrix::from_triplets(
            4,
            4,
            vec![0, 0, 1, 1, 2, 2, 3],
            vec![0, 1, 1, 2, 2, 3, 3],
            vec![2.0, -1.0, 2.0, -1.0, 2.0, -1.0, 2.0],
        )
        .unwrap();
        let b = [1.0, 1.0, 1.0, 2.0];
        let result = bicgstab(&a, &b, None, 100, 1e-10).unwrap();
        assert!(
            result.converged,
            "BiCGSTAB did not converge on tridiagonal system"
        );
        for i in 0..4 {
            assert!(
                (result.x[i] - 1.0).abs() < 1e-6,
                "x[{i}] = {}, expected 1.0",
                result.x[i]
            );
        }
    }
}
