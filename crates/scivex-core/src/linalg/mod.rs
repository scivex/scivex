//! Linear algebra operations.
//!
//! All routines are implemented from scratch — no external BLAS/LAPACK
//! bindings. The API is split into three BLAS levels plus matrix
//! decompositions:
//!
//! | Level | Operations | Complexity |
//! |-------|-----------|------------|
//! | L1 | `dot`, `axpy`, `nrm2`, `asum`, `scal`, `iamax` | O(n) |
//! | L2 | `gemv` (matrix-vector multiply) | O(n^2) |
//! | L3 | `gemm` (matrix-matrix multiply) | O(n^3) |
//!
//! Decompositions: [`LuDecomposition`], [`QrDecomposition`],
//! [`CholeskyDecomposition`], [`SvdDecomposition`], [`EigDecomposition`]

pub mod blas;
pub mod decomp;
pub mod sparse;

pub use blas::{asum, axpy, dot, gemm, gemv, iamax, nrm2, scal};
pub use decomp::CholeskyDecomposition;
pub use decomp::EigDecomposition;
pub use decomp::LuDecomposition;
pub use decomp::QrDecomposition;
pub use decomp::SvdDecomposition;
pub use decomp::lstsq;
pub use sparse::{CooMatrix, CscMatrix, CsrMatrix};

use crate::Float;
use crate::error::Result;
use crate::tensor::Tensor;

/// Solve the linear system `Ax = b` for a square matrix `A`.
///
/// Uses LU decomposition with partial pivoting internally.
///
/// ```
/// # use scivex_core::tensor::Tensor;
/// # use scivex_core::linalg;
/// let a = Tensor::from_vec(vec![2.0_f64, 1.0, 1.0, 4.0], vec![2, 2]).unwrap();
/// let b = Tensor::from_vec(vec![5.0_f64, 6.0], vec![2]).unwrap();
/// let x = linalg::solve(&a, &b).unwrap();
/// assert!((x.as_slice()[0] - 2.0).abs() < 1e-10);
/// assert!((x.as_slice()[1] - 1.0).abs() < 1e-10);
/// ```
pub fn solve<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>> {
    LuDecomposition::decompose(a)?.solve(b)
}

/// Compute the inverse of a square matrix.
///
/// Uses LU decomposition with partial pivoting internally.
/// Returns [`CoreError::SingularMatrix`](crate::CoreError::SingularMatrix) if the
/// matrix is singular.
///
/// ```
/// # use scivex_core::tensor::Tensor;
/// # use scivex_core::linalg;
/// let a = Tensor::from_vec(vec![2.0_f64, 1.0, 1.0, 4.0], vec![2, 2]).unwrap();
/// let inv = linalg::inv(&a).unwrap();
/// // A * A^-1 ≈ I
/// let eye = a.matmul(&inv).unwrap();
/// assert!((eye.as_slice()[0] - 1.0).abs() < 1e-10);
/// ```
pub fn inv<T: Float>(a: &Tensor<T>) -> Result<Tensor<T>> {
    LuDecomposition::decompose(a)?.inverse()
}

/// Compute the determinant of a square matrix.
///
/// Uses LU decomposition with partial pivoting internally.
///
/// ```
/// # use scivex_core::tensor::Tensor;
/// # use scivex_core::linalg;
/// let a = Tensor::from_vec(vec![2.0_f64, 1.0, 1.0, 4.0], vec![2, 2]).unwrap();
/// let det = linalg::det(&a).unwrap();
/// assert!((det - 7.0).abs() < 1e-10);
/// ```
pub fn det<T: Float>(a: &Tensor<T>) -> Result<T> {
    Ok(LuDecomposition::decompose(a)?.det())
}
