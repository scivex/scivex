//! Cholesky decomposition for symmetric positive-definite matrices.
//!
//! Decomposes a symmetric positive-definite matrix `A` into `A = L L^T`
//! where `L` is lower triangular with positive diagonal entries.

use crate::Float;
use crate::error::{CoreError, Result};
use crate::tensor::Tensor;

/// Result of a Cholesky decomposition.
///
/// Stores the factorization `A = L L^T` where `L` is lower triangular.
#[derive(Debug, Clone)]
pub struct CholeskyDecomposition<T: Float> {
    /// Lower triangular factor stored as a flat n x n array.
    l_data: Vec<T>,
    /// Matrix dimension.
    n: usize,
}

#[allow(clippy::many_single_char_names)]
impl<T: Float> CholeskyDecomposition<T> {
    /// Compute the Cholesky decomposition of a symmetric positive-definite matrix.
    ///
    /// Returns `A = L L^T` where `L` is lower triangular.
    ///
    /// ```
    /// # use scivex_core::tensor::Tensor;
    /// # use scivex_core::linalg::decomp::CholeskyDecomposition;
    /// let a = Tensor::from_vec(vec![4.0_f64, 2.0, 2.0, 3.0], vec![2, 2]).unwrap();
    /// let chol = CholeskyDecomposition::decompose(&a).unwrap();
    /// let l = chol.l();
    /// // Verify L L^T = A
    /// let lt = l.transpose().unwrap();
    /// let prod = l.matmul(&lt).unwrap();
    /// assert!((prod.as_slice()[0] - 4.0).abs() < 1e-10);
    /// ```
    pub fn decompose(a: &Tensor<T>) -> Result<Self> {
        if a.ndim() != 2 {
            return Err(CoreError::InvalidArgument {
                reason: "Cholesky decomposition requires a 2-D tensor (matrix)",
            });
        }
        let n = a.shape()[0];
        if a.shape()[1] != n {
            return Err(CoreError::InvalidArgument {
                reason: "Cholesky decomposition requires a square matrix",
            });
        }

        let a_data = a.as_slice();
        let mut l = vec![T::zero(); n * n];

        for j in 0..n {
            // Diagonal element
            let mut sum = a_data[j * n + j];
            for k in 0..j {
                sum -= l[j * n + k] * l[j * n + k];
            }
            if sum <= T::zero() {
                return Err(CoreError::InvalidArgument {
                    reason: "Cholesky: matrix is not positive definite",
                });
            }
            l[j * n + j] = sum.sqrt();

            // Off-diagonal elements
            let diag = l[j * n + j];
            for i in (j + 1)..n {
                let mut sum = a_data[i * n + j];
                for k in 0..j {
                    sum -= l[i * n + k] * l[j * n + k];
                }
                l[i * n + j] = sum / diag;
            }
        }

        Ok(Self { l_data: l, n })
    }

    /// Extract the lower triangular factor `L`.
    pub fn l(&self) -> Tensor<T> {
        Tensor::from_vec(self.l_data.clone(), vec![self.n, self.n]).unwrap()
    }

    /// Solve the linear system `Ax = b` using the Cholesky factorization.
    ///
    /// Since `A = L L^T`, solves `L y = b` (forward) then `L^T x = y` (backward).
    pub fn solve(&self, b: &Tensor<T>) -> Result<Tensor<T>> {
        if b.ndim() != 1 {
            return Err(CoreError::InvalidArgument {
                reason: "Cholesky solve: `b` must be a 1-D tensor",
            });
        }
        if b.numel() != self.n {
            return Err(CoreError::DimensionMismatch {
                expected: vec![self.n],
                got: b.shape().to_vec(),
            });
        }

        let n = self.n;
        let mut x: Vec<T> = b.as_slice().to_vec();

        // Forward substitution: L y = b
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            for j in 0..i {
                let l_xj = self.l_data[i * n + j] * x[j];
                x[i] -= l_xj;
            }
            x[i] /= self.l_data[i * n + i];
        }

        // Back substitution: L^T x = y
        #[allow(clippy::needless_range_loop)]
        for i in (0..n).rev() {
            for j in (i + 1)..n {
                let lt_xj = self.l_data[j * n + i] * x[j];
                x[i] -= lt_xj;
            }
            x[i] /= self.l_data[i * n + i];
        }

        Tensor::from_vec(x, vec![n])
    }

    /// Compute the inverse using the Cholesky factorization.
    pub fn inverse(&self) -> Result<Tensor<T>> {
        let n = self.n;
        let mut inv_data = vec![T::zero(); n * n];

        for col in 0..n {
            let mut e = vec![T::zero(); n];
            e[col] = T::one();
            let e_tensor = Tensor::from_vec(e, vec![n])?;
            let x = self.solve(&e_tensor)?;
            let x_data = x.as_slice();
            for row in 0..n {
                inv_data[row * n + col] = x_data[row];
            }
        }

        Tensor::from_vec(inv_data, vec![n, n])
    }

    /// Compute the log-determinant (useful for avoiding overflow).
    ///
    /// `log(det(A)) = 2 * sum(log(diag(L)))`
    pub fn log_det(&self) -> T {
        let n = self.n;
        let mut sum = T::zero();
        for i in 0..n {
            sum += self.l_data[i * n + i].ln();
        }
        sum + sum // 2 * sum
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    fn approx_eq(a: &[f64], b: &[f64], tol: f64) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(&x, &y)| (x - y).abs() < tol)
    }

    fn sym_pd(data: &[f64], n: usize) -> Tensor<f64> {
        // Build A^T A + I to guarantee symmetric positive definite
        let a = Tensor::from_vec(data.to_vec(), vec![n, n]).unwrap();
        let at = a.transpose().unwrap();
        let ata = at.matmul(&a).unwrap();
        let eye = Tensor::<f64>::eye(n);
        // ata += eye
        let ata_s = ata.as_slice().to_vec();
        let eye_s = eye.as_slice();
        let sum: Vec<f64> = ata_s.iter().zip(eye_s).map(|(a, b)| a + b).collect();
        Tensor::from_vec(sum, vec![n, n]).unwrap()
    }

    #[test]
    fn test_cholesky_2x2() {
        let a = Tensor::from_vec(vec![4.0, 2.0, 2.0, 3.0], vec![2, 2]).unwrap();
        let chol = CholeskyDecomposition::decompose(&a).unwrap();
        let l = chol.l();
        let lt = l.transpose().unwrap();
        let prod = l.matmul(&lt).unwrap();
        assert!(approx_eq(prod.as_slice(), a.as_slice(), 1e-12));
    }

    #[test]
    fn test_cholesky_3x3() {
        // A = [[25,15,-5],[15,18,0],[-5,0,11]]
        let a = Tensor::from_vec(
            vec![25.0, 15.0, -5.0, 15.0, 18.0, 0.0, -5.0, 0.0, 11.0],
            vec![3, 3],
        )
        .unwrap();
        let chol = CholeskyDecomposition::decompose(&a).unwrap();
        let l = chol.l();
        let lt = l.transpose().unwrap();
        let prod = l.matmul(&lt).unwrap();
        assert!(approx_eq(prod.as_slice(), a.as_slice(), 1e-10));
    }

    #[test]
    fn test_cholesky_identity() {
        let eye = Tensor::<f64>::eye(4);
        let chol = CholeskyDecomposition::decompose(&eye).unwrap();
        let l = chol.l();
        assert!(approx_eq(l.as_slice(), eye.as_slice(), 1e-14));
    }

    #[test]
    fn test_cholesky_solve() {
        let a = Tensor::from_vec(vec![4.0, 2.0, 2.0, 3.0], vec![2, 2]).unwrap();
        let b = Tensor::from_vec(vec![1.0, 2.0], vec![2]).unwrap();
        let chol = CholeskyDecomposition::decompose(&a).unwrap();
        let x = chol.solve(&b).unwrap();
        // Verify Ax = b
        let ax = a.matvec(&x).unwrap();
        assert!(approx_eq(ax.as_slice(), b.as_slice(), 1e-12));
    }

    #[test]
    fn test_cholesky_solve_3x3() {
        let a = Tensor::from_vec(
            vec![25.0, 15.0, -5.0, 15.0, 18.0, 0.0, -5.0, 0.0, 11.0],
            vec![3, 3],
        )
        .unwrap();
        let b = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let chol = CholeskyDecomposition::decompose(&a).unwrap();
        let x = chol.solve(&b).unwrap();
        let ax = a.matvec(&x).unwrap();
        assert!(approx_eq(ax.as_slice(), b.as_slice(), 1e-10));
    }

    #[test]
    fn test_cholesky_inverse() {
        let a = Tensor::from_vec(vec![4.0, 2.0, 2.0, 3.0], vec![2, 2]).unwrap();
        let chol = CholeskyDecomposition::decompose(&a).unwrap();
        let inv = chol.inverse().unwrap();
        let eye = a.matmul(&inv).unwrap();
        let identity = Tensor::<f64>::eye(2);
        assert!(approx_eq(eye.as_slice(), identity.as_slice(), 1e-12));
    }

    #[test]
    fn test_cholesky_log_det() {
        let a = Tensor::from_vec(vec![4.0, 2.0, 2.0, 3.0], vec![2, 2]).unwrap();
        let chol = CholeskyDecomposition::decompose(&a).unwrap();
        // det(A) = 4*3 - 2*2 = 8
        let log_det = chol.log_det();
        assert!((log_det - 8.0_f64.ln()).abs() < 1e-12);
    }

    #[test]
    fn test_cholesky_not_pd() {
        // Negative eigenvalue
        let a = Tensor::from_vec(vec![1.0, 2.0, 2.0, 1.0], vec![2, 2]).unwrap();
        assert!(CholeskyDecomposition::decompose(&a).is_err());
    }

    #[test]
    fn test_cholesky_not_square() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        assert!(CholeskyDecomposition::decompose(&a).is_err());
    }

    #[test]
    fn test_cholesky_generated_spd() {
        let spd = sym_pd(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 3);
        let chol = CholeskyDecomposition::decompose(&spd).unwrap();
        let l = chol.l();
        let lt = l.transpose().unwrap();
        let prod = l.matmul(&lt).unwrap();
        assert!(approx_eq(prod.as_slice(), spd.as_slice(), 1e-10));
    }
}
