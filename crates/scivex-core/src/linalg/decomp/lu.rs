//! LU decomposition with partial pivoting.
//!
//! Decomposes a square matrix `A` into `PA = LU` where:
//! - `P` is a permutation matrix (stored as a pivot vector)
//! - `L` is lower triangular with unit diagonal
//! - `U` is upper triangular

use crate::Float;
use crate::error::{CoreError, Result};
use crate::tensor::Tensor;

/// Result of an LU decomposition with partial pivoting.
///
/// Stores the factorization `PA = LU` in compact form: `L` and `U` are
/// packed into a single matrix (the unit diagonal of `L` is implicit),
/// and the permutation is stored as a pivot index vector.
#[derive(Debug, Clone)]
pub struct LuDecomposition<T: Float> {
    /// Packed LU matrix: lower triangle holds L (without diagonal),
    /// upper triangle (including diagonal) holds U.
    lu: Vec<T>,
    /// Pivot indices: row `i` was swapped with row `pivots[i]`.
    pivots: Vec<usize>,
    /// Matrix dimension (n x n).
    n: usize,
    /// Sign of the permutation (+1 or -1), for determinant computation.
    sign: T,
}

impl<T: Float> LuDecomposition<T> {
    /// Perform LU decomposition with partial pivoting on a square matrix.
    ///
    /// ```
    /// # use scivex_core::tensor::Tensor;
    /// # use scivex_core::linalg::decomp::LuDecomposition;
    /// let a = Tensor::from_vec(vec![2.0_f64, 1.0, 1.0, 4.0], vec![2, 2]).unwrap();
    /// let lu = LuDecomposition::decompose(&a).unwrap();
    /// let det = lu.det();
    /// assert!((det - 7.0).abs() < 1e-10);
    /// ```
    pub fn decompose(a: &Tensor<T>) -> Result<Self> {
        if a.ndim() != 2 {
            return Err(CoreError::InvalidArgument {
                reason: "LU decomposition requires a 2-D tensor (matrix)",
            });
        }
        let n = a.shape()[0];
        if a.shape()[1] != n {
            return Err(CoreError::InvalidArgument {
                reason: "LU decomposition requires a square matrix",
            });
        }

        // Copy matrix data into working buffer
        let mut lu: Vec<T> = a.as_slice().to_vec();
        let mut pivots: Vec<usize> = (0..n).collect();
        let mut sign = T::one();

        for k in 0..n {
            // Find pivot: row with largest |lu[i, k]| for i >= k
            let mut max_val = lu[k * n + k].abs();
            let mut max_row = k;
            for i in (k + 1)..n {
                let val = lu[i * n + k].abs();
                if val > max_val {
                    max_val = val;
                    max_row = i;
                }
            }

            // Swap rows if needed
            if max_row != k {
                for j in 0..n {
                    lu.swap(k * n + j, max_row * n + j);
                }
                pivots.swap(k, max_row);
                sign *= T::from_f64(-1.0);
            }

            let pivot = lu[k * n + k];
            if pivot.abs() < T::epsilon() * T::from_f64(1e3) {
                return Err(CoreError::SingularMatrix);
            }

            // Eliminate below the pivot
            for i in (k + 1)..n {
                let factor = lu[i * n + k] / pivot;
                lu[i * n + k] = factor; // Store L factor

                for j in (k + 1)..n {
                    let ukj = lu[k * n + j];
                    lu[i * n + j] -= factor * ukj;
                }
            }
        }

        Ok(Self {
            lu,
            pivots,
            n,
            sign,
        })
    }

    /// Extract the lower triangular matrix `L` (with unit diagonal).
    pub fn l(&self) -> Tensor<T> {
        let n = self.n;
        let mut data = vec![T::zero(); n * n];
        for i in 0..n {
            data[i * n + i] = T::one(); // Unit diagonal
            for j in 0..i {
                data[i * n + j] = self.lu[i * n + j];
            }
        }
        Tensor::from_vec(data, vec![n, n]).unwrap()
    }

    /// Extract the upper triangular matrix `U`.
    pub fn u(&self) -> Tensor<T> {
        let n = self.n;
        let mut data = vec![T::zero(); n * n];
        for i in 0..n {
            for j in i..n {
                data[i * n + j] = self.lu[i * n + j];
            }
        }
        Tensor::from_vec(data, vec![n, n]).unwrap()
    }

    /// Extract the permutation matrix `P`.
    pub fn p(&self) -> Tensor<T> {
        let n = self.n;
        let mut data = vec![T::zero(); n * n];
        for (i, &pi) in self.pivots.iter().enumerate() {
            data[i * n + pi] = T::one();
        }
        Tensor::from_vec(data, vec![n, n]).unwrap()
    }

    /// The permutation pivot vector.
    pub fn pivots(&self) -> &[usize] {
        &self.pivots
    }

    /// Compute the determinant from the LU factorization.
    ///
    /// `det(A) = sign * product(diag(U))`
    pub fn det(&self) -> T {
        let n = self.n;
        let mut d = self.sign;
        for i in 0..n {
            d *= self.lu[i * n + i];
        }
        d
    }

    /// Solve the linear system `Ax = b` using the precomputed LU factorization.
    ///
    /// `b` must be a 1-D tensor of length `n`.
    pub fn solve(&self, b: &Tensor<T>) -> Result<Tensor<T>> {
        if b.ndim() != 1 {
            return Err(CoreError::InvalidArgument {
                reason: "solve: `b` must be a 1-D tensor",
            });
        }
        if b.numel() != self.n {
            return Err(CoreError::DimensionMismatch {
                expected: vec![self.n],
                got: b.shape().to_vec(),
            });
        }

        let n = self.n;
        let b_data = b.as_slice();

        // Apply permutation: Pb
        let mut x: Vec<T> = vec![T::zero(); n];
        for (i, &pi) in self.pivots.iter().enumerate() {
            x[i] = b_data[pi];
        }

        // Forward substitution: Ly = Pb
        // We index x[j] while updating x[i] where j < i, so iterators
        // are safe but less clear — use index loops instead.
        #[allow(clippy::needless_range_loop)]
        for i in 1..n {
            for j in 0..i {
                let lij_xj = self.lu[i * n + j] * x[j];
                x[i] -= lij_xj;
            }
        }

        // Back substitution: Ux = y
        #[allow(clippy::needless_range_loop)]
        for i in (0..n).rev() {
            for j in (i + 1)..n {
                let uij_xj = self.lu[i * n + j] * x[j];
                x[i] -= uij_xj;
            }
            x[i] /= self.lu[i * n + i];
        }

        Tensor::from_vec(x, vec![n])
    }

    /// Compute the inverse matrix using the LU factorization.
    ///
    /// Solves `AX = I` column by column.
    pub fn inverse(&self) -> Result<Tensor<T>> {
        let n = self.n;
        let mut inv_data = vec![T::zero(); n * n];

        for col in 0..n {
            // Create unit vector e_col
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
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    fn mat(data: &[f64], n: usize) -> Tensor<f64> {
        Tensor::from_vec(data.to_vec(), vec![n, n]).unwrap()
    }

    fn approx_eq(a: &[f64], b: &[f64], tol: f64) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(&x, &y)| (x - y).abs() < tol)
    }

    #[test]
    fn test_lu_2x2() {
        let a = mat(&[2.0, 1.0, 1.0, 4.0], 2);
        let lu = LuDecomposition::decompose(&a).unwrap();

        // Verify PA = LU
        let p = lu.p();
        let l = lu.l();
        let u = lu.u();
        let pa = p.matmul(&a).unwrap();
        let lu_prod = l.matmul(&u).unwrap();
        assert!(approx_eq(pa.as_slice(), lu_prod.as_slice(), 1e-12));
    }

    #[test]
    fn test_lu_3x3() {
        let a = mat(&[2.0, 1.0, 1.0, 4.0, 3.0, 3.0, 8.0, 7.0, 9.0], 3);
        let lu = LuDecomposition::decompose(&a).unwrap();

        let p = lu.p();
        let l = lu.l();
        let u = lu.u();
        let pa = p.matmul(&a).unwrap();
        let lu_prod = l.matmul(&u).unwrap();
        assert!(approx_eq(pa.as_slice(), lu_prod.as_slice(), 1e-12));
    }

    #[test]
    fn test_lu_4x4() {
        // >>> np.linalg.det([[1,2,3,4],[5,6,7,8],[2,6,4,8],[3,1,1,2]])
        // 72.0
        let a = mat(
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 2.0, 6.0, 4.0, 8.0, 3.0, 1.0, 1.0, 2.0,
            ],
            4,
        );
        let lu = LuDecomposition::decompose(&a).unwrap();

        let p = lu.p();
        let l = lu.l();
        let u = lu.u();
        let pa = p.matmul(&a).unwrap();
        let lu_prod = l.matmul(&u).unwrap();
        assert!(approx_eq(pa.as_slice(), lu_prod.as_slice(), 1e-10));
    }

    #[test]
    fn test_det_2x2() {
        let a = mat(&[2.0, 1.0, 1.0, 4.0], 2);
        let lu = LuDecomposition::decompose(&a).unwrap();
        assert!((lu.det() - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_det_3x3() {
        // >>> np.linalg.det([[6,1,1],[4,-2,5],[2,8,7]])
        // -306.0
        let a = mat(&[6.0, 1.0, 1.0, 4.0, -2.0, 5.0, 2.0, 8.0, 7.0], 3);
        let lu = LuDecomposition::decompose(&a).unwrap();
        assert!((lu.det() - (-306.0)).abs() < 1e-10);
    }

    #[test]
    fn test_det_4x4_numpy() {
        let a = mat(
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 2.0, 6.0, 4.0, 8.0, 3.0, 1.0, 1.0, 2.0,
            ],
            4,
        );
        let lu = LuDecomposition::decompose(&a).unwrap();
        assert!((lu.det() - 72.0).abs() < 1e-10);
    }

    #[test]
    fn test_det_identity() {
        let eye = Tensor::<f64>::eye(5);
        let lu = LuDecomposition::decompose(&eye).unwrap();
        assert!((lu.det() - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_singular_matrix() {
        // Rows are linearly dependent
        let a = mat(&[1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 7.0, 8.0, 9.0], 3);
        assert!(LuDecomposition::decompose(&a).is_err());
    }

    #[test]
    fn test_solve_2x2() {
        // 2x + y = 5
        // x + 4y = 6
        // => x = 2, y = 1
        let a = mat(&[2.0, 1.0, 1.0, 4.0], 2);
        let b = Tensor::from_vec(vec![5.0, 6.0], vec![2]).unwrap();
        let lu = LuDecomposition::decompose(&a).unwrap();
        let x = lu.solve(&b).unwrap();
        assert!(approx_eq(x.as_slice(), &[2.0, 1.0], 1e-12));
    }

    #[test]
    fn test_solve_3x3() {
        // >>> A = np.array([[1,2,3],[4,5,6],[7,8,10]])
        // >>> b = np.array([1,2,3])
        // >>> np.linalg.solve(A, b)
        // array([-0.33333333,  0.66666667,  0.        ])
        let a = mat(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0], 3);
        let b = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let lu = LuDecomposition::decompose(&a).unwrap();
        let x = lu.solve(&b).unwrap();
        assert!(approx_eq(
            x.as_slice(),
            &[-1.0 / 3.0, 2.0 / 3.0, 0.0],
            1e-12
        ));
    }

    #[test]
    fn test_solve_4x4_numpy() {
        // >>> A = np.array([[1,2,3,4],[5,6,7,8],[2,6,4,8],[3,1,1,2]])
        // >>> b = np.array([10, 26, 20, 7])
        // >>> np.linalg.solve(A, b)
        // array([1., 1., 1., 1.])
        let a = mat(
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 2.0, 6.0, 4.0, 8.0, 3.0, 1.0, 1.0, 2.0,
            ],
            4,
        );
        let b = Tensor::from_vec(vec![10.0, 26.0, 20.0, 7.0], vec![4]).unwrap();
        let lu = LuDecomposition::decompose(&a).unwrap();
        let x = lu.solve(&b).unwrap();
        assert!(approx_eq(x.as_slice(), &[1.0, 1.0, 1.0, 1.0], 1e-10));
    }

    #[test]
    fn test_inverse_2x2() {
        // >>> np.linalg.inv([[2,1],[1,4]])
        // array([[ 0.57142857, -0.14285714],
        //        [-0.14285714,  0.28571429]])
        let a = mat(&[2.0, 1.0, 1.0, 4.0], 2);
        let lu = LuDecomposition::decompose(&a).unwrap();
        let inv = lu.inverse().unwrap();

        // Verify A * A^-1 = I
        let eye = a.matmul(&inv).unwrap();
        let identity = Tensor::<f64>::eye(2);
        assert!(approx_eq(eye.as_slice(), identity.as_slice(), 1e-12));
    }

    #[test]
    fn test_inverse_3x3() {
        let a = mat(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0], 3);
        let lu = LuDecomposition::decompose(&a).unwrap();
        let inv = lu.inverse().unwrap();

        let eye = a.matmul(&inv).unwrap();
        let identity = Tensor::<f64>::eye(3);
        assert!(approx_eq(eye.as_slice(), identity.as_slice(), 1e-10));
    }

    #[test]
    fn test_inverse_identity() {
        let eye = Tensor::<f64>::eye(4);
        let lu = LuDecomposition::decompose(&eye).unwrap();
        let inv = lu.inverse().unwrap();
        assert!(approx_eq(inv.as_slice(), eye.as_slice(), 1e-14));
    }

    #[test]
    fn test_not_square() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        assert!(LuDecomposition::decompose(&a).is_err());
    }

    #[test]
    fn test_not_2d() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        assert!(LuDecomposition::decompose(&a).is_err());
    }

    #[test]
    fn test_solve_dimension_mismatch() {
        let a = mat(&[1.0, 0.0, 0.0, 1.0], 2);
        let b = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let lu = LuDecomposition::decompose(&a).unwrap();
        assert!(lu.solve(&b).is_err());
    }
}
