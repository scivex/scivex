//! Singular Value Decomposition (SVD).
//!
//! Decomposes a matrix `A` (m x n) into `A = U S V^T` where:
//! - `U` is orthogonal (m x m)
//! - `S` is diagonal with non-negative entries (m x n)
//! - `V` is orthogonal (n x n)
//!
//! Implementation uses the one-sided Jacobi method.

use crate::Float;
use crate::error::{CoreError, Result};
use crate::tensor::Tensor;

/// Result of a Singular Value Decomposition.
#[derive(Debug, Clone)]
pub struct SvdDecomposition<T: Float> {
    /// Left singular vectors (m x m).
    u: Vec<T>,
    /// Singular values in descending order (length min(m,n)).
    s: Vec<T>,
    /// Right singular vectors (n x n).
    vt: Vec<T>,
    /// Number of rows.
    m: usize,
    /// Number of columns.
    n: usize,
}

/// Maximum number of Jacobi sweeps before giving up.
const MAX_SWEEPS: usize = 100;

#[allow(clippy::many_single_char_names)]
impl<T: Float> SvdDecomposition<T> {
    /// Compute the SVD of a matrix `A` (m x n).
    ///
    /// Returns `A = U diag(s) V^T`.
    ///
    /// ```
    /// # use scivex_core::tensor::Tensor;
    /// # use scivex_core::linalg::decomp::SvdDecomposition;
    /// let a = Tensor::from_vec(vec![3.0_f64, 0.0, 0.0, 4.0], vec![2, 2]).unwrap();
    /// let svd = SvdDecomposition::decompose(&a).unwrap();
    /// let s = svd.singular_values();
    /// // Singular values of diag(3,4) are 4 and 3
    /// assert!((s[0] - 4.0).abs() < 1e-10);
    /// assert!((s[1] - 3.0).abs() < 1e-10);
    /// ```
    pub fn decompose(a: &Tensor<T>) -> Result<Self> {
        if a.ndim() != 2 {
            return Err(CoreError::InvalidArgument {
                reason: "SVD requires a 2-D tensor (matrix)",
            });
        }
        let m = a.shape()[0];
        let n = a.shape()[1];

        if m >= n {
            Self::svd_tall(a.as_slice(), m, n)
        } else {
            // For wide matrices, transpose, compute SVD, then swap U and V
            let mut at = vec![T::zero(); m * n];
            for i in 0..m {
                for j in 0..n {
                    at[j * m + i] = a.as_slice()[i * n + j];
                }
            }
            let result = Self::svd_tall(&at, n, m)?;
            // A^T = U' S V'^T => A = V' S U'^T
            // result.u is U' (n x n), result.vt is V'^T (m x m)
            // For A: U_A = V' = transpose(V'^T), V_A^T = U'^T = transpose(U')
            // Transpose result.vt (m x m) to get V' as our U
            let mut u_a = vec![T::zero(); m * m];
            for i in 0..m {
                for j in 0..m {
                    u_a[i * m + j] = result.vt[j * m + i];
                }
            }
            // Transpose result.u (n x n) to get U'^T as our V^T
            let mut vt_a = vec![T::zero(); n * n];
            for i in 0..n {
                for j in 0..n {
                    vt_a[i * n + j] = result.u[j * n + i];
                }
            }
            Ok(Self {
                u: u_a,
                s: result.s,
                vt: vt_a,
                m,
                n,
            })
        }
    }

    /// SVD for tall/square matrices (m >= n) using one-sided Jacobi.
    #[allow(clippy::too_many_lines, clippy::unnecessary_wraps)]
    fn svd_tall(a_data: &[T], m: usize, n: usize) -> Result<Self> {
        // Working copy: columns of A that will converge to U * S
        let mut work = a_data.to_vec();
        // V starts as identity
        let mut v = vec![T::zero(); n * n];
        for i in 0..n {
            v[i * n + i] = T::one();
        }

        let tol = T::epsilon() * T::from_f64(100.0);

        for _sweep in 0..MAX_SWEEPS {
            let mut converged = true;

            // Apply Jacobi rotations to pairs of columns (p, q)
            for p in 0..n {
                for q in (p + 1)..n {
                    // Compute Gram matrix elements for columns p and q
                    let mut app = T::zero();
                    let mut aqq = T::zero();
                    let mut apq = T::zero();
                    for i in 0..m {
                        let wp = work[i * n + p];
                        let wq = work[i * n + q];
                        app += wp * wp;
                        aqq += wq * wq;
                        apq += wp * wq;
                    }

                    // Check convergence for this pair
                    if apq.abs() <= tol * (app * aqq).sqrt() {
                        continue;
                    }
                    converged = false;

                    // Compute Jacobi rotation angle
                    let tau = (aqq - app) / (apq + apq);
                    let t = if tau >= T::zero() {
                        T::one() / (tau + (T::one() + tau * tau).sqrt())
                    } else {
                        T::zero() - T::one() / (T::zero() - tau + (T::one() + tau * tau).sqrt())
                    };
                    let cs = T::one() / (T::one() + t * t).sqrt();
                    let sn = t * cs;

                    // Rotate columns p and q of work (R = [[c, s],[-s, c]])
                    for i in 0..m {
                        let wp = work[i * n + p];
                        let wq = work[i * n + q];
                        work[i * n + p] = cs * wp - sn * wq;
                        work[i * n + q] = sn * wp + cs * wq;
                    }

                    // Rotate columns p and q of V
                    for i in 0..n {
                        let vp = v[i * n + p];
                        let vq = v[i * n + q];
                        v[i * n + p] = cs * vp - sn * vq;
                        v[i * n + q] = sn * vp + cs * vq;
                    }
                }
            }

            if converged {
                break;
            }
        }

        // Extract singular values (column norms of work) and normalize to get U
        let mut s = vec![T::zero(); n];
        let mut u = vec![T::zero(); m * m];

        for j in 0..n {
            let mut norm_sq = T::zero();
            for i in 0..m {
                norm_sq += work[i * n + j] * work[i * n + j];
            }
            let norm = norm_sq.sqrt();
            s[j] = norm;

            if norm > tol {
                for i in 0..m {
                    u[i * m + j] = work[i * n + j] / norm;
                }
            }
        }

        // Complete U to an orthogonal basis using Gram-Schmidt for remaining columns
        for j in n..m {
            // Start with a unit vector
            u[j * m + j] = T::one();
            // Orthogonalize against all previous columns
            for k in 0..j {
                let mut dot = T::zero();
                for i in 0..m {
                    dot += u[i * m + j] * u[i * m + k];
                }
                for i in 0..m {
                    let uk = u[i * m + k];
                    u[i * m + j] -= dot * uk;
                }
            }
            // Normalize
            let mut norm_sq = T::zero();
            for i in 0..m {
                norm_sq += u[i * m + j] * u[i * m + j];
            }
            let norm = norm_sq.sqrt();
            if norm > tol {
                for i in 0..m {
                    u[i * m + j] /= norm;
                }
            }
        }

        // Sort singular values in descending order
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| s[b].partial_cmp(&s[a]).unwrap_or(std::cmp::Ordering::Equal));

        let mut s_sorted = vec![T::zero(); n];
        let mut u_sorted = vec![T::zero(); m * m];
        let mut v_sorted = vec![T::zero(); n * n];

        for (new_j, &old_j) in indices.iter().enumerate() {
            s_sorted[new_j] = s[old_j];
            for i in 0..m {
                u_sorted[i * m + new_j] = u[i * m + old_j];
            }
            for i in 0..n {
                v_sorted[i * n + new_j] = v[i * n + old_j];
            }
        }
        // Copy remaining U columns (beyond n) unchanged
        for j in n..m {
            for i in 0..m {
                u_sorted[i * m + j] = u[i * m + j];
            }
        }

        // Transpose V to get V^T
        let mut vt = vec![T::zero(); n * n];
        for i in 0..n {
            for j in 0..n {
                vt[i * n + j] = v_sorted[j * n + i];
            }
        }

        Ok(Self {
            u: u_sorted,
            s: s_sorted,
            vt,
            m,
            n,
        })
    }

    /// The singular values in descending order.
    pub fn singular_values(&self) -> &[T] {
        &self.s
    }

    /// The left singular vectors `U` (m x m).
    pub fn u(&self) -> Tensor<T> {
        Tensor::from_vec(self.u.clone(), vec![self.m, self.m]).unwrap()
    }

    /// The right singular vectors transposed `V^T` (n x n).
    pub fn vt(&self) -> Tensor<T> {
        Tensor::from_vec(self.vt.clone(), vec![self.n, self.n]).unwrap()
    }

    /// The singular values as a 1-D tensor.
    pub fn s(&self) -> Tensor<T> {
        Tensor::from_vec(self.s.clone(), vec![self.s.len()]).unwrap()
    }

    /// Compute the matrix rank (number of singular values above a tolerance).
    pub fn rank(&self, tol: T) -> usize {
        self.s.iter().filter(|&&sv| sv > tol).count()
    }

    /// Compute the condition number (ratio of largest to smallest singular value).
    pub fn condition_number(&self) -> T {
        if self.s.is_empty() {
            return T::zero();
        }
        let max_s = self.s[0];
        let min_s = self.s[self.s.len() - 1];
        if min_s.abs() < T::epsilon() {
            return T::from_f64(f64::INFINITY);
        }
        max_s / min_s
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp, clippy::many_single_char_names)]
mod tests {
    use super::*;

    fn approx_eq(a: &[f64], b: &[f64], tol: f64) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(&x, &y)| (x - y).abs() < tol)
    }

    #[test]
    fn test_svd_diagonal() {
        let a = Tensor::from_vec(vec![3.0_f64, 0.0, 0.0, 4.0], vec![2, 2]).unwrap();
        let svd = SvdDecomposition::decompose(&a).unwrap();
        let s = svd.singular_values();
        assert!((s[0] - 4.0).abs() < 1e-10);
        assert!((s[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_svd_identity() {
        let eye = Tensor::<f64>::eye(3);
        let svd = SvdDecomposition::decompose(&eye).unwrap();
        let s = svd.singular_values();
        for &sv in s {
            assert!((sv - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_svd_reconstruction() {
        let a = Tensor::from_vec(
            vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            vec![3, 3],
        )
        .unwrap();
        let svd = SvdDecomposition::decompose(&a).unwrap();

        let u = svd.u();
        let vt = svd.vt();
        let s = svd.singular_values();

        // Reconstruct: A = U * diag(s) * V^T
        // First build S matrix (m x n)
        let m = a.shape()[0];
        let n = a.shape()[1];
        let mut s_mat = vec![0.0_f64; m * n];
        for i in 0..s.len() {
            s_mat[i * n + i] = s[i];
        }
        let s_tensor = Tensor::from_vec(s_mat, vec![m, n]).unwrap();

        let us = u.matmul(&s_tensor).unwrap();
        let reconstructed = us.matmul(&vt).unwrap();

        assert!(approx_eq(reconstructed.as_slice(), a.as_slice(), 1e-8));
    }

    #[test]
    fn test_svd_tall_matrix() {
        let a = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]).unwrap();
        let svd = SvdDecomposition::decompose(&a).unwrap();
        assert_eq!(svd.singular_values().len(), 2);

        // Reconstruct
        let u = svd.u();
        let vt = svd.vt();
        let s = svd.singular_values();

        let mut s_mat = vec![0.0_f64; 3 * 2];
        for i in 0..2 {
            s_mat[i * 2 + i] = s[i];
        }
        let s_tensor = Tensor::from_vec(s_mat, vec![3, 2]).unwrap();
        let us = u.matmul(&s_tensor).unwrap();
        let reconstructed = us.matmul(&vt).unwrap();

        assert!(approx_eq(reconstructed.as_slice(), a.as_slice(), 1e-8));
    }

    #[test]
    fn test_svd_wide_matrix() {
        let a = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let svd = SvdDecomposition::decompose(&a).unwrap();
        assert_eq!(svd.singular_values().len(), 2);

        let u = svd.u();
        let vt = svd.vt();
        let s = svd.singular_values();

        let mut s_mat = vec![0.0_f64; 2 * 3];
        for i in 0..2 {
            s_mat[i * 3 + i] = s[i];
        }
        let s_tensor = Tensor::from_vec(s_mat, vec![2, 3]).unwrap();
        let us = u.matmul(&s_tensor).unwrap();
        let reconstructed = us.matmul(&vt).unwrap();

        assert!(approx_eq(reconstructed.as_slice(), a.as_slice(), 1e-8));
    }

    #[test]
    fn test_svd_rank() {
        // Rank-1 matrix
        let a = Tensor::from_vec(
            vec![1.0_f64, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0],
            vec![3, 3],
        )
        .unwrap();
        let svd = SvdDecomposition::decompose(&a).unwrap();
        assert_eq!(svd.rank(1e-8), 1);
    }

    #[test]
    fn test_svd_condition_number() {
        let eye = Tensor::<f64>::eye(3);
        let svd = SvdDecomposition::decompose(&eye).unwrap();
        assert!((svd.condition_number() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_svd_u_orthogonal() {
        let a = Tensor::from_vec(
            vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0],
            vec![3, 3],
        )
        .unwrap();
        let svd = SvdDecomposition::decompose(&a).unwrap();
        let u = svd.u();
        let ut = u.transpose().unwrap();
        let utu = ut.matmul(&u).unwrap();
        let eye = Tensor::<f64>::eye(3);
        assert!(approx_eq(utu.as_slice(), eye.as_slice(), 1e-8));
    }

    #[test]
    fn test_svd_vt_orthogonal() {
        let a = Tensor::from_vec(
            vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0],
            vec![3, 3],
        )
        .unwrap();
        let svd = SvdDecomposition::decompose(&a).unwrap();
        let vt = svd.vt();
        let v = vt.transpose().unwrap();
        let vtv = vt.matmul(&v).unwrap();
        let eye = Tensor::<f64>::eye(3);
        assert!(approx_eq(vtv.as_slice(), eye.as_slice(), 1e-8));
    }

    #[test]
    fn test_svd_not_2d() {
        let a = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0], vec![3]).unwrap();
        assert!(SvdDecomposition::decompose(&a).is_err());
    }
}
