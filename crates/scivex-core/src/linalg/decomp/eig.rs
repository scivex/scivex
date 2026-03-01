//! Eigendecomposition for symmetric matrices.
//!
//! Decomposes a real symmetric matrix `A` into `A = V D V^T` where:
//! - `V` is orthogonal (columns are eigenvectors)
//! - `D` is diagonal (eigenvalues on the diagonal)
//!
//! Implementation uses the Jacobi eigenvalue algorithm.

use crate::Float;
use crate::error::{CoreError, Result};
use crate::tensor::Tensor;

/// Result of an eigendecomposition for symmetric matrices.
#[derive(Debug, Clone)]
pub struct EigDecomposition<T: Float> {
    /// Eigenvalues in descending order of absolute value.
    eigenvalues: Vec<T>,
    /// Eigenvectors as columns of an n x n matrix (stored row-major).
    eigenvectors: Vec<T>,
    /// Matrix dimension.
    n: usize,
}

/// Maximum number of Jacobi sweeps.
const MAX_SWEEPS: usize = 100;

#[allow(clippy::many_single_char_names)]
impl<T: Float> EigDecomposition<T> {
    /// Compute the eigendecomposition of a symmetric matrix.
    ///
    /// Returns eigenvalues and eigenvectors such that `A = V diag(d) V^T`.
    /// The matrix must be symmetric; only the lower triangle is read.
    ///
    /// ```
    /// # use scivex_core::tensor::Tensor;
    /// # use scivex_core::linalg::decomp::EigDecomposition;
    /// let a = Tensor::from_vec(vec![2.0_f64, 1.0, 1.0, 3.0], vec![2, 2]).unwrap();
    /// let eig = EigDecomposition::decompose_symmetric(&a).unwrap();
    /// let vals = eig.eigenvalues();
    /// // Eigenvalues of [[2,1],[1,3]] are (5±sqrt(5))/2
    /// assert!(vals.len() == 2);
    /// ```
    pub fn decompose_symmetric(a: &Tensor<T>) -> Result<Self> {
        if a.ndim() != 2 {
            return Err(CoreError::InvalidArgument {
                reason: "Eigendecomposition requires a 2-D tensor (matrix)",
            });
        }
        let n = a.shape()[0];
        if a.shape()[1] != n {
            return Err(CoreError::InvalidArgument {
                reason: "Eigendecomposition requires a square matrix",
            });
        }

        // Copy A into working matrix (symmetric, so we use the full matrix)
        let mut s: Vec<T> = a.as_slice().to_vec();
        // V starts as identity
        let mut v = vec![T::zero(); n * n];
        for i in 0..n {
            v[i * n + i] = T::one();
        }

        let tol = T::epsilon() * T::from_f64(100.0);

        for _sweep in 0..MAX_SWEEPS {
            // Compute off-diagonal norm
            let mut off_norm = T::zero();
            for i in 0..n {
                for j in (i + 1)..n {
                    off_norm += s[i * n + j] * s[i * n + j];
                }
            }
            if off_norm.sqrt() < tol {
                break;
            }

            for p in 0..n {
                for q in (p + 1)..n {
                    let apq = s[p * n + q];
                    if apq.abs() < tol {
                        continue;
                    }

                    let app = s[p * n + p];
                    let aqq = s[q * n + q];

                    // Compute rotation angle
                    let theta = (aqq - app) / (apq + apq);
                    let t = if theta >= T::zero() {
                        T::one() / (theta + (T::one() + theta * theta).sqrt())
                    } else {
                        T::zero()
                            - T::one() / (T::zero() - theta + (T::one() + theta * theta).sqrt())
                    };
                    let cs = T::one() / (T::one() + t * t).sqrt();
                    let sn = t * cs;

                    // Apply rotation to S: S' = G^T S G
                    // Update rows/cols p and q
                    s[p * n + p] = app - t * apq;
                    s[q * n + q] = aqq + t * apq;
                    s[p * n + q] = T::zero();
                    s[q * n + p] = T::zero();

                    for r in 0..n {
                        if r == p || r == q {
                            continue;
                        }
                        let srp = s[r * n + p];
                        let srq = s[r * n + q];
                        s[r * n + p] = cs * srp - sn * srq;
                        s[p * n + r] = cs * srp - sn * srq;
                        s[r * n + q] = sn * srp + cs * srq;
                        s[q * n + r] = sn * srp + cs * srq;
                    }

                    // Update eigenvector matrix V (V' = V * R)
                    for i in 0..n {
                        let vp = v[i * n + p];
                        let vq = v[i * n + q];
                        v[i * n + p] = cs * vp - sn * vq;
                        v[i * n + q] = sn * vp + cs * vq;
                    }
                }
            }
        }

        // Extract eigenvalues from the diagonal of S
        let eigenvalues: Vec<T> = (0..n).map(|i| s[i * n + i]).collect();

        // Sort by descending absolute value
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| {
            eigenvalues[b]
                .abs()
                .partial_cmp(&eigenvalues[a].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let eigenvalues_sorted: Vec<T> = indices.iter().map(|&i| eigenvalues[i]).collect();
        let mut v_sorted = vec![T::zero(); n * n];
        for (new_j, &old_j) in indices.iter().enumerate() {
            for i in 0..n {
                v_sorted[i * n + new_j] = v[i * n + old_j];
            }
        }

        Ok(Self {
            eigenvalues: eigenvalues_sorted,
            eigenvectors: v_sorted,
            n,
        })
    }

    /// The eigenvalues, sorted by descending absolute value.
    pub fn eigenvalues(&self) -> &[T] {
        &self.eigenvalues
    }

    /// The eigenvalues as a 1-D tensor.
    pub fn eigenvalues_tensor(&self) -> Tensor<T> {
        Tensor::from_vec(self.eigenvalues.clone(), vec![self.n]).unwrap()
    }

    /// The eigenvector matrix `V` (n x n, columns are eigenvectors).
    pub fn eigenvectors(&self) -> Tensor<T> {
        Tensor::from_vec(self.eigenvectors.clone(), vec![self.n, self.n]).unwrap()
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    fn approx_eq(a: &[f64], b: &[f64], tol: f64) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(&x, &y)| (x - y).abs() < tol)
    }

    #[test]
    fn test_eig_diagonal() {
        let a = Tensor::from_vec(vec![3.0_f64, 0.0, 0.0, 5.0], vec![2, 2]).unwrap();
        let eig = EigDecomposition::decompose_symmetric(&a).unwrap();
        let vals = eig.eigenvalues();
        assert!((vals[0] - 5.0).abs() < 1e-10);
        assert!((vals[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_eig_identity() {
        let eye = Tensor::<f64>::eye(3);
        let eig = EigDecomposition::decompose_symmetric(&eye).unwrap();
        for &v in eig.eigenvalues() {
            assert!((v - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_eig_2x2() {
        // [[2,1],[1,3]] eigenvalues: (5 ± sqrt(5))/2
        let a = Tensor::from_vec(vec![2.0_f64, 1.0, 1.0, 3.0], vec![2, 2]).unwrap();
        let eig = EigDecomposition::decompose_symmetric(&a).unwrap();
        let vals = eig.eigenvalues();
        let sqrt5 = 5.0_f64.sqrt();
        let expected_1 = 2.5 + sqrt5 * 0.5;
        let expected_2 = 2.5 - sqrt5 * 0.5;
        assert!((vals[0] - expected_1).abs() < 1e-10);
        assert!((vals[1] - expected_2).abs() < 1e-10);
    }

    #[test]
    fn test_eig_reconstruction() {
        let a = Tensor::from_vec(
            vec![4.0_f64, 2.0, 1.0, 2.0, 5.0, 3.0, 1.0, 3.0, 6.0],
            vec![3, 3],
        )
        .unwrap();
        let eig = EigDecomposition::decompose_symmetric(&a).unwrap();
        let v = eig.eigenvectors();
        let vt = v.transpose().unwrap();

        // Reconstruct: A = V D V^T
        let d_vals = eig.eigenvalues();
        let n = 3;
        let mut d_data = vec![0.0_f64; n * n];
        for i in 0..n {
            d_data[i * n + i] = d_vals[i];
        }
        let d = Tensor::from_vec(d_data, vec![n, n]).unwrap();

        let vd = v.matmul(&d).unwrap();
        let reconstructed = vd.matmul(&vt).unwrap();

        assert!(approx_eq(reconstructed.as_slice(), a.as_slice(), 1e-8));
    }

    #[test]
    fn test_eig_orthogonal_eigenvectors() {
        let a = Tensor::from_vec(
            vec![4.0_f64, 2.0, 1.0, 2.0, 5.0, 3.0, 1.0, 3.0, 6.0],
            vec![3, 3],
        )
        .unwrap();
        let eig = EigDecomposition::decompose_symmetric(&a).unwrap();
        let v = eig.eigenvectors();
        let vt = v.transpose().unwrap();
        let vtv = vt.matmul(&v).unwrap();
        let eye = Tensor::<f64>::eye(3);
        assert!(approx_eq(vtv.as_slice(), eye.as_slice(), 1e-8));
    }

    #[test]
    fn test_eig_negative_eigenvalue() {
        // [[1, 2],[2, 1]] has eigenvalues 3 and -1
        let a = Tensor::from_vec(vec![1.0_f64, 2.0, 2.0, 1.0], vec![2, 2]).unwrap();
        let eig = EigDecomposition::decompose_symmetric(&a).unwrap();
        let vals = eig.eigenvalues();
        // Sorted by absolute value descending: 3, -1
        assert!((vals[0] - 3.0).abs() < 1e-10);
        assert!((vals[1] - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_eig_4x4() {
        // Symmetric 4x4
        let a = Tensor::from_vec(
            vec![
                5.0_f64, 1.0, 2.0, 0.0, 1.0, 4.0, 1.0, 1.0, 2.0, 1.0, 3.0, 0.0, 0.0, 1.0, 0.0, 2.0,
            ],
            vec![4, 4],
        )
        .unwrap();
        let eig = EigDecomposition::decompose_symmetric(&a).unwrap();
        let v = eig.eigenvectors();
        let vt = v.transpose().unwrap();

        // Reconstruct
        let d_vals = eig.eigenvalues();
        let mut d_data = vec![0.0_f64; 16];
        for i in 0..4 {
            d_data[i * 4 + i] = d_vals[i];
        }
        let d = Tensor::from_vec(d_data, vec![4, 4]).unwrap();
        let vd = v.matmul(&d).unwrap();
        let reconstructed = vd.matmul(&vt).unwrap();
        assert!(approx_eq(reconstructed.as_slice(), a.as_slice(), 1e-8));
    }

    #[test]
    fn test_eig_not_square() {
        let a = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        assert!(EigDecomposition::decompose_symmetric(&a).is_err());
    }

    #[test]
    fn test_eig_not_2d() {
        let a = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0], vec![3]).unwrap();
        assert!(EigDecomposition::decompose_symmetric(&a).is_err());
    }
}
