//! QR decomposition via Householder reflections.
//!
//! Decomposes a matrix `A` (m x n, m >= n) into `A = QR` where:
//! - `Q` is an orthogonal matrix (m x m) such that `Q^T Q = I`
//! - `R` is upper triangular (m x n)

use crate::Float;
use crate::error::{CoreError, Result};
use crate::tensor::Tensor;

/// Result of a QR decomposition via Householder reflections.
///
/// Stores the factorization `A = QR` in compact form: the Householder
/// vectors are stored in the lower triangle of the working matrix,
/// and `R` is stored in the upper triangle.
///
/// # Examples
///
/// ```
/// # use scivex_core::tensor::Tensor;
/// # use scivex_core::linalg::decomp::QrDecomposition;
/// let a = Tensor::from_vec(vec![1.0_f64, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
/// let qr = QrDecomposition::decompose(&a).unwrap();
/// assert!(qr.is_full_rank());
/// ```
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct QrDecomposition<T: Float> {
    /// Working matrix: upper triangle holds R, columns below the diagonal
    /// hold the Householder vectors (without their leading 1).
    qr: Vec<T>,
    /// Diagonal of R stored separately (the Householder reflections
    /// overwrite the diagonal of the working matrix).
    r_diag: Vec<T>,
    /// Number of rows.
    m: usize,
    /// Number of columns.
    n: usize,
}

#[allow(clippy::many_single_char_names)]
impl<T: Float> QrDecomposition<T> {
    /// Perform QR decomposition on a matrix `A` (m x n, m >= n).
    ///
    /// Uses Householder reflections for numerical stability.
    ///
    /// ```
    /// # use scivex_core::tensor::Tensor;
    /// # use scivex_core::linalg::decomp::QrDecomposition;
    /// let a = Tensor::from_vec(
    ///     vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
    ///     vec![3, 2],
    /// ).unwrap();
    /// let qr = QrDecomposition::decompose(&a).unwrap();
    /// let q = qr.q();
    /// let r = qr.r();
    /// // Verify Q is orthogonal: Q^T Q ≈ I
    /// let qt = q.transpose().unwrap();
    /// let qtq = qt.matmul(&q).unwrap();
    /// let eye = Tensor::<f64>::eye(3);
    /// for (a, b) in qtq.as_slice().iter().zip(eye.as_slice()) {
    ///     assert!((a - b).abs() < 1e-10);
    /// }
    /// ```
    /// Block size for the blocked Householder QR algorithm.
    /// Columns are processed in panels of this width; the accumulated
    /// block reflector is then applied to the trailing matrix using
    /// matrix-matrix operations (BLAS-3) for better cache utilization.
    const BLOCK_SIZE: usize = 32;

    pub fn decompose(a: &Tensor<T>) -> Result<Self> {
        if a.ndim() != 2 {
            return Err(CoreError::InvalidArgument {
                reason: "QR decomposition requires a 2-D tensor (matrix)",
            });
        }
        let m = a.shape()[0];
        let n = a.shape()[1];
        if m < n {
            return Err(CoreError::InvalidArgument {
                reason: "QR decomposition requires m >= n (tall or square matrix)",
            });
        }

        let mut qr: Vec<T> = a.as_slice().to_vec();
        let mut r_diag = vec![T::zero(); n];
        let nb = Self::BLOCK_SIZE.min(n);

        // Process columns in panels of width `nb`.
        let mut k0 = 0;
        while k0 < n {
            let k_end = (k0 + nb).min(n);
            let panel_width = k_end - k0;

            // tau[j] = qr[(k0+j)*n + (k0+j)] after reflection setup.
            let mut tau = vec![T::zero(); panel_width];

            // --- Panel factorization: column-by-column within the panel ---
            for (jj, tau_val) in tau.iter_mut().enumerate().take(panel_width) {
                let k = k0 + jj;
                // Compute norm of column k below the diagonal.
                let mut norm_sq = T::zero();
                for i in k..m {
                    norm_sq += qr[i * n + k] * qr[i * n + k];
                }
                let mut norm = norm_sq.sqrt();

                if norm.abs() < T::epsilon() * T::from_f64(1e3) {
                    r_diag[k] = T::zero();
                    *tau_val = T::zero();
                    continue;
                }

                // Choose sign to avoid cancellation.
                if qr[k * n + k] > T::zero() {
                    norm = T::zero() - norm;
                }

                // Scale the Householder vector.
                for i in k..m {
                    qr[i * n + k] /= T::zero() - norm;
                }
                qr[k * n + k] += T::one();
                *tau_val = qr[k * n + k];

                // Apply reflector to remaining columns *within the panel only*.
                for j in (k + 1)..k_end {
                    let mut s = T::zero();
                    for i in k..m {
                        s += qr[i * n + k] * qr[i * n + j];
                    }
                    s = T::zero() - s / qr[k * n + k];
                    for i in k..m {
                        let v = qr[i * n + k];
                        qr[i * n + j] += s * v;
                    }
                }

                r_diag[k] = norm;
            }

            // Apply accumulated block reflector to trailing columns.
            let trailing_cols = n - k_end;
            if trailing_cols > 0 && panel_width > 0 {
                Self::apply_block_reflector(&mut qr, &tau, m, n, k0, k_end, trailing_cols);
            }

            k0 = k_end;
        }

        Ok(Self { qr, r_diag, m, n })
    }

    /// Apply the block Householder reflector (compact WY form) to the trailing
    /// submatrix. This replaces per-column rank-1 updates with BLAS-3-style
    /// matrix-matrix operations for better cache utilization.
    #[allow(clippy::needless_range_loop)]
    fn apply_block_reflector(
        qr: &mut [T],
        tau: &[T],
        m: usize,
        n: usize,
        k0: usize,
        k_end: usize,
        trailing_cols: usize,
    ) {
        let pw = k_end - k0;

        // Build T (pw x pw upper triangular) for the compact WY representation:
        // Q_block = I - V T V^T
        let mut t_mat = vec![T::zero(); pw * pw];

        for j in 0..pw {
            let tauj = tau[j];
            if tauj.abs() < T::epsilon() {
                continue;
            }
            t_mat[j * pw + j] = tauj;
            let col_j = k0 + j;

            // w = V[:, 0..j]^T * v_j
            let mut w = vec![T::zero(); j];
            for (jj2, w_val) in w.iter_mut().enumerate() {
                let col_jj2 = k0 + jj2;
                let mut dot = T::zero();
                for i in col_j..m {
                    dot += qr[i * n + col_jj2] * qr[i * n + col_j];
                }
                *w_val = dot;
            }

            // T[0..j, j] = -tau[j] * T[0..j, 0..j] * w
            let mut temp = vec![T::zero(); j];
            for r in 0..j {
                let mut s = T::zero();
                for c in r..j {
                    s += t_mat[r * pw + c] * w[c];
                }
                temp[r] = s;
            }
            for r in 0..j {
                t_mat[r * pw + j] = T::zero() - tauj * temp[r];
            }
        }

        // Apply: A_trail -= V * T * (V^T * A_trail)

        // Step 1: vta = V^T * A_trail  (pw x trailing_cols)
        let mut vta = vec![T::zero(); pw * trailing_cols];
        for jj in 0..pw {
            let col_v = k0 + jj;
            for tc in 0..trailing_cols {
                let col_a = k_end + tc;
                let mut dot = T::zero();
                for i in col_v..m {
                    dot += qr[i * n + col_v] * qr[i * n + col_a];
                }
                vta[jj * trailing_cols + tc] = dot;
            }
        }

        // Step 2: tw = T * vta  (pw x trailing_cols)
        let mut tw = vec![T::zero(); pw * trailing_cols];
        for r in 0..pw {
            for tc in 0..trailing_cols {
                let mut s = T::zero();
                for c in r..pw {
                    s += t_mat[r * pw + c] * vta[c * trailing_cols + tc];
                }
                tw[r * trailing_cols + tc] = s;
            }
        }

        // Step 3: A_trail -= V * tw
        for jj in 0..pw {
            let col_v = k0 + jj;
            for i in col_v..m {
                let vi = qr[i * n + col_v];
                for tc in 0..trailing_cols {
                    let col_a = k_end + tc;
                    qr[i * n + col_a] -= vi * tw[jj * trailing_cols + tc];
                }
            }
        }
    }

    /// Whether the matrix has full column rank.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_core::tensor::Tensor;
    /// # use scivex_core::linalg::decomp::QrDecomposition;
    /// let a = Tensor::from_vec(vec![1.0_f64, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
    /// let qr = QrDecomposition::decompose(&a).unwrap();
    /// assert!(qr.is_full_rank());
    /// ```
    pub fn is_full_rank(&self) -> bool {
        let threshold = T::epsilon() * T::from_f64(1e3);
        self.r_diag.iter().all(|d| d.abs() > threshold)
    }

    /// Extract the upper triangular matrix `R` (m x n).
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_core::tensor::Tensor;
    /// # use scivex_core::linalg::decomp::QrDecomposition;
    /// let a = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    /// let qr = QrDecomposition::decompose(&a).unwrap();
    /// let r = qr.r();
    /// assert_eq!(r.shape(), &[2, 2]);
    /// ```
    pub fn r(&self) -> Tensor<T> {
        let (m, n) = (self.m, self.n);
        let mut data = vec![T::zero(); m * n];
        for i in 0..n {
            data[i * n + i] = self.r_diag[i];
            for j in (i + 1)..n {
                data[i * n + j] = self.qr[i * n + j];
            }
        }
        // SAFETY: data has exactly m*n elements, matching shape [m, n].
        Tensor::from_vec(data, vec![m, n]).expect("R data length equals m*n by construction")
    }

    /// Extract the orthogonal matrix `Q` (m x m).
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_core::tensor::Tensor;
    /// # use scivex_core::linalg::decomp::QrDecomposition;
    /// let a = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    /// let qr = QrDecomposition::decompose(&a).unwrap();
    /// let q = qr.q();
    /// assert_eq!(q.shape(), &[2, 2]);
    /// // Q is orthogonal: Q^T Q ≈ I
    /// let qtq = q.transpose().unwrap().matmul(&q).unwrap();
    /// let eye = Tensor::<f64>::eye(2);
    /// for (a, b) in qtq.as_slice().iter().zip(eye.as_slice()) {
    ///     assert!((a - b).abs() < 1e-10);
    /// }
    /// ```
    pub fn q(&self) -> Tensor<T> {
        let (m, n) = (self.m, self.n);
        // Start with identity
        let mut q_data = vec![T::zero(); m * m];
        for i in 0..m {
            q_data[i * m + i] = T::one();
        }

        // Apply Householder reflections in reverse order
        for k in (0..n).rev() {
            if self.qr[k * n + k].abs() < T::epsilon() {
                continue;
            }
            // Apply H_k = I - v*v^T/v[k] to Q
            for j in 0..m {
                let mut s = T::zero();
                for i in k..m {
                    s += self.qr[i * n + k] * q_data[i * m + j];
                }
                s = T::zero() - s / self.qr[k * n + k];
                for i in k..m {
                    q_data[i * m + j] += s * self.qr[i * n + k];
                }
            }
        }

        // SAFETY: q_data has exactly m*m elements, matching shape [m, m].
        Tensor::from_vec(q_data, vec![m, m]).expect("Q data length equals m*m by construction")
    }

    /// Extract the "thin" Q matrix (m x n) — only the first `n` columns.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_core::tensor::Tensor;
    /// # use scivex_core::linalg::decomp::QrDecomposition;
    /// let a = Tensor::from_vec(
    ///     vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
    ///     vec![3, 2],
    /// ).unwrap();
    /// let qr = QrDecomposition::decompose(&a).unwrap();
    /// let q_thin = qr.q_thin();
    /// assert_eq!(q_thin.shape(), &[3, 2]);
    /// ```
    pub fn q_thin(&self) -> Tensor<T> {
        let (m, n) = (self.m, self.n);
        let mut q_data = vec![T::zero(); m * n];
        for i in 0..n {
            q_data[i * n + i] = T::one();
        }

        // Apply Householder reflections in reverse order
        for k in (0..n).rev() {
            if self.qr[k * n + k].abs() < T::epsilon() {
                continue;
            }
            for j in 0..n {
                let mut s = T::zero();
                for i in k..m {
                    s += self.qr[i * n + k] * q_data[i * n + j];
                }
                s = T::zero() - s / self.qr[k * n + k];
                for i in k..m {
                    q_data[i * n + j] += s * self.qr[i * n + k];
                }
            }
        }

        // SAFETY: q_data has exactly m*n elements, matching shape [m, n].
        Tensor::from_vec(q_data, vec![m, n]).expect("Q_thin data length equals m*n by construction")
    }

    /// Solve the least-squares problem `min ||Ax - b||_2`.
    ///
    /// For a full-rank system where `m == n`, this is equivalent to solving
    /// `Ax = b`. For overdetermined systems (`m > n`), it returns the
    /// least-squares solution.
    ///
    /// `b` must be a 1-D tensor of length `m`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_core::tensor::Tensor;
    /// # use scivex_core::linalg::decomp::QrDecomposition;
    /// let a = Tensor::from_vec(vec![2.0_f64, 1.0, 1.0, 4.0], vec![2, 2]).unwrap();
    /// let b = Tensor::from_vec(vec![5.0_f64, 6.0], vec![2]).unwrap();
    /// let qr = QrDecomposition::decompose(&a).unwrap();
    /// let x = qr.solve(&b).unwrap();
    /// assert!((x.as_slice()[0] - 2.0).abs() < 1e-10);
    /// assert!((x.as_slice()[1] - 1.0).abs() < 1e-10);
    /// ```
    pub fn solve(&self, b: &Tensor<T>) -> Result<Tensor<T>> {
        if !self.is_full_rank() {
            return Err(CoreError::InvalidArgument {
                reason: "QR solve: matrix does not have full column rank",
            });
        }
        if b.ndim() != 1 {
            return Err(CoreError::InvalidArgument {
                reason: "QR solve: `b` must be a 1-D tensor",
            });
        }
        if b.numel() != self.m {
            return Err(CoreError::DimensionMismatch {
                expected: vec![self.m],
                got: b.shape().to_vec(),
            });
        }

        let (m, n) = (self.m, self.n);
        let mut x: Vec<T> = b.as_slice().to_vec();

        // Compute Q^T b by applying Householder reflections
        #[allow(clippy::needless_range_loop)]
        for k in 0..n {
            if self.qr[k * n + k].abs() < T::epsilon() {
                continue;
            }
            let mut s = T::zero();
            for i in k..m {
                s += self.qr[i * n + k] * x[i];
            }
            s = T::zero() - s / self.qr[k * n + k];
            for i in k..m {
                x[i] += s * self.qr[i * n + k];
            }
        }

        // Back substitution on the upper triangular part: Rx = Q^T b
        #[allow(clippy::needless_range_loop)]
        for i in (0..n).rev() {
            for j in (i + 1)..n {
                let xj = x[j];
                x[i] -= self.qr[i * n + j] * xj;
            }
            x[i] /= self.r_diag[i];
        }

        // Only the first n elements are the solution
        x.truncate(n);
        Tensor::from_vec(x, vec![n])
    }
}

/// Solve the least-squares problem `min ||Ax - b||_2` via QR decomposition.
///
/// For square full-rank systems this is equivalent to `solve`. For
/// overdetermined systems (`m > n`), returns the least-squares solution.
///
/// ```
/// # use scivex_core::tensor::Tensor;
/// # use scivex_core::linalg::decomp;
/// // Overdetermined system: 3 equations, 2 unknowns
/// let a = Tensor::from_vec(
///     vec![1.0_f64, 1.0, 1.0, 2.0, 1.0, 3.0],
///     vec![3, 2],
/// ).unwrap();
/// let b = Tensor::from_vec(vec![6.0_f64, 5.0, 7.0], vec![3]).unwrap();
/// let x = decomp::lstsq(&a, &b).unwrap();
/// assert_eq!(x.shape(), &[2]);
/// ```
pub fn lstsq<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>> {
    QrDecomposition::decompose(a)?.solve(b)
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    fn approx_eq(a: &[f64], b: &[f64], tol: f64) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(&x, &y)| (x - y).abs() < tol)
    }

    #[test]
    fn test_qr_3x3() {
        let a = Tensor::from_vec(
            vec![12.0_f64, -51.0, 4.0, 6.0, 167.0, -68.0, -4.0, 24.0, -41.0],
            vec![3, 3],
        )
        .unwrap();
        let qr = QrDecomposition::decompose(&a).unwrap();

        let q = qr.q();
        let r = qr.r();

        // Verify A = QR
        let qr_prod = q.matmul(&r).unwrap();
        assert!(approx_eq(qr_prod.as_slice(), a.as_slice(), 1e-10));

        // Verify Q is orthogonal: Q^T Q = I
        let qt = q.transpose().unwrap();
        let qtq = qt.matmul(&q).unwrap();
        let eye = Tensor::<f64>::eye(3);
        assert!(approx_eq(qtq.as_slice(), eye.as_slice(), 1e-10));
    }

    #[test]
    fn test_qr_identity() {
        let eye = Tensor::<f64>::eye(4);
        let qr = QrDecomposition::decompose(&eye).unwrap();

        let q = qr.q();
        let r = qr.r();
        let qr_prod = q.matmul(&r).unwrap();
        assert!(approx_eq(qr_prod.as_slice(), eye.as_slice(), 1e-14));
    }

    #[test]
    fn test_qr_tall_matrix() {
        // 4x2 matrix
        let a =
            Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![4, 2]).unwrap();
        let qr = QrDecomposition::decompose(&a).unwrap();

        let q = qr.q();
        let r = qr.r();

        // Q is 4x4, R is 4x2
        assert_eq!(q.shape(), &[4, 4]);
        assert_eq!(r.shape(), &[4, 2]);

        // Verify A = QR
        let qr_prod = q.matmul(&r).unwrap();
        assert!(approx_eq(qr_prod.as_slice(), a.as_slice(), 1e-10));

        // Q orthogonal
        let qt = q.transpose().unwrap();
        let qtq = qt.matmul(&q).unwrap();
        let eye = Tensor::<f64>::eye(4);
        assert!(approx_eq(qtq.as_slice(), eye.as_slice(), 1e-10));
    }

    #[test]
    fn test_qr_thin_q() {
        let a =
            Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![4, 2]).unwrap();
        let qr = QrDecomposition::decompose(&a).unwrap();

        let q_thin = qr.q_thin();
        assert_eq!(q_thin.shape(), &[4, 2]);

        // Q_thin^T Q_thin should be 2x2 identity
        let qt = q_thin.transpose().unwrap();
        let qtq = qt.matmul(&q_thin).unwrap();
        let eye = Tensor::<f64>::eye(2);
        assert!(approx_eq(qtq.as_slice(), eye.as_slice(), 1e-10));
    }

    #[test]
    fn test_qr_solve_square() {
        // Same system as LU test: 2x + y = 5, x + 4y = 6 => x=2, y=1
        let a = Tensor::from_vec(vec![2.0_f64, 1.0, 1.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::from_vec(vec![5.0_f64, 6.0], vec![2]).unwrap();
        let qr = QrDecomposition::decompose(&a).unwrap();
        let x = qr.solve(&b).unwrap();
        assert!(approx_eq(x.as_slice(), &[2.0, 1.0], 1e-10));
    }

    #[test]
    fn test_qr_solve_3x3() {
        // >>> A = np.array([[1,2,3],[4,5,6],[7,8,10]])
        // >>> b = np.array([1,2,3])
        // >>> np.linalg.lstsq(A, b, rcond=None)[0]
        // array([-0.33333333,  0.66666667,  0.        ])
        let a = Tensor::from_vec(
            vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0],
            vec![3, 3],
        )
        .unwrap();
        let b = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0], vec![3]).unwrap();
        let qr = QrDecomposition::decompose(&a).unwrap();
        let x = qr.solve(&b).unwrap();
        assert!(approx_eq(
            x.as_slice(),
            &[-1.0 / 3.0, 2.0 / 3.0, 0.0],
            1e-10
        ));
    }

    #[test]
    fn test_lstsq_overdetermined() {
        // Overdetermined system: fit y = a + b*x to points (1,6), (2,5), (3,7)
        // A = [[1,1],[1,2],[1,3]], b = [6,5,7]
        // Normal equations: A^T A = [[3,6],[6,14]], A^T b = [18,37]
        // Solution: x = [5.0, 0.5]
        let a = Tensor::from_vec(vec![1.0_f64, 1.0, 1.0, 2.0, 1.0, 3.0], vec![3, 2]).unwrap();
        let b = Tensor::from_vec(vec![6.0_f64, 5.0, 7.0], vec![3]).unwrap();
        let x = lstsq(&a, &b).unwrap();
        assert!(approx_eq(x.as_slice(), &[5.0, 0.5], 1e-10));
    }

    #[test]
    fn test_lstsq_perfect_fit() {
        // Exactly determined system through lstsq path
        // y = 2x + 1 at x=0,1,2 => b=[1,3,5]
        // A = [[1,0],[1,1],[1,2]], b = [1,3,5]
        let a = Tensor::from_vec(vec![1.0_f64, 0.0, 1.0, 1.0, 1.0, 2.0], vec![3, 2]).unwrap();
        let b = Tensor::from_vec(vec![1.0_f64, 3.0, 5.0], vec![3]).unwrap();
        let x = lstsq(&a, &b).unwrap();
        assert!(approx_eq(x.as_slice(), &[1.0, 2.0], 1e-10));
    }

    #[test]
    fn test_qr_is_full_rank() {
        let a = Tensor::from_vec(vec![1.0_f64, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let qr = QrDecomposition::decompose(&a).unwrap();
        assert!(qr.is_full_rank());
    }

    #[test]
    fn test_qr_not_full_rank() {
        // Columns are linearly dependent
        let a = Tensor::from_vec(vec![1.0_f64, 2.0, 2.0, 4.0, 3.0, 6.0], vec![3, 2]).unwrap();
        let qr = QrDecomposition::decompose(&a).unwrap();
        assert!(!qr.is_full_rank());
    }

    #[test]
    fn test_qr_solve_not_full_rank() {
        let a = Tensor::from_vec(vec![1.0_f64, 2.0, 2.0, 4.0, 3.0, 6.0], vec![3, 2]).unwrap();
        let b = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0], vec![3]).unwrap();
        let qr = QrDecomposition::decompose(&a).unwrap();
        assert!(qr.solve(&b).is_err());
    }

    #[test]
    fn test_qr_not_2d() {
        let a = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0], vec![3]).unwrap();
        assert!(QrDecomposition::decompose(&a).is_err());
    }

    #[test]
    fn test_qr_wide_matrix() {
        // m < n should fail
        let a = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        assert!(QrDecomposition::decompose(&a).is_err());
    }

    #[test]
    fn test_qr_solve_dimension_mismatch() {
        let a = Tensor::from_vec(vec![1.0_f64, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let b = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0], vec![3]).unwrap();
        let qr = QrDecomposition::decompose(&a).unwrap();
        assert!(qr.solve(&b).is_err());
    }

    #[test]
    fn test_lstsq_4x3_numpy() {
        // >>> A = np.array([[1,1,1],[1,2,4],[1,3,9],[1,4,16]], dtype=float)
        // >>> b = np.array([2,3,5,8], dtype=float)
        // Normal equations solution: x = [2.0, -0.5, 0.5]
        let a = Tensor::from_vec(
            vec![
                1.0_f64, 1.0, 1.0, 1.0, 2.0, 4.0, 1.0, 3.0, 9.0, 1.0, 4.0, 16.0,
            ],
            vec![4, 3],
        )
        .unwrap();
        let b = Tensor::from_vec(vec![2.0_f64, 3.0, 5.0, 8.0], vec![4]).unwrap();
        let x = lstsq(&a, &b).unwrap();
        assert!(approx_eq(x.as_slice(), &[2.0, -0.5, 0.5], 1e-10));
    }
}
