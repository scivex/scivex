//! Tensor decompositions: CP (CANDECOMP/PARAFAC), Tucker (HOSVD), and
//! Non-negative Tensor Factorization (NTF).
//!
//! These decompositions generalize matrix factorizations to higher-order
//! tensors. All algorithms are implemented from scratch using the existing
//! SVD and BLAS primitives in `scivex-core`.

use crate::Float;
use crate::error::{CoreError, Result};
use crate::linalg::decomp::SvdDecomposition;
use crate::tensor::Tensor;

// ======================================================================
// Helper functions
// ======================================================================

/// Mode-n unfolding (matricization) of an N-D tensor.
///
/// Reshapes an N-D tensor into a 2-D matrix where mode `n` becomes the row
/// dimension and all other modes are combined into columns (in row-major
/// order of the remaining modes).
///
/// The resulting matrix has shape `[shape[mode], prod(shape[i] for i != mode)]`.
pub fn unfold<T: Float>(x: &Tensor<T>, mode: usize) -> Result<Tensor<T>> {
    let ndim = x.ndim();
    if mode >= ndim {
        return Err(CoreError::AxisOutOfBounds { axis: mode, ndim });
    }

    let shape = x.shape();
    let rows = shape[mode];
    let cols: usize = shape
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != mode)
        .map(|(_, &d)| d)
        .product();

    if cols == 0 || rows == 0 {
        return Tensor::from_vec(vec![], vec![rows, cols]);
    }

    let mut result = vec![T::zero(); rows * cols];

    // Build the "other" dimensions list (all except mode)
    let other_dims: Vec<usize> = (0..ndim).filter(|&i| i != mode).collect();
    let _other_shape: Vec<usize> = other_dims.iter().map(|&i| shape[i]).collect();

    // Iterate over all elements via multi-index
    let numel = x.numel();
    let strides = x.strides();

    // For each element, compute its position in the unfolded matrix
    // Row = index along mode, Col = combined index of other dims
    let mut multi_idx = vec![0usize; ndim];
    for flat in 0..numel {
        // Compute multi-index from flat index
        let mut rem = flat;
        for d in 0..ndim {
            multi_idx[d] = rem / strides[d];
            rem %= strides[d];
        }

        let row = multi_idx[mode];

        // Compute column index from the other dimensions
        let mut col = 0;
        let mut col_stride = 1;
        for &d in other_dims.iter().rev() {
            col += multi_idx[d] * col_stride;
            col_stride *= shape[d];
        }

        result[row * cols + col] = *x.get(&multi_idx)?;
    }

    Tensor::from_vec(result, vec![rows, cols])
}

/// Fold (refold) a 2-D matrix back into an N-D tensor.
///
/// This is the inverse of [`unfold`]. `shape` is the target tensor shape
/// and `mode` is the mode that was unfolded.
fn fold<T: Float>(mat: &Tensor<T>, shape: &[usize], mode: usize) -> Result<Tensor<T>> {
    let ndim = shape.len();
    if mode >= ndim {
        return Err(CoreError::AxisOutOfBounds { axis: mode, ndim });
    }

    let numel: usize = shape.iter().product();
    let mut result = vec![T::zero(); numel];
    let result_strides = compute_strides(shape);

    let rows = shape[mode];
    let cols: usize = shape
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != mode)
        .map(|(_, &d)| d)
        .product();

    let other_dims: Vec<usize> = (0..ndim).filter(|&i| i != mode).collect();

    let mut multi_idx = vec![0usize; ndim];
    for (flat, result_val) in result.iter_mut().enumerate() {
        // Compute multi-index from flat index using result strides
        let mut rem = flat;
        for d in 0..ndim {
            multi_idx[d] = rem / result_strides[d];
            rem %= result_strides[d];
        }

        let row = multi_idx[mode];
        let mut col = 0;
        let mut col_stride = 1;
        for &d in other_dims.iter().rev() {
            col += multi_idx[d] * col_stride;
            col_stride *= shape[d];
        }

        if row < rows && col < cols {
            *result_val = mat.as_slice()[row * cols + col];
        }
    }

    Tensor::from_vec(result, shape.to_vec())
}

/// Compute strides for a given shape (row-major).
fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len()).rev().skip(1) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Mode-n product: multiply tensor `x` by matrix `a` along mode `n`.
///
/// If `x` has shape `[I_0, ..., I_n, ..., I_{N-1}]` and `a` has shape
/// `[J, I_n]`, the result has shape `[I_0, ..., J, ..., I_{N-1}]`.
pub fn mode_n_product<T: Float>(x: &Tensor<T>, a: &Tensor<T>, mode: usize) -> Result<Tensor<T>> {
    if a.ndim() != 2 {
        return Err(CoreError::InvalidArgument {
            reason: "mode_n_product: matrix must be 2-D",
        });
    }
    if mode >= x.ndim() {
        return Err(CoreError::AxisOutOfBounds {
            axis: mode,
            ndim: x.ndim(),
        });
    }
    if a.shape()[1] != x.shape()[mode] {
        return Err(CoreError::DimensionMismatch {
            expected: vec![a.shape()[0], x.shape()[mode]],
            got: a.shape().to_vec(),
        });
    }

    // Unfold X along mode, multiply A * X_(mode), then refold
    let x_unf = unfold(x, mode)?;
    let y_unf = a.matmul(&x_unf)?;

    let mut new_shape = x.shape().to_vec();
    new_shape[mode] = a.shape()[0];

    fold(&y_unf, &new_shape, mode)
}

/// Khatri-Rao product (column-wise Kronecker product) of two 2-D tensors.
///
/// Given `A` with shape `[I, R]` and `B` with shape `[J, R]`, the result
/// has shape `[I*J, R]` where each column is the Kronecker product of the
/// corresponding columns of A and B.
pub fn khatri_rao<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>> {
    if a.ndim() != 2 || b.ndim() != 2 {
        return Err(CoreError::InvalidArgument {
            reason: "khatri_rao: both arguments must be 2-D",
        });
    }
    if a.shape()[1] != b.shape()[1] {
        return Err(CoreError::DimensionMismatch {
            expected: vec![a.shape()[0], a.shape()[1]],
            got: b.shape().to_vec(),
        });
    }

    let ia = a.shape()[0];
    let jb = b.shape()[0];
    let r = a.shape()[1];
    let rows = ia * jb;

    let mut data = vec![T::zero(); rows * r];
    let a_data = a.as_slice();
    let b_data = b.as_slice();

    for col in 0..r {
        for i in 0..ia {
            for j in 0..jb {
                data[(i * jb + j) * r + col] = a_data[i * r + col] * b_data[j * r + col];
            }
        }
    }

    Tensor::from_vec(data, vec![rows, r])
}

/// Hadamard (element-wise) product of two 2-D tensors with the same shape.
fn hadamard<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>> {
    if a.shape() != b.shape() {
        return Err(CoreError::DimensionMismatch {
            expected: a.shape().to_vec(),
            got: b.shape().to_vec(),
        });
    }
    let data: Vec<T> = a
        .as_slice()
        .iter()
        .zip(b.as_slice().iter())
        .map(|(&x, &y)| x * y)
        .collect();
    Tensor::from_vec(data, a.shape().to_vec())
}

/// Compute A^T * A for a 2-D matrix.
fn ata<T: Float>(a: &Tensor<T>) -> Result<Tensor<T>> {
    let at = a.transpose()?;
    at.matmul(a)
}

/// Simple pseudoinverse via SVD for solving least-squares: pinv(A) = V * S^{-1} * U^T
/// Only inverts singular values above a tolerance.
fn pinv_solve<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>> {
    // Solve A * X = B using SVD-based pseudoinverse
    // A is [m, n], B is [m, p], result is [n, p]
    let svd = SvdDecomposition::decompose(a)?;
    let u = svd.u(); // [m, m]
    let vt = svd.vt(); // [n, n]
    let s = svd.singular_values();
    let _m = a.shape()[0];
    let n = a.shape()[1];
    let p = b.shape()[1];

    // Compute U^T * B (truncated to min(m,n) rows)
    let ut = u.transpose()?;
    let utb = ut.matmul(b)?; // [m, p]

    let k = s.len(); // min(m, n)
    let tol = T::epsilon() * T::from_f64(100.0) * s[0]; // relative tolerance

    // Compute S^{-1} * (U^T * B)[0..k, :]
    let mut sinv_utb = vec![T::zero(); n * p];
    for i in 0..k {
        if s[i] > tol {
            let inv_si = T::one() / s[i];
            for j in 0..p {
                sinv_utb[i * p + j] = inv_si * utb.as_slice()[i * p + j];
            }
        }
    }
    let sinv_utb_tensor = Tensor::from_vec(sinv_utb, vec![n, p])?;

    // Compute V * (S^{-1} * U^T * B)
    let v = vt.transpose()?; // [n, n]
    v.matmul(&sinv_utb_tensor)
}

// ======================================================================
// CP Decomposition
// ======================================================================

/// Result of a CP (CANDECOMP/PARAFAC) decomposition.
///
/// A tensor X is approximated as a sum of R rank-one components:
/// X ≈ Σ_r w_r * a_r ⊗ b_r ⊗ c_r ⊗ ...
///
/// where `w_r` are scalar weights and each factor matrix column
/// represents one component along that mode.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct CpDecomposition<T: Float> {
    /// Weights for each component (length R).
    weights: Vec<T>,
    /// Factor matrices, one per mode. Factor `n` has shape `[dim_n, R]`.
    factors: Vec<Tensor<T>>,
}

#[allow(clippy::many_single_char_names)]
impl<T: Float> CpDecomposition<T> {
    /// Decompose tensor `x` into a rank-R CP decomposition using Alternating
    /// Least Squares (ALS).
    ///
    /// # Arguments
    /// - `x`: input tensor (N-D, N >= 2)
    /// - `rank`: number of components R
    /// - `max_iter`: maximum number of ALS iterations
    /// - `tol`: convergence tolerance on relative reconstruction error change
    ///
    /// # Errors
    /// Returns an error if `rank` is zero or the tensor has fewer than 2 dimensions.
    pub fn decompose(x: &Tensor<T>, rank: usize, max_iter: usize, tol: T) -> Result<Self> {
        let ndim = x.ndim();
        if ndim < 2 {
            return Err(CoreError::InvalidArgument {
                reason: "CP decomposition requires at least a 2-D tensor",
            });
        }
        if rank == 0 {
            return Err(CoreError::InvalidArgument {
                reason: "CP decomposition rank must be positive",
            });
        }

        let shape = x.shape().to_vec();

        // Initialize factor matrices using SVD of mode-n unfoldings
        // (HOSVD-based initialization for better convergence)
        let mut factors: Vec<Tensor<T>> = Vec::with_capacity(ndim);
        for (mode, &dim) in shape.iter().enumerate() {
            let x_n = unfold(x, mode)?;
            let r_use = rank.min(dim);
            if let Ok(svd) = SvdDecomposition::decompose(&x_n) {
                let u = svd.u(); // [dim, dim]
                let u_data = u.as_slice();
                let u_cols = u.shape()[1];
                let mut data = vec![T::zero(); dim * rank];
                for i in 0..dim {
                    for r in 0..r_use {
                        data[i * rank + r] = u_data[i * u_cols + r];
                    }
                    // Fill remaining columns with small perturbations
                    for r in r_use..rank {
                        data[i * rank + r] = T::from_f64(
                            (i + 1) as f64 * 0.01
                                + (r + 1) as f64 * 0.007
                                + (mode + 1) as f64 * 0.003,
                        );
                    }
                }
                factors.push(Tensor::from_vec(data, vec![dim, rank])?);
            } else {
                // Fallback: diverse deterministic initialization
                let mut data = vec![T::zero(); dim * rank];
                for i in 0..dim {
                    for r in 0..rank {
                        let val = T::from_f64(
                            ((i + 1) as f64).sqrt() / (dim as f64)
                                + ((r * 7 + 3) as f64).sin() * 0.5,
                        );
                        data[i * rank + r] = val;
                    }
                }
                factors.push(Tensor::from_vec(data, vec![dim, rank])?);
            }
        }

        let mut prev_error = T::from_f64(f64::MAX);

        for _iter in 0..max_iter {
            // Update each factor matrix
            for n in 0..ndim {
                // Compute the Khatri-Rao product of all factors except n
                // and the Hadamard product of all (F_i^T F_i) except n
                let mut v_mat = all_ones_matrix(rank)?;
                let mut kr: Option<Tensor<T>> = None;

                // Build Khatri-Rao and V in reverse order (excluding mode n)
                for k in (0..ndim).rev() {
                    if k == n {
                        continue;
                    }
                    let fk = &factors[k];
                    let ftf = ata(fk)?;
                    v_mat = hadamard(&v_mat, &ftf)?;

                    kr = Some(match kr {
                        None => fk.clone(),
                        Some(prev) => khatri_rao(fk, &prev)?,
                    });
                }

                let kr = kr.ok_or(CoreError::InvalidArgument {
                    reason: "CP decomposition: need at least 2 modes",
                })?;

                // Unfold X along mode n
                let x_n = unfold(x, n)?;

                // A_n = X_(n) * KR * pinv(V)
                let x_kr = x_n.matmul(&kr)?; // [dim_n, R]
                let new_factor =
                    pinv_solve(&v_mat.transpose()?, &x_kr.transpose()?)?.transpose()?;

                factors[n] = new_factor;
            }

            // Check convergence via reconstruction error
            // During iterations, weights are implicitly 1 (absorbed into factors)
            let decomp = CpDecomposition {
                weights: vec![T::one(); rank],
                factors: factors.clone(),
            };
            let recon = decomp.reconstruct()?;
            let mut error = T::zero();
            for (&a, &b) in x.as_slice().iter().zip(recon.as_slice().iter()) {
                let d = a - b;
                error += d * d;
            }
            error = error.sqrt();

            if (prev_error - error).abs() < tol {
                break;
            }
            prev_error = error;
        }

        // Extract final weights
        let weights = extract_weights(&mut factors, &shape, rank)?;

        Ok(CpDecomposition { weights, factors })
    }

    /// Reconstruct the full tensor from the CP decomposition.
    pub fn reconstruct(&self) -> Result<Tensor<T>> {
        if self.factors.is_empty() {
            return Err(CoreError::InvalidArgument {
                reason: "CP decomposition has no factors",
            });
        }

        let rank = self.weights.len();
        let shape: Vec<usize> = self.factors.iter().map(|f| f.shape()[0]).collect();
        let numel: usize = shape.iter().product();
        let ndim = shape.len();
        let strides = compute_strides(&shape);

        let mut data = vec![T::zero(); numel];

        for r in 0..rank {
            let w = self.weights[r];
            // For each element, multiply the corresponding factor entries
            let mut multi_idx = vec![0usize; ndim];
            for (flat, data_val) in data.iter_mut().enumerate() {
                // Compute multi-index
                let mut rem = flat;
                for d in 0..ndim {
                    multi_idx[d] = rem / strides[d];
                    rem %= strides[d];
                }

                let mut val = w;
                for (n, &idx) in multi_idx.iter().enumerate() {
                    val *= self.factors[n].as_slice()[idx * rank + r];
                }
                *data_val += val;
            }
        }

        Tensor::from_vec(data, shape)
    }

    /// Return the factor matrices.
    pub fn factors(&self) -> &[Tensor<T>] {
        &self.factors
    }

    /// Return the weights.
    pub fn weights(&self) -> &[T] {
        &self.weights
    }
}

/// Create an R x R matrix of all ones.
fn all_ones_matrix<T: Float>(r: usize) -> Result<Tensor<T>> {
    Tensor::from_vec(vec![T::one(); r * r], vec![r, r])
}

/// Extract column norms as weights and normalize factor columns in-place.
/// Returns the extracted weights vector.
fn extract_weights<T: Float>(
    factors: &mut [Tensor<T>],
    shape: &[usize],
    rank: usize,
) -> Result<Vec<T>> {
    let ndim = factors.len();
    let mut weights = vec![T::one(); rank];
    for n in 0..ndim {
        let dim = shape[n];
        let f_data = factors[n].as_slice();
        let mut new_data = vec![T::zero(); dim * rank];
        for r in 0..rank {
            let mut col_norm = T::zero();
            for i in 0..dim {
                col_norm += f_data[i * rank + r] * f_data[i * rank + r];
            }
            col_norm = col_norm.sqrt();
            if col_norm > T::epsilon() {
                for i in 0..dim {
                    new_data[i * rank + r] = f_data[i * rank + r] / col_norm;
                }
                weights[r] *= col_norm;
            } else {
                for i in 0..dim {
                    new_data[i * rank + r] = f_data[i * rank + r];
                }
            }
        }
        factors[n] = Tensor::from_vec(new_data, vec![dim, rank])?;
    }
    Ok(weights)
}

// ======================================================================
// Tucker Decomposition (HOSVD)
// ======================================================================

/// Result of a Tucker decomposition (Higher-Order SVD).
///
/// A tensor X is decomposed as:
/// X ≈ G ×_1 U_1 ×_2 U_2 ... ×_N U_N
///
/// where G is the core tensor and U_n are orthogonal factor matrices.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct TuckerDecomposition<T: Float> {
    /// Core tensor with shape equal to `ranks`.
    core: Tensor<T>,
    /// Factor matrices, one per mode. Factor `n` has shape `[dim_n, rank_n]`.
    factors: Vec<Tensor<T>>,
}

#[allow(clippy::many_single_char_names)]
impl<T: Float> TuckerDecomposition<T> {
    /// Decompose tensor `x` using truncated HOSVD.
    ///
    /// `ranks` specifies the target rank for each mode. Its length must
    /// equal `x.ndim()`.
    ///
    /// # Errors
    /// Returns an error if `ranks.len() != x.ndim()` or any rank exceeds
    /// the corresponding dimension.
    pub fn decompose(x: &Tensor<T>, ranks: &[usize]) -> Result<Self> {
        let ndim = x.ndim();
        if ndim < 2 {
            return Err(CoreError::InvalidArgument {
                reason: "Tucker decomposition requires at least a 2-D tensor",
            });
        }
        if ranks.len() != ndim {
            return Err(CoreError::DimensionMismatch {
                expected: x.shape().to_vec(),
                got: ranks.to_vec(),
            });
        }

        let shape = x.shape();
        for (n, &r) in ranks.iter().enumerate() {
            if r == 0 || r > shape[n] {
                return Err(CoreError::InvalidArgument {
                    reason: "Tucker rank must be in [1, dim_n] for each mode",
                });
            }
        }

        // Step 1: For each mode, compute SVD of the mode-n unfolding
        // and take the first ranks[n] left singular vectors.
        let mut factors: Vec<Tensor<T>> = Vec::with_capacity(ndim);
        for n in 0..ndim {
            let x_n = unfold(x, n)?; // [shape[n], prod(other dims)]
            let svd = SvdDecomposition::decompose(&x_n)?;
            let u_full = svd.u(); // [shape[n], shape[n]]

            // Truncate to ranks[n] columns
            let dim_n = shape[n];
            let rn = ranks[n];
            let mut u_data = vec![T::zero(); dim_n * rn];
            for i in 0..dim_n {
                for j in 0..rn {
                    u_data[i * rn + j] = u_full.as_slice()[i * dim_n + j];
                }
            }
            factors.push(Tensor::from_vec(u_data, vec![dim_n, rn])?);
        }

        // Step 2: Core tensor = X ×_1 U_1^T ×_2 U_2^T ... ×_N U_N^T
        let mut core = x.clone();
        for (n, factor) in factors.iter().enumerate() {
            let ut = factor.transpose()?; // [rank_n, dim_n]
            core = mode_n_product(&core, &ut, n)?;
        }

        Ok(TuckerDecomposition { core, factors })
    }

    /// Reconstruct the full tensor from the Tucker decomposition.
    pub fn reconstruct(&self) -> Result<Tensor<T>> {
        let mut result = self.core.clone();
        for (n, factor) in self.factors.iter().enumerate() {
            result = mode_n_product(&result, factor, n)?;
        }
        Ok(result)
    }

    /// Return the core tensor.
    pub fn core(&self) -> &Tensor<T> {
        &self.core
    }

    /// Return the factor matrices.
    pub fn factors(&self) -> &[Tensor<T>] {
        &self.factors
    }
}

// ======================================================================
// Non-negative Tensor Factorization (NTF)
// ======================================================================

/// Result of a Non-negative Tensor Factorization (NTF).
///
/// Similar to CP decomposition but with the constraint that all factor
/// values and weights are non-negative. Uses multiplicative update rules.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct NtfDecomposition<T: Float> {
    /// Weights for each component (length R).
    weights: Vec<T>,
    /// Factor matrices, one per mode. Factor `n` has shape `[dim_n, R]`.
    factors: Vec<Tensor<T>>,
}

#[allow(clippy::many_single_char_names)]
impl<T: Float> NtfDecomposition<T> {
    /// Non-negative CP decomposition using multiplicative updates.
    ///
    /// All elements of `x` must be non-negative.
    ///
    /// # Arguments
    /// - `x`: input non-negative tensor (N-D, N >= 2)
    /// - `rank`: number of components R
    /// - `max_iter`: maximum number of iterations
    /// - `tol`: convergence tolerance on relative error change
    pub fn decompose(x: &Tensor<T>, rank: usize, max_iter: usize, tol: T) -> Result<Self> {
        let ndim = x.ndim();
        if ndim < 2 {
            return Err(CoreError::InvalidArgument {
                reason: "NTF decomposition requires at least a 2-D tensor",
            });
        }
        if rank == 0 {
            return Err(CoreError::InvalidArgument {
                reason: "NTF decomposition rank must be positive",
            });
        }

        // Check non-negativity
        for &val in x.as_slice() {
            if val < T::zero() {
                return Err(CoreError::InvalidArgument {
                    reason: "NTF decomposition requires non-negative tensor",
                });
            }
        }

        let shape = x.shape().to_vec();

        // Initialize factors with positive values
        let mut factors: Vec<Tensor<T>> = Vec::with_capacity(ndim);
        for (mode, &dim) in shape.iter().enumerate() {
            let mut data = vec![T::zero(); dim * rank];
            for i in 0..dim {
                for r in 0..rank {
                    data[i * rank + r] = T::from_f64(
                        1.0 + (i as f64 * 0.1) + (r as f64 * 0.05) + (mode as f64 * 0.01),
                    );
                }
            }
            factors.push(Tensor::from_vec(data, vec![dim, rank])?);
        }

        let eps = T::from_f64(1e-12);
        let mut prev_error = T::from_f64(f64::MAX);

        for _iter in 0..max_iter {
            for n in 0..ndim {
                // Numerator: X_(n) * KR (Khatri-Rao of all factors except n)
                let x_n = unfold(x, n)?;

                let mut kr: Option<Tensor<T>> = None;
                for (k, fk) in factors.iter().enumerate().rev() {
                    if k == n {
                        continue;
                    }
                    kr = Some(match kr {
                        None => fk.clone(),
                        Some(prev) => khatri_rao(fk, &prev)?,
                    });
                }
                let kr = kr.ok_or(CoreError::InvalidArgument {
                    reason: "NTF: need at least 2 modes",
                })?;

                let numerator = x_n.matmul(&kr)?; // [dim_n, R]

                // Denominator: factor_n * (Hadamard of all F_k^T F_k except n)
                let mut v_mat = all_ones_matrix(rank)?;
                for (k, fk) in factors.iter().enumerate() {
                    if k == n {
                        continue;
                    }
                    let ftf = ata(fk)?;
                    v_mat = hadamard(&v_mat, &ftf)?;
                }

                let denominator = factors[n].matmul(&v_mat)?; // [dim_n, R]

                // Multiplicative update
                let dim_n = shape[n];
                let f_data = factors[n].as_slice();
                let num_data = numerator.as_slice();
                let den_data = denominator.as_slice();
                let mut new_data = vec![T::zero(); dim_n * rank];
                for idx in 0..dim_n * rank {
                    new_data[idx] = f_data[idx] * num_data[idx] / (den_data[idx] + eps);
                }
                factors[n] = Tensor::from_vec(new_data, vec![dim_n, rank])?;
            }

            // Check convergence
            let decomp = NtfDecomposition {
                weights: vec![T::one(); rank],
                factors: factors.clone(),
            };
            let recon = decomp.reconstruct()?;
            let mut error = T::zero();
            for (&a, &b) in x.as_slice().iter().zip(recon.as_slice().iter()) {
                let d = a - b;
                error += d * d;
            }
            error = error.sqrt();

            if (prev_error - error).abs() < tol {
                break;
            }
            prev_error = error;
        }

        // Extract weights (column norms)
        let weights = extract_weights(&mut factors, &shape, rank)?;

        Ok(NtfDecomposition { weights, factors })
    }

    /// Reconstruct the full tensor from the NTF decomposition.
    pub fn reconstruct(&self) -> Result<Tensor<T>> {
        if self.factors.is_empty() {
            return Err(CoreError::InvalidArgument {
                reason: "NTF decomposition has no factors",
            });
        }

        let rank = self.weights.len();
        let shape: Vec<usize> = self.factors.iter().map(|f| f.shape()[0]).collect();
        let numel: usize = shape.iter().product();
        let ndim = shape.len();
        let strides = compute_strides(&shape);

        let mut data = vec![T::zero(); numel];

        for r in 0..rank {
            let w = self.weights[r];
            let mut multi_idx = vec![0usize; ndim];
            for (flat, data_val) in data.iter_mut().enumerate() {
                let mut rem = flat;
                for d in 0..ndim {
                    multi_idx[d] = rem / strides[d];
                    rem %= strides[d];
                }

                let mut val = w;
                for (n, &idx) in multi_idx.iter().enumerate() {
                    val *= self.factors[n].as_slice()[idx * rank + r];
                }
                *data_val += val;
            }
        }

        Tensor::from_vec(data, shape)
    }

    /// Return the factor matrices.
    pub fn factors(&self) -> &[Tensor<T>] {
        &self.factors
    }

    /// Return the weights.
    pub fn weights(&self) -> &[T] {
        &self.weights
    }
}

// ======================================================================
// Tests
// ======================================================================

#[cfg(test)]
#[allow(clippy::float_cmp, clippy::many_single_char_names)]
mod tests {
    use super::*;

    fn approx_eq_tensor(a: &Tensor<f64>, b: &Tensor<f64>, tol: f64) -> bool {
        if a.shape() != b.shape() {
            return false;
        }
        a.as_slice()
            .iter()
            .zip(b.as_slice().iter())
            .all(|(&x, &y)| (x - y).abs() < tol)
    }

    fn reconstruction_error(orig: &Tensor<f64>, recon: &Tensor<f64>) -> f64 {
        orig.as_slice()
            .iter()
            .zip(recon.as_slice().iter())
            .map(|(&a, &b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt()
    }

    fn tensor_norm(x: &Tensor<f64>) -> f64 {
        x.as_slice().iter().map(|&v| v * v).sum::<f64>().sqrt()
    }

    // Test 1: CP decomposition of rank-1 tensor recovers exact decomposition
    #[test]
    fn test_cp_rank1_exact() {
        // Create a rank-1 tensor: outer product of [1, 2] x [3, 4, 5] x [6, 7]
        let a = [1.0, 2.0];
        let b = [3.0, 4.0, 5.0];
        let c = [6.0, 7.0];

        let mut data = vec![0.0; 2 * 3 * 2];
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..2 {
                    data[i * 6 + j * 2 + k] = a[i] * b[j] * c[k];
                }
            }
        }
        let x = Tensor::from_vec(data, vec![2, 3, 2]).unwrap();

        let cp = CpDecomposition::decompose(&x, 1, 100, 1e-10).unwrap();
        let recon = cp.reconstruct().unwrap();

        let err = reconstruction_error(&x, &recon);
        let norm = tensor_norm(&x);
        assert!(
            err / norm < 1e-6,
            "rank-1 CP reconstruction error too large: {err}"
        );
    }

    // Test 2: CP decomposition of random 3D tensor reduces reconstruction error
    #[test]
    fn test_cp_reduces_error() {
        let data: Vec<f64> = (0..24).map(|i| (f64::from(i) + 1.0) * 0.5).collect();
        let x = Tensor::from_vec(data, vec![2, 3, 4]).unwrap();

        let cp = CpDecomposition::decompose(&x, 3, 50, 1e-12).unwrap();
        let recon = cp.reconstruct().unwrap();

        let err = reconstruction_error(&x, &recon);
        let norm = tensor_norm(&x);
        // With rank 3 on a 2x3x4 tensor, we should get reasonable approximation
        assert!(
            err / norm < 0.5,
            "CP reconstruction error too large: {err}/{norm}"
        );
    }

    // Test 3: Tucker HOSVD with full ranks is near-lossless
    #[test]
    fn test_tucker_full_ranks_lossless() {
        let data: Vec<f64> = (0..24).map(|i| f64::from(i) + 1.0).collect();
        let x = Tensor::from_vec(data, vec![2, 3, 4]).unwrap();

        let tucker = TuckerDecomposition::decompose(&x, &[2, 3, 4]).unwrap();
        let recon = tucker.reconstruct().unwrap();

        let err = reconstruction_error(&x, &recon);
        assert!(err < 1e-8, "Tucker full-rank reconstruction error: {err}");
    }

    // Test 4: Tucker with reduced ranks approximates well
    #[test]
    fn test_tucker_reduced_ranks() {
        let data: Vec<f64> = (0..24).map(|i| f64::from(i) + 1.0).collect();
        let x = Tensor::from_vec(data, vec![2, 3, 4]).unwrap();

        let tucker = TuckerDecomposition::decompose(&x, &[2, 2, 3]).unwrap();
        let recon = tucker.reconstruct().unwrap();

        // Core should have reduced shape
        assert_eq!(tucker.core().shape(), &[2, 2, 3]);

        let err = reconstruction_error(&x, &recon);
        let norm = tensor_norm(&x);
        // Reduced rank approximation should still be decent
        assert!(
            err / norm < 0.5,
            "Tucker reduced-rank error too large: {err}/{norm}"
        );
    }

    // Test 5: NTF of non-negative tensor
    #[test]
    fn test_ntf_nonnegative() {
        let data: Vec<f64> = (0..12).map(|i| f64::from(i) + 1.0).collect();
        let x = Tensor::from_vec(data, vec![2, 3, 2]).unwrap();

        let ntf = NtfDecomposition::decompose(&x, 2, 100, 1e-10).unwrap();
        let recon = ntf.reconstruct().unwrap();

        // All reconstructed values should be non-negative
        for &v in recon.as_slice() {
            assert!(v >= 0.0, "NTF reconstruction has negative value: {v}");
        }

        // All factor values should be non-negative
        for factor in ntf.factors() {
            for &v in factor.as_slice() {
                assert!(v >= 0.0, "NTF factor has negative value: {v}");
            }
        }

        // Weights should be non-negative
        for &w in ntf.weights() {
            assert!(w >= 0.0, "NTF weight is negative: {w}");
        }
    }

    // Test 6: Mode-n unfolding and refolding is identity
    #[test]
    fn test_unfold_fold_identity() {
        let data: Vec<f64> = (0..24).map(f64::from).collect();
        let x = Tensor::from_vec(data, vec![2, 3, 4]).unwrap();

        for mode in 0..3 {
            let unfolded = unfold(&x, mode).unwrap();

            // Check unfolded shape
            let expected_rows = x.shape()[mode];
            let expected_cols: usize = x
                .shape()
                .iter()
                .enumerate()
                .filter(|&(i, _)| i != mode)
                .map(|(_, &d)| d)
                .product();
            assert_eq!(
                unfolded.shape(),
                &[expected_rows, expected_cols],
                "unfolded shape mismatch for mode {mode}"
            );

            let refolded = fold(&unfolded, x.shape(), mode).unwrap();
            assert!(
                approx_eq_tensor(&x, &refolded, 1e-12),
                "unfold/fold round-trip failed for mode {mode}"
            );
        }
    }

    // Test 7: Khatri-Rao product shape check
    #[test]
    fn test_khatri_rao_shape() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]).unwrap();
        let b = Tensor::from_vec(vec![7.0, 8.0, 9.0, 10.0], vec![2, 2]).unwrap();

        let kr = khatri_rao(&a, &b).unwrap();
        assert_eq!(kr.shape(), &[6, 2]); // 3*2 = 6 rows, 2 columns

        // Column mismatch should error
        let c = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        assert!(khatri_rao(&a, &c).is_err());
    }

    // Test 8: Error on rank > dimension (Tucker)
    #[test]
    fn test_tucker_rank_too_large() {
        let data: Vec<f64> = (0..12).map(f64::from).collect();
        let x = Tensor::from_vec(data, vec![2, 3, 2]).unwrap();

        // Rank 5 > dim 3 on mode 1
        let result = TuckerDecomposition::decompose(&x, &[2, 5, 2]);
        assert!(result.is_err());
    }

    // Test 9: Mode-n product with identity gives same tensor
    #[test]
    fn test_mode_n_product_identity() {
        let data: Vec<f64> = (0..24).map(|i| f64::from(i) + 1.0).collect();
        let x = Tensor::from_vec(data, vec![2, 3, 4]).unwrap();

        for mode in 0..3 {
            let dim = x.shape()[mode];
            let eye = Tensor::<f64>::eye(dim);
            let result = mode_n_product(&x, &eye, mode).unwrap();
            assert!(
                approx_eq_tensor(&x, &result, 1e-10),
                "mode-{mode} product with identity failed"
            );
        }
    }

    // Test 10: CP with max_iter=0 returns initial guess
    #[test]
    fn test_cp_zero_iterations() {
        let data: Vec<f64> = (0..12).map(|i| f64::from(i) + 1.0).collect();
        let x = Tensor::from_vec(data, vec![2, 3, 2]).unwrap();

        let cp = CpDecomposition::decompose(&x, 2, 0, 1e-10).unwrap();
        // Should succeed and return some decomposition (the initial guess)
        assert_eq!(cp.factors().len(), 3);
        assert_eq!(cp.weights().len(), 2);
        assert_eq!(cp.factors()[0].shape(), &[2, 2]);
        assert_eq!(cp.factors()[1].shape(), &[3, 2]);
        assert_eq!(cp.factors()[2].shape(), &[2, 2]);
    }
}
