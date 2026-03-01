//! BLAS Level 1–3 operations on [`Tensor`].
//!
//! All functions operate on tensors and validate shapes, returning
//! [`Result`] on dimension mismatches.

use crate::error::{CoreError, Result};
use crate::tensor::Tensor;
use crate::{Float, Scalar};

// ======================================================================
// BLAS Level 1 — vector operations, O(n)
// ======================================================================

/// Inner (dot) product of two 1-D tensors: `sum(x_i * y_i)`.
///
/// Both tensors must be 1-D with the same length.
///
/// ```
/// # use scivex_core::tensor::Tensor;
/// # use scivex_core::linalg::dot;
/// let x = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0], vec![3]).unwrap();
/// let y = Tensor::from_vec(vec![4.0_f64, 5.0, 6.0], vec![3]).unwrap();
/// let d = dot(&x, &y).unwrap();
/// assert!((d - 32.0).abs() < 1e-10);
/// ```
pub fn dot<T: Scalar>(x: &Tensor<T>, y: &Tensor<T>) -> Result<T> {
    check_vectors(x, y, "dot")?;
    let result = x
        .as_slice()
        .iter()
        .zip(y.as_slice().iter())
        .fold(T::zero(), |acc, (&a, &b)| acc + a * b);
    Ok(result)
}

/// `y = alpha * x + y` (in-place update of `y`).
///
/// Both tensors must be 1-D with the same length.
///
/// ```
/// # use scivex_core::tensor::Tensor;
/// # use scivex_core::linalg::axpy;
/// let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
/// let mut y = Tensor::from_vec(vec![10.0, 20.0, 30.0], vec![3]).unwrap();
/// axpy(2.0, &x, &mut y).unwrap();
/// assert_eq!(y.as_slice(), &[12.0, 24.0, 36.0]);
/// ```
pub fn axpy<T: Scalar>(alpha: T, x: &Tensor<T>, y: &mut Tensor<T>) -> Result<()> {
    check_vectors(x, y, "axpy")?;
    let xs = x.as_slice();
    let ys = y.as_mut_slice();
    for (yi, &xi) in ys.iter_mut().zip(xs.iter()) {
        *yi += alpha * xi;
    }
    Ok(())
}

/// Euclidean norm (L2 norm) of a 1-D tensor: `sqrt(sum(x_i^2))`.
///
/// ```
/// # use scivex_core::tensor::Tensor;
/// # use scivex_core::linalg::nrm2;
/// let x = Tensor::from_vec(vec![3.0_f64, 4.0], vec![2]).unwrap();
/// let n = nrm2(&x).unwrap();
/// assert!((n - 5.0).abs() < 1e-10);
/// ```
pub fn nrm2<T: Float>(x: &Tensor<T>) -> Result<T> {
    check_vector(x, "nrm2")?;
    let sum_sq = x.as_slice().iter().fold(T::zero(), |acc, &v| acc + v * v);
    Ok(sum_sq.sqrt())
}

/// Sum of absolute values (L1 norm) of a 1-D tensor: `sum(|x_i|)`.
///
/// ```
/// # use scivex_core::tensor::Tensor;
/// # use scivex_core::linalg::asum;
/// let x = Tensor::from_vec(vec![-1.0_f64, 2.0, -3.0], vec![3]).unwrap();
/// let s = asum(&x).unwrap();
/// assert!((s - 6.0).abs() < 1e-10);
/// ```
pub fn asum<T: Float>(x: &Tensor<T>) -> Result<T> {
    check_vector(x, "asum")?;
    let result = x.as_slice().iter().fold(T::zero(), |acc, &v| acc + v.abs());
    Ok(result)
}

/// Scale a vector in place: `x = alpha * x`.
///
/// ```
/// # use scivex_core::tensor::Tensor;
/// # use scivex_core::linalg::scal;
/// let mut x = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
/// scal(10.0, &mut x).unwrap();
/// assert_eq!(x.as_slice(), &[10.0, 20.0, 30.0]);
/// ```
pub fn scal<T: Scalar>(alpha: T, x: &mut Tensor<T>) -> Result<()> {
    check_vector(x, "scal")?;
    for v in x.as_mut_slice() {
        *v *= alpha;
    }
    Ok(())
}

/// Index of the element with the largest absolute value.
///
/// Returns `None` for empty tensors.
///
/// ```
/// # use scivex_core::tensor::Tensor;
/// # use scivex_core::linalg::iamax;
/// let x = Tensor::from_vec(vec![1.0_f64, -5.0, 3.0], vec![3]).unwrap();
/// assert_eq!(iamax(&x).unwrap(), Some(1));
/// ```
pub fn iamax<T: Float>(x: &Tensor<T>) -> Result<Option<usize>> {
    check_vector(x, "iamax")?;
    if x.is_empty() {
        return Ok(None);
    }
    let mut max_idx = 0;
    let mut max_val = x.as_slice()[0].abs();
    for (i, &v) in x.as_slice().iter().enumerate().skip(1) {
        let av = v.abs();
        if av > max_val {
            max_val = av;
            max_idx = i;
        }
    }
    Ok(Some(max_idx))
}

// ======================================================================
// BLAS Level 2 — matrix-vector operations, O(n^2)
// ======================================================================

/// General matrix-vector multiply: `y = alpha * A * x + beta * y`.
///
/// - `a` must be 2-D with shape `[m, n]`.
/// - `x` must be 1-D with length `n`.
/// - `y` must be 1-D with length `m`.
///
/// If `beta` is zero, `y` is overwritten (not read).
///
/// ```
/// # use scivex_core::tensor::Tensor;
/// # use scivex_core::linalg::gemv;
/// // A = [[1, 2], [3, 4]], x = [5, 6]
/// let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
/// let x = Tensor::from_vec(vec![5.0, 6.0], vec![2]).unwrap();
/// let mut y = Tensor::<f64>::zeros(vec![2]);
/// gemv(1.0, &a, &x, 0.0, &mut y).unwrap();
/// assert_eq!(y.as_slice(), &[17.0, 39.0]);
/// ```
#[allow(clippy::many_single_char_names)]
pub fn gemv<T: Scalar>(
    alpha: T,
    a: &Tensor<T>,
    x: &Tensor<T>,
    beta: T,
    y: &mut Tensor<T>,
) -> Result<()> {
    if a.ndim() != 2 {
        return Err(CoreError::InvalidArgument {
            reason: "gemv: `a` must be a 2-D tensor (matrix)",
        });
    }
    if x.ndim() != 1 {
        return Err(CoreError::InvalidArgument {
            reason: "gemv: `x` must be a 1-D tensor (vector)",
        });
    }
    if y.ndim() != 1 {
        return Err(CoreError::InvalidArgument {
            reason: "gemv: `y` must be a 1-D tensor (vector)",
        });
    }

    let m = a.shape()[0];
    let n = a.shape()[1];

    if x.numel() != n {
        return Err(CoreError::DimensionMismatch {
            expected: vec![n],
            got: x.shape().to_vec(),
        });
    }
    if y.numel() != m {
        return Err(CoreError::DimensionMismatch {
            expected: vec![m],
            got: y.shape().to_vec(),
        });
    }

    let a_data = a.as_slice();
    let x_data = x.as_slice();
    let y_data = y.as_mut_slice();

    for (i, yi) in y_data.iter_mut().enumerate().take(m) {
        let mut sum = T::zero();
        let row_offset = i * n;
        for j in 0..n {
            sum += a_data[row_offset + j] * x_data[j];
        }
        *yi = alpha * sum + beta * *yi;
    }

    Ok(())
}

// ======================================================================
// BLAS Level 3 — matrix-matrix operations, O(n^3)
// ======================================================================

/// General matrix-matrix multiply: `C = alpha * A * B + beta * C`.
///
/// - `a` must be 2-D with shape `[m, k]`.
/// - `b` must be 2-D with shape `[k, n]`.
/// - `c` must be 2-D with shape `[m, n]`.
///
/// If `beta` is zero, `c` is overwritten (not read).
///
/// ```
/// # use scivex_core::tensor::Tensor;
/// # use scivex_core::linalg::gemm;
/// let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
/// let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
/// let mut c = Tensor::<f64>::zeros(vec![2, 2]);
/// gemm(1.0, &a, &b, 0.0, &mut c).unwrap();
/// assert_eq!(c.as_slice(), &[19.0, 22.0, 43.0, 50.0]);
/// ```
#[allow(clippy::many_single_char_names)]
pub fn gemm<T: Scalar>(
    alpha: T,
    a: &Tensor<T>,
    b: &Tensor<T>,
    beta: T,
    c: &mut Tensor<T>,
) -> Result<()> {
    if a.ndim() != 2 || b.ndim() != 2 || c.ndim() != 2 {
        return Err(CoreError::InvalidArgument {
            reason: "gemm: all arguments must be 2-D tensors (matrices)",
        });
    }

    let m = a.shape()[0];
    let k = a.shape()[1];
    let n = b.shape()[1];

    if b.shape()[0] != k {
        return Err(CoreError::DimensionMismatch {
            expected: vec![k, n],
            got: b.shape().to_vec(),
        });
    }
    if c.shape()[0] != m || c.shape()[1] != n {
        return Err(CoreError::DimensionMismatch {
            expected: vec![m, n],
            got: c.shape().to_vec(),
        });
    }

    let a_data = a.as_slice();
    let b_data = b.as_slice();
    let c_data = c.as_mut_slice();

    // ijk loop order (row-major friendly for A and C)
    for i in 0..m {
        for j in 0..n {
            let mut sum = T::zero();
            let a_row = i * k;
            for p in 0..k {
                sum += a_data[a_row + p] * b_data[p * n + j];
            }
            let c_idx = i * n + j;
            c_data[c_idx] = alpha * sum + beta * c_data[c_idx];
        }
    }

    Ok(())
}

// ======================================================================
// Convenience methods on Tensor
// ======================================================================

impl<T: Scalar> Tensor<T> {
    /// Matrix-vector multiply: returns `A @ x` as a new 1-D tensor.
    ///
    /// `self` must be 2-D `[m, n]`, `x` must be 1-D `[n]`.
    pub fn matvec(&self, x: &Tensor<T>) -> Result<Tensor<T>> {
        let m = self.shape().first().copied().unwrap_or(0);
        let mut y = Tensor::zeros(vec![m]);
        gemv(T::one(), self, x, T::zero(), &mut y)?;
        Ok(y)
    }

    /// Matrix-matrix multiply: returns `self @ other` as a new 2-D tensor.
    ///
    /// `self` must be 2-D `[m, k]`, `other` must be 2-D `[k, n]`.
    pub fn matmul(&self, other: &Tensor<T>) -> Result<Tensor<T>> {
        let m = self.shape().first().copied().unwrap_or(0);
        let n = other.shape().get(1).copied().unwrap_or(0);
        let mut c = Tensor::zeros(vec![m, n]);
        gemm(T::one(), self, other, T::zero(), &mut c)?;
        Ok(c)
    }

    /// Dot product with another 1-D tensor.
    pub fn dot(&self, other: &Tensor<T>) -> Result<T> {
        dot(self, other)
    }
}

impl<T: Float> Tensor<T> {
    /// Euclidean (L2) norm of a 1-D tensor.
    pub fn norm(&self) -> Result<T> {
        nrm2(self)
    }

    /// Solve the linear system `self * x = b` for a square matrix `self`.
    ///
    /// Uses LU decomposition with partial pivoting.
    pub fn solve(&self, b: &Tensor<T>) -> Result<Tensor<T>> {
        crate::linalg::solve(self, b)
    }

    /// Compute the inverse of a square matrix.
    ///
    /// Uses LU decomposition with partial pivoting.
    pub fn inv(&self) -> Result<Tensor<T>> {
        crate::linalg::inv(self)
    }

    /// Compute the determinant of a square matrix.
    ///
    /// Uses LU decomposition with partial pivoting.
    pub fn det(&self) -> Result<T> {
        crate::linalg::det(self)
    }

    /// Solve the least-squares problem `min ||self * x - b||_2`.
    ///
    /// Uses QR decomposition with Householder reflections.
    pub fn lstsq(&self, b: &Tensor<T>) -> Result<Tensor<T>> {
        crate::linalg::lstsq(self, b)
    }
}

// ======================================================================
// Internal helpers
// ======================================================================

fn check_vector<T: Scalar>(x: &Tensor<T>, name: &'static str) -> Result<()> {
    if x.ndim() != 1 {
        return Err(CoreError::InvalidArgument {
            reason: match name {
                "nrm2" => "nrm2: expected a 1-D tensor",
                "asum" => "asum: expected a 1-D tensor",
                "scal" => "scal: expected a 1-D tensor",
                "iamax" => "iamax: expected a 1-D tensor",
                _ => "expected a 1-D tensor",
            },
        });
    }
    Ok(())
}

fn check_vectors<T: Scalar>(x: &Tensor<T>, y: &Tensor<T>, name: &'static str) -> Result<()> {
    if x.ndim() != 1 || y.ndim() != 1 {
        return Err(CoreError::InvalidArgument {
            reason: match name {
                "dot" => "dot: both arguments must be 1-D tensors",
                "axpy" => "axpy: both arguments must be 1-D tensors",
                _ => "both arguments must be 1-D tensors",
            },
        });
    }
    if x.numel() != y.numel() {
        return Err(CoreError::DimensionMismatch {
            expected: x.shape().to_vec(),
            got: y.shape().to_vec(),
        });
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    fn vec_f64(data: &[f64]) -> Tensor<f64> {
        Tensor::from_vec(data.to_vec(), vec![data.len()]).unwrap()
    }

    fn mat_f64(data: &[f64], rows: usize, cols: usize) -> Tensor<f64> {
        Tensor::from_vec(data.to_vec(), vec![rows, cols]).unwrap()
    }

    // ------------------------------------------------------------------
    // BLAS L1
    // ------------------------------------------------------------------

    #[test]
    fn test_dot_basic() {
        let x = vec_f64(&[1.0, 2.0, 3.0]);
        let y = vec_f64(&[4.0, 5.0, 6.0]);
        assert_eq!(dot(&x, &y).unwrap(), 32.0);
    }

    #[test]
    fn test_dot_single() {
        let x = vec_f64(&[3.0]);
        let y = vec_f64(&[7.0]);
        assert_eq!(dot(&x, &y).unwrap(), 21.0);
    }

    #[test]
    fn test_dot_length_mismatch() {
        let x = vec_f64(&[1.0, 2.0]);
        let y = vec_f64(&[1.0, 2.0, 3.0]);
        assert!(dot(&x, &y).is_err());
    }

    #[test]
    fn test_dot_not_1d() {
        let x = mat_f64(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let y = vec_f64(&[1.0, 2.0]);
        assert!(dot(&x, &y).is_err());
    }

    #[test]
    fn test_axpy() {
        let x = vec_f64(&[1.0, 2.0, 3.0]);
        let mut y = vec_f64(&[10.0, 20.0, 30.0]);
        axpy(2.0, &x, &mut y).unwrap();
        assert_eq!(y.as_slice(), &[12.0, 24.0, 36.0]);
    }

    #[test]
    fn test_axpy_zero_alpha() {
        let x = vec_f64(&[1.0, 2.0, 3.0]);
        let mut y = vec_f64(&[10.0, 20.0, 30.0]);
        axpy(0.0, &x, &mut y).unwrap();
        assert_eq!(y.as_slice(), &[10.0, 20.0, 30.0]);
    }

    #[test]
    fn test_nrm2() {
        let x = vec_f64(&[3.0, 4.0]);
        assert!((nrm2(&x).unwrap() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_nrm2_single() {
        let x = vec_f64(&[-7.0]);
        assert!((nrm2(&x).unwrap() - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_asum() {
        let x = vec_f64(&[-1.0, 2.0, -3.0, 4.0]);
        assert!((asum(&x).unwrap() - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_scal() {
        let mut x = vec_f64(&[1.0, 2.0, 3.0]);
        scal(10.0, &mut x).unwrap();
        assert_eq!(x.as_slice(), &[10.0, 20.0, 30.0]);
    }

    #[test]
    fn test_scal_zero() {
        let mut x = vec_f64(&[1.0, 2.0, 3.0]);
        scal(0.0, &mut x).unwrap();
        assert_eq!(x.as_slice(), &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_iamax() {
        let x = vec_f64(&[1.0, -5.0, 3.0, -2.0]);
        assert_eq!(iamax(&x).unwrap(), Some(1));
    }

    #[test]
    fn test_iamax_first_is_max() {
        let x = vec_f64(&[100.0, 1.0, 2.0]);
        assert_eq!(iamax(&x).unwrap(), Some(0));
    }

    #[test]
    fn test_iamax_empty() {
        let x = Tensor::<f64>::zeros(vec![0]);
        assert_eq!(iamax(&x).unwrap(), None);
    }

    // ------------------------------------------------------------------
    // BLAS L2
    // ------------------------------------------------------------------

    #[test]
    fn test_gemv_basic() {
        // A = [[1, 2], [3, 4]], x = [5, 6]
        // y = A @ x = [17, 39]
        let a = mat_f64(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let x = vec_f64(&[5.0, 6.0]);
        let mut y = Tensor::<f64>::zeros(vec![2]);
        gemv(1.0, &a, &x, 0.0, &mut y).unwrap();
        assert_eq!(y.as_slice(), &[17.0, 39.0]);
    }

    #[test]
    fn test_gemv_with_alpha_beta() {
        // y = 2 * A @ x + 3 * y
        let a = mat_f64(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let x = vec_f64(&[1.0, 1.0]);
        let mut y = vec_f64(&[10.0, 10.0]);
        gemv(2.0, &a, &x, 3.0, &mut y).unwrap();
        // A @ x = [3, 7], 2*[3,7] + 3*[10,10] = [6+30, 14+30] = [36, 44]
        assert_eq!(y.as_slice(), &[36.0, 44.0]);
    }

    #[test]
    fn test_gemv_rectangular() {
        // A = [[1, 2, 3], [4, 5, 6]]  (2x3)
        // x = [1, 0, 1]  (3)
        // y = A @ x = [4, 10]
        let a = mat_f64(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let x = vec_f64(&[1.0, 0.0, 1.0]);
        let mut y = Tensor::<f64>::zeros(vec![2]);
        gemv(1.0, &a, &x, 0.0, &mut y).unwrap();
        assert_eq!(y.as_slice(), &[4.0, 10.0]);
    }

    #[test]
    fn test_gemv_dimension_mismatch() {
        let a = mat_f64(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let x = vec_f64(&[1.0, 2.0, 3.0]);
        let mut y = Tensor::<f64>::zeros(vec![2]);
        assert!(gemv(1.0, &a, &x, 0.0, &mut y).is_err());
    }

    #[test]
    fn test_gemv_y_dimension_mismatch() {
        let a = mat_f64(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let x = vec_f64(&[1.0, 2.0]);
        let mut y = Tensor::<f64>::zeros(vec![3]);
        assert!(gemv(1.0, &a, &x, 0.0, &mut y).is_err());
    }

    // ------------------------------------------------------------------
    // BLAS L3
    // ------------------------------------------------------------------

    #[test]
    fn test_gemm_square() {
        // A = [[1, 2], [3, 4]]
        // B = [[5, 6], [7, 8]]
        // C = A @ B = [[19, 22], [43, 50]]
        let a = mat_f64(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let b = mat_f64(&[5.0, 6.0, 7.0, 8.0], 2, 2);
        let mut c = Tensor::<f64>::zeros(vec![2, 2]);
        gemm(1.0, &a, &b, 0.0, &mut c).unwrap();
        assert_eq!(c.as_slice(), &[19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_gemm_rectangular() {
        // A (2x3) @ B (3x2) = C (2x2)
        let a = mat_f64(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let b = mat_f64(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], 3, 2);
        let mut c = Tensor::<f64>::zeros(vec![2, 2]);
        gemm(1.0, &a, &b, 0.0, &mut c).unwrap();
        // Row 0: 1*7+2*9+3*11 = 7+18+33 = 58, 1*8+2*10+3*12 = 8+20+36 = 64
        // Row 1: 4*7+5*9+6*11 = 28+45+66 = 139, 4*8+5*10+6*12 = 32+50+72 = 154
        assert_eq!(c.as_slice(), &[58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_gemm_with_alpha_beta() {
        // C = 2 * A @ B + 3 * C
        let a = mat_f64(&[1.0, 0.0, 0.0, 1.0], 2, 2); // identity
        let b = mat_f64(&[5.0, 6.0, 7.0, 8.0], 2, 2);
        let mut c = mat_f64(&[1.0, 1.0, 1.0, 1.0], 2, 2);
        gemm(2.0, &a, &b, 3.0, &mut c).unwrap();
        // 2*B + 3*ones = [10+3, 12+3, 14+3, 16+3] = [13, 15, 17, 19]
        assert_eq!(c.as_slice(), &[13.0, 15.0, 17.0, 19.0]);
    }

    #[test]
    fn test_gemm_identity() {
        let a = mat_f64(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 3, 3);
        let eye = Tensor::<f64>::eye(3);
        let mut c = Tensor::<f64>::zeros(vec![3, 3]);
        gemm(1.0, &a, &eye, 0.0, &mut c).unwrap();
        assert_eq!(c.as_slice(), a.as_slice());
    }

    #[test]
    fn test_gemm_single_element() {
        let a = mat_f64(&[3.0], 1, 1);
        let b = mat_f64(&[7.0], 1, 1);
        let mut c = Tensor::<f64>::zeros(vec![1, 1]);
        gemm(1.0, &a, &b, 0.0, &mut c).unwrap();
        assert_eq!(c.as_slice(), &[21.0]);
    }

    #[test]
    fn test_gemm_dimension_mismatch() {
        let a = mat_f64(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let b = mat_f64(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
        let mut c = Tensor::<f64>::zeros(vec![2, 2]);
        assert!(gemm(1.0, &a, &b, 0.0, &mut c).is_err());
    }

    #[test]
    fn test_gemm_c_shape_mismatch() {
        let a = mat_f64(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let b = mat_f64(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let mut c = Tensor::<f64>::zeros(vec![3, 3]);
        assert!(gemm(1.0, &a, &b, 0.0, &mut c).is_err());
    }

    // ------------------------------------------------------------------
    // Convenience methods
    // ------------------------------------------------------------------

    #[test]
    fn test_matvec() {
        let a = mat_f64(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let x = vec_f64(&[5.0, 6.0]);
        let y = a.matvec(&x).unwrap();
        assert_eq!(y.as_slice(), &[17.0, 39.0]);
    }

    #[test]
    fn test_matmul() {
        let a = mat_f64(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let b = mat_f64(&[5.0, 6.0, 7.0, 8.0], 2, 2);
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.as_slice(), &[19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_tensor_dot() {
        let x = vec_f64(&[1.0, 2.0, 3.0]);
        let y = vec_f64(&[4.0, 5.0, 6.0]);
        assert_eq!(x.dot(&y).unwrap(), 32.0);
    }

    #[test]
    fn test_tensor_norm() {
        let x = vec_f64(&[3.0, 4.0]);
        assert!((x.norm().unwrap() - 5.0).abs() < 1e-10);
    }

    // ------------------------------------------------------------------
    // NumPy reference values
    // ------------------------------------------------------------------

    #[test]
    fn test_gemm_numpy_reference() {
        // >>> import numpy as np
        // >>> a = np.array([[1,2,3],[4,5,6]], dtype=np.float64)
        // >>> b = np.array([[7,8],[9,10],[11,12]], dtype=np.float64)
        // >>> a @ b
        // array([[ 58.,  64.],
        //        [139., 154.]])
        let a = mat_f64(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let b = mat_f64(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], 3, 2);
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.as_slice(), &[58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_gemv_numpy_reference() {
        // >>> a = np.array([[1,2,3],[4,5,6]], dtype=np.float64)
        // >>> x = np.array([1,1,1], dtype=np.float64)
        // >>> a @ x
        // array([ 6., 15.])
        let a = mat_f64(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let x = vec_f64(&[1.0, 1.0, 1.0]);
        let y = a.matvec(&x).unwrap();
        assert_eq!(y.as_slice(), &[6.0, 15.0]);
    }

    #[test]
    fn test_dot_numpy_reference() {
        // >>> np.dot([1,2,3,4,5], [5,4,3,2,1])
        // 35
        let x = vec_f64(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = vec_f64(&[5.0, 4.0, 3.0, 2.0, 1.0]);
        assert_eq!(dot(&x, &y).unwrap(), 35.0);
    }

    #[test]
    fn test_nrm2_numpy_reference() {
        // >>> np.linalg.norm([1, 2, 3, 4, 5])
        // 7.416198487095663
        let x = vec_f64(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let n = nrm2(&x).unwrap();
        assert!((n - 7.416_198_487_095_663).abs() < 1e-12);
    }
}
