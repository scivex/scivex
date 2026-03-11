//! Rayon-based parallel execution utilities for tensors.
//!
//! All functions in this module require the `parallel` feature flag.
//! They provide parallel versions of element-wise operations, reductions,
//! and matrix operations that automatically use all available CPU cores.
//!
//! # Threshold
//!
//! Parallelism has overhead from thread scheduling. Operations on small
//! tensors are faster sequentially. The [`PAR_THRESHOLD`] constant defines
//! the minimum number of elements required to trigger parallel execution.

use rayon::prelude::*;

use crate::error::{CoreError, Result};
use crate::tensor::Tensor;
use crate::{Float, Scalar};

/// Minimum number of elements to trigger parallel execution.
///
/// Below this threshold, operations fall back to sequential execution
/// to avoid Rayon scheduling overhead.
pub const PAR_THRESHOLD: usize = 4096;

// ======================================================================
// Parallel element-wise operations on Tensor
// ======================================================================

impl<T: Scalar> Tensor<T> {
    /// Parallel element-wise addition. Falls back to sequential for small tensors.
    pub fn par_add(&self, other: &Tensor<T>) -> Result<Tensor<T>> {
        if self.shape() != other.shape() {
            return Err(CoreError::DimensionMismatch {
                expected: self.shape().to_vec(),
                got: other.shape().to_vec(),
            });
        }
        let data = if self.numel() >= PAR_THRESHOLD {
            self.as_slice()
                .par_iter()
                .zip(other.as_slice().par_iter())
                .map(|(&a, &b)| a + b)
                .collect()
        } else {
            self.as_slice()
                .iter()
                .zip(other.as_slice().iter())
                .map(|(&a, &b)| a + b)
                .collect()
        };
        Tensor::from_vec(data, self.shape().to_vec())
    }

    /// Parallel element-wise subtraction.
    pub fn par_sub(&self, other: &Tensor<T>) -> Result<Tensor<T>> {
        if self.shape() != other.shape() {
            return Err(CoreError::DimensionMismatch {
                expected: self.shape().to_vec(),
                got: other.shape().to_vec(),
            });
        }
        let data = if self.numel() >= PAR_THRESHOLD {
            self.as_slice()
                .par_iter()
                .zip(other.as_slice().par_iter())
                .map(|(&a, &b)| a - b)
                .collect()
        } else {
            self.as_slice()
                .iter()
                .zip(other.as_slice().iter())
                .map(|(&a, &b)| a - b)
                .collect()
        };
        Tensor::from_vec(data, self.shape().to_vec())
    }

    /// Parallel element-wise multiplication.
    pub fn par_mul(&self, other: &Tensor<T>) -> Result<Tensor<T>> {
        if self.shape() != other.shape() {
            return Err(CoreError::DimensionMismatch {
                expected: self.shape().to_vec(),
                got: other.shape().to_vec(),
            });
        }
        let data = if self.numel() >= PAR_THRESHOLD {
            self.as_slice()
                .par_iter()
                .zip(other.as_slice().par_iter())
                .map(|(&a, &b)| a * b)
                .collect()
        } else {
            self.as_slice()
                .iter()
                .zip(other.as_slice().iter())
                .map(|(&a, &b)| a * b)
                .collect()
        };
        Tensor::from_vec(data, self.shape().to_vec())
    }

    /// Parallel scalar broadcast addition.
    pub fn par_add_scalar(&self, scalar: T) -> Tensor<T> {
        let data = if self.numel() >= PAR_THRESHOLD {
            self.as_slice().par_iter().map(|&a| a + scalar).collect()
        } else {
            self.as_slice().iter().map(|&a| a + scalar).collect()
        };
        Tensor::from_vec(data, self.shape().to_vec()).unwrap()
    }

    /// Parallel scalar broadcast multiplication.
    pub fn par_mul_scalar(&self, scalar: T) -> Tensor<T> {
        let data = if self.numel() >= PAR_THRESHOLD {
            self.as_slice().par_iter().map(|&a| a * scalar).collect()
        } else {
            self.as_slice().iter().map(|&a| a * scalar).collect()
        };
        Tensor::from_vec(data, self.shape().to_vec()).unwrap()
    }

    /// Parallel map: apply a function to every element.
    pub fn par_map<F>(&self, f: F) -> Tensor<T>
    where
        F: Fn(T) -> T + Send + Sync,
    {
        let data = if self.numel() >= PAR_THRESHOLD {
            self.as_slice().par_iter().map(|&a| f(a)).collect()
        } else {
            self.as_slice().iter().map(|&a| f(a)).collect()
        };
        Tensor::from_vec(data, self.shape().to_vec()).unwrap()
    }

    /// Parallel sum of all elements.
    pub fn par_sum(&self) -> T {
        if self.numel() >= PAR_THRESHOLD {
            self.as_slice()
                .par_iter()
                .copied()
                .reduce(T::zero, |a, b| a + b)
        } else {
            self.sum()
        }
    }
}

impl<T: Float> Tensor<T> {
    /// Parallel mean of all elements.
    pub fn par_mean(&self) -> T {
        self.par_sum() / T::from_usize(self.numel())
    }
}

// ======================================================================
// Parallel matrix multiplication (block decomposition)
// ======================================================================

/// Parallel general matrix-matrix multiply: `C = alpha * A * B + beta * C`.
///
/// Uses row-parallel decomposition: each row of C is computed independently.
/// For small matrices (m < 64), falls back to sequential gemm.
#[allow(clippy::many_single_char_names)]
pub fn par_gemm<T: Scalar>(
    alpha: T,
    a: &Tensor<T>,
    b: &Tensor<T>,
    beta: T,
    c: &mut Tensor<T>,
) -> Result<()> {
    if a.ndim() != 2 || b.ndim() != 2 || c.ndim() != 2 {
        return Err(CoreError::InvalidArgument {
            reason: "par_gemm: all arguments must be 2-D tensors (matrices)",
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

    if m < 64 {
        return crate::linalg::gemm(alpha, a, b, beta, c);
    }

    let a_data = a.as_slice();
    let b_data = b.as_slice();
    let c_data = c.as_mut_slice();

    // Parallel over rows of C.
    c_data.par_chunks_mut(n).enumerate().for_each(|(i, c_row)| {
        // Scale existing row by beta.
        if beta == T::zero() {
            for v in c_row.iter_mut() {
                *v = T::zero();
            }
        } else if beta != T::one() {
            for v in c_row.iter_mut() {
                *v *= beta;
            }
        }

        // Accumulate alpha * A[i,:] * B into C[i,:].
        let a_row_start = i * k;
        for p in 0..k {
            let scale = alpha * a_data[a_row_start + p];
            let b_row = &b_data[p * n..(p + 1) * n];
            for j in 0..n {
                c_row[j] += scale * b_row[j];
            }
        }
    });

    Ok(())
}

/// Parallel matrix-matrix multiply convenience: returns `A @ B`.
pub fn par_matmul<T: Scalar>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>> {
    let m = a.shape().first().copied().unwrap_or(0);
    let n = b.shape().get(1).copied().unwrap_or(0);
    let mut c = Tensor::zeros(vec![m, n]);
    par_gemm(T::one(), a, b, T::zero(), &mut c)?;
    Ok(c)
}

/// Parallel matrix-vector multiply: returns `A @ x`.
pub fn par_matvec<T: Scalar>(a: &Tensor<T>, x: &Tensor<T>) -> Result<Tensor<T>> {
    if a.ndim() != 2 {
        return Err(CoreError::InvalidArgument {
            reason: "par_matvec: `a` must be a 2-D tensor",
        });
    }
    if x.ndim() != 1 {
        return Err(CoreError::InvalidArgument {
            reason: "par_matvec: `x` must be a 1-D tensor",
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

    if m < 64 {
        return a.matvec(x);
    }

    let a_data = a.as_slice();
    let x_data = x.as_slice();

    let y_data: Vec<T> = (0..m)
        .into_par_iter()
        .map(|i| {
            let row_start = i * n;
            let mut sum = T::zero();
            for j in 0..n {
                sum += a_data[row_start + j] * x_data[j];
            }
            sum
        })
        .collect();

    Tensor::from_vec(y_data, vec![m])
}

// ======================================================================
// Parallel Tensor convenience methods
// ======================================================================

impl<T: Scalar> Tensor<T> {
    /// Parallel matrix-matrix multiply: returns `self @ other`.
    pub fn par_matmul(&self, other: &Tensor<T>) -> Result<Tensor<T>> {
        par_matmul(self, other)
    }

    /// Parallel matrix-vector multiply: returns `self @ x`.
    pub fn par_matvec(&self, x: &Tensor<T>) -> Result<Tensor<T>> {
        par_matvec(self, x)
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    fn vec_f64(data: &[f64]) -> Tensor<f64> {
        Tensor::from_vec(data.to_vec(), vec![data.len()]).unwrap()
    }

    fn mat_f64(data: &[f64], rows: usize, cols: usize) -> Tensor<f64> {
        Tensor::from_vec(data.to_vec(), vec![rows, cols]).unwrap()
    }

    #[test]
    fn test_par_add() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let b = Tensor::from_vec(vec![10.0, 20.0, 30.0], vec![3]).unwrap();
        let c = a.par_add(&b).unwrap();
        assert_eq!(c.as_slice(), &[11.0, 22.0, 33.0]);
    }

    #[test]
    fn test_par_sub() {
        let a = Tensor::from_vec(vec![10.0, 20.0, 30.0], vec![3]).unwrap();
        let b = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let c = a.par_sub(&b).unwrap();
        assert_eq!(c.as_slice(), &[9.0, 18.0, 27.0]);
    }

    #[test]
    fn test_par_mul() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], vec![3]).unwrap();
        let c = a.par_mul(&b).unwrap();
        assert_eq!(c.as_slice(), &[4.0, 10.0, 18.0]);
    }

    #[test]
    fn test_par_add_shape_mismatch() {
        let a = Tensor::from_vec(vec![1.0, 2.0], vec![2]).unwrap();
        let b = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        assert!(a.par_add(&b).is_err());
    }

    #[test]
    fn test_par_add_scalar() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let c = a.par_add_scalar(10.0);
        assert_eq!(c.as_slice(), &[11.0, 12.0, 13.0]);
    }

    #[test]
    fn test_par_mul_scalar() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let c = a.par_mul_scalar(10.0);
        assert_eq!(c.as_slice(), &[10.0, 20.0, 30.0]);
    }

    #[test]
    fn test_par_map() {
        let a = Tensor::from_vec(vec![1.0_f64, 4.0, 9.0, 16.0], vec![4]).unwrap();
        let b = a.par_map(f64::sqrt);
        assert_eq!(b.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_par_sum() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        assert_eq!(a.par_sum(), 10.0);
    }

    #[test]
    fn test_par_mean() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        assert_eq!(a.par_mean(), 2.5);
    }

    #[test]
    fn test_par_gemm() {
        let a = mat_f64(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let b = mat_f64(&[5.0, 6.0, 7.0, 8.0], 2, 2);
        let mut c = Tensor::<f64>::zeros(vec![2, 2]);
        par_gemm(1.0, &a, &b, 0.0, &mut c).unwrap();
        assert_eq!(c.as_slice(), &[19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_par_matmul() {
        let a = mat_f64(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let b = mat_f64(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], 3, 2);
        let c = par_matmul(&a, &b).unwrap();
        assert_eq!(c.as_slice(), &[58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_par_matvec() {
        let a = mat_f64(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let x = vec_f64(&[5.0, 6.0]);
        let y = par_matvec(&a, &x).unwrap();
        assert_eq!(y.as_slice(), &[17.0, 39.0]);
    }

    #[test]
    fn test_par_matmul_convenience() {
        let a = mat_f64(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let b = mat_f64(&[5.0, 6.0, 7.0, 8.0], 2, 2);
        let c = a.par_matmul(&b).unwrap();
        assert_eq!(c.as_slice(), &[19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_par_add_large() {
        // Test with data above PAR_THRESHOLD
        let n = 5000;
        let a = Tensor::from_vec(vec![1.0_f64; n], vec![n]).unwrap();
        let b = Tensor::from_vec(vec![2.0_f64; n], vec![n]).unwrap();
        let c = a.par_add(&b).unwrap();
        assert!(c.as_slice().iter().all(|&v| v == 3.0));
    }

    #[test]
    fn test_par_sum_large() {
        let n = 5000;
        let a = Tensor::from_vec(vec![1.0_f64; n], vec![n]).unwrap();
        assert!((a.par_sum() - n as f64).abs() < 1e-6);
    }

    #[test]
    fn test_par_gemm_with_alpha_beta() {
        let a = mat_f64(&[1.0, 0.0, 0.0, 1.0], 2, 2);
        let b = mat_f64(&[5.0, 6.0, 7.0, 8.0], 2, 2);
        let mut c = mat_f64(&[1.0, 1.0, 1.0, 1.0], 2, 2);
        par_gemm(2.0, &a, &b, 3.0, &mut c).unwrap();
        assert_eq!(c.as_slice(), &[13.0, 15.0, 17.0, 19.0]);
    }

    #[test]
    fn test_par_gemm_dimension_mismatch() {
        let a = mat_f64(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let b = mat_f64(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
        let mut c = Tensor::<f64>::zeros(vec![2, 2]);
        assert!(par_gemm(1.0, &a, &b, 0.0, &mut c).is_err());
    }
}
