//! CUDA kernel launch wrappers for common tensor operations.

use crate::error::{GpuError, Result};

use super::memory::CudaBuffer;

// ---------------------------------------------------------------------------
// Custom CUDA kernel FFI (compiled from .cu sources)
// ---------------------------------------------------------------------------

unsafe extern "C" {
    fn scivex_cuda_add(a: *const f32, b: *const f32, c: *mut f32, n: i32);
    fn scivex_cuda_mul(a: *const f32, b: *const f32, c: *mut f32, n: i32);
    fn scivex_cuda_add_scalar(a: *const f32, scalar: f32, b: *mut f32, n: i32);
    fn scivex_cuda_mul_scalar(a: *const f32, scalar: f32, b: *mut f32, n: i32);
    fn scivex_cuda_relu(a: *const f32, b: *mut f32, n: i32);
    fn scivex_cuda_sigmoid(a: *const f32, b: *mut f32, n: i32);
    fn scivex_cuda_sum(a: *const f32, n: i32, result: *mut f32);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Validate that two buffers have at least `n * sizeof(f32)` bytes.
fn check_buffers(a: &CudaBuffer, b: &CudaBuffer, n: usize) -> Result<()> {
    let required = n * std::mem::size_of::<f32>();
    if a.size_bytes() < required {
        return Err(GpuError::CudaError {
            code: -1,
            reason: format!(
                "input buffer too small: need {required} bytes, have {}",
                a.size_bytes()
            ),
        });
    }
    if b.size_bytes() < required {
        return Err(GpuError::CudaError {
            code: -1,
            reason: format!(
                "output buffer too small: need {required} bytes, have {}",
                b.size_bytes()
            ),
        });
    }
    Ok(())
}

/// Cast `n` to `i32`, returning an error on overflow.
#[allow(clippy::cast_possible_wrap)]
fn to_i32(n: usize) -> Result<i32> {
    if n > i32::MAX as usize {
        return Err(GpuError::CudaError {
            code: -1,
            reason: format!("element count {n} exceeds i32::MAX"),
        });
    }
    Ok(n as i32)
}

// ---------------------------------------------------------------------------
// CudaKernels
// ---------------------------------------------------------------------------

/// CUDA kernel launchers for common tensor operations.
pub struct CudaKernels;

impl CudaKernels {
    /// Element-wise addition: `c[i] = a[i] + b[i]`.
    pub fn add(a: &CudaBuffer, b: &CudaBuffer, c: &mut CudaBuffer, n: usize) -> Result<()> {
        let ni = to_i32(n)?;
        check_buffers(a, c, n)?;
        check_buffers(b, c, n)?;
        // SAFETY: All three pointers are valid device pointers with at least
        // `n * sizeof(f32)` bytes. The kernel reads `n` elements from `a` and
        // `b` and writes `n` elements to `c`.
        unsafe {
            scivex_cuda_add(
                a.as_ptr().cast::<f32>(),
                b.as_ptr().cast::<f32>(),
                c.as_mut_ptr().cast::<f32>(),
                ni,
            );
        }
        Ok(())
    }

    /// Element-wise multiplication: `c[i] = a[i] * b[i]`.
    pub fn mul(a: &CudaBuffer, b: &CudaBuffer, c: &mut CudaBuffer, n: usize) -> Result<()> {
        let ni = to_i32(n)?;
        check_buffers(a, c, n)?;
        check_buffers(b, c, n)?;
        // SAFETY: Same as `add` — valid device pointers, sufficient size.
        unsafe {
            scivex_cuda_mul(
                a.as_ptr().cast::<f32>(),
                b.as_ptr().cast::<f32>(),
                c.as_mut_ptr().cast::<f32>(),
                ni,
            );
        }
        Ok(())
    }

    /// Scalar addition: `b[i] = a[i] + scalar`.
    pub fn add_scalar(a: &CudaBuffer, scalar: f32, b: &mut CudaBuffer, n: usize) -> Result<()> {
        let ni = to_i32(n)?;
        check_buffers(a, b, n)?;
        // SAFETY: `a` and `b` are valid device pointers of sufficient size.
        unsafe {
            scivex_cuda_add_scalar(
                a.as_ptr().cast::<f32>(),
                scalar,
                b.as_mut_ptr().cast::<f32>(),
                ni,
            );
        }
        Ok(())
    }

    /// Scalar multiplication: `b[i] = a[i] * scalar`.
    pub fn mul_scalar(a: &CudaBuffer, scalar: f32, b: &mut CudaBuffer, n: usize) -> Result<()> {
        let ni = to_i32(n)?;
        check_buffers(a, b, n)?;
        // SAFETY: `a` and `b` are valid device pointers of sufficient size.
        unsafe {
            scivex_cuda_mul_scalar(
                a.as_ptr().cast::<f32>(),
                scalar,
                b.as_mut_ptr().cast::<f32>(),
                ni,
            );
        }
        Ok(())
    }

    /// ReLU activation: `b[i] = max(0, a[i])`.
    pub fn relu(a: &CudaBuffer, b: &mut CudaBuffer, n: usize) -> Result<()> {
        let ni = to_i32(n)?;
        check_buffers(a, b, n)?;
        // SAFETY: `a` and `b` are valid device pointers of sufficient size.
        unsafe {
            scivex_cuda_relu(a.as_ptr().cast::<f32>(), b.as_mut_ptr().cast::<f32>(), ni);
        }
        Ok(())
    }

    /// Sigmoid activation: `b[i] = 1 / (1 + exp(-a[i]))`.
    pub fn sigmoid(a: &CudaBuffer, b: &mut CudaBuffer, n: usize) -> Result<()> {
        let ni = to_i32(n)?;
        check_buffers(a, b, n)?;
        // SAFETY: `a` and `b` are valid device pointers of sufficient size.
        unsafe {
            scivex_cuda_sigmoid(a.as_ptr().cast::<f32>(), b.as_mut_ptr().cast::<f32>(), ni);
        }
        Ok(())
    }

    /// Sum reduction: returns the sum of all `n` elements in `a`.
    pub fn sum(a: &CudaBuffer, n: usize) -> Result<f32> {
        let ni = to_i32(n)?;
        let required = n * std::mem::size_of::<f32>();
        if a.size_bytes() < required {
            return Err(GpuError::CudaError {
                code: -1,
                reason: format!(
                    "buffer too small for sum: need {required} bytes, have {}",
                    a.size_bytes()
                ),
            });
        }
        let mut result: f32 = 0.0;
        // SAFETY: `a` is a valid device pointer of sufficient size. `result`
        // is a valid host pointer to a single f32. The kernel writes the sum
        // into `result`.
        unsafe {
            scivex_cuda_sum(a.as_ptr().cast::<f32>(), ni, &mut result);
        }
        Ok(result)
    }
}
