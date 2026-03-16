//! cuBLAS integration for dense linear algebra on CUDA devices.

use std::ffi::c_void;

use crate::error::{GpuError, Result};

use super::memory::CudaBuffer;

// ---------------------------------------------------------------------------
// cuBLAS FFI
// ---------------------------------------------------------------------------

unsafe extern "C" {
    fn cublasCreate_v2(handle: *mut *mut c_void) -> i32;
    fn cublasDestroy_v2(handle: *mut c_void) -> i32;
    fn cublasSgemm_v2(
        handle: *mut c_void,
        transa: i32,
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: *const f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: *const f32,
        c: *mut f32,
        ldc: i32,
    ) -> i32;
    fn cublasSdot_v2(
        handle: *mut c_void,
        n: i32,
        x: *const f32,
        incx: i32,
        y: *const f32,
        incy: i32,
        result: *mut f32,
    ) -> i32;
    fn cublasSnrm2_v2(
        handle: *mut c_void,
        n: i32,
        x: *const f32,
        incx: i32,
        result: *mut f32,
    ) -> i32;
    fn cublasSaxpy_v2(
        handle: *mut c_void,
        n: i32,
        alpha: *const f32,
        x: *const f32,
        incx: i32,
        y: *mut f32,
        incy: i32,
    ) -> i32;
}

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

fn check_cublas(code: i32, context: &str) -> Result<()> {
    if code == 0 {
        Ok(())
    } else {
        Err(GpuError::CuBlasError {
            code,
            reason: context.to_owned(),
        })
    }
}

// ---------------------------------------------------------------------------
// CuBlasHandle
// ---------------------------------------------------------------------------

/// A handle to the cuBLAS library context.
///
/// Creating a handle initialises cuBLAS for the current CUDA device.
/// Destroying it (via `Drop`) releases all cuBLAS resources.
pub struct CuBlasHandle {
    handle: *mut c_void,
}

impl CuBlasHandle {
    /// Create a new cuBLAS handle.
    pub fn new() -> Result<Self> {
        let mut handle: *mut c_void = std::ptr::null_mut();
        // SAFETY: `cublasCreate_v2` writes a valid handle to the pointer.
        let code = unsafe { cublasCreate_v2(&mut handle) };
        check_cublas(code, "cublasCreate_v2")?;
        Ok(Self { handle })
    }

    /// Single-precision general matrix multiplication (SGEMM).
    ///
    /// Computes `C = alpha * A @ B + beta * C` where `A` is `m x k`,
    /// `B` is `k x n`, and `C` is `m x n` (all in column-major order).
    #[allow(clippy::too_many_arguments)]
    pub fn sgemm(
        &self,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: &CudaBuffer,
        lda: i32,
        b: &CudaBuffer,
        ldb: i32,
        beta: f32,
        c: &mut CudaBuffer,
        ldc: i32,
    ) -> Result<()> {
        // cuBLAS operation: CUBLAS_OP_N = 0 (no transpose)
        let op_n: i32 = 0;
        // SAFETY: All device pointers (`a`, `b`, `c`) are valid CUDA
        // allocations. `alpha` and `beta` are host-side scalars passed by
        // pointer. The handle was successfully created.
        let code = unsafe {
            cublasSgemm_v2(
                self.handle,
                op_n,
                op_n,
                m,
                n,
                k,
                &alpha,
                a.as_ptr().cast::<f32>(),
                lda,
                b.as_ptr().cast::<f32>(),
                ldb,
                &beta,
                c.as_mut_ptr().cast::<f32>(),
                ldc,
            )
        };
        check_cublas(code, "cublasSgemm_v2")
    }

    /// Single-precision dot product: `result = x . y`.
    pub fn sdot(&self, n: i32, x: &CudaBuffer, y: &CudaBuffer) -> Result<f32> {
        let mut result: f32 = 0.0;
        // SAFETY: `x` and `y` are valid device pointers with at least `n`
        // f32 elements. `result` is a valid host pointer.
        let code = unsafe {
            cublasSdot_v2(
                self.handle,
                n,
                x.as_ptr().cast::<f32>(),
                1,
                y.as_ptr().cast::<f32>(),
                1,
                &mut result,
            )
        };
        check_cublas(code, "cublasSdot_v2")?;
        Ok(result)
    }

    /// Single-precision Euclidean norm: `result = ||x||_2`.
    pub fn snrm2(&self, n: i32, x: &CudaBuffer) -> Result<f32> {
        let mut result: f32 = 0.0;
        // SAFETY: `x` is a valid device pointer with at least `n` f32
        // elements. `result` is a valid host pointer.
        let code =
            unsafe { cublasSnrm2_v2(self.handle, n, x.as_ptr().cast::<f32>(), 1, &mut result) };
        check_cublas(code, "cublasSnrm2_v2")?;
        Ok(result)
    }

    /// SAXPY: `y = alpha * x + y`.
    pub fn saxpy(&self, n: i32, alpha: f32, x: &CudaBuffer, y: &mut CudaBuffer) -> Result<()> {
        // SAFETY: `x` and `y` are valid device pointers with at least `n`
        // f32 elements. `alpha` is passed by host pointer.
        let code = unsafe {
            cublasSaxpy_v2(
                self.handle,
                n,
                &alpha,
                x.as_ptr().cast::<f32>(),
                1,
                y.as_mut_ptr().cast::<f32>(),
                1,
            )
        };
        check_cublas(code, "cublasSaxpy_v2")
    }
}

impl Drop for CuBlasHandle {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            // SAFETY: `self.handle` was successfully created and has not been
            // destroyed yet.
            unsafe {
                let _ = cublasDestroy_v2(self.handle);
            }
        }
    }
}

// SAFETY: cuBLAS handles can be sent between threads; the library serialises
// access internally.
unsafe impl Send for CuBlasHandle {}
