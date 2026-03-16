//! CUDA device memory management.

use std::ffi::c_void;

use crate::error::{GpuError, Result};

// ---------------------------------------------------------------------------
// CUDA memory FFI
// ---------------------------------------------------------------------------

/// `cudaMemcpyKind` enum values.
const CUDA_MEMCPY_HOST_TO_DEVICE: i32 = 1;
const CUDA_MEMCPY_DEVICE_TO_HOST: i32 = 2;
const CUDA_MEMCPY_DEVICE_TO_DEVICE: i32 = 3;

unsafe extern "C" {
    fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> i32;
    fn cudaFree(ptr: *mut c_void) -> i32;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
}

/// Convert a CUDA status code to a `Result`.
fn check_cuda(code: i32, context: &str) -> Result<()> {
    if code == 0 {
        Ok(())
    } else {
        Err(GpuError::CudaError {
            code,
            reason: context.to_owned(),
        })
    }
}

// ---------------------------------------------------------------------------
// CudaBuffer
// ---------------------------------------------------------------------------

/// A buffer of device memory allocated via `cudaMalloc`.
pub struct CudaBuffer {
    ptr: *mut c_void,
    size_bytes: usize,
    device_id: i32,
}

impl CudaBuffer {
    /// Allocate `size_bytes` of device memory.
    pub fn allocate(size_bytes: usize, device_id: i32) -> Result<Self> {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        // SAFETY: `cudaMalloc` writes a valid device pointer into `ptr`.
        // We check the return code before using it.
        let code = unsafe { cudaMalloc(&mut ptr, size_bytes) };
        check_cuda(code, "cudaMalloc")?;
        Ok(Self {
            ptr,
            size_bytes,
            device_id,
        })
    }

    /// Allocate device memory and copy host `f32` data into it (host-to-device).
    pub fn from_host(data: &[f32], device_id: i32) -> Result<Self> {
        let size_bytes = std::mem::size_of_val(data);
        let buf = Self::allocate(size_bytes, device_id)?;
        // SAFETY: `buf.ptr` is a valid device pointer of `size_bytes` bytes.
        // `data.as_ptr()` is a valid host pointer for `size_bytes` bytes.
        let code = unsafe {
            cudaMemcpy(
                buf.ptr,
                data.as_ptr().cast::<c_void>(),
                size_bytes,
                CUDA_MEMCPY_HOST_TO_DEVICE,
            )
        };
        check_cuda(code, "cudaMemcpy (H2D)")?;
        Ok(buf)
    }

    /// Copy device data back to host (device-to-host).
    pub fn to_host(&self, out: &mut [f32]) -> Result<()> {
        let copy_bytes = std::mem::size_of_val(out);
        if copy_bytes > self.size_bytes {
            return Err(GpuError::TransferError {
                reason: format!(
                    "output slice ({copy_bytes} bytes) exceeds buffer ({} bytes)",
                    self.size_bytes
                ),
            });
        }
        // SAFETY: `self.ptr` is a valid device pointer. `out` is a valid host
        // slice with room for `copy_bytes` bytes.
        let code = unsafe {
            cudaMemcpy(
                out.as_mut_ptr().cast::<c_void>(),
                self.ptr.cast_const(),
                copy_bytes,
                CUDA_MEMCPY_DEVICE_TO_HOST,
            )
        };
        check_cuda(code, "cudaMemcpy (D2H)")
    }

    /// Copy data from another device buffer (device-to-device).
    pub fn copy_from(&mut self, other: &CudaBuffer) -> Result<()> {
        let copy_bytes = other.size_bytes.min(self.size_bytes);
        // SAFETY: Both pointers are valid device pointers. We copy the
        // minimum of the two sizes so neither buffer is overrun.
        let code = unsafe {
            cudaMemcpy(
                self.ptr,
                other.ptr.cast_const(),
                copy_bytes,
                CUDA_MEMCPY_DEVICE_TO_DEVICE,
            )
        };
        check_cuda(code, "cudaMemcpy (D2D)")
    }

    /// Size of this buffer in bytes.
    pub fn size_bytes(&self) -> usize {
        self.size_bytes
    }

    /// Raw device pointer (const).
    pub fn as_ptr(&self) -> *const c_void {
        self.ptr.cast_const()
    }

    /// Raw device pointer (mutable).
    pub fn as_mut_ptr(&mut self) -> *mut c_void {
        self.ptr
    }

    /// Device ordinal this buffer was allocated on.
    pub fn device_id(&self) -> i32 {
        self.device_id
    }
}

impl Drop for CudaBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: `self.ptr` was allocated by `cudaMalloc` and has not
            // been freed yet (we own it).
            unsafe {
                let _ = cudaFree(self.ptr);
            }
        }
    }
}

// SAFETY: CUDA memory can be accessed from any host thread as long as the
// correct device context is active.
unsafe impl Send for CudaBuffer {}
