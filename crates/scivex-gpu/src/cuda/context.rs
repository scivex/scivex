//! CUDA context, device, and stream management.

use std::ffi::c_void;

use crate::error::{GpuError, Result};

// ---------------------------------------------------------------------------
// CUDA runtime API FFI declarations
// ---------------------------------------------------------------------------

/// Opaque struct matching the layout expected by `cudaGetDeviceProperties`.
///
/// Only the fields we read are typed; the rest is padding to cover the full
/// `cudaDeviceProp` structure (which is several KiB).
#[repr(C)]
struct CudaDeviceProp {
    /// Device name (null-terminated ASCII).
    name: [u8; 256],
    /// Total global memory in bytes.
    total_global_mem: usize,
    /// Padding for the many fields we don't read.
    _padding: [u8; 4096],
    /// Major compute capability.
    major: i32,
    /// Minor compute capability.
    minor: i32,
}

unsafe extern "C" {
    fn cudaGetDeviceCount(count: *mut i32) -> i32;
    fn cudaSetDevice(device: i32) -> i32;
    fn cudaGetDeviceProperties(prop: *mut CudaDeviceProp, device: i32) -> i32;
    fn cudaDeviceSynchronize() -> i32;
    fn cudaStreamCreate(stream: *mut *mut c_void) -> i32;
    fn cudaStreamSynchronize(stream: *mut c_void) -> i32;
    fn cudaStreamDestroy(stream: *mut c_void) -> i32;
}

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

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
// CudaDevice
// ---------------------------------------------------------------------------

/// Represents a single NVIDIA CUDA device.
pub struct CudaDevice {
    device_id: i32,
    name: String,
    compute_capability: (i32, i32),
    total_memory: usize,
}

impl CudaDevice {
    /// Return the number of CUDA-capable devices on the system.
    pub fn count() -> Result<i32> {
        let mut n: i32 = 0;
        // SAFETY: `cudaGetDeviceCount` writes a single i32 to the provided pointer.
        // We pass a valid, aligned, exclusively-owned pointer.
        let code = unsafe { cudaGetDeviceCount(&mut n) };
        check_cuda(code, "cudaGetDeviceCount")?;
        Ok(n)
    }

    /// Create a device handle for the given `device_id`.
    ///
    /// Queries device properties (name, memory, compute capability).
    pub fn new(device_id: i32) -> Result<Self> {
        let count = Self::count()?;
        if device_id < 0 || device_id >= count {
            return Err(GpuError::CudaError {
                code: -1,
                reason: format!("device_id {device_id} out of range (0..{count})"),
            });
        }

        // SAFETY: `cudaGetDeviceProperties` writes into a caller-allocated struct.
        // We zero-initialise it so all padding bytes are defined.
        let mut prop: CudaDeviceProp = unsafe { std::mem::zeroed() };
        let code = unsafe { cudaGetDeviceProperties(&mut prop, device_id) };
        check_cuda(code, "cudaGetDeviceProperties")?;

        let name_end = prop
            .name
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(prop.name.len());
        let name = String::from_utf8_lossy(&prop.name[..name_end]).into_owned();

        Ok(Self {
            device_id,
            name,
            compute_capability: (prop.major, prop.minor),
            total_memory: prop.total_global_mem,
        })
    }

    /// Device ordinal.
    pub fn id(&self) -> i32 {
        self.device_id
    }

    /// Human-readable device name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// `(major, minor)` compute capability.
    pub fn compute_capability(&self) -> (i32, i32) {
        self.compute_capability
    }

    /// Total global memory in bytes.
    pub fn total_memory(&self) -> usize {
        self.total_memory
    }

    /// Make this device the current CUDA device for the calling thread.
    pub fn set_current(&self) -> Result<()> {
        // SAFETY: `cudaSetDevice` is safe to call with any valid device id.
        let code = unsafe { cudaSetDevice(self.device_id) };
        check_cuda(code, "cudaSetDevice")
    }
}

// ---------------------------------------------------------------------------
// CudaContext
// ---------------------------------------------------------------------------

/// A CUDA execution context bound to a device.
pub struct CudaContext {
    device: CudaDevice,
}

impl CudaContext {
    /// Create a new context on the given device.
    pub fn new(device: CudaDevice) -> Result<Self> {
        device.set_current()?;
        Ok(Self { device })
    }

    /// Get a reference to the underlying device.
    pub fn device(&self) -> &CudaDevice {
        &self.device
    }

    /// Block until all previously issued work on this device completes.
    pub fn synchronize() -> Result<()> {
        // SAFETY: `cudaDeviceSynchronize` has no preconditions beyond a valid
        // CUDA context being active (ensured by `CudaContext::new`).
        let code = unsafe { cudaDeviceSynchronize() };
        check_cuda(code, "cudaDeviceSynchronize")
    }
}

// ---------------------------------------------------------------------------
// CudaStream
// ---------------------------------------------------------------------------

/// A CUDA stream for asynchronous kernel execution.
pub struct CudaStream {
    handle: *mut c_void,
}

impl CudaStream {
    /// Create a new CUDA stream on the current device.
    pub fn new(_ctx: &CudaContext) -> Result<Self> {
        let mut handle: *mut c_void = std::ptr::null_mut();
        // SAFETY: `cudaStreamCreate` writes a valid stream handle to the pointer.
        let code = unsafe { cudaStreamCreate(&mut handle) };
        check_cuda(code, "cudaStreamCreate")?;
        Ok(Self { handle })
    }

    /// Block until all operations in this stream complete.
    pub fn synchronize(&self) -> Result<()> {
        // SAFETY: `self.handle` was successfully created by `cudaStreamCreate`.
        let code = unsafe { cudaStreamSynchronize(self.handle) };
        check_cuda(code, "cudaStreamSynchronize")
    }

    /// Raw stream handle for use in kernel launches.
    pub fn as_ptr(&self) -> *mut c_void {
        self.handle
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        // SAFETY: `self.handle` was successfully created and has not been
        // destroyed yet (we own it).
        unsafe {
            let _ = cudaStreamDestroy(self.handle);
        }
    }
}

// We never share the raw pointer across threads, but the CUDA runtime is
// thread-safe for stream operations when used correctly.
// SAFETY: `CudaStream` owns its handle and CUDA streams are safe to send
// between threads.
unsafe impl Send for CudaStream {}
