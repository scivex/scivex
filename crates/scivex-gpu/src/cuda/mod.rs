//! CUDA backend for GPU-accelerated tensor operations.
//!
//! This module provides CUDA-based GPU compute when the `cuda` feature is enabled.
//! It requires the NVIDIA CUDA toolkit (nvcc, libcuda, libcublas) to be installed.

mod context;
mod cublas;
mod cudnn;
mod kernels;
mod memory;
mod tensor;

pub use context::{CudaContext, CudaDevice, CudaStream};
pub use cublas::CuBlasHandle;
pub use cudnn::CuDnnHandle;
pub use kernels::CudaKernels;
pub use memory::CudaBuffer;
pub use tensor::CudaTensor;
