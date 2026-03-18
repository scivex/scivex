//! GPU-accelerated tensor operations for Scivex.
//!
//! This crate provides GPU compute via [`wgpu`], supporting Vulkan, Metal, DX12,
//! and WebGPU backends. All GPU tensors operate on `f32` data — use
//! [`GpuTensor::from_tensor_f64`] for automatic conversion from `f64`.
//!
//! # Quick start
//!
//! ```rust,no_run
//! use scivex_gpu::{GpuDevice, GpuTensor, ops};
//! use scivex_core::Tensor;
//!
//! let dev = GpuDevice::new().expect("no GPU found");
//!
//! let a_cpu = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
//! let b_cpu = Tensor::from_vec(vec![5.0f32, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
//!
//! let a = GpuTensor::from_tensor(&dev, &a_cpu);
//! let b = GpuTensor::from_tensor(&dev, &b_cpu);
//!
//! let c = ops::matmul(&a, &b).unwrap();
//! let result = c.to_tensor().unwrap();
//! println!("{:?}", result.as_slice()); // [19.0, 22.0, 43.0, 50.0]
//! ```
//!
//! # Supported operations
//!
//! | Category | Functions |
//! |----------|-----------|
//! | Element-wise | `add`, `sub`, `mul`, `div` |
//! | Scalar | `add_scalar`, `mul_scalar` |
//! | Linear algebra | `matmul` |
//! | Reduction | `sum`, `mean` |

pub mod backend;
pub mod device;
pub mod error;
pub mod ops;
pub mod optim;
pub mod tensor;

#[cfg(feature = "cuda")]
pub mod cuda;

pub use backend::GpuBackend;
pub use device::{GpuDevice, GpuInfo};
pub use error::GpuError;
pub use ops::GpuBatch;
pub use optim::{GpuAdam, GpuSGD};
pub use tensor::GpuTensor;

#[cfg(feature = "cuda")]
pub use cuda::{
    CuBlasHandle, CuDnnHandle, CudaBuffer, CudaContext, CudaDevice, CudaKernels, CudaStream,
    CudaTensor,
};
