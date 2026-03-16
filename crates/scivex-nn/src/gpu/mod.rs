//! GPU-accelerated training backend for neural networks.
//!
//! This module provides GPU-backed autograd, operations, layers, and optimizers
//! for training neural networks on the GPU via wgpu.
//!
//! All GPU operations use `f32` precision.

pub mod functional;
pub mod layer;
pub mod loss;
pub mod ops;
pub mod optim;
pub mod variable;

pub use variable::GpuVariable;

/// Convenience re-exports for GPU training.
pub mod prelude {
    pub use super::layer::{GpuLayer, GpuLinear};
    pub use super::loss::{gpu_cross_entropy_loss, gpu_mse_loss};
    pub use super::optim::{GpuAdam, GpuOptimizer, GpuSGD};
    pub use super::variable::GpuVariable;
}
