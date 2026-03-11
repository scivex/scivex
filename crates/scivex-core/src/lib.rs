//! `scivex-core` — Foundation crate for the Scivex ecosystem.
//!
//! Provides tensors, numeric type traits, linear algebra, FFT, and math
//! primitives. All other `scivex-*` crates build on top of this one.
//!
//! # Design
//!
//! - **Zero external dependencies** for math — everything is from scratch.
//! - Generic over numeric types via the [`Scalar`] / [`Float`] / [`Real`] trait
//!   hierarchy.
//! - `unsafe` is confined to this crate; all higher-level crates use only safe
//!   abstractions.

/// Numeric type traits: [`Scalar`], [`Float`], [`Real`], [`Integer`].
pub mod dtype;
/// Core error types.
pub mod error;
/// Fast Fourier Transform (FFT / IFFT / RFFT).
pub mod fft;
/// Linear algebra: decompositions, solvers, and matrix operations.
pub mod linalg;
/// Elementary mathematical functions.
pub mod math;
/// Rayon-based parallel execution for tensors and matrix operations.
#[cfg(feature = "parallel")]
pub mod parallel;
/// Pseudo-random number generation.
pub mod random;
/// SIMD-accelerated kernels for core numerical operations.
#[cfg(feature = "simd")]
pub(crate) mod simd;
/// N-dimensional tensor type and operations.
pub mod tensor;

// Re-export key types at crate root for convenience.
pub use dtype::{Float, Integer, Real, Scalar};
pub use error::{CoreError, Result};
pub use tensor::Tensor;

/// Items intended for glob-import: `use scivex_core::prelude::*;`
pub mod prelude {
    pub use crate::dtype::{Float, Integer, Real, Scalar};
    pub use crate::error::{CoreError, Result};
    pub use crate::tensor::Tensor;
}
