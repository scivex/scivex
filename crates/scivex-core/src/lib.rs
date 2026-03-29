#![allow(
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::doc_markdown,
    clippy::module_name_repetitions
)]
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

/// Arena and slab allocators for temporary tensor buffers.
pub mod arena;
/// Native complex number type.
pub mod complex;
/// Numeric type traits: [`Scalar`], [`Float`], [`Real`], [`Integer`].
pub mod dtype;
/// Core error types.
pub mod error;
/// Fast Fourier Transform (FFT / IFFT / RFFT).
pub mod fft;
/// Expression JIT for fusing element-wise tensor operations.
pub mod jit;
/// Linear algebra: decompositions, solvers, and matrix operations.
pub mod linalg;
/// Elementary mathematical functions.
pub mod math;
/// Rayon-based parallel execution for tensors and matrix operations.
#[cfg(feature = "parallel")]
pub mod parallel;
/// Type promotion rules, casting utilities, and runtime dtype tags.
pub mod promote;
/// Pseudo-random number generation.
pub mod random;
/// SIMD-accelerated kernels for core numerical operations.
#[cfg(feature = "simd")]
pub(crate) mod simd;
/// Spatial data structures: KD-tree, ball tree.
pub mod spatial;
/// N-dimensional tensor type and operations.
pub mod tensor;

// Re-export key types at crate root for convenience.
pub use complex::Complex;
pub use dtype::{Float, Integer, Real, Scalar};
pub use error::{CoreError, Result};
pub use promote::{CastFrom, DType, DTypeOf, promote};
pub use tensor::Tensor;
pub use tensor::named::NamedTensor;
pub use tensor::sparse::SparseTensor;

// Re-export half-precision types when enabled.
#[cfg(feature = "mixed-precision")]
pub use dtype::{bf16, f16};

/// Items intended for glob-import: `use scivex_core::prelude::*;`
pub mod prelude {
    pub use crate::complex::Complex;
    pub use crate::dtype::{Float, Integer, Real, Scalar};
    pub use crate::error::{CoreError, Result};
    pub use crate::tensor::Tensor;
    pub use crate::tensor::named::NamedTensor;
    pub use crate::tensor::sparse::SparseTensor;
}
