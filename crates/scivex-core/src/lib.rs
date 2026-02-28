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

pub mod dtype;
pub mod error;

// Re-export key types at crate root for convenience.
pub use dtype::{Float, Integer, Real, Scalar};
pub use error::{CoreError, Result};

/// Items intended for glob-import: `use scivex_core::prelude::*;`
pub mod prelude {
    pub use crate::dtype::{Float, Integer, Real, Scalar};
    pub use crate::error::{CoreError, Result};
}
