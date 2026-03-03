//! # Scivex
//!
//! A comprehensive Rust library replacing the Python data science ecosystem.
//!
//! One `use scivex::prelude::*;` gives you tensors, dataframes, statistics,
//! machine learning, neural networks, visualization, and more — all
//! implemented from scratch in pure Rust.
//!
//! ## Feature Flags
//!
//! | Feature | Enables |
//! |---------|---------|
//! | `core` *(default)* | Tensors, linear algebra, FFT, math primitives |
//!
//! Additional sub-crates will be gated behind their own feature flags as
//! development progresses.

pub use scivex_core as core;

#[cfg(feature = "frame")]
pub use scivex_frame as frame;

#[cfg(feature = "stats")]
pub use scivex_stats as stats;

#[cfg(feature = "io")]
pub use scivex_io as io;

/// Glob-import convenience: `use scivex::prelude::*;`
pub mod prelude {
    pub use scivex_core::prelude::*;

    #[cfg(feature = "frame")]
    pub use scivex_frame::prelude::*;

    #[cfg(feature = "stats")]
    pub use scivex_stats::prelude::*;

    #[cfg(feature = "io")]
    pub use scivex_io::prelude::*;
}

// Re-export the Tensor type at the top level for ergonomics.
pub use scivex_core::Tensor;

#[cfg(feature = "frame")]
pub use scivex_frame::DataFrame;
