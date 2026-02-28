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

/// Glob-import convenience: `use scivex::prelude::*;`
pub mod prelude {
    pub use scivex_core::prelude::*;
}
