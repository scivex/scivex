//! One-dimensional minimization algorithms.
//!
//! Find `x` that minimizes `f(x)` within a bracket `[a, b]`.

mod brent;
mod golden;

pub use brent::brent_min;
pub use golden::golden_section;

use scivex_core::Float;

/// Result of a 1-D minimization algorithm.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct Minimize1dResult<T: Float> {
    /// The estimated minimizer.
    pub x_min: T,
    /// The function value at the minimizer: `f(x_min)`.
    pub f_min: T,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Whether the algorithm converged within tolerances.
    pub converged: bool,
}
