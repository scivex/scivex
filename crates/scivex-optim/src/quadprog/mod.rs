//! Quadratic programming via the active set method.
//!
//! Solves problems of the form:
//!
//! ```text
//! minimize    0.5 x^T H x + c^T x
//! subject to  A_ub x <= b_ub
//! ```
//!
//! where `H` is symmetric positive semi-definite.

pub mod active_set;

pub use active_set::quadprog;

use scivex_core::Float;

/// Result of a quadratic programming solve.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct QpResult<T: Float> {
    /// Optimal decision variables.
    pub x: Vec<T>,
    /// Optimal objective value (`0.5 x^T H x + c^T x`).
    pub fun: T,
    /// Number of active set iterations.
    pub iterations: usize,
    /// Whether the solver converged to an optimal solution.
    pub converged: bool,
}
