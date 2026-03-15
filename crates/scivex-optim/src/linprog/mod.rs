//! Linear programming via the revised simplex method.
//!
//! Solves problems of the form:
//!
//! ```text
//! minimize    c^T x
//! subject to  A_ub x <= b_ub
//!             x >= 0
//! ```

mod simplex;

pub use simplex::linprog;

use scivex_core::Float;

/// Result of a linear programming solve.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct LinProgResult<T: Float> {
    /// Optimal decision variables.
    pub x: Vec<T>,
    /// Optimal objective value.
    pub fun: T,
    /// Slack variables (b_ub - A_ub x).
    pub slack: Vec<T>,
    /// Number of simplex iterations.
    pub iterations: usize,
    /// Whether the solver found an optimal solution.
    pub converged: bool,
}
