//! Root-finding algorithms for scalar functions.
//!
//! Find `x` such that `f(x) = 0`.

mod bisection;
mod brent;
mod newton;

pub use bisection::bisection;
pub use brent::brent_root;
pub use newton::newton;

use scivex_core::Float;

/// Result of a root-finding algorithm.
///
/// # Examples
///
/// ```
/// # use scivex_optim::prelude::*;
/// let result = bisection(|x: f64| x * x - 2.0, 1.0, 2.0, &RootOptions::default()).unwrap();
/// assert!(result.converged);
/// assert!((result.root - std::f64::consts::SQRT_2).abs() < 1e-10);
/// ```
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct RootResult<T: Float> {
    /// The estimated root.
    pub root: T,
    /// The function value at the root: `f(root)`.
    pub f_root: T,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Whether the algorithm converged within tolerances.
    pub converged: bool,
}

/// Options controlling root-finding algorithms.
///
/// # Examples
///
/// ```
/// # use scivex_optim::prelude::*;
/// let opts = RootOptions {
///     xtol: 1e-8,
///     ftol: 1e-8,
///     max_iter: 50,
/// };
/// let result = bisection(|x: f64| x * x - 2.0, 1.0, 2.0, &opts).unwrap();
/// assert!(result.converged);
/// ```
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct RootOptions<T: Float> {
    /// Tolerance on the root position.
    pub xtol: T,
    /// Tolerance on the function value.
    pub ftol: T,
    /// Maximum number of iterations.
    pub max_iter: usize,
}

impl<T: Float> Default for RootOptions<T> {
    fn default() -> Self {
        Self {
            xtol: T::from_f64(1e-12),
            ftol: T::from_f64(1e-12),
            max_iter: 100,
        }
    }
}
