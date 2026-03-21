//! Numerical integration (quadrature) algorithms.

mod quad;
mod simpson;
mod trapezoid;

pub use quad::quad;
pub use simpson::simpson;
pub use trapezoid::trapezoid;

use scivex_core::Float;

/// Result of a numerical integration.
///
/// # Examples
///
/// ```
/// # use scivex_optim::integrate::{simpson, QuadResult};
/// let result = simpson(|x: f64| x * x, 0.0, 1.0, 100).unwrap();
/// assert!((result.value - 1.0 / 3.0).abs() < 1e-6);
/// ```
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct QuadResult<T: Float> {
    /// The estimated value of the integral.
    pub value: T,
    /// An estimate of the absolute error.
    pub error_estimate: T,
    /// Number of function evaluations.
    pub n_evals: usize,
}

/// Options controlling numerical integration.
///
/// # Examples
///
/// ```
/// # use scivex_optim::integrate::QuadOptions;
/// let opts = QuadOptions::<f64>::default();
/// assert!(opts.abs_tol < 1e-5);
/// ```
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct QuadOptions<T: Float> {
    /// Absolute error tolerance.
    pub abs_tol: T,
    /// Relative error tolerance.
    pub rel_tol: T,
    /// Maximum number of subdivisions (for adaptive methods).
    pub max_subdivisions: usize,
}

impl<T: Float> Default for QuadOptions<T> {
    fn default() -> Self {
        Self {
            abs_tol: T::from_f64(1e-10),
            rel_tol: T::from_f64(1e-10),
            max_subdivisions: 50,
        }
    }
}
