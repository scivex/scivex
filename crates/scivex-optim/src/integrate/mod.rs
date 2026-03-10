//! Numerical integration (quadrature) algorithms.

mod quad;
mod simpson;
mod trapezoid;

pub use quad::quad;
pub use simpson::simpson;
pub use trapezoid::trapezoid;

use scivex_core::Float;

/// Result of a numerical integration.
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
