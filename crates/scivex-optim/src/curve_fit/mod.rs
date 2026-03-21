//! Curve fitting via non-linear least squares.
//!
//! Provides the [`levenberg_marquardt`] algorithm and a high-level [`curve_fit()`]
//! convenience function.

mod levenberg_marquardt;

pub use self::levenberg_marquardt::levenberg_marquardt;

use scivex_core::Float;

use crate::error::Result;

/// Result of a least squares curve fit.
///
/// # Examples
///
/// ```
/// # use scivex_optim::curve_fit::curve_fit;
/// let result = curve_fit(
///     |x: f64, p: &[f64]| p[0] * x + p[1],
///     &[1.0_f64, 2.0, 3.0], &[2.0, 4.0, 6.0], &[1.0, 0.0],
/// ).unwrap();
/// assert!(result.converged);
/// assert!((result.params[0] - 2.0).abs() < 0.1);
/// ```
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct LeastSquaresResult<T: Float> {
    /// Optimal parameters.
    pub params: Vec<T>,
    /// Residuals at the optimal parameters.
    pub residuals: Vec<T>,
    /// Sum of squared residuals.
    pub cost: T,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Whether the algorithm converged.
    pub converged: bool,
}

/// High-level curve fitting using Levenberg-Marquardt with default tolerances.
///
/// Fits `model(x, params)` to the data `(x_data, y_data)` starting from initial
/// parameters `p0`. Uses `max_iter=200` and `tol=1e-10`.
///
/// # Examples
///
/// ```
/// # use scivex_optim::curve_fit::curve_fit;
/// // Fit y = a*x + b to data
/// let x_data = [1.0_f64, 2.0, 3.0, 4.0];
/// let y_data = [2.1, 3.9, 6.1, 7.9]; // ~2x
/// let result = curve_fit(
///     |x: f64, p: &[f64]| p[0] * x + p[1],
///     &x_data, &y_data, &[1.0, 0.0],
/// ).unwrap();
/// assert!((result.params[0] - 2.0).abs() < 0.2);
/// ```
pub fn curve_fit<T, F>(
    model: F,
    x_data: &[T],
    y_data: &[T],
    p0: &[T],
) -> Result<LeastSquaresResult<T>>
where
    T: Float,
    F: Fn(T, &[T]) -> T,
{
    levenberg_marquardt(model, x_data, y_data, p0, 200, T::from_f64(1e-10))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_curve_fit_gaussian() {
        // Model: y = a * exp(-(x - mu)^2 / (2 * sigma^2))
        // True params: a=3.0, mu=2.0, sigma=1.0
        let true_a = 3.0_f64;
        let true_mu = 2.0_f64;
        let true_sigma = 1.0_f64;

        let x_data: Vec<f64> = (-20..=60).map(|i| f64::from(i) * 0.1).collect();
        let y_data: Vec<f64> = x_data
            .iter()
            .map(|&x| true_a * (-((x - true_mu).powi(2)) / (2.0 * true_sigma * true_sigma)).exp())
            .collect();

        let model = |x: f64, p: &[f64]| -> f64 {
            p[0] * (-((x - p[1]).powi(2)) / (2.0 * p[2] * p[2])).exp()
        };

        let result = curve_fit(model, &x_data, &y_data, &[2.0, 1.5, 0.8]).unwrap();
        assert!(result.converged, "did not converge");
        assert!(
            (result.params[0] - true_a).abs() < 1e-3,
            "a = {}",
            result.params[0]
        );
        assert!(
            (result.params[1] - true_mu).abs() < 1e-3,
            "mu = {}",
            result.params[1]
        );
        assert!(
            (result.params[2] - true_sigma).abs() < 1e-2,
            "sigma = {}",
            result.params[2]
        );
    }

    #[test]
    fn test_curve_fit_sine_wave() {
        // Model: y = a * sin(b * x + c)
        // True params: a=2.0, b=1.5, c=0.5
        let x_data: Vec<f64> = (0..50).map(|i| f64::from(i) * 0.1).collect();
        let y_data: Vec<f64> = x_data
            .iter()
            .map(|&x| 2.0 * (1.5 * x + 0.5).sin())
            .collect();

        let model = |x: f64, p: &[f64]| -> f64 { p[0] * (p[1] * x + p[2]).sin() };

        let result = curve_fit(model, &x_data, &y_data, &[1.5, 1.2, 0.3]).unwrap();
        assert!(result.converged, "cost = {}", result.cost);
        assert!(
            (result.params[0] - 2.0).abs() < 0.1,
            "a = {}",
            result.params[0]
        );
    }

    #[test]
    fn test_curve_fit_default_tolerance() {
        // Simple linear fit to verify default tolerance works
        let x_data: Vec<f64> = (0..20).map(f64::from).collect();
        let y_data: Vec<f64> = x_data.iter().map(|&x| 3.0 * x + 7.0).collect();

        let model = |x: f64, p: &[f64]| -> f64 { p[0] * x + p[1] };

        let result = curve_fit(model, &x_data, &y_data, &[1.0, 1.0]).unwrap();
        assert!(result.converged);
        assert!(result.cost < 1e-8, "cost = {}", result.cost);
    }
}
