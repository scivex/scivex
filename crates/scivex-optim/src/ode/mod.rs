//! ODE initial value problem solvers.
//!
//! Provides multiple methods for solving systems of ordinary differential
//! equations of the form `dy/dt = f(t, y)`:
//!
//! - [`euler`] — Forward Euler (1st order, simple)
//! - [`rk45`] — Dormand-Prince RK4(5) (adaptive, general-purpose)
//! - [`bdf2`] — BDF-2 (implicit, for stiff systems)
//! - [`solve_ivp`] — Unified entry point with method selection
//!
//! ## Example
//!
//! ```ignore
//! use scivex_optim::ode::{solve_ivp, OdeMethod, OdeOptions};
//!
//! // dy/dt = -y, y(0) = 1  =>  y(t) = e^(-t)
//! let result = solve_ivp(
//!     |_t, y: &[f64]| vec![-y[0]],
//!     [0.0, 1.0],
//!     &[1.0],
//!     OdeMethod::RK45,
//!     &OdeOptions::default(),
//! ).unwrap();
//!
//! println!("y(1) = {}", result.y.last().unwrap()[0]);
//! ```

mod bdf;
mod euler;
mod rk45;

pub use bdf::bdf2;
pub use euler::euler;
pub use rk45::rk45;

use scivex_core::Float;

use crate::error::Result;

/// Result of an ODE integration.
///
/// # Examples
///
/// ```
/// # use scivex_optim::ode::{euler, OdeOptions};
/// let result = euler(|_t, y: &[f64]| vec![-y[0]], [0.0, 1.0], &[1.0], &OdeOptions::default()).unwrap();
/// assert!(result.success);
/// assert!(!result.t.is_empty());
/// ```
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct OdeResult<T: Float> {
    /// Time values at each accepted step.
    pub t: Vec<T>,
    /// Solution vectors at each accepted step. `y[i]` corresponds to `t[i]`.
    pub y: Vec<Vec<T>>,
    /// Total number of function evaluations.
    pub n_evals: usize,
    /// Total number of accepted steps.
    pub n_steps: usize,
    /// Whether the integration completed successfully.
    pub success: bool,
}

/// Event function type: `fn(t, y) -> T`. Integration stops when the return value crosses zero.
pub type EventFn<T> = Box<dyn Fn(T, &[T]) -> T>;

/// Options for ODE solvers.
///
/// # Examples
///
/// ```
/// # use scivex_optim::ode::OdeOptions;
/// let opts = OdeOptions::<f64>::default();
/// assert_eq!(opts.max_steps, 10_000);
/// ```
pub struct OdeOptions<T: Float> {
    /// Absolute tolerance for adaptive methods.
    pub atol: T,
    /// Relative tolerance for adaptive methods.
    pub rtol: T,
    /// Initial step size. If `None`, a reasonable default is chosen.
    pub first_step: Option<T>,
    /// Maximum number of steps before giving up.
    pub max_steps: usize,
    /// Optional event function. Integration terminates when the return
    /// value crosses zero.
    pub event_fn: Option<EventFn<T>>,
}

impl<T: Float> Default for OdeOptions<T> {
    fn default() -> Self {
        Self {
            atol: T::from_f64(1e-8),
            rtol: T::from_f64(1e-6),
            first_step: None,
            max_steps: 10_000,
            event_fn: None,
        }
    }
}

/// Available ODE solver methods.
///
/// # Examples
///
/// ```
/// # use scivex_optim::ode::OdeMethod;
/// let method = OdeMethod::RK45;
/// assert_eq!(method, OdeMethod::RK45);
/// ```
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OdeMethod {
    /// Forward Euler — first-order, fixed step. Simple but inaccurate.
    Euler,
    /// Dormand-Prince RK4(5) — adaptive, general-purpose. Best for non-stiff problems.
    RK45,
    /// BDF-2 — implicit, fixed step. Best for stiff problems.
    BDF2,
}

/// Solve an initial value problem (IVP) for a system of ODEs.
///
/// # Arguments
///
/// * `f` — right-hand side function: `f(t, y) -> dy/dt`
/// * `t_span` — integration interval `[t0, tf]`
/// * `y0` — initial state vector
/// * `method` — solver method to use
/// * `options` — solver options (tolerances, step size, etc.)
///
/// # Returns
///
/// An [`OdeResult`] containing the time values and solution trajectory.
///
/// # Examples
///
/// ```
/// # use scivex_optim::ode::{solve_ivp, OdeMethod, OdeOptions};
/// // dy/dt = -y, y(0) = 1  →  y(t) = e^(-t)
/// let result = solve_ivp(
///     |_t: f64, y: &[f64]| vec![-y[0]],
///     [0.0, 1.0],
///     &[1.0],
///     OdeMethod::RK45,
///     &OdeOptions::default(),
/// ).unwrap();
/// let y_final = result.y.last().unwrap()[0];
/// assert!((y_final - (-1.0_f64).exp()).abs() < 1e-6);
/// ```
pub fn solve_ivp<T, F>(
    f: F,
    t_span: [T; 2],
    y0: &[T],
    method: OdeMethod,
    options: &OdeOptions<T>,
) -> Result<OdeResult<T>>
where
    T: Float,
    F: Fn(T, &[T]) -> Vec<T>,
{
    match method {
        OdeMethod::Euler => euler::euler(f, t_span, y0, options),
        OdeMethod::RK45 => rk45::rk45(f, t_span, y0, options),
        OdeMethod::BDF2 => bdf::bdf2(f, t_span, y0, options),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solve_ivp_rk45() {
        let result = solve_ivp(
            |_t: f64, y: &[f64]| vec![-y[0]],
            [0.0, 1.0],
            &[1.0],
            OdeMethod::RK45,
            &OdeOptions::default(),
        )
        .unwrap();

        let y_final = result.y.last().unwrap()[0];
        let expected = (-1.0_f64).exp();
        assert!((y_final - expected).abs() < 1e-6);
    }

    #[test]
    fn test_solve_ivp_euler() {
        let result = solve_ivp(
            |_t: f64, y: &[f64]| vec![-y[0]],
            [0.0, 1.0],
            &[1.0],
            OdeMethod::Euler,
            &OdeOptions::default(),
        )
        .unwrap();

        let y_final = result.y.last().unwrap()[0];
        let expected = (-1.0_f64).exp();
        assert!((y_final - expected).abs() < 0.02);
    }

    #[test]
    fn test_solve_ivp_bdf2() {
        let result = solve_ivp(
            |_t: f64, y: &[f64]| vec![-y[0]],
            [0.0, 1.0],
            &[1.0],
            OdeMethod::BDF2,
            &OdeOptions::default(),
        )
        .unwrap();

        let y_final = result.y.last().unwrap()[0];
        let expected = (-1.0_f64).exp();
        assert!((y_final - expected).abs() < 1e-3);
    }

    #[test]
    fn test_event_detection() {
        // dy/dt = 1, y(0) = -1. Event: y = 0 at t = 1.
        let result = solve_ivp(
            |_t: f64, _y: &[f64]| vec![1.0],
            [0.0, 5.0],
            &[-1.0],
            OdeMethod::RK45,
            &OdeOptions {
                event_fn: Some(Box::new(|_t: f64, y: &[f64]| y[0])),
                ..OdeOptions::default()
            },
        )
        .unwrap();

        // Should stop early, around t=1
        let t_final = *result.t.last().unwrap();
        assert!(
            t_final < 2.0,
            "Should have stopped early at event, t_final={t_final}"
        );
    }

    #[test]
    fn test_ode_result_trajectory() {
        let result = solve_ivp(
            |_t: f64, _y: &[f64]| vec![1.0],
            [0.0, 1.0],
            &[0.0],
            OdeMethod::RK45,
            &OdeOptions::default(),
        )
        .unwrap();

        // Trajectory should be monotonically increasing
        for i in 1..result.y.len() {
            assert!(result.y[i][0] >= result.y[i - 1][0]);
            assert!(result.t[i] > result.t[i - 1]);
        }
    }

    #[test]
    fn test_lotka_volterra() {
        // Classic predator-prey: dx/dt = x - x*y, dy/dt = -y + x*y
        // Oscillatory, conservative system
        let result = solve_ivp(
            |_t: f64, y: &[f64]| vec![y[0] - y[0] * y[1], -y[1] + y[0] * y[1]],
            [0.0, 10.0],
            &[1.0, 0.5],
            OdeMethod::RK45,
            &OdeOptions::default(),
        )
        .unwrap();

        assert!(result.success);
        // Both populations should remain positive
        for y in &result.y {
            assert!(y[0] > 0.0, "prey went negative");
            assert!(y[1] > 0.0, "predator went negative");
        }
    }
}
