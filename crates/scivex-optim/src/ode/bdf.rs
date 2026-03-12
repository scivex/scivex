//! BDF (Backward Differentiation Formula) method for stiff ODE systems.
//!
//! Implements BDF-2: `y_{n+1} = (4/3)*y_n - (1/3)*y_{n-1} + (2/3)*h*f(t_{n+1}, y_{n+1})`.
//! Implicit method suitable for stiff problems. Uses fixed-point iteration
//! to solve the implicit equation at each step.

use scivex_core::Float;

use super::{OdeOptions, OdeResult};
use crate::error::{OptimError, Result};

const FIXED_POINT_MAX_ITER: usize = 50;
const FIXED_POINT_TOL: f64 = 1e-10;

/// Solve a stiff ODE system using the BDF-2 method.
///
/// `f(t, y) -> dy/dt` defines the system. `y0` is the initial state vector.
/// Uses BDF-1 (backward Euler) for the first step, then BDF-2 for subsequent steps.
#[allow(clippy::too_many_lines)]
pub fn bdf2<T, F>(f: F, t_span: [T; 2], y0: &[T], options: &OdeOptions<T>) -> Result<OdeResult<T>>
where
    T: Float,
    F: Fn(T, &[T]) -> Vec<T>,
{
    let t0 = t_span[0];
    let tf = t_span[1];
    let n = y0.len();
    let h = options
        .first_step
        .unwrap_or_else(|| (tf - t0) / T::from_f64(200.0));
    let max_steps = options.max_steps;

    let mut t = t0;
    let mut t_values = vec![t];
    let mut y_values = vec![y0.to_vec()];
    let mut n_evals: usize = 0;
    let mut n_steps: usize = 0;

    // BDF-1 (backward Euler) for first step: y1 = y0 + h*f(t1, y1)
    // Solve by fixed-point iteration: y1^{k+1} = y0 + h*f(t1, y1^k)
    let t1 = t + h.min(tf - t);
    let h_actual = t1 - t;
    let mut y1 = y0.to_vec();

    // Initial guess using forward Euler
    let dy0 = f(t, &y1);
    n_evals += 1;
    for i in 0..n {
        y1[i] += h_actual * dy0[i];
    }

    // Fixed-point iteration for backward Euler
    for _ in 0..FIXED_POINT_MAX_ITER {
        let dy = f(t1, &y1);
        n_evals += 1;
        let mut max_diff = T::zero();
        for i in 0..n {
            let y_new = y0[i] + h_actual * dy[i];
            let diff = (y_new - y1[i]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            y1[i] = y_new;
        }
        if max_diff < T::from_f64(FIXED_POINT_TOL) {
            break;
        }
    }

    t = t1;
    t_values.push(t);
    y_values.push(y1.clone());
    n_steps += 1;

    if t >= tf {
        return Ok(OdeResult {
            t: t_values,
            y: y_values,
            n_evals,
            n_steps,
            success: true,
        });
    }

    // BDF-2 for subsequent steps
    // y_{n+1} = (4/3)*y_n - (1/3)*y_{n-1} + (2/3)*h*f(t_{n+1}, y_{n+1})
    let four_thirds = T::from_f64(4.0 / 3.0);
    let one_third = T::from_f64(1.0 / 3.0);
    let two_thirds = T::from_f64(2.0 / 3.0);

    while t < tf {
        if n_steps >= max_steps {
            return Err(OptimError::ConvergenceFailure {
                iterations: n_steps,
            });
        }

        let step = h.min(tf - t);
        let t_next = t + step;

        let y_nm1 = &y_values[y_values.len() - 2];
        let y_n = &y_values[y_values.len() - 1];

        // Predictor: extrapolate from previous two points
        let mut y_next = vec![T::zero(); n];
        for i in 0..n {
            y_next[i] = four_thirds * y_n[i] - one_third * y_nm1[i];
        }

        // Fixed-point iteration for BDF-2
        let mut converged = false;
        for _ in 0..FIXED_POINT_MAX_ITER {
            let dy = f(t_next, &y_next);
            n_evals += 1;
            let mut max_diff = T::zero();
            for i in 0..n {
                let y_new = four_thirds * y_n[i] - one_third * y_nm1[i] + two_thirds * step * dy[i];
                let diff = (y_new - y_next[i]).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
                y_next[i] = y_new;
            }
            if max_diff < T::from_f64(FIXED_POINT_TOL) {
                converged = true;
                break;
            }
        }

        if !converged {
            return Err(OptimError::ConvergenceFailure {
                iterations: n_steps,
            });
        }

        t = t_next;
        t_values.push(t);
        y_values.push(y_next);
        n_steps += 1;

        // Event detection
        if let Some(ref event_fn) = options.event_fn {
            let y_cur = &y_values[y_values.len() - 1];
            let val = event_fn(t, y_cur);
            if val.abs() < T::from_f64(1e-12)
                || (t_values.len() > 1 && {
                    let prev_y = &y_values[y_values.len() - 2];
                    let prev_t = t_values[t_values.len() - 2];
                    let prev_val = event_fn(prev_t, prev_y);
                    (prev_val > T::zero()) != (val > T::zero())
                })
            {
                break;
            }
        }
    }

    Ok(OdeResult {
        t: t_values,
        y: y_values,
        n_evals,
        n_steps,
        success: true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bdf2_exponential_decay() {
        // dy/dt = -y, y(0) = 1 => y(t) = e^(-t)
        let result = bdf2(
            |_t: f64, y: &[f64]| vec![-y[0]],
            [0.0, 1.0],
            &[1.0],
            &OdeOptions::default(),
        )
        .unwrap();

        let y_final = result.y.last().unwrap()[0];
        let expected = (-1.0_f64).exp();
        assert!(
            (y_final - expected).abs() < 1e-4,
            "y_final={y_final}, expected={expected}"
        );
    }

    #[test]
    fn test_bdf2_stiff_system() {
        // dy/dt = -50*y, y(0) = 1 => y(t) = e^(-50t)
        // This is stiff — BDF should handle it well
        let result = bdf2(
            |_t: f64, y: &[f64]| vec![-50.0 * y[0]],
            [0.0, 0.5],
            &[1.0],
            &OdeOptions {
                first_step: Some(0.002),
                max_steps: 5000,
                ..OdeOptions::default()
            },
        )
        .unwrap();

        let y_final = result.y.last().unwrap()[0];
        let expected = (-25.0_f64).exp();
        // BDF-2 on stiff problems: allow moderate tolerance
        assert!(
            (y_final - expected).abs() < 1e-3,
            "y_final={y_final}, expected={expected}, err={}",
            (y_final - expected).abs()
        );
        assert!(result.success);
    }

    #[test]
    fn test_bdf2_linear() {
        // dy/dt = 1, y(0) = 0 => y(t) = t
        let result = bdf2(
            |_t: f64, _y: &[f64]| vec![1.0],
            [0.0, 2.0],
            &[0.0],
            &OdeOptions::default(),
        )
        .unwrap();

        let y_final = result.y.last().unwrap()[0];
        assert!(
            (y_final - 2.0).abs() < 1e-6,
            "y_final={y_final}, expected=2.0"
        );
    }

    #[test]
    fn test_bdf2_system() {
        // Coupled stiff system: dy0/dt = -20*y0 + y1, dy1/dt = y0 - 20*y1
        // Both decay rapidly
        let result = bdf2(
            |_t: f64, y: &[f64]| vec![-20.0 * y[0] + y[1], y[0] - 20.0 * y[1]],
            [0.0, 1.0],
            &[1.0, 0.0],
            &OdeOptions {
                first_step: Some(0.002),
                max_steps: 5000,
                ..OdeOptions::default()
            },
        )
        .unwrap();

        let y_final = &result.y.last().unwrap();
        // Both components should be very small after t=1
        assert!(
            y_final[0].abs() < 1e-3,
            "y0={} should be near zero",
            y_final[0]
        );
        assert!(
            y_final[1].abs() < 1e-3,
            "y1={} should be near zero",
            y_final[1]
        );
    }
}
