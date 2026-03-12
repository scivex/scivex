//! Forward Euler method for ODE initial value problems.
//!
//! The simplest first-order explicit method: `y_{n+1} = y_n + h * f(t_n, y_n)`.
//! First-order accurate (`O(h)`), not recommended for stiff problems.

use scivex_core::Float;

use super::{OdeOptions, OdeResult};
use crate::error::Result;

/// Solve an ODE system using the forward Euler method.
///
/// `f(t, y) -> dy/dt` defines the system. `y0` is the initial state vector.
pub fn euler<T, F>(f: F, t_span: [T; 2], y0: &[T], options: &OdeOptions<T>) -> Result<OdeResult<T>>
where
    T: Float,
    F: Fn(T, &[T]) -> Vec<T>,
{
    let t0 = t_span[0];
    let tf = t_span[1];
    let h = options
        .first_step
        .unwrap_or_else(|| (tf - t0) / T::from_f64(100.0));
    let n = y0.len();

    let mut t = t0;
    let mut y = y0.to_vec();
    let mut t_values = vec![t];
    let mut y_values = vec![y.clone()];
    let mut n_evals: usize = 0;

    while t < tf {
        // Don't step past tf
        let step = if t + h > tf { tf - t } else { h };

        let dy = f(t, &y);
        n_evals += 1;

        for i in 0..n {
            y[i] += step * dy[i];
        }
        t += step;

        t_values.push(t);
        y_values.push(y.clone());

        // Event detection
        if let Some(ref event_fn) = options.event_fn {
            let val = event_fn(t, &y);
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

    let n_steps = y_values.len() - 1;
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
    fn test_euler_exponential_decay() {
        // dy/dt = -y, y(0) = 1 => y(t) = e^(-t)
        let result = euler(
            |_t: f64, y: &[f64]| vec![-y[0]],
            [0.0, 1.0],
            &[1.0],
            &OdeOptions::default(),
        )
        .unwrap();

        let y_final = result.y.last().unwrap()[0];
        let expected = (-1.0_f64).exp();
        // Euler is only first-order, so allow some error
        assert!(
            (y_final - expected).abs() < 0.02,
            "y_final={y_final}, expected={expected}"
        );
    }

    #[test]
    fn test_euler_linear() {
        // dy/dt = 1, y(0) = 0 => y(t) = t
        let result = euler(
            |_t: f64, _y: &[f64]| vec![1.0],
            [0.0, 2.0],
            &[0.0],
            &OdeOptions::default(),
        )
        .unwrap();

        let y_final = result.y.last().unwrap()[0];
        assert!(
            (y_final - 2.0).abs() < 1e-10,
            "y_final={y_final}, expected=2.0"
        );
    }

    #[test]
    fn test_euler_system() {
        // dy0/dt = y1, dy1/dt = -y0 (harmonic oscillator)
        // y(0) = [1, 0] => y(t) = [cos(t), -sin(t)]
        let opts = OdeOptions {
            first_step: Some(0.001),
            ..OdeOptions::default()
        };
        let result = euler(
            |_t: f64, y: &[f64]| vec![y[1], -y[0]],
            [0.0, 1.0],
            &[1.0, 0.0],
            &opts,
        )
        .unwrap();

        let y_final = &result.y.last().unwrap();
        let expected_y0 = 1.0_f64.cos();
        let expected_y1 = -(1.0_f64.sin());
        assert!(
            (y_final[0] - expected_y0).abs() < 0.01,
            "y0={}, expected={}",
            y_final[0],
            expected_y0
        );
        assert!(
            (y_final[1] - expected_y1).abs() < 0.01,
            "y1={}, expected={}",
            y_final[1],
            expected_y1
        );
    }

    #[test]
    fn test_euler_stores_trajectory() {
        let result = euler(
            |_t: f64, _y: &[f64]| vec![1.0],
            [0.0, 1.0],
            &[0.0],
            &OdeOptions::default(),
        )
        .unwrap();

        assert!(result.t.len() > 2);
        assert_eq!(result.t.len(), result.y.len());
        assert!((result.t[0] - 0.0).abs() < 1e-12);
        assert!((*result.t.last().unwrap() - 1.0).abs() < 1e-12);
    }
}
