//! Dormand-Prince RK4(5) adaptive ODE solver.
//!
//! An embedded Runge-Kutta method that provides both 4th and 5th order
//! solutions, using the difference for local error estimation and
//! automatic step size control.

use scivex_core::Float;

use super::{OdeOptions, OdeResult};
use crate::error::{OptimError, Result};

// Dormand-Prince Butcher tableau coefficients
const A21: f64 = 1.0 / 5.0;
const A31: f64 = 3.0 / 40.0;
const A32: f64 = 9.0 / 40.0;
const A41: f64 = 44.0 / 45.0;
const A42: f64 = -56.0 / 15.0;
const A43: f64 = 32.0 / 9.0;
const A51: f64 = 19_372.0 / 6_561.0;
const A52: f64 = -25_360.0 / 2_187.0;
const A53: f64 = 64_448.0 / 6_561.0;
const A54: f64 = -212.0 / 729.0;
const A61: f64 = 9_017.0 / 3_168.0;
const A62: f64 = -355.0 / 33.0;
const A63: f64 = 46_732.0 / 5_247.0;
const A64: f64 = 49.0 / 176.0;
const A65: f64 = -5_103.0 / 18_656.0;
const A71: f64 = 35.0 / 384.0;
// A72 = 0
const A73: f64 = 500.0 / 1_113.0;
const A74: f64 = 125.0 / 192.0;
const A75: f64 = -2_187.0 / 6_784.0;
const A76: f64 = 11.0 / 84.0;

// 5th order weights (for the solution)
const B1: f64 = A71;
// B2 = 0
const B3: f64 = A73;
const B4: f64 = A74;
const B5: f64 = A75;
const B6: f64 = A76;
// B7 = 0

// 4th order weights (for error estimation)
const E1: f64 = 5_179.0 / 57_600.0;
// E2 = 0
const E3: f64 = 7_571.0 / 16_695.0;
const E4: f64 = 393.0 / 640.0;
const E5: f64 = -92_097.0 / 339_200.0;
const E6: f64 = 187.0 / 2_100.0;
const E7: f64 = 1.0 / 40.0;

// Nodes
const C2: f64 = 1.0 / 5.0;
const C3: f64 = 3.0 / 10.0;
const C4: f64 = 4.0 / 5.0;
const C5: f64 = 8.0 / 9.0;
// C6 = 1.0, C7 = 1.0

/// Solve an ODE system using the Dormand-Prince RK4(5) adaptive method.
///
/// `f(t, y) -> dy/dt` defines the system. `y0` is the initial state vector.
/// Step size is automatically adjusted to maintain the specified tolerances.
#[allow(clippy::too_many_lines)]
pub fn rk45<T, F>(f: F, t_span: [T; 2], y0: &[T], options: &OdeOptions<T>) -> Result<OdeResult<T>>
where
    T: Float,
    F: Fn(T, &[T]) -> Vec<T>,
{
    let t0 = t_span[0];
    let tf = t_span[1];
    let n = y0.len();
    let atol = options.atol;
    let rtol = options.rtol;
    let max_steps = options.max_steps;

    // Initial step size
    let mut h = options.first_step.unwrap_or_else(|| {
        let span = tf - t0;
        let h0 = span / T::from_f64(100.0);
        // Clamp to reasonable range
        if h0 > span / T::from_f64(10.0) {
            span / T::from_f64(10.0)
        } else {
            h0
        }
    });

    let h_min = (tf - t0) * T::from_f64(1e-12);
    let h_max = tf - t0;

    let mut t = t0;
    let mut y = y0.to_vec();
    let mut t_values = vec![t];
    let mut y_values = vec![y.clone()];
    let mut n_evals: usize = 0;
    let mut n_steps: usize = 0;

    // Temporary stage vectors
    let mut k1 = vec![T::zero(); n];
    let mut k2 = vec![T::zero(); n];
    let mut k3 = vec![T::zero(); n];
    let mut k4 = vec![T::zero(); n];
    let mut k5 = vec![T::zero(); n];
    let mut k6 = vec![T::zero(); n];
    let mut k7 = vec![T::zero(); n];
    let mut y_tmp = vec![T::zero(); n];

    while t < tf {
        if n_steps >= max_steps {
            return Err(OptimError::ConvergenceFailure {
                iterations: n_steps,
            });
        }

        // Don't step past tf
        if t + h > tf {
            h = tf - t;
        }

        let hf = h;

        // k1 = f(t, y)
        let k1_res = f(t, &y);
        k1[..n].copy_from_slice(&k1_res[..n]);

        // k2 = f(t + c2*h, y + h*(a21*k1))
        for i in 0..n {
            y_tmp[i] = y[i] + hf * T::from_f64(A21) * k1[i];
        }
        let k2_res = f(t + T::from_f64(C2) * hf, &y_tmp);
        k2[..n].copy_from_slice(&k2_res[..n]);

        // k3 = f(t + c3*h, y + h*(a31*k1 + a32*k2))
        for i in 0..n {
            y_tmp[i] = y[i] + hf * (T::from_f64(A31) * k1[i] + T::from_f64(A32) * k2[i]);
        }
        let k3_res = f(t + T::from_f64(C3) * hf, &y_tmp);
        k3[..n].copy_from_slice(&k3_res[..n]);

        // k4 = f(t + c4*h, y + h*(a41*k1 + a42*k2 + a43*k3))
        for i in 0..n {
            y_tmp[i] = y[i]
                + hf * (T::from_f64(A41) * k1[i]
                    + T::from_f64(A42) * k2[i]
                    + T::from_f64(A43) * k3[i]);
        }
        let k4_res = f(t + T::from_f64(C4) * hf, &y_tmp);
        k4[..n].copy_from_slice(&k4_res[..n]);

        // k5 = f(t + c5*h, y + h*(a51*k1 + ... + a54*k4))
        for i in 0..n {
            y_tmp[i] = y[i]
                + hf * (T::from_f64(A51) * k1[i]
                    + T::from_f64(A52) * k2[i]
                    + T::from_f64(A53) * k3[i]
                    + T::from_f64(A54) * k4[i]);
        }
        let k5_res = f(t + T::from_f64(C5) * hf, &y_tmp);
        k5[..n].copy_from_slice(&k5_res[..n]);

        // k6 = f(t + h, y + h*(a61*k1 + ... + a65*k5))
        for i in 0..n {
            y_tmp[i] = y[i]
                + hf * (T::from_f64(A61) * k1[i]
                    + T::from_f64(A62) * k2[i]
                    + T::from_f64(A63) * k3[i]
                    + T::from_f64(A64) * k4[i]
                    + T::from_f64(A65) * k5[i]);
        }
        let k6_res = f(t + hf, &y_tmp);
        k6[..n].copy_from_slice(&k6_res[..n]);

        // 5th order solution: y_new = y + h * (b1*k1 + b3*k3 + b4*k4 + b5*k5 + b6*k6)
        let mut y_new = vec![T::zero(); n];
        for i in 0..n {
            y_new[i] = y[i]
                + hf * (T::from_f64(B1) * k1[i]
                    + T::from_f64(B3) * k3[i]
                    + T::from_f64(B4) * k4[i]
                    + T::from_f64(B5) * k5[i]
                    + T::from_f64(B6) * k6[i]);
        }

        // k7 = f(t + h, y_new)
        let k7_res = f(t + hf, &y_new);
        k7[..n].copy_from_slice(&k7_res[..n]);

        n_evals += 7;

        // Error estimate: difference between 5th and 4th order solutions
        let mut err_norm = T::zero();
        for i in 0..n {
            let err_i = hf
                * ((T::from_f64(E1) - T::from_f64(B1)) * k1[i]
                    + (T::from_f64(E3) - T::from_f64(B3)) * k3[i]
                    + (T::from_f64(E4) - T::from_f64(B4)) * k4[i]
                    + (T::from_f64(E5) - T::from_f64(B5)) * k5[i]
                    + (T::from_f64(E6) - T::from_f64(B6)) * k6[i]
                    + T::from_f64(E7) * k7[i]);
            let scale = atol + rtol * y[i].abs().max(y_new[i].abs());
            let ratio = err_i / scale;
            err_norm += ratio * ratio;
        }
        err_norm = (err_norm / T::from_f64(n as f64)).sqrt();

        if err_norm <= T::one() {
            // Step accepted
            t += hf;
            y = y_new;
            t_values.push(t);
            y_values.push(y.clone());
            n_steps += 1;

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

        // Adjust step size
        let safety = T::from_f64(0.9);
        let min_factor = T::from_f64(0.2);
        let max_factor = T::from_f64(5.0);

        let factor = if err_norm > T::from_f64(1e-15) {
            safety * (T::one() / err_norm).powf(T::from_f64(0.2))
        } else {
            max_factor
        };
        let factor = factor.max(min_factor).min(max_factor);
        h = (hf * factor).max(h_min).min(h_max);
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
    fn test_rk45_exponential_decay() {
        // dy/dt = -y, y(0) = 1 => y(t) = e^(-t)
        let result = rk45(
            |_t: f64, y: &[f64]| vec![-y[0]],
            [0.0, 1.0],
            &[1.0],
            &OdeOptions::default(),
        )
        .unwrap();

        let y_final = result.y.last().unwrap()[0];
        let expected = (-1.0_f64).exp();
        assert!(
            (y_final - expected).abs() < 1e-6,
            "y_final={y_final}, expected={expected}"
        );
    }

    #[test]
    fn test_rk45_linear() {
        // dy/dt = 2*t, y(0) = 0 => y(t) = t^2
        let result = rk45(
            |t: f64, _y: &[f64]| vec![2.0 * t],
            [0.0, 3.0],
            &[0.0],
            &OdeOptions::default(),
        )
        .unwrap();

        let y_final = result.y.last().unwrap()[0];
        assert!(
            (y_final - 9.0).abs() < 1e-8,
            "y_final={y_final}, expected=9.0"
        );
    }

    #[test]
    fn test_rk45_harmonic_oscillator() {
        // dy0/dt = y1, dy1/dt = -y0
        // y(0) = [1, 0] => y(t) = [cos(t), -sin(t)]
        let result = rk45(
            |_t: f64, y: &[f64]| vec![y[1], -y[0]],
            [0.0, std::f64::consts::TAU],
            &[1.0, 0.0],
            &OdeOptions::default(),
        )
        .unwrap();

        let y_final = &result.y.last().unwrap();
        // After one full period, should return to [1, 0]
        assert!(
            (y_final[0] - 1.0).abs() < 1e-6,
            "y0={}, expected=1.0",
            y_final[0]
        );
        assert!(y_final[1].abs() < 1e-6, "y1={}, expected=0.0", y_final[1]);
    }

    #[test]
    fn test_rk45_much_more_accurate_than_euler() {
        // dy/dt = -y, y(0) = 1
        let result = rk45(
            |_t: f64, y: &[f64]| vec![-y[0]],
            [0.0, 2.0],
            &[1.0],
            &OdeOptions::default(),
        )
        .unwrap();

        let y_final = result.y.last().unwrap()[0];
        let expected = (-2.0_f64).exp();
        // RK45 should be very accurate
        assert!(
            (y_final - expected).abs() < 1e-6,
            "RK45 error={} is too large",
            (y_final - expected).abs()
        );
    }

    #[test]
    fn test_rk45_adaptive_step_count() {
        // A smooth problem should use far fewer steps than 100
        let result = rk45(
            |_t: f64, y: &[f64]| vec![-y[0]],
            [0.0, 1.0],
            &[1.0],
            &OdeOptions::default(),
        )
        .unwrap();

        assert!(
            result.n_steps < 50,
            "n_steps={} is too many",
            result.n_steps
        );
        assert!(result.success);
    }

    #[test]
    fn test_rk45_stiff_problem_completes() {
        // Mildly stiff: dy/dt = -15*y
        let result = rk45(
            |_t: f64, y: &[f64]| vec![-15.0 * y[0]],
            [0.0, 1.0],
            &[1.0],
            &OdeOptions {
                max_steps: 5000,
                ..OdeOptions::default()
            },
        )
        .unwrap();

        let y_final = result.y.last().unwrap()[0];
        let expected = (-15.0_f64).exp();
        assert!(
            (y_final - expected).abs() < 1e-6,
            "y_final={y_final}, expected={expected}"
        );
    }
}
