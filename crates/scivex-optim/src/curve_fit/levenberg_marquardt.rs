//! Levenberg-Marquardt algorithm for non-linear least squares.
//!
//! Solves: minimize Σ (y_i - model(x_i, params))^2
//!
//! Uses damped Gauss-Newton with numerical Jacobian via forward differences.
//! The damping parameter λ is adjusted by the gain ratio.

use scivex_core::Float;

use crate::error::{OptimError, Result};

use super::LeastSquaresResult;

/// Solve a non-linear least squares problem using the Levenberg-Marquardt algorithm.
///
/// `model(x, params)` computes the model prediction at point `x` given parameters.
/// `x_data` and `y_data` are the observed data points. `p0` is the initial parameter
/// guess. The algorithm iterates until the cost change is below `tol` or `max_iter`
/// iterations are reached.
pub fn levenberg_marquardt<T, F>(
    model: F,
    x_data: &[T],
    y_data: &[T],
    p0: &[T],
    max_iter: usize,
    tol: T,
) -> Result<LeastSquaresResult<T>>
where
    T: Float,
    F: Fn(T, &[T]) -> T,
{
    let n_data = x_data.len();
    let n_params = p0.len();

    if n_data != y_data.len() {
        return Err(OptimError::InvalidParameter {
            name: "y_data",
            reason: "must have same length as x_data",
        });
    }
    if n_data < n_params {
        return Err(OptimError::InvalidParameter {
            name: "p0",
            reason: "number of data points must be >= number of parameters",
        });
    }

    let mut params = p0.to_vec();
    let mut residuals = compute_residuals(&model, x_data, y_data, &params);
    let mut cost = residuals.iter().map(|&r| r * r).sum::<T>();

    // Initial damping parameter
    let mut lambda = T::from_f64(1e-3);
    let nu = T::from_f64(2.0);

    for iter in 0..max_iter {
        // Compute Jacobian numerically (forward differences)
        let jac = numerical_jacobian(&model, x_data, &params, n_data, n_params);

        // Compute J^T J and J^T r
        let mut jtj = vec![T::zero(); n_params * n_params];
        let mut jtr = vec![T::zero(); n_params];

        for i in 0..n_params {
            for j in 0..n_params {
                let mut sum = T::zero();
                for k in 0..n_data {
                    sum += jac[k * n_params + i] * jac[k * n_params + j];
                }
                jtj[i * n_params + j] = sum;
            }
            let mut sum = T::zero();
            for k in 0..n_data {
                sum += jac[k * n_params + i] * residuals[k];
            }
            jtr[i] = sum;
        }

        // Solve (J^T J + λ I) δ = -J^T r
        let mut a = jtj.clone();
        for i in 0..n_params {
            a[i * n_params + i] += lambda;
        }
        let neg_jtr: Vec<T> = jtr.iter().map(|&v| -v).collect();

        let Some(delta) = solve_linear_system(&a, &neg_jtr, n_params) else {
            // Singular system; increase damping
            lambda *= nu;
            continue;
        };

        // Trial step
        let params_trial: Vec<T> = params
            .iter()
            .zip(delta.iter())
            .map(|(&p, &d)| p + d)
            .collect();
        let residuals_trial = compute_residuals(&model, x_data, y_data, &params_trial);
        let cost_trial = residuals_trial.iter().map(|&r| r * r).sum::<T>();

        // Gain ratio
        // predicted reduction = -2 * delta^T * J^T r - delta^T * J^T J * delta
        let mut predicted = T::zero();
        for i in 0..n_params {
            predicted += delta[i] * (lambda * delta[i] - jtr[i]);
        }

        if predicted.abs() < T::epsilon() {
            // No predicted improvement
            lambda *= nu;
            continue;
        }

        let gain = (cost - cost_trial) / predicted;

        if gain > T::zero() {
            // Accept step
            params = params_trial;
            residuals = residuals_trial;
            let old_cost = cost;
            cost = cost_trial;

            // Adjust lambda
            let three = T::from_f64(3.0);
            let factor = T::one() - (T::from_f64(2.0) * gain - T::one()).powi(3);
            lambda *= factor.max(T::one() / three);

            // Check convergence
            let cost_change = (old_cost - cost).abs();
            if cost < tol || cost_change < tol * cost.max(T::one()) {
                return Ok(LeastSquaresResult {
                    params,
                    residuals,
                    cost,
                    iterations: iter + 1,
                    converged: true,
                });
            }
        } else {
            // Reject step, increase damping
            lambda *= nu;
        }
    }

    Ok(LeastSquaresResult {
        params,
        residuals,
        cost,
        iterations: max_iter,
        converged: false,
    })
}

/// Compute residuals: r_i = y_i - model(x_i, params).
fn compute_residuals<T: Float, F: Fn(T, &[T]) -> T>(
    model: &F,
    x_data: &[T],
    y_data: &[T],
    params: &[T],
) -> Vec<T> {
    x_data
        .iter()
        .zip(y_data.iter())
        .map(|(&x, &y)| y - model(x, params))
        .collect()
}

/// Compute the Jacobian matrix numerically using forward differences.
///
/// J[i][j] = ∂r_i/∂p_j ≈ (r_i(p + h*e_j) - r_i(p)) / h
/// Since r_i = y_i - model(x_i, p), we get ∂r_i/∂p_j = -∂model/∂p_j.
fn numerical_jacobian<T: Float, F: Fn(T, &[T]) -> T>(
    model: &F,
    x_data: &[T],
    params: &[T],
    n_data: usize,
    n_params: usize,
) -> Vec<T> {
    let h = T::from_f64(1e-8);
    let mut jac = vec![T::zero(); n_data * n_params];

    let mut params_plus = params.to_vec();

    for j in 0..n_params {
        let orig = params[j];
        let step = if orig.abs() > T::epsilon() {
            h * orig.abs()
        } else {
            h
        };
        params_plus[j] = orig + step;

        for i in 0..n_data {
            let f_plus = model(x_data[i], &params_plus);
            let f_orig = model(x_data[i], params);
            // dr_i/dp_j = -(df/dp_j) since r = y - f
            jac[i * n_params + j] = -(f_plus - f_orig) / step;
        }

        params_plus[j] = orig;
    }

    jac
}

/// Solve Ax = b for a small dense system using Gaussian elimination with partial pivoting.
fn solve_linear_system<T: Float>(a: &[T], b: &[T], n: usize) -> Option<Vec<T>> {
    let mut aug = vec![T::zero(); n * (n + 1)];
    for i in 0..n {
        for j in 0..n {
            aug[i * (n + 1) + j] = a[i * n + j];
        }
        aug[i * (n + 1) + n] = b[i];
    }

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_val = aug[col * (n + 1) + col].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            let val = aug[row * (n + 1) + col].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        if max_val < T::from_f64(1e-14) {
            return None; // Singular
        }

        // Swap rows
        if max_row != col {
            for j in 0..=n {
                aug.swap(col * (n + 1) + j, max_row * (n + 1) + j);
            }
        }

        // Eliminate
        let pivot = aug[col * (n + 1) + col];
        for row in (col + 1)..n {
            let factor = aug[row * (n + 1) + col] / pivot;
            for j in col..=n {
                let val = aug[col * (n + 1) + j];
                aug[row * (n + 1) + j] -= factor * val;
            }
        }
    }

    // Back substitution
    let mut x = vec![T::zero(); n];
    for i in (0..n).rev() {
        let mut sum = aug[i * (n + 1) + n];
        for j in (i + 1)..n {
            sum -= aug[i * (n + 1) + j] * x[j];
        }
        x[i] = sum / aug[i * (n + 1) + i];
    }

    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lm_exponential_decay() {
        // Model: y = a * exp(-b * x)
        // True params: a=2.0, b=0.5
        let true_a = 2.0_f64;
        let true_b = 0.5_f64;

        let x_data: Vec<f64> = (0..20).map(|i| f64::from(i) * 0.5).collect();
        let y_data: Vec<f64> = x_data
            .iter()
            .map(|&x| true_a * (-true_b * x).exp())
            .collect();

        let model = |x: f64, p: &[f64]| -> f64 { p[0] * (-p[1] * x).exp() };

        let result = levenberg_marquardt(model, &x_data, &y_data, &[1.0, 1.0], 200, 1e-12).unwrap();
        assert!(
            result.converged,
            "did not converge after {} iters",
            result.iterations
        );
        assert!(
            (result.params[0] - true_a).abs() < 1e-4,
            "a = {}",
            result.params[0]
        );
        assert!(
            (result.params[1] - true_b).abs() < 1e-4,
            "b = {}",
            result.params[1]
        );
    }

    #[test]
    fn test_lm_polynomial() {
        // Model: y = a*x^2 + b*x + c
        // True params: a=1.0, b=-2.0, c=3.0
        let x_data: Vec<f64> = (-5..=5).map(f64::from).collect();
        let y_data: Vec<f64> = x_data
            .iter()
            .map(|&x| 1.0 * x * x - 2.0 * x + 3.0)
            .collect();

        let model = |x: f64, p: &[f64]| -> f64 { p[0] * x * x + p[1] * x + p[2] };

        let result =
            levenberg_marquardt(model, &x_data, &y_data, &[0.5, -1.0, 1.0], 100, 1e-12).unwrap();
        assert!(result.converged);
        assert!(
            (result.params[0] - 1.0).abs() < 1e-3,
            "a = {}",
            result.params[0]
        );
        assert!(
            (result.params[1] - (-2.0)).abs() < 1e-3,
            "b = {}",
            result.params[1]
        );
        assert!(
            (result.params[2] - 3.0).abs() < 1e-3,
            "c = {}",
            result.params[2]
        );
    }

    #[test]
    fn test_lm_linear_exact() {
        // Model: y = a*x + b
        // True: a=2.0, b=1.0
        let x_data: Vec<f64> = (0..10).map(f64::from).collect();
        let y_data: Vec<f64> = x_data.iter().map(|&x| 2.0 * x + 1.0).collect();

        let model = |x: f64, p: &[f64]| -> f64 { p[0] * x + p[1] };

        let result = levenberg_marquardt(model, &x_data, &y_data, &[0.0, 0.0], 100, 1e-12).unwrap();
        assert!(result.converged);
        assert!((result.params[0] - 2.0).abs() < 1e-6);
        assert!((result.params[1] - 1.0).abs() < 1e-6);
        assert!(result.cost < 1e-12);
    }

    #[test]
    fn test_lm_bad_initial_guess() {
        // Same exponential model but with a very poor initial guess
        let x_data: Vec<f64> = (0..20).map(|i| f64::from(i) * 0.5).collect();
        let y_data: Vec<f64> = x_data.iter().map(|&x| 2.0 * (-0.5 * x).exp()).collect();

        let model = |x: f64, p: &[f64]| -> f64 { p[0] * (-p[1] * x).exp() };

        let result =
            levenberg_marquardt(model, &x_data, &y_data, &[10.0, 5.0], 200, 1e-10).unwrap();
        // Should still converge (LM is robust)
        assert!(result.converged, "cost = {}", result.cost);
        assert!((result.params[0] - 2.0).abs() < 1e-2);
        assert!((result.params[1] - 0.5).abs() < 1e-2);
    }
}
