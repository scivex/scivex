//! BFGS quasi-Newton method for unconstrained optimization.
//!
//! Uses a rank-2 update to approximate the inverse Hessian, combined
//! with a Wolfe line search for step-size selection.

use scivex_core::{Float, Tensor};

use crate::error::Result;

use super::{MinimizeOptions, MinimizeResult};

/// Minimize `f` using the BFGS quasi-Newton method.
///
/// Takes the objective function `f`, its gradient `grad`, an initial
/// point `x0`, and options. Returns a [`MinimizeResult`] with the
/// optimal point and convergence information.
pub fn bfgs<T, F, G>(
    f: F,
    grad: G,
    x0: &Tensor<T>,
    options: &MinimizeOptions<T>,
) -> Result<MinimizeResult<T>>
where
    T: Float,
    F: Fn(&Tensor<T>) -> T,
    G: Fn(&Tensor<T>) -> Tensor<T>,
{
    let n = x0.numel();
    let mut x = x0.clone();
    let mut fx = f(&x);
    let mut g = grad(&x);
    let mut f_evals = 1usize;
    let mut g_evals = 1usize;

    // Initialize inverse Hessian approximation as identity
    let mut h_inv = identity_matrix::<T>(n);

    for i in 0..options.max_iter {
        // Check gradient norm
        let grad_norm = vec_norm(&g);
        if grad_norm < options.gtol {
            return Ok(MinimizeResult {
                x,
                f_val: fx,
                grad: Some(g),
                iterations: i,
                f_evals,
                g_evals,
                converged: true,
            });
        }

        // Search direction: p = -H_inv * g
        let p = mat_vec_mul(&h_inv, &g, n).map(|v| -v);

        // Wolfe line search
        let (alpha, fx_new, evals) = wolfe_line_search(&f, &x, &p, fx, &g, options);
        f_evals += evals;

        // Compute s = alpha * p
        let s = &p * alpha;
        let x_new = &x + &s;

        // Check step size
        let step_norm = vec_norm(&s);
        if step_norm < options.xtol {
            let g_new = grad(&x_new);
            g_evals += 1;
            return Ok(MinimizeResult {
                x: x_new,
                f_val: fx_new,
                grad: Some(g_new),
                iterations: i + 1,
                f_evals,
                g_evals,
                converged: true,
            });
        }

        // Compute y = g_new - g_old
        let g_new = grad(&x_new);
        g_evals += 1;
        let y = &g_new - &g;

        // Compute s^T y
        let sy = vec_dot(&s, &y);

        // Skip update if curvature condition fails
        if sy > T::epsilon() {
            // BFGS inverse Hessian update:
            // H_{k+1} = (I - rho*s*y^T) * H_k * (I - rho*y*s^T) + rho*s*s^T
            // where rho = 1 / (y^T s)
            let rho = T::one() / sy;
            bfgs_update(&mut h_inv, &s, &y, rho, n);
        }

        // Check function value change
        let f_change = (fx - fx_new).abs();

        x = x_new;
        fx = fx_new;
        g = g_new;

        if f_change < options.ftol && f_change < options.ftol * fx.abs() {
            return Ok(MinimizeResult {
                x,
                f_val: fx,
                grad: Some(g),
                iterations: i + 1,
                f_evals,
                g_evals,
                converged: true,
            });
        }
    }

    Ok(MinimizeResult {
        x,
        f_val: fx,
        grad: Some(g),
        iterations: options.max_iter,
        f_evals,
        g_evals,
        converged: false,
    })
}

/// Create an n×n identity matrix as a flat Tensor.
fn identity_matrix<T: Float>(n: usize) -> Tensor<T> {
    let mut data = vec![T::zero(); n * n];
    for i in 0..n {
        data[i * n + i] = T::one();
    }
    Tensor::from_vec(data, vec![n, n]).expect("identity matrix data length matches n*n shape")
}

/// Matrix-vector product for an n×n matrix stored as a Tensor.
fn mat_vec_mul<T: Float>(mat: &Tensor<T>, v: &Tensor<T>, n: usize) -> Tensor<T> {
    let m = mat.as_slice();
    let v = v.as_slice();
    let mut result = vec![T::zero(); n];
    for i in 0..n {
        let mut sum = T::zero();
        for j in 0..n {
            sum += m[i * n + j] * v[j];
        }
        result[i] = sum;
    }
    Tensor::from_vec(result, vec![n]).expect("mat-vec product length matches n")
}

/// Dot product of two 1-D tensors.
fn vec_dot<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> T {
    a.as_slice()
        .iter()
        .zip(b.as_slice().iter())
        .map(|(&ai, &bi)| ai * bi)
        .sum()
}

/// L2 norm of a 1-D tensor.
fn vec_norm<T: Float>(v: &Tensor<T>) -> T {
    vec_dot(v, v).sqrt()
}

/// Apply the BFGS rank-2 update to the inverse Hessian in-place.
///
/// H_{k+1} = (I - rho*s*y^T) * H_k * (I - rho*y*s^T) + rho*s*s^T
fn bfgs_update<T: Float>(h: &mut Tensor<T>, s: &Tensor<T>, y: &Tensor<T>, rho: T, n: usize) {
    let s_data = s.as_slice();
    let y_data = y.as_slice();
    let h_data = h.as_mut_slice();

    // Compute H * y
    let mut hy = vec![T::zero(); n];
    for i in 0..n {
        let mut sum = T::zero();
        for j in 0..n {
            sum += h_data[i * n + j] * y_data[j];
        }
        hy[i] = sum;
    }

    // Compute y^T H y
    let yhy: T = y_data
        .iter()
        .zip(hy.iter())
        .map(|(&yi, &hyi)| yi * hyi)
        .sum();

    // Sherman-Morrison form of the BFGS update:
    // H_{k+1}[i][j] = H[i][j]
    //                - rho * (hy[i]*s[j] + s[i]*hy[j])
    //                + rho * (rho * yhy + 1) * s[i] * s[j]
    let factor = rho * (rho * yhy + T::one());
    for i in 0..n {
        for j in 0..n {
            h_data[i * n + j] = h_data[i * n + j] - rho * (hy[i] * s_data[j] + s_data[i] * hy[j])
                + factor * s_data[i] * s_data[j];
        }
    }
}

/// Backtracking line search with Armijo condition.
///
/// Returns `(step_size, f_new, n_function_evals)`.
fn wolfe_line_search<T, F>(
    f: &F,
    x: &Tensor<T>,
    p: &Tensor<T>,
    fx: T,
    g: &Tensor<T>,
    options: &MinimizeOptions<T>,
) -> (T, T, usize)
where
    T: Float,
    F: Fn(&Tensor<T>) -> T,
{
    let c1 = T::from_f64(1e-4);
    let rho = T::from_f64(0.5);

    let dg = vec_dot(g, p); // directional derivative g^T p

    if dg > T::zero() {
        // Not a descent direction; just take a tiny step
        let alpha = options.learning_rate;
        let x_new = x + &(p * alpha);
        let fx_new = f(&x_new);
        return (alpha, fx_new, 1);
    }

    let mut alpha = T::one();
    let mut evals = 0usize;
    let max_ls = 20usize;

    for _ in 0..max_ls {
        let x_trial = x + &(p * alpha);
        let f_trial = f(&x_trial);
        evals += 1;

        // Armijo condition
        if f_trial <= fx + c1 * alpha * dg {
            return (alpha, f_trial, evals);
        }

        alpha *= rho;
    }

    // If line search fails, return the last alpha
    let x_trial = x + &(p * alpha);
    let f_trial = f(&x_trial);
    evals += 1;
    (alpha, f_trial, evals)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bfgs_quadratic() {
        // f(x, y) = x^2 + y^2
        let f = |x: &Tensor<f64>| {
            let s = x.as_slice();
            s[0] * s[0] + s[1] * s[1]
        };
        let grad = |x: &Tensor<f64>| {
            let s = x.as_slice();
            Tensor::from_vec(vec![2.0 * s[0], 2.0 * s[1]], vec![2]).unwrap()
        };

        let x0 = Tensor::from_vec(vec![5.0, 5.0], vec![2]).unwrap();
        let result = bfgs(f, grad, &x0, &MinimizeOptions::default()).unwrap();
        assert!(
            result.converged,
            "did not converge after {} iters",
            result.iterations
        );
        let s = result.x.as_slice();
        assert!(s[0].abs() < 1e-6, "x = {}", s[0]);
        assert!(s[1].abs() < 1e-6, "y = {}", s[1]);
    }

    #[test]
    fn test_bfgs_rosenbrock() {
        // f(x, y) = (1 - x)^2 + 100(y - x^2)^2
        let f = |x: &Tensor<f64>| {
            let s = x.as_slice();
            let a = 1.0 - s[0];
            let b = s[1] - s[0] * s[0];
            a * a + 100.0 * b * b
        };
        let grad = |x: &Tensor<f64>| {
            let s = x.as_slice();
            let dx = -2.0 * (1.0 - s[0]) - 400.0 * s[0] * (s[1] - s[0] * s[0]);
            let dy = 200.0 * (s[1] - s[0] * s[0]);
            Tensor::from_vec(vec![dx, dy], vec![2]).unwrap()
        };

        let x0 = Tensor::from_vec(vec![-1.0, 1.0], vec![2]).unwrap();
        let opts = MinimizeOptions {
            max_iter: 5000,
            gtol: 1e-6,
            ..MinimizeOptions::default()
        };

        let result = bfgs(f, grad, &x0, &opts).unwrap();
        assert!(
            result.converged,
            "did not converge after {} iters, f = {}",
            result.iterations, result.f_val
        );
        let s = result.x.as_slice();
        assert!((s[0] - 1.0).abs() < 1e-4, "x = {}, expected 1.0", s[0]);
        assert!((s[1] - 1.0).abs() < 1e-4, "y = {}, expected 1.0", s[1]);
    }
}
