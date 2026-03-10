//! Gradient descent with fixed learning rate.

use scivex_core::{Float, Tensor};

use crate::error::Result;

use super::{MinimizeOptions, MinimizeResult};

/// Minimize `f` using gradient descent with a fixed learning rate.
///
/// The gradient is provided by the `grad` closure. For numerical gradients,
/// use [`numerical_gradient`](super::numerical_gradient) to construct one.
pub fn gradient_descent<T, F, G>(
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
    let mut x = x0.clone();
    let mut fx = f(&x);
    let mut f_evals = 1usize;
    let mut g_evals = 0usize;

    for i in 0..options.max_iter {
        let g = grad(&x);
        g_evals += 1;

        // Check gradient norm for convergence
        let grad_norm_sq: T = g.as_slice().iter().map(|&v| v * v).sum();
        let grad_norm = grad_norm_sq.sqrt();

        if grad_norm < options.gtol {
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

        // x_new = x - lr * grad
        let step = &g * options.learning_rate;
        let x_new = &x - &step;
        let fx_new = f(&x_new);
        f_evals += 1;

        // Check function value change
        let f_change = (fx - fx_new).abs();
        x = x_new;
        fx = fx_new;

        if f_change < options.ftol {
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

    let g = grad(&x);
    g_evals += 1;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_descent_quadratic() {
        // f(x, y) = x^2 + y^2, grad = (2x, 2y)
        let f = |x: &Tensor<f64>| {
            let s = x.as_slice();
            s[0] * s[0] + s[1] * s[1]
        };
        let grad = |x: &Tensor<f64>| {
            let s = x.as_slice();
            Tensor::from_vec(vec![2.0 * s[0], 2.0 * s[1]], vec![2]).unwrap()
        };

        let x0 = Tensor::from_vec(vec![5.0, 5.0], vec![2]).unwrap();
        let opts = MinimizeOptions {
            learning_rate: 0.1,
            max_iter: 500,
            gtol: 1e-6,
            ..MinimizeOptions::default()
        };

        let result = gradient_descent(f, grad, &x0, &opts).unwrap();
        assert!(
            result.converged,
            "did not converge after {} iters",
            result.iterations
        );
        let s = result.x.as_slice();
        assert!(s[0].abs() < 1e-4, "x = {}", s[0]);
        assert!(s[1].abs() < 1e-4, "y = {}", s[1]);
    }
}
