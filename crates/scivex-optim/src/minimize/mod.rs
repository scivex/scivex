//! Multi-dimensional unconstrained optimization.

mod bfgs;
mod gradient_descent;

pub use bfgs::bfgs;
pub use gradient_descent::gradient_descent;

use scivex_core::{Float, Tensor};

/// Result of a multi-dimensional minimization algorithm.
#[derive(Debug, Clone)]
pub struct MinimizeResult<T: Float> {
    /// The estimated minimizer.
    pub x: Tensor<T>,
    /// The function value at the minimizer.
    pub f_val: T,
    /// The gradient at the minimizer (if available).
    pub grad: Option<Tensor<T>>,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Number of function evaluations.
    pub f_evals: usize,
    /// Number of gradient evaluations.
    pub g_evals: usize,
    /// Whether the algorithm converged.
    pub converged: bool,
}

/// Options controlling multi-dimensional minimization.
#[derive(Debug, Clone)]
pub struct MinimizeOptions<T: Float> {
    /// Gradient norm tolerance for convergence.
    pub gtol: T,
    /// Step-size tolerance.
    pub xtol: T,
    /// Function value change tolerance.
    pub ftol: T,
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Learning rate (for gradient descent).
    pub learning_rate: T,
}

impl<T: Float> Default for MinimizeOptions<T> {
    fn default() -> Self {
        Self {
            gtol: T::from_f64(1e-8),
            xtol: T::from_f64(1e-12),
            ftol: T::from_f64(1e-12),
            max_iter: 1000,
            learning_rate: T::from_f64(0.01),
        }
    }
}

/// Compute the gradient of `f` at `x` using central finite differences.
///
/// `grad_i = (f(x + h*e_i) - f(x - h*e_i)) / (2*h)` where `h` is a
/// small perturbation and `e_i` is the i-th unit vector.
pub fn numerical_gradient<T: Float, F: Fn(&Tensor<T>) -> T>(f: &F, x: &Tensor<T>) -> Tensor<T> {
    let n = x.numel();
    let h = T::from_f64(1e-8);
    let two_h = h * T::from_f64(2.0);
    let mut grad_data = vec![T::zero(); n];

    let x_data = x.as_slice();
    let mut x_plus = x_data.to_vec();
    let mut x_minus = x_data.to_vec();

    for i in 0..n {
        let orig = x_data[i];

        x_plus[i] = orig + h;
        x_minus[i] = orig - h;

        let t_plus = Tensor::from_vec(x_plus.clone(), vec![n])
            .expect("perturbed vector length matches original dimension n");
        let t_minus = Tensor::from_vec(x_minus.clone(), vec![n])
            .expect("perturbed vector length matches original dimension n");

        grad_data[i] = (f(&t_plus) - f(&t_minus)) / two_h;

        x_plus[i] = orig;
        x_minus[i] = orig;
    }

    Tensor::from_vec(grad_data, vec![n]).expect("gradient vector length matches dimension n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_numerical_gradient_quadratic() {
        // f(x, y) = x^2 + y^2, grad = (2x, 2y)
        let f = |x: &Tensor<f64>| {
            let s = x.as_slice();
            s[0] * s[0] + s[1] * s[1]
        };

        let x = Tensor::from_vec(vec![3.0, 4.0], vec![2]).unwrap();
        let g = numerical_gradient(&f, &x);
        let gs = g.as_slice();

        assert!((gs[0] - 6.0).abs() < 1e-5, "got {}", gs[0]);
        assert!((gs[1] - 8.0).abs() < 1e-5, "got {}", gs[1]);
    }
}
