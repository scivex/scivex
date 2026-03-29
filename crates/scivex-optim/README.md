# scivex-optim

Optimization and numerical methods for Scivex. Root finding, minimization,
integration, ODE solvers, linear programming, and curve fitting.

## Highlights

- **Root finding** — Bisection, Newton-Raphson, Brent's method, secant method
- **Minimization** — Gradient descent, BFGS, L-BFGS-B, Nelder-Mead
- **Linear programming** — Revised simplex method for LP problems
- **Curve fitting** — Levenberg-Marquardt non-linear least squares
- **Numerical integration** — Trapezoidal, Simpson's, Gauss-Legendre quadrature
- **ODE solvers** — Euler, RK4, RK45, BDF2 for stiff systems
- **PDE solvers** — Wave equation (1D), Laplace equation (2D)
- **Interpolation** — 1D and 2D interpolation, B-splines
- **Numerical differentiation** — Forward, central, and Richardson extrapolation

## Usage

```rust
use scivex_optim::prelude::*;

// Minimize Rosenbrock function
let f = |x: &Tensor<f64>| { /* Rosenbrock */ };
let grad = |x: &Tensor<f64>| { /* gradient */ };
let x0 = Tensor::from_vec(vec![-1.0, -1.0], &[2]);
let result = bfgs(f, grad, &x0, &MinimizeOptions::default()).unwrap();

// Root finding
let root = brent(|x| x * x - 2.0, 0.0, 2.0, 1e-10).unwrap();

// Numerical integration
let integral = simpson(|x| x.sin(), 0.0, std::f64::consts::PI, 100).unwrap();
```

## License

MIT
