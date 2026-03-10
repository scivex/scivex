# scivex-optim

Optimization, root finding, and numerical integration for Scivex.

## Highlights

- **Root finding** — Bisection, Brent's method, Newton's method
- **1D minimization** — Brent's method, golden section search
- **Multi-D optimization** — Gradient descent, BFGS quasi-Newton
- **Numerical integration** — Trapezoid, Simpson, adaptive Gauss-Kronrod quadrature
- **Numerical gradient** — Central finite differences

## Usage

```rust
use scivex_optim::prelude::*;

// Find root of f(x) = x^2 - 2
let root = brent_root(|x| x * x - 2.0, 0.0, 2.0, &RootOptions::default()).unwrap();

// Minimize with BFGS
let f = |x: &[f64]| (x[0] - 1.0).powi(2) + (x[1] - 2.5).powi(2);
let grad = |x: &[f64]| vec![2.0 * (x[0] - 1.0), 2.0 * (x[1] - 2.5)];
let result = bfgs(f, grad, &[0.0, 0.0], &MinimizeOptions::default()).unwrap();

// Numerical integration
let area = quad(|x| x.sin(), 0.0, std::f64::consts::PI, &QuadOptions::default()).unwrap();
```

## License

MIT
