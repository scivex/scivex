//! `scivex-optim` — Optimization, root finding, numerical integration, and ODE solvers.
//!
//! Built on top of [`scivex_core`] with zero external dependencies for the
//! math itself. All functions are generic over [`Float`](scivex_core::Float).
//!
//! # Modules
//!
//! - [`roots`] — Scalar root-finding (bisection, Newton, Brent)
//! - [`minimize_1d`] — 1-D minimization (golden section, Brent)
//! - [`minimize`] — Multi-dimensional unconstrained optimization (gradient descent, BFGS)
//! - [`integrate`] — Numerical integration (trapezoid, Simpson, adaptive Gauss-Kronrod)
//! - [`interpolate`] — Interpolation (linear, cubic spline, B-spline, bilinear, bicubic)
//! - [`ode`] — ODE initial value problem solvers (Euler, RK45, BDF-2)

/// Curve fitting via non-linear least squares (Levenberg-Marquardt).
pub mod curve_fit;
/// Optimization error types.
pub mod error;
/// Numerical integration (trapezoid, Simpson, Gauss-Kronrod).
pub mod integrate;
/// Interpolation (linear, cubic spline, B-spline, bilinear, bicubic).
pub mod interpolate;
/// Linear programming (revised simplex method).
pub mod linprog;
/// Multi-dimensional optimization (gradient descent, BFGS, Nelder-Mead, L-BFGS-B).
pub mod minimize;
/// 1-D minimization (golden section, Brent).
pub mod minimize_1d;
/// ODE initial value problem solvers (Euler, RK45, BDF-2).
pub mod ode;
/// Scalar root-finding (bisection, Newton, Brent).
pub mod roots;

pub use curve_fit::{LeastSquaresResult, curve_fit, levenberg_marquardt};
pub use error::{OptimError, Result};
pub use integrate::{QuadOptions, QuadResult, quad, simpson, trapezoid};
pub use interpolate::{
    BSpline, Bicubic2d, Bilinear2d, CubicSpline, Extrapolate, Interp1dMethod, Interp2dMethod,
    Linear1d, SplineBoundary, interp1d, interp2d,
};
pub use linprog::{LinProgResult, linprog};
pub use minimize::{
    Bounds, MinimizeOptions, MinimizeResult, bfgs, gradient_descent, lbfgsb, nelder_mead,
    numerical_gradient,
};
pub use minimize_1d::{Minimize1dResult, brent_min, golden_section};
pub use ode::{OdeMethod, OdeOptions, OdeResult, bdf2, euler, rk45, solve_ivp};
pub use roots::{RootOptions, RootResult, bisection, brent_root, newton};

/// Items intended for glob-import: `use scivex_optim::prelude::*;`
pub mod prelude {
    pub use crate::curve_fit::{LeastSquaresResult, curve_fit, levenberg_marquardt};
    pub use crate::error::{OptimError, Result};
    pub use crate::integrate::{QuadOptions, QuadResult, quad, simpson, trapezoid};
    pub use crate::interpolate::{
        BSpline, Bicubic2d, Bilinear2d, CubicSpline, Extrapolate, Interp1dMethod, Interp2dMethod,
        Linear1d, SplineBoundary, interp1d, interp2d,
    };
    pub use crate::linprog::{LinProgResult, linprog};
    pub use crate::minimize::{
        Bounds, MinimizeOptions, MinimizeResult, bfgs, gradient_descent, lbfgsb, nelder_mead,
        numerical_gradient,
    };
    pub use crate::minimize_1d::{Minimize1dResult, brent_min, golden_section};
    pub use crate::ode::{OdeMethod, OdeOptions, OdeResult, bdf2, euler, rk45, solve_ivp};
    pub use crate::roots::{RootOptions, RootResult, bisection, brent_root, newton};
}
