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
//! - [`ode`] — ODE initial value problem solvers (Euler, RK45, BDF-2)

/// Optimization error types.
pub mod error;
/// Numerical integration (trapezoid, Simpson, Gauss-Kronrod).
pub mod integrate;
/// Multi-dimensional unconstrained optimization (gradient descent, BFGS).
pub mod minimize;
/// 1-D minimization (golden section, Brent).
pub mod minimize_1d;
/// ODE initial value problem solvers (Euler, RK45, BDF-2).
pub mod ode;
/// Scalar root-finding (bisection, Newton, Brent).
pub mod roots;

pub use error::{OptimError, Result};
pub use integrate::{QuadOptions, QuadResult, quad, simpson, trapezoid};
pub use minimize::{MinimizeOptions, MinimizeResult, bfgs, gradient_descent, numerical_gradient};
pub use minimize_1d::{Minimize1dResult, brent_min, golden_section};
pub use ode::{OdeMethod, OdeOptions, OdeResult, bdf2, euler, rk45, solve_ivp};
pub use roots::{RootOptions, RootResult, bisection, brent_root, newton};

/// Items intended for glob-import: `use scivex_optim::prelude::*;`
pub mod prelude {
    pub use crate::error::{OptimError, Result};
    pub use crate::integrate::{QuadOptions, QuadResult, quad, simpson, trapezoid};
    pub use crate::minimize::{
        MinimizeOptions, MinimizeResult, bfgs, gradient_descent, numerical_gradient,
    };
    pub use crate::minimize_1d::{Minimize1dResult, brent_min, golden_section};
    pub use crate::ode::{OdeMethod, OdeOptions, OdeResult, bdf2, euler, rk45, solve_ivp};
    pub use crate::roots::{RootOptions, RootResult, bisection, brent_root, newton};
}
