//! Partial differential equation solvers using finite difference methods.

pub mod finite_diff;
pub use finite_diff::{
    BoundaryCondition, PdeResult, heat_equation_1d, laplace_2d, wave_equation_1d,
};
