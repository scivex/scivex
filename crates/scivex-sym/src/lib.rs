//! `scivex-sym` — Symbolic math, CAS, and expression simplification.
//!
//! Provides from-scratch implementations of:
//! - Symbolic expression AST with evaluation, substitution, and operator overloading
//! - Algebraic simplification with constant folding and identity reduction
//! - Symbolic differentiation with chain rule support
//! - Algebraic expansion and factoring
//! - Linear and quadratic equation solving
//! - Coefficient-based polynomials with Horner evaluation and root finding
//! - Symbolic integration (power rule, trig, exp, integration by parts)
//! - Taylor / Maclaurin series expansion

/// Algebraic expansion and factoring.
pub mod algebra;
/// Symbolic differentiation.
pub mod diff;
/// Symbolic math error types.
pub mod error;
/// Expression AST, evaluation, substitution, and operator overloading.
pub mod expr;
/// Symbolic integration (indefinite and definite).
pub mod integrate;

/// Coefficient-based polynomials with Horner evaluation and root finding.
pub mod polynomial;
/// Algebraic simplification and constant folding.
pub mod simplify;
/// Linear and quadratic equation solving.
pub mod solve;
/// Taylor and Maclaurin series expansion.
pub mod taylor;

pub use algebra::{expand, factor_out};
pub use diff::{diff, diff_n};
pub use error::{Result, SymError};
pub use expr::{Expr, MathFn, abs, constant, cos, e, exp, ln, one, pi, sin, sqrt, tan, var, zero};
pub use integrate::{definite_integral, integrate};
pub use polynomial::Polynomial;
pub use simplify::simplify;
pub use solve::{solve_linear, solve_quadratic};
pub use taylor::{maclaurin, taylor};

/// Items intended for glob-import: `use scivex_sym::prelude::*;`
pub mod prelude {
    pub use crate::algebra::{expand, factor_out};
    pub use crate::diff::{diff, diff_n};
    pub use crate::error::{Result, SymError};
    pub use crate::expr::{
        Expr, MathFn, abs, constant, cos, e, exp, ln, one, pi, sin, sqrt, tan, var, zero,
    };
    pub use crate::integrate::{definite_integral, integrate};
    pub use crate::polynomial::Polynomial;
    pub use crate::simplify::simplify;
    pub use crate::solve::{solve_linear, solve_quadratic};
    pub use crate::taylor::{maclaurin, taylor};
}
