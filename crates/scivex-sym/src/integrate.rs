//! Symbolic integration.
//!
//! Supports a useful subset of antiderivatives:
//!
//! - Constants and polynomials (power rule)
//! - Exponential and logarithmic functions
//! - Trigonometric functions (sin, cos, tan)
//! - Simple chain rule patterns (`f(a*x + b)`)
//! - Integration by parts (heuristic, one level)
//! - Definite integrals via the fundamental theorem of calculus

use std::collections::HashMap;

use crate::diff::diff;
use crate::error::{Result, SymError};
use crate::expr::{Expr, MathFn, constant, cos, exp, ln, sin, var};
use crate::simplify::simplify;

/// Compute the indefinite integral (antiderivative) of `expr` with respect to
/// `var_name`.
///
/// The constant of integration is omitted.
///
/// Returns `Err(SymError::UnsupportedOperation)` if the integral cannot be
/// computed symbolically.
pub fn integrate(expr: &Expr, var_name: &str) -> Result<Expr> {
    let expr = simplify(expr);
    let result = integrate_inner(&expr, var_name)?;
    Ok(simplify(&result))
}

/// Compute the definite integral of `expr` from `a` to `b`.
///
/// Uses the fundamental theorem: `F(b) - F(a)` where `F` is the antiderivative.
pub fn definite_integral(expr: &Expr, var_name: &str, a: f64, b: f64) -> Result<f64> {
    let antideriv = integrate(expr, var_name)?;
    let fa = antideriv.substitute(var_name, &constant(a));
    let fb = antideriv.substitute(var_name, &constant(b));
    let fa = simplify(&fa);
    let fb = simplify(&fb);
    let empty = HashMap::new();
    let va = fa
        .eval(&empty)
        .map_err(|_| SymError::UnsupportedOperation {
            reason: "could not evaluate antiderivative at lower bound",
        })?;
    let vb = fb
        .eval(&empty)
        .map_err(|_| SymError::UnsupportedOperation {
            reason: "could not evaluate antiderivative at upper bound",
        })?;
    Ok(vb - va)
}

// ---------------------------------------------------------------------------
// Core integration engine
// ---------------------------------------------------------------------------

fn integrate_inner(expr: &Expr, v: &str) -> Result<Expr> {
    // ── Constant (w.r.t. v) ────────────────────────────────────────────
    if !contains_var(expr, v) {
        // ∫ c dx = c*x
        return Ok(Expr::Mul(Box::new(expr.clone()), Box::new(var(v))));
    }

    match expr {
        // ── Variable x ─────────────────────────────────────────────────
        Expr::Var(name) if name == v => {
            // ∫ x dx = x^2 / 2
            Ok(Expr::Mul(
                Box::new(constant(0.5)),
                Box::new(Expr::Pow(Box::new(var(v)), Box::new(constant(2.0)))),
            ))
        }

        // ── Addition: ∫ (a + b) = ∫a + ∫b ────────────────────────────
        Expr::Add(a, b) => {
            let ia = integrate_inner(a, v)?;
            let ib = integrate_inner(b, v)?;
            Ok(Expr::Add(Box::new(ia), Box::new(ib)))
        }

        // ── Negation: ∫ -f = -(∫f) ──────────────────────────────────
        Expr::Neg(inner) => {
            let ii = integrate_inner(inner, v)?;
            Ok(Expr::Neg(Box::new(ii)))
        }

        // ── Multiplication ───────────────────────────────────────────
        Expr::Mul(a, b) => integrate_mul(a, b, v),

        // ── Power: x^n ──────────────────────────────────────────────
        Expr::Pow(base, exp) => integrate_pow(base, exp, v),

        // ── Functions ────────────────────────────────────────────────
        Expr::Fn(func, arg) => integrate_fn(*func, arg, v),

        _ => Err(SymError::UnsupportedOperation {
            reason: "cannot integrate this expression",
        }),
    }
}

// ---------------------------------------------------------------------------
// Multiplication
// ---------------------------------------------------------------------------

fn integrate_mul(a: &Expr, b: &Expr, v: &str) -> Result<Expr> {
    let a_has = contains_var(a, v);
    let b_has = contains_var(b, v);

    // c * f(x)  →  c * ∫f(x)
    if !a_has {
        let ib = integrate_inner(b, v)?;
        return Ok(Expr::Mul(Box::new(a.clone()), Box::new(ib)));
    }
    // f(x) * c  →  c * ∫f(x)
    if !b_has {
        let ia = integrate_inner(a, v)?;
        return Ok(Expr::Mul(Box::new(b.clone()), Box::new(ia)));
    }

    // Try integration by parts: ∫ u dv = u*v - ∫ v du
    // Heuristic: if one factor is a polynomial and the other is exp/trig,
    // choose the polynomial as u.
    if let Some(result) = try_by_parts(a, b, v) {
        return Ok(result);
    }
    if let Some(result) = try_by_parts(b, a, v) {
        return Ok(result);
    }

    Err(SymError::UnsupportedOperation {
        reason: "cannot integrate this product",
    })
}

// ---------------------------------------------------------------------------
// Power rule
// ---------------------------------------------------------------------------

#[allow(clippy::collapsible_if)]
fn integrate_pow(base: &Expr, exp: &Expr, v: &str) -> Result<Expr> {
    let base_has = contains_var(base, v);
    let exp_has = contains_var(exp, v);

    // x^n (constant exponent, base is the variable) → x^(n+1) / (n+1)
    if base_has && !exp_has {
        if let Expr::Var(name) = base {
            if name == v {
                if let Some(n) = exp.as_const() {
                    if (n - (-1.0)).abs() < f64::EPSILON {
                        return Ok(ln(var(v)));
                    }
                    let n1 = n + 1.0;
                    return Ok(Expr::Mul(
                        Box::new(constant(1.0 / n1)),
                        Box::new(Expr::Pow(Box::new(var(v)), Box::new(constant(n1)))),
                    ));
                }
            }
        }
        // Try linear substitution: (a*x + b)^n
        if let Some((a_coeff, _b_coeff)) = as_linear(base, v) {
            if let Some(n) = exp.as_const() {
                if (n - (-1.0)).abs() < f64::EPSILON {
                    return Ok(Expr::Mul(
                        Box::new(constant(1.0 / a_coeff)),
                        Box::new(ln(base.clone())),
                    ));
                }
                let n1 = n + 1.0;
                return Ok(Expr::Mul(
                    Box::new(constant(1.0 / (a_coeff * n1))),
                    Box::new(Expr::Pow(Box::new(base.clone()), Box::new(constant(n1)))),
                ));
            }
        }
    }

    // c^x  →  c^x / ln(c) (exponential with constant base)
    if !base_has && exp_has {
        if let Expr::Var(name) = exp {
            if name == v {
                if let Some(c) = base.as_const() {
                    if c > 0.0 && (c - 1.0).abs() > f64::EPSILON {
                        return Ok(Expr::Mul(
                            Box::new(constant(1.0 / c.ln())),
                            Box::new(Expr::Pow(Box::new(base.clone()), Box::new(var(v)))),
                        ));
                    }
                }
            }
        }
    }

    Err(SymError::UnsupportedOperation {
        reason: "cannot integrate this power expression",
    })
}

// ---------------------------------------------------------------------------
// Function integration
// ---------------------------------------------------------------------------

#[allow(clippy::collapsible_if)]
fn integrate_fn(func: MathFn, arg: &Expr, v: &str) -> Result<Expr> {
    // Simple case: f(x) where arg = x
    if let Expr::Var(name) = arg {
        if name == v {
            return match func {
                // ∫ sin(x) = -cos(x)
                MathFn::Sin => Ok(Expr::Neg(Box::new(cos(var(v))))),
                // ∫ cos(x) = sin(x)
                MathFn::Cos => Ok(sin(var(v))),
                // ∫ exp(x) = exp(x)
                MathFn::Exp => Ok(exp(var(v))),
                // ∫ ln(x) = x*ln(x) - x
                MathFn::Ln => Ok(Expr::Add(
                    Box::new(Expr::Mul(Box::new(var(v)), Box::new(ln(var(v))))),
                    Box::new(Expr::Neg(Box::new(var(v)))),
                )),
                _ => Err(SymError::UnsupportedOperation {
                    reason: "cannot integrate this function",
                }),
            };
        }
    }

    // Linear substitution: f(a*x + b) → (1/a) * F(a*x + b)
    if let Some((a_coeff, _b_coeff)) = as_linear(arg, v) {
        let inv_a = constant(1.0 / a_coeff);
        let antideriv = match func {
            MathFn::Sin => Ok(Expr::Neg(Box::new(cos(arg.clone())))),
            MathFn::Cos => Ok(sin(arg.clone())),
            MathFn::Exp => Ok(exp(arg.clone())),
            _ => Err(SymError::UnsupportedOperation {
                reason: "cannot integrate this function with linear argument",
            }),
        }?;
        return Ok(Expr::Mul(Box::new(inv_a), Box::new(antideriv)));
    }

    Err(SymError::UnsupportedOperation {
        reason: "cannot integrate this function expression",
    })
}

// ---------------------------------------------------------------------------
// Integration by parts (one level)
// ---------------------------------------------------------------------------

/// Try integration by parts: ∫ u * dv_expr = u * V - ∫ V * du
/// where `u_candidate` is the "u" and `dv_candidate` is "dv/dx".
fn try_by_parts(u_candidate: &Expr, dv_candidate: &Expr, v: &str) -> Option<Expr> {
    // Only attempt if u is "simple" (polynomial-like) and dv is integrable
    if !is_polynomial_like(u_candidate, v) {
        return None;
    }
    let big_v = integrate_inner(dv_candidate, v).ok()?;
    let big_v = simplify(&big_v);
    let du = diff(u_candidate, v);

    // u * V - ∫ V * du
    let v_du = Expr::Mul(Box::new(big_v.clone()), Box::new(du));
    let v_du = simplify(&v_du);
    let int_v_du = integrate_inner(&v_du, v).ok()?;

    Some(Expr::Add(
        Box::new(Expr::Mul(Box::new(u_candidate.clone()), Box::new(big_v))),
        Box::new(Expr::Neg(Box::new(int_v_du))),
    ))
}

/// Check if expr is polynomial-like (var, const, or simple products/sums of those).
#[allow(clippy::collapsible_if)]
fn is_polynomial_like(expr: &Expr, v: &str) -> bool {
    match expr {
        Expr::Const(_) | Expr::Var(_) => true,
        Expr::Add(a, b) | Expr::Mul(a, b) => is_polynomial_like(a, v) && is_polynomial_like(b, v),
        Expr::Neg(inner) => is_polynomial_like(inner, v),
        Expr::Pow(base, exp) => {
            if let Expr::Var(name) = base.as_ref() {
                if name == v {
                    if let Some(n) = exp.as_const() {
                        return n >= 0.0 && (n - n.floor()).abs() < f64::EPSILON;
                    }
                }
            }
            false
        }
        Expr::Fn(_, _) => false,
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Check if `expr` contains `var`.
fn contains_var(expr: &Expr, v: &str) -> bool {
    match expr {
        Expr::Const(_) => false,
        Expr::Var(name) => name == v,
        Expr::Add(a, b) | Expr::Mul(a, b) | Expr::Pow(a, b) => {
            contains_var(a, v) || contains_var(b, v)
        }
        Expr::Neg(inner) | Expr::Fn(_, inner) => contains_var(inner, v),
    }
}

/// If `expr` is of the form `a*x + b` (linear in `v`), return `(a, b)`.
fn as_linear(expr: &Expr, v: &str) -> Option<(f64, f64)> {
    let d = diff(expr, v);
    let d = simplify(&d);
    // Derivative must be a constant for linearity
    let a = d.as_const()?;
    // b = expr at v=0
    let at_zero = simplify(&expr.substitute(v, &constant(0.0)));
    let b = at_zero.eval(&HashMap::new()).ok()?;
    // Verify it's actually linear (second derivative is zero)
    let d2 = diff(&d, v);
    let d2 = simplify(&d2);
    if !d2.is_zero() {
        return None;
    }
    Some((a, b))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn eval_at(e: &Expr, x_val: f64) -> f64 {
        let mut vars = HashMap::new();
        vars.insert("x".into(), x_val);
        e.eval(&vars).unwrap()
    }

    #[test]
    fn integrate_constant() {
        // ∫ 5 dx = 5x
        let e = constant(5.0);
        let result = integrate(&e, "x").unwrap();
        assert!((eval_at(&result, 3.0) - 15.0).abs() < 1e-10);
    }

    #[test]
    fn integrate_x() {
        // ∫ x dx = x^2/2
        let result = integrate(&var("x"), "x").unwrap();
        assert!((eval_at(&result, 4.0) - 8.0).abs() < 1e-10);
    }

    #[test]
    fn integrate_x_squared() {
        // ∫ x^2 dx = x^3/3
        let e = Expr::Pow(Box::new(var("x")), Box::new(constant(2.0)));
        let result = integrate(&e, "x").unwrap();
        assert!((eval_at(&result, 3.0) - 9.0).abs() < 1e-10);
    }

    #[test]
    fn integrate_x_inverse() {
        // ∫ 1/x dx = ln(x)
        let e = Expr::Pow(Box::new(var("x")), Box::new(constant(-1.0)));
        let result = integrate(&e, "x").unwrap();
        // ln(e) = 1
        assert!((eval_at(&result, std::f64::consts::E) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn integrate_sin() {
        // ∫ sin(x) dx = -cos(x)
        let e = sin(var("x"));
        let result = integrate(&e, "x").unwrap();
        // -cos(0) = -1
        assert!((eval_at(&result, 0.0) - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn integrate_cos() {
        // ∫ cos(x) dx = sin(x)
        let e = cos(var("x"));
        let result = integrate(&e, "x").unwrap();
        // sin(pi/2) = 1
        assert!((eval_at(&result, std::f64::consts::FRAC_PI_2) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn integrate_exp() {
        // ∫ exp(x) dx = exp(x)
        let e = exp(var("x"));
        let result = integrate(&e, "x").unwrap();
        assert!((eval_at(&result, 1.0) - std::f64::consts::E).abs() < 1e-10);
    }

    #[test]
    fn integrate_ln() {
        // ∫ ln(x) dx = x*ln(x) - x
        let e = ln(var("x"));
        let result = integrate(&e, "x").unwrap();
        // At x=e: e*1 - e = 0
        assert!((eval_at(&result, std::f64::consts::E)).abs() < 1e-10);
    }

    #[test]
    fn integrate_constant_times_x() {
        // ∫ 3x dx = 3x^2/2
        let e = constant(3.0) * var("x");
        let result = integrate(&e, "x").unwrap();
        assert!((eval_at(&result, 2.0) - 6.0).abs() < 1e-10);
    }

    #[test]
    fn integrate_sum() {
        // ∫ (x + 1) dx = x^2/2 + x
        let e = var("x") + constant(1.0);
        let result = integrate(&e, "x").unwrap();
        // At x=2: 2 + 2 = 4
        assert!((eval_at(&result, 2.0) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn integrate_polynomial() {
        // ∫ (x^2 + 2x + 1) dx = x^3/3 + x^2 + x
        let e = Expr::Pow(Box::new(var("x")), Box::new(constant(2.0)))
            + constant(2.0) * var("x")
            + constant(1.0);
        let result = integrate(&e, "x").unwrap();
        // At x=3: 9 + 9 + 3 = 21
        assert!((eval_at(&result, 3.0) - 21.0).abs() < 1e-10);
    }

    #[test]
    fn integrate_linear_sub_sin() {
        // ∫ sin(2x) dx = -cos(2x)/2
        let e = sin(constant(2.0) * var("x"));
        let result = integrate(&e, "x").unwrap();
        // Verify numerically: d/dx[-cos(2x)/2] = sin(2x)
        let d = diff(&result, "x");
        let d = simplify(&d);
        // At x=1: sin(2) ≈ 0.9093
        assert!((eval_at(&d, 1.0) - (2.0_f64).sin()).abs() < 1e-10);
    }

    #[test]
    fn integrate_linear_sub_exp() {
        // ∫ exp(3x) dx = exp(3x)/3
        let e = exp(constant(3.0) * var("x"));
        let result = integrate(&e, "x").unwrap();
        let d = diff(&result, "x");
        let d = simplify(&d);
        assert!((eval_at(&d, 1.0) - (3.0_f64).exp()).abs() < 1e-8);
    }

    #[test]
    fn definite_integral_x_squared() {
        // ∫₀² x^2 dx = 8/3
        let e = Expr::Pow(Box::new(var("x")), Box::new(constant(2.0)));
        let result = definite_integral(&e, "x", 0.0, 2.0).unwrap();
        assert!((result - 8.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn definite_integral_sin() {
        // ∫₀^π sin(x) dx = 2
        let e = sin(var("x"));
        let result = definite_integral(&e, "x", 0.0, std::f64::consts::PI).unwrap();
        assert!((result - 2.0).abs() < 1e-10);
    }

    #[test]
    fn integrate_by_parts_x_exp() {
        // ∫ x * exp(x) dx = x*exp(x) - exp(x)
        let e = var("x") * exp(var("x"));
        let result = integrate(&e, "x").unwrap();
        // Verify via differentiation
        let d = diff(&result, "x");
        let d = simplify(&d);
        // At x=1: 1*e = e ≈ 2.718
        assert!((eval_at(&d, 1.0) - std::f64::consts::E).abs() < 1e-8);
    }

    #[test]
    fn integrate_unsupported_returns_error() {
        // tan(x^2) — no closed form
        let e = Expr::Fn(
            MathFn::Tan,
            Box::new(Expr::Pow(Box::new(var("x")), Box::new(constant(2.0)))),
        );
        assert!(integrate(&e, "x").is_err());
    }

    #[test]
    fn integrate_wrt_other_var() {
        // ∫ x dy = x*y (x is constant w.r.t. y)
        let result = integrate(&var("x"), "y").unwrap();
        let mut vars = HashMap::new();
        vars.insert("x".into(), 3.0);
        vars.insert("y".into(), 4.0);
        assert!((result.eval(&vars).unwrap() - 12.0).abs() < 1e-10);
    }
}
