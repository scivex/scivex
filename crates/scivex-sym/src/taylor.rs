//! Taylor series expansion.
//!
//! Computes the Taylor (or Maclaurin) polynomial of a symbolic expression
//! around a given point.

use crate::diff::diff;
use crate::error::{Result, SymError};
use crate::expr::{Expr, constant, var};
use crate::simplify::simplify;

/// Compute the Taylor series expansion of `expr` around `center` up to
/// order `n`.
///
/// Returns the polynomial:
///
/// ```text
/// f(a) + f'(a)*(x-a) + f''(a)/2!*(x-a)^2 + ... + f^(n)(a)/n!*(x-a)^n
/// ```
///
/// The result is a simplified symbolic expression.
///
/// # Examples
///
/// ```
/// # use std::collections::HashMap;
/// # use scivex_sym::expr::{var, exp};
/// # use scivex_sym::taylor::taylor;
/// // Taylor of e^x around 0 to order 3: 1 + x + x²/2 + x³/6
/// let ex = exp(var("x"));
/// let approx = taylor(&ex, "x", 0.0, 3).unwrap();
/// let vars = HashMap::from([("x".to_string(), 0.5)]);
/// let val = approx.eval(&vars).unwrap();
/// assert!((val - 0.5_f64.exp()).abs() < 0.01);
/// ```
pub fn taylor(expr: &Expr, var_name: &str, center: f64, n: usize) -> Result<Expr> {
    if n > 20 {
        return Err(SymError::InvalidExpr {
            reason: "Taylor expansion order must be <= 20",
        });
    }

    let a = constant(center);
    let x_minus_a = if center == 0.0 {
        var(var_name)
    } else {
        Expr::Add(
            Box::new(var(var_name)),
            Box::new(Expr::Neg(Box::new(a.clone()))),
        )
    };

    let empty = std::collections::HashMap::new();
    let mut current = expr.clone();
    let mut factorial = 1.0_f64;
    let mut terms: Vec<Expr> = Vec::with_capacity(n + 1);

    for k in 0..=n {
        if k > 0 {
            factorial *= k as f64;
        }

        // Evaluate the k-th derivative at center
        let at_center = simplify(&current.substitute(var_name, &a));
        let coeff_val = at_center
            .eval(&empty)
            .map_err(|_| SymError::UnsupportedOperation {
                reason: "could not evaluate derivative at expansion point",
            })?;

        if coeff_val.abs() > f64::EPSILON {
            let coeff = constant(coeff_val / factorial);
            let term = if k == 0 {
                coeff
            } else if k == 1 {
                Expr::Mul(Box::new(coeff), Box::new(x_minus_a.clone()))
            } else {
                Expr::Mul(
                    Box::new(coeff),
                    Box::new(Expr::Pow(
                        Box::new(x_minus_a.clone()),
                        Box::new(constant(k as f64)),
                    )),
                )
            };
            terms.push(term);
        }

        // Compute next derivative (unless at last iteration)
        if k < n {
            current = diff(&current, var_name);
        }
    }

    if terms.is_empty() {
        return Ok(constant(0.0));
    }

    let result = terms
        .into_iter()
        .reduce(|acc, t| Expr::Add(Box::new(acc), Box::new(t)))
        .unwrap_or_else(|| constant(0.0));

    Ok(simplify(&result))
}

/// Compute the Maclaurin series (Taylor series around 0) of `expr` up to
/// order `n`.
///
/// # Examples
///
/// ```
/// # use scivex_sym::expr::{var, exp};
/// # use scivex_sym::taylor::maclaurin;
/// # use std::collections::HashMap;
/// let approx = maclaurin(&exp(var("x")), "x", 4).unwrap();
/// let vars = HashMap::from([("x".to_string(), 0.5)]);
/// let val = approx.eval(&vars).unwrap();
/// assert!((val - 0.5_f64.exp()).abs() < 0.001);
/// ```
pub fn maclaurin(expr: &Expr, var_name: &str, n: usize) -> Result<Expr> {
    taylor(expr, var_name, 0.0, n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{cos, exp, sin, var};
    use std::collections::HashMap;

    fn eval_at(e: &Expr, x_val: f64) -> f64 {
        let mut vars = HashMap::new();
        vars.insert("x".into(), x_val);
        e.eval(&vars).unwrap()
    }

    #[test]
    fn maclaurin_exp_order_4() {
        // exp(x) ≈ 1 + x + x^2/2 + x^3/6 + x^4/24
        let e = exp(var("x"));
        let t = maclaurin(&e, "x", 4).unwrap();
        // At x=0.5: exp(0.5) ≈ 1.6487, Taylor ≈ 1.6484
        let approx = eval_at(&t, 0.5);
        let exact = 0.5_f64.exp();
        assert!(
            (approx - exact).abs() < 0.01,
            "approx={approx}, exact={exact}"
        );
    }

    #[test]
    fn maclaurin_sin_order_5() {
        // sin(x) ≈ x - x^3/6 + x^5/120
        let e = sin(var("x"));
        let t = maclaurin(&e, "x", 5).unwrap();
        let approx = eval_at(&t, 0.3);
        let exact = 0.3_f64.sin();
        assert!(
            (approx - exact).abs() < 1e-6,
            "approx={approx}, exact={exact}"
        );
    }

    #[test]
    fn maclaurin_cos_order_4() {
        // cos(x) ≈ 1 - x^2/2 + x^4/24
        let e = cos(var("x"));
        let t = maclaurin(&e, "x", 4).unwrap();
        let approx = eval_at(&t, 0.5);
        let exact = 0.5_f64.cos();
        assert!(
            (approx - exact).abs() < 1e-4,
            "approx={approx}, exact={exact}"
        );
    }

    #[test]
    fn taylor_around_nonzero() {
        // Taylor of x^2 around x=1: 1 + 2(x-1) + (x-1)^2
        let e = Expr::Pow(Box::new(var("x")), Box::new(constant(2.0)));
        let t = taylor(&e, "x", 1.0, 2).unwrap();
        // At x=2: should be 4
        assert!((eval_at(&t, 2.0) - 4.0).abs() < 1e-10);
        // At x=3: should be 9
        assert!((eval_at(&t, 3.0) - 9.0).abs() < 1e-10);
    }

    #[test]
    fn taylor_constant() {
        let e = constant(7.0);
        let t = taylor(&e, "x", 0.0, 3).unwrap();
        assert!((eval_at(&t, 100.0) - 7.0).abs() < 1e-10);
    }

    #[test]
    fn taylor_high_order_rejected() {
        let e = var("x");
        assert!(taylor(&e, "x", 0.0, 25).is_err());
    }

    #[test]
    fn maclaurin_polynomial_exact() {
        // Taylor of x^3 around 0, order 3 should be exact
        let e = Expr::Pow(Box::new(var("x")), Box::new(constant(3.0)));
        let t = maclaurin(&e, "x", 3).unwrap();
        assert!((eval_at(&t, 2.0) - 8.0).abs() < 1e-10);
        assert!((eval_at(&t, -1.0) - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn maclaurin_1_over_1_minus_x() {
        // 1/(1-x) = (1-x)^(-1) ≈ 1 + x + x^2 + x^3 + ...
        let e = Expr::Pow(
            Box::new(constant(1.0) + Expr::Neg(Box::new(var("x")))),
            Box::new(constant(-1.0)),
        );
        let t = maclaurin(&e, "x", 4).unwrap();
        // At x=0.5: exact = 2.0, Taylor = 1 + 0.5 + 0.25 + 0.125 + 0.0625 = 1.9375
        let approx = eval_at(&t, 0.5);
        assert!((approx - 2.0).abs() < 0.1, "approx={approx}");
    }
}
