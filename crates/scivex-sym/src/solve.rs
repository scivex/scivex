use crate::diff::diff;
use crate::error::{Result, SymError};
use crate::expr::{Expr, constant};
use crate::simplify::simplify;

/// Solve a linear equation `expr = 0` for `var`.
///
/// Extracts `a` and `b` where `expr = a*var + b` via differentiation,
/// then returns `-b / a`.
///
/// # Examples
///
/// ```
/// # use std::collections::HashMap;
/// # use scivex_sym::expr::{var, constant};
/// # use scivex_sym::solve::solve_linear;
/// // Solve 2x - 6 = 0  →  x = 3
/// let expr = constant(2.0) * var("x") - constant(6.0);
/// let sol = solve_linear(&expr, "x").unwrap();
/// let val = sol.eval(&HashMap::new()).unwrap();
/// assert!((val - 3.0).abs() < 1e-10);
/// ```
pub fn solve_linear(expr: &Expr, var: &str) -> Result<Expr> {
    let expr = simplify(expr);

    // a = d(expr)/d(var) — should be constant for a linear equation.
    let a = diff(&expr, var);
    if a.free_variables().contains(var) {
        return Err(SymError::SolveFailure {
            reason: "expression is not linear in the variable",
        });
    }

    // b = expr evaluated at var=0
    let b = simplify(&expr.substitute(var, &constant(0.0)));

    // result = -b / a
    let neg_b = Expr::Neg(Box::new(b));
    let result = Expr::Mul(
        Box::new(neg_b),
        Box::new(Expr::Pow(Box::new(a), Box::new(constant(-1.0)))),
    );
    Ok(simplify(&result))
}

/// Solve a quadratic equation `expr = 0` for `var`.
///
/// Returns up to two roots using the quadratic formula.
/// Roots are returned as constant `Expr` values.
///
/// # Examples
///
/// ```
/// # use scivex_sym::expr::{var, constant};
/// # use scivex_sym::solve::solve_quadratic;
/// # use std::collections::HashMap;
/// // x^2 - 5x + 6 = 0 → roots at 2 and 3
/// let x = var("x");
/// let expr = x.clone() * x - constant(5.0) * var("x") + constant(6.0);
/// let roots = solve_quadratic(&expr, "x").unwrap();
/// assert_eq!(roots.len(), 2);
/// ```
pub fn solve_quadratic(expr: &Expr, var: &str) -> Result<Vec<Expr>> {
    let expr = simplify(expr);

    // Extract coefficients: expr = a*var^2 + b*var + c
    // a = (1/2) * d²(expr)/d(var)²
    let d1 = diff(&expr, var);
    let d2 = diff(&d1, var);

    // d2 should be constant (2*a).
    if d2.free_variables().contains(var) {
        return Err(SymError::SolveFailure {
            reason: "expression is not quadratic in the variable",
        });
    }

    // c = expr at var=0
    let c = simplify(&expr.substitute(var, &constant(0.0)));
    // b = d1 at var=0
    let b = simplify(&d1.substitute(var, &constant(0.0)));

    // Evaluate to f64 for the quadratic formula.
    let empty = std::collections::HashMap::new();
    let a_val = d2.eval(&empty).map_err(|_| SymError::SolveFailure {
        reason: "could not evaluate quadratic coefficient a",
    })? / 2.0;
    let b_val = b.eval(&empty).map_err(|_| SymError::SolveFailure {
        reason: "could not evaluate quadratic coefficient b",
    })?;
    let c_val = c.eval(&empty).map_err(|_| SymError::SolveFailure {
        reason: "could not evaluate quadratic coefficient c",
    })?;

    if a_val.abs() < f64::EPSILON {
        return Err(SymError::SolveFailure {
            reason: "leading coefficient is zero; not a quadratic",
        });
    }

    let discriminant = b_val * b_val - 4.0 * a_val * c_val;

    if discriminant < -f64::EPSILON {
        return Ok(Vec::new()); // No real roots.
    }

    let sqrt_d = discriminant.abs().sqrt();

    if discriminant.abs() < f64::EPSILON {
        // One repeated root.
        let root = -b_val / (2.0 * a_val);
        Ok(vec![constant(root)])
    } else {
        let r1 = (-b_val + sqrt_d) / (2.0 * a_val);
        let r2 = (-b_val - sqrt_d) / (2.0 * a_val);
        let mut roots = vec![r1, r2];
        roots.sort_by(f64::total_cmp);
        Ok(roots.into_iter().map(constant).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{constant, var};
    use std::collections::HashMap;

    #[test]
    fn solve_linear_simple() {
        // 2x - 6 = 0 → x = 3
        let e = constant(2.0) * var("x") + constant(-6.0);
        let sol = solve_linear(&e, "x").unwrap();
        let val = sol.eval(&HashMap::new()).unwrap();
        assert!((val - 3.0).abs() < 1e-10);
    }

    #[test]
    fn solve_linear_with_offset() {
        // x + 5 = 0 → x = -5
        let e = var("x") + constant(5.0);
        let sol = solve_linear(&e, "x").unwrap();
        let val = sol.eval(&HashMap::new()).unwrap();
        assert!((val - (-5.0)).abs() < 1e-10);
    }

    #[test]
    fn solve_linear_non_linear_fails() {
        // x^2 is not linear.
        let e = Expr::Pow(Box::new(var("x")), Box::new(constant(2.0)));
        let err = solve_linear(&e, "x").unwrap_err();
        assert!(matches!(err, SymError::SolveFailure { .. }));
    }

    #[test]
    fn solve_quadratic_two_roots() {
        // x^2 - 5x + 6 = 0 → x = 2 or x = 3
        let x = var("x");
        let e = Expr::Pow(Box::new(x.clone()), Box::new(constant(2.0)))
            + constant(-5.0) * x
            + constant(6.0);
        let roots = solve_quadratic(&e, "x").unwrap();
        assert_eq!(roots.len(), 2);
        let empty = HashMap::new();
        let r0 = roots[0].eval(&empty).unwrap();
        let r1 = roots[1].eval(&empty).unwrap();
        assert!((r0 - 2.0).abs() < 1e-10);
        assert!((r1 - 3.0).abs() < 1e-10);
    }

    #[test]
    fn solve_quadratic_one_root() {
        // x^2 - 4x + 4 = 0 → x = 2 (repeated)
        let x = var("x");
        let e = Expr::Pow(Box::new(x.clone()), Box::new(constant(2.0)))
            + constant(-4.0) * x
            + constant(4.0);
        let roots = solve_quadratic(&e, "x").unwrap();
        assert_eq!(roots.len(), 1);
        let val = roots[0].eval(&HashMap::new()).unwrap();
        assert!((val - 2.0).abs() < 1e-10);
    }

    #[test]
    fn solve_quadratic_no_real_roots() {
        // x^2 + 1 = 0 → no real roots
        let e = Expr::Pow(Box::new(var("x")), Box::new(constant(2.0))) + constant(1.0);
        let roots = solve_quadratic(&e, "x").unwrap();
        assert!(roots.is_empty());
    }
}
