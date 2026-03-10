use crate::expr::Expr;
use crate::simplify::simplify;

/// Expand products over sums: `a * (b + c)` → `a*b + a*c`.
///
/// Recursively expands all sub-expressions and simplifies the result.
#[must_use]
pub fn expand(expr: &Expr) -> Expr {
    let expanded = expand_inner(expr);
    simplify(&expanded)
}

fn expand_inner(expr: &Expr) -> Expr {
    match expr {
        Expr::Const(_) | Expr::Var(_) => expr.clone(),
        Expr::Add(a, b) => {
            let a = expand_inner(a);
            let b = expand_inner(b);
            Expr::Add(Box::new(a), Box::new(b))
        }
        Expr::Mul(a, b) => {
            let a = expand_inner(a);
            let b = expand_inner(b);
            distribute(a, b)
        }
        Expr::Pow(base, exp) => {
            let base = expand_inner(base);
            let exp = expand_inner(exp);
            // Expand integer powers of sums: (a+b)^2 → (a+b)*(a+b) expanded.
            if let Some(n) = exp.as_const()
                && n > 0.0
                && (n - n.floor()).abs() < f64::EPSILON
                && n <= 8.0
            {
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let ni = n as u32;
                if ni >= 2 {
                    let mut result = base.clone();
                    for _ in 1..ni {
                        result = distribute(result, base.clone());
                    }
                    return result;
                }
            }
            Expr::Pow(Box::new(base), Box::new(exp))
        }
        Expr::Neg(inner) => {
            let inner = expand_inner(inner);
            Expr::Neg(Box::new(inner))
        }
        Expr::Fn(func, arg) => {
            let arg = expand_inner(arg);
            Expr::Fn(*func, Box::new(arg))
        }
    }
}

/// Distribute multiplication: if either side is a sum, expand it.
fn distribute(a: Expr, b: Expr) -> Expr {
    match (a, b) {
        // (a1 + a2) * b → a1*b + a2*b
        (Expr::Add(a1, a2), b) => {
            let left = distribute(*a1, b.clone());
            let right = distribute(*a2, b);
            Expr::Add(Box::new(left), Box::new(right))
        }
        // a * (b1 + b2) → a*b1 + a*b2
        (a, Expr::Add(b1, b2)) => {
            let left = distribute(a.clone(), *b1);
            let right = distribute(a, *b2);
            Expr::Add(Box::new(left), Box::new(right))
        }
        (a, b) => Expr::Mul(Box::new(a), Box::new(b)),
    }
}

/// Factor out a common `term` from a sum expression.
///
/// Given `term*a + term*b`, returns `term * (a + b)`.
/// If the term cannot be factored from every addend, the expression is
/// returned unchanged.
#[must_use]
pub fn factor_out(expr: &Expr, term: &Expr) -> Expr {
    let addends = collect_addends(expr);
    let mut remainders = Vec::with_capacity(addends.len());

    for addend in &addends {
        if let Some(remainder) = try_divide(addend, term) {
            remainders.push(remainder);
        } else {
            // Cannot factor this term from every addend; return as-is.
            return expr.clone();
        }
    }

    let sum = remainders
        .into_iter()
        .reduce(|acc, r| Expr::Add(Box::new(acc), Box::new(r)))
        .unwrap_or(Expr::Const(0.0));

    simplify(&Expr::Mul(Box::new(term.clone()), Box::new(sum)))
}

/// Collect all top-level addends from a sum tree.
fn collect_addends(expr: &Expr) -> Vec<Expr> {
    match expr {
        Expr::Add(a, b) => {
            let mut v = collect_addends(a);
            v.extend(collect_addends(b));
            v
        }
        _ => vec![expr.clone()],
    }
}

/// Try to symbolically divide `expr` by `term`.
///
/// Very simple: handles the case where `expr = Mul(term, something)` or
/// `expr = term` (quotient is 1).
fn try_divide(expr: &Expr, term: &Expr) -> Option<Expr> {
    if expr == term {
        return Some(Expr::Const(1.0));
    }
    // Check if expr = term * rest or rest * term.
    if let Expr::Mul(a, b) = expr {
        if a.as_ref() == term {
            return Some(*b.clone());
        }
        if b.as_ref() == term {
            return Some(*a.clone());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{constant, var};
    use std::collections::HashMap;

    fn eval(e: &Expr, x_val: f64) -> f64 {
        let mut vars = HashMap::new();
        vars.insert("x".into(), x_val);
        vars.insert("a".into(), 2.0);
        vars.insert("b".into(), 3.0);
        vars.insert("c".into(), 4.0);
        e.eval(&vars).unwrap()
    }

    #[test]
    fn expand_distributes() {
        // a * (b + c) → a*b + a*c
        let e = var("a") * (var("b") + var("c"));
        let expanded = expand(&e);
        // Should evaluate the same: 2*(3+4) = 14, 2*3 + 2*4 = 14
        assert!((eval(&e, 0.0) - eval(&expanded, 0.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn expand_square_of_sum() {
        // (x + 1)^2 → x^2 + 2x + 1
        let e = Expr::Pow(Box::new(var("x") + constant(1.0)), Box::new(constant(2.0)));
        let expanded = expand(&e);
        // At x=3: (3+1)^2 = 16
        assert!((eval(&expanded, 3.0) - 16.0).abs() < 1e-10);
    }

    #[test]
    fn factor_out_common_term() {
        // x*a + x*b → x*(a+b)
        let x = var("x");
        let e = Expr::Add(
            Box::new(Expr::Mul(Box::new(x.clone()), Box::new(var("a")))),
            Box::new(Expr::Mul(Box::new(x.clone()), Box::new(var("b")))),
        );
        let factored = factor_out(&e, &x);
        // Should evaluate the same.
        assert!((eval(&e, 5.0) - eval(&factored, 5.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn factor_out_fails_gracefully() {
        // x + y — cannot factor out x from y.
        let e = var("x") + var("y");
        let factored = factor_out(&e, &var("x"));
        // Should return original expression unchanged.
        assert_eq!(factored, e);
    }
}
