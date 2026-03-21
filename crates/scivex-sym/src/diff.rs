use crate::expr::{Expr, MathFn, cos, exp, ln, one, sin, zero};
use crate::simplify::simplify;

/// Compute the symbolic derivative of `expr` with respect to `var`.
///
/// The result is automatically simplified.
///
/// # Examples
///
/// ```
/// # use scivex_sym::{var, diff};
/// # use std::collections::HashMap;
/// let x = var("x");
/// // d/dx(x²) = 2x
/// let x_squared = x.clone() * x.clone();
/// let derivative = diff(&x_squared, "x");
/// let vars = HashMap::from([("x".to_string(), 3.0)]);
/// let val = derivative.eval(&vars).unwrap();
/// assert!((val - 6.0).abs() < 1e-10);
/// ```
#[must_use]
pub fn diff(expr: &Expr, var: &str) -> Expr {
    let raw = diff_raw(expr, var);
    simplify(&raw)
}

/// Compute the `n`-th derivative of `expr` with respect to `var`.
///
/// # Examples
///
/// ```
/// # use scivex_sym::{var, diff_n};
/// # use std::collections::HashMap;
/// let x = var("x");
/// let cubic = x.clone() * x.clone() * x.clone(); // x³
/// let d2 = diff_n(&cubic, "x", 2); // 6x
/// let vars = HashMap::from([("x".to_string(), 1.0)]);
/// assert!((d2.eval(&vars).unwrap() - 6.0).abs() < 1e-10);
/// ```
#[must_use]
pub fn diff_n(expr: &Expr, var: &str, n: usize) -> Expr {
    let mut result = expr.clone();
    for _ in 0..n {
        result = diff(&result, var);
    }
    result
}

/// Raw differentiation without simplification.
fn diff_raw(expr: &Expr, var: &str) -> Expr {
    match expr {
        // d/dx(c) = 0
        Expr::Const(_) => zero(),

        // d/dx(x) = 1, d/dx(y) = 0
        Expr::Var(name) => {
            if name == var {
                one()
            } else {
                zero()
            }
        }

        // d/dx(a + b) = da + db
        Expr::Add(a, b) => {
            let da = diff_raw(a, var);
            let db = diff_raw(b, var);
            Expr::Add(Box::new(da), Box::new(db))
        }

        // Product rule: d/dx(a * b) = da * b + a * db
        Expr::Mul(a, b) => {
            let da = diff_raw(a, var);
            let db = diff_raw(b, var);
            Expr::Add(
                Box::new(Expr::Mul(Box::new(da), b.clone())),
                Box::new(Expr::Mul(a.clone(), Box::new(db))),
            )
        }

        // Power rule: d/dx(f^g)
        // For f^c (constant exponent): c * f^(c-1) * f'
        // General case: f^g * (g' * ln(f) + g * f'/f)
        Expr::Pow(base, exp) => {
            let base_has_var = base.free_variables_contains(var);
            let exp_has_var = exp.free_variables_contains(var);

            if !base_has_var && !exp_has_var {
                // Both constant w.r.t. var.
                zero()
            } else if base_has_var && !exp_has_var {
                // f(x)^c  →  c * f^(c-1) * f'
                let df = diff_raw(base, var);
                let c_minus_1 = Expr::Add(Box::new(*exp.clone()), Box::new(Expr::Const(-1.0)));
                // c * f^(c-1) * f'
                Expr::Mul(
                    Box::new(Expr::Mul(
                        exp.clone(),
                        Box::new(Expr::Pow(base.clone(), Box::new(c_minus_1))),
                    )),
                    Box::new(df),
                )
            } else if !base_has_var && exp_has_var {
                // c^g(x) → c^g * ln(c) * g'
                let dg = diff_raw(exp, var);
                Expr::Mul(
                    Box::new(Expr::Mul(
                        Box::new(expr.clone()),
                        Box::new(ln(*base.clone())),
                    )),
                    Box::new(dg),
                )
            } else {
                // General: f^g * (g' * ln(f) + g * f'/f)
                let df = diff_raw(base, var);
                let dg = diff_raw(exp, var);
                let ln_f = ln(*base.clone());
                let f_prime_over_f = Expr::Mul(
                    Box::new(df),
                    Box::new(Expr::Pow(base.clone(), Box::new(Expr::Const(-1.0)))),
                );
                let inner = Expr::Add(
                    Box::new(Expr::Mul(Box::new(dg), Box::new(ln_f))),
                    Box::new(Expr::Mul(exp.clone(), Box::new(f_prime_over_f))),
                );
                Expr::Mul(Box::new(expr.clone()), Box::new(inner))
            }
        }

        // d/dx(-f) = -(df)
        Expr::Neg(inner) => Expr::Neg(Box::new(diff_raw(inner, var))),

        // Chain rule: d/dx(f(g)) = f'(g) * g'
        Expr::Fn(func, arg) => {
            let darg = diff_raw(arg, var);
            let outer_deriv = match func {
                MathFn::Sin => cos(*arg.clone()),
                MathFn::Cos => Expr::Neg(Box::new(sin(*arg.clone()))),
                MathFn::Tan => {
                    // sec^2(x) = 1 + tan^2(x)  →  but simpler: 1/cos^2(x)
                    Expr::Pow(Box::new(cos(*arg.clone())), Box::new(Expr::Const(-2.0)))
                }
                MathFn::Exp => exp(*arg.clone()),
                MathFn::Ln => {
                    // 1/arg
                    Expr::Pow(arg.clone(), Box::new(Expr::Const(-1.0)))
                }
                MathFn::Sqrt => {
                    // 1 / (2 * sqrt(arg))
                    Expr::Mul(
                        Box::new(Expr::Const(0.5)),
                        Box::new(Expr::Pow(
                            Box::new(Expr::Fn(MathFn::Sqrt, arg.clone())),
                            Box::new(Expr::Const(-1.0)),
                        )),
                    )
                }
                MathFn::Abs => {
                    // |x|' = x / |x| (undefined at 0, but symbolically fine)
                    Expr::Mul(
                        arg.clone(),
                        Box::new(Expr::Pow(
                            Box::new(Expr::Fn(MathFn::Abs, arg.clone())),
                            Box::new(Expr::Const(-1.0)),
                        )),
                    )
                }
            };
            Expr::Mul(Box::new(outer_deriv), Box::new(darg))
        }
    }
}

/// Helper trait to avoid `HashSet` allocation for a simple check.
trait ContainsVar {
    fn free_variables_contains(&self, var: &str) -> bool;
}

impl ContainsVar for Expr {
    fn free_variables_contains(&self, var: &str) -> bool {
        match self {
            Expr::Const(_) => false,
            Expr::Var(name) => name == var,
            Expr::Add(a, b) | Expr::Mul(a, b) | Expr::Pow(a, b) => {
                a.free_variables_contains(var) || b.free_variables_contains(var)
            }
            Expr::Neg(inner) | Expr::Fn(_, inner) => inner.free_variables_contains(var),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{constant, var};
    use std::collections::HashMap;

    fn eval(e: &Expr, x_val: f64) -> f64 {
        let mut vars = HashMap::new();
        vars.insert("x".into(), x_val);
        e.eval(&vars).unwrap()
    }

    #[test]
    fn diff_constant() {
        let e = constant(5.0);
        assert_eq!(diff(&e, "x"), zero());
    }

    #[test]
    fn diff_variable() {
        assert_eq!(diff(&var("x"), "x"), one());
        assert_eq!(diff(&var("y"), "x"), zero());
    }

    #[test]
    fn diff_power_rule() {
        // d/dx(x^3) = 3x^2
        let e = Expr::Pow(Box::new(var("x")), Box::new(constant(3.0)));
        let d = diff(&e, "x");
        // Evaluate at x=2: should be 3*4 = 12
        assert!((eval(&d, 2.0) - 12.0).abs() < 1e-10);
    }

    #[test]
    fn diff_product_rule() {
        // d/dx(x * x) = 2x
        let e = var("x") * var("x");
        let d = diff(&e, "x");
        assert!((eval(&d, 3.0) - 6.0).abs() < 1e-10);
    }

    #[test]
    fn diff_sin() {
        // d/dx(sin(x)) = cos(x)
        let e = sin(var("x"));
        let d = diff(&e, "x");
        // At x=0: cos(0) = 1
        assert!((eval(&d, 0.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn diff_cos() {
        // d/dx(cos(x)) = -sin(x)
        let e = cos(var("x"));
        let d = diff(&e, "x");
        // At x=pi/2: -sin(pi/2) = -1
        assert!((eval(&d, std::f64::consts::FRAC_PI_2) - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn diff_exp_func() {
        // d/dx(exp(x)) = exp(x)
        let e = exp(var("x"));
        let d = diff(&e, "x");
        assert!((eval(&d, 1.0) - std::f64::consts::E).abs() < 1e-10);
    }

    #[test]
    fn diff_ln_func() {
        // d/dx(ln(x)) = 1/x
        let e = ln(var("x"));
        let d = diff(&e, "x");
        assert!((eval(&d, 2.0) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn diff_chain_rule() {
        // d/dx(sin(x^2)) = cos(x^2) * 2x
        let x_sq = Expr::Pow(Box::new(var("x")), Box::new(constant(2.0)));
        let e = sin(x_sq);
        let d = diff(&e, "x");
        // At x=1: cos(1) * 2 ≈ 1.0806
        let expected = 1.0_f64.cos() * 2.0;
        assert!((eval(&d, 1.0) - expected).abs() < 1e-10);
    }

    #[test]
    fn diff_n_second_derivative() {
        // d²/dx²(x^3) = 6x
        let e = Expr::Pow(Box::new(var("x")), Box::new(constant(3.0)));
        let d2 = diff_n(&e, "x", 2);
        assert!((eval(&d2, 2.0) - 12.0).abs() < 1e-10);
    }
}
