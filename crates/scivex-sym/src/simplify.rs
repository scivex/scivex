use crate::expr::Expr;

/// Apply algebraic simplification rules bottom-up.
///
/// Rules applied:
/// - Constant folding (`Const(a) op Const(b)` → `Const(result)`)
/// - `x + 0 → x`, `0 + x → x`
/// - `x * 1 → x`, `1 * x → x`
/// - `x * 0 → 0`, `0 * x → 0`
/// - `x ^ 0 → 1`
/// - `x ^ 1 → x`
/// - `Neg(Neg(x)) → x`
/// - `Neg(Const(v)) → Const(-v)`
///
/// # Examples
///
/// ```
/// # use scivex_sym::{var, constant, simplify};
/// let x = var("x");
/// let expr = x.clone() + constant(0.0); // x + 0
/// let simplified = simplify(&expr);
/// assert_eq!(format!("{}", simplified), "x");
/// ```
#[must_use]
pub fn simplify(expr: &Expr) -> Expr {
    // First, recursively simplify children.
    match expr {
        Expr::Const(_) | Expr::Var(_) => expr.clone(),
        Expr::Add(a, b) => {
            let a = simplify(a);
            let b = simplify(b);
            simplify_add(a, b)
        }
        Expr::Mul(a, b) => {
            let a = simplify(a);
            let b = simplify(b);
            simplify_mul(a, b)
        }
        Expr::Pow(base, exp) => {
            let base = simplify(base);
            let exp = simplify(exp);
            simplify_pow(base, exp)
        }
        Expr::Neg(inner) => {
            let inner = simplify(inner);
            simplify_neg(inner)
        }
        Expr::Fn(func, arg) => {
            let arg = simplify(arg);
            // Constant folding for functions.
            if let Some(v) = arg.as_const() {
                let result = match func {
                    crate::expr::MathFn::Sin => v.sin(),
                    crate::expr::MathFn::Cos => v.cos(),
                    crate::expr::MathFn::Tan => v.tan(),
                    crate::expr::MathFn::Exp => v.exp(),
                    crate::expr::MathFn::Ln => v.ln(),
                    crate::expr::MathFn::Sqrt => v.sqrt(),
                    crate::expr::MathFn::Abs => v.abs(),
                };
                return Expr::Const(result);
            }
            Expr::Fn(*func, Box::new(arg))
        }
    }
}

fn simplify_add(a: Expr, b: Expr) -> Expr {
    // Constant folding.
    if let (Some(av), Some(bv)) = (a.as_const(), b.as_const()) {
        return Expr::Const(av + bv);
    }
    // x + 0 → x
    if b.is_zero() {
        return a;
    }
    // 0 + x → x
    if a.is_zero() {
        return b;
    }
    Expr::Add(Box::new(a), Box::new(b))
}

fn simplify_mul(a: Expr, b: Expr) -> Expr {
    // Constant folding.
    if let (Some(av), Some(bv)) = (a.as_const(), b.as_const()) {
        return Expr::Const(av * bv);
    }
    // x * 0 → 0 or 0 * x → 0
    if a.is_zero() || b.is_zero() {
        return Expr::Const(0.0);
    }
    // x * 1 → x
    if b.is_one() {
        return a;
    }
    // 1 * x → x
    if a.is_one() {
        return b;
    }
    Expr::Mul(Box::new(a), Box::new(b))
}

fn simplify_pow(base: Expr, exp: Expr) -> Expr {
    // Constant folding.
    if let (Some(bv), Some(ev)) = (base.as_const(), exp.as_const()) {
        return Expr::Const(bv.powf(ev));
    }
    // x ^ 0 → 1
    if exp.is_zero() {
        return Expr::Const(1.0);
    }
    // x ^ 1 → x
    if exp.is_one() {
        return base;
    }
    Expr::Pow(Box::new(base), Box::new(exp))
}

fn simplify_neg(inner: Expr) -> Expr {
    // Neg(Neg(x)) → x
    if let Expr::Neg(x) = inner {
        return *x;
    }
    // Neg(Const(v)) → Const(-v)
    if let Some(v) = inner.as_const() {
        return Expr::Const(-v);
    }
    Expr::Neg(Box::new(inner))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{constant, one, var, zero};

    #[test]
    fn x_plus_zero() {
        let e = Expr::Add(Box::new(var("x")), Box::new(zero()));
        let s = simplify(&e);
        assert_eq!(s, var("x"));
    }

    #[test]
    fn zero_plus_x() {
        let e = Expr::Add(Box::new(zero()), Box::new(var("x")));
        let s = simplify(&e);
        assert_eq!(s, var("x"));
    }

    #[test]
    fn x_times_one() {
        let e = Expr::Mul(Box::new(var("x")), Box::new(one()));
        let s = simplify(&e);
        assert_eq!(s, var("x"));
    }

    #[test]
    fn one_times_x() {
        let e = Expr::Mul(Box::new(one()), Box::new(var("x")));
        let s = simplify(&e);
        assert_eq!(s, var("x"));
    }

    #[test]
    fn x_times_zero() {
        let e = Expr::Mul(Box::new(var("x")), Box::new(zero()));
        let s = simplify(&e);
        assert_eq!(s, zero());
    }

    #[test]
    fn x_pow_zero() {
        let e = Expr::Pow(Box::new(var("x")), Box::new(zero()));
        let s = simplify(&e);
        assert_eq!(s, one());
    }

    #[test]
    fn x_pow_one() {
        let e = Expr::Pow(Box::new(var("x")), Box::new(one()));
        let s = simplify(&e);
        assert_eq!(s, var("x"));
    }

    #[test]
    fn neg_neg_x() {
        let e = Expr::Neg(Box::new(Expr::Neg(Box::new(var("x")))));
        let s = simplify(&e);
        assert_eq!(s, var("x"));
    }

    #[test]
    fn neg_const() {
        let e = Expr::Neg(Box::new(constant(5.0)));
        let s = simplify(&e);
        assert_eq!(s, constant(-5.0));
    }

    #[test]
    fn constant_folding() {
        let e = constant(2.0) + constant(3.0);
        let s = simplify(&e);
        assert_eq!(s, constant(5.0));

        let e = constant(4.0) * constant(5.0);
        let s = simplify(&e);
        assert_eq!(s, constant(20.0));
    }
}
