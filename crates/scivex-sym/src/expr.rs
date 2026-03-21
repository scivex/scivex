use std::collections::{HashMap, HashSet};
use std::fmt;
use std::ops;

use crate::error::{Result, SymError};

/// Built-in mathematical functions.
///
/// # Examples
///
/// ```
/// # use scivex_sym::MathFn;
/// let f = MathFn::Sin;
/// assert_eq!(format!("{f}"), "sin");
/// ```
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MathFn {
    Sin,
    Cos,
    Tan,
    Exp,
    Ln,
    Sqrt,
    Abs,
}

impl fmt::Display for MathFn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Sin => "sin",
            Self::Cos => "cos",
            Self::Tan => "tan",
            Self::Exp => "exp",
            Self::Ln => "ln",
            Self::Sqrt => "sqrt",
            Self::Abs => "abs",
        };
        f.write_str(name)
    }
}

/// A symbolic expression AST.
///
/// `Sub` is represented as `Add(a, Neg(b))` and `Div` as `Mul(a, Pow(b, Const(-1)))`.
///
/// # Examples
///
/// ```
/// # use scivex_sym::{var, constant};
/// # use std::collections::HashMap;
/// let expr = var("x") + constant(1.0);
/// let mut vars = HashMap::new();
/// vars.insert("x".to_string(), 2.0);
/// assert!((expr.eval(&vars).unwrap() - 3.0).abs() < 1e-10);
/// ```
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// Numeric constant.
    Const(f64),
    /// Named variable.
    Var(String),
    /// Addition: `lhs + rhs`.
    Add(Box<Expr>, Box<Expr>),
    /// Multiplication: `lhs * rhs`.
    Mul(Box<Expr>, Box<Expr>),
    /// Exponentiation: `base ^ exp`.
    Pow(Box<Expr>, Box<Expr>),
    /// Negation: `-expr`.
    Neg(Box<Expr>),
    /// Function application: `f(arg)`.
    Fn(MathFn, Box<Expr>),
}

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

/// Create a constant expression.
///
/// # Examples
///
/// ```
/// # use scivex_sym::constant;
/// # use std::collections::HashMap;
/// let five = constant(5.0);
/// let val = five.eval(&HashMap::new()).unwrap();
/// assert!((val - 5.0).abs() < 1e-10);
/// ```
#[must_use]
pub fn constant(v: f64) -> Expr {
    Expr::Const(v)
}

/// Create a variable expression.
///
/// # Examples
///
/// ```
/// # use scivex_sym::var;
/// # use std::collections::HashMap;
/// let x = var("x");
/// let mut vars = HashMap::new();
/// vars.insert("x".to_string(), 7.0);
/// assert!((x.eval(&vars).unwrap() - 7.0).abs() < 1e-10);
/// ```
#[must_use]
pub fn var(name: &str) -> Expr {
    Expr::Var(name.to_owned())
}

/// The additive identity.
///
/// # Examples
///
/// ```
/// # use scivex_sym::zero;
/// assert!(zero().is_zero());
/// ```
#[must_use]
pub fn zero() -> Expr {
    Expr::Const(0.0)
}

/// The multiplicative identity.
///
/// # Examples
///
/// ```
/// # use scivex_sym::one;
/// assert!(one().is_one());
/// ```
#[must_use]
pub fn one() -> Expr {
    Expr::Const(1.0)
}

/// The constant pi.
///
/// # Examples
///
/// ```
/// # use scivex_sym::pi;
/// assert!(pi().as_const().unwrap() > 3.14);
/// ```
#[must_use]
pub fn pi() -> Expr {
    Expr::Const(std::f64::consts::PI)
}

/// The constant e.
///
/// # Examples
///
/// ```
/// # use scivex_sym::e;
/// assert!(e().as_const().unwrap() > 2.71);
/// ```
#[must_use]
pub fn e() -> Expr {
    Expr::Const(std::f64::consts::E)
}

/// `sin(expr)`
///
/// # Examples
///
/// ```
/// # use scivex_sym::{sin, pi};
/// # use std::collections::HashMap;
/// let val = sin(pi()).eval(&HashMap::new()).unwrap();
/// assert!(val.abs() < 1e-10);
/// ```
#[must_use]
pub fn sin(expr: Expr) -> Expr {
    Expr::Fn(MathFn::Sin, Box::new(expr))
}

/// `cos(expr)`
///
/// # Examples
///
/// ```
/// # use scivex_sym::{cos, constant};
/// # use std::collections::HashMap;
/// let val = cos(constant(0.0)).eval(&HashMap::new()).unwrap();
/// assert!((val - 1.0).abs() < 1e-10);
/// ```
#[must_use]
pub fn cos(expr: Expr) -> Expr {
    Expr::Fn(MathFn::Cos, Box::new(expr))
}

/// `tan(expr)`
///
/// # Examples
///
/// ```
/// # use scivex_sym::{tan, constant};
/// # use std::collections::HashMap;
/// let val = tan(constant(0.0)).eval(&HashMap::new()).unwrap();
/// assert!(val.abs() < 1e-10);
/// ```
#[must_use]
pub fn tan(expr: Expr) -> Expr {
    Expr::Fn(MathFn::Tan, Box::new(expr))
}

/// `exp(expr)`
///
/// # Examples
///
/// ```
/// # use scivex_sym::{exp, constant};
/// # use std::collections::HashMap;
/// let val = exp(constant(0.0)).eval(&HashMap::new()).unwrap();
/// assert!((val - 1.0).abs() < 1e-10);
/// ```
#[must_use]
pub fn exp(expr: Expr) -> Expr {
    Expr::Fn(MathFn::Exp, Box::new(expr))
}

/// `ln(expr)`
///
/// # Examples
///
/// ```
/// # use scivex_sym::{ln, e};
/// # use std::collections::HashMap;
/// let val = ln(e()).eval(&HashMap::new()).unwrap();
/// assert!((val - 1.0).abs() < 1e-10);
/// ```
#[must_use]
pub fn ln(expr: Expr) -> Expr {
    Expr::Fn(MathFn::Ln, Box::new(expr))
}

/// `sqrt(expr)`
///
/// # Examples
///
/// ```
/// # use scivex_sym::{sqrt, constant};
/// # use std::collections::HashMap;
/// let val = sqrt(constant(4.0)).eval(&HashMap::new()).unwrap();
/// assert!((val - 2.0).abs() < 1e-10);
/// ```
#[must_use]
pub fn sqrt(expr: Expr) -> Expr {
    Expr::Fn(MathFn::Sqrt, Box::new(expr))
}

/// `|expr|`
///
/// # Examples
///
/// ```
/// # use scivex_sym::{abs, constant};
/// # use std::collections::HashMap;
/// let val = abs(constant(-3.0)).eval(&HashMap::new()).unwrap();
/// assert!((val - 3.0).abs() < 1e-10);
/// ```
#[must_use]
pub fn abs(expr: Expr) -> Expr {
    Expr::Fn(MathFn::Abs, Box::new(expr))
}

// ---------------------------------------------------------------------------
// Core methods
// ---------------------------------------------------------------------------

impl Expr {
    /// Evaluate the expression given concrete variable bindings.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_sym::expr::{var, constant};
    /// # use std::collections::HashMap;
    /// let expr = constant(2.0) * var("x") + constant(1.0);
    /// let vars = HashMap::from([("x".to_string(), 3.0)]);
    /// assert!((expr.eval(&vars).unwrap() - 7.0).abs() < 1e-10);
    /// ```
    pub fn eval(&self, vars: &HashMap<String, f64>) -> Result<f64> {
        match self {
            Self::Const(v) => Ok(*v),
            Self::Var(name) => vars
                .get(name)
                .copied()
                .ok_or_else(|| SymError::UndefinedVariable { name: name.clone() }),
            Self::Add(a, b) => Ok(a.eval(vars)? + b.eval(vars)?),
            Self::Mul(a, b) => {
                let av = a.eval(vars)?;
                let bv = b.eval(vars)?;
                Ok(av * bv)
            }
            Self::Pow(base, exp) => {
                let bv = base.eval(vars)?;
                let ev = exp.eval(vars)?;
                // Check for 0^negative (division by zero).
                if bv == 0.0 && ev < 0.0 {
                    return Err(SymError::DivisionByZero);
                }
                Ok(bv.powf(ev))
            }
            Self::Neg(inner) => Ok(-inner.eval(vars)?),
            Self::Fn(func, arg) => {
                let v = arg.eval(vars)?;
                Ok(match func {
                    MathFn::Sin => v.sin(),
                    MathFn::Cos => v.cos(),
                    MathFn::Tan => v.tan(),
                    MathFn::Exp => v.exp(),
                    MathFn::Ln => v.ln(),
                    MathFn::Sqrt => v.sqrt(),
                    MathFn::Abs => v.abs(),
                })
            }
        }
    }

    /// Replace every occurrence of `var` with `replacement`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_sym::expr::{var, constant};
    /// # use std::collections::HashMap;
    /// let expr = var("x") + constant(1.0);
    /// let replaced = expr.substitute("x", &constant(5.0));
    /// assert!((replaced.eval(&HashMap::new()).unwrap() - 6.0).abs() < 1e-10);
    /// ```
    #[must_use]
    pub fn substitute(&self, var: &str, replacement: &Expr) -> Expr {
        match self {
            Self::Const(_) => self.clone(),
            Self::Var(name) => {
                if name == var {
                    replacement.clone()
                } else {
                    self.clone()
                }
            }
            Self::Add(a, b) => Expr::Add(
                Box::new(a.substitute(var, replacement)),
                Box::new(b.substitute(var, replacement)),
            ),
            Self::Mul(a, b) => Expr::Mul(
                Box::new(a.substitute(var, replacement)),
                Box::new(b.substitute(var, replacement)),
            ),
            Self::Pow(base, exp) => Expr::Pow(
                Box::new(base.substitute(var, replacement)),
                Box::new(exp.substitute(var, replacement)),
            ),
            Self::Neg(inner) => Expr::Neg(Box::new(inner.substitute(var, replacement))),
            Self::Fn(func, arg) => Expr::Fn(*func, Box::new(arg.substitute(var, replacement))),
        }
    }

    /// Collect all free variable names in the expression.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_sym::expr::{var, constant};
    /// let expr = var("x") + var("y") * constant(2.0);
    /// let vars = expr.free_variables();
    /// assert!(vars.contains("x"));
    /// assert!(vars.contains("y"));
    /// assert_eq!(vars.len(), 2);
    /// ```
    #[must_use]
    pub fn free_variables(&self) -> HashSet<String> {
        let mut set = HashSet::new();
        self.collect_vars(&mut set);
        set
    }

    fn collect_vars(&self, set: &mut HashSet<String>) {
        match self {
            Self::Const(_) => {}
            Self::Var(name) => {
                set.insert(name.clone());
            }
            Self::Add(a, b) | Self::Mul(a, b) | Self::Pow(a, b) => {
                a.collect_vars(set);
                b.collect_vars(set);
            }
            Self::Neg(inner) | Self::Fn(_, inner) => inner.collect_vars(set),
        }
    }

    /// Returns `true` if the expression is `Const(0.0)`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_sym::expr::constant;
    /// assert!(constant(0.0).is_zero());
    /// assert!(!constant(1.0).is_zero());
    /// ```
    #[must_use]
    pub fn is_zero(&self) -> bool {
        matches!(self, Self::Const(v) if *v == 0.0)
    }

    /// Returns `true` if the expression is `Const(1.0)`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_sym::expr::constant;
    /// assert!(constant(1.0).is_one());
    /// assert!(!constant(2.0).is_one());
    /// ```
    #[must_use]
    pub fn is_one(&self) -> bool {
        matches!(self, Self::Const(v) if (*v - 1.0).abs() < f64::EPSILON)
    }

    /// Returns `true` if the expression is a constant.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_sym::expr::{constant, var};
    /// assert!(constant(3.14).is_const());
    /// assert!(!var("x").is_const());
    /// ```
    #[must_use]
    pub fn is_const(&self) -> bool {
        matches!(self, Self::Const(_))
    }

    /// If the expression is a constant, return its value.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_sym::expr::{constant, var};
    /// assert_eq!(constant(42.0).as_const(), Some(42.0));
    /// assert_eq!(var("x").as_const(), None);
    /// ```
    #[must_use]
    pub fn as_const(&self) -> Option<f64> {
        match self {
            Self::Const(v) => Some(*v),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Operator overloading
// ---------------------------------------------------------------------------

impl ops::Add for Expr {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Expr::Add(Box::new(self), Box::new(rhs))
    }
}

impl ops::Sub for Expr {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Expr::Add(Box::new(self), Box::new(Expr::Neg(Box::new(rhs))))
    }
}

impl ops::Mul for Expr {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Expr::Mul(Box::new(self), Box::new(rhs))
    }
}

impl ops::Div for Expr {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        Expr::Mul(
            Box::new(self),
            Box::new(Expr::Pow(Box::new(rhs), Box::new(Expr::Const(-1.0)))),
        )
    }
}

impl ops::Neg for Expr {
    type Output = Self;
    fn neg(self) -> Self {
        Expr::Neg(Box::new(self))
    }
}

// ---------------------------------------------------------------------------
// Display
// ---------------------------------------------------------------------------

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Const(v) => {
                if (*v - std::f64::consts::PI).abs() < f64::EPSILON {
                    write!(f, "pi")
                } else if *v < 0.0 {
                    write!(f, "({v})")
                } else {
                    write!(f, "{v}")
                }
            }
            Self::Var(name) => f.write_str(name),
            Self::Add(a, b) => write!(f, "({a} + {b})"),
            Self::Mul(a, b) => write!(f, "({a} * {b})"),
            Self::Pow(base, exp) => write!(f, "({base}^{exp})"),
            Self::Neg(inner) => write!(f, "(-{inner})"),
            Self::Fn(func, arg) => write!(f, "{func}({arg})"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eval_const_and_var() {
        let e = constant(42.0);
        assert!((e.eval(&HashMap::new()).unwrap() - 42.0).abs() < f64::EPSILON);

        let x = var("x");
        let mut vars = HashMap::new();
        vars.insert("x".into(), 3.0);
        assert!((x.eval(&vars).unwrap() - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn eval_undefined_variable() {
        let x = var("x");
        let err = x.eval(&HashMap::new()).unwrap_err();
        assert!(matches!(err, SymError::UndefinedVariable { name } if name == "x"));
    }

    #[test]
    fn eval_division_by_zero() {
        // 1 / 0 = 1 * 0^(-1)
        let e = constant(1.0) / constant(0.0);
        let err = e.eval(&HashMap::new()).unwrap_err();
        assert!(matches!(err, SymError::DivisionByZero));
    }

    #[test]
    fn eval_arithmetic() {
        let mut vars = HashMap::new();
        vars.insert("x".into(), 2.0);
        // (x + 3) * 4 = 20
        let e = (var("x") + constant(3.0)) * constant(4.0);
        assert!((e.eval(&vars).unwrap() - 20.0).abs() < f64::EPSILON);
    }

    #[test]
    fn eval_functions() {
        let vars = HashMap::new();
        let e = sin(constant(0.0));
        assert!(e.eval(&vars).unwrap().abs() < f64::EPSILON);

        let e = cos(constant(0.0));
        assert!((e.eval(&vars).unwrap() - 1.0).abs() < f64::EPSILON);

        let e = exp(constant(0.0));
        assert!((e.eval(&vars).unwrap() - 1.0).abs() < f64::EPSILON);

        let e = ln(constant(1.0));
        assert!(e.eval(&vars).unwrap().abs() < f64::EPSILON);
    }

    #[test]
    fn substitute_works() {
        let e = var("x") + constant(1.0);
        let replaced = e.substitute("x", &constant(5.0));
        assert!((replaced.eval(&HashMap::new()).unwrap() - 6.0).abs() < f64::EPSILON);
    }

    #[test]
    fn free_variables_collected() {
        let e = var("x") * var("y") + sin(var("x"));
        let fv = e.free_variables();
        assert!(fv.contains("x"));
        assert!(fv.contains("y"));
        assert_eq!(fv.len(), 2);
    }

    #[test]
    fn display_formatting() {
        let e = var("x") + constant(1.0);
        let s = format!("{e}");
        assert_eq!(s, "(x + 1)");
    }

    #[test]
    fn is_predicates() {
        assert!(zero().is_zero());
        assert!(one().is_one());
        assert!(constant(3.5).is_const());
        assert!(!var("x").is_const());
        assert_eq!(constant(7.0).as_const(), Some(7.0));
        assert_eq!(var("x").as_const(), None);
    }
}
