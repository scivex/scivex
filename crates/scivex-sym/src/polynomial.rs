use crate::error::{Result, SymError};
use crate::expr::{Expr, constant, var};
use scivex_core::Float;

/// A polynomial with coefficients of type `T`.
///
/// Coefficients are stored in ascending power order: `coeffs[i]` is the
/// coefficient of `x^i`.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone, PartialEq)]
pub struct Polynomial<T: Float> {
    coeffs: Vec<T>,
}

impl<T: Float> Polynomial<T> {
    /// Create a polynomial from coefficients in ascending power order.
    ///
    /// `coeffs[0]` is the constant term, `coeffs[1]` is the `x` coefficient, etc.
    #[must_use]
    pub fn new(coeffs: Vec<T>) -> Self {
        let mut p = Self { coeffs };
        p.trim();
        p
    }

    /// The zero polynomial.
    #[must_use]
    pub fn zero() -> Self {
        Self {
            coeffs: vec![T::zero()],
        }
    }

    /// A constant polynomial.
    #[must_use]
    pub fn constant(c: T) -> Self {
        Self { coeffs: vec![c] }
    }

    /// Degree of the polynomial (0 for the zero polynomial).
    #[must_use]
    pub fn degree(&self) -> usize {
        if self.coeffs.len() <= 1 {
            0
        } else {
            self.coeffs.len() - 1
        }
    }

    /// Borrow the coefficients slice.
    #[must_use]
    pub fn coeffs(&self) -> &[T] {
        &self.coeffs
    }

    /// Evaluate the polynomial at `x` using Horner's method.
    #[must_use]
    pub fn eval(&self, x: T) -> T {
        let mut result = T::zero();
        for c in self.coeffs.iter().rev() {
            result = result * x + *c;
        }
        result
    }

    /// Add two polynomials.
    #[must_use]
    pub fn add(&self, other: &Self) -> Self {
        let len = self.coeffs.len().max(other.coeffs.len());
        let mut coeffs = Vec::with_capacity(len);
        for i in 0..len {
            let a = self.coeffs.get(i).copied().unwrap_or(T::zero());
            let b = other.coeffs.get(i).copied().unwrap_or(T::zero());
            coeffs.push(a + b);
        }
        let mut p = Self { coeffs };
        p.trim();
        p
    }

    /// Multiply two polynomials.
    #[must_use]
    pub fn mul(&self, other: &Self) -> Self {
        if self.coeffs.is_empty() || other.coeffs.is_empty() {
            return Self::zero();
        }
        let len = self.coeffs.len() + other.coeffs.len() - 1;
        let mut coeffs = vec![T::zero(); len];
        for (i, &a) in self.coeffs.iter().enumerate() {
            for (j, &b) in other.coeffs.iter().enumerate() {
                coeffs[i + j] += a * b;
            }
        }
        let mut p = Self { coeffs };
        p.trim();
        p
    }

    /// Remove trailing zero coefficients (but keep at least one).
    fn trim(&mut self) {
        while self.coeffs.len() > 1 && self.coeffs.last().copied() == Some(T::zero()) {
            self.coeffs.pop();
        }
    }

    /// Find real roots for polynomials of degree <= 2.
    pub fn roots(&self) -> Result<Vec<T>> {
        match self.degree() {
            0 => {
                // Constant: 0 if zero polynomial (infinite roots → empty vec for simplicity),
                // otherwise no roots.
                Ok(Vec::new())
            }
            1 => {
                // ax + b = 0  →  x = -b/a
                let a = self.coeffs[1];
                let b = self.coeffs[0];
                if a == T::zero() {
                    return Ok(Vec::new());
                }
                Ok(vec![T::zero() - b / a])
            }
            2 => {
                let a = self.coeffs[2];
                let b = self.coeffs[1];
                let c = self.coeffs[0];
                let four = T::one() + T::one() + T::one() + T::one();
                let two = T::one() + T::one();
                let disc = b * b - four * a * c;
                if disc < T::zero() {
                    return Ok(Vec::new());
                }
                let sqrt_disc = disc.sqrt();
                if disc == T::zero() {
                    Ok(vec![(T::zero() - b) / (two * a)])
                } else {
                    let r1 = (T::zero() - b + sqrt_disc) / (two * a);
                    let r2 = (T::zero() - b - sqrt_disc) / (two * a);
                    let mut roots = vec![r1, r2];
                    roots.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    Ok(roots)
                }
            }
            _ => Err(SymError::UnsupportedOperation {
                reason: "root finding for degree > 2 is not yet supported",
            }),
        }
    }
}

impl Polynomial<f64> {
    /// Convert to a symbolic `Expr` using the given variable name.
    #[must_use]
    #[allow(clippy::float_cmp)]
    pub fn to_expr(&self, var_name: &str) -> Expr {
        let mut terms: Vec<Expr> = Vec::new();
        for (i, &c) in self.coeffs.iter().enumerate() {
            if c == 0.0 {
                continue;
            }
            let term = if i == 0 {
                constant(c)
            } else if i == 1 {
                if c == 1.0 {
                    var(var_name)
                } else {
                    constant(c) * var(var_name)
                }
            } else {
                let power = Expr::Pow(Box::new(var(var_name)), Box::new(constant(i as f64)));
                if c == 1.0 { power } else { constant(c) * power }
            };
            terms.push(term);
        }
        if terms.is_empty() {
            return constant(0.0);
        }
        terms
            .into_iter()
            .reduce(|acc, t| acc + t)
            .unwrap_or_else(|| constant(0.0))
    }
}

impl Polynomial<f64> {
    /// Try to extract a polynomial from a symbolic expression.
    ///
    /// The expression must be a polynomial in `var` with constant coefficients.
    /// Currently supports expressions built from `Const`, `Var`, `Add`, `Mul`,
    /// `Neg`, and `Pow` with non-negative integer exponents.
    pub fn from_expr(expr: &Expr, var_name: &str) -> Result<Self> {
        Self::extract(expr, var_name)
    }

    fn extract(expr: &Expr, var_name: &str) -> Result<Self> {
        match expr {
            Expr::Const(v) => Ok(Polynomial::constant(*v)),
            Expr::Var(name) => {
                if name == var_name {
                    // x = 0 + 1*x
                    Ok(Polynomial::new(vec![0.0, 1.0]))
                } else {
                    // Treat other variables as unknown → error
                    Err(SymError::UndefinedVariable { name: name.clone() })
                }
            }
            Expr::Add(a, b) => {
                let pa = Self::extract(a, var_name)?;
                let pb = Self::extract(b, var_name)?;
                Ok(pa.add(&pb))
            }
            Expr::Mul(a, b) => {
                let pa = Self::extract(a, var_name)?;
                let pb = Self::extract(b, var_name)?;
                Ok(pa.mul(&pb))
            }
            Expr::Neg(inner) => {
                let p = Self::extract(inner, var_name)?;
                let neg_one = Polynomial::constant(-1.0);
                Ok(neg_one.mul(&p))
            }
            Expr::Pow(base, exp) => {
                let pb = Self::extract(base, var_name)?;
                #[allow(clippy::collapsible_if)]
                if let Some(n) = exp.as_const() {
                    if n >= 0.0 && (n - n.floor()).abs() < f64::EPSILON && n <= 20.0 {
                        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                        let ni = n as u32;
                        let mut result = Polynomial::constant(1.0);
                        for _ in 0..ni {
                            result = result.mul(&pb);
                        }
                        return Ok(result);
                    }
                }
                Err(SymError::UnsupportedOperation {
                    reason: "only non-negative integer exponents are supported for polynomial extraction",
                })
            }
            Expr::Fn(_, _) => Err(SymError::UnsupportedOperation {
                reason: "function calls cannot be represented as polynomials",
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eval_horner() {
        // 3x^2 + 2x + 1 at x=2 → 3*4 + 2*2 + 1 = 17
        let p = Polynomial::new(vec![1.0, 2.0, 3.0]);
        assert!((p.eval(2.0) - 17.0).abs() < f64::EPSILON);
    }

    #[test]
    fn degree() {
        let p = Polynomial::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(p.degree(), 2);

        let p = Polynomial::constant(5.0);
        assert_eq!(p.degree(), 0);
    }

    #[test]
    fn add_polynomials() {
        let p1 = Polynomial::new(vec![1.0, 2.0]);
        let p2 = Polynomial::new(vec![3.0, 4.0, 5.0]);
        let sum = p1.add(&p2);
        assert_eq!(sum.coeffs(), &[4.0, 6.0, 5.0]);
    }

    #[test]
    fn mul_polynomials() {
        // (1 + x) * (1 + x) = 1 + 2x + x^2
        let p = Polynomial::new(vec![1.0, 1.0]);
        let sq = p.mul(&p);
        assert_eq!(sq.coeffs(), &[1.0, 2.0, 1.0]);
    }

    #[test]
    fn roots_linear() {
        // 2x + 6 = 0 → x = -3
        let p = Polynomial::new(vec![6.0, 2.0]);
        let r = p.roots().unwrap();
        assert_eq!(r.len(), 1);
        assert!((r[0] - (-3.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn roots_quadratic() {
        // x^2 - 5x + 6 = 0 → x = 2, 3
        let p = Polynomial::new(vec![6.0, -5.0, 1.0]);
        let r = p.roots().unwrap();
        assert_eq!(r.len(), 2);
        assert!((r[0] - 2.0).abs() < 1e-10);
        assert!((r[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn roots_no_real() {
        // x^2 + 1 = 0 → no real roots
        let p = Polynomial::new(vec![1.0, 0.0, 1.0]);
        let r = p.roots().unwrap();
        assert!(r.is_empty());
    }

    #[test]
    fn to_expr_and_back() {
        let p = Polynomial::new(vec![1.0, 2.0, 3.0]);
        let e = p.to_expr("x");
        let p2 = Polynomial::from_expr(&e, "x").unwrap();
        // Evaluate at a few points to confirm equivalence.
        for &x in &[0.0, 1.0, 2.0, -1.0] {
            assert!((p.eval(x) - p2.eval(x)).abs() < 1e-10);
        }
    }

    #[test]
    fn from_expr_roundtrip() {
        // x^2 + 3x + 2
        let x = var("x");
        let e = Expr::Pow(Box::new(x.clone()), Box::new(constant(2.0)))
            + constant(3.0) * x
            + constant(2.0);
        let p = Polynomial::from_expr(&e, "x").unwrap();
        assert!((p.eval(0.0) - 2.0).abs() < 1e-10);
        assert!((p.eval(1.0) - 6.0).abs() < 1e-10);
    }

    #[test]
    fn trim_trailing_zeros() {
        let p = Polynomial::new(vec![1.0, 0.0, 0.0]);
        assert_eq!(p.degree(), 0);
        assert_eq!(p.coeffs(), &[1.0]);
    }
}
