//! Expression JIT for element-wise tensor operations.
//!
//! Instead of allocating intermediate tensors for each element-wise operation,
//! this module builds an expression tree and evaluates it in a single fused
//! pass over the data. This eliminates temporary allocations and improves
//! cache locality for chains of element-wise ops.
//!
//! # Examples
//!
//! ```
//! use scivex_core::Tensor;
//! use scivex_core::jit::Expr;
//!
//! let a = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0], vec![3]).unwrap();
//! let b = Tensor::from_vec(vec![4.0_f64, 5.0, 6.0], vec![3]).unwrap();
//! let c = Tensor::from_vec(vec![0.5_f64, 0.5, 0.5], vec![3]).unwrap();
//!
//! // (a + b) * c — no intermediate tensor for a + b
//! let result = Expr::input(&a)
//!     .add(Expr::input(&b))
//!     .mul(Expr::input(&c))
//!     .eval()
//!     .unwrap();
//!
//! assert_eq!(result.as_slice(), &[2.5, 3.5, 4.5]);
//! ```

use crate::Float;
use crate::Tensor;
use crate::error::{CoreError, Result};

/// An expression node in the computation graph.
///
/// Each variant represents either a leaf (tensor input or scalar constant)
/// or a fused element-wise operation. The tree is evaluated in one pass
/// via [`Expr::eval`], avoiding intermediate tensor allocations.
pub enum Expr<'a, T: Float> {
    /// A tensor input (leaf node).
    Input(&'a Tensor<T>),
    /// A scalar constant, broadcast to match the output shape.
    Scalar(T),
    /// Element-wise addition.
    Add(Box<Expr<'a, T>>, Box<Expr<'a, T>>),
    /// Element-wise subtraction.
    Sub(Box<Expr<'a, T>>, Box<Expr<'a, T>>),
    /// Element-wise multiplication.
    Mul(Box<Expr<'a, T>>, Box<Expr<'a, T>>),
    /// Element-wise division.
    Div(Box<Expr<'a, T>>, Box<Expr<'a, T>>),
    /// Unary negation.
    Neg(Box<Expr<'a, T>>),
    /// Element-wise square root.
    Sqrt(Box<Expr<'a, T>>),
    /// Element-wise exponential.
    Exp(Box<Expr<'a, T>>),
    /// Element-wise natural logarithm.
    Ln(Box<Expr<'a, T>>),
    /// Element-wise absolute value.
    Abs(Box<Expr<'a, T>>),
    /// Element-wise sine.
    Sin(Box<Expr<'a, T>>),
    /// Element-wise cosine.
    Cos(Box<Expr<'a, T>>),
    /// Element-wise power.
    Pow(Box<Expr<'a, T>>, Box<Expr<'a, T>>),
    /// Fused multiply-add: `a * b + c`.
    Fma(Box<Expr<'a, T>>, Box<Expr<'a, T>>, Box<Expr<'a, T>>),
    /// Clamp values to `[min, max]`.
    Clamp(Box<Expr<'a, T>>, T, T),
}

#[allow(clippy::should_implement_trait)]
impl<'a, T: Float> Expr<'a, T> {
    /// Create an input expression referencing a tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// use scivex_core::Tensor;
    /// use scivex_core::jit::Expr;
    ///
    /// let t = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0], vec![3]).unwrap();
    /// let result = Expr::input(&t).eval().unwrap();
    /// assert_eq!(result.as_slice(), &[1.0, 2.0, 3.0]);
    /// ```
    pub fn input(tensor: &'a Tensor<T>) -> Self {
        Expr::Input(tensor)
    }

    /// Create a scalar constant expression.
    pub fn scalar(val: T) -> Self {
        Expr::Scalar(val)
    }

    /// Element-wise addition: `self + other`.
    pub fn add(self, other: Self) -> Self {
        Expr::Add(Box::new(self), Box::new(other))
    }

    /// Element-wise subtraction: `self - other`.
    pub fn sub(self, other: Self) -> Self {
        Expr::Sub(Box::new(self), Box::new(other))
    }

    /// Element-wise multiplication: `self * other`.
    pub fn mul(self, other: Self) -> Self {
        Expr::Mul(Box::new(self), Box::new(other))
    }

    /// Element-wise division: `self / other`.
    pub fn div(self, other: Self) -> Self {
        Expr::Div(Box::new(self), Box::new(other))
    }

    /// Unary negation: `-self`.
    pub fn neg(self) -> Self {
        Expr::Neg(Box::new(self))
    }

    /// Element-wise square root.
    pub fn sqrt(self) -> Self {
        Expr::Sqrt(Box::new(self))
    }

    /// Element-wise exponential.
    pub fn exp(self) -> Self {
        Expr::Exp(Box::new(self))
    }

    /// Element-wise natural logarithm.
    pub fn ln(self) -> Self {
        Expr::Ln(Box::new(self))
    }

    /// Element-wise absolute value.
    pub fn abs(self) -> Self {
        Expr::Abs(Box::new(self))
    }

    /// Element-wise sine.
    pub fn sin(self) -> Self {
        Expr::Sin(Box::new(self))
    }

    /// Element-wise cosine.
    pub fn cos(self) -> Self {
        Expr::Cos(Box::new(self))
    }

    /// Element-wise power: `self ^ other`.
    pub fn pow(self, other: Self) -> Self {
        Expr::Pow(Box::new(self), Box::new(other))
    }

    /// Fused multiply-add: `self * b + c`.
    pub fn fma(self, b: Self, c: Self) -> Self {
        Expr::Fma(Box::new(self), Box::new(b), Box::new(c))
    }

    /// Clamp values to `[min, max]`.
    pub fn clamp(self, min: T, max: T) -> Self {
        Expr::Clamp(Box::new(self), min, max)
    }

    /// Evaluate the expression tree, producing a result tensor.
    ///
    /// All [`Expr::Input`] tensors referenced anywhere in the tree must have
    /// the same shape. Scalar nodes are broadcast to match. Returns an error
    /// if input shapes disagree or if no shape can be determined (i.e., the
    /// entire expression is purely scalar with no tensor inputs).
    pub fn eval(&self) -> Result<Tensor<T>> {
        let shape = collect_shape(self)?;
        let numel: usize = shape.iter().product();
        let mut result = Vec::with_capacity(numel);
        for i in 0..numel {
            result.push(self.eval_at(i));
        }
        Tensor::from_vec(result, shape)
    }

    /// Evaluate the expression at a single flat index.
    fn eval_at(&self, idx: usize) -> T {
        match self {
            Expr::Input(t) => t.as_slice()[idx],
            Expr::Scalar(v) => *v,
            Expr::Add(a, b) => a.eval_at(idx) + b.eval_at(idx),
            Expr::Sub(a, b) => a.eval_at(idx) - b.eval_at(idx),
            Expr::Mul(a, b) => a.eval_at(idx) * b.eval_at(idx),
            Expr::Div(a, b) => a.eval_at(idx) / b.eval_at(idx),
            Expr::Neg(a) => T::zero() - a.eval_at(idx),
            Expr::Sqrt(a) => a.eval_at(idx).sqrt(),
            Expr::Exp(a) => a.eval_at(idx).exp(),
            Expr::Ln(a) => a.eval_at(idx).ln(),
            Expr::Abs(a) => a.eval_at(idx).abs(),
            Expr::Sin(a) => a.eval_at(idx).sin(),
            Expr::Cos(a) => a.eval_at(idx).cos(),
            Expr::Pow(a, b) => a.eval_at(idx).powf(b.eval_at(idx)),
            Expr::Fma(a, b, c) => a.eval_at(idx) * b.eval_at(idx) + c.eval_at(idx),
            Expr::Clamp(a, min, max) => {
                let v = a.eval_at(idx);
                if v < *min {
                    *min
                } else if v > *max {
                    *max
                } else {
                    v
                }
            }
        }
    }
}

/// Traverse the expression tree, collect all input tensor shapes, and verify
/// they are identical. Returns the common shape, or an error if shapes differ.
///
/// If the expression contains no `Input` nodes (pure scalar), returns a
/// scalar shape `[1]`.
fn collect_shape<T: Float>(expr: &Expr<'_, T>) -> Result<Vec<usize>> {
    let mut shape: Option<Vec<usize>> = None;
    collect_shape_inner(expr, &mut shape)?;
    Ok(shape.unwrap_or_else(|| vec![1]))
}

fn collect_shape_inner<T: Float>(expr: &Expr<'_, T>, shape: &mut Option<Vec<usize>>) -> Result<()> {
    match expr {
        Expr::Input(t) => {
            let s = t.shape();
            match shape {
                Some(existing) if existing.as_slice() != s => {
                    return Err(CoreError::DimensionMismatch {
                        expected: existing.clone(),
                        got: s.to_vec(),
                    });
                }
                None => {
                    *shape = Some(s.to_vec());
                }
                _ => {}
            }
            Ok(())
        }
        Expr::Scalar(_) => Ok(()),
        Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) | Expr::Div(a, b) | Expr::Pow(a, b) => {
            collect_shape_inner(a, shape)?;
            collect_shape_inner(b, shape)
        }
        Expr::Neg(a)
        | Expr::Sqrt(a)
        | Expr::Exp(a)
        | Expr::Ln(a)
        | Expr::Abs(a)
        | Expr::Sin(a)
        | Expr::Cos(a)
        | Expr::Clamp(a, _, _) => collect_shape_inner(a, shape),
        Expr::Fma(a, b, c) => {
            collect_shape_inner(a, shape)?;
            collect_shape_inner(b, shape)?;
            collect_shape_inner(c, shape)
        }
    }
}

/// Convenience function: evaluate an expression built from tensors.
///
/// Equivalent to calling [`Expr::eval`] directly.
pub fn eval_expr<T: Float>(expr: &Expr<'_, T>) -> Result<Tensor<T>> {
    expr.eval()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expr_basic_arithmetic() {
        let a = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::from_vec(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2]).unwrap();
        let c = Tensor::from_vec(vec![2.0, 2.0, 2.0, 2.0], vec![2, 2]).unwrap();

        // (a + b) * c
        let result = Expr::input(&a)
            .add(Expr::input(&b))
            .mul(Expr::input(&c))
            .eval()
            .unwrap();

        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.as_slice(), &[22.0, 44.0, 66.0, 88.0]);
    }

    #[test]
    fn test_expr_unary_ops() {
        let a = Tensor::from_vec(vec![-4.0_f64, -9.0, -16.0], vec![3]).unwrap();

        // sqrt(abs(a))
        let result = Expr::input(&a).abs().sqrt().eval().unwrap();

        assert_eq!(result.shape(), &[3]);
        assert_eq!(result.as_slice(), &[2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_expr_fma() {
        let a = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0], vec![3]).unwrap();
        let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], vec![3]).unwrap();
        let c = Tensor::from_vec(vec![10.0, 20.0, 30.0], vec![3]).unwrap();

        // a * b + c via fma
        let result = Expr::input(&a)
            .fma(Expr::input(&b), Expr::input(&c))
            .eval()
            .unwrap();

        // Expected: [1*4+10, 2*5+20, 3*6+30] = [14, 30, 48]
        assert_eq!(result.as_slice(), &[14.0, 30.0, 48.0]);
    }

    #[test]
    fn test_expr_scalar_broadcast() {
        let a = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0], vec![4]).unwrap();

        // a + 2.0
        let result = Expr::input(&a).add(Expr::scalar(2.0)).eval().unwrap();

        assert_eq!(result.as_slice(), &[3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_expr_shape_mismatch() {
        let a = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0], vec![3]).unwrap();
        let b = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();

        let err = Expr::input(&a).add(Expr::input(&b)).eval();
        assert!(err.is_err());

        match err.unwrap_err() {
            CoreError::DimensionMismatch { expected, got } => {
                assert_eq!(expected, vec![3]);
                assert_eq!(got, vec![4]);
            }
            other => panic!("expected DimensionMismatch, got {other:?}"),
        }
    }

    #[test]
    fn test_expr_complex_chain() {
        // exp(a * 0.5) + cos(b)
        let a = Tensor::from_vec(vec![0.0_f64, 2.0, 4.0], vec![3]).unwrap();
        let b = Tensor::from_vec(vec![0.0, core::f64::consts::PI, 0.0], vec![3]).unwrap();

        let result = Expr::input(&a)
            .mul(Expr::scalar(0.5))
            .exp()
            .add(Expr::input(&b).cos())
            .eval()
            .unwrap();

        let expected = [
            (0.0_f64 * 0.5).exp() + 0.0_f64.cos(), // 1.0 + 1.0 = 2.0
            (2.0_f64 * 0.5).exp() + core::f64::consts::PI.cos(), // e^1 + (-1)
            (4.0_f64 * 0.5).exp() + 0.0_f64.cos(), // e^2 + 1.0
        ];

        let result_slice = result.as_slice();
        for (i, (&got, &exp)) in result_slice.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-12,
                "index {i}: got {got}, expected {exp}"
            );
        }
    }
}
