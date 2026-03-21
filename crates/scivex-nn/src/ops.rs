//! Differentiable operations on [`Variable`]s.
//!
//! Each operation records its inputs and a closure (`grad_fn`) so that
//! [`Variable::backward`] can compute gradients via reverse-mode autodiff.

use std::ops;

use scivex_core::{Float, Tensor};

use crate::variable::Variable;

// ── Element-wise binary ops ─────────────────────────────────────────

/// Element-wise addition of two variables.
///
/// # Examples
///
/// ```
/// # use scivex_core::Tensor;
/// # use scivex_nn::variable::Variable;
/// # use scivex_nn::ops::add;
/// let a = Variable::new(Tensor::from_vec(vec![1.0_f64, 2.0], vec![2]).unwrap(), false);
/// let b = Variable::new(Tensor::from_vec(vec![3.0_f64, 4.0], vec![2]).unwrap(), false);
/// let c = add(&a, &b);
/// assert_eq!(c.data().as_slice(), &[4.0, 6.0]);
/// ```
pub fn add<T: Float>(a: &Variable<T>, b: &Variable<T>) -> Variable<T> {
    let data = &a.data() + &b.data();
    Variable::from_op(
        data,
        vec![a.clone(), b.clone()],
        Box::new(|g: &Tensor<T>| vec![g.clone(), g.clone()]),
    )
}

/// Element-wise subtraction.
///
/// # Examples
///
/// ```
/// # use scivex_core::Tensor;
/// # use scivex_nn::variable::Variable;
/// # use scivex_nn::ops::sub;
/// let a = Variable::new(Tensor::from_vec(vec![5.0_f64, 3.0], vec![2]).unwrap(), false);
/// let b = Variable::new(Tensor::from_vec(vec![1.0_f64, 1.0], vec![2]).unwrap(), false);
/// let c = sub(&a, &b);
/// assert_eq!(c.data().as_slice(), &[4.0, 2.0]);
/// ```
pub fn sub<T: Float>(a: &Variable<T>, b: &Variable<T>) -> Variable<T> {
    let data = &a.data() - &b.data();
    Variable::from_op(
        data,
        vec![a.clone(), b.clone()],
        Box::new(|g: &Tensor<T>| vec![g.clone(), -g]),
    )
}

/// Element-wise multiplication (Hadamard product).
///
/// # Examples
///
/// ```
/// # use scivex_core::Tensor;
/// # use scivex_nn::variable::Variable;
/// # use scivex_nn::ops::mul;
/// let a = Variable::new(Tensor::from_vec(vec![2.0_f64, 3.0], vec![2]).unwrap(), true);
/// let b = Variable::new(Tensor::from_vec(vec![4.0_f64, 5.0], vec![2]).unwrap(), true);
/// let c = mul(&a, &b);
/// assert_eq!(c.data().as_slice(), &[8.0, 15.0]);
/// ```
pub fn mul<T: Float>(a: &Variable<T>, b: &Variable<T>) -> Variable<T> {
    let a_data = a.data();
    let b_data = b.data();
    let data = &a_data * &b_data;
    Variable::from_op(
        data,
        vec![a.clone(), b.clone()],
        Box::new(move |g: &Tensor<T>| {
            let ga = g
                .zip_map(&b_data, |gi, bi| gi * bi)
                .expect("shapes match from forward pass");
            let gb = g
                .zip_map(&a_data, |gi, ai| gi * ai)
                .expect("shapes match from forward pass");
            vec![ga, gb]
        }),
    )
}

/// Negation.
///
/// # Examples
///
/// ```
/// # use scivex_core::Tensor;
/// # use scivex_nn::variable::Variable;
/// # use scivex_nn::ops::neg;
/// let a = Variable::new(Tensor::from_vec(vec![1.0_f64, -2.0], vec![2]).unwrap(), false);
/// let b = neg(&a);
/// assert_eq!(b.data().as_slice(), &[-1.0, 2.0]);
/// ```
pub fn neg<T: Float>(a: &Variable<T>) -> Variable<T> {
    let data = -&a.data();
    Variable::from_op(data, vec![a.clone()], Box::new(|g: &Tensor<T>| vec![-g]))
}

// ── Matrix operations ───────────────────────────────────────────────

/// Matrix multiplication: `a @ b`.
///
/// `a` has shape `[m, k]`, `b` has shape `[k, n]`, result is `[m, n]`.
///
/// # Examples
///
/// ```
/// # use scivex_core::Tensor;
/// # use scivex_nn::variable::Variable;
/// # use scivex_nn::ops::matmul;
/// let a = Variable::new(Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0], vec![2, 2]).unwrap(), false);
/// let b = Variable::new(Tensor::from_vec(vec![1.0_f64, 0.0, 0.0, 1.0], vec![2, 2]).unwrap(), false);
/// let c = matmul(&a, &b); // identity matmul
/// assert_eq!(c.data().as_slice(), &[1.0, 2.0, 3.0, 4.0]);
/// ```
pub fn matmul<T: Float>(a: &Variable<T>, b: &Variable<T>) -> Variable<T> {
    let a_data = a.data();
    let b_data = b.data();
    let data = a_data
        .matmul(&b_data)
        .expect("matmul shapes validated at call site");
    Variable::from_op(
        data,
        vec![a.clone(), b.clone()],
        Box::new(move |g: &Tensor<T>| {
            // grad_a = g @ b^T
            let bt = b_data.transpose().expect("2-D from forward pass");
            let ga = g.matmul(&bt).expect("shapes match from forward pass");
            // grad_b = a^T @ g
            let at = a_data.transpose().expect("2-D from forward pass");
            let gb = at.matmul(g).expect("shapes match from forward pass");
            vec![ga, gb]
        }),
    )
}

// ── Reductions ──────────────────────────────────────────────────────

/// Sum all elements to a scalar variable.
///
/// # Examples
///
/// ```
/// # use scivex_core::Tensor;
/// # use scivex_nn::variable::Variable;
/// # use scivex_nn::ops::sum;
/// let a = Variable::new(Tensor::from_vec(vec![1.0_f64, 2.0, 3.0], vec![3]).unwrap(), false);
/// let s = sum(&a);
/// assert_eq!(s.data().as_slice(), &[6.0]);
/// ```
pub fn sum<T: Float>(a: &Variable<T>) -> Variable<T> {
    let s = a.data().sum();
    let shape = a.shape();
    let data = Tensor::from_vec(vec![s], vec![1]).expect("scalar tensor");
    Variable::from_op(
        data,
        vec![a.clone()],
        Box::new(move |g: &Tensor<T>| {
            // Broadcast scalar grad to input shape.
            let g_val = g.as_slice()[0];
            vec![Tensor::full(shape.clone(), g_val)]
        }),
    )
}

/// Mean of all elements to a scalar variable.
///
/// # Examples
///
/// ```
/// # use scivex_core::Tensor;
/// # use scivex_nn::variable::Variable;
/// # use scivex_nn::ops::mean;
/// let a = Variable::new(Tensor::from_vec(vec![2.0_f64, 4.0], vec![2]).unwrap(), false);
/// let m = mean(&a);
/// assert_eq!(m.data().as_slice(), &[3.0]);
/// ```
pub fn mean<T: Float>(a: &Variable<T>) -> Variable<T> {
    let n = a.data().numel();
    let m = a.data().mean();
    let shape = a.shape();
    let data = Tensor::from_vec(vec![m], vec![1]).expect("scalar tensor");
    Variable::from_op(
        data,
        vec![a.clone()],
        Box::new(move |g: &Tensor<T>| {
            let g_val = g.as_slice()[0];
            let scale = g_val / T::from_usize(n);
            vec![Tensor::full(shape.clone(), scale)]
        }),
    )
}

/// Element-wise power: `a^exponent`.
///
/// # Examples
///
/// ```
/// # use scivex_core::Tensor;
/// # use scivex_nn::variable::Variable;
/// # use scivex_nn::ops::pow;
/// let a = Variable::new(Tensor::from_vec(vec![2.0_f64, 3.0], vec![2]).unwrap(), false);
/// let b = pow(&a, 2.0);
/// assert_eq!(b.data().as_slice(), &[4.0, 9.0]);
/// ```
pub fn pow<T: Float>(a: &Variable<T>, exponent: T) -> Variable<T> {
    let a_data = a.data();
    let data = a_data.powf(exponent);
    Variable::from_op(
        data,
        vec![a.clone()],
        Box::new(move |g: &Tensor<T>| {
            // d/da (a^n) = n * a^(n-1)
            let n_minus_1 = exponent - T::one();
            let deriv = a_data.powf(n_minus_1).map(|v| exponent * v);
            let grad = g
                .zip_map(&deriv, |gi, di| gi * di)
                .expect("shapes match from forward pass");
            vec![grad]
        }),
    )
}

/// Scalar multiplication: `variable * scalar_value`.
///
/// # Examples
///
/// ```
/// # use scivex_core::Tensor;
/// # use scivex_nn::variable::Variable;
/// # use scivex_nn::ops::scalar_mul;
/// let a = Variable::new(Tensor::from_vec(vec![2.0_f64, 3.0], vec![2]).unwrap(), false);
/// let b = scalar_mul(&a, 5.0);
/// assert_eq!(b.data().as_slice(), &[10.0, 15.0]);
/// ```
pub fn scalar_mul<T: Float>(a: &Variable<T>, scalar: T) -> Variable<T> {
    let data = &a.data() * scalar;
    Variable::from_op(
        data,
        vec![a.clone()],
        Box::new(move |g: &Tensor<T>| vec![g.map(|v| v * scalar)]),
    )
}

/// Scalar division: `variable / scalar_value`.
///
/// # Examples
///
/// ```
/// # use scivex_core::Tensor;
/// # use scivex_nn::variable::Variable;
/// # use scivex_nn::ops::scalar_div;
/// let a = Variable::new(Tensor::from_vec(vec![10.0_f64, 6.0], vec![2]).unwrap(), false);
/// let b = scalar_div(&a, 2.0);
/// assert_eq!(b.data().as_slice(), &[5.0, 3.0]);
/// ```
pub fn scalar_div<T: Float>(a: &Variable<T>, scalar: T) -> Variable<T> {
    scalar_mul(a, T::one() / scalar)
}

// ── Bias-add helper (manual broadcasting) ───────────────────────────

/// Add a 1-D bias `[out]` to a 2-D input `[batch, out]` (row-wise broadcast).
///
/// # Examples
///
/// ```
/// # use scivex_core::Tensor;
/// # use scivex_nn::variable::Variable;
/// # use scivex_nn::ops::add_bias;
/// let x = Variable::new(Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0], vec![2, 2]).unwrap(), false);
/// let b = Variable::new(Tensor::from_vec(vec![0.1_f64, 0.2], vec![2]).unwrap(), false);
/// let y = add_bias(&x, &b);
/// assert!((y.data().as_slice()[0] - 1.1).abs() < 1e-10);
/// ```
pub fn add_bias<T: Float>(input: &Variable<T>, bias: &Variable<T>) -> Variable<T> {
    let x = input.data();
    let b = bias.data();
    let shape = x.shape().to_vec();
    let rows = shape[0];
    let cols = shape[1];

    // Manually broadcast: each row gets bias added.
    let mut out_data = Vec::with_capacity(rows * cols);
    let b_slice = b.as_slice();
    let x_slice = x.as_slice();
    for r in 0..rows {
        for c in 0..cols {
            out_data.push(x_slice[r * cols + c] + b_slice[c]);
        }
    }
    let data =
        Tensor::from_vec(out_data, shape).expect("output data length matches shape from input");

    let cols_copy = cols;
    Variable::from_op(
        data,
        vec![input.clone(), bias.clone()],
        Box::new(move |g: &Tensor<T>| {
            // grad_input = g (same shape)
            let g_input = g.clone();
            // grad_bias = sum over rows (reduce axis 0)
            let g_slice = g.as_slice();
            let g_rows = g.shape()[0];
            let mut bias_grad = vec![T::zero(); cols_copy];
            for r in 0..g_rows {
                for c in 0..cols_copy {
                    bias_grad[c] += g_slice[r * cols_copy + c];
                }
            }
            let g_bias = Tensor::from_vec(bias_grad, vec![cols_copy])
                .expect("bias grad length matches feature count");
            vec![g_input, g_bias]
        }),
    )
}

// ── Operator overloads ──────────────────────────────────────────────

impl<T: Float> ops::Add for &Variable<T> {
    type Output = Variable<T>;
    fn add(self, rhs: Self) -> Variable<T> {
        add(self, rhs)
    }
}

impl<T: Float> ops::Add for Variable<T> {
    type Output = Variable<T>;
    fn add(self, rhs: Self) -> Variable<T> {
        add(&self, &rhs)
    }
}

impl<T: Float> ops::Sub for &Variable<T> {
    type Output = Variable<T>;
    fn sub(self, rhs: Self) -> Variable<T> {
        sub(self, rhs)
    }
}

impl<T: Float> ops::Sub for Variable<T> {
    type Output = Variable<T>;
    fn sub(self, rhs: Self) -> Variable<T> {
        sub(&self, &rhs)
    }
}

impl<T: Float> ops::Mul for &Variable<T> {
    type Output = Variable<T>;
    fn mul(self, rhs: Self) -> Variable<T> {
        mul(self, rhs)
    }
}

impl<T: Float> ops::Mul for Variable<T> {
    type Output = Variable<T>;
    fn mul(self, rhs: Self) -> Variable<T> {
        mul(&self, &rhs)
    }
}

impl<T: Float> ops::Neg for &Variable<T> {
    type Output = Variable<T>;
    fn neg(self) -> Variable<T> {
        neg(self)
    }
}

impl<T: Float> ops::Neg for Variable<T> {
    type Output = Variable<T>;
    fn neg(self) -> Variable<T> {
        neg(&self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn var(vals: &[f64]) -> Variable<f64> {
        let t = Tensor::from_vec(vals.to_vec(), vec![vals.len()]).unwrap();
        Variable::new(t, true)
    }

    #[test]
    fn test_add_backward() {
        let a = var(&[2.0, 3.0]);
        let b = var(&[4.0, 5.0]);
        let c = add(&a, &b);
        let s = sum(&c);
        s.backward();
        // dc/da = 1, dc/db = 1, ds/dc = 1 => each grad = [1,1]
        assert_eq!(a.grad().unwrap().as_slice(), &[1.0, 1.0]);
        assert_eq!(b.grad().unwrap().as_slice(), &[1.0, 1.0]);
    }

    #[test]
    fn test_mul_backward() {
        let a = var(&[2.0, 3.0]);
        let b = var(&[4.0, 5.0]);
        let c = mul(&a, &b);
        let s = sum(&c);
        s.backward();
        // dc/da = b, dc/db = a
        assert_eq!(a.grad().unwrap().as_slice(), &[4.0, 5.0]);
        assert_eq!(b.grad().unwrap().as_slice(), &[2.0, 3.0]);
    }

    #[test]
    fn test_sub_backward() {
        let a = var(&[5.0]);
        let b = var(&[3.0]);
        let c = sub(&a, &b);
        let s = sum(&c);
        s.backward();
        assert_eq!(a.grad().unwrap().as_slice(), &[1.0]);
        assert_eq!(b.grad().unwrap().as_slice(), &[-1.0]);
    }

    #[test]
    fn test_matmul_backward() {
        // a: [2,3], b: [3,2]
        let a = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap(),
            true,
        );
        let b = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]).unwrap(),
            true,
        );
        let c = matmul(&a, &b);
        let s = sum(&c);
        s.backward();
        // grad_a should be [2,3], grad_b should be [3,2]
        assert_eq!(a.grad().unwrap().shape(), &[2, 3]);
        assert_eq!(b.grad().unwrap().shape(), &[3, 2]);
    }

    #[test]
    fn test_pow_backward() {
        let a = var(&[2.0, 3.0]);
        let c = pow(&a, 2.0);
        let s = sum(&c);
        s.backward();
        // d/da (a^2) = 2a
        assert_eq!(a.grad().unwrap().as_slice(), &[4.0, 6.0]);
    }

    #[test]
    fn test_mean_backward() {
        let a = var(&[2.0, 4.0]);
        let m = mean(&a);
        m.backward();
        // d/da mean = 1/n = 0.5
        assert_eq!(a.grad().unwrap().as_slice(), &[0.5, 0.5]);
    }

    #[test]
    fn test_neg_backward() {
        let a = var(&[3.0]);
        let c = neg(&a);
        let s = sum(&c);
        s.backward();
        assert_eq!(a.grad().unwrap().as_slice(), &[-1.0]);
    }

    #[test]
    fn test_operator_overloads() {
        let a = var(&[1.0, 2.0]);
        let b = var(&[3.0, 4.0]);
        let c = &a + &b;
        let d = &a * &b;
        let s = sum(&(&c + &d));
        s.backward();
        // c = a+b, d = a*b, s = sum(c+d) = sum(a+b+a*b)
        // ds/da = 1 + b = [4, 5]
        // ds/db = 1 + a = [2, 3]
        assert_eq!(a.grad().unwrap().as_slice(), &[4.0, 5.0]);
        assert_eq!(b.grad().unwrap().as_slice(), &[2.0, 3.0]);
    }

    #[test]
    fn test_scalar_mul_backward() {
        let a = var(&[2.0, 3.0]);
        let c = scalar_mul(&a, 5.0);
        let s = sum(&c);
        s.backward();
        // d/da (5*a) = 5
        assert_eq!(a.grad().unwrap().as_slice(), &[5.0, 5.0]);
    }

    #[test]
    fn test_scalar_div_backward() {
        let a = var(&[4.0, 8.0]);
        let c = scalar_div(&a, 2.0);
        let s = sum(&c);
        s.backward();
        // d/da (a/2) = 0.5
        assert_eq!(a.grad().unwrap().as_slice(), &[0.5, 0.5]);
    }

    #[test]
    fn test_add_bias_forward_and_backward() {
        // input: [2, 3], bias: [3]
        let input = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap(),
            true,
        );
        let bias = Variable::new(
            Tensor::from_vec(vec![0.1, 0.2, 0.3], vec![3]).unwrap(),
            true,
        );
        let y = add_bias(&input, &bias);
        // Check forward values
        let y_data = y.data();
        let y_s = y_data.as_slice();
        assert!((y_s[0] - 1.1).abs() < 1e-10);
        assert!((y_s[4] - 5.2).abs() < 1e-10);

        let s = sum(&y);
        s.backward();
        // grad_input = ones (same shape as input)
        let g_input = input.grad().unwrap();
        assert_eq!(g_input.shape(), &[2, 3]);
        for &v in g_input.as_slice() {
            assert!((v - 1.0).abs() < 1e-10);
        }
        // grad_bias = sum over rows = [2.0, 2.0, 2.0] (2 rows of ones)
        let g_bias = bias.grad().unwrap();
        assert_eq!(g_bias.shape(), &[3]);
        for &v in g_bias.as_slice() {
            assert!((v - 2.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_single_element_sum() {
        let a = var(&[7.0]);
        let s = sum(&a);
        assert_eq!(s.data().as_slice(), &[7.0]);
        s.backward();
        assert_eq!(a.grad().unwrap().as_slice(), &[1.0]);
    }

    #[test]
    fn test_pow_cubic_backward() {
        let a = var(&[2.0]);
        let c = pow(&a, 3.0);
        let s = sum(&c);
        s.backward();
        // d/da (a^3) = 3*a^2 = 3*4 = 12
        assert!((a.grad().unwrap().as_slice()[0] - 12.0).abs() < 1e-10);
    }
}
