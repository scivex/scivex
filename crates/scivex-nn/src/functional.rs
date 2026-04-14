//! Differentiable activation functions.

use scivex_core::{Float, Tensor};

use crate::error::{NnError, Result};
use crate::ops;
use crate::variable::Variable;

/// ReLU activation: `max(0, x)`.
///
/// # Examples
///
/// ```
/// # use scivex_core::Tensor;
/// # use scivex_nn::variable::Variable;
/// # use scivex_nn::functional::relu;
/// let x = Variable::new(Tensor::from_vec(vec![-1.0_f64, 0.0, 2.0], vec![3]).unwrap(), false);
/// let y = relu(&x);
/// assert_eq!(y.data().as_slice(), &[0.0, 0.0, 2.0]);
/// ```
pub fn relu<T: Float>(x: &Variable<T>) -> Variable<T> {
    let x_data = x.data();
    let out = x_data.relu();
    Variable::from_op(
        out,
        vec![x.clone()],
        Box::new(move |g: &Tensor<T>| {
            let mask = x_data.map(|v| if v > T::zero() { T::one() } else { T::zero() });
            vec![g * &mask]
        }),
    )
}

/// Sigmoid activation: `1 / (1 + exp(-x))`.
///
/// # Examples
///
/// ```
/// # use scivex_core::Tensor;
/// # use scivex_nn::variable::Variable;
/// # use scivex_nn::functional::sigmoid;
/// let x = Variable::new(Tensor::from_vec(vec![0.0_f64], vec![1]).unwrap(), false);
/// let y = sigmoid(&x);
/// assert!((y.data().as_slice()[0] - 0.5).abs() < 1e-10);
/// ```
pub fn sigmoid<T: Float>(x: &Variable<T>) -> Variable<T> {
    let x_data = x.data();
    let sig = x_data.map(|v| T::one() / (T::one() + (-v).exp()));
    let sig_clone = sig.clone();
    Variable::from_op(
        sig,
        vec![x.clone()],
        Box::new(move |g: &Tensor<T>| {
            // grad = g * s * (1 - s)
            let deriv = sig_clone.map(|s| s * (T::one() - s));
            vec![g * &deriv]
        }),
    )
}

/// Tanh activation: `(exp(x) - exp(-x)) / (exp(x) + exp(-x))`.
///
/// The Float trait has no `tanh`, so we compute it manually.
///
/// # Examples
///
/// ```
/// # use scivex_core::Tensor;
/// # use scivex_nn::variable::Variable;
/// # use scivex_nn::functional::tanh_fn;
/// let x = Variable::new(Tensor::from_vec(vec![0.0_f64], vec![1]).unwrap(), false);
/// let y = tanh_fn(&x);
/// assert!(y.data().as_slice()[0].abs() < 1e-10); // tanh(0) = 0
/// ```
pub fn tanh_fn<T: Float>(x: &Variable<T>) -> Variable<T> {
    let x_data = x.data();
    let out = x_data.map(|v| {
        let ep = v.exp();
        let en = (-v).exp();
        (ep - en) / (ep + en)
    });
    let out_clone = out.clone();
    Variable::from_op(
        out,
        vec![x.clone()],
        Box::new(move |g: &Tensor<T>| {
            // grad = g * (1 - tanh^2)
            let deriv = out_clone.map(|t| T::one() - t * t);
            vec![g * &deriv]
        }),
    )
}

/// Softmax along the last axis of a 2-D variable `[batch, classes]`.
///
/// For numerical stability, subtracts the row-max before exponentiating.
///
/// # Examples
///
/// ```
/// # use scivex_core::Tensor;
/// # use scivex_nn::variable::Variable;
/// # use scivex_nn::functional::softmax;
/// let x = Variable::new(Tensor::from_vec(vec![1.0_f64, 2.0, 3.0], vec![1, 3]).unwrap(), false);
/// let y = softmax(&x).unwrap();
/// let sum: f64 = y.data().as_slice().iter().sum();
/// assert!((sum - 1.0).abs() < 1e-10); // probabilities sum to 1
/// ```
pub fn softmax<T: Float>(x: &Variable<T>) -> Result<Variable<T>> {
    let x_data = x.data();
    let shape = x_data.shape().to_vec();
    if shape.len() != 2 {
        return Err(NnError::ShapeMismatch {
            expected: vec![0, 0],
            got: shape,
        });
    }
    let rows = shape[0];
    let cols = shape[1];
    let x_slice = x_data.as_slice();

    let mut out_data = vec![T::zero(); rows * cols];
    for r in 0..rows {
        let row_start = r * cols;
        // Find row max for numerical stability.
        let mut max_val = x_slice[row_start];
        for c in 1..cols {
            let v = x_slice[row_start + c];
            if v > max_val {
                max_val = v;
            }
        }
        // exp(x - max)
        let mut sum_exp = T::zero();
        for c in 0..cols {
            let e = (x_slice[row_start + c] - max_val).exp();
            out_data[row_start + c] = e;
            sum_exp += e;
        }
        // Normalize.
        for c in 0..cols {
            out_data[row_start + c] /= sum_exp;
        }
    }

    let out = Tensor::from_vec(out_data, shape.clone())?;
    let out_clone = out.clone();

    Ok(Variable::from_op(
        out,
        vec![x.clone()],
        Box::new(move |g: &Tensor<T>| {
            // Jacobian-vector product: grad_i = s_i * (g_i - sum_j(g_j * s_j))
            let s = out_clone.as_slice();
            let g_slice = g.as_slice();
            let mut result = vec![T::zero(); rows * cols];
            for r in 0..rows {
                let row_start = r * cols;
                // dot(g_row, s_row)
                let mut dot = T::zero();
                for c in 0..cols {
                    dot += g_slice[row_start + c] * s[row_start + c];
                }
                for c in 0..cols {
                    let idx = row_start + c;
                    result[idx] = s[idx] * (g_slice[idx] - dot);
                }
            }
            vec![Tensor::from_vec(result, shape.clone()).expect("grad shape matches forward pass")]
        }),
    ))
}

/// Log-softmax along the last axis: `log(softmax(x))`.
///
/// Computed as `x - max - log(sum(exp(x - max)))` for numerical stability.
///
/// # Examples
///
/// ```
/// # use scivex_core::Tensor;
/// # use scivex_nn::variable::Variable;
/// # use scivex_nn::functional::log_softmax;
/// let x = Variable::new(Tensor::from_vec(vec![1.0_f64, 2.0, 3.0], vec![1, 3]).unwrap(), false);
/// let y = log_softmax(&x).unwrap();
/// // All log-softmax values are negative.
/// for &v in y.data().as_slice() {
///     assert!(v < 0.0);
/// }
/// ```
pub fn log_softmax<T: Float>(x: &Variable<T>) -> Result<Variable<T>> {
    let x_data = x.data();
    let shape = x_data.shape().to_vec();
    if shape.len() != 2 {
        return Err(NnError::ShapeMismatch {
            expected: vec![0, 0],
            got: shape,
        });
    }
    let rows = shape[0];
    let cols = shape[1];
    let x_slice = x_data.as_slice();

    let mut out_data = vec![T::zero(); rows * cols];
    // Also store softmax for backward.
    let mut sm_data = vec![T::zero(); rows * cols];

    for r in 0..rows {
        let row_start = r * cols;
        let mut max_val = x_slice[row_start];
        for c in 1..cols {
            let v = x_slice[row_start + c];
            if v > max_val {
                max_val = v;
            }
        }
        let mut sum_exp = T::zero();
        for c in 0..cols {
            let e = (x_slice[row_start + c] - max_val).exp();
            sm_data[row_start + c] = e;
            sum_exp += e;
        }
        let log_sum = sum_exp.ln();
        for c in 0..cols {
            sm_data[row_start + c] /= sum_exp;
            out_data[row_start + c] = x_slice[row_start + c] - max_val - log_sum;
        }
    }

    let out = Tensor::from_vec(out_data, shape.clone())?;
    let sm = Tensor::from_vec(sm_data, shape.clone())?;

    Ok(Variable::from_op(
        out,
        vec![x.clone()],
        Box::new(move |g: &Tensor<T>| {
            // d log_softmax / dx = g - softmax * sum(g, axis=1)
            let g_slice = g.as_slice();
            let s_slice = sm.as_slice();
            let mut result = vec![T::zero(); rows * cols];
            for r in 0..rows {
                let row_start = r * cols;
                let mut g_sum = T::zero();
                for c in 0..cols {
                    g_sum += g_slice[row_start + c];
                }
                for c in 0..cols {
                    let idx = row_start + c;
                    result[idx] = g_slice[idx] - s_slice[idx] * g_sum;
                }
            }
            vec![Tensor::from_vec(result, shape.clone()).expect("grad shape matches forward pass")]
        }),
    ))
}

/// Element-wise exponential.
///
/// # Examples
///
/// ```
/// # use scivex_core::Tensor;
/// # use scivex_nn::variable::Variable;
/// # use scivex_nn::functional::exp;
/// let x = Variable::new(Tensor::from_vec(vec![0.0_f64, 1.0], vec![2]).unwrap(), false);
/// let y = exp(&x);
/// assert!((y.data().as_slice()[0] - 1.0).abs() < 1e-10); // exp(0) = 1
/// ```
pub fn exp<T: Float>(x: &Variable<T>) -> Variable<T> {
    let x_data = x.data();
    let out = x_data.exp();
    let out_clone = out.clone();
    Variable::from_op(
        out,
        vec![x.clone()],
        Box::new(move |g: &Tensor<T>| {
            vec![g * &out_clone]
        }),
    )
}

/// Element-wise natural logarithm.
///
/// # Examples
///
/// ```
/// # use scivex_core::Tensor;
/// # use scivex_nn::variable::Variable;
/// # use scivex_nn::functional::ln;
/// let x = Variable::new(Tensor::from_vec(vec![1.0_f64, 2.718281828], vec![2]).unwrap(), false);
/// let y = ln(&x);
/// assert!(y.data().as_slice()[0].abs() < 1e-10); // ln(1) = 0
/// ```
pub fn ln<T: Float>(x: &Variable<T>) -> Variable<T> {
    let x_data = x.data();
    let out = x_data.ln();
    Variable::from_op(
        out,
        vec![x.clone()],
        Box::new(move |g: &Tensor<T>| {
            vec![g / &x_data]
        }),
    )
}

/// Clamp (not differentiable at boundaries, gradient passes through in range).
///
/// # Examples
///
/// ```
/// # use scivex_core::Tensor;
/// # use scivex_nn::variable::Variable;
/// # use scivex_nn::functional::clamp;
/// let x = Variable::new(Tensor::from_vec(vec![-5.0_f64, 0.5, 10.0], vec![3]).unwrap(), false);
/// let y = clamp(&x, 0.0, 1.0);
/// assert_eq!(y.data().as_slice(), &[0.0, 0.5, 1.0]);
/// ```
pub fn clamp<T: Float>(x: &Variable<T>, min: T, max: T) -> Variable<T> {
    let x_data = x.data();
    let out = x_data.clamp(min, max);
    Variable::from_op(
        out,
        vec![x.clone()],
        Box::new(move |g: &Tensor<T>| {
            let mask = x_data.map(|v| {
                if v >= min && v <= max {
                    T::one()
                } else {
                    T::zero()
                }
            });
            vec![g * &mask]
        }),
    )
}

/// Convenience: `mean` re-exported from ops.
pub use ops::mean;
/// Convenience: `sum` re-exported from ops.
pub use ops::sum;

#[cfg(test)]
mod tests {
    use super::*;

    fn var1d(vals: &[f64]) -> Variable<f64> {
        Variable::new(
            Tensor::from_vec(vals.to_vec(), vec![vals.len()]).unwrap(),
            true,
        )
    }

    #[test]
    fn test_relu_forward() {
        let x = var1d(&[-1.0, 0.0, 1.0, 2.0]);
        let y = relu(&x);
        assert_eq!(y.data().as_slice(), &[0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_relu_backward() {
        let x = var1d(&[-1.0, 0.0, 2.0]);
        let y = relu(&x);
        let s = ops::sum(&y);
        s.backward();
        assert_eq!(x.grad().unwrap().as_slice(), &[0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_sigmoid_forward() {
        let x = var1d(&[0.0]);
        let y = sigmoid(&x);
        assert!((y.data().as_slice()[0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_tanh_forward() {
        let x = var1d(&[0.0]);
        let y = tanh_fn(&x);
        assert!(y.data().as_slice()[0].abs() < 1e-10);
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let x = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 1.0, 1.0, 1.0], vec![2, 3]).unwrap(),
            true,
        );
        let y = softmax(&x).unwrap();
        let y_data = y.data();
        let y_slice = y_data.as_slice();
        // Each row should sum to 1.
        let sum_row0: f64 = y_slice[..3].iter().sum();
        let sum_row1: f64 = y_slice[3..].iter().sum();
        assert!((sum_row0 - 1.0).abs() < 1e-10);
        assert!((sum_row1 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_log_softmax() {
        let x = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap(),
            true,
        );
        let y = log_softmax(&x).unwrap();
        // log_softmax values should all be negative.
        for &v in y.data().as_slice() {
            assert!(v < 0.0);
        }
        // exp(log_softmax) should sum to 1.
        let s: f64 = y.data().as_slice().iter().map(|v| v.exp()).sum();
        assert!((s - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sigmoid_backward() {
        let x = var1d(&[0.0]);
        let y = sigmoid(&x);
        let s = ops::sum(&y);
        s.backward();
        // sigmoid'(0) = sigmoid(0) * (1 - sigmoid(0)) = 0.25
        let g = x.grad().unwrap().as_slice()[0];
        assert!((g - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_tanh_backward() {
        let x = var1d(&[0.0]);
        let y = tanh_fn(&x);
        let s = ops::sum(&y);
        s.backward();
        // tanh'(0) = 1 - tanh(0)^2 = 1
        let g = x.grad().unwrap().as_slice()[0];
        assert!((g - 1.0).abs() < 1e-10);
    }
}
