use scivex_core::{Float, Tensor};

use crate::error::{MlError, Result};
use crate::traits::{Classifier, Predictor};

/// Binary logistic regression trained via gradient descent.
#[derive(Debug, Clone)]
pub struct LogisticRegression<T: Float> {
    learning_rate: T,
    max_iter: usize,
    tol: T,
    weights: Option<Vec<T>>,
    bias: Option<T>,
}

impl<T: Float> LogisticRegression<T> {
    /// Create a new logistic regression with gradient descent hyper-parameters.
    pub fn new(learning_rate: T, max_iter: usize, tol: T) -> Result<Self> {
        if learning_rate <= T::zero() {
            return Err(MlError::InvalidParameter {
                name: "learning_rate",
                reason: "must be positive",
            });
        }
        if max_iter == 0 {
            return Err(MlError::InvalidParameter {
                name: "max_iter",
                reason: "must be at least 1",
            });
        }
        Ok(Self {
            learning_rate,
            max_iter,
            tol,
            weights: None,
            bias: None,
        })
    }

    /// Fitted coefficients (excluding bias).
    pub fn weights(&self) -> Option<&[T]> {
        self.weights.as_deref()
    }

    /// Fitted intercept.
    pub fn bias(&self) -> Option<T> {
        self.bias
    }
}

impl<T: Float> Predictor<T> for LogisticRegression<T> {
    fn fit(&mut self, x: &Tensor<T>, y: &Tensor<T>) -> Result<()> {
        let (n, p) = matrix_shape(x)?;
        check_y(y, n)?;
        let x_data = x.as_slice();
        let y_data = y.as_slice();
        let nf = T::from_usize(n);

        let mut w = vec![T::zero(); p];
        let mut b = T::zero();

        for _iter in 0..self.max_iter {
            let mut dw = vec![T::zero(); p];
            let mut db = T::zero();
            let mut max_grad = T::zero();

            for i in 0..n {
                let z = dot_row(x_data, &w, i, p) + b;
                let pred = sigmoid(z);
                let err = pred - y_data[i];
                for j in 0..p {
                    dw[j] += err * x_data[i * p + j];
                }
                db += err;
            }

            for j in 0..p {
                dw[j] /= nf;
                let step = self.learning_rate * dw[j];
                if step.abs() > max_grad {
                    max_grad = step.abs();
                }
                w[j] -= step;
            }
            db /= nf;
            b -= self.learning_rate * db;

            if max_grad < self.tol {
                break;
            }
        }

        self.weights = Some(w);
        self.bias = Some(b);
        Ok(())
    }

    fn predict(&self, x: &Tensor<T>) -> Result<Tensor<T>> {
        let proba = self.predict_proba(x)?;
        let half = T::from_usize(1) / T::from_usize(2);
        let data: Vec<T> = proba
            .as_slice()
            .iter()
            .map(|&p| if p >= half { T::one() } else { T::zero() })
            .collect();
        let n = data.len();
        Ok(Tensor::from_vec(data, vec![n])?)
    }
}

impl<T: Float> Classifier<T> for LogisticRegression<T> {
    fn predict_proba(&self, x: &Tensor<T>) -> Result<Tensor<T>> {
        let w = self.weights.as_ref().ok_or(MlError::NotFitted)?;
        let b = self.bias.ok_or(MlError::NotFitted)?;
        let (n, p) = matrix_shape(x)?;
        if p != w.len() {
            return Err(MlError::DimensionMismatch {
                expected: w.len(),
                got: p,
            });
        }
        let x_data = x.as_slice();
        let out: Vec<T> = (0..n)
            .map(|i| sigmoid(dot_row(x_data, w, i, p) + b))
            .collect();
        Ok(Tensor::from_vec(out, vec![n])?)
    }
}

fn sigmoid<T: Float>(z: T) -> T {
    T::one() / (T::one() + (T::zero() - z).exp())
}

fn dot_row<T: Float>(data: &[T], w: &[T], row: usize, p: usize) -> T {
    let mut sum = T::zero();
    for j in 0..p {
        sum += data[row * p + j] * w[j];
    }
    sum
}

fn matrix_shape<T: Float>(x: &Tensor<T>) -> Result<(usize, usize)> {
    let s = x.shape();
    if s.len() != 2 {
        return Err(MlError::DimensionMismatch {
            expected: 2,
            got: s.len(),
        });
    }
    if s[0] == 0 {
        return Err(MlError::EmptyInput);
    }
    Ok((s[0], s[1]))
}

fn check_y<T: Float>(y: &Tensor<T>, n: usize) -> Result<()> {
    if y.ndim() != 1 || y.shape()[0] != n {
        return Err(MlError::DimensionMismatch {
            expected: n,
            got: y.shape()[0],
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logistic_separable() {
        // Two linearly separable clusters
        let x = Tensor::from_vec(
            vec![
                -2.0_f64, -1.0, -1.5, -0.5, -1.0, -1.0, 2.0, 1.0, 1.5, 0.5, 1.0, 1.0,
            ],
            vec![6, 2],
        )
        .unwrap();
        let y = Tensor::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0], vec![6]).unwrap();

        let mut lr = LogisticRegression::new(0.5, 1000, 1e-8).unwrap();
        lr.fit(&x, &y).unwrap();

        let preds = lr.predict(&x).unwrap();
        let pred_slice = preds.as_slice();
        // First three should be 0, last three should be 1
        for &p in &pred_slice[..3] {
            assert!((p - 0.0).abs() < 1e-6, "expected 0, got {p}");
        }
        for &p in &pred_slice[3..] {
            assert!((p - 1.0).abs() < 1e-6, "expected 1, got {p}");
        }
    }

    #[test]
    fn test_predict_proba_range() {
        let x = Tensor::from_vec(vec![-5.0_f64, 0.0, 5.0], vec![3, 1]).unwrap();
        let y = Tensor::from_vec(vec![0.0, 0.0, 1.0], vec![3]).unwrap();
        let mut lr = LogisticRegression::new(0.1, 500, 1e-8).unwrap();
        lr.fit(&x, &y).unwrap();
        let proba = lr.predict_proba(&x).unwrap();
        for &p in proba.as_slice() {
            assert!((0.0..=1.0).contains(&p));
        }
    }
}
