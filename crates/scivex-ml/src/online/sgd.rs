//! Stochastic gradient descent models for online learning.

use scivex_core::{Float, Tensor};

use crate::error::{MlError, Result};
use crate::online::IncrementalPredictor;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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

fn sigmoid<T: Float>(z: T) -> T {
    T::one() / (T::one() + (T::zero() - z).exp())
}

// ---------------------------------------------------------------------------
// SGDRegressor
// ---------------------------------------------------------------------------

/// Online linear regression trained with mini-batch SGD.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct SGDRegressor<T: Float> {
    learning_rate: T,
    l2_penalty: T,
    weights: Option<Vec<T>>,
    bias: T,
    n_features: usize,
    n_samples_seen: usize,
}

impl<T: Float> SGDRegressor<T> {
    /// Create a new `SGDRegressor`.
    ///
    /// # Errors
    ///
    /// Returns [`MlError::InvalidParameter`] if `learning_rate` is not positive.
    pub fn new(learning_rate: f64) -> Result<Self> {
        if learning_rate <= 0.0 {
            return Err(MlError::InvalidParameter {
                name: "learning_rate",
                reason: "must be positive",
            });
        }
        Ok(Self {
            learning_rate: T::from_f64(learning_rate),
            l2_penalty: T::zero(),
            weights: None,
            bias: T::zero(),
            n_features: 0,
            n_samples_seen: 0,
        })
    }

    /// Set the L2 regularisation coefficient (weight decay).
    pub fn set_l2_penalty(&mut self, alpha: f64) -> &mut Self {
        self.l2_penalty = T::from_f64(alpha);
        self
    }

    fn init_weights(&mut self, n_features: usize) {
        self.weights = Some(vec![T::zero(); n_features]);
        self.bias = T::zero();
        self.n_features = n_features;
    }
}

impl<T: Float> IncrementalPredictor<T> for SGDRegressor<T> {
    fn partial_fit(&mut self, x: &Tensor<T>, y: &Tensor<T>) -> Result<()> {
        let (n, p) = matrix_shape(x)?;
        let y_slice = y.as_slice();
        if y_slice.len() != n {
            return Err(MlError::DimensionMismatch {
                expected: n,
                got: y_slice.len(),
            });
        }

        // Lazy initialisation on first call.
        if self.weights.is_none() {
            self.init_weights(p);
        }
        if p != self.n_features {
            return Err(MlError::DimensionMismatch {
                expected: self.n_features,
                got: p,
            });
        }

        let x_data = x.as_slice();
        let lr = self.learning_rate;
        let l2 = self.l2_penalty;
        let w = self.weights.as_mut().expect("weights initialised above");

        for i in 0..n {
            let row = &x_data[i * p..(i + 1) * p];
            // prediction = w . x + bias
            let mut pred = self.bias;
            for j in 0..p {
                pred += w[j] * row[j];
            }
            let error = pred - y_slice[i];
            // update weights: w -= lr * (error * x + l2 * w)
            for j in 0..p {
                let grad = lr * (error * row[j] + l2 * w[j]);
                w[j] -= grad;
            }
            self.bias -= lr * error;
        }

        self.n_samples_seen += n;
        Ok(())
    }

    fn predict(&self, x: &Tensor<T>) -> Result<Tensor<T>> {
        let w = self.weights.as_ref().ok_or(MlError::NotFitted)?;
        let (n, p) = matrix_shape(x)?;
        if p != self.n_features {
            return Err(MlError::DimensionMismatch {
                expected: self.n_features,
                got: p,
            });
        }
        let x_data = x.as_slice();
        let mut preds = vec![T::zero(); n];
        for i in 0..n {
            let row = &x_data[i * p..(i + 1) * p];
            let mut pred = self.bias;
            for j in 0..p {
                pred += w[j] * row[j];
            }
            preds[i] = pred;
        }
        Ok(Tensor::from_vec(preds, vec![n])?)
    }

    fn n_samples_seen(&self) -> usize {
        self.n_samples_seen
    }
}

// ---------------------------------------------------------------------------
// SGDClassifier
// ---------------------------------------------------------------------------

/// Online binary classifier trained with logistic-loss SGD.
///
/// Outputs class labels `0.0` or `1.0`.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct SGDClassifier<T: Float> {
    learning_rate: T,
    l2_penalty: T,
    weights: Option<Vec<T>>,
    bias: T,
    n_features: usize,
    n_samples_seen: usize,
}

impl<T: Float> SGDClassifier<T> {
    /// Create a new `SGDClassifier`.
    ///
    /// # Errors
    ///
    /// Returns [`MlError::InvalidParameter`] if `learning_rate` is not positive.
    pub fn new(learning_rate: f64) -> Result<Self> {
        if learning_rate <= 0.0 {
            return Err(MlError::InvalidParameter {
                name: "learning_rate",
                reason: "must be positive",
            });
        }
        Ok(Self {
            learning_rate: T::from_f64(learning_rate),
            l2_penalty: T::zero(),
            weights: None,
            bias: T::zero(),
            n_features: 0,
            n_samples_seen: 0,
        })
    }

    /// Set the L2 regularisation coefficient (weight decay).
    pub fn set_l2_penalty(&mut self, alpha: f64) -> &mut Self {
        self.l2_penalty = T::from_f64(alpha);
        self
    }

    fn init_weights(&mut self, n_features: usize) {
        self.weights = Some(vec![T::zero(); n_features]);
        self.bias = T::zero();
        self.n_features = n_features;
    }
}

impl<T: Float> IncrementalPredictor<T> for SGDClassifier<T> {
    fn partial_fit(&mut self, x: &Tensor<T>, y: &Tensor<T>) -> Result<()> {
        let (n, p) = matrix_shape(x)?;
        let y_slice = y.as_slice();
        if y_slice.len() != n {
            return Err(MlError::DimensionMismatch {
                expected: n,
                got: y_slice.len(),
            });
        }

        if self.weights.is_none() {
            self.init_weights(p);
        }
        if p != self.n_features {
            return Err(MlError::DimensionMismatch {
                expected: self.n_features,
                got: p,
            });
        }

        let x_data = x.as_slice();
        let lr = self.learning_rate;
        let l2 = self.l2_penalty;
        let w = self.weights.as_mut().expect("weights initialised above");

        for i in 0..n {
            let row = &x_data[i * p..(i + 1) * p];
            let mut z = self.bias;
            for j in 0..p {
                z += w[j] * row[j];
            }
            let prob = sigmoid(z);
            // gradient of logistic loss: (prob - y)
            let error = prob - y_slice[i];
            for j in 0..p {
                let grad = lr * (error * row[j] + l2 * w[j]);
                w[j] -= grad;
            }
            self.bias -= lr * error;
        }

        self.n_samples_seen += n;
        Ok(())
    }

    fn predict(&self, x: &Tensor<T>) -> Result<Tensor<T>> {
        let w = self.weights.as_ref().ok_or(MlError::NotFitted)?;
        let (n, p) = matrix_shape(x)?;
        if p != self.n_features {
            return Err(MlError::DimensionMismatch {
                expected: self.n_features,
                got: p,
            });
        }
        let x_data = x.as_slice();
        let half = T::from_f64(0.5);
        let mut labels = vec![T::zero(); n];
        for i in 0..n {
            let row = &x_data[i * p..(i + 1) * p];
            let mut z = self.bias;
            for j in 0..p {
                z += w[j] * row[j];
            }
            let prob = sigmoid(z);
            labels[i] = if prob >= half { T::one() } else { T::zero() };
        }
        Ok(Tensor::from_vec(labels, vec![n])?)
    }

    fn n_samples_seen(&self) -> usize {
        self.n_samples_seen
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sgd_regressor_linear() {
        // y = 2*x + 1
        let mut reg = SGDRegressor::<f64>::new(0.01).unwrap();
        // Train many small batches so it converges.
        for _ in 0..200 {
            let x = Tensor::from_vec(
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                vec![10, 1],
            )
            .unwrap();
            let y = Tensor::from_vec(
                vec![3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0],
                vec![10],
            )
            .unwrap();
            reg.partial_fit(&x, &y).unwrap();
        }
        // Predict x = 20 -> expected ~41
        let x_test = Tensor::from_vec(vec![20.0], vec![1, 1]).unwrap();
        let pred = reg.predict(&x_test).unwrap();
        assert!(
            (pred.as_slice()[0] - 41.0).abs() < 1.0,
            "predicted {} but expected ~41",
            pred.as_slice()[0]
        );
    }

    #[test]
    fn test_sgd_regressor_not_fitted() {
        let reg = SGDRegressor::<f64>::new(0.01).unwrap();
        let x = Tensor::from_vec(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let result = reg.predict(&x);
        assert!(result.is_err());
    }

    #[test]
    fn test_sgd_classifier_separable() {
        // Class 0: x[0] < 0, Class 1: x[0] > 0
        let mut clf = SGDClassifier::<f64>::new(0.1).unwrap();
        for _ in 0..100 {
            let x = Tensor::from_vec(vec![-3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0], vec![8, 1])
                .unwrap();
            let y =
                Tensor::from_vec(vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], vec![8]).unwrap();
            clf.partial_fit(&x, &y).unwrap();
        }
        let x_test = Tensor::from_vec(vec![-5.0, 5.0], vec![2, 1]).unwrap();
        let pred = clf.predict(&x_test).unwrap();
        let labels = pred.as_slice();
        assert!(
            (labels[0] - 0.0).abs() < 1e-12,
            "expected 0.0 for x=-5, got {}",
            labels[0]
        );
        assert!(
            (labels[1] - 1.0).abs() < 1e-12,
            "expected 1.0 for x=5, got {}",
            labels[1]
        );
    }

    #[test]
    fn test_sgd_classifier_predict_labels() {
        let mut clf = SGDClassifier::<f64>::new(0.1).unwrap();
        let x = Tensor::from_vec(vec![-1.0, 1.0], vec![2, 1]).unwrap();
        let y = Tensor::from_vec(vec![0.0, 1.0], vec![2]).unwrap();
        for _ in 0..200 {
            clf.partial_fit(&x, &y).unwrap();
        }
        let pred = clf.predict(&x).unwrap();
        for &v in pred.as_slice() {
            assert!(
                (v - 0.0).abs() < 1e-12 || (v - 1.0).abs() < 1e-12,
                "label must be 0.0 or 1.0, got {v}"
            );
        }
    }

    #[test]
    fn test_n_samples_seen() {
        let mut reg = SGDRegressor::<f64>::new(0.01).unwrap();
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let y = Tensor::from_vec(vec![2.0, 4.0, 6.0], vec![3]).unwrap();
        reg.partial_fit(&x, &y).unwrap();
        assert_eq!(reg.n_samples_seen(), 3);
        reg.partial_fit(&x, &y).unwrap();
        assert_eq!(reg.n_samples_seen(), 6);
    }
}
