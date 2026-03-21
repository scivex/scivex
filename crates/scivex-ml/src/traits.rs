use scivex_core::{Float, Tensor};

use crate::Result;

/// An unsupervised transformation: fit on data, then transform new data.
///
/// # Examples
///
/// ```
/// # use scivex_ml::prelude::*;
/// # use scivex_core::prelude::*;
/// let x = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
/// let mut scaler = StandardScaler::<f64>::new();
/// scaler.fit(&x).unwrap();
/// let transformed = scaler.transform(&x).unwrap();
/// ```
pub trait Transformer<T: Float> {
    /// Learn parameters from `x` (shape `[n_samples, n_features]`).
    fn fit(&mut self, x: &Tensor<T>) -> Result<()>;

    /// Apply the learned transformation to `x`.
    fn transform(&self, x: &Tensor<T>) -> Result<Tensor<T>>;

    /// Convenience: fit and transform in one step.
    fn fit_transform(&mut self, x: &Tensor<T>) -> Result<Tensor<T>> {
        self.fit(x)?;
        self.transform(x)
    }
}

/// A supervised learner: fit on (x, y) pairs, then predict.
///
/// # Examples
///
/// ```
/// # use scivex_ml::prelude::*;
/// # use scivex_core::prelude::*;
/// let x = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
/// let y = Tensor::from_vec(vec![3.0, 5.0, 7.0, 9.0], vec![4]).unwrap();
/// let mut model = LinearRegression::<f64>::new();
/// model.fit(&x, &y).unwrap();
/// let preds = model.predict(&x).unwrap();
/// ```
pub trait Predictor<T: Float> {
    /// Learn from features `x` `[n_samples, n_features]` and targets `y` `[n_samples]`.
    fn fit(&mut self, x: &Tensor<T>, y: &Tensor<T>) -> Result<()>;

    /// Predict targets for new features `x`.
    fn predict(&self, x: &Tensor<T>) -> Result<Tensor<T>>;
}

/// A classifier that can also produce class probabilities.
///
/// # Examples
///
/// ```
/// # use scivex_ml::prelude::*;
/// # use scivex_core::prelude::*;
/// let x = Tensor::from_vec(
///     vec![-2.0_f64, -1.0, -1.5, -0.5, 2.0, 1.0, 1.5, 0.5],
///     vec![4, 2],
/// ).unwrap();
/// let y = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0], vec![4]).unwrap();
/// let mut lr = LogisticRegression::new(0.5, 1000, 1e-8).unwrap();
/// lr.fit(&x, &y).unwrap();
/// let proba = lr.predict_proba(&x).unwrap();
/// ```
pub trait Classifier<T: Float>: Predictor<T> {
    /// Return class probabilities, shape `[n_samples, n_classes]`.
    fn predict_proba(&self, x: &Tensor<T>) -> Result<Tensor<T>>;
}
