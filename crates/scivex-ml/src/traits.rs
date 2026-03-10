use scivex_core::{Float, Tensor};

use crate::Result;

/// An unsupervised transformation: fit on data, then transform new data.
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
pub trait Predictor<T: Float> {
    /// Learn from features `x` `[n_samples, n_features]` and targets `y` `[n_samples]`.
    fn fit(&mut self, x: &Tensor<T>, y: &Tensor<T>) -> Result<()>;

    /// Predict targets for new features `x`.
    fn predict(&self, x: &Tensor<T>) -> Result<Tensor<T>>;
}

/// A classifier that can also produce class probabilities.
pub trait Classifier<T: Float>: Predictor<T> {
    /// Return class probabilities, shape `[n_samples, n_classes]`.
    fn predict_proba(&self, x: &Tensor<T>) -> Result<Tensor<T>>;
}
