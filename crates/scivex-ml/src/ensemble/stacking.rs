//! Stacking (stacked generalization) ensemble methods.
//!
//! A stacking ensemble trains multiple base estimators and combines their
//! predictions via a meta-learner. Out-of-fold (OOF) predictions are used
//! to train the meta-learner, avoiding data leakage.

use scivex_core::random::Rng;
use scivex_core::{Float, Tensor};

use crate::error::{MlError, Result};
use crate::model_selection::KFold;
use crate::traits::Predictor;

/// A factory closure that produces a fresh boxed predictor.
pub type EstimatorFactory<T> = Box<dyn Fn() -> Box<dyn Predictor<T>>>;

// ======================================================================
// Helpers (crate-local pattern: each file defines its own)
// ======================================================================

fn matrix_shape<T: Float>(x: &Tensor<T>) -> Result<(usize, usize)> {
    let s = x.shape();
    if s.len() != 2 {
        return Err(MlError::DimensionMismatch {
            expected: 2,
            got: s.len(),
        });
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

/// Subset rows from a 2-D tensor by index.
fn subset_rows<T: Float>(x: &Tensor<T>, indices: &[usize], n_features: usize) -> Tensor<T> {
    let data = x.as_slice();
    let mut out = Vec::with_capacity(indices.len() * n_features);
    for &i in indices {
        out.extend_from_slice(&data[i * n_features..(i + 1) * n_features]);
    }
    Tensor::from_vec(out, vec![indices.len(), n_features]).unwrap()
}

/// Subset elements from a 1-D tensor by index.
fn subset_vec<T: Float>(y: &Tensor<T>, indices: &[usize]) -> Tensor<T> {
    let data = y.as_slice();
    let out: Vec<T> = indices.iter().map(|&i| data[i]).collect();
    Tensor::from_vec(out, vec![indices.len()]).unwrap()
}

// ======================================================================
// StackingRegressor
// ======================================================================

/// Stacking regressor that trains multiple base regressors and combines their
/// predictions via a meta-learner trained on out-of-fold predictions.
///
/// # Algorithm
///
/// 1. **Generate meta-features** using k-fold cross-validation:
///    - For each fold, train each base estimator on the training fold
///    - Predict on the held-out fold to generate OOF predictions
/// 2. **Stack** all OOF predictions into a meta-feature matrix `[n_samples, n_estimators]`
/// 3. **Fit** the meta-learner on the meta-feature matrix and the original targets
/// 4. **Re-fit** all base estimators on the full training data for prediction
///
/// # Examples
///
/// ```
/// # use scivex_ml::prelude::*;
/// # use scivex_core::prelude::*;
/// let x = Tensor::from_vec(
///     vec![1.0_f64, 10.0, 2.0, 8.0, 3.0, 6.0, 4.0, 9.0,
///          5.0, 3.0, 6.0, 7.0, 7.0, 2.0, 8.0, 5.0,
///          9.0, 4.0, 10.0, 1.0, 11.0, 11.0, 12.0, 0.5],
///     vec![12, 2],
/// ).unwrap();
/// let y = Tensor::from_vec(
///     vec![11.0, 10.0, 9.0, 13.0, 8.0, 13.0, 9.0, 13.0, 13.0, 11.0, 22.0, 12.5],
///     vec![12],
/// ).unwrap();
///
/// let base_factories: Vec<EstimatorFactory<f64>> = vec![
///     Box::new(|| Box::new(LinearRegression::<f64>::new())),
///     Box::new(|| Box::new(Ridge::new(1.0).unwrap())),
/// ];
/// let meta = Box::new(LinearRegression::<f64>::new());
/// let mut stacker = StackingRegressor::new(base_factories, meta, 3, 42).unwrap();
/// stacker.fit(&x, &y).unwrap();
/// let preds = stacker.predict(&x).unwrap();
/// assert_eq!(preds.shape(), &[12]);
/// ```
pub struct StackingRegressor<T: Float> {
    /// Factory closures that produce fresh base estimators.
    base_factories: Vec<EstimatorFactory<T>>,
    /// The meta-learner trained on stacked OOF predictions.
    meta_learner: Box<dyn Predictor<T>>,
    /// Number of cross-validation folds for OOF generation.
    cv_folds: usize,
    /// Random seed for KFold shuffling.
    seed: u64,
    /// Fitted base estimators (trained on full data for prediction).
    fitted_base: Option<Vec<Box<dyn Predictor<T>>>>,
}

impl<T: Float> StackingRegressor<T> {
    /// Create a new stacking regressor.
    ///
    /// # Arguments
    /// - `base_factories`: Closures that create fresh base estimators
    /// - `meta_learner`: The meta-learner used to combine base predictions
    /// - `cv_folds`: Number of cross-validation folds (>= 2)
    /// - `seed`: Random seed for fold shuffling
    pub fn new(
        base_factories: Vec<EstimatorFactory<T>>,
        meta_learner: Box<dyn Predictor<T>>,
        cv_folds: usize,
        seed: u64,
    ) -> Result<Self> {
        if base_factories.is_empty() {
            return Err(MlError::InvalidParameter {
                name: "base_factories",
                reason: "must provide at least one base estimator",
            });
        }
        if cv_folds < 2 {
            return Err(MlError::InvalidParameter {
                name: "cv_folds",
                reason: "must be at least 2",
            });
        }
        Ok(Self {
            base_factories,
            meta_learner,
            cv_folds,
            seed,
            fitted_base: None,
        })
    }
}

impl<T: Float> Predictor<T> for StackingRegressor<T> {
    fn fit(&mut self, x: &Tensor<T>, y: &Tensor<T>) -> Result<()> {
        let (n, p) = matrix_shape(x)?;
        check_y(y, n)?;

        let n_base = self.base_factories.len();
        let mut rng = Rng::new(self.seed);
        let kfold = KFold::new(self.cv_folds, n, &mut rng)?;

        // Generate out-of-fold predictions for each base estimator.
        // oof_preds[i][sample_idx] = prediction from estimator i for sample sample_idx
        let mut oof_preds = vec![vec![T::zero(); n]; n_base];

        for (train_idx, test_idx) in &kfold {
            let x_train = subset_rows(x, &train_idx, p);
            let y_train = subset_vec(y, &train_idx);
            let x_test = subset_rows(x, &test_idx, p);

            for (est_i, factory) in self.base_factories.iter().enumerate() {
                let mut estimator = factory();
                estimator.fit(&x_train, &y_train)?;
                let preds = estimator.predict(&x_test)?;
                let pred_data = preds.as_slice();
                for (j, &idx) in test_idx.iter().enumerate() {
                    oof_preds[est_i][idx] = pred_data[j];
                }
            }
        }

        // Build meta-feature matrix: [n_samples, n_base_estimators]
        let mut meta_data = Vec::with_capacity(n * n_base);
        for i in 0..n {
            for est_preds in &oof_preds {
                meta_data.push(est_preds[i]);
            }
        }
        let meta_x = Tensor::from_vec(meta_data, vec![n, n_base])?;

        // Fit the meta-learner on meta-features
        self.meta_learner.fit(&meta_x, y)?;

        // Re-fit all base estimators on the full training data
        let mut fitted = Vec::with_capacity(n_base);
        for factory in &self.base_factories {
            let mut est = factory();
            est.fit(x, y)?;
            fitted.push(est);
        }
        self.fitted_base = Some(fitted);

        Ok(())
    }

    fn predict(&self, x: &Tensor<T>) -> Result<Tensor<T>> {
        let fitted = self.fitted_base.as_ref().ok_or(MlError::NotFitted)?;
        let (n, _p) = matrix_shape(x)?;
        let n_base = fitted.len();

        // Get predictions from all fitted base estimators
        let mut meta_data = Vec::with_capacity(n * n_base);
        let base_preds: Vec<Tensor<T>> = fitted
            .iter()
            .map(|est| est.predict(x))
            .collect::<Result<Vec<_>>>()?;

        for i in 0..n {
            for pred_tensor in &base_preds {
                meta_data.push(pred_tensor.as_slice()[i]);
            }
        }
        let meta_x = Tensor::from_vec(meta_data, vec![n, n_base])?;

        self.meta_learner.predict(&meta_x)
    }
}

// ======================================================================
// StackingClassifier
// ======================================================================

/// Stacking classifier that trains multiple base classifiers and combines
/// their predictions via a meta-learner trained on out-of-fold predictions.
///
/// By default, uses the base estimators' `predict` outputs as meta-features.
/// Set `use_proba(true)` to use `predict_proba` outputs instead (requires
/// base estimators to implement `Classifier`).
///
/// # Examples
///
/// ```
/// # use scivex_ml::prelude::*;
/// # use scivex_core::prelude::*;
/// let x = Tensor::from_vec(
///     vec![-3.0_f64, -2.0, -2.0, -1.0, -1.0, -0.5,
///          1.0, 0.5, 2.0, 1.0, 3.0, 2.0],
///     vec![6, 2],
/// ).unwrap();
/// let y = Tensor::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0], vec![6]).unwrap();
///
/// let base_factories: Vec<EstimatorFactory<f64>> = vec![
///     Box::new(|| Box::new(LogisticRegression::new(0.1, 100, 1e-6).unwrap())),
/// ];
/// let meta: Box<dyn Predictor<f64>> =
///     Box::new(LogisticRegression::new(0.1, 100, 1e-6).unwrap());
/// let mut stacker = StackingClassifier::new(base_factories, meta, 2, 42).unwrap();
/// stacker.fit(&x, &y).unwrap();
/// let preds = stacker.predict(&x).unwrap();
/// assert_eq!(preds.shape(), &[6]);
/// ```
pub struct StackingClassifier<T: Float> {
    /// Factory closures that produce fresh base estimators.
    base_factories: Vec<EstimatorFactory<T>>,
    /// The meta-learner.
    meta_learner: Box<dyn Predictor<T>>,
    /// Number of cross-validation folds.
    cv_folds: usize,
    /// Random seed.
    seed: u64,
    /// Fitted base estimators.
    fitted_base: Option<Vec<Box<dyn Predictor<T>>>>,
}

impl<T: Float> StackingClassifier<T> {
    /// Create a new stacking classifier.
    pub fn new(
        base_factories: Vec<EstimatorFactory<T>>,
        meta_learner: Box<dyn Predictor<T>>,
        cv_folds: usize,
        seed: u64,
    ) -> Result<Self> {
        if base_factories.is_empty() {
            return Err(MlError::InvalidParameter {
                name: "base_factories",
                reason: "must provide at least one base estimator",
            });
        }
        if cv_folds < 2 {
            return Err(MlError::InvalidParameter {
                name: "cv_folds",
                reason: "must be at least 2",
            });
        }
        Ok(Self {
            base_factories,
            meta_learner,
            cv_folds,
            seed,
            fitted_base: None,
        })
    }
}

impl<T: Float> Predictor<T> for StackingClassifier<T> {
    fn fit(&mut self, x: &Tensor<T>, y: &Tensor<T>) -> Result<()> {
        let (n, p) = matrix_shape(x)?;
        check_y(y, n)?;

        let n_base = self.base_factories.len();
        let mut rng = Rng::new(self.seed);
        let kfold = KFold::new(self.cv_folds, n, &mut rng)?;

        let mut oof_preds = vec![vec![T::zero(); n]; n_base];

        for (train_idx, test_idx) in &kfold {
            let x_train = subset_rows(x, &train_idx, p);
            let y_train = subset_vec(y, &train_idx);
            let x_test = subset_rows(x, &test_idx, p);

            for (est_i, factory) in self.base_factories.iter().enumerate() {
                let mut estimator = factory();
                estimator.fit(&x_train, &y_train)?;
                let preds = estimator.predict(&x_test)?;
                let pred_data = preds.as_slice();
                for (j, &idx) in test_idx.iter().enumerate() {
                    oof_preds[est_i][idx] = pred_data[j];
                }
            }
        }

        // Build meta-feature matrix
        let mut meta_data = Vec::with_capacity(n * n_base);
        for i in 0..n {
            for est_preds in &oof_preds {
                meta_data.push(est_preds[i]);
            }
        }
        let meta_x = Tensor::from_vec(meta_data, vec![n, n_base])?;

        self.meta_learner.fit(&meta_x, y)?;

        // Re-fit on full data
        let mut fitted = Vec::with_capacity(n_base);
        for factory in &self.base_factories {
            let mut est = factory();
            est.fit(x, y)?;
            fitted.push(est);
        }
        self.fitted_base = Some(fitted);

        Ok(())
    }

    fn predict(&self, x: &Tensor<T>) -> Result<Tensor<T>> {
        let fitted = self.fitted_base.as_ref().ok_or(MlError::NotFitted)?;
        let (n, _p) = matrix_shape(x)?;
        let n_base = fitted.len();

        let base_preds: Vec<Tensor<T>> = fitted
            .iter()
            .map(|est| est.predict(x))
            .collect::<Result<Vec<_>>>()?;

        let mut meta_data = Vec::with_capacity(n * n_base);
        for i in 0..n {
            for pred_tensor in &base_preds {
                meta_data.push(pred_tensor.as_slice()[i]);
            }
        }
        let meta_x = Tensor::from_vec(meta_data, vec![n, n_base])?;

        self.meta_learner.predict(&meta_x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linear::{LinearRegression, LogisticRegression, Ridge};

    #[test]
    fn test_stacking_regressor_basic() {
        // Use uncorrelated features so QR solve succeeds on each fold
        #[rustfmt::skip]
        let x = Tensor::from_vec(
            vec![
                1.0_f64, 10.0,
                2.0,     8.0,
                3.0,     6.0,
                4.0,     9.0,
                5.0,     3.0,
                6.0,     7.0,
                7.0,     2.0,
                8.0,     5.0,
                9.0,     4.0,
                10.0,    1.0,
                11.0,    11.0,
                12.0,    0.5,
            ],
            vec![12, 2],
        )
        .unwrap();
        let y = Tensor::from_vec(
            vec![
                11.0, 10.0, 9.0, 13.0, 8.0, 13.0, 9.0, 13.0, 13.0, 11.0, 22.0, 12.5,
            ],
            vec![12],
        )
        .unwrap();

        let base_factories: Vec<EstimatorFactory<f64>> = vec![
            Box::new(|| Box::new(LinearRegression::<f64>::new())),
            Box::new(|| Box::new(Ridge::new(0.1).unwrap())),
        ];
        let meta = Box::new(LinearRegression::<f64>::new());
        let mut stacker = StackingRegressor::new(base_factories, meta, 3, 42).unwrap();
        stacker.fit(&x, &y).unwrap();
        let preds = stacker.predict(&x).unwrap();
        assert_eq!(preds.shape(), &[12]);
    }

    #[test]
    fn test_stacking_regressor_predict_before_fit() {
        let x = Tensor::from_vec(vec![1.0_f64, 2.0], vec![1, 2]).unwrap();
        let base_factories: Vec<EstimatorFactory<f64>> =
            vec![Box::new(|| Box::new(LinearRegression::<f64>::new()))];
        let meta = Box::new(LinearRegression::<f64>::new());
        let stacker = StackingRegressor::new(base_factories, meta, 2, 42).unwrap();
        assert!(stacker.predict(&x).is_err());
    }

    #[test]
    fn test_stacking_regressor_no_base_error() {
        let base_factories: Vec<EstimatorFactory<f64>> = vec![];
        let meta = Box::new(LinearRegression::<f64>::new());
        assert!(StackingRegressor::new(base_factories, meta, 2, 42).is_err());
    }

    #[test]
    fn test_stacking_classifier_basic() {
        // Simple binary classification
        #[rustfmt::skip]
        let x = Tensor::from_vec(
            vec![
                -3.0_f64, -2.0,
                -2.0, -1.0,
                -1.0, -0.5,
                 1.0,  0.5,
                 2.0,  1.0,
                 3.0,  2.0,
            ],
            vec![6, 2],
        ).unwrap();
        let y = Tensor::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0], vec![6]).unwrap();

        let base_factories: Vec<EstimatorFactory<f64>> = vec![Box::new(|| {
            Box::new(LogisticRegression::new(0.1, 200, 1e-6).unwrap())
        })];
        let meta: Box<dyn Predictor<f64>> =
            Box::new(LogisticRegression::new(0.1, 200, 1e-6).unwrap());
        let mut stacker = StackingClassifier::new(base_factories, meta, 2, 42).unwrap();
        stacker.fit(&x, &y).unwrap();
        let preds = stacker.predict(&x).unwrap();
        assert_eq!(preds.shape(), &[6]);
    }

    #[test]
    fn test_stacking_classifier_invalid_folds() {
        let base_factories: Vec<EstimatorFactory<f64>> =
            vec![Box::new(|| Box::new(LinearRegression::<f64>::new()))];
        let meta = Box::new(LinearRegression::<f64>::new());
        assert!(StackingClassifier::new(base_factories, meta, 1, 42).is_err());
    }
}
