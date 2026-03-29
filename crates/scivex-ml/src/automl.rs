//! AutoML pipeline optimization.
//!
//! Provides [`PipelineOptimizer`]-style functionality that searches over
//! combinations of preprocessing transformers and models, evaluating each
//! via k-fold cross-validation.
//!
//! # Examples
//!
//! ```ignore
//! use scivex_ml::automl::{SearchSpace, pipeline_optimize};
//! use scivex_ml::preprocessing::{StandardScaler, MinMaxScaler};
//! use scivex_ml::linear::{Ridge, LinearRegression};
//! use scivex_ml::metrics::regression::r2_score;
//! use scivex_core::random::Rng;
//!
//! let space = SearchSpace {
//!     scalers: vec![
//!         Box::new(|| Box::new(StandardScaler::<f64>::new()) as Box<dyn Transformer<f64>>),
//!         Box::new(|| Box::new(MinMaxScaler::<f64>::new()) as Box<dyn Transformer<f64>>),
//!     ],
//!     models: vec![
//!         Box::new(|| Box::new(LinearRegression::<f64>::new()) as Box<dyn Predictor<f64>>),
//!         Box::new(|| Box::new(Ridge::<f64>::new(1.0).unwrap()) as Box<dyn Predictor<f64>>),
//!     ],
//! };
//! let mut rng = Rng::new(42);
//! let result = pipeline_optimize(&space, &x, &y, 3, |yt, yp| r2_score(yt, yp), &mut rng).unwrap();
//! ```

use scivex_core::{Float, Tensor, random::Rng};

use crate::error::{MlError, Result};
use crate::model_selection::KFold;
use crate::traits::{Predictor, Transformer};

/// Type alias for a factory closure that produces a fresh transformer instance.
pub type TransformerFactory<T> = Box<dyn Fn() -> Box<dyn Transformer<T>>>;

/// Type alias for a factory closure that produces a fresh predictor instance.
pub type PredictorFactory<T> = Box<dyn Fn() -> Box<dyn Predictor<T>>>;

/// A search space of preprocessing transformers and models.
///
/// Each entry is a factory closure that creates a fresh (unfitted) instance.
/// The optimizer evaluates every (scaler, model) combination via cross-validation.
pub struct SearchSpace<T: Float> {
    /// Factory closures that produce fresh transformer instances.
    pub scalers: Vec<TransformerFactory<T>>,
    /// Factory closures that produce fresh predictor instances.
    pub models: Vec<PredictorFactory<T>>,
}

/// Result of an AutoML pipeline optimization run.
///
/// # Examples
///
/// ```
/// # use scivex_ml::automl::AutoMlResult;
/// let result = AutoMlResult::<f64> {
///     best_score: 0.95,
///     best_scaler_idx: 0,
///     best_model_idx: 1,
///     scores: vec![vec![0.90, 0.95], vec![0.88, 0.92]],
/// };
/// assert_eq!(result.best_model_idx, 1);
/// ```
#[derive(Debug, Clone)]
pub struct AutoMlResult<T: Float> {
    /// Best mean cross-validation score across all combinations.
    pub best_score: T,
    /// Index of the best scaler in the search space.
    pub best_scaler_idx: usize,
    /// Index of the best model in the search space.
    pub best_model_idx: usize,
    /// Mean CV scores indexed as `scores[scaler_idx][model_idx]`.
    pub scores: Vec<Vec<T>>,
}

/// Search over all (scaler, model) combinations via k-fold cross-validation.
///
/// For each combination the optimizer:
/// 1. Creates a fresh scaler via its factory, fits on the training fold, and
///    transforms both train and test folds.
/// 2. Creates a fresh model via its factory, fits on the transformed training
///    data, and predicts on the transformed test data.
/// 3. Computes the supplied `metric` (higher is better) and averages across
///    folds.
///
/// Returns an [`AutoMlResult`] with the best combination and full score matrix.
///
/// # Errors
///
/// Returns [`MlError::InvalidParameter`] if the search space is empty or
/// `k_folds < 2`. Propagates any error from transformers, predictors, or
/// the metric function.
pub fn pipeline_optimize<T, M>(
    search_space: &SearchSpace<T>,
    x: &Tensor<T>,
    y: &Tensor<T>,
    k_folds: usize,
    metric: M,
    rng: &mut Rng,
) -> Result<AutoMlResult<T>>
where
    T: Float,
    M: Fn(&[T], &[T]) -> Result<T>,
{
    if search_space.scalers.is_empty() {
        return Err(MlError::InvalidParameter {
            name: "scalers",
            reason: "search space must have at least one scaler",
        });
    }
    if search_space.models.is_empty() {
        return Err(MlError::InvalidParameter {
            name: "models",
            reason: "search space must have at least one model",
        });
    }

    let s = x.shape();
    if s.len() != 2 {
        return Err(MlError::DimensionMismatch {
            expected: 2,
            got: s.len(),
        });
    }
    let n = s[0];
    let p = s[1];
    if n == 0 {
        return Err(MlError::EmptyInput);
    }

    let kfold = KFold::new(k_folds, n, rng)?;
    let x_data = x.as_slice();
    let y_data = y.as_slice();

    // Collect folds so we can reuse them for every combination.
    let folds: Vec<(Vec<usize>, Vec<usize>)> = kfold.iter().collect();

    let mut best_score = T::neg_infinity();
    let mut best_scaler_idx = 0;
    let mut best_model_idx = 0;
    let mut scores: Vec<Vec<T>> = Vec::with_capacity(search_space.scalers.len());

    for (si, scaler_factory) in search_space.scalers.iter().enumerate() {
        let mut row = Vec::with_capacity(search_space.models.len());

        for (mi, model_factory) in search_space.models.iter().enumerate() {
            let mut fold_scores = Vec::with_capacity(folds.len());

            for (train_idx, test_idx) in &folds {
                let train_n = train_idx.len();
                let test_n = test_idx.len();

                // Build train/test tensors
                let mut x_train_v = vec![T::zero(); train_n * p];
                let mut y_train_v = vec![T::zero(); train_n];
                for (out_i, &idx) in train_idx.iter().enumerate() {
                    x_train_v[out_i * p..(out_i + 1) * p]
                        .copy_from_slice(&x_data[idx * p..(idx + 1) * p]);
                    y_train_v[out_i] = y_data[idx];
                }
                let mut x_test_v = vec![T::zero(); test_n * p];
                let mut y_test_v = vec![T::zero(); test_n];
                for (out_i, &idx) in test_idx.iter().enumerate() {
                    x_test_v[out_i * p..(out_i + 1) * p]
                        .copy_from_slice(&x_data[idx * p..(idx + 1) * p]);
                    y_test_v[out_i] = y_data[idx];
                }

                let x_train = Tensor::from_vec(x_train_v, vec![train_n, p])?;
                let y_train = Tensor::from_vec(y_train_v, vec![train_n])?;
                let x_test = Tensor::from_vec(x_test_v, vec![test_n, p])?;

                // Fit scaler on train, transform train and test
                let mut scaler = scaler_factory();
                let x_train_t = scaler.fit_transform(&x_train)?;
                let x_test_t = scaler.transform(&x_test)?;

                // Fit model on transformed train, predict on transformed test
                let mut model = model_factory();
                model.fit(&x_train_t, &y_train)?;
                let preds = model.predict(&x_test_t)?;

                let score = metric(&y_test_v, preds.as_slice())?;
                fold_scores.push(score);
            }

            let mean = fold_scores.iter().copied().fold(T::zero(), |a, b| a + b)
                / T::from_usize(fold_scores.len());

            if mean > best_score {
                best_score = mean;
                best_scaler_idx = si;
                best_model_idx = mi;
            }

            row.push(mean);
        }

        scores.push(row);
    }

    Ok(AutoMlResult {
        best_score,
        best_scaler_idx,
        best_model_idx,
        scores,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linear::{LinearRegression, Ridge};
    use crate::metrics::regression::r2_score;
    use crate::preprocessing::{MinMaxScaler, StandardScaler};

    fn make_linear_data() -> (Tensor<f64>, Tensor<f64>) {
        let x = Tensor::from_vec(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0, 17.0, 18.0, 19.0, 20.0,
            ],
            vec![20, 1],
        )
        .unwrap();
        let y = Tensor::from_vec(
            vec![
                3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0, 23.0, 25.0, 27.0, 29.0,
                31.0, 33.0, 35.0, 37.0, 39.0, 41.0,
            ],
            vec![20],
        )
        .unwrap();
        (x, y)
    }

    #[test]
    fn test_pipeline_optimize_basic() {
        let (x, y) = make_linear_data();

        let space = SearchSpace {
            scalers: vec![
                Box::new(|| Box::new(StandardScaler::<f64>::new()) as Box<dyn Transformer<f64>>),
                Box::new(|| Box::new(MinMaxScaler::<f64>::new()) as Box<dyn Transformer<f64>>),
            ],
            models: vec![
                Box::new(|| Box::new(LinearRegression::<f64>::new()) as Box<dyn Predictor<f64>>),
                Box::new(|| Box::new(Ridge::<f64>::new(0.1).unwrap()) as Box<dyn Predictor<f64>>),
            ],
        };

        let mut rng = Rng::new(42);
        let result = pipeline_optimize(&space, &x, &y, 3, r2_score, &mut rng).unwrap();

        assert_eq!(result.scores.len(), 2); // 2 scalers
        assert_eq!(result.scores[0].len(), 2); // 2 models each
        assert!(result.best_score > 0.5, "best R2 should be reasonable");
        assert!(result.best_scaler_idx < 2);
        assert!(result.best_model_idx < 2);
    }

    #[test]
    fn test_pipeline_optimize_empty_scalers() {
        let (x, y) = make_linear_data();
        let space = SearchSpace::<f64> {
            scalers: vec![],
            models: vec![Box::new(|| {
                Box::new(LinearRegression::<f64>::new()) as Box<dyn Predictor<f64>>
            })],
        };
        let mut rng = Rng::new(42);
        assert!(pipeline_optimize(&space, &x, &y, 3, r2_score, &mut rng).is_err());
    }

    #[test]
    fn test_pipeline_optimize_empty_models() {
        let (x, y) = make_linear_data();
        let space = SearchSpace::<f64> {
            scalers: vec![Box::new(|| {
                Box::new(StandardScaler::<f64>::new()) as Box<dyn Transformer<f64>>
            })],
            models: vec![],
        };
        let mut rng = Rng::new(42);
        assert!(pipeline_optimize(&space, &x, &y, 3, r2_score, &mut rng).is_err());
    }

    #[test]
    fn test_pipeline_optimize_score_matrix_shape() {
        let (x, y) = make_linear_data();

        let space = SearchSpace {
            scalers: vec![
                Box::new(|| Box::new(StandardScaler::<f64>::new()) as Box<dyn Transformer<f64>>),
                Box::new(|| Box::new(MinMaxScaler::<f64>::new()) as Box<dyn Transformer<f64>>),
                Box::new(|| Box::new(StandardScaler::<f64>::new()) as Box<dyn Transformer<f64>>),
            ],
            models: vec![Box::new(|| {
                Box::new(LinearRegression::<f64>::new()) as Box<dyn Predictor<f64>>
            })],
        };

        let mut rng = Rng::new(99);
        let result = pipeline_optimize(&space, &x, &y, 3, r2_score, &mut rng).unwrap();

        assert_eq!(result.scores.len(), 3);
        for row in &result.scores {
            assert_eq!(row.len(), 1);
        }
    }
}
