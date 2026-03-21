use scivex_core::{Float, Tensor};

use crate::error::{MlError, Result};
use crate::traits::Predictor;
use crate::tree::DecisionTreeRegressor;

// ── Loss functions ──

/// Supported loss functions for gradient boosting.
///
/// # Examples
///
/// ```
/// # use scivex_ml::ensemble::GBLoss;
/// let loss = GBLoss::Mse;
/// assert_eq!(loss, GBLoss::Mse);
/// ```
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Loss {
    /// Mean squared error (L2). Gradient = -(y - F(x)).
    Mse,
    /// Mean absolute error (L1). Gradient = -sign(y - F(x)).
    Mae,
    /// Huber loss with configurable delta. Smooth combination of L1/L2.
    Huber,
}

/// Compute the negative gradient (pseudo-residuals) for each loss function.
fn negative_gradient<T: Float>(y: &[T], f: &[T], loss: Loss, delta: T) -> Vec<T> {
    match loss {
        Loss::Mse => y.iter().zip(f).map(|(&yi, &fi)| yi - fi).collect(),
        Loss::Mae => y
            .iter()
            .zip(f)
            .map(|(&yi, &fi)| {
                let r = yi - fi;
                if r > T::zero() {
                    T::one()
                } else if r < T::zero() {
                    -T::one()
                } else {
                    T::zero()
                }
            })
            .collect(),
        Loss::Huber => y
            .iter()
            .zip(f)
            .map(|(&yi, &fi)| {
                let r = yi - fi;
                if r.abs() <= delta {
                    r
                } else if r > T::zero() {
                    delta
                } else {
                    -delta
                }
            })
            .collect(),
    }
}

// ── Gradient Boosting Regressor ──

/// Gradient boosting regressor using Friedman's algorithm.
///
/// Sequentially fits decision tree regressors to the negative gradient
/// (pseudo-residuals) of the loss function. The final prediction is the
/// sum of the initial estimate plus learning-rate-scaled tree predictions.
///
/// # Examples
///
/// ```
/// # use scivex_ml::prelude::*;
/// # use scivex_core::prelude::*;
/// let x = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
/// let y = Tensor::from_vec(vec![2.0, 4.0, 6.0, 8.0], vec![4]).unwrap();
/// let mut gbr = GradientBoostingRegressor::new(50, 0.1, Some(3), GBLoss::Mse).unwrap();
/// gbr.fit(&x, &y).unwrap();
/// let preds = gbr.predict(&x).unwrap();
/// ```
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct GradientBoostingRegressor<T: Float> {
    n_estimators: usize,
    learning_rate: T,
    max_depth: Option<usize>,
    min_samples_split: usize,
    loss: Loss,
    huber_delta: T,
    subsample: f64,
    seed: u64,
    // Fitted state
    init_prediction: Option<T>,
    trees: Option<Vec<DecisionTreeRegressor<T>>>,
}

impl<T: Float> GradientBoostingRegressor<T> {
    /// Create a new gradient boosting regressor.
    ///
    /// - `n_estimators`: number of boosting rounds (trees)
    /// - `learning_rate`: shrinkage applied to each tree's contribution
    /// - `max_depth`: maximum depth per tree (default: 3)
    /// - `loss`: loss function (`Mse`, `Mae`, or `Huber`)
    pub fn new(
        n_estimators: usize,
        learning_rate: f64,
        max_depth: Option<usize>,
        loss: Loss,
    ) -> Result<Self> {
        if n_estimators == 0 {
            return Err(MlError::InvalidParameter {
                name: "n_estimators",
                reason: "must be at least 1",
            });
        }
        if learning_rate <= 0.0 || learning_rate > 1.0 {
            return Err(MlError::InvalidParameter {
                name: "learning_rate",
                reason: "must be in (0, 1]",
            });
        }
        Ok(Self {
            n_estimators,
            learning_rate: T::from_f64(learning_rate),
            max_depth: Some(max_depth.unwrap_or(3)),
            min_samples_split: 2,
            loss,
            huber_delta: T::from_f64(1.35),
            subsample: 1.0,
            seed: 42,
            init_prediction: None,
            trees: None,
        })
    }

    /// Set the subsample fraction (stochastic gradient boosting).
    /// Values < 1.0 enable stochastic gradient boosting.
    pub fn set_subsample(&mut self, frac: f64) -> &mut Self {
        self.subsample = frac.clamp(0.1, 1.0);
        self
    }

    /// Set the random seed for subsampling.
    pub fn set_seed(&mut self, seed: u64) -> &mut Self {
        self.seed = seed;
        self
    }

    /// Set the Huber delta parameter (only used with `Loss::Huber`).
    pub fn set_huber_delta(&mut self, delta: f64) -> &mut Self {
        self.huber_delta = T::from_f64(delta.max(0.01));
        self
    }

    /// Set minimum samples required to split a node.
    pub fn set_min_samples_split(&mut self, min: usize) -> &mut Self {
        self.min_samples_split = min.max(2);
        self
    }

    /// Return feature importances based on impurity reduction across all trees.
    /// Returns a 1-D tensor of shape `[n_features]`.
    pub fn feature_importances(&self, n_features: usize) -> Result<Tensor<T>> {
        let trees = self.trees.as_ref().ok_or(MlError::NotFitted)?;
        let mut importances = vec![T::zero(); n_features];

        for tree in trees {
            let tree_imp = tree.feature_importances(n_features);
            for (i, &v) in tree_imp.iter().enumerate() {
                importances[i] += v;
            }
        }

        // Normalise
        let total: T = importances.iter().copied().fold(T::zero(), |a, b| a + b);
        if total > T::zero() {
            for v in &mut importances {
                *v /= total;
            }
        }

        Tensor::from_vec(importances, vec![n_features]).map_err(MlError::from)
    }

    /// Staged predictions: returns the prediction after each boosting round.
    /// Useful for early stopping analysis.
    pub fn staged_predict(&self, x: &Tensor<T>) -> Result<Vec<Tensor<T>>> {
        let trees = self.trees.as_ref().ok_or(MlError::NotFitted)?;
        let init = self.init_prediction.ok_or(MlError::NotFitted)?;
        let (n, _p) = matrix_shape(x)?;

        let mut f = vec![init; n];
        let mut stages = Vec::with_capacity(trees.len());

        for tree in trees {
            let preds = tree.predict(x)?;
            for (i, &p) in preds.as_slice().iter().enumerate() {
                f[i] += self.learning_rate * p;
            }
            stages.push(Tensor::from_vec(f.clone(), vec![n])?);
        }

        Ok(stages)
    }
}

impl<T: Float> Predictor<T> for GradientBoostingRegressor<T> {
    fn fit(&mut self, x: &Tensor<T>, y: &Tensor<T>) -> Result<()> {
        let (n, _p) = matrix_shape(x)?;
        check_y(y, n)?;
        let y_data = y.as_slice();

        // Initial prediction: mean of y (optimal constant for MSE)
        let init = y_data.iter().copied().fold(T::zero(), |a, b| a + b) / T::from_usize(n);
        self.init_prediction = Some(init);

        // Current predictions
        let mut f = vec![init; n];
        let mut trees = Vec::with_capacity(self.n_estimators);

        let use_subsample = self.subsample < 1.0;
        let mut rng = scivex_core::random::Rng::new(self.seed);

        for _ in 0..self.n_estimators {
            // Compute pseudo-residuals
            let residuals = negative_gradient(y_data, &f, self.loss, self.huber_delta);

            // Subsample indices if needed
            let (fit_x, fit_residuals) = if use_subsample {
                let sub_n = (self.subsample * n as f64).ceil() as usize;
                let sub_n = sub_n.max(1).min(n);
                subsample_data(x, &residuals, n, sub_n, &mut rng)?
            } else {
                (x.clone(), Tensor::from_vec(residuals, vec![n])?)
            };

            // Fit a tree to pseudo-residuals
            let mut tree = DecisionTreeRegressor::new(self.max_depth, self.min_samples_split);
            tree.fit(&fit_x, &fit_residuals)?;

            // Update predictions using the FULL dataset
            let update = tree.predict(x)?;
            for (i, &u) in update.as_slice().iter().enumerate() {
                f[i] += self.learning_rate * u;
            }

            trees.push(tree);
        }

        self.trees = Some(trees);
        Ok(())
    }

    fn predict(&self, x: &Tensor<T>) -> Result<Tensor<T>> {
        let trees = self.trees.as_ref().ok_or(MlError::NotFitted)?;
        let init = self.init_prediction.ok_or(MlError::NotFitted)?;
        let (n, _p) = matrix_shape(x)?;

        let mut f = vec![init; n];
        for tree in trees {
            let preds = tree.predict(x)?;
            for (i, &p) in preds.as_slice().iter().enumerate() {
                f[i] += self.learning_rate * p;
            }
        }

        Tensor::from_vec(f, vec![n]).map_err(MlError::from)
    }
}

// ── Gradient Boosting Classifier ──

/// Gradient boosting classifier for binary and multi-class problems.
///
/// Uses log-loss (cross-entropy) with one-vs-all for multi-class.
/// Each class gets its own sequence of regression trees fitted to
/// the negative gradient of the log-loss.
///
/// # Examples
///
/// ```
/// # use scivex_ml::prelude::*;
/// # use scivex_core::prelude::*;
/// let x = Tensor::from_vec(
///     vec![1.0_f64, 1.0, 2.0, 2.0, 8.0, 8.0, 9.0, 9.0],
///     vec![4, 2],
/// ).unwrap();
/// let y = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0], vec![4]).unwrap();
/// let mut gbc = GradientBoostingClassifier::new(50, 0.1, Some(3)).unwrap();
/// gbc.fit(&x, &y).unwrap();
/// let preds = gbc.predict(&x).unwrap();
/// ```
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct GradientBoostingClassifier<T: Float> {
    n_estimators: usize,
    learning_rate: T,
    max_depth: Option<usize>,
    min_samples_split: usize,
    seed: u64,
    // Fitted state
    classes: Option<Vec<T>>,
    init_log_odds: Option<Vec<T>>,
    /// trees[class_idx][round] — one tree sequence per class
    trees: Option<Vec<Vec<DecisionTreeRegressor<T>>>>,
}

impl<T: Float> GradientBoostingClassifier<T> {
    /// Create a new gradient boosting classifier.
    pub fn new(n_estimators: usize, learning_rate: f64, max_depth: Option<usize>) -> Result<Self> {
        if n_estimators == 0 {
            return Err(MlError::InvalidParameter {
                name: "n_estimators",
                reason: "must be at least 1",
            });
        }
        if learning_rate <= 0.0 || learning_rate > 1.0 {
            return Err(MlError::InvalidParameter {
                name: "learning_rate",
                reason: "must be in (0, 1]",
            });
        }
        Ok(Self {
            n_estimators,
            learning_rate: T::from_f64(learning_rate),
            max_depth: Some(max_depth.unwrap_or(3)),
            min_samples_split: 2,
            seed: 42,
            classes: None,
            init_log_odds: None,
            trees: None,
        })
    }

    /// Set minimum samples required to split a node.
    pub fn set_min_samples_split(&mut self, min: usize) -> &mut Self {
        self.min_samples_split = min.max(2);
        self
    }

    /// Set the random seed.
    pub fn set_seed(&mut self, seed: u64) -> &mut Self {
        self.seed = seed;
        self
    }

    /// Predict class probabilities, shape `[n_samples, n_classes]`.
    pub fn predict_proba(&self, x: &Tensor<T>) -> Result<Tensor<T>> {
        let trees = self.trees.as_ref().ok_or(MlError::NotFitted)?;
        let init = self.init_log_odds.as_ref().ok_or(MlError::NotFitted)?;
        let classes = self.classes.as_ref().ok_or(MlError::NotFitted)?;
        let n_classes = classes.len();
        let (n, _p) = matrix_shape(x)?;

        // Compute raw scores per class
        let mut scores = vec![vec![T::zero(); n]; n_classes];
        for (c, class_trees) in trees.iter().enumerate() {
            scores[c].fill(init[c]);
            for tree in class_trees {
                let preds = tree.predict(x)?;
                for (i, &p) in preds.as_slice().iter().enumerate() {
                    scores[c][i] += self.learning_rate * p;
                }
            }
        }

        // Softmax across classes
        let mut proba = vec![T::zero(); n * n_classes];
        for i in 0..n {
            let max_score = (0..n_classes)
                .map(|c| scores[c][i])
                .fold(T::neg_infinity(), |a, b| if b > a { b } else { a });
            let mut sum_exp = T::zero();
            for c in 0..n_classes {
                let e = (scores[c][i] - max_score).exp();
                proba[i * n_classes + c] = e;
                sum_exp += e;
            }
            for c in 0..n_classes {
                proba[i * n_classes + c] /= sum_exp;
            }
        }

        Tensor::from_vec(proba, vec![n, n_classes]).map_err(MlError::from)
    }
}

impl<T: Float> Predictor<T> for GradientBoostingClassifier<T> {
    fn fit(&mut self, x: &Tensor<T>, y: &Tensor<T>) -> Result<()> {
        let (n, _p) = matrix_shape(x)?;
        check_y(y, n)?;
        let y_data = y.as_slice();

        // Discover classes
        let mut classes: Vec<T> = Vec::new();
        for &v in y_data {
            if !classes.iter().any(|&c| (c - v).abs() < T::epsilon()) {
                classes.push(v);
            }
        }
        classes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n_classes = classes.len();

        if n_classes < 2 {
            return Err(MlError::InvalidParameter {
                name: "y",
                reason: "must contain at least 2 classes",
            });
        }

        // Encode y as one-hot indicators per class
        let mut y_one_hot = vec![vec![T::zero(); n]; n_classes];
        for i in 0..n {
            for (c, &cls) in classes.iter().enumerate() {
                if (y_data[i] - cls).abs() < T::epsilon() {
                    y_one_hot[c][i] = T::one();
                }
            }
        }

        // Initial log-odds: log(class_count / total)
        let mut init_log_odds = Vec::with_capacity(n_classes);
        for class_indicators in &y_one_hot {
            let count: T = class_indicators
                .iter()
                .copied()
                .fold(T::zero(), |a, b| a + b);
            let p = count / T::from_usize(n);
            // Clamp to avoid log(0)
            let p_clamped = if p < T::from_f64(1e-10) {
                T::from_f64(1e-10)
            } else if p > T::from_f64(1.0 - 1e-10) {
                T::from_f64(1.0 - 1e-10)
            } else {
                p
            };
            init_log_odds.push(p_clamped.ln());
        }

        // Current scores: scores[class][sample]
        let mut scores = vec![vec![T::zero(); n]; n_classes];
        for (score_row, &log_odds) in scores.iter_mut().zip(init_log_odds.iter()) {
            score_row.fill(log_odds);
        }

        let mut all_trees = vec![Vec::with_capacity(self.n_estimators); n_classes];

        for _ in 0..self.n_estimators {
            // Compute softmax probabilities
            let proba = softmax_scores(&scores, n, n_classes);

            // For each class, fit a tree to (y_one_hot[c] - proba[c])
            for c in 0..n_classes {
                let residuals: Vec<T> = (0..n).map(|i| y_one_hot[c][i] - proba[c][i]).collect();

                let residual_tensor = Tensor::from_vec(residuals, vec![n])?;
                let mut tree = DecisionTreeRegressor::new(self.max_depth, self.min_samples_split);
                tree.fit(x, &residual_tensor)?;

                let update = tree.predict(x)?;
                // Apply Newton-Raphson scaling factor for multi-class: (K-1)/K
                let scale =
                    self.learning_rate * T::from_f64((n_classes - 1) as f64 / n_classes as f64);
                for (i, &u) in update.as_slice().iter().enumerate() {
                    scores[c][i] += scale * u;
                }

                all_trees[c].push(tree);
            }
        }

        self.classes = Some(classes);
        self.init_log_odds = Some(init_log_odds);
        self.trees = Some(all_trees);
        Ok(())
    }

    fn predict(&self, x: &Tensor<T>) -> Result<Tensor<T>> {
        let classes = self.classes.as_ref().ok_or(MlError::NotFitted)?;
        let proba_tensor = self.predict_proba(x)?;
        let proba = proba_tensor.as_slice();
        let n_classes = classes.len();
        let n = proba.len() / n_classes;

        let mut out = vec![T::zero(); n];
        for i in 0..n {
            let mut best_c = 0;
            let mut best_p = proba[i * n_classes];
            for c in 1..n_classes {
                if proba[i * n_classes + c] > best_p {
                    best_p = proba[i * n_classes + c];
                    best_c = c;
                }
            }
            out[i] = classes[best_c];
        }

        Tensor::from_vec(out, vec![n]).map_err(MlError::from)
    }
}

// ── helpers ──

fn softmax_scores<T: Float>(scores: &[Vec<T>], n: usize, n_classes: usize) -> Vec<Vec<T>> {
    let mut proba = vec![vec![T::zero(); n]; n_classes];
    for i in 0..n {
        let max_s = (0..n_classes)
            .map(|c| scores[c][i])
            .fold(T::neg_infinity(), |a, b| if b > a { b } else { a });
        let mut sum_exp = T::zero();
        for (p_row, s_row) in proba.iter_mut().zip(scores.iter()) {
            let e = (s_row[i] - max_s).exp();
            p_row[i] = e;
            sum_exp += e;
        }
        for p_row in &mut proba {
            p_row[i] /= sum_exp;
        }
    }
    proba
}

fn subsample_data<T: Float>(
    x: &Tensor<T>,
    residuals: &[T],
    n: usize,
    sub_n: usize,
    rng: &mut scivex_core::random::Rng,
) -> Result<(Tensor<T>, Tensor<T>)> {
    let p = x.shape()[1];
    let x_data = x.as_slice();
    let mut sub_x = vec![T::zero(); sub_n * p];
    let mut sub_r = vec![T::zero(); sub_n];

    for i in 0..sub_n {
        let idx = (rng.next_f64() * n as f64) as usize % n;
        sub_x[i * p..(i + 1) * p].copy_from_slice(&x_data[idx * p..(idx + 1) * p]);
        sub_r[i] = residuals[idx];
    }

    Ok((
        Tensor::from_vec(sub_x, vec![sub_n, p])?,
        Tensor::from_vec(sub_r, vec![sub_n])?,
    ))
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
    fn test_gbr_mse_linear() {
        // y = 2x, should fit well
        let x =
            Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![8, 1]).unwrap();
        let y =
            Tensor::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0], vec![8]).unwrap();

        let mut gbr = GradientBoostingRegressor::new(100, 0.1, Some(3), Loss::Mse).unwrap();
        gbr.fit(&x, &y).unwrap();
        let preds = gbr.predict(&x).unwrap();

        for (&p, &t) in preds.as_slice().iter().zip(y.as_slice()) {
            assert!(
                (p - t).abs() < 1.5,
                "prediction {p} too far from target {t}"
            );
        }
    }

    #[test]
    fn test_gbr_mae_loss() {
        let x = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
        let y = Tensor::from_vec(vec![2.0, 4.0, 6.0, 8.0], vec![4]).unwrap();

        let mut gbr = GradientBoostingRegressor::new(50, 0.1, Some(3), Loss::Mae).unwrap();
        gbr.fit(&x, &y).unwrap();
        let preds = gbr.predict(&x).unwrap();

        for &p in preds.as_slice() {
            assert!(p > 0.0 && p < 12.0, "prediction {p} out of range");
        }
    }

    #[test]
    fn test_gbr_huber_loss() {
        let x = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
        let y = Tensor::from_vec(vec![2.0, 4.0, 6.0, 8.0], vec![4]).unwrap();

        let mut gbr = GradientBoostingRegressor::new(50, 0.1, Some(3), Loss::Huber).unwrap();
        gbr.fit(&x, &y).unwrap();
        let preds = gbr.predict(&x).unwrap();

        for &p in preds.as_slice() {
            assert!(p > 0.0 && p < 12.0, "prediction {p} out of range");
        }
    }

    #[test]
    fn test_gbr_staged_predict() {
        let x = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
        let y = Tensor::from_vec(vec![2.0, 4.0, 6.0, 8.0], vec![4]).unwrap();

        let mut gbr = GradientBoostingRegressor::new(10, 0.1, Some(3), Loss::Mse).unwrap();
        gbr.fit(&x, &y).unwrap();

        let stages = gbr.staged_predict(&x).unwrap();
        assert_eq!(stages.len(), 10);

        // Final staged prediction should match predict()
        let final_pred = gbr.predict(&x).unwrap();
        for (&s, &p) in stages[9].as_slice().iter().zip(final_pred.as_slice()) {
            assert!((s - p).abs() < 1e-10);
        }
    }

    #[test]
    fn test_gbr_subsample() {
        let x =
            Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![8, 1]).unwrap();
        let y =
            Tensor::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0], vec![8]).unwrap();

        let mut gbr = GradientBoostingRegressor::new(50, 0.1, Some(3), Loss::Mse).unwrap();
        gbr.set_subsample(0.5).set_seed(123);
        gbr.fit(&x, &y).unwrap();
        let preds = gbr.predict(&x).unwrap();

        for &p in preds.as_slice() {
            assert!(p > 0.0 && p < 20.0, "prediction {p} out of range");
        }
    }

    #[test]
    fn test_gbr_feature_importances() {
        // Feature 0 is predictive, feature 1 is noise
        let x = Tensor::from_vec(
            vec![
                1.0_f64, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0, 5.0, 50.0, 6.0, 60.0,
            ],
            vec![6, 2],
        )
        .unwrap();
        let y = Tensor::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0], vec![6]).unwrap();

        let mut gbr = GradientBoostingRegressor::new(20, 0.1, Some(2), Loss::Mse).unwrap();
        gbr.fit(&x, &y).unwrap();

        let imp = gbr.feature_importances(2).unwrap();
        let imp_slice = imp.as_slice();
        assert_eq!(imp_slice.len(), 2);
        // Sum should be ~1.0
        let total: f64 = imp_slice.iter().sum();
        assert!((total - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_gbc_binary() {
        // Simple binary classification
        let x = Tensor::from_vec(
            vec![
                1.0_f64, 1.0, 2.0, 2.0, 1.5, 1.5, 8.0, 8.0, 9.0, 9.0, 8.5, 8.5,
            ],
            vec![6, 2],
        )
        .unwrap();
        let y = Tensor::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0], vec![6]).unwrap();

        let mut gbc = GradientBoostingClassifier::new(50, 0.1, Some(3)).unwrap();
        gbc.fit(&x, &y).unwrap();
        let preds = gbc.predict(&x).unwrap();

        let correct: usize = preds
            .as_slice()
            .iter()
            .zip(y.as_slice())
            .filter(|(a, b)| (**a - **b).abs() < 0.5)
            .count();
        assert!(correct >= 4, "expected at least 4/6 correct, got {correct}");
    }

    #[test]
    fn test_gbc_predict_proba() {
        let x =
            Tensor::from_vec(vec![1.0_f64, 1.0, 2.0, 2.0, 8.0, 8.0, 9.0, 9.0], vec![4, 2]).unwrap();
        let y = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0], vec![4]).unwrap();

        let mut gbc = GradientBoostingClassifier::new(30, 0.1, Some(3)).unwrap();
        gbc.fit(&x, &y).unwrap();

        let proba = gbc.predict_proba(&x).unwrap();
        assert_eq!(proba.shape(), &[4, 2]);

        // Each row should sum to ~1.0
        let p = proba.as_slice();
        for i in 0..4 {
            let row_sum = p[i * 2] + p[i * 2 + 1];
            assert!(
                (row_sum - 1.0).abs() < 1e-6,
                "row {i} sum = {row_sum}, expected 1.0"
            );
        }
    }

    #[test]
    fn test_gbc_multiclass() {
        // 3-class problem
        let x = Tensor::from_vec(
            vec![
                1.0_f64, 0.0, 1.5, 0.0, // class 0
                0.0, 5.0, 0.0, 5.5, // class 1
                5.0, 5.0, 5.5, 5.5, // class 2
            ],
            vec![6, 2],
        )
        .unwrap();
        let y = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0], vec![6]).unwrap();

        let mut gbc = GradientBoostingClassifier::new(50, 0.1, Some(3)).unwrap();
        gbc.fit(&x, &y).unwrap();
        let preds = gbc.predict(&x).unwrap();

        let correct: usize = preds
            .as_slice()
            .iter()
            .zip(y.as_slice())
            .filter(|(a, b)| (**a - **b).abs() < 0.5)
            .count();
        assert!(
            correct >= 4,
            "expected at least 4/6 correct in multiclass, got {correct}"
        );
    }

    #[test]
    fn test_gbr_not_fitted() {
        let gbr = GradientBoostingRegressor::<f64>::new(10, 0.1, Some(3), Loss::Mse).unwrap();
        let x = Tensor::from_vec(vec![1.0_f64, 2.0], vec![2, 1]).unwrap();
        assert!(gbr.predict(&x).is_err());
    }

    #[test]
    fn test_gbc_not_fitted() {
        let gbc = GradientBoostingClassifier::<f64>::new(10, 0.1, Some(3)).unwrap();
        let x = Tensor::from_vec(vec![1.0_f64, 2.0], vec![2, 1]).unwrap();
        assert!(gbc.predict(&x).is_err());
    }

    #[test]
    fn test_gbr_invalid_params() {
        assert!(GradientBoostingRegressor::<f64>::new(0, 0.1, Some(3), Loss::Mse).is_err());
        assert!(GradientBoostingRegressor::<f64>::new(10, 0.0, Some(3), Loss::Mse).is_err());
        assert!(GradientBoostingRegressor::<f64>::new(10, 1.5, Some(3), Loss::Mse).is_err());
    }

    #[test]
    fn test_gbr_residuals_decrease() {
        // MSE should decrease over boosting rounds
        let x = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0], vec![5, 1]).unwrap();
        let y = Tensor::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0], vec![5]).unwrap();

        let mut gbr = GradientBoostingRegressor::new(20, 0.1, Some(3), Loss::Mse).unwrap();
        gbr.fit(&x, &y).unwrap();

        let stages = gbr.staged_predict(&x).unwrap();
        let y_data = y.as_slice();

        // Compute MSE at stage 0 and final stage
        let mse = |preds: &[f64]| -> f64 {
            preds
                .iter()
                .zip(y_data)
                .map(|(&p, &t)| (p - t) * (p - t))
                .sum::<f64>()
                / preds.len() as f64
        };

        let mse_early = mse(stages[0].as_slice());
        let mse_late = mse(stages[stages.len() - 1].as_slice());
        assert!(
            mse_late < mse_early,
            "MSE should decrease: early={mse_early}, late={mse_late}"
        );
    }
}
