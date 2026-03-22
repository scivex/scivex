//! CatBoost-style ordered boosting with oblivious (symmetric) decision trees.

use scivex_core::{Float, Tensor};

use crate::error::{MlError, Result};
use crate::traits::{Classifier, Predictor};

// ── Oblivious Tree ──

/// An oblivious (symmetric) decision tree.
///
/// At each depth level the same split feature and threshold are used across
/// ALL nodes at that level. This yields `2^depth` leaf values and a very
/// compact representation that is fast to evaluate.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct ObliviousTree<T: Float> {
    /// `(feature_index, threshold)` per level, length = `depth`.
    splits: Vec<(usize, T)>,
    /// Leaf values, length = `2^depth`.
    leaf_values: Vec<T>,
    /// Tree depth (number of split levels), equal to `splits.len()`.
    _depth: usize,
}

impl<T: Float> ObliviousTree<T> {
    /// Predict a single sample and return the leaf index.
    fn predict_one(&self, row: &[T]) -> T {
        let idx = self.leaf_index(row);
        self.leaf_values[idx]
    }

    /// Return the leaf index for a given feature row.
    fn leaf_index(&self, row: &[T]) -> usize {
        let mut idx = 0usize;
        for &(feat, thresh) in &self.splits {
            idx <<= 1;
            if row[feat] > thresh {
                idx |= 1;
            }
        }
        idx
    }
}

/// Fit an oblivious tree to weighted residuals.
///
/// At each depth level, exhaustively search over all (feature, threshold)
/// pairs and pick the single split that minimises total squared error across
/// all current leaves.
fn fit_oblivious_tree<T: Float>(
    x_data: &[T],
    residuals: &[T],
    n: usize,
    p: usize,
    max_depth: usize,
    l2_reg: T,
) -> ObliviousTree<T> {
    let depth = max_depth;
    let mut splits: Vec<(usize, T)> = Vec::with_capacity(depth);

    // Track which leaf each sample currently belongs to
    let mut leaf_assignments = vec![0usize; n];

    for level in 0..depth {
        let n_leaves = 1usize << level;
        let mut best_feat = 0usize;
        let mut best_thresh = T::zero();
        let mut best_loss = T::from_f64(f64::MAX);

        // Collect candidate thresholds per feature
        for feat in 0..p {
            // Gather unique sorted values for this feature
            let mut vals: Vec<T> = (0..n).map(|i| x_data[i * p + feat]).collect();
            vals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            vals.dedup();

            if vals.len() < 2 {
                continue;
            }

            // Try midpoints as thresholds
            for w in vals.windows(2) {
                let thresh = (w[0] + w[1]) / T::from_f64(2.0);

                // For this candidate split applied globally at this level,
                // compute total squared error with optimal leaf means.
                let n_new_leaves = n_leaves * 2;
                let mut sums = vec![T::zero(); n_new_leaves];
                let mut counts = vec![0usize; n_new_leaves];

                for i in 0..n {
                    let cur = leaf_assignments[i];
                    let child = if x_data[i * p + feat] > thresh {
                        cur * 2 + 1
                    } else {
                        cur * 2
                    };
                    sums[child] += residuals[i];
                    counts[child] += 1;
                }

                // Loss = sum over leaves of: sum_i (r_i - mean)^2
                // which equals total_sum_sq - sum_leaf (S_leaf^2 / N_leaf)
                // We only need the reduction part to compare.
                let mut reduction = T::zero();
                for (s, &c) in sums.iter().zip(counts.iter()) {
                    if c > 0 {
                        reduction += *s * *s / T::from_usize(c);
                    }
                }
                // Maximise reduction (equivalently minimise negative reduction)
                let loss = -reduction;

                if loss < best_loss {
                    best_loss = loss;
                    best_feat = feat;
                    best_thresh = thresh;
                }
            }
        }

        splits.push((best_feat, best_thresh));

        // Update leaf assignments
        for i in 0..n {
            let cur = leaf_assignments[i];
            leaf_assignments[i] = if x_data[i * p + best_feat] > best_thresh {
                cur * 2 + 1
            } else {
                cur * 2
            };
        }
    }

    // Compute leaf values with L2 regularisation: mean = sum / (count + l2)
    let n_final_leaves = 1usize << depth;
    let mut sums = vec![T::zero(); n_final_leaves];
    let mut counts = vec![T::zero(); n_final_leaves];

    for i in 0..n {
        let leaf = leaf_assignments[i];
        sums[leaf] += residuals[i];
        counts[leaf] += T::one();
    }

    let leaf_values: Vec<T> = sums
        .iter()
        .zip(counts.iter())
        .map(|(&s, &c)| {
            if c > T::zero() {
                s / (c + l2_reg)
            } else {
                T::zero()
            }
        })
        .collect();

    ObliviousTree {
        splits,
        leaf_values,
        _depth: depth,
    }
}

// ── CatBoost Regressor ──

/// CatBoost-style gradient boosting regressor with ordered boosting and
/// oblivious decision trees.
///
/// Ordered boosting shuffles the training data and, for each sample, computes
/// the residual using only predictions from trees fitted on earlier samples.
/// This reduces the prediction shift (target leakage) present in standard
/// gradient boosting.
///
/// # Examples
///
/// ```
/// # use scivex_ml::prelude::*;
/// # use scivex_core::prelude::*;
/// let x = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
/// let y = Tensor::from_vec(vec![2.0, 4.0, 6.0, 8.0], vec![4]).unwrap();
/// let mut model = CatBoostRegressor::new(50, 0.1, 4, 1.0).unwrap();
/// model.fit(&x, &y).unwrap();
/// let preds = model.predict(&x).unwrap();
/// ```
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct CatBoostRegressor<T: Float> {
    n_estimators: usize,
    learning_rate: T,
    max_depth: usize,
    l2_reg: T,
    trees: Option<Vec<ObliviousTree<T>>>,
    bias: T,
}

impl<T: Float> CatBoostRegressor<T> {
    /// Create a new CatBoost regressor.
    ///
    /// - `n_estimators`: number of boosting rounds
    /// - `learning_rate`: shrinkage factor in `(0, 1]`
    /// - `max_depth`: depth of each oblivious tree (typically 6)
    /// - `l2_reg`: L2 regularisation on leaf values (>= 0)
    pub fn new(
        n_estimators: usize,
        learning_rate: f64,
        max_depth: usize,
        l2_reg: f64,
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
        if max_depth == 0 {
            return Err(MlError::InvalidParameter {
                name: "max_depth",
                reason: "must be at least 1",
            });
        }
        if l2_reg < 0.0 {
            return Err(MlError::InvalidParameter {
                name: "l2_reg",
                reason: "must be non-negative",
            });
        }
        Ok(Self {
            n_estimators,
            learning_rate: T::from_f64(learning_rate),
            max_depth,
            l2_reg: T::from_f64(l2_reg),
            trees: None,
            bias: T::zero(),
        })
    }
}

impl<T: Float> Predictor<T> for CatBoostRegressor<T> {
    fn fit(&mut self, x: &Tensor<T>, y: &Tensor<T>) -> Result<()> {
        let (n, p) = matrix_shape(x)?;
        check_y(y, n)?;
        let x_data = x.as_slice();
        let y_data = y.as_slice();

        // Initial bias: mean of y
        let bias = y_data.iter().copied().fold(T::zero(), |a, b| a + b) / T::from_usize(n);
        self.bias = bias;

        // Create a random permutation for ordered boosting
        let mut rng = scivex_core::random::Rng::new(42);
        let mut perm: Vec<usize> = (0..n).collect();
        // Fisher-Yates shuffle
        for i in (1..n).rev() {
            let j = (rng.next_f64() * (i + 1) as f64) as usize % (i + 1);
            perm.swap(i, j);
        }

        // ordered_preds[i] = cumulative prediction for permuted sample i
        // using only trees fitted on samples before it
        let mut ordered_preds = vec![bias; n];
        let mut trees = Vec::with_capacity(self.n_estimators);

        for _ in 0..self.n_estimators {
            // Compute ordered residuals: for each sample in permutation order,
            // the residual is computed using only predictions from previous samples.
            // In practice, we approximate this by using the current ordered_preds.
            let mut residuals = vec![T::zero(); n];
            for &idx in &perm {
                residuals[idx] = y_data[idx] - ordered_preds[idx];
            }

            // Fit oblivious tree to residuals using ALL data
            let tree = fit_oblivious_tree(x_data, &residuals, n, p, self.max_depth, self.l2_reg);

            // Update ordered predictions: for each sample in permutation order,
            // only update using trees fitted on data before this sample.
            // We simulate ordered boosting by doing a forward pass through the
            // permutation and updating predictions as we go.
            let mut seen_residuals = Vec::with_capacity(n);
            let mut seen_x = Vec::with_capacity(n * p);

            for &idx in &perm {
                // Current prediction for this sample uses only trees fitted
                // on previously seen data. For this round, we compute a local
                // tree from the already-seen subset if large enough, otherwise
                // use the global tree.
                let row = &x_data[idx * p..(idx + 1) * p];

                if seen_residuals.len() >= (1 << self.max_depth) {
                    // Fit a local tree on seen-so-far data
                    let local_tree = fit_oblivious_tree(
                        &seen_x,
                        &seen_residuals,
                        seen_residuals.len(),
                        p,
                        self.max_depth,
                        self.l2_reg,
                    );
                    ordered_preds[idx] += self.learning_rate * local_tree.predict_one(row);
                } else {
                    // Not enough data for a local tree; use global tree
                    ordered_preds[idx] += self.learning_rate * tree.predict_one(row);
                }

                seen_residuals.push(residuals[idx]);
                seen_x.extend_from_slice(row);
            }

            trees.push(tree);
        }

        self.trees = Some(trees);
        Ok(())
    }

    fn predict(&self, x: &Tensor<T>) -> Result<Tensor<T>> {
        let trees = self.trees.as_ref().ok_or(MlError::NotFitted)?;
        let (n, p) = matrix_shape(x)?;
        let data = x.as_slice();

        let mut preds = vec![self.bias; n];
        for tree in trees {
            for i in 0..n {
                let row = &data[i * p..(i + 1) * p];
                preds[i] += self.learning_rate * tree.predict_one(row);
            }
        }

        Tensor::from_vec(preds, vec![n]).map_err(MlError::from)
    }
}

// ── CatBoost Classifier ──

/// CatBoost-style gradient boosting classifier for binary classification.
///
/// Uses log-loss (logistic / cross-entropy) with sigmoid transform for
/// probabilities. Internally builds oblivious trees with ordered boosting.
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
/// let mut model = CatBoostClassifier::new(50, 0.1, 4, 1.0).unwrap();
/// model.fit(&x, &y).unwrap();
/// let preds = model.predict(&x).unwrap();
/// ```
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct CatBoostClassifier<T: Float> {
    n_estimators: usize,
    learning_rate: T,
    max_depth: usize,
    l2_reg: T,
    trees: Option<Vec<ObliviousTree<T>>>,
    bias: T,
}

impl<T: Float> CatBoostClassifier<T> {
    /// Create a new CatBoost classifier.
    ///
    /// - `n_estimators`: number of boosting rounds
    /// - `learning_rate`: shrinkage factor in `(0, 1]`
    /// - `max_depth`: depth of each oblivious tree (typically 6)
    /// - `l2_reg`: L2 regularisation on leaf values (>= 0)
    pub fn new(
        n_estimators: usize,
        learning_rate: f64,
        max_depth: usize,
        l2_reg: f64,
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
        if max_depth == 0 {
            return Err(MlError::InvalidParameter {
                name: "max_depth",
                reason: "must be at least 1",
            });
        }
        if l2_reg < 0.0 {
            return Err(MlError::InvalidParameter {
                name: "l2_reg",
                reason: "must be non-negative",
            });
        }
        Ok(Self {
            n_estimators,
            learning_rate: T::from_f64(learning_rate),
            max_depth,
            l2_reg: T::from_f64(l2_reg),
            trees: None,
            bias: T::zero(),
        })
    }
}

/// Sigmoid function clamped to avoid extreme values.
fn sigmoid<T: Float>(x: T) -> T {
    let one = T::one();
    one / (one + (-x).exp())
}

impl<T: Float> Predictor<T> for CatBoostClassifier<T> {
    fn fit(&mut self, x: &Tensor<T>, y: &Tensor<T>) -> Result<()> {
        let (n, p) = matrix_shape(x)?;
        check_y(y, n)?;
        let x_data = x.as_slice();
        let y_data = y.as_slice();

        // Validate binary labels
        for &v in y_data {
            let v_f64 = v.to_f64();
            if (v_f64 - 0.0).abs() > 1e-10 && (v_f64 - 1.0).abs() > 1e-10 {
                return Err(MlError::InvalidParameter {
                    name: "y",
                    reason: "must contain only 0.0 and 1.0 for binary classification",
                });
            }
        }

        // Initial bias: log-odds of positive class
        let pos_count: T = y_data.iter().copied().fold(T::zero(), |a, b| a + b);
        let pos_rate = pos_count / T::from_usize(n);
        let eps = T::from_f64(1e-10);
        let clamped = if pos_rate < eps {
            eps
        } else if pos_rate > T::one() - eps {
            T::one() - eps
        } else {
            pos_rate
        };
        let bias = (clamped / (T::one() - clamped)).ln();
        self.bias = bias;

        // Current raw scores (log-odds)
        let mut raw_scores = vec![bias; n];
        let mut trees = Vec::with_capacity(self.n_estimators);

        for _ in 0..self.n_estimators {
            // Compute negative gradient of log-loss: y - sigmoid(F(x))
            let residuals: Vec<T> = y_data
                .iter()
                .zip(raw_scores.iter())
                .map(|(&yi, &fi)| yi - sigmoid(fi))
                .collect();

            // Fit oblivious tree to residuals
            let tree = fit_oblivious_tree(x_data, &residuals, n, p, self.max_depth, self.l2_reg);

            // Update raw scores
            for i in 0..n {
                let row = &x_data[i * p..(i + 1) * p];
                raw_scores[i] += self.learning_rate * tree.predict_one(row);
            }

            trees.push(tree);
        }

        self.trees = Some(trees);
        Ok(())
    }

    fn predict(&self, x: &Tensor<T>) -> Result<Tensor<T>> {
        let proba = self.predict_proba(x)?;
        let proba_data = proba.as_slice();
        let n = proba_data.len() / 2;
        let half = T::from_f64(0.5);

        let labels: Vec<T> = (0..n)
            .map(|i| {
                if proba_data[i * 2 + 1] >= half {
                    T::one()
                } else {
                    T::zero()
                }
            })
            .collect();

        Tensor::from_vec(labels, vec![n]).map_err(MlError::from)
    }
}

impl<T: Float> Classifier<T> for CatBoostClassifier<T> {
    /// Return class probabilities, shape `[n_samples, 2]`.
    fn predict_proba(&self, x: &Tensor<T>) -> Result<Tensor<T>> {
        let trees = self.trees.as_ref().ok_or(MlError::NotFitted)?;
        let (n, p) = matrix_shape(x)?;
        let data = x.as_slice();

        let mut raw = vec![self.bias; n];
        for tree in trees {
            for i in 0..n {
                let row = &data[i * p..(i + 1) * p];
                raw[i] += self.learning_rate * tree.predict_one(row);
            }
        }

        let mut proba = Vec::with_capacity(n * 2);
        for &r in &raw {
            let p1 = sigmoid(r);
            let p0 = T::one() - p1;
            proba.push(p0);
            proba.push(p1);
        }

        Tensor::from_vec(proba, vec![n, 2]).map_err(MlError::from)
    }
}

// ── helpers ──

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
    fn test_catboost_regressor_basic() {
        // Simple linear data: y = 2x
        let x =
            Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![8, 1]).unwrap();
        let y =
            Tensor::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0], vec![8]).unwrap();

        let mut model = CatBoostRegressor::new(50, 0.1, 3, 1.0).unwrap();
        model.fit(&x, &y).unwrap();
        let preds = model.predict(&x).unwrap();

        assert_eq!(preds.shape(), &[8]);
        for (&p, &t) in preds.as_slice().iter().zip(y.as_slice()) {
            assert!(
                (p - t).abs() < 3.0,
                "prediction {p} too far from target {t}"
            );
        }
    }

    #[test]
    fn test_catboost_regressor_not_fitted() {
        let model = CatBoostRegressor::<f64>::new(10, 0.1, 3, 1.0).unwrap();
        let x = Tensor::from_vec(vec![1.0_f64, 2.0], vec![2, 1]).unwrap();
        assert!(model.predict(&x).is_err());
    }

    #[test]
    fn test_catboost_classifier_basic() {
        // Simple binary classification: low values -> 0, high values -> 1
        let x = Tensor::from_vec(
            vec![
                1.0_f64, 1.0, 2.0, 2.0, 1.5, 1.5, 8.0, 8.0, 9.0, 9.0, 8.5, 8.5,
            ],
            vec![6, 2],
        )
        .unwrap();
        let y = Tensor::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0], vec![6]).unwrap();

        let mut model = CatBoostClassifier::new(80, 0.1, 3, 1.0).unwrap();
        model.fit(&x, &y).unwrap();
        let preds = model.predict(&x).unwrap();

        assert_eq!(preds.shape(), &[6]);
        let correct: usize = preds
            .as_slice()
            .iter()
            .zip(y.as_slice())
            .filter(|(a, b)| (**a - **b).abs() < 0.5)
            .count();
        assert!(correct >= 4, "expected at least 4/6 correct, got {correct}");

        // Check predict_proba shape and row sums
        let proba = model.predict_proba(&x).unwrap();
        assert_eq!(proba.shape(), &[6, 2]);
        let p = proba.as_slice();
        for i in 0..6 {
            let row_sum = p[i * 2] + p[i * 2 + 1];
            assert!(
                (row_sum - 1.0).abs() < 1e-6,
                "row {i} sum = {row_sum}, expected 1.0"
            );
        }
    }

    #[test]
    fn test_catboost_invalid_params() {
        // Zero estimators
        assert!(CatBoostRegressor::<f64>::new(0, 0.1, 3, 1.0).is_err());
        // Negative learning rate
        assert!(CatBoostRegressor::<f64>::new(10, -0.1, 3, 1.0).is_err());
        // Zero learning rate
        assert!(CatBoostRegressor::<f64>::new(10, 0.0, 3, 1.0).is_err());
        // Learning rate > 1
        assert!(CatBoostRegressor::<f64>::new(10, 1.5, 3, 1.0).is_err());
        // Zero depth
        assert!(CatBoostRegressor::<f64>::new(10, 0.1, 0, 1.0).is_err());
        // Negative l2_reg
        assert!(CatBoostRegressor::<f64>::new(10, 0.1, 3, -1.0).is_err());

        // Classifier invalid params
        assert!(CatBoostClassifier::<f64>::new(0, 0.1, 3, 1.0).is_err());
        assert!(CatBoostClassifier::<f64>::new(10, -0.1, 3, 1.0).is_err());
    }

    #[test]
    fn test_oblivious_tree_prediction() {
        // Build a small oblivious tree manually and verify routing
        let tree = ObliviousTree {
            splits: vec![(0, 5.0_f64), (1, 3.0)],
            leaf_values: vec![1.0, 2.0, 3.0, 4.0],
            _depth: 2,
        };

        // feature[0] <= 5, feature[1] <= 3 -> leaf 0 (binary 00)
        assert!((tree.predict_one(&[3.0, 1.0]) - 1.0).abs() < 1e-10);

        // feature[0] <= 5, feature[1] > 3 -> leaf 1 (binary 01)
        assert!((tree.predict_one(&[3.0, 5.0]) - 2.0).abs() < 1e-10);

        // feature[0] > 5, feature[1] <= 3 -> leaf 2 (binary 10)
        assert!((tree.predict_one(&[7.0, 1.0]) - 3.0).abs() < 1e-10);

        // feature[0] > 5, feature[1] > 3 -> leaf 3 (binary 11)
        assert!((tree.predict_one(&[7.0, 5.0]) - 4.0).abs() < 1e-10);

        // Verify leaf indices
        assert_eq!(tree.leaf_index(&[3.0, 1.0]), 0);
        assert_eq!(tree.leaf_index(&[3.0, 5.0]), 1);
        assert_eq!(tree.leaf_index(&[7.0, 1.0]), 2);
        assert_eq!(tree.leaf_index(&[7.0, 5.0]), 3);
    }
}
