use scivex_core::{Float, Tensor};

use crate::error::{MlError, Result};
use crate::traits::Predictor;

// ── Internal tree node ──

#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub(crate) enum Node<T: Float> {
    Leaf {
        value: T,
    },
    Split {
        feature: usize,
        threshold: T,
        left: Box<Node<T>>,
        right: Box<Node<T>>,
    },
}

impl<T: Float> Node<T> {
    fn predict_one(&self, row: &[T]) -> T {
        match self {
            Self::Leaf { value } => *value,
            Self::Split {
                feature,
                threshold,
                left,
                right,
            } => {
                if row[*feature] <= *threshold {
                    left.predict_one(row)
                } else {
                    right.predict_one(row)
                }
            }
        }
    }
}

// ── Split criterion helpers ──

/// Gini impurity for a set of class counts.
fn gini<T: Float>(class_counts: &[usize], total: usize) -> T {
    if total == 0 {
        return T::zero();
    }
    let n = T::from_usize(total);
    let mut sum = T::zero();
    for &c in class_counts {
        let p = T::from_usize(c) / n;
        sum += p * p;
    }
    T::one() - sum
}

/// MSE impurity for regression: variance of values.
fn mse_impurity<T: Float>(values: &[T]) -> T {
    if values.is_empty() {
        return T::zero();
    }
    let n = T::from_usize(values.len());
    let mean = values.iter().copied().fold(T::zero(), |a, b| a + b) / n;
    values
        .iter()
        .map(|&v| {
            let d = v - mean;
            d * d
        })
        .fold(T::zero(), |a, b| a + b)
        / n
}

/// Find unique class labels and the maximum class index.
fn class_info<T: Float>(y: &[T]) -> (Vec<T>, usize) {
    let mut classes: Vec<T> = Vec::new();
    for &v in y {
        if !classes.iter().any(|&c| (c - v).abs() < T::epsilon()) {
            classes.push(v);
        }
    }
    classes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = classes.len();
    (classes, n)
}

fn class_index<T: Float>(classes: &[T], v: T) -> usize {
    classes
        .iter()
        .position(|&c| (c - v).abs() < T::epsilon())
        .unwrap_or(0)
}

// ── Decision Tree Classifier ──

/// CART decision tree for classification (Gini impurity).
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct DecisionTreeClassifier<T: Float> {
    pub(crate) max_depth: Option<usize>,
    pub(crate) min_samples_split: usize,
    pub(crate) root: Option<Node<T>>,
    pub(crate) classes: Option<Vec<T>>,
}

impl<T: Float> Default for DecisionTreeClassifier<T> {
    fn default() -> Self {
        Self::new(None, 2)
    }
}

impl<T: Float> DecisionTreeClassifier<T> {
    /// Create a new decision tree classifier with optional depth limit.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_core::Tensor;
    /// # use scivex_ml::tree::DecisionTreeClassifier;
    /// # use scivex_ml::traits::Predictor;
    /// let x = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
    /// let y = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0], vec![4]).unwrap();
    /// let mut tree = DecisionTreeClassifier::<f64>::new(Some(3), 1);
    /// tree.fit(&x, &y).unwrap();
    /// let preds = tree.predict(&x).unwrap();
    /// assert_eq!(preds.as_slice(), &[0.0, 0.0, 1.0, 1.0]);
    /// ```
    pub fn new(max_depth: Option<usize>, min_samples_split: usize) -> Self {
        Self {
            max_depth,
            min_samples_split,
            root: None,
            classes: None,
        }
    }

    fn build(
        &self,
        x: &[T],
        y: &[T],
        indices: &[usize],
        p: usize,
        depth: usize,
        classes: &[T],
    ) -> Node<T> {
        let n_classes = classes.len();
        // Leaf conditions
        let at_max_depth = self.max_depth.is_some_and(|d| depth >= d);
        if indices.len() < self.min_samples_split || at_max_depth || indices.len() <= 1 {
            return Node::Leaf {
                value: majority_class(y, indices, classes),
            };
        }

        // Check if all same class
        let first_class = y[indices[0]];
        if indices
            .iter()
            .all(|&i| (y[i] - first_class).abs() < T::epsilon())
        {
            return Node::Leaf { value: first_class };
        }

        // Find best split
        let mut best_gain = T::neg_infinity();
        let mut best_feature = 0;
        let mut best_threshold = T::zero();

        let parent_counts = count_classes(y, indices, classes);
        let parent_gini: T = gini(&parent_counts, indices.len());

        for feat in 0..p {
            // Collect and sort unique thresholds
            let mut vals: Vec<T> = indices.iter().map(|&i| x[i * p + feat]).collect();
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            vals.dedup();

            if vals.len() <= 1 {
                continue;
            }

            for w in vals.windows(2) {
                let threshold = (w[0] + w[1]) / T::from_usize(2);
                let (mut left_counts, mut right_counts) =
                    (vec![0usize; n_classes], vec![0usize; n_classes]);
                let (mut n_left, mut n_right) = (0usize, 0usize);
                for &i in indices {
                    let ci = class_index(classes, y[i]);
                    if x[i * p + feat] <= threshold {
                        left_counts[ci] += 1;
                        n_left += 1;
                    } else {
                        right_counts[ci] += 1;
                        n_right += 1;
                    }
                }
                if n_left == 0 || n_right == 0 {
                    continue;
                }
                let left_gini: T = gini(&left_counts, n_left);
                let right_gini: T = gini(&right_counts, n_right);
                let n_total = T::from_usize(indices.len());
                let weighted = T::from_usize(n_left) / n_total * left_gini
                    + T::from_usize(n_right) / n_total * right_gini;
                let gain = parent_gini - weighted;
                if gain > best_gain {
                    best_gain = gain;
                    best_feature = feat;
                    best_threshold = threshold;
                }
            }
        }

        if best_gain < T::zero() {
            return Node::Leaf {
                value: majority_class(y, indices, classes),
            };
        }

        let (left_idx, right_idx): (Vec<usize>, Vec<usize>) = indices
            .iter()
            .partition(|&&i| x[i * p + best_feature] <= best_threshold);

        if left_idx.is_empty() || right_idx.is_empty() {
            return Node::Leaf {
                value: majority_class(y, indices, classes),
            };
        }

        Node::Split {
            feature: best_feature,
            threshold: best_threshold,
            left: Box::new(self.build(x, y, &left_idx, p, depth + 1, classes)),
            right: Box::new(self.build(x, y, &right_idx, p, depth + 1, classes)),
        }
    }
}

impl<T: Float> Predictor<T> for DecisionTreeClassifier<T> {
    fn fit(&mut self, x: &Tensor<T>, y: &Tensor<T>) -> Result<()> {
        let (n, p) = matrix_shape(x)?;
        check_y(y, n)?;
        let x_data = x.as_slice();
        let y_data = y.as_slice();
        let (classes, _n_classes) = class_info(y_data);
        let indices: Vec<usize> = (0..n).collect();
        let root = self.build(x_data, y_data, &indices, p, 0, &classes);
        self.root = Some(root);
        self.classes = Some(classes);
        Ok(())
    }

    fn predict(&self, x: &Tensor<T>) -> Result<Tensor<T>> {
        let root = self.root.as_ref().ok_or(MlError::NotFitted)?;
        let (n, p) = matrix_shape(x)?;
        let data = x.as_slice();
        let mut out = vec![T::zero(); n];
        for i in 0..n {
            out[i] = root.predict_one(&data[i * p..(i + 1) * p]);
        }
        Ok(Tensor::from_vec(out, vec![n])?)
    }
}

// ── Decision Tree Regressor ──

/// CART decision tree for regression (MSE criterion).
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct DecisionTreeRegressor<T: Float> {
    pub(crate) max_depth: Option<usize>,
    pub(crate) min_samples_split: usize,
    pub(crate) root: Option<Node<T>>,
}

impl<T: Float> Default for DecisionTreeRegressor<T> {
    fn default() -> Self {
        Self::new(None, 2)
    }
}

impl<T: Float> DecisionTreeRegressor<T> {
    /// Create a new decision tree regressor with optional depth limit.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_ml::prelude::*;
    /// # use scivex_core::prelude::*;
    /// let x = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0], vec![3, 1]).unwrap();
    /// let y = Tensor::from_vec(vec![10.0, 20.0, 30.0], vec![3]).unwrap();
    /// let mut tree = DecisionTreeRegressor::<f64>::new(None, 1);
    /// tree.fit(&x, &y).unwrap();
    /// let preds = tree.predict(&x).unwrap();
    /// ```
    pub fn new(max_depth: Option<usize>, min_samples_split: usize) -> Self {
        Self {
            max_depth,
            min_samples_split,
            root: None,
        }
    }

    fn build(&self, x: &[T], y: &[T], indices: &[usize], p: usize, depth: usize) -> Node<T> {
        let at_max_depth = self.max_depth.is_some_and(|d| depth >= d);
        if indices.len() < self.min_samples_split || at_max_depth || indices.len() <= 1 {
            return Node::Leaf {
                value: mean_of(y, indices),
            };
        }

        let parent_mse = mse_of(y, indices);
        let mut best_reduction = T::neg_infinity();
        let mut best_feature = 0;
        let mut best_threshold = T::zero();

        for feat in 0..p {
            let mut vals: Vec<T> = indices.iter().map(|&i| x[i * p + feat]).collect();
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            vals.dedup();
            if vals.len() <= 1 {
                continue;
            }

            for w in vals.windows(2) {
                let threshold = (w[0] + w[1]) / T::from_usize(2);
                let (left, right): (Vec<usize>, Vec<usize>) =
                    indices.iter().partition(|&&i| x[i * p + feat] <= threshold);
                if left.is_empty() || right.is_empty() {
                    continue;
                }
                let left_vals: Vec<T> = left.iter().map(|&i| y[i]).collect();
                let right_vals: Vec<T> = right.iter().map(|&i| y[i]).collect();
                let n_total = T::from_usize(indices.len());
                let weighted = T::from_usize(left.len()) / n_total * mse_impurity(&left_vals)
                    + T::from_usize(right.len()) / n_total * mse_impurity(&right_vals);
                let reduction = parent_mse - weighted;
                if reduction > best_reduction {
                    best_reduction = reduction;
                    best_feature = feat;
                    best_threshold = threshold;
                }
            }
        }

        if best_reduction <= T::zero() {
            return Node::Leaf {
                value: mean_of(y, indices),
            };
        }

        let (left_idx, right_idx): (Vec<usize>, Vec<usize>) = indices
            .iter()
            .partition(|&&i| x[i * p + best_feature] <= best_threshold);

        Node::Split {
            feature: best_feature,
            threshold: best_threshold,
            left: Box::new(self.build(x, y, &left_idx, p, depth + 1)),
            right: Box::new(self.build(x, y, &right_idx, p, depth + 1)),
        }
    }
}

impl<T: Float> DecisionTreeRegressor<T> {
    /// Compute feature importance based on the number of times each feature is
    /// used as a split. Returns a vec of length `n_features`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_ml::prelude::*;
    /// # use scivex_core::prelude::*;
    /// let x = Tensor::from_vec(vec![1.0_f64, 0.0, 2.0, 1.0, 3.0, 0.0], vec![3, 2]).unwrap();
    /// let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
    /// let mut tree = DecisionTreeRegressor::<f64>::new(None, 1);
    /// tree.fit(&x, &y).unwrap();
    /// let imp = tree.feature_importances(2);
    /// assert_eq!(imp.len(), 2);
    /// ```
    pub fn feature_importances(&self, n_features: usize) -> Vec<T> {
        let mut counts = vec![0usize; n_features];
        if let Some(ref root) = self.root {
            count_splits(root, &mut counts);
        }
        let total: usize = counts.iter().sum();
        if total == 0 {
            return vec![T::zero(); n_features];
        }
        counts
            .iter()
            .map(|&c| T::from_usize(c) / T::from_usize(total))
            .collect()
    }
}

impl<T: Float> Predictor<T> for DecisionTreeRegressor<T> {
    fn fit(&mut self, x: &Tensor<T>, y: &Tensor<T>) -> Result<()> {
        let (n, p) = matrix_shape(x)?;
        check_y(y, n)?;
        let indices: Vec<usize> = (0..n).collect();
        let root = self.build(x.as_slice(), y.as_slice(), &indices, p, 0);
        self.root = Some(root);
        Ok(())
    }

    fn predict(&self, x: &Tensor<T>) -> Result<Tensor<T>> {
        let root = self.root.as_ref().ok_or(MlError::NotFitted)?;
        let (n, p) = matrix_shape(x)?;
        let data = x.as_slice();
        let mut out = vec![T::zero(); n];
        for i in 0..n {
            out[i] = root.predict_one(&data[i * p..(i + 1) * p]);
        }
        Ok(Tensor::from_vec(out, vec![n])?)
    }
}

// ── helpers ──

fn count_classes<T: Float>(y: &[T], indices: &[usize], classes: &[T]) -> Vec<usize> {
    let mut counts = vec![0usize; classes.len()];
    for &i in indices {
        counts[class_index(classes, y[i])] += 1;
    }
    counts
}

fn majority_class<T: Float>(y: &[T], indices: &[usize], classes: &[T]) -> T {
    let counts = count_classes(y, indices, classes);
    let best = counts
        .iter()
        .enumerate()
        .max_by_key(|&(_, &c)| c)
        .map_or(0, |(i, _)| i);
    classes[best]
}

fn mean_of<T: Float>(y: &[T], indices: &[usize]) -> T {
    if indices.is_empty() {
        return T::zero();
    }
    let sum = indices.iter().map(|&i| y[i]).fold(T::zero(), |a, b| a + b);
    sum / T::from_usize(indices.len())
}

fn mse_of<T: Float>(y: &[T], indices: &[usize]) -> T {
    let vals: Vec<T> = indices.iter().map(|&i| y[i]).collect();
    mse_impurity(&vals)
}

fn count_splits<T: Float>(node: &Node<T>, counts: &mut [usize]) {
    if let Node::Split {
        feature,
        left,
        right,
        ..
    } = node
    {
        if *feature < counts.len() {
            counts[*feature] += 1;
        }
        count_splits(left, counts);
        count_splits(right, counts);
    }
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
    fn test_tree_classifier_simple() {
        // XOR-like data with depth sufficient to learn it
        let x =
            Tensor::from_vec(vec![0.0_f64, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], vec![4, 2]).unwrap();
        let y = Tensor::from_vec(vec![0.0, 1.0, 1.0, 0.0], vec![4]).unwrap();

        let mut tree = DecisionTreeClassifier::new(Some(4), 1);
        tree.fit(&x, &y).unwrap();
        let preds = tree.predict(&x).unwrap();
        assert_eq!(preds.as_slice(), y.as_slice());
    }

    #[test]
    fn test_tree_regressor_memorises() {
        let x = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0], vec![3, 1]).unwrap();
        let y = Tensor::from_vec(vec![10.0, 20.0, 30.0], vec![3]).unwrap();

        let mut tree = DecisionTreeRegressor::new(None, 1);
        tree.fit(&x, &y).unwrap();
        let preds = tree.predict(&x).unwrap();
        for (a, b) in preds.as_slice().iter().zip(y.as_slice()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_depth_1_stump() {
        let x = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
        let y = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0], vec![4]).unwrap();
        let mut tree = DecisionTreeClassifier::new(Some(1), 1);
        tree.fit(&x, &y).unwrap();
        let preds = tree.predict(&x).unwrap();
        // A single split should separate [0,0] from [1,1]
        assert_eq!(preds.as_slice(), &[0.0, 0.0, 1.0, 1.0]);
    }
}
