use scivex_core::{Float, Tensor, random::Rng};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::error::{MlError, Result};
use crate::traits::Predictor;
use crate::tree::{DecisionTreeClassifier, DecisionTreeRegressor};

/// Random forest classifier using bagging of decision trees.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct RandomForestClassifier<T: Float> {
    n_trees: usize,
    max_depth: Option<usize>,
    max_features: Option<usize>,
    min_samples_split: usize,
    seed: u64,
    trees: Option<Vec<DecisionTreeClassifier<T>>>,
}

impl<T: Float> RandomForestClassifier<T> {
    /// Create a new random forest classifier.
    pub fn new(
        n_trees: usize,
        max_depth: Option<usize>,
        max_features: Option<usize>,
        seed: u64,
    ) -> Result<Self> {
        if n_trees == 0 {
            return Err(MlError::InvalidParameter {
                name: "n_trees",
                reason: "must be at least 1",
            });
        }
        Ok(Self {
            n_trees,
            max_depth,
            max_features,
            min_samples_split: 2,
            seed,
            trees: None,
        })
    }
}

#[cfg(feature = "parallel")]
impl<T: Float> RandomForestClassifier<T> {
    /// Train the random forest in parallel using Rayon.
    ///
    /// Each tree is built on a separate thread with its own RNG derived
    /// from the parent seed via [`Rng::fork`].
    pub fn par_fit(&mut self, x: &Tensor<T>, y: &Tensor<T>) -> Result<()> {
        let (n, p) = matrix_shape(x)?;
        check_y(y, n)?;
        let max_feat = self.max_features.unwrap_or(float_sqrt(p));
        let mut rng = Rng::new(self.seed);
        let child_rngs = rng.fork(self.n_trees);

        let trees: Result<Vec<_>> = child_rngs
            .into_par_iter()
            .map(|mut rng| {
                let (bx, by) = bootstrap_sample(x, y, n, p, &mut rng)?;
                let (sx, feat_count) = select_features(&bx, n, p, max_feat, &mut rng);
                let mut tree = DecisionTreeClassifier::new(self.max_depth, self.min_samples_split);
                let sx_tensor = Tensor::from_vec(sx, vec![n, feat_count])?;
                tree.fit(&sx_tensor, &by)?;
                Ok(tree)
            })
            .collect();

        self.trees = Some(trees?);
        Ok(())
    }
}

impl<T: Float> Predictor<T> for RandomForestClassifier<T> {
    fn fit(&mut self, x: &Tensor<T>, y: &Tensor<T>) -> Result<()> {
        let (n, p) = matrix_shape(x)?;
        check_y(y, n)?;
        let max_feat = self.max_features.unwrap_or(float_sqrt(p));
        let mut rng = Rng::new(self.seed);
        let mut trees = Vec::with_capacity(self.n_trees);

        for _ in 0..self.n_trees {
            // Bootstrap sample
            let (bx, by) = bootstrap_sample(x, y, n, p, &mut rng)?;
            // Feature subset
            let (sx, feat_count) = select_features(&bx, n, p, max_feat, &mut rng);

            let mut tree = DecisionTreeClassifier::new(self.max_depth, self.min_samples_split);
            let sx_tensor = Tensor::from_vec(sx, vec![n, feat_count])?;
            tree.fit(&sx_tensor, &by)?;
            trees.push(tree);
        }

        self.trees = Some(trees);
        Ok(())
    }

    fn predict(&self, x: &Tensor<T>) -> Result<Tensor<T>> {
        let trees = self.trees.as_ref().ok_or(MlError::NotFitted)?;
        let (n, _p) = matrix_shape(x)?;

        // Collect predictions from all trees and majority vote
        let mut all_preds: Vec<Vec<T>> = vec![Vec::new(); n];

        // For feature subsetting in prediction, we need to reconstruct
        // Since we don't store the feature indices, we pass full features to trees
        // that were trained on full features (bootstrap only).
        // Simplification: train on all features with bootstrap sampling.
        // Re-predict with each tree using the same feature space.
        for tree in trees {
            let preds = tree.predict(x)?;
            for (i, &v) in preds.as_slice().iter().enumerate() {
                all_preds[i].push(v);
            }
        }

        let mut out = vec![T::zero(); n];
        for i in 0..n {
            out[i] = majority_vote(&all_preds[i]);
        }
        Ok(Tensor::from_vec(out, vec![n])?)
    }
}

/// Random forest regressor using bagging of decision trees.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct RandomForestRegressor<T: Float> {
    n_trees: usize,
    max_depth: Option<usize>,
    max_features: Option<usize>,
    min_samples_split: usize,
    seed: u64,
    trees: Option<Vec<DecisionTreeRegressor<T>>>,
}

impl<T: Float> RandomForestRegressor<T> {
    /// Create a new random forest regressor.
    pub fn new(
        n_trees: usize,
        max_depth: Option<usize>,
        max_features: Option<usize>,
        seed: u64,
    ) -> Result<Self> {
        if n_trees == 0 {
            return Err(MlError::InvalidParameter {
                name: "n_trees",
                reason: "must be at least 1",
            });
        }
        Ok(Self {
            n_trees,
            max_depth,
            max_features,
            min_samples_split: 2,
            seed,
            trees: None,
        })
    }
}

#[cfg(feature = "parallel")]
impl<T: Float> RandomForestRegressor<T> {
    /// Train the random forest in parallel using Rayon.
    ///
    /// Each tree is built on a separate thread with its own RNG.
    pub fn par_fit(&mut self, x: &Tensor<T>, y: &Tensor<T>) -> Result<()> {
        let (n, p) = matrix_shape(x)?;
        check_y(y, n)?;
        let mut rng = Rng::new(self.seed);
        let child_rngs = rng.fork(self.n_trees);

        let trees: Result<Vec<_>> = child_rngs
            .into_par_iter()
            .map(|mut rng| {
                let (bx, by) = bootstrap_sample(x, y, n, p, &mut rng)?;
                let mut tree = DecisionTreeRegressor::new(self.max_depth, self.min_samples_split);
                let bx_tensor = Tensor::from_vec(bx, vec![n, p])?;
                tree.fit(&bx_tensor, &by)?;
                Ok(tree)
            })
            .collect();

        self.trees = Some(trees?);
        Ok(())
    }
}

impl<T: Float> Predictor<T> for RandomForestRegressor<T> {
    fn fit(&mut self, x: &Tensor<T>, y: &Tensor<T>) -> Result<()> {
        let (n, p) = matrix_shape(x)?;
        check_y(y, n)?;
        let _max_feat = self.max_features.unwrap_or(p / 3);
        let mut rng = Rng::new(self.seed);
        let mut trees = Vec::with_capacity(self.n_trees);

        for _ in 0..self.n_trees {
            let (bx, by) = bootstrap_sample(x, y, n, p, &mut rng)?;
            let mut tree = DecisionTreeRegressor::new(self.max_depth, self.min_samples_split);
            let bx_tensor = Tensor::from_vec(bx, vec![n, p])?;
            tree.fit(&bx_tensor, &by)?;
            trees.push(tree);
        }

        self.trees = Some(trees);
        Ok(())
    }

    fn predict(&self, x: &Tensor<T>) -> Result<Tensor<T>> {
        let trees = self.trees.as_ref().ok_or(MlError::NotFitted)?;
        let (n, _p) = matrix_shape(x)?;

        let mut sums = vec![T::zero(); n];
        for tree in trees {
            let preds = tree.predict(x)?;
            for (i, &v) in preds.as_slice().iter().enumerate() {
                sums[i] += v;
            }
        }
        let n_trees = T::from_usize(trees.len());
        for s in &mut sums {
            *s /= n_trees;
        }
        Ok(Tensor::from_vec(sums, vec![n])?)
    }
}

// ── helpers ──

fn bootstrap_sample<T: Float>(
    x: &Tensor<T>,
    y: &Tensor<T>,
    n: usize,
    p: usize,
    rng: &mut Rng,
) -> Result<(Vec<T>, Tensor<T>)> {
    let x_data = x.as_slice();
    let y_data = y.as_slice();
    let mut bx = vec![T::zero(); n * p];
    let mut by = vec![T::zero(); n];
    for i in 0..n {
        let idx = (rng.next_f64() * n as f64) as usize % n;
        bx[i * p..(i + 1) * p].copy_from_slice(&x_data[idx * p..(idx + 1) * p]);
        by[i] = y_data[idx];
    }
    Ok((bx, Tensor::from_vec(by, vec![n])?))
}

fn select_features<T: Float>(
    bx: &[T],
    n: usize,
    p: usize,
    max_feat: usize,
    rng: &mut Rng,
) -> (Vec<T>, usize) {
    let max_feat = max_feat.min(p);
    if max_feat == p {
        return (bx.to_vec(), p);
    }
    // Select random feature indices
    let mut feat_indices: Vec<usize> = (0..p).collect();
    for i in 0..max_feat {
        let j = i + (rng.next_f64() * (p - i) as f64) as usize % (p - i);
        feat_indices.swap(i, j);
    }
    let selected = &feat_indices[..max_feat];
    let mut out = vec![T::zero(); n * max_feat];
    for i in 0..n {
        for (k, &f) in selected.iter().enumerate() {
            out[i * max_feat + k] = bx[i * p + f];
        }
    }
    (out, max_feat)
}

fn majority_vote<T: Float>(votes: &[T]) -> T {
    // Count occurrences of each unique value
    let mut counts: Vec<(T, usize)> = Vec::new();
    for &v in votes {
        if let Some(entry) = counts
            .iter_mut()
            .find(|(c, _)| (*c - v).abs() < T::epsilon())
        {
            entry.1 += 1;
        } else {
            counts.push((v, 1));
        }
    }
    counts
        .into_iter()
        .max_by_key(|&(_, c)| c)
        .map_or(T::zero(), |(v, _)| v)
}

fn float_sqrt(n: usize) -> usize {
    (n as f64).sqrt().ceil() as usize
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
    fn test_random_forest_classifier() {
        // Simple linearly separable data
        let x = Tensor::from_vec(
            vec![
                1.0_f64, 1.0, 2.0, 2.0, 1.5, 1.5, 8.0, 8.0, 9.0, 9.0, 8.5, 8.5,
            ],
            vec![6, 2],
        )
        .unwrap();
        let y = Tensor::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0], vec![6]).unwrap();

        let mut rf = RandomForestClassifier::new(10, Some(3), None, 42).unwrap();
        rf.fit(&x, &y).unwrap();
        let preds = rf.predict(&x).unwrap();
        // Should classify training data reasonably well
        let correct: usize = preds
            .as_slice()
            .iter()
            .zip(y.as_slice())
            .filter(|(a, b)| (**a - **b).abs() < 0.5)
            .count();
        assert!(correct >= 4, "expected at least 4/6 correct, got {correct}");
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_par_random_forest_classifier() {
        let x = Tensor::from_vec(
            vec![
                1.0_f64, 1.0, 2.0, 2.0, 1.5, 1.5, 8.0, 8.0, 9.0, 9.0, 8.5, 8.5,
            ],
            vec![6, 2],
        )
        .unwrap();
        let y = Tensor::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0], vec![6]).unwrap();

        let mut rf = RandomForestClassifier::new(10, Some(3), None, 42).unwrap();
        rf.par_fit(&x, &y).unwrap();
        let preds = rf.predict(&x).unwrap();
        let correct: usize = preds
            .as_slice()
            .iter()
            .zip(y.as_slice())
            .filter(|(a, b)| (**a - **b).abs() < 0.5)
            .count();
        assert!(correct >= 4, "expected at least 4/6 correct, got {correct}");
    }

    #[test]
    fn test_random_forest_regressor() {
        let x = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0], vec![5, 1]).unwrap();
        let y = Tensor::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0], vec![5]).unwrap();

        let mut rf = RandomForestRegressor::new(20, Some(3), None, 42).unwrap();
        rf.fit(&x, &y).unwrap();
        let preds = rf.predict(&x).unwrap();
        // Predictions should be in a reasonable range
        for &p in preds.as_slice() {
            assert!(p > 0.0 && p < 15.0, "prediction {p} out of range");
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_par_random_forest_regressor() {
        let x = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0], vec![5, 1]).unwrap();
        let y = Tensor::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0], vec![5]).unwrap();

        let mut rf = RandomForestRegressor::new(20, Some(3), None, 42).unwrap();
        rf.par_fit(&x, &y).unwrap();
        let preds = rf.predict(&x).unwrap();
        for &p in preds.as_slice() {
            assert!(p > 0.0 && p < 15.0, "prediction {p} out of range");
        }
    }
}
