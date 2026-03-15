use std::collections::BinaryHeap;

use scivex_core::{Float, Tensor};

use crate::error::{MlError, Result};
use crate::traits::Predictor;

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

// ── Bin Mapper ──

/// Discretises continuous features into integer bins via quantile-based edges.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct BinMapper<T: Float> {
    bin_edges: Vec<Vec<T>>,
    n_bins_per_feature: Vec<usize>,
    max_bins: usize,
}

impl<T: Float> BinMapper<T> {
    fn fit(x_data: &[T], n: usize, p: usize, max_bins: usize) -> Self {
        let mut bin_edges = Vec::with_capacity(p);
        let mut n_bins_per_feature = Vec::with_capacity(p);

        for feat in 0..p {
            let mut vals: Vec<T> = (0..n).map(|i| x_data[i * p + feat]).collect();
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let mut unique: Vec<T> = Vec::new();
            for &v in &vals {
                if let Some(&last) = unique.last() {
                    if (v - last).abs() > T::epsilon() {
                        unique.push(v);
                    }
                } else {
                    unique.push(v);
                }
            }

            let n_unique = unique.len();
            if n_unique <= max_bins {
                let mut edges = Vec::with_capacity(n_unique + 1);
                edges.push(unique[0] - T::one());
                for w in unique.windows(2) {
                    edges.push((w[0] + w[1]) / T::from_usize(2));
                }
                edges.push(unique[n_unique - 1] + T::one());
                let nb = edges.len() - 1;
                n_bins_per_feature.push(nb);
                bin_edges.push(edges);
            } else {
                let mut edges = Vec::with_capacity(max_bins + 1);
                edges.push(unique[0] - T::one());
                for b in 1..max_bins {
                    let idx = b * n_unique / max_bins;
                    let idx = idx.min(n_unique - 1);
                    edges.push(unique[idx]);
                }
                edges.push(unique[n_unique - 1] + T::one());
                edges.dedup_by(|a, b| (*a - *b).abs() <= T::epsilon());
                let nb = edges.len() - 1;
                n_bins_per_feature.push(nb);
                bin_edges.push(edges);
            }
        }

        Self {
            bin_edges,
            n_bins_per_feature,
            max_bins,
        }
    }

    #[inline]
    fn bin_value(&self, feat: usize, val: T) -> u8 {
        let edges = &self.bin_edges[feat];
        let mut lo: usize = 0;
        let mut hi: usize = edges.len();
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if edges[mid] <= val {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        let bin = if lo == 0 { 0 } else { lo - 1 };
        let max_bin = self.n_bins_per_feature[feat].saturating_sub(1);
        bin.min(max_bin) as u8
    }

    /// Column-major binned representation: `binned[feat * n + sample]`.
    fn transform(&self, x_data: &[T], n: usize, p: usize) -> Vec<u8> {
        let mut binned = vec![0u8; n * p];
        for feat in 0..p {
            for i in 0..n {
                binned[feat * n + i] = self.bin_value(feat, x_data[i * p + feat]);
            }
        }
        binned
    }

    fn threshold_value(&self, feat: usize, bin: usize) -> T {
        let edges = &self.bin_edges[feat];
        let idx = (bin + 1).min(edges.len() - 1);
        edges[idx]
    }
}

// ── Feature Histogram ──

#[derive(Debug, Clone)]
struct FeatureHistogram<T: Float> {
    sum_gradients: Vec<T>,
    sum_hessians: Vec<T>,
    counts: Vec<u32>,
}

impl<T: Float> FeatureHistogram<T> {
    fn new(n_bins: usize) -> Self {
        Self {
            sum_gradients: vec![T::zero(); n_bins],
            sum_hessians: vec![T::zero(); n_bins],
            counts: vec![0; n_bins],
        }
    }

    fn subtract(&self, child: &Self) -> Self {
        let n = self.sum_gradients.len();
        let mut result = Self::new(n);
        for i in 0..n {
            result.sum_gradients[i] = self.sum_gradients[i] - child.sum_gradients[i];
            result.sum_hessians[i] = self.sum_hessians[i] - child.sum_hessians[i];
            result.counts[i] = self.counts[i] - child.counts[i];
        }
        result
    }
}

// ── Hist Node ──

#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
enum HistNode<T: Float> {
    Leaf {
        value: T,
    },
    Split {
        feature: usize,
        threshold: T,
        left: Box<HistNode<T>>,
        right: Box<HistNode<T>>,
    },
}

impl<T: Float> HistNode<T> {
    fn predict_one(&self, row: &[T]) -> T {
        match self {
            Self::Leaf { value } => *value,
            Self::Split {
                feature,
                threshold,
                left,
                right,
                ..
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

// ── Split Candidate ──

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct SplitCandidate<T: Float> {
    feature: usize,
    bin: usize,
    gain: T,
    sum_gradient_left: T,
    sum_hessian_left: T,
    n_left: u32,
}

/// Entry in the `BinaryHeap` for leaf-wise growth: `(split_candidate, leaf_id)`.
#[derive(Debug)]
struct HeapEntry<T: Float> {
    split: SplitCandidate<T>,
    leaf_id: usize,
}

impl<T: Float> PartialEq for HeapEntry<T> {
    fn eq(&self, other: &Self) -> bool {
        self.split.gain == other.split.gain
    }
}

impl<T: Float> Eq for HeapEntry<T> {}

impl<T: Float> PartialOrd for HeapEntry<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Float> Ord for HeapEntry<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.split
            .gain
            .partial_cmp(&other.split.gain)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Metadata for a leaf node pending potential splitting.
#[derive(Debug)]
struct PendingLeaf<T: Float> {
    sample_indices: Vec<usize>,
    sum_gradient: T,
    sum_hessian: T,
    depth: usize,
    parent_histograms: Option<Vec<FeatureHistogram<T>>>,
}

// ── Importance Type ──

/// Feature importance metric.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImportanceType {
    /// Total gain contributed by splits on the feature.
    Gain,
    /// Number of times the feature was used for a split.
    SplitCount,
}

// ── Histogram Gradient Boosting Regressor ──

/// Histogram-based gradient boosting regressor (LightGBM-style).
///
/// Uses quantile-based feature binning, leaf-wise (best-first) tree growth,
/// and the histogram subtraction trick for efficient split finding.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct HistGradientBoostingRegressor<T: Float> {
    n_estimators: usize,
    learning_rate: T,
    max_leaf_nodes: usize,
    max_depth: Option<usize>,
    min_samples_leaf: usize,
    max_bins: usize,
    l2_regularization: T,
    min_gain_to_split: T,
    subsample: f64,
    seed: u64,
    early_stopping_rounds: Option<usize>,
    validation_fraction: f64,
    // Fitted state
    trees: Option<Vec<HistNode<T>>>,
    bin_mapper: Option<BinMapper<T>>,
    baseline_prediction: T,
    feature_importances_gain: Option<Vec<T>>,
    feature_importances_split: Option<Vec<usize>>,
    n_features: usize,
}

impl<T: Float> HistGradientBoostingRegressor<T> {
    /// Create a new histogram-based gradient boosting regressor.
    ///
    /// # Errors
    ///
    /// Returns `MlError::InvalidParameter` if any parameter is out of range.
    pub fn new(
        n_estimators: usize,
        learning_rate: f64,
        max_leaf_nodes: usize,
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
        if max_leaf_nodes < 2 {
            return Err(MlError::InvalidParameter {
                name: "max_leaf_nodes",
                reason: "must be at least 2",
            });
        }
        Ok(Self {
            n_estimators,
            learning_rate: T::from_f64(learning_rate),
            max_leaf_nodes,
            max_depth: None,
            min_samples_leaf: 1,
            max_bins: 256,
            l2_regularization: T::zero(),
            min_gain_to_split: T::zero(),
            subsample: 1.0,
            seed: 42,
            early_stopping_rounds: None,
            validation_fraction: 0.1,
            trees: None,
            bin_mapper: None,
            baseline_prediction: T::zero(),
            feature_importances_gain: None,
            feature_importances_split: None,
            n_features: 0,
        })
    }

    /// Set the maximum tree depth (default: unlimited).
    pub fn set_max_depth(&mut self, depth: usize) -> &mut Self {
        self.max_depth = Some(depth);
        self
    }

    /// Set the minimum number of samples in a leaf (default: 1).
    pub fn set_min_samples_leaf(&mut self, min: usize) -> &mut Self {
        self.min_samples_leaf = min.max(1);
        self
    }

    /// Set the maximum number of bins for feature discretisation (default: 256).
    pub fn set_max_bins(&mut self, bins: usize) -> &mut Self {
        self.max_bins = bins.clamp(2, 255);
        self
    }

    /// Set the L2 regularisation parameter (default: 0.0).
    pub fn set_l2_regularization(&mut self, lambda: f64) -> &mut Self {
        self.l2_regularization = T::from_f64(lambda.max(0.0));
        self
    }

    /// Set the minimum gain required to make a split (default: 0.0).
    pub fn set_min_gain_to_split(&mut self, min_gain: f64) -> &mut Self {
        self.min_gain_to_split = T::from_f64(min_gain.max(0.0));
        self
    }

    /// Set the subsample fraction for stochastic gradient boosting (default: 1.0).
    pub fn set_subsample(&mut self, frac: f64) -> &mut Self {
        self.subsample = frac.clamp(0.1, 1.0);
        self
    }

    /// Set the random seed for subsampling.
    pub fn set_seed(&mut self, seed: u64) -> &mut Self {
        self.seed = seed;
        self
    }

    /// Set the number of rounds with no improvement to trigger early stopping.
    pub fn set_early_stopping_rounds(&mut self, rounds: usize) -> &mut Self {
        self.early_stopping_rounds = Some(rounds);
        self
    }

    /// Set the fraction of training data held out for early stopping validation
    /// (default: 0.1).
    pub fn set_validation_fraction(&mut self, frac: f64) -> &mut Self {
        self.validation_fraction = frac.clamp(0.05, 0.5);
        self
    }

    /// Return feature importances as a vector of length `n_features`.
    ///
    /// # Errors
    ///
    /// Returns `MlError::NotFitted` if the model has not been fitted.
    pub fn feature_importances(&self, importance_type: ImportanceType) -> Result<Vec<T>> {
        if self.trees.is_none() {
            return Err(MlError::NotFitted);
        }
        match importance_type {
            ImportanceType::Gain => {
                let gains = self
                    .feature_importances_gain
                    .as_ref()
                    .ok_or(MlError::NotFitted)?;
                let total: T = gains.iter().copied().fold(T::zero(), |a, b| a + b);
                if total > T::zero() {
                    Ok(gains.iter().map(|&g| g / total).collect())
                } else {
                    Ok(vec![T::zero(); self.n_features])
                }
            }
            ImportanceType::SplitCount => {
                let splits = self
                    .feature_importances_split
                    .as_ref()
                    .ok_or(MlError::NotFitted)?;
                let total: usize = splits.iter().sum();
                if total > 0 {
                    Ok(splits
                        .iter()
                        .map(|&s| T::from_usize(s) / T::from_usize(total))
                        .collect())
                } else {
                    Ok(vec![T::zero(); self.n_features])
                }
            }
        }
    }

    /// Return the number of trees actually fitted (may be less than `n_estimators`
    /// if early stopping triggered).
    pub fn n_estimators_fitted(&self) -> usize {
        self.trees.as_ref().map_or(0, Vec::len)
    }
}

// ── Tree building ──

/// Build histograms for all features given a set of sample indices.
fn build_histograms<T: Float>(
    binned: &[u8],
    gradients: &[T],
    hessians: &[T],
    n: usize,
    p: usize,
    bin_mapper: &BinMapper<T>,
    indices: &[usize],
) -> Vec<FeatureHistogram<T>> {
    let mut histograms: Vec<FeatureHistogram<T>> = (0..p)
        .map(|feat| FeatureHistogram::new(bin_mapper.n_bins_per_feature[feat]))
        .collect();

    for &idx in indices {
        let g = gradients[idx];
        let h = hessians[idx];
        for feat in 0..p {
            let bin = binned[feat * n + idx] as usize;
            let hist = &mut histograms[feat];
            if bin < hist.sum_gradients.len() {
                hist.sum_gradients[bin] += g;
                hist.sum_hessians[bin] += h;
                hist.counts[bin] += 1;
            }
        }
    }

    histograms
}

/// Find the best split across all features from precomputed histograms.
fn find_best_split<T: Float>(
    histograms: &[FeatureHistogram<T>],
    p: usize,
    bin_mapper: &BinMapper<T>,
    lambda: T,
    min_gain_to_split: T,
    min_samples_leaf: usize,
) -> Option<SplitCandidate<T>> {
    let mut best: Option<SplitCandidate<T>> = None;
    let half = T::from_f64(0.5);

    for (feat, hist) in histograms.iter().enumerate().take(p) {
        let n_bins = bin_mapper.n_bins_per_feature[feat];

        let total_g: T = hist.sum_gradients.iter().copied().fold(T::zero(), |a, b| a + b);
        let total_h: T = hist.sum_hessians.iter().copied().fold(T::zero(), |a, b| a + b);
        let total_count: u32 = hist.counts.iter().sum();

        let mut cum_g = T::zero();
        let mut cum_h = T::zero();
        let mut cum_count: u32 = 0;

        for bin in 0..n_bins.saturating_sub(1) {
            cum_g += hist.sum_gradients[bin];
            cum_h += hist.sum_hessians[bin];
            cum_count += hist.counts[bin];

            let right_count = total_count - cum_count;
            if (cum_count as usize) < min_samples_leaf
                || (right_count as usize) < min_samples_leaf
            {
                continue;
            }

            let right_g = total_g - cum_g;
            let right_h = total_h - cum_h;

            let gain = half
                * (cum_g * cum_g / (cum_h + lambda)
                    + right_g * right_g / (right_h + lambda)
                    - total_g * total_g / (total_h + lambda));

            if gain > min_gain_to_split {
                let is_better = best.as_ref().is_none_or(|b| gain > b.gain);
                if is_better {
                    best = Some(SplitCandidate {
                        feature: feat,
                        bin,
                        gain,
                        sum_gradient_left: cum_g,
                        sum_hessian_left: cum_h,
                        n_left: cum_count,
                    });
                }
            }
        }
    }

    best
}

/// Recursively assemble the tree from the flat `tree_nodes` / `leaves` vectors.
fn assemble_tree<T: Float>(
    node_id: usize,
    split_info: &[(usize, usize, usize, T)], // (node_id, feature, bin, threshold)
    child_map: &[(usize, usize)],             // (parent_id, first_child_id)
    leaves: &[PendingLeaf<T>],
    lambda: T,
) -> HistNode<T> {
    // Check if this node was split.
    for (map_idx, &(parent, children_start)) in child_map.iter().enumerate() {
        if parent == node_id {
            let (_, feature, _bin, threshold) = split_info[map_idx];
            let left = assemble_tree(children_start, split_info, child_map, leaves, lambda);
            let right =
                assemble_tree(children_start + 1, split_info, child_map, leaves, lambda);
            return HistNode::Split {
                feature,
                threshold,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
    }
    // Leaf node.
    let g = leaves[node_id].sum_gradient;
    let h = leaves[node_id].sum_hessian;
    HistNode::Leaf {
        value: -g / (h + lambda),
    }
}

/// Build a single histogram-based tree using leaf-wise (best-first) growth.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn build_hist_tree<T: Float>(
    binned: &[u8],
    gradients: &[T],
    hessians: &[T],
    n: usize,
    p: usize,
    bin_mapper: &BinMapper<T>,
    max_leaf_nodes: usize,
    max_depth: Option<usize>,
    min_samples_leaf: usize,
    lambda: T,
    min_gain_to_split: T,
    sample_indices: &[usize],
    gains_acc: &mut [T],
    splits_acc: &mut [usize],
) -> HistNode<T> {
    // Compute total gradient/hessian for the root.
    let mut total_grad = T::zero();
    let mut total_hess = T::zero();
    for &idx in sample_indices {
        total_grad += gradients[idx];
        total_hess += hessians[idx];
    }

    if sample_indices.len() < 2 * min_samples_leaf {
        return HistNode::Leaf {
            value: -total_grad / (total_hess + lambda),
        };
    }

    let mut leaves: Vec<PendingLeaf<T>> = Vec::new();
    // Track which nodes have been split: (node_id, feature, bin, threshold).
    let mut split_info: Vec<(usize, usize, usize, T)> = Vec::new();
    // Map from split index to (parent_id, first_child_id).
    let mut child_map: Vec<(usize, usize)> = Vec::new();

    // Push root leaf.
    leaves.push(PendingLeaf {
        sample_indices: sample_indices.to_vec(),
        sum_gradient: total_grad,
        sum_hessian: total_hess,
        depth: 0,
        parent_histograms: None,
    });

    let mut heap: BinaryHeap<HeapEntry<T>> = BinaryHeap::new();

    // Build histograms for root and find best split.
    let root_histograms =
        build_histograms(binned, gradients, hessians, n, p, bin_mapper, sample_indices);
    if let Some(best) =
        find_best_split(&root_histograms, p, bin_mapper, lambda, min_gain_to_split, min_samples_leaf)
    {
        heap.push(HeapEntry {
            split: best,
            leaf_id: 0,
        });
    }
    leaves[0].parent_histograms = Some(root_histograms);

    let mut n_leaves: usize = 1;

    while let Some(entry) = heap.pop() {
        if n_leaves >= max_leaf_nodes {
            break;
        }

        let leaf_id = entry.leaf_id;
        let split = entry.split;

        let at_max_depth = if let Some(md) = max_depth {
            leaves[leaf_id].depth >= md
        } else {
            false
        };
        if at_max_depth {
            continue;
        }

        // Partition samples.
        let feat = split.feature;
        let bin_thresh = split.bin as u8;
        let leaf_indices = std::mem::take(&mut leaves[leaf_id].sample_indices);

        let mut left_indices = Vec::new();
        let mut right_indices = Vec::new();
        for &idx in &leaf_indices {
            if binned[feat * n + idx] <= bin_thresh {
                left_indices.push(idx);
            } else {
                right_indices.push(idx);
            }
        }

        if left_indices.len() < min_samples_leaf || right_indices.len() < min_samples_leaf {
            leaves[leaf_id].sample_indices = leaf_indices;
            continue;
        }

        let sum_grad_right = leaves[leaf_id].sum_gradient - split.sum_gradient_left;
        let sum_hess_right = leaves[leaf_id].sum_hessian - split.sum_hessian_left;
        let depth = leaves[leaf_id].depth;

        // Record importance.
        gains_acc[feat] += split.gain;
        splits_acc[feat] += 1;

        // Subtraction trick: build histogram for the smaller child,
        // subtract from parent to get the larger child's histogram.
        let parent_hists = leaves[leaf_id].parent_histograms.take();
        let left_is_smaller = left_indices.len() <= right_indices.len();
        let (smaller_indices, larger_indices) = if left_is_smaller {
            (&left_indices, &right_indices)
        } else {
            (&right_indices, &left_indices)
        };

        let smaller_hists =
            build_histograms(binned, gradients, hessians, n, p, bin_mapper, smaller_indices);
        let larger_hists = if let Some(ref ph) = parent_hists {
            ph.iter()
                .zip(smaller_hists.iter())
                .map(|(par, small)| par.subtract(small))
                .collect::<Vec<_>>()
        } else {
            build_histograms(binned, gradients, hessians, n, p, bin_mapper, larger_indices)
        };

        let (left_hists, right_hists) = if left_is_smaller {
            (smaller_hists, larger_hists)
        } else {
            (larger_hists, smaller_hists)
        };

        // Record the split.
        let threshold = bin_mapper.threshold_value(feat, split.bin);
        let left_id = leaves.len();
        let right_id = left_id + 1;

        split_info.push((leaf_id, feat, split.bin, threshold));
        child_map.push((leaf_id, left_id));

        // Create left and right leaf entries.
        leaves.push(PendingLeaf {
            sample_indices: left_indices,
            sum_gradient: split.sum_gradient_left,
            sum_hessian: split.sum_hessian_left,
            depth: depth + 1,
            parent_histograms: Some(left_hists),
        });

        leaves.push(PendingLeaf {
            sample_indices: right_indices,
            sum_gradient: sum_grad_right,
            sum_hessian: sum_hess_right,
            depth: depth + 1,
            parent_histograms: Some(right_hists),
        });

        // Find best splits for children.
        let can_split_depth = max_depth.is_none_or(|md| depth + 1 < md);
        if can_split_depth {
            let left_hist_ref = leaves[left_id].parent_histograms.as_ref().unwrap();
            if let Some(best_left) = find_best_split(
                left_hist_ref,
                p,
                bin_mapper,
                lambda,
                min_gain_to_split,
                min_samples_leaf,
            ) {
                heap.push(HeapEntry {
                    split: best_left,
                    leaf_id: left_id,
                });
            }

            let right_hist_ref = leaves[right_id].parent_histograms.as_ref().unwrap();
            if let Some(best_right) = find_best_split(
                right_hist_ref,
                p,
                bin_mapper,
                lambda,
                min_gain_to_split,
                min_samples_leaf,
            ) {
                heap.push(HeapEntry {
                    split: best_right,
                    leaf_id: right_id,
                });
            }
        }

        n_leaves += 1;
    }

    if split_info.is_empty() {
        return HistNode::Leaf {
            value: -total_grad / (total_hess + lambda),
        };
    }

    assemble_tree(0, &split_info, &child_map, &leaves, lambda)
}

impl<T: Float> Predictor<T> for HistGradientBoostingRegressor<T> {
    #[allow(clippy::too_many_lines)]
    fn fit(&mut self, x: &Tensor<T>, y: &Tensor<T>) -> Result<()> {
        let (n, p) = matrix_shape(x)?;
        check_y(y, n)?;
        let x_data = x.as_slice();
        let y_data = y.as_slice();
        self.n_features = p;

        let mut rng = scivex_core::random::Rng::new(self.seed);

        // Split into train/validation if early stopping enabled.
        let (train_indices, val_indices) = if self.early_stopping_rounds.is_some() {
            let val_n = (self.validation_fraction * n as f64).ceil() as usize;
            let val_n = val_n.max(1).min(n - 1);
            let mut all_indices: Vec<usize> = (0..n).collect();
            for i in (1..all_indices.len()).rev() {
                let j = (rng.next_f64() * (i + 1) as f64) as usize % (i + 1);
                all_indices.swap(i, j);
            }
            let val_idx = all_indices[..val_n].to_vec();
            let train_idx = all_indices[val_n..].to_vec();
            (train_idx, val_idx)
        } else {
            ((0..n).collect(), Vec::new())
        };

        // Build bin mapper.
        let bin_mapper = BinMapper::fit(x_data, n, p, self.max_bins);
        let binned = bin_mapper.transform(x_data, n, p);

        // Baseline prediction: mean of training y.
        let train_n = train_indices.len();
        let mean_y: T = train_indices
            .iter()
            .map(|&i| y_data[i])
            .fold(T::zero(), |a, b| a + b)
            / T::from_usize(train_n);
        self.baseline_prediction = mean_y;

        // Current predictions for ALL samples.
        let mut predictions = vec![mean_y; n];

        let mut trees: Vec<HistNode<T>> = Vec::with_capacity(self.n_estimators);
        let mut imp_gains = vec![T::zero(); p];
        let mut imp_splits = vec![0_usize; p];

        let mut best_val_loss = T::from_f64(f64::MAX);
        let mut rounds_without_improvement: usize = 0;

        for _round in 0..self.n_estimators {
            // Compute gradients and hessians (MSE loss).
            let mut gradients = vec![T::zero(); n];
            let mut hessians = vec![T::one(); n];
            for &i in &train_indices {
                gradients[i] = predictions[i] - y_data[i];
                hessians[i] = T::one();
            }

            // Subsample training indices.
            let round_indices = if self.subsample < 1.0 {
                let sub_n = (self.subsample * train_n as f64).ceil() as usize;
                let sub_n = sub_n.max(1).min(train_n);
                let mut sampled = Vec::with_capacity(sub_n);
                for _ in 0..sub_n {
                    let idx = (rng.next_f64() * train_n as f64) as usize % train_n;
                    sampled.push(train_indices[idx]);
                }
                sampled
            } else {
                train_indices.clone()
            };

            let tree = build_hist_tree(
                &binned,
                &gradients,
                &hessians,
                n,
                p,
                &bin_mapper,
                self.max_leaf_nodes,
                self.max_depth,
                self.min_samples_leaf,
                self.l2_regularization,
                self.min_gain_to_split,
                &round_indices,
                &mut imp_gains,
                &mut imp_splits,
            );

            // Update predictions for all samples.
            for i in 0..n {
                let row = &x_data[i * p..(i + 1) * p];
                predictions[i] += self.learning_rate * tree.predict_one(row);
            }

            trees.push(tree);

            // Early stopping check.
            #[allow(clippy::collapsible_if)]
            if let Some(patience) = self.early_stopping_rounds {
                if !val_indices.is_empty() {
                    let val_mse: T = val_indices
                        .iter()
                        .map(|&i| {
                            let diff = predictions[i] - y_data[i];
                            diff * diff
                        })
                        .fold(T::zero(), |a, b| a + b)
                        / T::from_usize(val_indices.len());
                    if val_mse < best_val_loss {
                        best_val_loss = val_mse;
                        rounds_without_improvement = 0;
                    } else {
                        rounds_without_improvement += 1;
                        if rounds_without_improvement >= patience {
                            break;
                        }
                    }
                }
            }
        }

        self.trees = Some(trees);
        self.bin_mapper = Some(bin_mapper);
        self.feature_importances_gain = Some(imp_gains);
        self.feature_importances_split = Some(imp_splits);
        Ok(())
    }

    fn predict(&self, x: &Tensor<T>) -> Result<Tensor<T>> {
        let trees = self.trees.as_ref().ok_or(MlError::NotFitted)?;
        let (n, p) = matrix_shape(x)?;
        let x_data = x.as_slice();

        let mut out = vec![self.baseline_prediction; n];
        for tree in trees {
            for i in 0..n {
                let row = &x_data[i * p..(i + 1) * p];
                out[i] += self.learning_rate * tree.predict_one(row);
            }
        }

        Tensor::from_vec(out, vec![n]).map_err(MlError::from)
    }
}

// ── Histogram Gradient Boosting Classifier ──

/// Histogram-based gradient boosting classifier for binary classification.
///
/// Uses log-loss (binary cross-entropy) with histogram-based tree growing.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct HistGradientBoostingClassifier<T: Float> {
    n_estimators: usize,
    learning_rate: T,
    max_leaf_nodes: usize,
    max_depth: Option<usize>,
    min_samples_leaf: usize,
    max_bins: usize,
    l2_regularization: T,
    min_gain_to_split: T,
    subsample: f64,
    seed: u64,
    early_stopping_rounds: Option<usize>,
    validation_fraction: f64,
    // Fitted state
    trees: Option<Vec<HistNode<T>>>,
    bin_mapper: Option<BinMapper<T>>,
    baseline_prediction: T,
    classes: Option<Vec<T>>,
    n_features: usize,
}

impl<T: Float> HistGradientBoostingClassifier<T> {
    /// Create a new histogram-based gradient boosting classifier.
    ///
    /// # Errors
    ///
    /// Returns `MlError::InvalidParameter` if any parameter is out of range.
    pub fn new(
        n_estimators: usize,
        learning_rate: f64,
        max_leaf_nodes: usize,
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
        if max_leaf_nodes < 2 {
            return Err(MlError::InvalidParameter {
                name: "max_leaf_nodes",
                reason: "must be at least 2",
            });
        }
        Ok(Self {
            n_estimators,
            learning_rate: T::from_f64(learning_rate),
            max_leaf_nodes,
            max_depth: None,
            min_samples_leaf: 1,
            max_bins: 256,
            l2_regularization: T::zero(),
            min_gain_to_split: T::zero(),
            subsample: 1.0,
            seed: 42,
            early_stopping_rounds: None,
            validation_fraction: 0.1,
            trees: None,
            bin_mapper: None,
            baseline_prediction: T::zero(),
            classes: None,
            n_features: 0,
        })
    }

    /// Set the maximum tree depth.
    pub fn set_max_depth(&mut self, depth: usize) -> &mut Self {
        self.max_depth = Some(depth);
        self
    }

    /// Set the minimum number of samples in a leaf.
    pub fn set_min_samples_leaf(&mut self, min: usize) -> &mut Self {
        self.min_samples_leaf = min.max(1);
        self
    }

    /// Set the maximum number of bins.
    pub fn set_max_bins(&mut self, bins: usize) -> &mut Self {
        self.max_bins = bins.clamp(2, 255);
        self
    }

    /// Set the L2 regularisation parameter.
    pub fn set_l2_regularization(&mut self, lambda: f64) -> &mut Self {
        self.l2_regularization = T::from_f64(lambda.max(0.0));
        self
    }

    /// Set the minimum gain required to split.
    pub fn set_min_gain_to_split(&mut self, min_gain: f64) -> &mut Self {
        self.min_gain_to_split = T::from_f64(min_gain.max(0.0));
        self
    }

    /// Set the subsample fraction.
    pub fn set_subsample(&mut self, frac: f64) -> &mut Self {
        self.subsample = frac.clamp(0.1, 1.0);
        self
    }

    /// Set the random seed.
    pub fn set_seed(&mut self, seed: u64) -> &mut Self {
        self.seed = seed;
        self
    }

    /// Set early stopping patience.
    pub fn set_early_stopping_rounds(&mut self, rounds: usize) -> &mut Self {
        self.early_stopping_rounds = Some(rounds);
        self
    }

    /// Set the validation fraction for early stopping.
    pub fn set_validation_fraction(&mut self, frac: f64) -> &mut Self {
        self.validation_fraction = frac.clamp(0.05, 0.5);
        self
    }

    /// Predict class probabilities, shape `[n_samples, 2]`.
    ///
    /// # Errors
    ///
    /// Returns `MlError::NotFitted` if the model has not been fitted.
    pub fn predict_proba(&self, x: &Tensor<T>) -> Result<Tensor<T>> {
        let trees = self.trees.as_ref().ok_or(MlError::NotFitted)?;
        let (n, p) = matrix_shape(x)?;
        let x_data = x.as_slice();

        let mut raw_preds = vec![self.baseline_prediction; n];
        for tree in trees {
            for i in 0..n {
                let row = &x_data[i * p..(i + 1) * p];
                raw_preds[i] += self.learning_rate * tree.predict_one(row);
            }
        }

        let mut proba = vec![T::zero(); n * 2];
        for i in 0..n {
            let p1 = sigmoid_t(raw_preds[i]);
            proba[i * 2] = T::one() - p1;
            proba[i * 2 + 1] = p1;
        }

        Tensor::from_vec(proba, vec![n, 2]).map_err(MlError::from)
    }

    /// Return the number of trees fitted.
    pub fn n_estimators_fitted(&self) -> usize {
        self.trees.as_ref().map_or(0, Vec::len)
    }
}

/// Sigmoid function using the `Float` trait.
#[inline]
fn sigmoid_t<T: Float>(x: T) -> T {
    T::one() / (T::one() + (-x).exp())
}

impl<T: Float> Predictor<T> for HistGradientBoostingClassifier<T> {
    #[allow(clippy::too_many_lines)]
    fn fit(&mut self, x: &Tensor<T>, y: &Tensor<T>) -> Result<()> {
        let (n, p) = matrix_shape(x)?;
        check_y(y, n)?;
        let x_data = x.as_slice();
        let y_data = y.as_slice();
        self.n_features = p;

        // Discover classes (binary only).
        let mut classes: Vec<T> = Vec::new();
        for &v in y_data {
            if !classes.iter().any(|&c| (c - v).abs() < T::epsilon()) {
                classes.push(v);
            }
        }
        classes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        if classes.len() != 2 {
            return Err(MlError::InvalidParameter {
                name: "y",
                reason: "histogram gradient boosting classifier requires exactly 2 classes",
            });
        }

        // Encode y as 0/1 (positive class = classes[1]).
        let positive_class = classes[1];
        let y_binary: Vec<T> = y_data
            .iter()
            .map(|&v| {
                if (v - positive_class).abs() < T::epsilon() {
                    T::one()
                } else {
                    T::zero()
                }
            })
            .collect();

        let mut rng = scivex_core::random::Rng::new(self.seed);

        // Train/validation split.
        let (train_indices, val_indices) = if self.early_stopping_rounds.is_some() {
            let val_n = (self.validation_fraction * n as f64).ceil() as usize;
            let val_n = val_n.max(1).min(n - 1);
            let mut all_indices: Vec<usize> = (0..n).collect();
            for i in (1..all_indices.len()).rev() {
                let j = (rng.next_f64() * (i + 1) as f64) as usize % (i + 1);
                all_indices.swap(i, j);
            }
            let val_idx = all_indices[..val_n].to_vec();
            let train_idx = all_indices[val_n..].to_vec();
            (train_idx, val_idx)
        } else {
            ((0..n).collect(), Vec::new())
        };

        // Bin mapper.
        let bin_mapper = BinMapper::fit(x_data, n, p, self.max_bins);
        let binned = bin_mapper.transform(x_data, n, p);

        // Baseline: log-odds of positive class in training set.
        let train_n = train_indices.len();
        let pos_count: T = train_indices
            .iter()
            .map(|&i| y_binary[i])
            .fold(T::zero(), |a, b| a + b);
        let pos_frac = pos_count / T::from_usize(train_n);
        let eps = T::from_f64(1e-10);
        let pos_clamped = if pos_frac < eps {
            eps
        } else if pos_frac > T::one() - eps {
            T::one() - eps
        } else {
            pos_frac
        };
        let baseline = (pos_clamped / (T::one() - pos_clamped)).ln();
        self.baseline_prediction = baseline;

        let mut raw_predictions = vec![baseline; n];
        let mut trees: Vec<HistNode<T>> = Vec::with_capacity(self.n_estimators);

        let mut imp_gains = vec![T::zero(); p];
        let mut imp_splits = vec![0_usize; p];

        let mut best_val_loss = T::from_f64(f64::MAX);
        let mut rounds_without_improvement: usize = 0;

        for _round in 0..self.n_estimators {
            // Gradients and hessians for log-loss.
            let mut gradients = vec![T::zero(); n];
            let mut hessians = vec![T::zero(); n];
            let h_floor = T::from_f64(1e-10);
            for &i in &train_indices {
                let prob = sigmoid_t(raw_predictions[i]);
                gradients[i] = prob - y_binary[i];
                let h = prob * (T::one() - prob);
                hessians[i] = if h < h_floor { h_floor } else { h };
            }

            let round_indices = if self.subsample < 1.0 {
                let sub_n = (self.subsample * train_n as f64).ceil() as usize;
                let sub_n = sub_n.max(1).min(train_n);
                let mut sampled = Vec::with_capacity(sub_n);
                for _ in 0..sub_n {
                    let idx = (rng.next_f64() * train_n as f64) as usize % train_n;
                    sampled.push(train_indices[idx]);
                }
                sampled
            } else {
                train_indices.clone()
            };

            let tree = build_hist_tree(
                &binned,
                &gradients,
                &hessians,
                n,
                p,
                &bin_mapper,
                self.max_leaf_nodes,
                self.max_depth,
                self.min_samples_leaf,
                self.l2_regularization,
                self.min_gain_to_split,
                &round_indices,
                &mut imp_gains,
                &mut imp_splits,
            );

            for i in 0..n {
                let row = &x_data[i * p..(i + 1) * p];
                raw_predictions[i] += self.learning_rate * tree.predict_one(row);
            }

            trees.push(tree);

            // Early stopping.
            #[allow(clippy::collapsible_if)]
            if let Some(patience) = self.early_stopping_rounds {
                if !val_indices.is_empty() {
                    let log_loss_eps = T::from_f64(1e-15);
                    let val_loss: T = val_indices
                        .iter()
                        .map(|&i| {
                            let prob = sigmoid_t(raw_predictions[i]);
                            let prob_clamped = if prob < log_loss_eps {
                                log_loss_eps
                            } else if prob > T::one() - log_loss_eps {
                                T::one() - log_loss_eps
                            } else {
                                prob
                            };
                            -(y_binary[i] * prob_clamped.ln()
                                + (T::one() - y_binary[i]) * (T::one() - prob_clamped).ln())
                        })
                        .fold(T::zero(), |a, b| a + b)
                        / T::from_usize(val_indices.len());
                    if val_loss < best_val_loss {
                        best_val_loss = val_loss;
                        rounds_without_improvement = 0;
                    } else {
                        rounds_without_improvement += 1;
                        if rounds_without_improvement >= patience {
                            break;
                        }
                    }
                }
            }
        }

        self.trees = Some(trees);
        self.bin_mapper = Some(bin_mapper);
        self.classes = Some(classes);
        Ok(())
    }

    fn predict(&self, x: &Tensor<T>) -> Result<Tensor<T>> {
        let classes = self.classes.as_ref().ok_or(MlError::NotFitted)?;
        let proba = self.predict_proba(x)?;
        let proba_data = proba.as_slice();
        let n = proba_data.len() / 2;

        let mut out = vec![T::zero(); n];
        for i in 0..n {
            if proba_data[i * 2 + 1] > proba_data[i * 2] {
                out[i] = classes[1];
            } else {
                out[i] = classes[0];
            }
        }

        Tensor::from_vec(out, vec![n]).map_err(MlError::from)
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hist_gb_regressor_linear() {
        // y = 2*x0 + x1
        let x = Tensor::from_vec(
            vec![
                1.0_f64, 0.5, 2.0, 1.0, 3.0, 1.5, 4.0, 2.0, 5.0, 2.5, 6.0, 3.0, 7.0, 3.5, 8.0,
                4.0,
            ],
            vec![8, 2],
        )
        .unwrap();
        let y = Tensor::from_vec(
            vec![2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0],
            vec![8],
        )
        .unwrap();

        let mut model = HistGradientBoostingRegressor::new(100, 0.1, 8).unwrap();
        model.fit(&x, &y).unwrap();
        let preds = model.predict(&x).unwrap();

        let mse: f64 = preds
            .as_slice()
            .iter()
            .zip(y.as_slice())
            .map(|(&p, &t)| (p - t) * (p - t))
            .sum::<f64>()
            / 8.0;
        assert!(mse < 1.0, "MSE = {mse}, expected < 1.0");
    }

    #[test]
    fn test_hist_gb_regressor_converges() {
        let x = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0], vec![5, 1]).unwrap();
        let y = Tensor::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0], vec![5]).unwrap();

        let mut early_model = HistGradientBoostingRegressor::new(5, 0.1, 4).unwrap();
        early_model.fit(&x, &y).unwrap();

        let mut late_model = HistGradientBoostingRegressor::new(50, 0.1, 4).unwrap();
        late_model.fit(&x, &y).unwrap();

        let preds_early = early_model.predict(&x).unwrap();
        let preds_late = late_model.predict(&x).unwrap();

        let mse_early: f64 = preds_early
            .as_slice()
            .iter()
            .zip(y.as_slice())
            .map(|(&p, &t)| (p - t) * (p - t))
            .sum::<f64>()
            / 5.0;
        let mse_late: f64 = preds_late
            .as_slice()
            .iter()
            .zip(y.as_slice())
            .map(|(&p, &t)| (p - t) * (p - t))
            .sum::<f64>()
            / 5.0;

        assert!(
            mse_late <= mse_early + 0.1,
            "Training loss should decrease: early={mse_early}, late={mse_late}"
        );
    }

    #[test]
    fn test_hist_gb_regressor_not_fitted() {
        let model = HistGradientBoostingRegressor::<f64>::new(10, 0.1, 4).unwrap();
        let x = Tensor::from_vec(vec![1.0_f64, 2.0], vec![2, 1]).unwrap();
        assert!(model.predict(&x).is_err());
    }

    #[test]
    fn test_hist_gb_classifier_binary() {
        let x = Tensor::from_vec(
            vec![
                1.0_f64, 1.0, 1.5, 1.5, 2.0, 2.0, 8.0, 8.0, 9.0, 9.0, 8.5, 8.5,
            ],
            vec![6, 2],
        )
        .unwrap();
        let y = Tensor::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0], vec![6]).unwrap();

        let mut model = HistGradientBoostingClassifier::new(100, 0.1, 8).unwrap();
        model.fit(&x, &y).unwrap();
        let preds = model.predict(&x).unwrap();

        let correct: usize = preds
            .as_slice()
            .iter()
            .zip(y.as_slice())
            .filter(|(a, b)| (**a - **b).abs() < 0.5)
            .count();
        assert!(
            correct >= 4,
            "expected at least 4/6 correct, got {correct}"
        );
    }

    #[test]
    fn test_hist_gb_classifier_multiclass() {
        // Binary classifier should reject 3 classes.
        let x = Tensor::from_vec(vec![1.0_f64, 0.0, 5.0, 5.0, 0.0, 5.0], vec![3, 2]).unwrap();
        let y = Tensor::from_vec(vec![0.0, 1.0, 2.0], vec![3]).unwrap();

        let mut model = HistGradientBoostingClassifier::new(10, 0.1, 4).unwrap();
        let result = model.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_hist_gb_early_stopping() {
        let n = 40;
        let mut x_data = Vec::with_capacity(n);
        let mut y_data = Vec::with_capacity(n);
        for i in 0..n {
            let v = i as f64;
            x_data.push(v);
            y_data.push(2.0 * v + 1.0);
        }
        let x = Tensor::from_vec(x_data, vec![n, 1]).unwrap();
        let y = Tensor::from_vec(y_data, vec![n]).unwrap();

        let mut model = HistGradientBoostingRegressor::new(500, 0.1, 8).unwrap();
        model
            .set_early_stopping_rounds(5)
            .set_validation_fraction(0.2);
        model.fit(&x, &y).unwrap();

        let fitted = model.n_estimators_fitted();
        assert!(
            fitted < 500,
            "expected early stopping before 500 rounds, got {fitted}"
        );
    }

    #[test]
    fn test_hist_gb_feature_importances_gain() {
        // Feature 0 is predictive; feature 1 is constant.
        let x = Tensor::from_vec(
            vec![
                1.0_f64, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0, 6.0, 0.0, 7.0, 0.0, 8.0,
                0.0,
            ],
            vec![8, 2],
        )
        .unwrap();
        let y = Tensor::from_vec(
            vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0],
            vec![8],
        )
        .unwrap();

        let mut model = HistGradientBoostingRegressor::new(50, 0.1, 8).unwrap();
        model.fit(&x, &y).unwrap();

        let imp = model.feature_importances(ImportanceType::Gain).unwrap();
        assert_eq!(imp.len(), 2);
        assert!(imp[0] > 0.0, "feature 0 importance should be > 0");
    }

    #[test]
    fn test_hist_gb_feature_importances_split() {
        let x = Tensor::from_vec(
            vec![
                1.0_f64, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0, 6.0, 0.0, 7.0, 0.0, 8.0,
                0.0,
            ],
            vec![8, 2],
        )
        .unwrap();
        let y = Tensor::from_vec(
            vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0],
            vec![8],
        )
        .unwrap();

        let mut model = HistGradientBoostingRegressor::new(20, 0.1, 4).unwrap();
        model.fit(&x, &y).unwrap();

        let imp = model
            .feature_importances(ImportanceType::SplitCount)
            .unwrap();
        assert_eq!(imp.len(), 2);
        assert!(imp[0] > 0.0, "feature 0 split importance should be > 0");
    }

    #[test]
    fn test_hist_gb_subsample() {
        let x = Tensor::from_vec(
            vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![8, 1],
        )
        .unwrap();
        let y = Tensor::from_vec(
            vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0],
            vec![8],
        )
        .unwrap();

        let mut model = HistGradientBoostingRegressor::new(50, 0.1, 4).unwrap();
        model.set_subsample(0.5).set_seed(123);
        model.fit(&x, &y).unwrap();

        let preds = model.predict(&x).unwrap();
        for &p in preds.as_slice() {
            assert!(
                p > -5.0 && p < 25.0,
                "prediction {p} out of reasonable range"
            );
        }
    }

    #[test]
    fn test_hist_gb_max_bins() {
        let x = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();

        for bins in [4, 16, 64, 128] {
            let mut model = HistGradientBoostingRegressor::new(20, 0.1, 4).unwrap();
            model.set_max_bins(bins);
            model.fit(&x, &y).unwrap();
            let preds = model.predict(&x).unwrap();
            assert_eq!(preds.as_slice().len(), 4);
        }
    }

    #[test]
    fn test_binning_basic() {
        let x_data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let mapper = BinMapper::fit(&x_data, 10, 1, 4);
        assert!(mapper.n_bins_per_feature[0] <= 10);
        assert!(!mapper.bin_edges[0].is_empty());

        for &v in &x_data {
            let bin = mapper.bin_value(0, v);
            assert!(
                (bin as usize) < mapper.n_bins_per_feature[0],
                "bin {bin} >= n_bins {}",
                mapper.n_bins_per_feature[0]
            );
        }
    }

    #[test]
    fn test_histogram_subtraction() {
        let mut parent = FeatureHistogram::<f64>::new(4);
        parent.sum_gradients = vec![1.0, 2.0, 3.0, 4.0];
        parent.sum_hessians = vec![1.0, 1.0, 1.0, 1.0];
        parent.counts = vec![10, 20, 30, 40];

        let mut child = FeatureHistogram::<f64>::new(4);
        child.sum_gradients = vec![0.5, 1.0, 1.5, 2.0];
        child.sum_hessians = vec![0.5, 0.5, 0.5, 0.5];
        child.counts = vec![5, 10, 15, 20];

        let sibling = parent.subtract(&child);
        assert_eq!(sibling.sum_gradients, vec![0.5, 1.0, 1.5, 2.0]);
        assert_eq!(sibling.sum_hessians, vec![0.5, 0.5, 0.5, 0.5]);
        assert_eq!(sibling.counts, vec![5, 10, 15, 20]);
    }

    #[test]
    fn test_parameter_validation() {
        assert!(HistGradientBoostingRegressor::<f64>::new(0, 0.1, 4).is_err());
        assert!(HistGradientBoostingRegressor::<f64>::new(10, 0.0, 4).is_err());
        assert!(HistGradientBoostingRegressor::<f64>::new(10, 1.5, 4).is_err());
        assert!(HistGradientBoostingRegressor::<f64>::new(10, 0.1, 1).is_err());

        assert!(HistGradientBoostingClassifier::<f64>::new(0, 0.1, 4).is_err());
        assert!(HistGradientBoostingClassifier::<f64>::new(10, 0.0, 4).is_err());
        assert!(HistGradientBoostingClassifier::<f64>::new(10, 1.5, 4).is_err());
        assert!(HistGradientBoostingClassifier::<f64>::new(10, 0.1, 1).is_err());
    }

    #[test]
    fn test_hist_gb_n_estimators_fitted() {
        let x = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();

        let model = HistGradientBoostingRegressor::<f64>::new(10, 0.1, 4).unwrap();
        assert_eq!(model.n_estimators_fitted(), 0);

        let mut model = HistGradientBoostingRegressor::new(10, 0.1, 4).unwrap();
        model.fit(&x, &y).unwrap();
        assert_eq!(model.n_estimators_fitted(), 10);
    }

    #[test]
    fn test_hist_gb_single_tree() {
        let x = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
        let y = Tensor::from_vec(vec![10.0, 20.0, 30.0, 40.0], vec![4]).unwrap();

        let mut model = HistGradientBoostingRegressor::new(1, 1.0, 4).unwrap();
        model.fit(&x, &y).unwrap();
        let preds = model.predict(&x).unwrap();
        assert_eq!(preds.as_slice().len(), 4);
        assert_eq!(model.n_estimators_fitted(), 1);
    }
}
