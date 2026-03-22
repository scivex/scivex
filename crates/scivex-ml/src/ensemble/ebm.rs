//! Explainable Boosting Machine (EBM / GA²M).
//!
//! An interpretable model of the form:
//!
//! ```text
//! f(x) = intercept + Σ_j g_j(x_j) + Σ_{i<j} g_{ij}(x_i, x_j)
//! ```
//!
//! Each shape function `g_j` is learned via cyclic gradient boosting on
//! binned features. Pairwise interaction terms `g_{ij}` are optionally
//! included (GA²M).

use scivex_core::{Float, Tensor};

use crate::error::{MlError, Result};
use crate::traits::{Classifier, Predictor};

// ── Type aliases to avoid type_complexity lint ──

/// Per-feature shape function scores: `shape_fns[feature][bin]`.
type ShapeFns<T> = Vec<Vec<T>>;

/// Per-feature bin edges: `bin_edges[feature][edge]`.
type BinEdges<T> = Vec<Vec<T>>;

/// Interaction term: `(feature_i, feature_j, score_grid[bin_i][bin_j])`.
type InteractionTerm<T> = (usize, usize, Vec<Vec<T>>);

/// Residual function type alias.
type ResidualFn<T> = fn(&[T], &[T]) -> Vec<T>;

// ── EBM Regressor ──

/// Explainable Boosting Machine for regression (GA²M).
///
/// Fits an additive model with optional pairwise interactions using
/// cyclic gradient boosting on binned features. Each feature's contribution
/// is a shape function that can be inspected for interpretability.
///
/// # Examples
///
/// ```
/// # use scivex_ml::prelude::*;
/// # use scivex_core::prelude::*;
/// let x = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
/// let y = Tensor::from_vec(vec![2.0, 4.0, 6.0, 8.0], vec![4]).unwrap();
/// let mut ebm = EbmRegressor::new(100, 256, 0.1).unwrap();
/// ebm.fit(&x, &y).unwrap();
/// let preds = ebm.predict(&x).unwrap();
/// ```
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct EbmRegressor<T: Float> {
    max_rounds: usize,
    max_bins: usize,
    learning_rate: T,
    max_interaction_bins: usize,
    include_interactions: bool,
    // Fitted state
    intercept: Option<T>,
    shape_fns: Option<ShapeFns<T>>,
    bin_edges: Option<BinEdges<T>>,
    interactions: Option<Vec<InteractionTerm<T>>>,
}

impl<T: Float> EbmRegressor<T> {
    /// Create a new EBM regressor.
    ///
    /// - `max_rounds`: number of cyclic boosting rounds
    /// - `max_bins`: maximum number of bins per feature
    /// - `learning_rate`: shrinkage applied to each gradient step
    pub fn new(max_rounds: usize, max_bins: usize, learning_rate: f64) -> Result<Self> {
        validate_ebm_params(max_rounds, max_bins, learning_rate)?;
        Ok(Self {
            max_rounds,
            max_bins,
            learning_rate: T::from_f64(learning_rate),
            max_interaction_bins: 32,
            include_interactions: false,
            intercept: None,
            shape_fns: None,
            bin_edges: None,
            interactions: None,
        })
    }

    /// Enable or disable pairwise interaction terms.
    pub fn set_interactions(&mut self, enable: bool) -> &mut Self {
        self.include_interactions = enable;
        self
    }

    /// Set the number of bins for pairwise interaction terms.
    pub fn set_max_interaction_bins(&mut self, bins: usize) -> &mut Self {
        self.max_interaction_bins = bins.max(2);
        self
    }

    /// Return the per-feature shape functions (bin scores).
    ///
    /// Each inner `Vec<T>` contains one score per bin for the corresponding
    /// feature. Only available after fitting.
    pub fn shape_functions(&self) -> Result<&ShapeFns<T>> {
        self.shape_fns.as_ref().ok_or(MlError::NotFitted)
    }

    /// Return the per-feature bin edges.
    pub fn bin_edges(&self) -> Result<&BinEdges<T>> {
        self.bin_edges.as_ref().ok_or(MlError::NotFitted)
    }

    /// Return the interaction terms.
    pub fn interaction_terms(&self) -> Result<&[InteractionTerm<T>]> {
        self.interactions.as_deref().ok_or(MlError::NotFitted)
    }
}

impl<T: Float> Predictor<T> for EbmRegressor<T> {
    fn fit(&mut self, x: &Tensor<T>, y: &Tensor<T>) -> Result<()> {
        let (n, p) = matrix_shape(x)?;
        check_y(y, n)?;
        let x_data = x.as_slice();
        let y_data = y.as_slice();

        let intercept = mean_value(y_data);
        self.intercept = Some(intercept);

        let (bin_edges, binned) = compute_bins(x_data, n, p, self.max_bins);
        self.bin_edges = Some(bin_edges);

        let n_bins = count_bins(self.bin_edges.as_ref().unwrap());
        let mut shape_fns: ShapeFns<T> = n_bins.iter().map(|&nb| vec![T::zero(); nb]).collect();
        let mut f = vec![intercept; n];

        boost_shape_fns(
            y_data,
            &mut f,
            &mut shape_fns,
            &binned,
            &n_bins,
            self.max_rounds,
            p,
            n,
            self.learning_rate,
            mse_residuals,
        );

        let interactions = fit_interactions(
            self.include_interactions,
            p,
            x_data,
            n,
            self.max_interaction_bins,
            y_data,
            &mut f,
            self.max_rounds,
            self.learning_rate,
            mse_residuals,
        );

        self.shape_fns = Some(shape_fns);
        self.interactions = Some(interactions);
        Ok(())
    }

    fn predict(&self, x: &Tensor<T>) -> Result<Tensor<T>> {
        let scores = compute_scores(
            x,
            self.shape_fns.as_ref().ok_or(MlError::NotFitted)?,
            self.bin_edges.as_ref().ok_or(MlError::NotFitted)?,
            self.intercept.ok_or(MlError::NotFitted)?,
            self.interactions.as_ref().ok_or(MlError::NotFitted)?,
        )?;
        let n = scores.len();
        Tensor::from_vec(scores, vec![n]).map_err(MlError::from)
    }
}

// ── EBM Classifier ──

/// Explainable Boosting Machine for binary classification (GA²M).
///
/// Uses log-loss with sigmoid transform. The additive model structure
/// provides interpretable per-feature shape functions.
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
/// let mut ebm = EbmClassifier::new(100, 256, 0.1).unwrap();
/// ebm.fit(&x, &y).unwrap();
/// let preds = ebm.predict(&x).unwrap();
/// ```
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct EbmClassifier<T: Float> {
    max_rounds: usize,
    max_bins: usize,
    learning_rate: T,
    max_interaction_bins: usize,
    include_interactions: bool,
    // Fitted state
    intercept: Option<T>,
    shape_fns: Option<ShapeFns<T>>,
    bin_edges: Option<BinEdges<T>>,
    interactions: Option<Vec<InteractionTerm<T>>>,
}

impl<T: Float> EbmClassifier<T> {
    /// Create a new EBM classifier.
    ///
    /// - `max_rounds`: number of cyclic boosting rounds
    /// - `max_bins`: maximum number of bins per feature
    /// - `learning_rate`: shrinkage applied to each gradient step
    pub fn new(max_rounds: usize, max_bins: usize, learning_rate: f64) -> Result<Self> {
        validate_ebm_params(max_rounds, max_bins, learning_rate)?;
        Ok(Self {
            max_rounds,
            max_bins,
            learning_rate: T::from_f64(learning_rate),
            max_interaction_bins: 32,
            include_interactions: false,
            intercept: None,
            shape_fns: None,
            bin_edges: None,
            interactions: None,
        })
    }

    /// Enable or disable pairwise interaction terms.
    pub fn set_interactions(&mut self, enable: bool) -> &mut Self {
        self.include_interactions = enable;
        self
    }

    /// Set the number of bins for pairwise interaction terms.
    pub fn set_max_interaction_bins(&mut self, bins: usize) -> &mut Self {
        self.max_interaction_bins = bins.max(2);
        self
    }

    /// Return the per-feature shape functions (bin scores).
    pub fn shape_functions(&self) -> Result<&ShapeFns<T>> {
        self.shape_fns.as_ref().ok_or(MlError::NotFitted)
    }

    /// Return the per-feature bin edges.
    pub fn bin_edges(&self) -> Result<&BinEdges<T>> {
        self.bin_edges.as_ref().ok_or(MlError::NotFitted)
    }

    /// Compute raw log-odds for a given input.
    fn raw_scores(&self, x: &Tensor<T>) -> Result<Vec<T>> {
        compute_scores(
            x,
            self.shape_fns.as_ref().ok_or(MlError::NotFitted)?,
            self.bin_edges.as_ref().ok_or(MlError::NotFitted)?,
            self.intercept.ok_or(MlError::NotFitted)?,
            self.interactions.as_ref().ok_or(MlError::NotFitted)?,
        )
    }
}

impl<T: Float> Predictor<T> for EbmClassifier<T> {
    fn fit(&mut self, x: &Tensor<T>, y: &Tensor<T>) -> Result<()> {
        let (n, p) = matrix_shape(x)?;
        check_y(y, n)?;
        let x_data = x.as_slice();
        let y_data = y.as_slice();

        validate_binary_labels(y_data)?;

        let mean_y = mean_value::<T>(y_data);
        let p_clamped = clamp_probability(mean_y);
        let intercept = (p_clamped / (T::one() - p_clamped)).ln();
        self.intercept = Some(intercept);

        let (bin_edges, binned) = compute_bins(x_data, n, p, self.max_bins);
        self.bin_edges = Some(bin_edges);

        let n_bins = count_bins(self.bin_edges.as_ref().unwrap());
        let mut shape_fns: ShapeFns<T> = n_bins.iter().map(|&nb| vec![T::zero(); nb]).collect();
        let mut f = vec![intercept; n];

        boost_shape_fns(
            y_data,
            &mut f,
            &mut shape_fns,
            &binned,
            &n_bins,
            self.max_rounds,
            p,
            n,
            self.learning_rate,
            logloss_residuals,
        );

        let interactions = fit_interactions(
            self.include_interactions,
            p,
            x_data,
            n,
            self.max_interaction_bins,
            y_data,
            &mut f,
            self.max_rounds,
            self.learning_rate,
            logloss_residuals,
        );

        self.shape_fns = Some(shape_fns);
        self.interactions = Some(interactions);
        Ok(())
    }

    fn predict(&self, x: &Tensor<T>) -> Result<Tensor<T>> {
        let scores = self.raw_scores(x)?;
        let n = scores.len();
        let preds: Vec<T> = scores
            .into_iter()
            .map(|s| {
                if sigmoid(s) >= T::from_f64(0.5) {
                    T::one()
                } else {
                    T::zero()
                }
            })
            .collect();
        Tensor::from_vec(preds, vec![n]).map_err(MlError::from)
    }
}

impl<T: Float> Classifier<T> for EbmClassifier<T> {
    fn predict_proba(&self, x: &Tensor<T>) -> Result<Tensor<T>> {
        let scores = self.raw_scores(x)?;
        let n = scores.len();
        let mut proba = Vec::with_capacity(n * 2);
        for s in &scores {
            let p1 = sigmoid(*s);
            let p0 = T::one() - p1;
            proba.push(p0);
            proba.push(p1);
        }
        Tensor::from_vec(proba, vec![n, 2]).map_err(MlError::from)
    }
}

// ── Shared helpers ──

/// Validate common EBM hyper-parameters.
fn validate_ebm_params(max_rounds: usize, max_bins: usize, learning_rate: f64) -> Result<()> {
    if max_rounds == 0 {
        return Err(MlError::InvalidParameter {
            name: "max_rounds",
            reason: "must be at least 1",
        });
    }
    if max_bins < 2 {
        return Err(MlError::InvalidParameter {
            name: "max_bins",
            reason: "must be at least 2",
        });
    }
    if learning_rate <= 0.0 || learning_rate > 1.0 {
        return Err(MlError::InvalidParameter {
            name: "learning_rate",
            reason: "must be in (0, 1]",
        });
    }
    Ok(())
}

/// Validate that all labels are 0.0 or 1.0.
fn validate_binary_labels<T: Float>(y_data: &[T]) -> Result<()> {
    for &v in y_data {
        let v64 = v.to_f64();
        if v64.abs() > 1e-10 && (v64 - 1.0).abs() > 1e-10 {
            return Err(MlError::InvalidParameter {
                name: "y",
                reason: "must contain only 0.0 or 1.0 for binary classification",
            });
        }
    }
    Ok(())
}

/// Compute the mean of a slice.
fn mean_value<T: Float>(data: &[T]) -> T {
    data.iter().copied().fold(T::zero(), |a, b| a + b) / T::from_usize(data.len())
}

/// Count the number of bins per feature from bin edges.
fn count_bins<T>(edges: &[Vec<T>]) -> Vec<usize> {
    edges.iter().map(|e| e.len() + 1).collect()
}

/// Residuals for MSE: y - f.
fn mse_residuals<T: Float>(y: &[T], f: &[T]) -> Vec<T> {
    y.iter().zip(f).map(|(&yi, &fi)| yi - fi).collect()
}

/// Residuals for log-loss: y - sigmoid(f).
fn logloss_residuals<T: Float>(y: &[T], f: &[T]) -> Vec<T> {
    y.iter().zip(f).map(|(&yi, &fi)| yi - sigmoid(fi)).collect()
}

/// Core round-robin boosting loop over features.
///
/// Updates `shape_fns` and predictions `f` in place for `max_rounds` rounds.
#[allow(clippy::too_many_arguments)]
fn boost_shape_fns<T: Float>(
    y_data: &[T],
    f: &mut [T],
    shape_fns: &mut ShapeFns<T>,
    binned: &[Vec<usize>],
    n_bins: &[usize],
    max_rounds: usize,
    p: usize,
    n: usize,
    lr: T,
    residual_fn: ResidualFn<T>,
) {
    for _ in 0..max_rounds {
        for j in 0..p {
            let residuals = residual_fn(y_data, f);
            let nb = n_bins[j];
            let mut bin_sum = vec![T::zero(); nb];
            let mut bin_count = vec![0usize; nb];

            for i in 0..n {
                let b = binned[j][i];
                bin_sum[b] += residuals[i];
                bin_count[b] += 1;
            }

            update_bins_and_preds(
                f,
                &mut shape_fns[j],
                &bin_sum,
                &bin_count,
                &binned[j],
                lr,
                n,
            );
        }
    }
}

/// Update shape function bins and predictions from accumulated residuals.
fn update_bins_and_preds<T: Float>(
    f: &mut [T],
    shape_bins: &mut [T],
    bin_sum: &[T],
    bin_count: &[usize],
    sample_bins: &[usize],
    lr: T,
    n: usize,
) {
    // Update shape function
    for (b, (sum, count)) in bin_sum.iter().zip(bin_count).enumerate() {
        if *count > 0 {
            shape_bins[b] += lr * *sum / T::from_usize(*count);
        }
    }

    // Update predictions
    for i in 0..n {
        let b = sample_bins[i];
        if bin_count[b] > 0 {
            f[i] += lr * bin_sum[b] / T::from_usize(bin_count[b]);
        }
    }
}

/// Fit pairwise interaction terms via boosting on a 2D bin grid.
#[allow(clippy::too_many_arguments)]
fn fit_interactions<T: Float>(
    include: bool,
    p: usize,
    x_data: &[T],
    n: usize,
    max_interaction_bins: usize,
    y_data: &[T],
    f: &mut [T],
    max_rounds: usize,
    lr: T,
    residual_fn: ResidualFn<T>,
) -> Vec<InteractionTerm<T>> {
    if !include || p < 2 {
        return Vec::new();
    }

    let (inter_edges, inter_binned) = compute_bins(x_data, n, p, max_interaction_bins);
    let inter_n_bins: Vec<usize> = inter_edges.iter().map(|e| e.len() + 1).collect();
    let mut inter = Vec::new();

    for j1 in 0..p {
        for j2 in (j1 + 1)..p {
            let grid = boost_interaction_pair(
                &inter_binned,
                j1,
                j2,
                inter_n_bins[j1],
                inter_n_bins[j2],
                y_data,
                f,
                n,
                max_rounds,
                lr,
                residual_fn,
            );
            inter.push((j1, j2, grid));
        }
    }

    inter
}

/// Boost a single pairwise interaction term.
#[allow(clippy::too_many_arguments)]
fn boost_interaction_pair<T: Float>(
    inter_binned: &[Vec<usize>],
    j1: usize,
    j2: usize,
    nb1: usize,
    nb2: usize,
    y_data: &[T],
    f: &mut [T],
    n: usize,
    max_rounds: usize,
    lr: T,
    residual_fn: ResidualFn<T>,
) -> Vec<Vec<T>> {
    let mut grid = vec![vec![T::zero(); nb2]; nb1];

    for _ in 0..max_rounds {
        let residuals = residual_fn(y_data, f);
        let mut grid_sum = vec![vec![T::zero(); nb2]; nb1];
        let mut grid_count = vec![vec![0usize; nb2]; nb1];

        for i in 0..n {
            let b1 = inter_binned[j1][i];
            let b2 = inter_binned[j2][i];
            grid_sum[b1][b2] += residuals[i];
            grid_count[b1][b2] += 1;
        }

        for b1 in 0..nb1 {
            for b2 in 0..nb2 {
                if grid_count[b1][b2] > 0 {
                    let update = lr * grid_sum[b1][b2] / T::from_usize(grid_count[b1][b2]);
                    grid[b1][b2] += update;
                }
            }
        }

        for i in 0..n {
            let b1 = inter_binned[j1][i];
            let b2 = inter_binned[j2][i];
            if grid_count[b1][b2] > 0 {
                f[i] += lr * grid_sum[b1][b2] / T::from_usize(grid_count[b1][b2]);
            }
        }
    }

    grid
}

/// Compute additive scores from shape functions and interactions.
fn compute_scores<T: Float>(
    x: &Tensor<T>,
    shape_fns: &ShapeFns<T>,
    bin_edges: &BinEdges<T>,
    intercept: T,
    interactions: &[InteractionTerm<T>],
) -> Result<Vec<T>> {
    let (n, p) = matrix_shape(x)?;
    let x_data = x.as_slice();
    let mut out = vec![intercept; n];

    for j in 0..p {
        for i in 0..n {
            let val = x_data[i * p + j];
            let b = find_bin(val, &bin_edges[j]);
            out[i] += shape_fns[j][b];
        }
    }

    for (j1, j2, grid) in interactions {
        let nb1 = grid.len();
        let nb2 = grid[0].len();
        for i in 0..n {
            let v1 = x_data[i * p + j1];
            let v2 = x_data[i * p + j2];
            let b1 = find_bin_with_nbins(v1, &bin_edges[*j1], nb1);
            let b2 = find_bin_with_nbins(v2, &bin_edges[*j2], nb2);
            out[i] += grid[b1][b2];
        }
    }

    Ok(out)
}

/// Sigmoid function: 1 / (1 + exp(-x)).
fn sigmoid<T: Float>(x: T) -> T {
    T::one() / (T::one() + (-x).exp())
}

/// Clamp a probability away from 0 and 1.
fn clamp_probability<T: Float>(p: T) -> T {
    let lo = T::from_f64(1e-10);
    let hi = T::from_f64(1.0 - 1e-10);
    if p < lo {
        lo
    } else if p > hi {
        hi
    } else {
        p
    }
}

/// Compute equal-frequency bin edges and bin assignments for all features.
///
/// Returns `(bin_edges, binned)` where `bin_edges[j]` are the cut points
/// and `binned[j][i]` is the bin index for sample `i`, feature `j`.
fn compute_bins<T: Float>(
    x_data: &[T],
    n: usize,
    p: usize,
    max_bins: usize,
) -> (BinEdges<T>, Vec<Vec<usize>>) {
    let mut all_edges = Vec::with_capacity(p);
    let mut all_binned = Vec::with_capacity(p);

    for j in 0..p {
        let mut col: Vec<T> = (0..n).map(|i| x_data[i * p + j]).collect();
        col.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let actual_bins = max_bins.min(n);
        let mut edges = Vec::new();
        if actual_bins > 1 {
            for b in 1..actual_bins {
                let idx = (b * n / actual_bins).min(n - 1);
                let edge = col[idx];
                if edges.is_empty() || edge > *edges.last().unwrap() {
                    edges.push(edge);
                }
            }
        }

        let binned: Vec<usize> = (0..n)
            .map(|i| find_bin(x_data[i * p + j], &edges))
            .collect();

        all_edges.push(edges);
        all_binned.push(binned);
    }

    (all_edges, all_binned)
}

/// Find the bin index for a value given sorted bin edges.
fn find_bin<T: Float>(val: T, edges: &[T]) -> usize {
    match edges.binary_search_by(|e| e.partial_cmp(&val).unwrap_or(std::cmp::Ordering::Equal)) {
        Ok(pos) => {
            if pos + 1 < edges.len() {
                pos + 1
            } else {
                pos
            }
        }
        Err(pos) => pos,
    }
}

/// Find bin index, clamping to `[0, n_bins - 1]`.
fn find_bin_with_nbins<T: Float>(val: T, edges: &[T], n_bins: usize) -> usize {
    find_bin(val, edges).min(n_bins - 1)
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
    fn test_ebm_regressor_basic() {
        let x =
            Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![8, 1]).unwrap();
        let y =
            Tensor::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0], vec![8]).unwrap();

        let mut ebm = EbmRegressor::new(100, 256, 0.3).unwrap();
        ebm.fit(&x, &y).unwrap();
        let preds = ebm.predict(&x).unwrap();

        assert_eq!(preds.shape(), &[8]);
        for (&p, &t) in preds.as_slice().iter().zip(y.as_slice()) {
            assert!(
                (p - t).abs() < 3.0,
                "prediction {p} too far from target {t}"
            );
        }
    }

    #[test]
    fn test_ebm_regressor_not_fitted() {
        let ebm = EbmRegressor::<f64>::new(10, 32, 0.1).unwrap();
        let x = Tensor::from_vec(vec![1.0_f64, 2.0], vec![2, 1]).unwrap();
        assert!(ebm.predict(&x).is_err());
    }

    #[test]
    fn test_ebm_classifier_basic() {
        let x = Tensor::from_vec(
            vec![
                1.0_f64, 1.0, 1.5, 1.5, 2.0, 2.0, 8.0, 8.0, 8.5, 8.5, 9.0, 9.0,
            ],
            vec![6, 2],
        )
        .unwrap();
        let y = Tensor::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0], vec![6]).unwrap();

        let mut ebm = EbmClassifier::new(100, 32, 0.3).unwrap();
        ebm.fit(&x, &y).unwrap();
        let preds = ebm.predict(&x).unwrap();
        assert_eq!(preds.shape(), &[6]);

        let proba = ebm.predict_proba(&x).unwrap();
        assert_eq!(proba.shape(), &[6, 2]);
        let p = proba.as_slice();
        for i in 0..6 {
            let row_sum = p[i * 2] + p[i * 2 + 1];
            assert!(
                (row_sum - 1.0).abs() < 1e-10,
                "row {i} proba sum = {row_sum}"
            );
        }
    }

    #[test]
    fn test_ebm_shape_functions() {
        let x =
            Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![8, 1]).unwrap();
        let y =
            Tensor::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0], vec![8]).unwrap();

        let mut ebm = EbmRegressor::new(50, 32, 0.1).unwrap();
        assert!(ebm.shape_functions().is_err());

        ebm.fit(&x, &y).unwrap();

        let sf = ebm.shape_functions().unwrap();
        assert_eq!(sf.len(), 1);
        assert!(!sf[0].is_empty());

        let edges = ebm.bin_edges().unwrap();
        assert_eq!(edges.len(), 1);

        let any_nonzero = sf[0].iter().any(|&v| v.abs() > 1e-15);
        assert!(any_nonzero, "shape function should have nonzero entries");
    }

    #[test]
    fn test_ebm_invalid_params() {
        assert!(EbmRegressor::<f64>::new(0, 32, 0.1).is_err());
        assert!(EbmRegressor::<f64>::new(10, 1, 0.1).is_err());
        assert!(EbmRegressor::<f64>::new(10, 32, 0.0).is_err());
        assert!(EbmRegressor::<f64>::new(10, 32, 1.5).is_err());
        assert!(EbmClassifier::<f64>::new(0, 32, 0.1).is_err());
        assert!(EbmClassifier::<f64>::new(10, 1, 0.1).is_err());
    }
}
