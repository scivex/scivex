//! Online / streaming machine learning algorithms.
//!
//! This module provides incremental learners that process data one
//! mini-batch at a time, making them suitable for streaming or
//! out-of-core workloads.

/// Mini-batch K-Means clustering.
pub mod online_kmeans;
/// Stochastic gradient descent models.
pub mod sgd;
/// Streaming mean and variance (Welford's algorithm).
pub mod streaming_stats;

pub use online_kmeans::OnlineKMeans;
pub use sgd::{SGDClassifier, SGDRegressor};
pub use streaming_stats::{StreamingMean, StreamingVariance};

use scivex_core::{Float, Tensor};

use crate::Result;

/// An incremental learner that can be updated with mini-batches.
pub trait IncrementalPredictor<T: Float> {
    /// Update the model with a new mini-batch of features `x` and targets `y`.
    fn partial_fit(&mut self, x: &Tensor<T>, y: &Tensor<T>) -> Result<()>;

    /// Predict targets for new features `x`.
    fn predict(&self, x: &Tensor<T>) -> Result<Tensor<T>>;

    /// Total number of training samples seen so far.
    fn n_samples_seen(&self) -> usize;
}
