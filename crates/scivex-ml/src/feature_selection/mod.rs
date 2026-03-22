//! Feature selection methods.
//!
//! Provides scoring functions and selection strategies to identify the most
//! informative features in a dataset.
//!
//! ## Scoring Functions
//!
//! - [`chi2`] — Chi-squared statistic between each feature and the target.
//! - [`f_classif`] — One-way ANOVA F-value between each feature and the target.
//!
//! ## Selection Methods
//!
//! - [`SelectKBest`] — Keep the *k* highest-scoring features.
//! - [`RFE`] — Recursive Feature Elimination using an estimator's importances.

mod rfe;
mod scoring;
mod select_k_best;

pub use rfe::RFE;
pub use scoring::{chi2, f_classif};
pub use select_k_best::SelectKBest;

/// Which scoring function to use for univariate feature selection.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScoringFunction {
    /// Chi-squared test (non-negative features, discrete target).
    Chi2,
    /// One-way ANOVA F-test (continuous features, discrete target).
    FClassif,
}
