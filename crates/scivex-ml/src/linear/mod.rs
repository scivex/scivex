//! Linear models for regression and classification.

/// Logistic regression classifier.
pub mod logistic;
/// Linear and ridge regression.
pub mod regression;

pub use logistic::LogisticRegression;
pub use regression::{LinearRegression, Ridge};
