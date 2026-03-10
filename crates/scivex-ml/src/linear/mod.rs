//! Linear models for regression and classification.

pub mod logistic;
pub mod regression;

pub use logistic::LogisticRegression;
pub use regression::{LinearRegression, Ridge};
