//! Evaluation metrics for classification and regression.

/// Classification metrics (accuracy, precision, recall, F1).
pub mod classification;
/// Regression metrics (MSE, MAE, R², RMSE).
pub mod regression;

pub use classification::{accuracy, confusion_matrix, f1_score, precision, recall};
pub use regression::{mae, mse, r2_score, rmse};
