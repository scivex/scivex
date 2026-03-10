//! Evaluation metrics for classification and regression.

pub mod classification;
pub mod regression;

pub use classification::{accuracy, confusion_matrix, f1_score, precision, recall};
pub use regression::{mae, mse, r2_score, rmse};
