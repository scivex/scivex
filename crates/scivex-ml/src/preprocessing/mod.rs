//! Data preprocessing: scaling, encoding, and feature transformations.

/// Label encoding for categorical features.
pub mod encoder;
/// Feature scaling (StandardScaler, MinMaxScaler).
pub mod scaler;

pub use encoder::LabelEncoder;
pub use scaler::{MinMaxScaler, StandardScaler};
