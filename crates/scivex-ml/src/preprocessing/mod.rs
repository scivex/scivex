//! Data preprocessing: scaling, encoding, and feature transformations.

/// Label encoding for categorical features.
pub mod encoder;
/// One-hot encoding for categorical features.
pub mod onehot;
/// Feature scaling (StandardScaler, MinMaxScaler).
pub mod scaler;

pub use encoder::LabelEncoder;
pub use onehot::OneHotEncoder;
pub use scaler::{MinMaxScaler, StandardScaler};
