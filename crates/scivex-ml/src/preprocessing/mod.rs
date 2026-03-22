//! Data preprocessing: scaling, encoding, and feature transformations.

/// Binary encoding for categorical features.
pub mod binary_encoder;
/// Label encoding for categorical features.
pub mod encoder;
/// One-hot encoding for categorical features.
pub mod onehot;
/// Ordinal encoding for categorical features.
pub mod ordinal_encoder;
/// Feature scaling (StandardScaler, MinMaxScaler).
pub mod scaler;
/// Target (supervised) encoding for categorical features.
pub mod target_encoder;

pub use binary_encoder::BinaryEncoder;
pub use encoder::LabelEncoder;
pub use onehot::OneHotEncoder;
pub use ordinal_encoder::OrdinalEncoder;
pub use scaler::{MinMaxScaler, StandardScaler};
pub use target_encoder::TargetEncoder;
