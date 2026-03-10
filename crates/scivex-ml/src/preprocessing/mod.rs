//! Data preprocessing: scaling, encoding, and feature transformations.

pub mod encoder;
pub mod scaler;

pub use encoder::LabelEncoder;
pub use scaler::{MinMaxScaler, StandardScaler};
