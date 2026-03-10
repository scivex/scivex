use std::fmt;

/// Errors produced by `scivex-nn`.
#[derive(Debug, Clone, PartialEq)]
pub enum NnError {
    /// Shape mismatch between expected and actual.
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    /// Gradient is not available (variable does not require grad or backward not called).
    NoGradient,
    /// An invalid hyper-parameter was supplied.
    InvalidParameter {
        name: &'static str,
        reason: &'static str,
    },
    /// Input data is empty.
    EmptyInput,
    /// Index out of bounds.
    IndexOutOfBounds { index: usize, len: usize },
    /// Error propagated from `scivex-core`.
    CoreError(scivex_core::CoreError),
}

impl fmt::Display for NnError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeMismatch { expected, got } => {
                write!(f, "shape mismatch: expected {expected:?}, got {got:?}")
            }
            Self::NoGradient => write!(f, "gradient is not available"),
            Self::InvalidParameter { name, reason } => {
                write!(f, "invalid parameter `{name}`: {reason}")
            }
            Self::EmptyInput => write!(f, "input data is empty"),
            Self::IndexOutOfBounds { index, len } => {
                write!(f, "index {index} out of bounds for length {len}")
            }
            Self::CoreError(e) => write!(f, "core: {e}"),
        }
    }
}

impl std::error::Error for NnError {}

impl From<scivex_core::CoreError> for NnError {
    fn from(e: scivex_core::CoreError) -> Self {
        Self::CoreError(e)
    }
}

/// Alias for `std::result::Result<T, NnError>`.
pub type Result<T> = std::result::Result<T, NnError>;
