use core::fmt;

/// All errors returned by `scivex-signal`.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum SignalError {
    /// A parameter has an invalid value.
    InvalidParameter {
        name: &'static str,
        reason: &'static str,
    },

    /// The input signal is empty.
    EmptyInput,

    /// A dimension does not match what was expected.
    DimensionMismatch { expected: usize, got: usize },

    /// An error propagated from `scivex-core`.
    CoreError(scivex_core::CoreError),
}

impl fmt::Display for SignalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameter { name, reason } => {
                write!(f, "invalid parameter `{name}`: {reason}")
            }
            Self::EmptyInput => write!(f, "input signal is empty"),
            Self::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            Self::CoreError(e) => write!(f, "core error: {e}"),
        }
    }
}

impl std::error::Error for SignalError {}

impl From<scivex_core::CoreError> for SignalError {
    fn from(e: scivex_core::CoreError) -> Self {
        Self::CoreError(e)
    }
}

/// Convenience alias used throughout `scivex-signal`.
pub type Result<T> = std::result::Result<T, SignalError>;
