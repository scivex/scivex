use core::fmt;

/// All errors returned by `scivex-stats`.
#[derive(Debug, Clone, PartialEq)]
pub enum StatsError {
    /// Input slice is empty when at least one element is required.
    EmptyInput,

    /// Not enough data points for the requested operation.
    InsufficientData { need: usize, got: usize },

    /// Two input slices have different lengths when they must match.
    LengthMismatch { expected: usize, got: usize },

    /// A distribution or function parameter is invalid.
    InvalidParameter {
        name: &'static str,
        reason: &'static str,
    },

    /// An iterative algorithm did not converge.
    ConvergenceFailure { iterations: usize },

    /// A required matrix is singular.
    SingularMatrix,

    /// An error propagated from `scivex-core`.
    CoreError(scivex_core::CoreError),
}

impl fmt::Display for StatsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "input is empty"),
            Self::InsufficientData { need, got } => {
                write!(f, "insufficient data: need at least {need}, got {got}")
            }
            Self::LengthMismatch { expected, got } => {
                write!(f, "length mismatch: expected {expected}, got {got}")
            }
            Self::InvalidParameter { name, reason } => {
                write!(f, "invalid parameter `{name}`: {reason}")
            }
            Self::ConvergenceFailure { iterations } => {
                write!(f, "convergence failure after {iterations} iterations")
            }
            Self::SingularMatrix => write!(f, "singular matrix"),
            Self::CoreError(e) => write!(f, "core error: {e}"),
        }
    }
}

impl std::error::Error for StatsError {}

impl From<scivex_core::CoreError> for StatsError {
    fn from(e: scivex_core::CoreError) -> Self {
        Self::CoreError(e)
    }
}

/// Convenience alias used throughout `scivex-stats`.
pub type Result<T> = std::result::Result<T, StatsError>;
