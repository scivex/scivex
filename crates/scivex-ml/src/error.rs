use std::fmt;

/// Errors produced by `scivex-ml` algorithms.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone, PartialEq)]
pub enum MlError {
    /// Model has not been fitted yet.
    NotFitted,
    /// Input data is empty.
    EmptyInput,
    /// Dimension mismatch between expected and actual.
    DimensionMismatch { expected: usize, got: usize },
    /// An invalid hyper-parameter was supplied.
    InvalidParameter {
        name: &'static str,
        reason: &'static str,
    },
    /// Iterative algorithm did not converge.
    ConvergenceFailure { iterations: usize },
    /// Singular matrix encountered during computation.
    SingularMatrix,
    /// Error propagated from `scivex-core`.
    CoreError(scivex_core::CoreError),
    /// Error propagated from `scivex-stats`.
    StatsError(scivex_stats::StatsError),
}

impl fmt::Display for MlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFitted => write!(f, "model has not been fitted"),
            Self::EmptyInput => write!(f, "input data is empty"),
            Self::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            Self::InvalidParameter { name, reason } => {
                write!(f, "invalid parameter `{name}`: {reason}")
            }
            Self::ConvergenceFailure { iterations } => {
                write!(f, "did not converge after {iterations} iterations")
            }
            Self::SingularMatrix => write!(f, "singular matrix"),
            Self::CoreError(e) => write!(f, "core: {e}"),
            Self::StatsError(e) => write!(f, "stats: {e}"),
        }
    }
}

impl std::error::Error for MlError {}

impl From<scivex_core::CoreError> for MlError {
    fn from(e: scivex_core::CoreError) -> Self {
        Self::CoreError(e)
    }
}

impl From<scivex_stats::StatsError> for MlError {
    fn from(e: scivex_stats::StatsError) -> Self {
        Self::StatsError(e)
    }
}

/// Alias for `std::result::Result<T, MlError>`.
pub type Result<T> = std::result::Result<T, MlError>;
