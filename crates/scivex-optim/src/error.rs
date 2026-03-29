use core::fmt;

/// All errors returned by `scivex-optim`.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum OptimError {
    /// An iterative algorithm did not converge within the iteration budget.
    ConvergenceFailure { iterations: usize },

    /// A computation produced a non-finite value (NaN or infinity).
    NonFiniteValue { context: &'static str },

    /// A parameter has an invalid value.
    InvalidParameter {
        name: &'static str,
        reason: &'static str,
    },

    /// The bracket does not contain a root (signs are not opposite).
    BracketError,

    /// An error propagated from `scivex-core`.
    CoreError(scivex_core::CoreError),
}

impl fmt::Display for OptimError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ConvergenceFailure { iterations } => {
                write!(f, "convergence failure after {iterations} iterations")
            }
            Self::NonFiniteValue { context } => {
                write!(f, "non-finite value encountered in {context}")
            }
            Self::InvalidParameter { name, reason } => {
                write!(f, "invalid parameter `{name}`: {reason}")
            }
            Self::BracketError => {
                write!(
                    f,
                    "bracket does not contain a root (f(a) and f(b) must have opposite signs)"
                )
            }
            Self::CoreError(e) => write!(f, "core error: {e}"),
        }
    }
}

impl std::error::Error for OptimError {}

impl From<scivex_core::CoreError> for OptimError {
    fn from(e: scivex_core::CoreError) -> Self {
        Self::CoreError(e)
    }
}

/// Convenience alias used throughout `scivex-optim`.
pub type Result<T> = std::result::Result<T, OptimError>;
