use core::fmt;

use scivex_core::CoreError;

/// All errors returned by `scivex-sym`.
///
/// # Examples
///
/// ```
/// # use scivex_sym::SymError;
/// let err = SymError::DivisionByZero;
/// assert_eq!(format!("{err}"), "division by zero");
/// ```
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone, PartialEq)]
pub enum SymError {
    /// The expression is malformed or unsupported.
    InvalidExpr { reason: &'static str },

    /// A variable was referenced but no value was provided.
    UndefinedVariable { name: String },

    /// Division by zero during evaluation.
    DivisionByZero,

    /// The requested operation is not supported.
    UnsupportedOperation { reason: &'static str },

    /// A solve operation could not find a solution.
    SolveFailure { reason: &'static str },

    /// Error propagated from `scivex-core`.
    CoreError(CoreError),
}

impl fmt::Display for SymError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidExpr { reason } => write!(f, "invalid expression: {reason}"),
            Self::UndefinedVariable { name } => write!(f, "undefined variable: {name}"),
            Self::DivisionByZero => write!(f, "division by zero"),
            Self::UnsupportedOperation { reason } => {
                write!(f, "unsupported operation: {reason}")
            }
            Self::SolveFailure { reason } => write!(f, "solve failure: {reason}"),
            Self::CoreError(e) => write!(f, "core error: {e}"),
        }
    }
}

impl std::error::Error for SymError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::CoreError(e) => Some(e),
            _ => None,
        }
    }
}

impl From<CoreError> for SymError {
    fn from(e: CoreError) -> Self {
        Self::CoreError(e)
    }
}

/// Convenience alias used throughout `scivex-sym`.
pub type Result<T> = std::result::Result<T, SymError>;
