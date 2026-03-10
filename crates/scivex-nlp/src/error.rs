use std::fmt;

/// Errors that can occur in NLP operations.
#[derive(Debug, Clone, PartialEq)]
pub enum NlpError {
    /// The input text or document list is empty.
    EmptyInput,
    /// No vocabulary could be built from the input.
    EmptyVocabulary,
    /// A configuration parameter is invalid.
    InvalidParameter {
        /// Parameter name.
        name: &'static str,
        /// Why the value is invalid.
        reason: &'static str,
    },
    /// A token was not found in the vocabulary.
    UnknownToken {
        /// The unrecognised token string.
        token: String,
    },
    /// The model has not been fitted yet.
    NotFitted,
    /// An error propagated from `scivex-core`.
    CoreError(scivex_core::CoreError),
}

impl fmt::Display for NlpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "input is empty"),
            Self::EmptyVocabulary => write!(f, "vocabulary is empty"),
            Self::InvalidParameter { name, reason } => {
                write!(f, "invalid parameter `{name}`: {reason}")
            }
            Self::UnknownToken { token } => write!(f, "unknown token: {token}"),
            Self::NotFitted => write!(f, "model has not been fitted"),
            Self::CoreError(e) => write!(f, "core error: {e}"),
        }
    }
}

impl std::error::Error for NlpError {}

impl From<scivex_core::CoreError> for NlpError {
    fn from(e: scivex_core::CoreError) -> Self {
        Self::CoreError(e)
    }
}

/// Convenience alias for `Result<T, NlpError>`.
pub type Result<T> = std::result::Result<T, NlpError>;
