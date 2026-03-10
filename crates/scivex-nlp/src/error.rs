use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum NlpError {
    EmptyInput,
    EmptyVocabulary,
    InvalidParameter {
        name: &'static str,
        reason: &'static str,
    },
    UnknownToken {
        token: String,
    },
    NotFitted,
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

pub type Result<T> = std::result::Result<T, NlpError>;
