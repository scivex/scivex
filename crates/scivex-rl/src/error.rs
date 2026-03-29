//! Error types for the reinforcement learning crate.

use std::fmt;

/// Errors produced by the RL crate.
#[derive(Debug)]
#[non_exhaustive]
pub enum RlError {
    /// A configuration parameter is invalid.
    InvalidParameter(String),
    /// An environment-specific error occurred.
    EnvironmentError(String),
    /// An operation was attempted on an agent that has not been trained.
    NotTrained(String),
}

impl fmt::Display for RlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameter(msg) => write!(f, "invalid parameter: {msg}"),
            Self::EnvironmentError(msg) => write!(f, "environment error: {msg}"),
            Self::NotTrained(msg) => write!(f, "not trained: {msg}"),
        }
    }
}

impl std::error::Error for RlError {}

/// Convenience alias for `std::result::Result<T, RlError>`.
pub type Result<T> = std::result::Result<T, RlError>;
