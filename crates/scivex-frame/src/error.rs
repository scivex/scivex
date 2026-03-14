use core::fmt;

/// All errors returned by `scivex-frame`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FrameError {
    /// A requested column was not found.
    ColumnNotFound { name: String },

    /// Attempted to add a column whose name already exists.
    DuplicateColumnName { name: String },

    /// Column lengths do not match the expected row count.
    RowCountMismatch { expected: usize, got: usize },

    /// Runtime type mismatch during downcast.
    TypeMismatch {
        expected: &'static str,
        got: &'static str,
    },

    /// An index is out of bounds.
    IndexOutOfBounds { index: usize, length: usize },

    /// An invalid argument was provided.
    InvalidArgument { reason: &'static str },

    /// Operation requires a non-empty `DataFrame`.
    EmptyDataFrame,

    /// Join key column counts do not match.
    JoinKeyMismatch { left: usize, right: usize },

    /// An invalid value was encountered (e.g. invalid regex pattern).
    InvalidValue { reason: String },

    /// An error propagated from `scivex-core`.
    CoreError(scivex_core::CoreError),
}

impl fmt::Display for FrameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ColumnNotFound { name } => write!(f, "column not found: {name:?}"),
            Self::DuplicateColumnName { name } => write!(f, "duplicate column name: {name:?}"),
            Self::RowCountMismatch { expected, got } => {
                write!(f, "row count mismatch: expected {expected}, got {got}")
            }
            Self::TypeMismatch { expected, got } => {
                write!(f, "type mismatch: expected {expected}, got {got}")
            }
            Self::IndexOutOfBounds { index, length } => {
                write!(f, "index {index} out of bounds for length {length}")
            }
            Self::InvalidArgument { reason } => write!(f, "invalid argument: {reason}"),
            Self::EmptyDataFrame => write!(f, "operation requires a non-empty dataframe"),
            Self::JoinKeyMismatch { left, right } => {
                write!(
                    f,
                    "join key count mismatch: left has {left}, right has {right}"
                )
            }
            Self::InvalidValue { reason } => write!(f, "invalid value: {reason}"),
            Self::CoreError(e) => write!(f, "core error: {e}"),
        }
    }
}

impl std::error::Error for FrameError {}

impl From<scivex_core::CoreError> for FrameError {
    fn from(e: scivex_core::CoreError) -> Self {
        Self::CoreError(e)
    }
}

/// Convenience alias used throughout `scivex-frame`.
pub type Result<T> = std::result::Result<T, FrameError>;
