use core::fmt;

/// All errors returned by `scivex-io`.
///
/// # Examples
///
/// ```
/// use scivex_io::error::IoError;
///
/// let err = IoError::EmptyInput;
/// assert!(err.to_string().contains("empty"));
///
/// let err2 = IoError::JsonError("bad json".to_string());
/// assert!(err2.to_string().contains("JSON"));
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub enum IoError {
    /// Wraps a [`std::io::Error`].
    Io(std::io::Error),

    /// A CSV parsing error at a specific location.
    CsvParse {
        line: usize,
        column: usize,
        reason: String,
    },

    /// A JSON-related error.
    JsonError(String),

    /// Type inference failed for a column.
    TypeInference { column: String, reason: String },

    /// The input was empty (no data to read).
    EmptyInput,

    /// The header row is invalid.
    InvalidHeader { reason: String },

    /// An error propagated from `scivex-frame`.
    FrameError(scivex_frame::FrameError),

    /// An error propagated from `scivex-core`.
    CoreError(scivex_core::CoreError),

    /// A SQL database error.
    SqlError(String),

    /// An Arrow/Parquet error.
    ArrowError(String),

    /// A file format error (invalid magic bytes, corrupt header, etc.).
    FormatError(String),
}

impl fmt::Display for IoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::CsvParse {
                line,
                column,
                reason,
            } => write!(
                f,
                "CSV parse error at line {line}, column {column}: {reason}"
            ),
            Self::JsonError(msg) => write!(f, "JSON error: {msg}"),
            Self::TypeInference { column, reason } => {
                write!(f, "type inference failed for column {column:?}: {reason}")
            }
            Self::EmptyInput => write!(f, "empty input: no data to read"),
            Self::InvalidHeader { reason } => write!(f, "invalid header: {reason}"),
            Self::FrameError(e) => write!(f, "frame error: {e}"),
            Self::CoreError(e) => write!(f, "core error: {e}"),
            Self::SqlError(msg) => write!(f, "SQL error: {msg}"),
            Self::ArrowError(msg) => write!(f, "Arrow/Parquet error: {msg}"),
            Self::FormatError(msg) => write!(f, "format error: {msg}"),
        }
    }
}

impl std::error::Error for IoError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::FrameError(e) => Some(e),
            Self::CoreError(e) => Some(e),
            Self::CsvParse { .. }
            | Self::JsonError(_)
            | Self::TypeInference { .. }
            | Self::EmptyInput
            | Self::InvalidHeader { .. }
            | Self::SqlError(_)
            | Self::ArrowError(_)
            | Self::FormatError(_) => None,
        }
    }
}

impl From<std::io::Error> for IoError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<scivex_frame::FrameError> for IoError {
    fn from(e: scivex_frame::FrameError) -> Self {
        Self::FrameError(e)
    }
}

impl From<scivex_core::CoreError> for IoError {
    fn from(e: scivex_core::CoreError) -> Self {
        Self::CoreError(e)
    }
}

/// Convenience alias used throughout `scivex-io`.
///
/// # Examples
///
/// ```
/// use scivex_io::error::{IoError, Result};
///
/// fn always_ok() -> Result<i32> {
///     Ok(42)
/// }
/// assert_eq!(always_ok().unwrap(), 42);
/// ```
pub type Result<T> = std::result::Result<T, IoError>;
