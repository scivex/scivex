use core::fmt;

/// All errors returned by `scivex-viz`.
#[derive(Debug)]
pub enum VizError {
    /// The input data is empty.
    EmptyData,

    /// Lengths of two related data arrays do not match.
    DimensionMismatch { expected: usize, got: usize },

    /// A parameter has an invalid value.
    InvalidParameter {
        name: &'static str,
        reason: &'static str,
    },

    /// An error during rendering.
    RenderError(String),

    /// An I/O error (e.g. writing SVG to file).
    IoError(std::io::Error),

    /// Animation has no frames.
    NoFrames,

    /// LaTeX parsing error.
    LatexParseError(String),

    /// An error propagated from `scivex-core`.
    CoreError(scivex_core::CoreError),
}

impl fmt::Display for VizError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyData => write!(f, "input data is empty"),
            Self::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            Self::InvalidParameter { name, reason } => {
                write!(f, "invalid parameter `{name}`: {reason}")
            }
            Self::RenderError(msg) => write!(f, "render error: {msg}"),
            Self::NoFrames => write!(f, "animation has no frames"),
            Self::LatexParseError(msg) => write!(f, "LaTeX parse error: {msg}"),
            Self::IoError(e) => write!(f, "I/O error: {e}"),
            Self::CoreError(e) => write!(f, "core error: {e}"),
        }
    }
}

impl std::error::Error for VizError {}

impl From<scivex_core::CoreError> for VizError {
    fn from(e: scivex_core::CoreError) -> Self {
        Self::CoreError(e)
    }
}

impl From<std::io::Error> for VizError {
    fn from(e: std::io::Error) -> Self {
        Self::IoError(e)
    }
}

impl Clone for VizError {
    fn clone(&self) -> Self {
        match self {
            Self::EmptyData => Self::EmptyData,
            Self::DimensionMismatch { expected, got } => Self::DimensionMismatch {
                expected: *expected,
                got: *got,
            },
            Self::InvalidParameter { name, reason } => Self::InvalidParameter { name, reason },
            Self::RenderError(msg) => Self::RenderError(msg.clone()),
            Self::NoFrames => Self::NoFrames,
            Self::LatexParseError(msg) => Self::LatexParseError(msg.clone()),
            Self::IoError(e) => Self::IoError(std::io::Error::new(e.kind(), e.to_string())),
            Self::CoreError(e) => Self::CoreError(e.clone()),
        }
    }
}

/// Convenience alias used throughout `scivex-viz`.
pub type Result<T> = std::result::Result<T, VizError>;
