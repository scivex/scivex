use core::fmt;

/// All errors returned by `scivex-core`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoreError {
    /// Operand shapes do not match the required layout.
    DimensionMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    /// A shape or stride specification is invalid.
    InvalidShape {
        shape: Vec<usize>,
        reason: &'static str,
    },

    /// An axis index is out of bounds for the tensor's rank.
    AxisOutOfBounds { axis: usize, ndim: usize },

    /// A flat or multi-dimensional index is out of bounds.
    IndexOutOfBounds {
        index: Vec<usize>,
        shape: Vec<usize>,
    },

    /// Matrix is singular and cannot be inverted / decomposed.
    SingularMatrix,

    /// The operation is not supported for the given input.
    InvalidArgument { reason: &'static str },

    /// Shapes cannot be broadcast together.
    BroadcastError {
        shape_a: Vec<usize>,
        shape_b: Vec<usize>,
    },
}

impl fmt::Display for CoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected:?}, got {got:?}")
            }
            Self::InvalidShape { shape, reason } => {
                write!(f, "invalid shape {shape:?}: {reason}")
            }
            Self::AxisOutOfBounds { axis, ndim } => {
                write!(
                    f,
                    "axis {axis} out of bounds for tensor with {ndim} dimensions"
                )
            }
            Self::IndexOutOfBounds { index, shape } => {
                write!(f, "index {index:?} out of bounds for shape {shape:?}")
            }
            Self::SingularMatrix => write!(f, "singular matrix"),
            Self::InvalidArgument { reason } => write!(f, "invalid argument: {reason}"),
            Self::BroadcastError { shape_a, shape_b } => {
                write!(f, "cannot broadcast shapes {shape_a:?} and {shape_b:?}")
            }
        }
    }
}

impl std::error::Error for CoreError {}

/// Convenience alias used throughout `scivex-core`.
pub type Result<T> = std::result::Result<T, CoreError>;
