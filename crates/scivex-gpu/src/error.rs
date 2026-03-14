use std::fmt;

/// GPU-specific errors.
#[derive(Debug, Clone)]
pub enum GpuError {
    /// No suitable GPU adapter found.
    NoAdapter,
    /// Failed to obtain GPU device.
    DeviceCreationFailed { reason: String },
    /// Shape mismatch for a GPU operation.
    DimensionMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    /// Invalid shape for the requested operation.
    InvalidShape { reason: &'static str },
    /// Error propagated from scivex-core.
    CoreError(scivex_core::CoreError),
    /// Buffer mapping or data transfer failed.
    TransferError { reason: String },
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoAdapter => write!(f, "no suitable GPU adapter found"),
            Self::DeviceCreationFailed { reason } => {
                write!(f, "GPU device creation failed: {reason}")
            }
            Self::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected:?}, got {got:?}")
            }
            Self::InvalidShape { reason } => write!(f, "invalid shape: {reason}"),
            Self::CoreError(e) => write!(f, "core error: {e}"),
            Self::TransferError { reason } => write!(f, "GPU transfer error: {reason}"),
        }
    }
}

impl std::error::Error for GpuError {}

impl From<scivex_core::CoreError> for GpuError {
    fn from(e: scivex_core::CoreError) -> Self {
        Self::CoreError(e)
    }
}

pub type Result<T> = std::result::Result<T, GpuError>;
