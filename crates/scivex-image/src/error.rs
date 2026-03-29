use core::fmt;

/// All errors returned by `scivex-image`.
///
/// # Examples
///
/// ```
/// # use scivex_image::error::ImageError;
/// let e = ImageError::InvalidDimensions { width: 0, height: 10 };
/// assert!(e.to_string().contains("invalid dimensions"));
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub enum ImageError {
    /// Image dimensions are invalid (e.g. zero width or height).
    InvalidDimensions { width: usize, height: usize },

    /// The raw data length does not match expected dimensions.
    DataLengthMismatch { expected: usize, got: usize },

    /// The image format is not supported.
    UnsupportedFormat { format: String },

    /// The number of channels is not supported for the operation.
    UnsupportedChannels { channels: usize },

    /// A convolution kernel is invalid.
    InvalidKernel { reason: &'static str },

    /// A parameter has an invalid value.
    InvalidParameter {
        name: &'static str,
        reason: &'static str,
    },

    /// An I/O error (e.g. reading/writing image files).
    IoError(std::io::Error),

    /// An error propagated from `scivex-core`.
    CoreError(scivex_core::CoreError),
}

impl fmt::Display for ImageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDimensions { width, height } => {
                write!(f, "invalid dimensions: {width}x{height}")
            }
            Self::DataLengthMismatch { expected, got } => {
                write!(f, "data length mismatch: expected {expected}, got {got}")
            }
            Self::UnsupportedFormat { format } => {
                write!(f, "unsupported image format: {format}")
            }
            Self::UnsupportedChannels { channels } => {
                write!(f, "unsupported channel count: {channels}")
            }
            Self::InvalidKernel { reason } => write!(f, "invalid kernel: {reason}"),
            Self::InvalidParameter { name, reason } => {
                write!(f, "invalid parameter `{name}`: {reason}")
            }
            Self::IoError(e) => write!(f, "I/O error: {e}"),
            Self::CoreError(e) => write!(f, "core error: {e}"),
        }
    }
}

impl std::error::Error for ImageError {}

impl From<scivex_core::CoreError> for ImageError {
    fn from(e: scivex_core::CoreError) -> Self {
        Self::CoreError(e)
    }
}

impl From<std::io::Error> for ImageError {
    fn from(e: std::io::Error) -> Self {
        Self::IoError(e)
    }
}

impl Clone for ImageError {
    fn clone(&self) -> Self {
        match self {
            Self::InvalidDimensions { width, height } => Self::InvalidDimensions {
                width: *width,
                height: *height,
            },
            Self::DataLengthMismatch { expected, got } => Self::DataLengthMismatch {
                expected: *expected,
                got: *got,
            },
            Self::UnsupportedFormat { format } => Self::UnsupportedFormat {
                format: format.clone(),
            },
            Self::UnsupportedChannels { channels } => Self::UnsupportedChannels {
                channels: *channels,
            },
            Self::InvalidKernel { reason } => Self::InvalidKernel { reason },
            Self::InvalidParameter { name, reason } => Self::InvalidParameter { name, reason },
            Self::IoError(e) => Self::IoError(std::io::Error::new(e.kind(), e.to_string())),
            Self::CoreError(e) => Self::CoreError(e.clone()),
        }
    }
}

/// Convenience alias used throughout `scivex-image`.
///
/// # Examples
///
/// ```
/// # use scivex_image::error::{ImageError, Result};
/// fn always_ok() -> Result<u32> { Ok(42) }
/// assert_eq!(always_ok().unwrap(), 42);
/// ```
pub type Result<T> = std::result::Result<T, ImageError>;
