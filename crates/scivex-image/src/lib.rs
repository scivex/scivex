//! `scivex-image` — Image loading, transforms, and filters.
//!
//! Provides a from-scratch image processing library with support for:
//! - Image I/O (PPM, PGM, BMP formats; PNG and JPEG via feature flags)
//! - Color space conversions (grayscale, RGB, HSV)
//! - Geometric transforms (resize, crop, flip, rotate, pad)
//! - Spatial filters (convolution, Gaussian blur, Sobel edge detection)
//! - Histogram operations (histogram, equalization)
//! - Drawing primitives (lines, rectangles, circles)

/// Color types and conversions (RGB, HSV, grayscale).
pub mod color;
/// Drawing primitives (lines, rectangles, circles).
pub mod draw;
/// Image-specific error types.
pub mod error;
/// Spatial filters (convolution, Gaussian blur, Sobel).
pub mod filter;
/// Histogram computation and equalization.
pub mod histogram;
/// Core [`Image`] type and pixel formats.
pub mod image;
/// Image I/O (PPM, PGM, BMP).
pub mod io;
/// Geometric transforms (resize, crop, flip, rotate, pad).
pub mod transform;

pub use error::{ImageError, Result};
pub use image::{Image, PixelFormat};

/// Items intended for glob-import: `use scivex_image::prelude::*;`
pub mod prelude {
    pub use crate::error::{ImageError, Result};
    pub use crate::image::{Image, PixelFormat};
    pub use crate::transform::ResizeMethod;
}
