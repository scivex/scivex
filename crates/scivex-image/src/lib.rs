//! `scivex-image` — Image loading, transforms, and filters.
//!
//! Provides a from-scratch image processing library with support for:
//! - Image I/O (PPM, PGM, BMP formats; PNG and JPEG via feature flags)
//! - Color space conversions (grayscale, RGB, HSV)
//! - Geometric transforms (resize, crop, flip, rotate, pad, Lanczos resampling)
//! - Spatial filters (convolution, Gaussian blur, Sobel edge detection)
//! - Histogram operations (histogram, equalization)
//! - Morphological operations (erosion, dilation, opening, closing)
//! - Feature detection (Harris, FAST, ORB)
//! - Feature matching (brute-force, FLANN/LSH)
//! - Image segmentation (connected components, region growing, watershed)
//! - Hough transform (line and circle detection)
//! - Drawing primitives (lines, rectangles, circles)

/// Data augmentation pipeline for training.
pub mod augment;
/// Color types and conversions (RGB, HSV, grayscale).
pub mod color;
/// Contour detection and analysis.
pub mod contour;
/// Drawing primitives (lines, rectangles, circles).
pub mod draw;
/// Image-specific error types.
pub mod error;
/// Feature detection (Harris corners, FAST features).
pub mod features;
/// Spatial filters (convolution, Gaussian blur, Sobel).
pub mod filter;
/// Histogram computation and equalization.
pub mod histogram;
/// Hough transform for line and circle detection.
pub mod hough;
/// Core [`Image`] type and pixel formats.
pub mod image;
/// Image I/O (PPM, PGM, BMP).
pub mod io;
/// Lanczos resampling for high-quality image resizing.
pub mod lanczos;
/// Feature matching (brute-force, FLANN/LSH).
pub mod matching;
/// Morphological operations (erosion, dilation, opening, closing, gradient).
pub mod morphology;
/// Optical flow estimation (Lucas-Kanade, Farneback).
pub mod optical_flow;
/// ORB (Oriented FAST and Rotated BRIEF) feature descriptor.
pub mod orb;
/// Image segmentation (connected components, region growing, watershed).
pub mod segment;
/// Geometric transforms (resize, crop, flip, rotate, pad).
pub mod transform;

pub use error::{ImageError, Result};
pub use image::{Image, PixelFormat};

/// Items intended for glob-import: `use scivex_image::prelude::*;`
pub mod prelude {
    pub use crate::contour::Contour;
    pub use crate::error::{ImageError, Result};
    pub use crate::features::Corner;
    pub use crate::hough::{HoughCircle, HoughLine};
    pub use crate::image::{Image, PixelFormat};
    pub use crate::matching::{BruteForceMatcher, FeatureMatch, FlannMatcher};
    pub use crate::morphology::StructuringElement;
    pub use crate::orb::{Keypoint, OrbDescriptor, OrbDetector};
    pub use crate::transform::ResizeMethod;
}
