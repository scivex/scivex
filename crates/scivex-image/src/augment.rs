//! Data augmentation pipeline for image training.
//!
//! Each augmentation step is probabilistic and controlled by the caller's
//! [`Rng`](scivex_core::random::Rng) instance for reproducibility.

use scivex_core::Float;
use scivex_core::random::Rng;

use crate::error::{ImageError, Result};
use crate::image::Image;

/// A single augmentation operation.
///
/// # Examples
///
/// ```
/// # use scivex_image::augment::AugmentStep;
/// let flip = AugmentStep::RandomFlipH { prob: 0.5 };
/// let noise = AugmentStep::GaussianNoise { sigma: 0.01 };
/// ```
#[derive(Debug, Clone)]
pub enum AugmentStep {
    /// Flip horizontally with probability `prob`.
    RandomFlipH {
        /// Probability of applying the flip.
        prob: f64,
    },
    /// Flip vertically with probability `prob`.
    RandomFlipV {
        /// Probability of applying the flip.
        prob: f64,
    },
    /// Adjust brightness by a random amount in `[-delta, delta]`.
    RandomBrightness {
        /// Maximum brightness shift (in the same scale as pixel values).
        delta: f64,
    },
    /// Add Gaussian noise with standard deviation `sigma`.
    GaussianNoise {
        /// Standard deviation of the noise.
        sigma: f64,
    },
    /// Randomly crop a `width x height` region from the image.
    RandomCrop {
        /// Target crop width.
        width: usize,
        /// Target crop height.
        height: usize,
    },
}

/// A composable augmentation pipeline.
///
/// Steps are applied in the order they are added.
///
/// # Examples
///
/// ```
/// # use scivex_image::prelude::*;
/// # use scivex_image::augment::{AugmentPipeline, AugmentStep};
/// # use scivex_core::random::Rng;
/// let img = Image::from_raw(vec![0.5f32; 12], 2, 2, PixelFormat::Rgb).unwrap();
/// let pipeline = AugmentPipeline::new()
///     .add(AugmentStep::RandomFlipH { prob: 1.0 })
///     .add(AugmentStep::RandomBrightness { delta: 0.1 });
/// let mut rng = Rng::new(42);
/// let out = pipeline.apply(&img, &mut rng).unwrap();
/// assert_eq!(out.dimensions(), (2, 2));
/// ```
#[derive(Debug, Clone)]
pub struct AugmentPipeline {
    steps: Vec<AugmentStep>,
}

impl AugmentPipeline {
    /// Create a new empty pipeline.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_image::augment::AugmentPipeline;
    /// let pipeline = AugmentPipeline::new();
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }

    /// Append a step to the pipeline (builder pattern).
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_image::augment::{AugmentPipeline, AugmentStep};
    /// let pipeline = AugmentPipeline::new()
    ///     .add(AugmentStep::RandomFlipH { prob: 0.5_f64 });
    /// ```
    #[must_use]
    #[allow(clippy::should_implement_trait)]
    pub fn add(mut self, step: AugmentStep) -> Self {
        self.steps.push(step);
        self
    }

    /// Apply all steps in order, returning a new image.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_image::prelude::*;
    /// # use scivex_image::augment::{AugmentPipeline, AugmentStep};
    /// # use scivex_core::random::Rng;
    /// let img = Image::from_raw(vec![0.5f32; 4], 2, 2, PixelFormat::Gray).unwrap();
    /// let pipeline = AugmentPipeline::new().add(AugmentStep::RandomFlipH { prob: 1.0_f64 });
    /// let mut rng = Rng::new(0);
    /// let out = pipeline.apply(&img, &mut rng).unwrap();
    /// assert_eq!(out.dimensions(), (2, 2));
    /// ```
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    pub fn apply<T: Float>(&self, img: &Image<T>, rng: &mut Rng) -> Result<Image<T>> {
        let mut current = img.clone();

        for step in &self.steps {
            current = apply_step(&current, step, rng)?;
        }

        Ok(current)
    }
}

impl Default for AugmentPipeline {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
fn apply_step<T: Float>(img: &Image<T>, step: &AugmentStep, rng: &mut Rng) -> Result<Image<T>> {
    match *step {
        AugmentStep::RandomFlipH { prob } => {
            if rng.next_f64() < prob {
                Ok(flip_horizontal(img))
            } else {
                Ok(img.clone())
            }
        }
        AugmentStep::RandomFlipV { prob } => {
            if rng.next_f64() < prob {
                Ok(flip_vertical(img))
            } else {
                Ok(img.clone())
            }
        }
        AugmentStep::RandomBrightness { delta } => {
            let shift = T::from_f64((rng.next_f64() * 2.0 - 1.0) * delta);
            let out = img.map_pixels(|v| v + shift);
            Ok(out)
        }
        AugmentStep::GaussianNoise { sigma } => {
            let (w, h) = img.dimensions();
            let src = img.as_slice();
            let mut out = Vec::with_capacity(src.len());
            for &v in src {
                let noise = T::from_f64(rng.next_normal_f64() * sigma);
                out.push(v + noise);
            }
            Image::from_raw(out, w, h, img.format())
        }
        AugmentStep::RandomCrop { width, height } => {
            let (img_w, img_h) = img.dimensions();
            if width > img_w || height > img_h {
                return Err(ImageError::InvalidParameter {
                    name: "crop size",
                    reason: "crop dimensions exceed image dimensions",
                });
            }
            let max_x = img_w - width;
            let max_y = img_h - height;
            let x0 = if max_x > 0 {
                (rng.next_f64() * (max_x + 1) as f64) as usize
            } else {
                0
            };
            let y0 = if max_y > 0 {
                (rng.next_f64() * (max_y + 1) as f64) as usize
            } else {
                0
            };
            let x0 = x0.min(max_x);
            let y0 = y0.min(max_y);
            crop(img, x0, y0, width, height)
        }
    }
}

/// Flip an image horizontally (mirror left-right).
fn flip_horizontal<T: Float>(img: &Image<T>) -> Image<T> {
    let (w, h) = img.dimensions();
    let c = img.channels();
    let src = img.as_slice();
    let mut out = vec![T::zero(); src.len()];
    for row in 0..h {
        for col in 0..w {
            let src_off = (row * w + col) * c;
            let dst_off = (row * w + (w - 1 - col)) * c;
            out[dst_off..dst_off + c].copy_from_slice(&src[src_off..src_off + c]);
        }
    }
    Image::from_raw(out, w, h, img.format()).expect("dimensions unchanged")
}

/// Flip an image vertically (mirror top-bottom).
fn flip_vertical<T: Float>(img: &Image<T>) -> Image<T> {
    let (w, h) = img.dimensions();
    let c = img.channels();
    let src = img.as_slice();
    let mut out = vec![T::zero(); src.len()];
    for row in 0..h {
        let src_row = row * w * c;
        let dst_row = (h - 1 - row) * w * c;
        out[dst_row..dst_row + w * c].copy_from_slice(&src[src_row..src_row + w * c]);
    }
    Image::from_raw(out, w, h, img.format()).expect("dimensions unchanged")
}

/// Crop a region from the image.
fn crop<T: Float>(
    img: &Image<T>,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
) -> Result<Image<T>> {
    let (img_w, _img_h) = img.dimensions();
    let c = img.channels();
    let src = img.as_slice();
    let fmt = img.format();

    let mut out = Vec::with_capacity(height * width * c);
    for row in y..y + height {
        let start = (row * img_w + x) * c;
        let end = start + width * c;
        out.extend_from_slice(&src[start..end]);
    }

    Image::from_raw(out, width, height, fmt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::PixelFormat;

    #[test]
    fn test_flip_preserves_dimensions() {
        let data = vec![0.5f32; 4 * 6 * 3];
        let img = Image::from_raw(data, 6, 4, PixelFormat::Rgb).unwrap();
        let mut rng = Rng::new(42);

        let pipeline = AugmentPipeline::new()
            .add(AugmentStep::RandomFlipH { prob: 1.0 })
            .add(AugmentStep::RandomFlipV { prob: 1.0 });

        let out = pipeline.apply(&img, &mut rng).unwrap();
        assert_eq!(out.width(), 6);
        assert_eq!(out.height(), 4);
        assert_eq!(out.channels(), 3);
    }

    #[test]
    fn test_brightness_changes_pixels() {
        let data = vec![0.5f32; 4 * 4];
        let img = Image::from_raw(data, 4, 4, PixelFormat::Gray).unwrap();
        let mut rng = Rng::new(123);

        let pipeline = AugmentPipeline::new().add(AugmentStep::RandomBrightness { delta: 0.2 });

        let out = pipeline.apply(&img, &mut rng).unwrap();
        // At least some pixels should differ from 0.5.
        let changed = out.as_slice().iter().any(|&v| (v - 0.5f32).abs() > 1e-6);
        assert!(
            changed,
            "brightness augmentation should change pixel values"
        );
    }

    #[test]
    fn test_pipeline_chains() {
        let data = vec![0.5f32; 8 * 8 * 3];
        let img = Image::from_raw(data, 8, 8, PixelFormat::Rgb).unwrap();
        let mut rng = Rng::new(7);

        let pipeline = AugmentPipeline::new()
            .add(AugmentStep::RandomFlipH { prob: 0.5 })
            .add(AugmentStep::GaussianNoise { sigma: 0.01 })
            .add(AugmentStep::RandomCrop {
                width: 4,
                height: 4,
            });

        let out = pipeline.apply(&img, &mut rng).unwrap();
        assert_eq!(out.width(), 4);
        assert_eq!(out.height(), 4);
        assert_eq!(out.channels(), 3);
    }
}
