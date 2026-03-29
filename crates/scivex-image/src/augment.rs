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
    /// Randomly adjust brightness, contrast, saturation, and hue.
    ///
    /// Each field is a maximum delta: the actual adjustment is sampled
    /// uniformly from `[-delta, delta]`.
    ColorJitter {
        /// Maximum brightness shift.
        brightness: f64,
        /// Maximum contrast scaling delta (actual scale = 1 + uniform(-contrast, contrast)).
        contrast: f64,
        /// Maximum saturation scaling delta (actual scale = 1 + uniform(-saturation, saturation)).
        saturation: f64,
        /// Maximum hue shift in degrees.
        hue: f64,
    },
    /// Rotate the image by a random angle within `[-max_angle, max_angle]` degrees.
    ///
    /// Uses bilinear interpolation to remap pixels. Pixels that fall outside
    /// the source image are filled with zero.
    RandomRotation {
        /// Maximum rotation angle in degrees.
        max_angle: f64,
    },
    /// Erase a random rectangular region by setting pixels to zero.
    CutOut {
        /// Width of the erased rectangle.
        width: usize,
        /// Height of the erased rectangle.
        height: usize,
    },
    /// Normalize each channel with per-channel mean and standard deviation.
    ///
    /// Applies `pixel = (pixel - mean[c]) / std[c]` for each channel `c`.
    Normalize {
        /// Per-channel means.
        mean: Vec<f64>,
        /// Per-channel standard deviations.
        std: Vec<f64>,
    },
    /// Blend the image with a uniform gray image using a random lambda.
    ///
    /// `output = lambda * image + (1 - lambda) * gray`, where `lambda` is
    /// sampled from a simplified Beta-like distribution parameterised by
    /// `alpha`. The gray value is 0.5.
    MixUp {
        /// Beta distribution parameter (higher = lambda closer to 0.5).
        alpha: f64,
    },
    /// Replace a random rectangular patch with a uniform gray fill.
    ///
    /// Similar to [`CutOut`](AugmentStep::CutOut) but fills with 0.5 instead
    /// of zero. The patch size is derived from `alpha`.
    CutMix {
        /// Controls the size of the replaced patch (higher = smaller patch on average).
        alpha: f64,
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
        AugmentStep::ColorJitter {
            brightness,
            contrast,
            saturation,
            hue,
        } => apply_color_jitter(img, brightness, contrast, saturation, hue, rng),
        AugmentStep::RandomRotation { max_angle } => apply_random_rotation(img, max_angle, rng),
        AugmentStep::CutOut { width, height } => apply_cutout(img, width, height, rng),
        AugmentStep::Normalize { ref mean, ref std } => apply_normalize(img, mean, std),
        AugmentStep::MixUp { alpha } => apply_mixup(img, alpha, rng),
        AugmentStep::CutMix { alpha } => apply_cutmix(img, alpha, rng),
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

// ---------------------------------------------------------------------------
// ColorJitter
// ---------------------------------------------------------------------------

/// Randomly adjust brightness, contrast, saturation, and hue.
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
fn apply_color_jitter<T: Float>(
    img: &Image<T>,
    brightness: f64,
    contrast: f64,
    saturation: f64,
    hue: f64,
    rng: &mut Rng,
) -> Result<Image<T>> {
    let (w, h) = img.dimensions();
    let c = img.channels();
    let src = img.as_slice();
    let mut out = src.to_vec();

    // Brightness: shift all channels uniformly.
    let b_shift = (rng.next_f64() * 2.0 - 1.0) * brightness;
    for v in &mut out {
        *v += T::from_f64(b_shift);
    }

    // Contrast: scale each pixel relative to channel mean.
    let c_scale = 1.0 + (rng.next_f64() * 2.0 - 1.0) * contrast;
    // Compute per-channel mean.
    let total_pixels = w * h;
    if total_pixels > 0 && c > 0 {
        let mut means = vec![0.0f64; c];
        for i in 0..total_pixels {
            for ch in 0..c {
                means[ch] += out[i * c + ch].to_f64();
            }
        }
        for m in &mut means {
            *m /= total_pixels as f64;
        }
        for i in 0..total_pixels {
            for (ch, &mean) in means.iter().enumerate() {
                let idx = i * c + ch;
                let val = out[idx].to_f64();
                out[idx] = T::from_f64(mean + (val - mean) * c_scale);
            }
        }
    }

    // Saturation: for multi-channel images, scale distance from grayscale.
    if c >= 3 {
        let s_scale = 1.0 + (rng.next_f64() * 2.0 - 1.0) * saturation;
        for i in 0..total_pixels {
            let base = i * c;
            let gray = (out[base].to_f64() + out[base + 1].to_f64() + out[base + 2].to_f64()) / 3.0;
            for ch in 0..3 {
                let val = out[base + ch].to_f64();
                out[base + ch] = T::from_f64(gray + (val - gray) * s_scale);
            }
        }
    }

    // Hue: for multi-channel images, rotate hue in RGB space.
    if c >= 3 {
        let h_shift = (rng.next_f64() * 2.0 - 1.0) * hue;
        let angle = h_shift * std::f64::consts::PI / 180.0;
        let cos_a = angle.cos();
        let sin_a = angle.sin();
        // Simplified hue rotation matrix for RGB.
        let sqrt3 = (3.0f64).sqrt();
        for i in 0..total_pixels {
            let base = i * c;
            let r = out[base].to_f64();
            let g = out[base + 1].to_f64();
            let b = out[base + 2].to_f64();
            let nr = r * (cos_a + (1.0 - cos_a) / 3.0)
                + g * ((1.0 - cos_a) / 3.0 - sqrt3 * sin_a / 3.0)
                + b * ((1.0 - cos_a) / 3.0 + sqrt3 * sin_a / 3.0);
            let ng = r * ((1.0 - cos_a) / 3.0 + sqrt3 * sin_a / 3.0)
                + g * (cos_a + (1.0 - cos_a) / 3.0)
                + b * ((1.0 - cos_a) / 3.0 - sqrt3 * sin_a / 3.0);
            let nb = r * ((1.0 - cos_a) / 3.0 - sqrt3 * sin_a / 3.0)
                + g * ((1.0 - cos_a) / 3.0 + sqrt3 * sin_a / 3.0)
                + b * (cos_a + (1.0 - cos_a) / 3.0);
            out[base] = T::from_f64(nr);
            out[base + 1] = T::from_f64(ng);
            out[base + 2] = T::from_f64(nb);
        }
    }

    Image::from_raw(out, w, h, img.format())
}

// ---------------------------------------------------------------------------
// RandomRotation
// ---------------------------------------------------------------------------

/// Rotate image by a random angle using bilinear interpolation.
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
fn apply_random_rotation<T: Float>(
    img: &Image<T>,
    max_angle: f64,
    rng: &mut Rng,
) -> Result<Image<T>> {
    let (w, h) = img.dimensions();
    let c = img.channels();
    let src = img.as_slice();

    let angle_deg = (rng.next_f64() * 2.0 - 1.0) * max_angle;
    let angle = angle_deg * std::f64::consts::PI / 180.0;
    let cos_a = angle.cos();
    let sin_a = angle.sin();

    let cx = w as f64 / 2.0;
    let cy = h as f64 / 2.0;

    let mut out = vec![T::zero(); src.len()];

    for row in 0..h {
        for col in 0..w {
            // Map destination to source (inverse rotation).
            let dx = col as f64 - cx;
            let dy = row as f64 - cy;
            let sx = dx * cos_a + dy * sin_a + cx;
            let sy = -dx * sin_a + dy * cos_a + cy;

            // Bilinear interpolation.
            let x0 = sx.floor();
            let y0 = sy.floor();
            let fx = sx - x0;
            let fy = sy - y0;
            let ix0 = x0 as i64;
            let iy0 = y0 as i64;

            for ch in 0..c {
                let mut val = 0.0;
                for (dy_off, wy) in [(0i64, 1.0 - fy), (1, fy)] {
                    for (dx_off, wx) in [(0i64, 1.0 - fx), (1, fx)] {
                        let px = ix0 + dx_off;
                        let py = iy0 + dy_off;
                        #[allow(clippy::cast_possible_wrap)]
                        if px >= 0 && px < w as i64 && py >= 0 && py < h as i64 {
                            let idx = (py as usize * w + px as usize) * c + ch;
                            val += src[idx].to_f64() * wx * wy;
                        }
                    }
                }
                out[(row * w + col) * c + ch] = T::from_f64(val);
            }
        }
    }

    Image::from_raw(out, w, h, img.format())
}

// ---------------------------------------------------------------------------
// CutOut
// ---------------------------------------------------------------------------

/// Erase a random rectangular region by setting to zero.
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
fn apply_cutout<T: Float>(
    img: &Image<T>,
    cut_w: usize,
    cut_h: usize,
    rng: &mut Rng,
) -> Result<Image<T>> {
    let (w, h) = img.dimensions();
    let c = img.channels();
    let mut out = img.as_slice().to_vec();

    // Clamp cut dimensions to image size.
    let cw = cut_w.min(w);
    let ch = cut_h.min(h);

    let x0 = if w > cw {
        ((rng.next_f64() * (w - cw + 1) as f64) as usize).min(w - cw)
    } else {
        0
    };
    let y0 = if h > ch {
        ((rng.next_f64() * (h - ch + 1) as f64) as usize).min(h - ch)
    } else {
        0
    };

    for row in y0..y0 + ch {
        for col in x0..x0 + cw {
            let base = (row * w + col) * c;
            for k in 0..c {
                out[base + k] = T::zero();
            }
        }
    }

    Image::from_raw(out, w, h, img.format())
}

// ---------------------------------------------------------------------------
// Normalize
// ---------------------------------------------------------------------------

/// Normalize with per-channel mean and std: `pixel = (pixel - mean[c]) / std[c]`.
fn apply_normalize<T: Float>(img: &Image<T>, mean: &[f64], std: &[f64]) -> Result<Image<T>> {
    let c = img.channels();
    if mean.len() != c || std.len() != c {
        return Err(ImageError::InvalidParameter {
            name: "normalize mean/std",
            reason: "mean and std vectors must have one element per channel",
        });
    }
    let (w, h) = img.dimensions();
    let src = img.as_slice();
    let mut out = Vec::with_capacity(src.len());

    for i in 0..w * h {
        for ch in 0..c {
            let val = src[i * c + ch].to_f64();
            let normalized = (val - mean[ch]) / std[ch];
            out.push(T::from_f64(normalized));
        }
    }

    Image::from_raw(out, w, h, img.format())
}

// ---------------------------------------------------------------------------
// MixUp
// ---------------------------------------------------------------------------

/// Blend the image with a uniform gray (0.5) image.
///
/// Lambda is sampled from a simplified Beta-like distribution: we average
/// `alpha` uniform samples (central-limit approximation) to concentrate
/// lambda around 0.5 when alpha is large.
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
fn apply_mixup<T: Float>(img: &Image<T>, alpha: f64, rng: &mut Rng) -> Result<Image<T>> {
    // Sample lambda using a simple Beta-like approach:
    // average of `n` uniform samples, where n = max(1, round(alpha)).
    let n = (alpha.round() as usize).max(1);
    let lambda: f64 = (0..n).map(|_| rng.next_f64()).sum::<f64>() / n as f64;

    let gray = 0.5;
    let (w, h) = img.dimensions();
    let src = img.as_slice();
    let out: Vec<T> = src
        .iter()
        .map(|&v| {
            let blended = lambda * v.to_f64() + (1.0 - lambda) * gray;
            T::from_f64(blended)
        })
        .collect();

    Image::from_raw(out, w, h, img.format())
}

// ---------------------------------------------------------------------------
// CutMix
// ---------------------------------------------------------------------------

/// Replace a random rectangular patch with a uniform gray (0.5) fill.
///
/// The patch area ratio is `1 - lambda`, where lambda is sampled similarly
/// to [`apply_mixup`]. The patch width/height are derived from the area ratio.
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
fn apply_cutmix<T: Float>(img: &Image<T>, alpha: f64, rng: &mut Rng) -> Result<Image<T>> {
    let (w, h) = img.dimensions();
    let c = img.channels();

    // Sample lambda.
    let n = (alpha.round() as usize).max(1);
    let lambda: f64 = (0..n).map(|_| rng.next_f64()).sum::<f64>() / n as f64;

    // Patch area ratio = 1 - lambda.
    let ratio = (1.0 - lambda).sqrt();
    let cut_w = ((w as f64 * ratio) as usize).max(1).min(w);
    let cut_h = ((h as f64 * ratio) as usize).max(1).min(h);

    let x0 = if w > cut_w {
        ((rng.next_f64() * (w - cut_w + 1) as f64) as usize).min(w - cut_w)
    } else {
        0
    };
    let y0 = if h > cut_h {
        ((rng.next_f64() * (h - cut_h + 1) as f64) as usize).min(h - cut_h)
    } else {
        0
    };

    let gray = T::from_f64(0.5);
    let mut out = img.as_slice().to_vec();

    for row in y0..y0 + cut_h {
        for col in x0..x0 + cut_w {
            let base = (row * w + col) * c;
            for k in 0..c {
                out[base + k] = gray;
            }
        }
    }

    Image::from_raw(out, w, h, img.format())
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

    // ------------------------------------------------------------------
    // ColorJitter tests
    // ------------------------------------------------------------------

    #[test]
    fn test_color_jitter_preserves_dimensions() {
        let data = vec![0.5f32; 4 * 4 * 3];
        let img = Image::from_raw(data, 4, 4, PixelFormat::Rgb).unwrap();
        let mut rng = Rng::new(99);

        let pipeline = AugmentPipeline::new().add(AugmentStep::ColorJitter {
            brightness: 0.2,
            contrast: 0.2,
            saturation: 0.2,
            hue: 10.0,
        });

        let out = pipeline.apply(&img, &mut rng).unwrap();
        assert_eq!(out.dimensions(), (4, 4));
        assert_eq!(out.channels(), 3);
    }

    #[test]
    fn test_color_jitter_changes_pixels() {
        let data: Vec<f32> = (0..4 * 4 * 3).map(|i| (i as f32) / 48.0).collect();
        let img = Image::from_raw(data, 4, 4, PixelFormat::Rgb).unwrap();
        let mut rng = Rng::new(42);

        let out = apply_color_jitter(&img, 0.1, 0.1, 0.1, 5.0, &mut rng).unwrap();
        // With non-zero deltas and varied input, output should differ.
        let any_changed = img
            .as_slice()
            .iter()
            .zip(out.as_slice().iter())
            .any(|(&a, &b)| (a - b).abs() > 1e-7);
        assert!(
            any_changed,
            "color jitter should change at least some pixels"
        );
    }

    #[test]
    fn test_color_jitter_zero_deltas_preserves() {
        let data = vec![0.3f32; 2 * 2 * 3];
        let img = Image::from_raw(data.clone(), 2, 2, PixelFormat::Rgb).unwrap();
        let mut rng = Rng::new(0);

        let out = apply_color_jitter(&img, 0.0, 0.0, 0.0, 0.0, &mut rng).unwrap();
        for (&a, &b) in img.as_slice().iter().zip(out.as_slice().iter()) {
            assert!((a - b).abs() < 1e-6, "zero jitter should preserve pixels");
        }
    }

    // ------------------------------------------------------------------
    // RandomRotation tests
    // ------------------------------------------------------------------

    #[test]
    fn test_random_rotation_preserves_dimensions() {
        let data = vec![0.5f32; 6 * 6 * 3];
        let img = Image::from_raw(data, 6, 6, PixelFormat::Rgb).unwrap();
        let mut rng = Rng::new(55);

        let pipeline = AugmentPipeline::new().add(AugmentStep::RandomRotation { max_angle: 30.0 });
        let out = pipeline.apply(&img, &mut rng).unwrap();
        assert_eq!(out.dimensions(), (6, 6));
        assert_eq!(out.channels(), 3);
    }

    #[test]
    fn test_random_rotation_zero_angle_preserves() {
        // With max_angle=0 the rotation is identity so pixels should be unchanged.
        let data: Vec<f32> = (0..4 * 4).map(|i| i as f32 / 16.0).collect();
        let img = Image::from_raw(data, 4, 4, PixelFormat::Gray).unwrap();
        let mut rng = Rng::new(0);

        let out = apply_random_rotation(&img, 0.0, &mut rng).unwrap();
        for (&a, &b) in img.as_slice().iter().zip(out.as_slice().iter()) {
            assert!((a - b).abs() < 1e-5, "zero rotation should preserve pixels");
        }
    }

    #[test]
    fn test_random_rotation_changes_pixels() {
        let data: Vec<f32> = (0..8 * 8).map(|i| i as f32 / 64.0).collect();
        let img = Image::from_raw(data, 8, 8, PixelFormat::Gray).unwrap();
        let mut rng = Rng::new(77);

        let out = apply_random_rotation(&img, 45.0, &mut rng).unwrap();
        let any_changed = img
            .as_slice()
            .iter()
            .zip(out.as_slice().iter())
            .any(|(&a, &b)| (a - b).abs() > 1e-5);
        assert!(any_changed, "rotation should change some pixels");
    }

    // ------------------------------------------------------------------
    // CutOut tests
    // ------------------------------------------------------------------

    #[test]
    fn test_cutout_preserves_dimensions() {
        let data = vec![0.8f32; 6 * 6 * 3];
        let img = Image::from_raw(data, 6, 6, PixelFormat::Rgb).unwrap();
        let mut rng = Rng::new(11);

        let pipeline = AugmentPipeline::new().add(AugmentStep::CutOut {
            width: 2,
            height: 2,
        });
        let out = pipeline.apply(&img, &mut rng).unwrap();
        assert_eq!(out.dimensions(), (6, 6));
    }

    #[test]
    fn test_cutout_zeros_pixels() {
        let data = vec![1.0f32; 4 * 4];
        let img = Image::from_raw(data, 4, 4, PixelFormat::Gray).unwrap();
        let mut rng = Rng::new(33);

        let out = apply_cutout(&img, 2, 2, &mut rng).unwrap();
        // There should be exactly 4 zeroed pixels (2x2 patch).
        let zero_count = out.as_slice().iter().filter(|&&v| v == 0.0).count();
        assert_eq!(zero_count, 4, "cutout should zero a 2x2 region");
    }

    #[test]
    fn test_cutout_larger_than_image_clamps() {
        let data = vec![1.0f32; 2 * 2];
        let img = Image::from_raw(data, 2, 2, PixelFormat::Gray).unwrap();
        let mut rng = Rng::new(0);

        // CutOut larger than image: should clamp and zero entire image.
        let out = apply_cutout(&img, 10, 10, &mut rng).unwrap();
        assert!(out.as_slice().iter().all(|&v| v == 0.0));
    }

    // ------------------------------------------------------------------
    // Normalize tests
    // ------------------------------------------------------------------

    #[test]
    fn test_normalize_basic() {
        // 2x2 gray image, all 0.5. mean=0.5, std=0.25 -> (0.5-0.5)/0.25 = 0.0
        let data = vec![0.5f32; 2 * 2];
        let img = Image::from_raw(data, 2, 2, PixelFormat::Gray).unwrap();
        let mut rng = Rng::new(0);

        let pipeline = AugmentPipeline::new().add(AugmentStep::Normalize {
            mean: vec![0.5],
            std: vec![0.25],
        });
        let out = pipeline.apply(&img, &mut rng).unwrap();
        for &v in out.as_slice() {
            assert!((v - 0.0f32).abs() < 1e-6);
        }
    }

    #[test]
    fn test_normalize_rgb() {
        // 1x1 RGB pixel = [0.6, 0.4, 0.8]
        let data = vec![0.6f32, 0.4, 0.8];
        let img = Image::from_raw(data, 1, 1, PixelFormat::Rgb).unwrap();

        let mean = vec![0.5, 0.5, 0.5];
        let std = vec![0.2, 0.2, 0.2];
        let out = apply_normalize(&img, &mean, &std).unwrap();
        let s = out.as_slice();
        assert!((s[0] - 0.5).abs() < 1e-5); // (0.6 - 0.5) / 0.2 = 0.5
        assert!((s[1] - (-0.5)).abs() < 1e-5); // (0.4 - 0.5) / 0.2 = -0.5
        assert!((s[2] - 1.5).abs() < 1e-5); // (0.8 - 0.5) / 0.2 = 1.5
    }

    #[test]
    fn test_normalize_wrong_channel_count() {
        let data = vec![0.5f32; 2 * 2 * 3];
        let img = Image::from_raw(data, 2, 2, PixelFormat::Rgb).unwrap();

        // Provide only 1 mean/std for a 3-channel image.
        let result = apply_normalize(&img, &[0.5], &[0.25]);
        assert!(result.is_err());
    }

    // ------------------------------------------------------------------
    // MixUp tests
    // ------------------------------------------------------------------

    #[test]
    fn test_mixup_preserves_dimensions() {
        let data = vec![0.8f32; 4 * 4 * 3];
        let img = Image::from_raw(data, 4, 4, PixelFormat::Rgb).unwrap();
        let mut rng = Rng::new(10);

        let pipeline = AugmentPipeline::new().add(AugmentStep::MixUp { alpha: 1.0 });
        let out = pipeline.apply(&img, &mut rng).unwrap();
        assert_eq!(out.dimensions(), (4, 4));
        assert_eq!(out.channels(), 3);
    }

    #[test]
    fn test_mixup_blends_toward_gray() {
        // All white (1.0) image blended with gray (0.5) -> values should be between 0.5 and 1.0.
        let data = vec![1.0f32; 4 * 4];
        let img = Image::from_raw(data, 4, 4, PixelFormat::Gray).unwrap();
        let mut rng = Rng::new(42);

        let out = apply_mixup(&img, 1.0, &mut rng).unwrap();
        for &v in out.as_slice() {
            assert!(
                (0.5 - 1e-6..=1.0 + 1e-6).contains(&v),
                "mixed value {v} out of range"
            );
        }
    }

    #[test]
    fn test_mixup_with_black_image() {
        // All zero image blended with gray (0.5) -> values should be between 0.0 and 0.5.
        let data = vec![0.0f32; 3 * 3];
        let img = Image::from_raw(data, 3, 3, PixelFormat::Gray).unwrap();
        let mut rng = Rng::new(7);

        let out = apply_mixup(&img, 2.0, &mut rng).unwrap();
        for &v in out.as_slice() {
            assert!(
                (-1e-6..=0.5 + 1e-6).contains(&v),
                "mixed value {v} out of range"
            );
        }
    }

    // ------------------------------------------------------------------
    // CutMix tests
    // ------------------------------------------------------------------

    #[test]
    fn test_cutmix_preserves_dimensions() {
        let data = vec![0.2f32; 6 * 6 * 3];
        let img = Image::from_raw(data, 6, 6, PixelFormat::Rgb).unwrap();
        let mut rng = Rng::new(50);

        let pipeline = AugmentPipeline::new().add(AugmentStep::CutMix { alpha: 1.0 });
        let out = pipeline.apply(&img, &mut rng).unwrap();
        assert_eq!(out.dimensions(), (6, 6));
        assert_eq!(out.channels(), 3);
    }

    #[test]
    fn test_cutmix_introduces_gray() {
        let data = vec![0.0f32; 8 * 8];
        let img = Image::from_raw(data, 8, 8, PixelFormat::Gray).unwrap();
        let mut rng = Rng::new(22);

        let out = apply_cutmix(&img, 1.0, &mut rng).unwrap();
        // Some pixels should now be 0.5 (gray fill).
        let gray_count = out
            .as_slice()
            .iter()
            .filter(|&&v| (v - 0.5).abs() < 1e-6)
            .count();
        assert!(gray_count > 0, "cutmix should introduce gray pixels");
    }

    #[test]
    fn test_cutmix_non_gray_pixels_unchanged() {
        let data = vec![0.9f32; 4 * 4 * 3];
        let img = Image::from_raw(data, 4, 4, PixelFormat::Rgb).unwrap();
        let mut rng = Rng::new(88);

        let out = apply_cutmix(&img, 1.0, &mut rng).unwrap();
        // Every pixel should be either 0.9 (original) or 0.5 (gray fill).
        for &v in out.as_slice() {
            assert!(
                (v - 0.9).abs() < 1e-6 || (v - 0.5).abs() < 1e-6,
                "unexpected pixel value: {v}"
            );
        }
    }

    // ------------------------------------------------------------------
    // Pipeline integration with new steps
    // ------------------------------------------------------------------

    #[test]
    fn test_full_pipeline_with_new_steps() {
        let data: Vec<f32> = (0..8 * 8 * 3).map(|i| (i as f32) / 192.0).collect();
        let img = Image::from_raw(data, 8, 8, PixelFormat::Rgb).unwrap();
        let mut rng = Rng::new(2025);

        let pipeline = AugmentPipeline::new()
            .add(AugmentStep::ColorJitter {
                brightness: 0.1,
                contrast: 0.1,
                saturation: 0.1,
                hue: 5.0,
            })
            .add(AugmentStep::RandomRotation { max_angle: 15.0 })
            .add(AugmentStep::CutOut {
                width: 2,
                height: 2,
            })
            .add(AugmentStep::Normalize {
                mean: vec![0.5, 0.5, 0.5],
                std: vec![0.25, 0.25, 0.25],
            });

        let out = pipeline.apply(&img, &mut rng).unwrap();
        assert_eq!(out.dimensions(), (8, 8));
        assert_eq!(out.channels(), 3);
    }
}
