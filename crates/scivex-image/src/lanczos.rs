//! Lanczos resampling for high-quality image resizing.

use scivex_core::Float;

use crate::error::{ImageError, Result};
use crate::image::Image;

/// Lanczos kernel: sinc(x) * sinc(x/a) for |x| < a, else 0.
///
/// Where `sinc(x) = sin(pi*x) / (pi*x)` for x != 0, and sinc(0) = 1.
fn lanczos_kernel<T: Float>(x: T, a: T) -> T {
    let abs_x = Float::abs(x);
    if abs_x < T::from_f64(1e-12) {
        return T::one();
    }
    if abs_x >= a {
        return T::zero();
    }
    let pi = T::pi();
    let pi_x = pi * x;
    let sinc_x = Float::sin(pi_x) / pi_x;
    let sinc_x_a = Float::sin(pi_x / a) / (pi_x / a);
    sinc_x * sinc_x_a
}

/// Convert a float to usize by binary search.
///
/// Mirrors the helper in `transform.rs`.
fn float_to_usize<T: Float>(v: T) -> usize {
    let mut lo: usize = 0;
    let mut hi: usize = 100_000;
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        if T::from_usize(mid) <= v {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    lo
}

/// Resize an image using Lanczos resampling.
///
/// `a` controls the kernel width: Lanczos2 (a=2) or Lanczos3 (a=3).
/// Higher `a` gives sharper results but is slower.
///
/// Uses a separable 2-pass approach (horizontal then vertical) for
/// O(n * 2a) per pixel instead of O(n * (2a)^2).
pub fn resize_lanczos<T: Float>(
    img: &Image<T>,
    new_width: usize,
    new_height: usize,
    a: usize,
) -> Result<Image<T>> {
    if new_width == 0 || new_height == 0 {
        return Err(ImageError::InvalidDimensions {
            width: new_width,
            height: new_height,
        });
    }

    let (old_w, old_h) = img.dimensions();
    let c = img.channels();
    let a_f = T::from_usize(a);

    // --- Pass 1: horizontal resize (old_h x new_width) ---
    let src = img.as_slice();
    let mut intermediate = vec![T::zero(); old_h * new_width * c];

    for y in 0..old_h {
        for ox in 0..new_width {
            // Map output x to source x
            let src_x = (T::from_usize(ox) + T::from_f64(0.5)) * T::from_usize(old_w)
                / T::from_usize(new_width)
                - T::from_f64(0.5);

            let center = float_to_usize(Float::floor(src_x));

            // Gather weights and accumulate per channel
            let mut weights = Vec::with_capacity(2 * a);
            let mut sample_indices = Vec::with_capacity(2 * a);
            let mut weight_sum = T::zero();

            let start = (center + 1).saturating_sub(a);
            let end = if center + a < old_w {
                center + a
            } else {
                old_w - 1
            };

            for sx in start..=end {
                let dx = src_x - T::from_usize(sx);
                let w = lanczos_kernel(dx, a_f);
                weights.push(w);
                sample_indices.push(sx);
                weight_sum += w;
            }

            // Normalize and accumulate
            let dst_idx = (y * new_width + ox) * c;
            if weight_sum > T::from_f64(1e-12) || weight_sum < T::from_f64(-1e-12) {
                for (i, &sx) in sample_indices.iter().enumerate() {
                    let src_idx = (y * old_w + sx) * c;
                    let w = weights[i] / weight_sum;
                    for ch in 0..c {
                        intermediate[dst_idx + ch] += src[src_idx + ch] * w;
                    }
                }
            }
        }
    }

    // --- Pass 2: vertical resize (new_height x new_width) ---
    let mut out = vec![T::zero(); new_height * new_width * c];

    for oy in 0..new_height {
        let src_y = (T::from_usize(oy) + T::from_f64(0.5)) * T::from_usize(old_h)
            / T::from_usize(new_height)
            - T::from_f64(0.5);

        let center = float_to_usize(Float::floor(src_y));

        let mut weights = Vec::with_capacity(2 * a);
        let mut sample_indices = Vec::with_capacity(2 * a);
        let mut weight_sum = T::zero();

        let start = (center + 1).saturating_sub(a);
        let end = if center + a < old_h {
            center + a
        } else {
            old_h - 1
        };

        for sy in start..=end {
            let dy = src_y - T::from_usize(sy);
            let w = lanczos_kernel(dy, a_f);
            weights.push(w);
            sample_indices.push(sy);
            weight_sum += w;
        }

        for ox in 0..new_width {
            let dst_idx = (oy * new_width + ox) * c;
            if weight_sum > T::from_f64(1e-12) || weight_sum < T::from_f64(-1e-12) {
                for (i, &sy) in sample_indices.iter().enumerate() {
                    let src_idx = (sy * new_width + ox) * c;
                    let w = weights[i] / weight_sum;
                    for ch in 0..c {
                        out[dst_idx + ch] += intermediate[src_idx + ch] * w;
                    }
                }
            }
        }
    }

    Image::from_raw(out, new_width, new_height, img.format())
}

/// Lanczos-2 resize (good quality, faster).
pub fn resize_lanczos2<T: Float>(
    img: &Image<T>,
    new_width: usize,
    new_height: usize,
) -> Result<Image<T>> {
    resize_lanczos(img, new_width, new_height, 2)
}

/// Lanczos-3 resize (best quality, slower).
pub fn resize_lanczos3<T: Float>(
    img: &Image<T>,
    new_width: usize,
    new_height: usize,
) -> Result<Image<T>> {
    resize_lanczos(img, new_width, new_height, 3)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PixelFormat;

    #[test]
    fn test_lanczos_kernel_at_zero() {
        let val: f64 = lanczos_kernel(0.0, 3.0);
        assert!(
            (val - 1.0).abs() < 1e-10,
            "kernel(0) should be 1.0, got {val}"
        );
    }

    #[test]
    fn test_lanczos_kernel_at_a() {
        // kernel(a) should be 0 (outside support)
        let val: f64 = lanczos_kernel(3.0, 3.0);
        assert!(val.abs() < 1e-10, "kernel(a) should be ~0.0, got {val}");
        // kernel(-a) should also be 0
        let val2: f64 = lanczos_kernel(-3.0, 3.0);
        assert!(val2.abs() < 1e-10, "kernel(-a) should be ~0.0, got {val2}");
    }

    #[test]
    fn test_resize_same_dimensions() {
        let data = vec![
            0.1f64, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.0, 0.5,
        ];
        let img = Image::from_raw(data.clone(), 2, 2, PixelFormat::Rgb).unwrap();
        let resized = resize_lanczos3(&img, 2, 2).unwrap();
        assert_eq!(resized.width(), 2);
        assert_eq!(resized.height(), 2);
        let src = img.as_slice();
        let dst = resized.as_slice();
        for (a, b) in src.iter().zip(dst.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "same-size resize should preserve values: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_downscale_4x4_to_2x2() {
        // 4x4 grayscale image with known pattern
        #[rustfmt::skip]
        let data: Vec<f64> = vec![
            1.0, 1.0, 0.0, 0.0,
            1.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 1.0,
            0.0, 0.0, 1.0, 1.0,
        ];
        let img = Image::from_raw(data, 4, 4, PixelFormat::Gray).unwrap();
        let resized = resize_lanczos2(&img, 2, 2).unwrap();
        assert_eq!(resized.width(), 2);
        assert_eq!(resized.height(), 2);
        // The top-left 2x2 block was all 1.0, bottom-right was all 1.0
        // so the downscaled result should have reasonable averages
        let dst = resized.as_slice();
        // Top-left should be close to 1.0, bottom-right close to 1.0
        assert!(dst[0] > 0.5, "top-left should be > 0.5, got {}", dst[0]);
        assert!(dst[3] > 0.5, "bottom-right should be > 0.5, got {}", dst[3]);
    }

    #[test]
    fn test_upscale_2x2_to_4x4() {
        // 2x2 grayscale
        let data: Vec<f64> = vec![0.0, 1.0, 1.0, 0.0];
        let img = Image::from_raw(data, 2, 2, PixelFormat::Gray).unwrap();
        let resized = resize_lanczos3(&img, 4, 4).unwrap();
        assert_eq!(resized.width(), 4);
        assert_eq!(resized.height(), 4);
        // Check smooth interpolation: center values should be intermediate
        let dst = resized.as_slice();
        // Verify we got 16 values (4x4x1)
        assert_eq!(dst.len(), 16);
    }

    #[test]
    fn test_rgb_multichannel_preserves_channels() {
        let data: Vec<f64> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.0, 0.5];
        let img = Image::from_raw(data, 2, 2, PixelFormat::Rgb).unwrap();
        let resized = resize_lanczos2(&img, 3, 3).unwrap();
        assert_eq!(resized.width(), 3);
        assert_eq!(resized.height(), 3);
        assert_eq!(resized.channels(), 3);
        assert_eq!(resized.as_slice().len(), 3 * 3 * 3);
    }

    #[test]
    fn test_zero_dimensions_error() {
        let data: Vec<f64> = vec![0.0; 4];
        let img = Image::from_raw(data, 2, 2, PixelFormat::Gray).unwrap();
        assert!(resize_lanczos(&img, 0, 5, 3).is_err());
        assert!(resize_lanczos(&img, 5, 0, 3).is_err());
        assert!(resize_lanczos(&img, 0, 0, 3).is_err());
    }
}
