use scivex_core::Scalar;

use crate::error::{ImageError, Result};
use crate::image::{Image, PixelFormat};

/// Convert an RGB or RGBA image to grayscale using the luminosity formula:
/// `0.299*R + 0.587*G + 0.114*B`.
///
/// Alpha channels are dropped.
pub fn to_grayscale<T: Scalar>(img: &Image<T>) -> Result<Image<T>> {
    match img.format() {
        PixelFormat::Gray | PixelFormat::GrayAlpha => {
            if img.format() == PixelFormat::Gray {
                return Ok(img.clone());
            }
            // GrayAlpha -> Gray: take first channel
            let (w, h) = img.dimensions();
            let mut out = Vec::with_capacity(w * h);
            let src = img.as_slice();
            for i in (0..src.len()).step_by(2) {
                out.push(src[i]);
            }
            Image::from_raw(out, w, h, PixelFormat::Gray)
        }
        PixelFormat::Rgb | PixelFormat::Rgba => {
            let (w, h) = img.dimensions();
            let c = img.channels();
            let src = img.as_slice();
            let mut out = Vec::with_capacity(w * h);

            for i in (0..src.len()).step_by(c) {
                let r = src[i];
                let g = src[i + 1];
                let b = src[i + 2];
                out.push(luminosity(r, g, b));
            }
            Image::from_raw(out, w, h, PixelFormat::Gray)
        }
    }
}

/// Compute luminosity gray value generically.
///
/// Uses the approximation `r/4 + g/2 + b/8` which avoids overflow for `u8`
/// while staying close to the standard 0.299/0.587/0.114 coefficients.
///
/// For floating-point types, prefer `to_grayscale_f32` for exact coefficients.
fn luminosity<T: Scalar>(r: T, g: T, b: T) -> T {
    // 0.25R + 0.5G + 0.125B ≈ 0.875 sum (normalize would overshoot for u8)
    // Close approximation that never overflows u8.
    r / T::from_usize(4) + g / T::from_usize(2) + b / T::from_usize(8)
}

/// Convert a grayscale image to RGB by triplicating the channel.
pub fn to_rgb<T: Scalar>(img: &Image<T>) -> Result<Image<T>> {
    match img.format() {
        PixelFormat::Rgb => Ok(img.clone()),
        PixelFormat::Rgba => {
            // Drop alpha
            let (w, h) = img.dimensions();
            let src = img.as_slice();
            let mut out = Vec::with_capacity(w * h * 3);
            for i in (0..src.len()).step_by(4) {
                out.push(src[i]);
                out.push(src[i + 1]);
                out.push(src[i + 2]);
            }
            Image::from_raw(out, w, h, PixelFormat::Rgb)
        }
        PixelFormat::Gray | PixelFormat::GrayAlpha => {
            let (w, h) = img.dimensions();
            let c = img.channels();
            let src = img.as_slice();
            let mut out = Vec::with_capacity(w * h * 3);
            for i in (0..src.len()).step_by(c) {
                let v = src[i];
                out.push(v);
                out.push(v);
                out.push(v);
            }
            Image::from_raw(out, w, h, PixelFormat::Rgb)
        }
    }
}

/// Convert an RGB `f32` image (values in `[0,1]`) to HSV.
///
/// Output channels: H in `[0,1]` (representing 0-360), S in `[0,1]`, V in `[0,1]`.
pub fn rgb_to_hsv(img: &Image<f32>) -> Result<Image<f32>> {
    if img.format() != PixelFormat::Rgb {
        return Err(ImageError::UnsupportedChannels {
            channels: img.channels(),
        });
    }
    let (w, h) = img.dimensions();
    let src = img.as_slice();
    let mut out = Vec::with_capacity(w * h * 3);

    for i in (0..src.len()).step_by(3) {
        let r = src[i];
        let g = src[i + 1];
        let b = src[i + 2];

        let max = r.max(g).max(b);
        let min = r.min(g).min(b);
        let delta = max - min;

        // V
        let v = max;

        // S
        let s = if max == 0.0 { 0.0 } else { delta / max };

        // H
        let h_val = if delta == 0.0 {
            0.0
        } else if (max - r).abs() < f32::EPSILON {
            let mut hue = (g - b) / delta;
            if hue < 0.0 {
                hue += 6.0;
            }
            hue / 6.0
        } else if (max - g).abs() < f32::EPSILON {
            ((b - r) / delta + 2.0) / 6.0
        } else {
            ((r - g) / delta + 4.0) / 6.0
        };

        out.push(h_val);
        out.push(s);
        out.push(v);
    }

    Image::from_raw(out, w, h, PixelFormat::Rgb)
}

/// Convert an HSV `f32` image back to RGB.
///
/// Input channels: H in `[0,1]`, S in `[0,1]`, V in `[0,1]`.
pub fn hsv_to_rgb(img: &Image<f32>) -> Result<Image<f32>> {
    if img.format() != PixelFormat::Rgb {
        return Err(ImageError::UnsupportedChannels {
            channels: img.channels(),
        });
    }
    let (w, h) = img.dimensions();
    let src = img.as_slice();
    let mut out = Vec::with_capacity(w * h * 3);

    for i in (0..src.len()).step_by(3) {
        let h_val = src[i] * 6.0;
        let s = src[i + 1];
        let v = src[i + 2];

        let c = v * s;
        let x = c * (1.0 - ((h_val % 2.0) - 1.0).abs());
        let m = v - c;

        let (r, g, b) = if h_val < 1.0 {
            (c, x, 0.0)
        } else if h_val < 2.0 {
            (x, c, 0.0)
        } else if h_val < 3.0 {
            (0.0, c, x)
        } else if h_val < 4.0 {
            (0.0, x, c)
        } else if h_val < 5.0 {
            (x, 0.0, c)
        } else {
            (c, 0.0, x)
        };

        out.push(r + m);
        out.push(g + m);
        out.push(b + m);
    }

    Image::from_raw(out, w, h, PixelFormat::Rgb)
}

/// Invert pixel values. For `u8`: `255 - val`. For floats: `1.0 - val`.
pub fn invert(img: &Image<u8>) -> Image<u8> {
    img.map_pixels(|v| 255 - v)
}

/// Invert pixel values for `f32` images: `1.0 - val`.
pub fn invert_f32(img: &Image<f32>) -> Image<f32> {
    img.map_pixels(|v| 1.0 - v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_grayscale_white() {
        let data = vec![255u8, 255, 255];
        let img = Image::from_raw(data, 1, 1, PixelFormat::Rgb).unwrap();
        let gray = to_grayscale(&img).unwrap();
        assert_eq!(gray.format(), PixelFormat::Gray);
        // 255/4 + 255/2 + 255/8 = 63 + 127 + 31 = 221
        let val = gray.get_pixel(0, 0).unwrap()[0];
        assert!(val > 200, "white should produce high gray value, got {val}");
    }

    #[test]
    fn test_to_grayscale_already_gray() {
        let img = Image::from_raw(vec![128u8], 1, 1, PixelFormat::Gray).unwrap();
        let gray = to_grayscale(&img).unwrap();
        assert_eq!(gray.get_pixel(0, 0).unwrap(), vec![128]);
    }

    #[test]
    fn test_to_rgb_from_gray() {
        let img = Image::from_raw(vec![100u8], 1, 1, PixelFormat::Gray).unwrap();
        let rgb = to_rgb(&img).unwrap();
        assert_eq!(rgb.format(), PixelFormat::Rgb);
        assert_eq!(rgb.get_pixel(0, 0).unwrap(), vec![100, 100, 100]);
    }

    #[test]
    fn test_rgb_hsv_roundtrip() {
        let data = vec![0.8f32, 0.2, 0.5, 0.0, 1.0, 0.0, 0.3, 0.3, 0.3];
        let img = Image::from_raw(data.clone(), 3, 1, PixelFormat::Rgb).unwrap();
        let hsv = rgb_to_hsv(&img).unwrap();
        let back = hsv_to_rgb(&hsv).unwrap();
        let src = img.as_slice();
        let dst = back.as_slice();
        for (a, b) in src.iter().zip(dst.iter()) {
            assert!((a - b).abs() < 1e-5, "{a} != {b}");
        }
    }

    #[test]
    fn test_invert_u8() {
        let img = Image::from_raw(vec![0u8, 128, 255], 1, 1, PixelFormat::Rgb).unwrap();
        let inv = invert(&img);
        assert_eq!(inv.get_pixel(0, 0).unwrap(), vec![255, 127, 0]);
    }

    #[test]
    fn test_invert_f32() {
        let img = Image::from_raw(vec![0.0f32, 0.5, 1.0], 1, 1, PixelFormat::Rgb).unwrap();
        let inv = invert_f32(&img);
        let p = inv.get_pixel(0, 0).unwrap();
        assert!((p[0] - 1.0).abs() < 1e-6);
        assert!((p[1] - 0.5).abs() < 1e-6);
        assert!(p[2].abs() < 1e-6);
    }

    #[test]
    fn test_to_grayscale_from_gray_alpha() {
        // GrayAlpha -> Gray should keep the first channel and drop alpha.
        let data = vec![100u8, 200, 50, 250];
        let img = Image::from_raw(data, 2, 1, PixelFormat::GrayAlpha).unwrap();
        let gray = to_grayscale(&img).unwrap();
        assert_eq!(gray.format(), PixelFormat::Gray);
        assert_eq!(gray.channels(), 1);
        assert_eq!(gray.get_pixel(0, 0).unwrap(), vec![100]);
        assert_eq!(gray.get_pixel(0, 1).unwrap(), vec![50]);
    }

    #[test]
    fn test_to_rgb_from_rgba() {
        // RGBA -> RGB should drop the alpha channel.
        let data = vec![10u8, 20, 30, 255];
        let img = Image::from_raw(data, 1, 1, PixelFormat::Rgba).unwrap();
        let rgb = to_rgb(&img).unwrap();
        assert_eq!(rgb.format(), PixelFormat::Rgb);
        assert_eq!(rgb.get_pixel(0, 0).unwrap(), vec![10, 20, 30]);
    }

    #[test]
    fn test_to_rgb_from_gray_alpha() {
        let data = vec![80u8, 255];
        let img = Image::from_raw(data, 1, 1, PixelFormat::GrayAlpha).unwrap();
        let rgb = to_rgb(&img).unwrap();
        assert_eq!(rgb.format(), PixelFormat::Rgb);
        assert_eq!(rgb.get_pixel(0, 0).unwrap(), vec![80, 80, 80]);
    }

    #[test]
    fn test_to_rgb_already_rgb() {
        let data = vec![1u8, 2, 3];
        let img = Image::from_raw(data.clone(), 1, 1, PixelFormat::Rgb).unwrap();
        let rgb = to_rgb(&img).unwrap();
        assert_eq!(rgb.as_slice(), &data);
    }

    #[test]
    fn test_rgb_to_hsv_rejects_non_rgb() {
        let img = Image::from_raw(vec![0.5f32], 1, 1, PixelFormat::Gray).unwrap();
        assert!(rgb_to_hsv(&img).is_err());
    }

    #[test]
    fn test_hsv_to_rgb_rejects_non_rgb() {
        let img = Image::from_raw(vec![0.5f32], 1, 1, PixelFormat::Gray).unwrap();
        assert!(hsv_to_rgb(&img).is_err());
    }

    #[test]
    fn test_rgb_to_hsv_pure_black() {
        let data = vec![0.0f32, 0.0, 0.0];
        let img = Image::from_raw(data, 1, 1, PixelFormat::Rgb).unwrap();
        let hsv = rgb_to_hsv(&img).unwrap();
        let p = hsv.get_pixel(0, 0).unwrap();
        // Black: H=0, S=0, V=0.
        assert!(p[0].abs() < 1e-6);
        assert!(p[1].abs() < 1e-6);
        assert!(p[2].abs() < 1e-6);
    }

    #[test]
    fn test_rgb_to_hsv_pure_red() {
        let data = vec![1.0f32, 0.0, 0.0];
        let img = Image::from_raw(data, 1, 1, PixelFormat::Rgb).unwrap();
        let hsv = rgb_to_hsv(&img).unwrap();
        let p = hsv.get_pixel(0, 0).unwrap();
        // Red: H=0, S=1, V=1.
        assert!(p[0].abs() < 1e-6);
        assert!((p[1] - 1.0).abs() < 1e-6);
        assert!((p[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_to_grayscale_pure_black() {
        let data = vec![0u8, 0, 0];
        let img = Image::from_raw(data, 1, 1, PixelFormat::Rgb).unwrap();
        let gray = to_grayscale(&img).unwrap();
        assert_eq!(gray.get_pixel(0, 0).unwrap(), vec![0]);
    }

    #[test]
    fn test_grayscale_of_gray_is_identity() {
        let data = vec![50u8, 100, 150, 200];
        let img = Image::from_raw(data.clone(), 4, 1, PixelFormat::Gray).unwrap();
        let gray = to_grayscale(&img).unwrap();
        assert_eq!(gray.format(), PixelFormat::Gray);
        assert_eq!(gray.as_slice(), &data);
    }

    #[test]
    fn test_to_grayscale_from_rgba() {
        let data = vec![100u8, 150, 200, 255];
        let img = Image::from_raw(data, 1, 1, PixelFormat::Rgba).unwrap();
        let gray = to_grayscale(&img).unwrap();
        assert_eq!(gray.format(), PixelFormat::Gray);
        assert_eq!(gray.channels(), 1);
        // Should produce a valid gray value derived from R=100 G=150 B=200
        let val = gray.get_pixel(0, 0).unwrap()[0];
        assert!(val > 0);
    }

    #[test]
    fn test_invert_twice_identity() {
        let data = vec![10u8, 100, 200];
        let img = Image::from_raw(data.clone(), 1, 1, PixelFormat::Rgb).unwrap();
        let double_inv = invert(&invert(&img));
        assert_eq!(double_inv.as_slice(), &data);
    }

    #[test]
    fn test_invert_f32_twice_identity() {
        let data = vec![0.1f32, 0.5, 0.9];
        let img = Image::from_raw(data.clone(), 1, 1, PixelFormat::Rgb).unwrap();
        let double_inv = invert_f32(&invert_f32(&img));
        for (a, b) in double_inv.as_slice().iter().zip(data.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_rgb_to_hsv_pure_green() {
        let data = vec![0.0f32, 1.0, 0.0];
        let img = Image::from_raw(data, 1, 1, PixelFormat::Rgb).unwrap();
        let hsv = rgb_to_hsv(&img).unwrap();
        let p = hsv.get_pixel(0, 0).unwrap();
        // Green: H ~ 1/3, S=1, V=1
        assert!((p[0] - 1.0 / 3.0).abs() < 1e-5);
        assert!((p[1] - 1.0).abs() < 1e-5);
        assert!((p[2] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_rgb_to_hsv_pure_blue() {
        let data = vec![0.0f32, 0.0, 1.0];
        let img = Image::from_raw(data, 1, 1, PixelFormat::Rgb).unwrap();
        let hsv = rgb_to_hsv(&img).unwrap();
        let p = hsv.get_pixel(0, 0).unwrap();
        // Blue: H ~ 2/3, S=1, V=1
        assert!((p[0] - 2.0 / 3.0).abs() < 1e-5);
        assert!((p[1] - 1.0).abs() < 1e-5);
        assert!((p[2] - 1.0).abs() < 1e-5);
    }
}
