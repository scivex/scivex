use scivex_core::tensor::SliceRange;
use scivex_core::{Float, Scalar};

use crate::error::{ImageError, Result};
use crate::image::Image;

/// Resize interpolation method.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResizeMethod {
    /// Nearest-neighbor interpolation.
    Nearest,
    /// Bilinear interpolation (requires `Float` types).
    Bilinear,
}

/// Resize an image to the given dimensions using nearest-neighbor interpolation.
///
/// # Examples
///
/// ```
/// # use scivex_image::{Image, PixelFormat};
/// # use scivex_image::transform::{resize, ResizeMethod};
/// let img = Image::from_raw(vec![10u8, 20, 30, 40], 2, 2, PixelFormat::Gray).unwrap();
/// let resized = resize(&img, 4, 4, ResizeMethod::Nearest).unwrap();
/// assert_eq!(resized.width(), 4);
/// assert_eq!(resized.height(), 4);
/// ```
pub fn resize<T: Scalar>(
    img: &Image<T>,
    new_width: usize,
    new_height: usize,
    method: ResizeMethod,
) -> Result<Image<T>> {
    if new_width == 0 || new_height == 0 {
        return Err(ImageError::InvalidDimensions {
            width: new_width,
            height: new_height,
        });
    }

    match method {
        ResizeMethod::Nearest => resize_nearest(img, new_width, new_height),
        ResizeMethod::Bilinear => Err(ImageError::InvalidParameter {
            name: "method",
            reason: "bilinear requires Float types; use resize_bilinear",
        }),
    }
}

/// Resize using bilinear interpolation (float images only).
///
/// # Examples
///
/// ```
/// # use scivex_image::prelude::*;
/// # use scivex_image::transform;
/// let img = Image::from_raw(vec![0.0f32; 12], 2, 2, PixelFormat::Rgb).unwrap();
/// let big = transform::resize_bilinear(&img, 4, 4).unwrap();
/// assert_eq!(big.dimensions(), (4, 4));
/// ```
pub fn resize_bilinear<T: Float>(
    img: &Image<T>,
    new_width: usize,
    new_height: usize,
) -> Result<Image<T>> {
    if new_width == 0 || new_height == 0 {
        return Err(ImageError::InvalidDimensions {
            width: new_width,
            height: new_height,
        });
    }

    let (old_w, old_h) = img.dimensions();
    let c = img.channels();
    let src = img.as_slice();
    let mut out = vec![T::zero(); new_height * new_width * c];

    let x_ratio = if new_width > 1 {
        T::from_f64((old_w - 1) as f64 / (new_width - 1) as f64)
    } else {
        T::zero()
    };
    let y_ratio = if new_height > 1 {
        T::from_f64((old_h - 1) as f64 / (new_height - 1) as f64)
    } else {
        T::zero()
    };

    for y in 0..new_height {
        for x in 0..new_width {
            let fy = T::from_f64(y as f64) * y_ratio;
            let fx = T::from_f64(x as f64) * x_ratio;
            let iy0_f = Float::floor(fy);
            let ix0_f = Float::floor(fx);
            let dy = fy - iy0_f;
            let dx = fx - ix0_f;

            // Convert to usize safely
            let iy0 = float_to_usize(iy0_f);
            let ix0 = float_to_usize(ix0_f);
            let iy1 = if iy0 + 1 < old_h { iy0 + 1 } else { iy0 };
            let ix1 = if ix0 + 1 < old_w { ix0 + 1 } else { ix0 };

            let dst_idx = (y * new_width + x) * c;
            for ch in 0..c {
                let v00 = src[(iy0 * old_w + ix0) * c + ch];
                let v01 = src[(iy0 * old_w + ix1) * c + ch];
                let v10 = src[(iy1 * old_w + ix0) * c + ch];
                let v11 = src[(iy1 * old_w + ix1) * c + ch];

                let one = T::one();
                let top = v00 * (one - dx) + v01 * dx;
                let bot = v10 * (one - dx) + v11 * dx;
                out[dst_idx + ch] = top * (one - dy) + bot * dy;
            }
        }
    }

    Image::from_raw(out, new_width, new_height, img.format())
}

fn float_to_usize<T: Float>(v: T) -> usize {
    // Binary search: find largest n such that T::from_usize(n) <= v
    // For image coordinates this is bounded by reasonable image dimensions.
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

fn resize_nearest<T: Scalar>(
    img: &Image<T>,
    new_width: usize,
    new_height: usize,
) -> Result<Image<T>> {
    let (old_w, old_h) = img.dimensions();
    let c = img.channels();
    let src = img.as_slice();
    let mut out = vec![T::zero(); new_height * new_width * c];

    // Pre-compute x-coordinate mapping to avoid per-pixel division in inner loop.
    let x_map: Vec<usize> = (0..new_width).map(|x| x * old_w / new_width).collect();

    for y in 0..new_height {
        let src_y = y * old_h / new_height;
        let src_row = src_y * old_w;
        let dst_row = y * new_width;
        for (x, &sx) in x_map.iter().enumerate() {
            let dst_idx = (dst_row + x) * c;
            let src_idx = (src_row + sx) * c;
            out[dst_idx..dst_idx + c].copy_from_slice(&src[src_idx..src_idx + c]);
        }
    }

    Image::from_raw(out, new_width, new_height, img.format())
}

/// Crop a rectangular region from the image.
///
/// # Examples
///
/// ```
/// # use scivex_image::prelude::*;
/// # use scivex_image::transform;
/// let img = Image::from_raw(vec![1u8, 2, 3, 4], 2, 2, PixelFormat::Gray).unwrap();
/// let cropped = transform::crop(&img, 0, 0, 1, 2).unwrap();
/// assert_eq!(cropped.dimensions(), (1, 2));
/// ```
pub fn crop<T: Scalar>(
    img: &Image<T>,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
) -> Result<Image<T>> {
    let (img_w, img_h) = img.dimensions();
    if x + width > img_w || y + height > img_h {
        return Err(ImageError::InvalidParameter {
            name: "crop region",
            reason: "crop region exceeds image bounds",
        });
    }
    if width == 0 || height == 0 {
        return Err(ImageError::InvalidDimensions { width, height });
    }

    let c = img.channels();
    let tensor = img.as_tensor();
    let cropped = tensor.slice(&[
        SliceRange::range(y, y + height),
        SliceRange::range(x, x + width),
        SliceRange::full(c),
    ])?;

    Image::from_tensor(cropped, img.format())
}

/// Flip an image horizontally (mirror left-right).
///
/// # Examples
///
/// ```
/// # use scivex_image::{Image, PixelFormat};
/// # use scivex_image::transform::flip_horizontal;
/// let img = Image::from_raw(vec![1u8, 2], 2, 1, PixelFormat::Gray).unwrap();
/// let flipped = flip_horizontal(&img);
/// assert_eq!(flipped.as_slice(), &[2, 1]);
/// ```
pub fn flip_horizontal<T: Scalar>(img: &Image<T>) -> Image<T> {
    let (w, h) = img.dimensions();
    let c = img.channels();
    let src = img.as_slice();
    let mut out = vec![T::zero(); src.len()];

    for row in 0..h {
        for col in 0..w {
            let src_idx = (row * w + col) * c;
            let dst_idx = (row * w + (w - 1 - col)) * c;
            out[dst_idx..dst_idx + c].copy_from_slice(&src[src_idx..src_idx + c]);
        }
    }

    Image::from_raw(out, w, h, img.format()).expect("dimensions unchanged")
}

/// Flip an image vertically (mirror top-bottom).
///
/// # Examples
///
/// ```
/// # use scivex_image::prelude::*;
/// # use scivex_image::transform;
/// let img = Image::from_raw(vec![1u8, 2], 1, 2, PixelFormat::Gray).unwrap();
/// let flipped = transform::flip_vertical(&img);
/// assert_eq!(flipped.as_slice(), &[2, 1]);
/// ```
pub fn flip_vertical<T: Scalar>(img: &Image<T>) -> Image<T> {
    let (w, h) = img.dimensions();
    let c = img.channels();
    let src = img.as_slice();
    let mut out = vec![T::zero(); src.len()];

    for row in 0..h {
        let src_start = row * w * c;
        let dst_start = (h - 1 - row) * w * c;
        let row_len = w * c;
        out[dst_start..dst_start + row_len].copy_from_slice(&src[src_start..src_start + row_len]);
    }

    Image::from_raw(out, w, h, img.format()).expect("dimensions unchanged")
}

/// Rotate 90 degrees clockwise.
///
/// # Examples
///
/// ```
/// # use scivex_image::{Image, PixelFormat};
/// # use scivex_image::transform::rotate90;
/// let img = Image::from_raw(vec![1u8, 2, 3, 4], 2, 2, PixelFormat::Gray).unwrap();
/// let rotated = rotate90(&img);
/// assert_eq!(rotated.width(), 2);
/// assert_eq!(rotated.height(), 2);
/// ```
pub fn rotate90<T: Scalar>(img: &Image<T>) -> Image<T> {
    let (w, h) = img.dimensions();
    let c = img.channels();
    let src = img.as_slice();
    // New dimensions: width=h, height=w
    let mut out = vec![T::zero(); src.len()];

    for row in 0..h {
        for col in 0..w {
            let src_idx = (row * w + col) * c;
            // (row, col) -> (col, h - 1 - row) in new image
            let dst_idx = (col * h + (h - 1 - row)) * c;
            out[dst_idx..dst_idx + c].copy_from_slice(&src[src_idx..src_idx + c]);
        }
    }

    Image::from_raw(out, h, w, img.format()).expect("dimensions valid")
}

/// Rotate 180 degrees.
///
/// # Examples
///
/// ```
/// # use scivex_image::prelude::*;
/// # use scivex_image::transform;
/// let img = Image::from_raw(vec![1u8, 2, 3, 4], 2, 2, PixelFormat::Gray).unwrap();
/// let r = transform::rotate180(&img);
/// assert_eq!(r.as_slice(), &[4, 3, 2, 1]);
/// ```
pub fn rotate180<T: Scalar>(img: &Image<T>) -> Image<T> {
    let (w, h) = img.dimensions();
    let c = img.channels();
    let src = img.as_slice();
    let mut out = vec![T::zero(); src.len()];
    let total_pixels = w * h;

    for i in 0..total_pixels {
        let src_idx = i * c;
        let dst_idx = (total_pixels - 1 - i) * c;
        out[dst_idx..dst_idx + c].copy_from_slice(&src[src_idx..src_idx + c]);
    }

    Image::from_raw(out, w, h, img.format()).expect("dimensions unchanged")
}

/// Rotate 270 degrees clockwise (= 90 degrees counter-clockwise).
///
/// # Examples
///
/// ```
/// # use scivex_image::prelude::*;
/// # use scivex_image::transform;
/// let img = Image::from_raw(vec![1u8, 2, 3, 4], 2, 2, PixelFormat::Gray).unwrap();
/// let r = transform::rotate270(&img);
/// assert_eq!(r.dimensions(), (2, 2));
/// ```
pub fn rotate270<T: Scalar>(img: &Image<T>) -> Image<T> {
    let (w, h) = img.dimensions();
    let c = img.channels();
    let src = img.as_slice();
    let mut out = vec![T::zero(); src.len()];

    for row in 0..h {
        for col in 0..w {
            let src_idx = (row * w + col) * c;
            // (row, col) -> (w - 1 - col, row) in new image
            let dst_idx = ((w - 1 - col) * h + row) * c;
            out[dst_idx..dst_idx + c].copy_from_slice(&src[src_idx..src_idx + c]);
        }
    }

    Image::from_raw(out, h, w, img.format()).expect("dimensions valid")
}

/// Pad an image with a constant value on all sides.
///
/// # Examples
///
/// ```
/// # use scivex_image::prelude::*;
/// # use scivex_image::transform;
/// let img = Image::from_raw(vec![100u8], 1, 1, PixelFormat::Gray).unwrap();
/// let padded = transform::pad(&img, 1, 1, 1, 1, 0);
/// assert_eq!(padded.dimensions(), (3, 3));
/// assert_eq!(padded.get_pixel(1, 1).unwrap(), vec![100]);
/// ```
pub fn pad<T: Scalar>(
    img: &Image<T>,
    top: usize,
    bottom: usize,
    left: usize,
    right: usize,
    value: T,
) -> Image<T> {
    let (w, h) = img.dimensions();
    let c = img.channels();
    let new_w = w + left + right;
    let new_h = h + top + bottom;
    let src = img.as_slice();
    let mut out = vec![value; new_w * new_h * c];

    for row in 0..h {
        for col in 0..w {
            let src_idx = (row * w + col) * c;
            let dst_idx = ((row + top) * new_w + (col + left)) * c;
            out[dst_idx..dst_idx + c].copy_from_slice(&src[src_idx..src_idx + c]);
        }
    }

    Image::from_raw(out, new_w, new_h, img.format()).expect("dimensions valid")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PixelFormat;

    fn make_2x2_rgb() -> Image<u8> {
        // Pixel (0,0)=R, (0,1)=G, (1,0)=B, (1,1)=White
        let data = vec![
            255, 0, 0, 0, 255, 0, // row 0
            0, 0, 255, 255, 255, 255, // row 1
        ];
        Image::from_raw(data, 2, 2, PixelFormat::Rgb).unwrap()
    }

    #[test]
    fn test_resize_same_size() {
        let img = make_2x2_rgb();
        let resized = resize(&img, 2, 2, ResizeMethod::Nearest).unwrap();
        assert_eq!(resized.as_slice(), img.as_slice());
    }

    #[test]
    fn test_resize_nearest_upscale() {
        let img = make_2x2_rgb();
        let resized = resize(&img, 4, 4, ResizeMethod::Nearest).unwrap();
        assert_eq!(resized.width(), 4);
        assert_eq!(resized.height(), 4);
        // Top-left 2x2 block should all be red
        assert_eq!(resized.get_pixel(0, 0).unwrap(), vec![255, 0, 0]);
        assert_eq!(resized.get_pixel(0, 1).unwrap(), vec![255, 0, 0]);
        assert_eq!(resized.get_pixel(1, 0).unwrap(), vec![255, 0, 0]);
    }

    #[test]
    fn test_crop() {
        let img = make_2x2_rgb();
        let cropped = crop(&img, 1, 0, 1, 2).unwrap();
        assert_eq!(cropped.width(), 1);
        assert_eq!(cropped.height(), 2);
        assert_eq!(cropped.get_pixel(0, 0).unwrap(), vec![0, 255, 0]);
        assert_eq!(cropped.get_pixel(1, 0).unwrap(), vec![255, 255, 255]);
    }

    #[test]
    fn test_crop_out_of_bounds() {
        let img = make_2x2_rgb();
        assert!(crop(&img, 1, 1, 2, 2).is_err());
    }

    #[test]
    fn test_flip_horizontal_twice_identity() {
        let img = make_2x2_rgb();
        let flipped = flip_horizontal(&flip_horizontal(&img));
        assert_eq!(flipped.as_slice(), img.as_slice());
    }

    #[test]
    fn test_flip_vertical_twice_identity() {
        let img = make_2x2_rgb();
        let flipped = flip_vertical(&flip_vertical(&img));
        assert_eq!(flipped.as_slice(), img.as_slice());
    }

    #[test]
    fn test_rotate90_four_times_identity() {
        let img = make_2x2_rgb();
        let r = rotate90(&rotate90(&rotate90(&rotate90(&img))));
        assert_eq!(r.as_slice(), img.as_slice());
    }

    #[test]
    fn test_rotate180() {
        let img = make_2x2_rgb();
        let r = rotate180(&img);
        // (0,0) was Red, now at (1,1)
        assert_eq!(r.get_pixel(1, 1).unwrap(), vec![255, 0, 0]);
        // (1,1) was White, now at (0,0)
        assert_eq!(r.get_pixel(0, 0).unwrap(), vec![255, 255, 255]);
    }

    #[test]
    fn test_pad() {
        let img = Image::from_raw(vec![100u8], 1, 1, PixelFormat::Gray).unwrap();
        let padded = pad(&img, 1, 1, 1, 1, 0);
        assert_eq!(padded.width(), 3);
        assert_eq!(padded.height(), 3);
        assert_eq!(padded.get_pixel(1, 1).unwrap(), vec![100]);
        assert_eq!(padded.get_pixel(0, 0).unwrap(), vec![0]);
    }

    #[test]
    fn test_resize_bilinear() {
        let data = vec![
            0.0f32, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
        ];
        let img = Image::from_raw(data, 2, 2, PixelFormat::Rgb).unwrap();
        // Same size should be identity
        let resized = resize_bilinear(&img, 2, 2).unwrap();
        let src = img.as_slice();
        let dst = resized.as_slice();
        for (a, b) in src.iter().zip(dst.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn test_resize_to_1x1() {
        let img = make_2x2_rgb();
        let resized = resize(&img, 1, 1, ResizeMethod::Nearest).unwrap();
        assert_eq!(resized.width(), 1);
        assert_eq!(resized.height(), 1);
        assert_eq!(resized.channels(), 3);
    }

    #[test]
    fn test_resize_zero_width() {
        let img = make_2x2_rgb();
        assert!(resize(&img, 0, 5, ResizeMethod::Nearest).is_err());
    }

    #[test]
    fn test_resize_zero_height() {
        let img = make_2x2_rgb();
        assert!(resize(&img, 5, 0, ResizeMethod::Nearest).is_err());
    }

    #[test]
    fn test_resize_bilinear_to_1x1() {
        let data = vec![
            0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.0, 0.5,
        ];
        let img = Image::from_raw(data, 2, 2, PixelFormat::Rgb).unwrap();
        let resized = resize_bilinear(&img, 1, 1).unwrap();
        assert_eq!(resized.width(), 1);
        assert_eq!(resized.height(), 1);
    }

    #[test]
    fn test_resize_bilinear_zero_dims() {
        let data = vec![0.0f32; 12];
        let img = Image::from_raw(data, 2, 2, PixelFormat::Rgb).unwrap();
        assert!(resize_bilinear(&img, 0, 5).is_err());
        assert!(resize_bilinear(&img, 5, 0).is_err());
    }

    #[test]
    fn test_crop_full_image() {
        let img = make_2x2_rgb();
        let cropped = crop(&img, 0, 0, 2, 2).unwrap();
        assert_eq!(cropped.as_slice(), img.as_slice());
    }

    #[test]
    fn test_crop_zero_size() {
        let img = make_2x2_rgb();
        assert!(crop(&img, 0, 0, 0, 1).is_err());
        assert!(crop(&img, 0, 0, 1, 0).is_err());
    }

    #[test]
    fn test_flip_horizontal_1x1() {
        let img = Image::from_raw(vec![42u8], 1, 1, PixelFormat::Gray).unwrap();
        let flipped = flip_horizontal(&img);
        assert_eq!(flipped.as_slice(), &[42]);
    }

    #[test]
    fn test_flip_vertical_1x1() {
        let img = Image::from_raw(vec![42u8], 1, 1, PixelFormat::Gray).unwrap();
        let flipped = flip_vertical(&img);
        assert_eq!(flipped.as_slice(), &[42]);
    }

    #[test]
    fn test_rotate90_1x1() {
        let img = Image::from_raw(vec![7u8, 8, 9], 1, 1, PixelFormat::Rgb).unwrap();
        let rotated = rotate90(&img);
        assert_eq!(rotated.width(), 1);
        assert_eq!(rotated.height(), 1);
        assert_eq!(rotated.as_slice(), &[7, 8, 9]);
    }

    #[test]
    fn test_rotate180_twice_identity() {
        let img = make_2x2_rgb();
        let r = rotate180(&rotate180(&img));
        assert_eq!(r.as_slice(), img.as_slice());
    }

    #[test]
    fn test_rotate270_is_three_rotate90() {
        let img = make_2x2_rgb();
        let r270 = rotate270(&img);
        let r90_3 = rotate90(&rotate90(&rotate90(&img)));
        assert_eq!(r270.as_slice(), r90_3.as_slice());
    }

    #[test]
    fn test_pad_zero_padding() {
        let img = Image::from_raw(vec![50u8], 1, 1, PixelFormat::Gray).unwrap();
        let padded = pad(&img, 0, 0, 0, 0, 0);
        assert_eq!(padded.width(), 1);
        assert_eq!(padded.height(), 1);
        assert_eq!(padded.as_slice(), &[50]);
    }
}
