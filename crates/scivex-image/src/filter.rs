use scivex_core::{Scalar, Tensor};

use crate::error::{ImageError, Result};
use crate::image::Image;

/// Apply 2D convolution to each channel of an `f32` image.
///
/// The kernel must be a 2D tensor with odd dimensions.
/// The output has the same size as the input (zero-padded borders).
///
/// # Examples
///
/// ```
/// # use scivex_image::prelude::*;
/// # use scivex_image::filter;
/// # use scivex_core::Tensor;
/// let img = Image::from_raw(vec![1.0f32; 9], 3, 3, PixelFormat::Gray).unwrap();
/// let kernel = Tensor::from_vec(vec![0.0,0.0,0.0, 0.0,1.0,0.0, 0.0,0.0,0.0], vec![3,3]).unwrap();
/// let out = filter::convolve2d(&img, &kernel).unwrap();
/// assert!((out.get_pixel(1, 1).unwrap()[0] - 1.0).abs() < 1e-5);
/// ```
#[allow(clippy::cast_possible_wrap)]
pub fn convolve2d(img: &Image<f32>, kernel: &Tensor<f32>) -> Result<Image<f32>> {
    let kshape = kernel.shape();
    if kshape.len() != 2 {
        return Err(ImageError::InvalidKernel {
            reason: "kernel must be 2D",
        });
    }
    let kh = kshape[0];
    let kw = kshape[1];
    #[allow(clippy::manual_is_multiple_of)]
    if kh % 2 == 0 || kw % 2 == 0 {
        return Err(ImageError::InvalidKernel {
            reason: "kernel dimensions must be odd",
        });
    }

    let (w, h) = img.dimensions();
    let c = img.channels();
    let src = img.as_slice();
    let k = kernel.as_slice();
    let pad_y = kh / 2;
    let pad_x = kw / 2;

    let mut out = vec![0.0f32; h * w * c];

    // Split into interior (no bounds checks) and border regions.
    // Interior: rows [pad_y..h-pad_y), cols [pad_x..w-pad_x).
    let row_start = pad_y;
    let row_end = h.saturating_sub(pad_y);
    let col_start = pad_x;
    let col_end = w.saturating_sub(pad_x);

    // Interior: all kernel taps are in-bounds — use unsafe indexing.
    for row in row_start..row_end {
        for col in col_start..col_end {
            for ch in 0..c {
                let mut sum = 0.0f32;
                // SAFETY: row-pad_y..row+pad_y+1 is within [0,h),
                //         col-pad_x..col+pad_x+1 is within [0,w).
                unsafe {
                    for ky in 0..kh {
                        let sy = row + ky - pad_y;
                        let row_base = sy * w;
                        let k_row = ky * kw;
                        for kx in 0..kw {
                            let sx = col + kx - pad_x;
                            sum += *src.get_unchecked((row_base + sx) * c + ch)
                                * *k.get_unchecked(k_row + kx);
                        }
                    }
                    *out.get_unchecked_mut((row * w + col) * c + ch) = sum;
                }
            }
        }
    }

    // Border rows (top and bottom) — need bounds checks.
    let border_rows = (0..row_start).chain(row_end..h);
    for row in border_rows {
        for col in 0..w {
            for ch in 0..c {
                let mut sum = 0.0f32;
                for ky in 0..kh {
                    let sy = row as isize + ky as isize - pad_y as isize;
                    if sy < 0 || sy >= h as isize {
                        continue;
                    }
                    for kx in 0..kw {
                        let sx = col as isize + kx as isize - pad_x as isize;
                        if sx >= 0 && sx < w as isize {
                            sum += src[(sy as usize * w + sx as usize) * c + ch] * k[ky * kw + kx];
                        }
                    }
                }
                out[(row * w + col) * c + ch] = sum;
            }
        }
    }

    // Interior rows, border columns (left and right edges).
    for row in row_start..row_end {
        let border_cols = (0..col_start).chain(col_end..w);
        for col in border_cols {
            for ch in 0..c {
                let mut sum = 0.0f32;
                for ky in 0..kh {
                    let sy = row + ky - pad_y;
                    for kx in 0..kw {
                        let sx = col as isize + kx as isize - pad_x as isize;
                        if sx >= 0 && sx < w as isize {
                            sum += src[(sy * w + sx as usize) * c + ch] * k[ky * kw + kx];
                        }
                    }
                }
                out[(row * w + col) * c + ch] = sum;
            }
        }
    }

    Image::from_raw(out, w, h, img.format())
}

/// Apply Gaussian blur with the given sigma.
///
/// # Examples
///
/// ```
/// # use scivex_image::prelude::*;
/// # use scivex_image::filter;
/// let img = Image::from_raw(vec![0.5f32; 25], 5, 5, PixelFormat::Gray).unwrap();
/// let blurred = filter::gaussian_blur(&img, 1.0).unwrap();
/// assert_eq!(blurred.dimensions(), (5, 5));
/// ```
#[allow(
    clippy::needless_range_loop,
    clippy::manual_range_contains,
    clippy::too_many_lines
)]
pub fn gaussian_blur(img: &Image<f32>, sigma: f32) -> Result<Image<f32>> {
    if sigma <= 0.0 {
        return Err(ImageError::InvalidParameter {
            name: "sigma",
            reason: "must be positive",
        });
    }

    let radius = (3.0 * sigma).ceil() as usize;
    let size = 2 * radius + 1;

    // Build 1-D Gaussian kernel.
    let mut k1d = vec![0.0f32; size];
    let mut sum = 0.0f32;
    for i in 0..size {
        let d = i as f32 - radius as f32;
        let val = (-d * d / (2.0 * sigma * sigma)).exp();
        k1d[i] = val;
        sum += val;
    }
    for v in &mut k1d {
        *v /= sum;
    }

    let (w, h) = img.dimensions();
    let c = img.channels();
    let src = img.as_slice();

    // Horizontal pass: convolve each row with the 1-D kernel.
    // Split into: left edge, branchless middle, right edge.
    let mut tmp = vec![0.0f32; h * w * c];
    for row in 0..h {
        let row_off = row * w;
        // Left edge: col < radius
        for col in 0..radius.min(w) {
            for ch in 0..c {
                let mut acc = 0.0f32;
                for ki in 0..size {
                    let sx = col + ki;
                    if sx >= radius && sx < w + radius {
                        acc += src[(row_off + sx - radius) * c + ch] * k1d[ki];
                    }
                }
                tmp[(row_off + col) * c + ch] = acc;
            }
        }
        // Middle: no boundary checks needed.
        let mid_start = radius;
        let mid_end = w.saturating_sub(radius);
        for col in mid_start..mid_end {
            for ch in 0..c {
                let mut acc = 0.0f32;
                let base = row_off + col - radius;
                for ki in 0..size {
                    acc += unsafe { *src.get_unchecked((base + ki) * c + ch) } * k1d[ki];
                }
                tmp[(row_off + col) * c + ch] = acc;
            }
        }
        // Right edge: col >= w - radius
        for col in mid_end.max(radius)..w {
            for ch in 0..c {
                let mut acc = 0.0f32;
                for ki in 0..size {
                    let sx = col + ki;
                    if sx >= radius && sx < w + radius {
                        acc += src[(row_off + sx - radius) * c + ch] * k1d[ki];
                    }
                }
                tmp[(row_off + col) * c + ch] = acc;
            }
        }
    }

    // Vertical pass: convolve each column with the 1-D kernel.
    // Same split: top edge, branchless middle, bottom edge.
    let mut out = vec![0.0f32; h * w * c];
    // Top edge: row < radius
    for row in 0..radius.min(h) {
        for col in 0..w {
            for ch in 0..c {
                let mut acc = 0.0f32;
                for ki in 0..size {
                    let sy = row + ki;
                    if sy >= radius && sy < h + radius {
                        acc += tmp[((sy - radius) * w + col) * c + ch] * k1d[ki];
                    }
                }
                out[(row * w + col) * c + ch] = acc;
            }
        }
    }
    // Middle: no boundary checks needed.
    let mid_start = radius;
    let mid_end = h.saturating_sub(radius);
    for row in mid_start..mid_end {
        for col in 0..w {
            for ch in 0..c {
                let mut acc = 0.0f32;
                let base_row = row - radius;
                for ki in 0..size {
                    acc += unsafe { *tmp.get_unchecked(((base_row + ki) * w + col) * c + ch) }
                        * k1d[ki];
                }
                out[(row * w + col) * c + ch] = acc;
            }
        }
    }
    // Bottom edge: row >= h - radius
    for row in mid_end.max(radius)..h {
        for col in 0..w {
            for ch in 0..c {
                let mut acc = 0.0f32;
                for ki in 0..size {
                    let sy = row + ki;
                    if sy >= radius && sy < h + radius {
                        acc += tmp[((sy - radius) * w + col) * c + ch] * k1d[ki];
                    }
                }
                out[(row * w + col) * c + ch] = acc;
            }
        }
    }

    Image::from_raw(out, w, h, img.format())
}

/// Apply a box (uniform) blur with the given radius.
///
/// # Examples
///
/// ```
/// # use scivex_image::prelude::*;
/// # use scivex_image::filter;
/// let img = Image::from_raw(vec![1.0f32; 9], 3, 3, PixelFormat::Gray).unwrap();
/// let blurred = filter::box_blur(&img, 1).unwrap();
/// assert!((blurred.get_pixel(1, 1).unwrap()[0] - 1.0).abs() < 1e-5);
/// ```
pub fn box_blur(img: &Image<f32>, radius: usize) -> Result<Image<f32>> {
    if radius == 0 {
        return Err(ImageError::InvalidParameter {
            name: "radius",
            reason: "must be positive",
        });
    }

    let (w, h) = img.dimensions();
    let c = img.channels();
    let src = img.as_slice();
    let size = 2 * radius + 1;
    let inv = 1.0 / size as f32;

    // Horizontal pass: running sum per row.
    let mut tmp = vec![0.0f32; h * w * c];
    for row in 0..h {
        for ch in 0..c {
            // Build initial window sum for col=0.
            let mut sum = 0.0f32;
            for ki in 0..=radius.min(w - 1) {
                sum += src[(row * w + ki) * c + ch];
            }
            tmp[row * w * c + ch] = sum * inv;

            for col in 1..w {
                // Add the new right pixel entering the window.
                let add_col = col + radius;
                if add_col < w {
                    sum += src[(row * w + add_col) * c + ch];
                }
                // Remove the old left pixel leaving the window.
                if col > radius {
                    sum -= src[(row * w + col - radius - 1) * c + ch];
                }
                tmp[(row * w + col) * c + ch] = sum * inv;
            }
        }
    }

    // Vertical pass: running sum per column.
    let mut out = vec![0.0f32; h * w * c];
    for col in 0..w {
        for ch in 0..c {
            let mut sum = 0.0f32;
            for ki in 0..=radius.min(h - 1) {
                sum += tmp[(ki * w + col) * c + ch];
            }
            out[col * c + ch] = sum * inv;

            for row in 1..h {
                let add_row = row + radius;
                if add_row < h {
                    sum += tmp[(add_row * w + col) * c + ch];
                }
                if row > radius {
                    sum -= tmp[((row - radius - 1) * w + col) * c + ch];
                }
                out[(row * w + col) * c + ch] = sum * inv;
            }
        }
    }

    Image::from_raw(out, w, h, img.format())
}

/// Apply a sharpening filter (Laplacian-based).
///
/// # Examples
///
/// ```
/// # use scivex_image::prelude::*;
/// # use scivex_image::filter;
/// let img = Image::from_raw(vec![0.5f32; 25], 5, 5, PixelFormat::Gray).unwrap();
/// let sharp = filter::sharpen(&img).unwrap();
/// assert_eq!(sharp.dimensions(), (5, 5));
/// ```
pub fn sharpen(img: &Image<f32>) -> Result<Image<f32>> {
    #[rustfmt::skip]
    let kernel_data = vec![
         0.0, -1.0,  0.0,
        -1.0,  5.0, -1.0,
         0.0, -1.0,  0.0,
    ];
    let kernel = Tensor::from_vec(kernel_data, vec![3, 3])?;
    convolve2d(img, &kernel)
}

/// Apply separable convolution: first convolve rows with `h_kernel`, then columns with `v_kernel`.
///
/// Both kernels must have odd length.
#[allow(clippy::needless_range_loop, clippy::cast_possible_wrap)]
fn separable_convolve(img: &Image<f32>, v_kernel: &[f32], h_kernel: &[f32]) -> Result<Image<f32>> {
    let (w, h) = img.dimensions();
    let c = img.channels();
    let src = img.as_slice();
    let h_rad = h_kernel.len() / 2;
    let v_rad = v_kernel.len() / 2;

    // Horizontal pass.
    let mut tmp = vec![0.0f32; h * w * c];
    for row in 0..h {
        let row_off = row * w;
        // Branchless middle.
        let mid_start = h_rad;
        let mid_end = w.saturating_sub(h_rad);
        for col in mid_start..mid_end {
            for ch in 0..c {
                let mut acc = 0.0f32;
                let base = row_off + col - h_rad;
                for ki in 0..h_kernel.len() {
                    acc += unsafe { *src.get_unchecked((base + ki) * c + ch) } * h_kernel[ki];
                }
                tmp[(row_off + col) * c + ch] = acc;
            }
        }
        // Edges with bounds checks.
        let edge_cols = (0..mid_start).chain(mid_end..w);
        for col in edge_cols {
            for ch in 0..c {
                let mut acc = 0.0f32;
                for ki in 0..h_kernel.len() {
                    let sx = col as isize + ki as isize - h_rad as isize;
                    if sx >= 0 && sx < w as isize {
                        acc += src[(row_off + sx as usize) * c + ch] * h_kernel[ki];
                    }
                }
                tmp[(row_off + col) * c + ch] = acc;
            }
        }
    }

    // Vertical pass.
    let mut out = vec![0.0f32; h * w * c];
    let mid_start = v_rad;
    let mid_end = h.saturating_sub(v_rad);
    for row in mid_start..mid_end {
        for col in 0..w {
            for ch in 0..c {
                let mut acc = 0.0f32;
                let base_row = row - v_rad;
                for ki in 0..v_kernel.len() {
                    acc += unsafe { *tmp.get_unchecked(((base_row + ki) * w + col) * c + ch) }
                        * v_kernel[ki];
                }
                out[(row * w + col) * c + ch] = acc;
            }
        }
    }
    // Edge rows.
    let edge_rows = (0..mid_start).chain(mid_end..h);
    for row in edge_rows {
        for col in 0..w {
            for ch in 0..c {
                let mut acc = 0.0f32;
                for ki in 0..v_kernel.len() {
                    let sy = row as isize + ki as isize - v_rad as isize;
                    if sy >= 0 && sy < h as isize {
                        acc += tmp[(sy as usize * w + col) * c + ch] * v_kernel[ki];
                    }
                }
                out[(row * w + col) * c + ch] = acc;
            }
        }
    }

    Image::from_raw(out, w, h, img.format())
}

/// Sobel edge detection in the X direction.
///
/// # Examples
///
/// ```
/// # use scivex_image::prelude::*;
/// # use scivex_image::filter;
/// let img = Image::from_raw(vec![0.5f32; 9], 3, 3, PixelFormat::Gray).unwrap();
/// let gx = filter::sobel_x(&img).unwrap();
/// assert_eq!(gx.dimensions(), (3, 3));
/// ```
pub fn sobel_x(img: &Image<f32>) -> Result<Image<f32>> {
    // Separable: Sobel X = [1, 2, 1]^T * [-1, 0, 1]
    separable_convolve(img, &[1.0, 2.0, 1.0], &[-1.0, 0.0, 1.0])
}

/// Sobel edge detection in the Y direction.
///
/// # Examples
///
/// ```
/// # use scivex_image::prelude::*;
/// # use scivex_image::filter;
/// let img = Image::from_raw(vec![0.5f32; 9], 3, 3, PixelFormat::Gray).unwrap();
/// let gy = filter::sobel_y(&img).unwrap();
/// assert_eq!(gy.dimensions(), (3, 3));
/// ```
pub fn sobel_y(img: &Image<f32>) -> Result<Image<f32>> {
    // Separable: Sobel Y = [-1, 0, 1]^T * [1, 2, 1]
    separable_convolve(img, &[-1.0, 0.0, 1.0], &[1.0, 2.0, 1.0])
}

/// Sobel edge detection: magnitude of X and Y gradients.
///
/// # Examples
///
/// ```
/// # use scivex_image::prelude::*;
/// # use scivex_image::filter;
/// let img = Image::from_raw(vec![0.5f32; 9], 3, 3, PixelFormat::Gray).unwrap();
/// let edges = filter::sobel(&img).unwrap();
/// // Uniform image has zero gradients at the center
/// assert!(edges.get_pixel(1, 1).unwrap()[0].abs() < 1e-5);
/// ```
pub fn sobel(img: &Image<f32>) -> Result<Image<f32>> {
    let gx = sobel_x(img)?;
    let gy = sobel_y(img)?;
    let gx_data = gx.as_slice();
    let gy_data = gy.as_slice();
    let out: Vec<f32> = gx_data
        .iter()
        .zip(gy_data.iter())
        .map(|(&x, &y)| (x * x + y * y).sqrt())
        .collect();
    Image::from_raw(out, img.width(), img.height(), img.format())
}

/// Apply a median filter with the given radius.
///
/// # Examples
///
/// ```
/// # use scivex_image::prelude::*;
/// # use scivex_image::filter;
/// let img = Image::from_raw(vec![1u8, 5, 2, 3, 100, 4, 6, 7, 8], 3, 3, PixelFormat::Gray).unwrap();
/// let filtered = filter::median_filter(&img, 1).unwrap();
/// assert_eq!(filtered.get_pixel(1, 1).unwrap(), vec![5]); // median of 3x3 window
/// ```
#[allow(clippy::cast_possible_wrap)]
pub fn median_filter<T: Scalar + Ord>(img: &Image<T>, radius: usize) -> Result<Image<T>> {
    if radius == 0 {
        return Err(ImageError::InvalidParameter {
            name: "radius",
            reason: "must be positive",
        });
    }

    let (w, h) = img.dimensions();
    let c = img.channels();
    let src = img.as_slice();
    let mut out = vec![T::zero(); src.len()];

    for row in 0..h {
        for col in 0..w {
            for ch in 0..c {
                let mut window = Vec::new();
                let r = radius as isize;
                for dy in -r..=r {
                    for dx in -r..=r {
                        let sy = row as isize + dy;
                        let sx = col as isize + dx;
                        if sy >= 0 && sy < h as isize && sx >= 0 && sx < w as isize {
                            let idx = (sy as usize * w + sx as usize) * c + ch;
                            window.push(src[idx]);
                        }
                    }
                }
                window.sort();
                out[(row * w + col) * c + ch] = window[window.len() / 2];
            }
        }
    }

    Image::from_raw(out, w, h, img.format())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PixelFormat;

    #[test]
    fn test_identity_kernel() {
        let data = vec![0.5f32, 0.3, 0.7, 0.2, 0.8, 0.1, 0.9, 0.4, 0.6];
        let img = Image::from_raw(data.clone(), 3, 3, PixelFormat::Gray).unwrap();

        #[rustfmt::skip]
        let kernel = Tensor::from_vec(vec![
            0.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 0.0,
        ], vec![3, 3]).unwrap();

        let result = convolve2d(&img, &kernel).unwrap();
        // Interior pixel (1,1) should be unchanged
        let p = result.get_pixel(1, 1).unwrap();
        assert!((p[0] - 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_gaussian_blur_reduces_variance() {
        // Create image with a spike
        let mut data = vec![0.0f32; 25];
        data[12] = 1.0; // center pixel
        let img = Image::from_raw(data, 5, 5, PixelFormat::Gray).unwrap();

        let blurred = gaussian_blur(&img, 1.0).unwrap();
        // Center should be lower than 1.0 after blur
        let center = blurred.get_pixel(2, 2).unwrap();
        assert!(center[0] < 1.0);
        assert!(center[0] > 0.0);
    }

    #[test]
    fn test_sobel_detects_edge() {
        // Vertical edge: left half black, right half white
        let mut data = vec![0.0f32; 9];
        data[1] = 1.0; // (0,1)
        data[2] = 1.0; // (0,2)
        data[4] = 1.0; // (1,1)
        data[5] = 1.0; // (1,2)
        data[7] = 1.0; // (2,1)
        data[8] = 1.0; // (2,2)
        let img = Image::from_raw(data, 3, 3, PixelFormat::Gray).unwrap();

        let edges = sobel(&img).unwrap();
        // Center pixel should have non-zero edge response
        let center = edges.get_pixel(1, 1).unwrap();
        assert!(center[0] > 0.0);
    }

    #[test]
    fn test_box_blur() {
        let data = vec![1.0f32; 9];
        let img = Image::from_raw(data, 3, 3, PixelFormat::Gray).unwrap();
        let blurred = box_blur(&img, 1).unwrap();
        // Center pixel of uniform image should stay ~1.0
        let center = blurred.get_pixel(1, 1).unwrap();
        assert!((center[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_median_filter() {
        let data = vec![1u8, 5, 2, 3, 100, 4, 6, 7, 8];
        let img = Image::from_raw(data, 3, 3, PixelFormat::Gray).unwrap();
        let filtered = median_filter(&img, 1).unwrap();
        // Center pixel: window is [1,5,2,3,100,4,6,7,8] sorted=[1,2,3,4,5,6,7,8,100] median=5
        let center = filtered.get_pixel(1, 1).unwrap();
        assert_eq!(center[0], 5);
    }

    #[test]
    fn test_invalid_kernel() {
        let img = Image::from_raw(vec![0.0f32; 4], 2, 2, PixelFormat::Gray).unwrap();
        let kernel = Tensor::from_vec(vec![1.0f32; 4], vec![2, 2]).unwrap();
        assert!(convolve2d(&img, &kernel).is_err());
    }

    #[test]
    fn test_blur_1x1_image() {
        let img = Image::from_raw(vec![0.5f32], 1, 1, PixelFormat::Gray).unwrap();
        let blurred = gaussian_blur(&img, 1.0).unwrap();
        // Single pixel, zero-padded borders: value should be attenuated
        assert_eq!(blurred.width(), 1);
        assert_eq!(blurred.height(), 1);
        let p = blurred.get_pixel(0, 0).unwrap();
        assert!(p[0] > 0.0);
    }

    #[test]
    fn test_sobel_on_uniform_image() {
        // Uniform image should have zero gradients everywhere (interior)
        let data = vec![0.5f32; 9];
        let img = Image::from_raw(data, 3, 3, PixelFormat::Gray).unwrap();
        let edges = sobel(&img).unwrap();
        let center = edges.get_pixel(1, 1).unwrap();
        assert!(
            center[0].abs() < 1e-5,
            "uniform image edge should be ~0, got {}",
            center[0]
        );
    }

    #[test]
    fn test_sobel_x_on_uniform() {
        let data = vec![1.0f32; 9];
        let img = Image::from_raw(data, 3, 3, PixelFormat::Gray).unwrap();
        let gx = sobel_x(&img).unwrap();
        let center = gx.get_pixel(1, 1).unwrap();
        assert!(center[0].abs() < 1e-5);
    }

    #[test]
    fn test_sobel_y_on_uniform() {
        let data = vec![1.0f32; 9];
        let img = Image::from_raw(data, 3, 3, PixelFormat::Gray).unwrap();
        let gy = sobel_y(&img).unwrap();
        let center = gy.get_pixel(1, 1).unwrap();
        assert!(center[0].abs() < 1e-5);
    }

    #[test]
    fn test_gaussian_blur_invalid_sigma() {
        let img = Image::from_raw(vec![0.5f32; 9], 3, 3, PixelFormat::Gray).unwrap();
        assert!(gaussian_blur(&img, 0.0).is_err());
        assert!(gaussian_blur(&img, -1.0).is_err());
    }

    #[test]
    fn test_box_blur_invalid_radius() {
        let img = Image::from_raw(vec![0.5f32; 9], 3, 3, PixelFormat::Gray).unwrap();
        assert!(box_blur(&img, 0).is_err());
    }

    #[test]
    fn test_median_filter_invalid_radius() {
        let img = Image::from_raw(vec![1u8; 9], 3, 3, PixelFormat::Gray).unwrap();
        assert!(median_filter(&img, 0).is_err());
    }

    #[test]
    fn test_convolve2d_1d_kernel_rejected() {
        let img = Image::from_raw(vec![1.0f32; 9], 3, 3, PixelFormat::Gray).unwrap();
        let kernel = Tensor::from_vec(vec![1.0f32; 3], vec![3]).unwrap();
        assert!(convolve2d(&img, &kernel).is_err());
    }

    #[test]
    fn test_sharpen_preserves_dimensions() {
        let data = vec![0.5f32; 25 * 3];
        let img = Image::from_raw(data, 5, 5, PixelFormat::Rgb).unwrap();
        let sharpened = sharpen(&img).unwrap();
        assert_eq!(sharpened.width(), 5);
        assert_eq!(sharpened.height(), 5);
        assert_eq!(sharpened.channels(), 3);
    }

    #[test]
    fn test_median_filter_1x1() {
        let img = Image::from_raw(vec![42u8], 1, 1, PixelFormat::Gray).unwrap();
        let filtered = median_filter(&img, 1).unwrap();
        assert_eq!(filtered.get_pixel(0, 0).unwrap(), vec![42]);
    }
}
