use scivex_core::{Scalar, Tensor};

use crate::error::{ImageError, Result};
use crate::image::Image;

/// Apply 2D convolution to each channel of an `f32` image.
///
/// The kernel must be a 2D tensor with odd dimensions.
/// The output has the same size as the input (zero-padded borders).
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

    for row in 0..h {
        for col in 0..w {
            for ch in 0..c {
                let mut sum = 0.0f32;
                for ky in 0..kh {
                    for kx in 0..kw {
                        let sy = row as isize + ky as isize - pad_y as isize;
                        let sx = col as isize + kx as isize - pad_x as isize;

                        if sy >= 0 && sy < h as isize && sx >= 0 && sx < w as isize {
                            let src_idx = (sy as usize * w + sx as usize) * c + ch;
                            let k_idx = ky * kw + kx;
                            sum += src[src_idx] * k[k_idx];
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
pub fn gaussian_blur(img: &Image<f32>, sigma: f32) -> Result<Image<f32>> {
    if sigma <= 0.0 {
        return Err(ImageError::InvalidParameter {
            name: "sigma",
            reason: "must be positive",
        });
    }

    let radius = (3.0 * sigma).ceil() as usize;
    let size = 2 * radius + 1;
    let mut kernel_data = vec![0.0f32; size * size];
    let mut sum = 0.0f32;

    for y in 0..size {
        for x in 0..size {
            let dy = y as f32 - radius as f32;
            let dx = x as f32 - radius as f32;
            let val = (-(dy * dy + dx * dx) / (2.0 * sigma * sigma)).exp();
            kernel_data[y * size + x] = val;
            sum += val;
        }
    }

    // Normalize
    for v in &mut kernel_data {
        *v /= sum;
    }

    let kernel = Tensor::from_vec(kernel_data, vec![size, size])?;
    convolve2d(img, &kernel)
}

/// Apply a box (uniform) blur with the given radius.
pub fn box_blur(img: &Image<f32>, radius: usize) -> Result<Image<f32>> {
    if radius == 0 {
        return Err(ImageError::InvalidParameter {
            name: "radius",
            reason: "must be positive",
        });
    }

    let size = 2 * radius + 1;
    let val = 1.0 / (size * size) as f32;
    let kernel_data = vec![val; size * size];
    let kernel = Tensor::from_vec(kernel_data, vec![size, size])?;
    convolve2d(img, &kernel)
}

/// Apply a sharpening filter (Laplacian-based).
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

/// Sobel edge detection in the X direction.
pub fn sobel_x(img: &Image<f32>) -> Result<Image<f32>> {
    #[rustfmt::skip]
    let kernel_data = vec![
        -1.0, 0.0, 1.0,
        -2.0, 0.0, 2.0,
        -1.0, 0.0, 1.0,
    ];
    let kernel = Tensor::from_vec(kernel_data, vec![3, 3])?;
    convolve2d(img, &kernel)
}

/// Sobel edge detection in the Y direction.
pub fn sobel_y(img: &Image<f32>) -> Result<Image<f32>> {
    #[rustfmt::skip]
    let kernel_data = vec![
        -1.0, -2.0, -1.0,
         0.0,  0.0,  0.0,
         1.0,  2.0,  1.0,
    ];
    let kernel = Tensor::from_vec(kernel_data, vec![3, 3])?;
    convolve2d(img, &kernel)
}

/// Sobel edge detection: magnitude of X and Y gradients.
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
}
