//! Morphological image operations: erosion, dilation, opening, closing, gradient.

use scivex_core::Scalar;

use crate::error::{ImageError, Result};
use crate::image::{Image, PixelFormat};

/// Structuring element for morphological operations.
///
/// # Examples
///
/// ```
/// # use scivex_image::morphology::StructuringElement;
/// let se = StructuringElement::Rect(3, 3);
/// let cross = StructuringElement::Cross(1);
/// let disk = StructuringElement::Disk(2);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StructuringElement {
    /// Rectangular structuring element with the given height and width.
    Rect(usize, usize),
    /// Cross-shaped structuring element with the given arm length.
    /// Total size is `(2*size + 1) x (2*size + 1)`.
    Cross(usize),
    /// Disk-shaped structuring element with the given radius.
    /// Total size is `(2*radius + 1) x (2*radius + 1)`.
    Disk(usize),
}

impl StructuringElement {
    /// Return the size `(height, width)` of the bounding box.
    fn size(&self) -> (usize, usize) {
        match *self {
            Self::Rect(h, w) => (h, w),
            Self::Cross(size) => (2 * size + 1, 2 * size + 1),
            Self::Disk(radius) => (2 * radius + 1, 2 * radius + 1),
        }
    }

    /// Check whether position `(r, c)` within the bounding box is active.
    fn active(&self, r: usize, c: usize) -> bool {
        match *self {
            Self::Rect(_, _) => true,
            Self::Cross(size) => r == size || c == size,
            Self::Disk(radius) => {
                let dr = r.abs_diff(radius);
                let dc = c.abs_diff(radius);
                dr * dr + dc * dc <= radius * radius
            }
        }
    }
}

/// Erode a grayscale image with the given structuring element.
///
/// For each output pixel, the minimum of all overlapping source pixels
/// covered by the structuring element is taken.
///
/// # Examples
///
/// ```
/// # use scivex_image::{Image, PixelFormat};
/// # use scivex_image::morphology::{erode, StructuringElement};
/// let mut img = Image::<u8>::new(5, 5, PixelFormat::Gray).unwrap();
/// img.set_pixel(2, 2, &[255]).unwrap();
/// let se = StructuringElement::Rect(3, 3);
/// let eroded = erode(&img, &se).unwrap();
/// assert_eq!(eroded.get_pixel(2, 2).unwrap(), vec![0]); // single pixel removed
/// ```
#[allow(clippy::cast_possible_wrap)]
pub fn erode<T: Scalar + Ord>(img: &Image<T>, se: &StructuringElement) -> Result<Image<T>> {
    if img.format() != PixelFormat::Gray {
        return Err(ImageError::UnsupportedChannels {
            channels: img.channels(),
        });
    }

    let (se_h, se_w) = se.size();
    if se_h == 0 || se_w == 0 {
        return Err(ImageError::InvalidParameter {
            name: "structuring element",
            reason: "dimensions must be positive",
        });
    }

    let (w, h) = img.dimensions();
    let src = img.as_slice();
    let pad_y = se_h / 2;
    let pad_x = se_w / 2;

    // For grayscale, channels == 1, so index = row * w + col.
    let mut out = vec![T::zero(); h * w];

    for row in 0..h {
        for col in 0..w {
            let mut min_val: Option<T> = None;
            for ky in 0..se_h {
                for kx in 0..se_w {
                    if !se.active(ky, kx) {
                        continue;
                    }
                    let sy = row as isize + ky as isize - pad_y as isize;
                    let sx = col as isize + kx as isize - pad_x as isize;
                    if sy >= 0 && sy < h as isize && sx >= 0 && sx < w as isize {
                        let val = src[sy as usize * w + sx as usize];
                        min_val = Some(match min_val {
                            Some(cur) => core::cmp::min(cur, val),
                            None => val,
                        });
                    }
                }
            }
            out[row * w + col] = min_val.unwrap_or(T::zero());
        }
    }

    Image::from_raw(out, w, h, PixelFormat::Gray)
}

/// Dilate a grayscale image with the given structuring element.
///
/// For each output pixel, the maximum of all overlapping source pixels
/// covered by the structuring element is taken.
///
/// # Examples
///
/// ```
/// # use scivex_image::{Image, PixelFormat};
/// # use scivex_image::morphology::{dilate, StructuringElement};
/// let mut img = Image::<u8>::new(5, 5, PixelFormat::Gray).unwrap();
/// img.set_pixel(2, 2, &[255]).unwrap();
/// let se = StructuringElement::Rect(3, 3);
/// let dilated = dilate(&img, &se).unwrap();
/// assert_eq!(dilated.get_pixel(2, 2).unwrap(), vec![255]);
/// assert_eq!(dilated.get_pixel(1, 2).unwrap(), vec![255]); // grew
/// ```
#[allow(clippy::cast_possible_wrap)]
pub fn dilate<T: Scalar + Ord>(img: &Image<T>, se: &StructuringElement) -> Result<Image<T>> {
    if img.format() != PixelFormat::Gray {
        return Err(ImageError::UnsupportedChannels {
            channels: img.channels(),
        });
    }

    let (se_h, se_w) = se.size();
    if se_h == 0 || se_w == 0 {
        return Err(ImageError::InvalidParameter {
            name: "structuring element",
            reason: "dimensions must be positive",
        });
    }

    let (w, h) = img.dimensions();
    let src = img.as_slice();
    let pad_y = se_h / 2;
    let pad_x = se_w / 2;

    let mut out = vec![T::zero(); h * w];

    for row in 0..h {
        for col in 0..w {
            let mut max_val: Option<T> = None;
            for ky in 0..se_h {
                for kx in 0..se_w {
                    if !se.active(ky, kx) {
                        continue;
                    }
                    let sy = row as isize + ky as isize - pad_y as isize;
                    let sx = col as isize + kx as isize - pad_x as isize;
                    if sy >= 0 && sy < h as isize && sx >= 0 && sx < w as isize {
                        let val = src[sy as usize * w + sx as usize];
                        max_val = Some(match max_val {
                            Some(cur) => core::cmp::max(cur, val),
                            None => val,
                        });
                    }
                }
            }
            out[row * w + col] = max_val.unwrap_or(T::zero());
        }
    }

    Image::from_raw(out, w, h, PixelFormat::Gray)
}

/// Morphological opening: erosion followed by dilation.
///
/// Removes small bright features while preserving overall shape.
///
/// # Examples
///
/// ```
/// # use scivex_image::{Image, PixelFormat};
/// # use scivex_image::morphology::{opening, StructuringElement};
/// let mut img = Image::<u8>::new(5, 5, PixelFormat::Gray).unwrap();
/// img.set_pixel(2, 2, &[255]).unwrap(); // single bright pixel
/// let se = StructuringElement::Rect(3, 3);
/// let opened = opening(&img, &se).unwrap();
/// assert_eq!(opened.get_pixel(2, 2).unwrap(), vec![0]); // removed
/// ```
pub fn opening<T: Scalar + Ord>(img: &Image<T>, se: &StructuringElement) -> Result<Image<T>> {
    let eroded = erode(img, se)?;
    dilate(&eroded, se)
}

/// Morphological closing: dilation followed by erosion.
///
/// Fills small dark holes while preserving overall shape.
///
/// # Examples
///
/// ```
/// # use scivex_image::{Image, PixelFormat};
/// # use scivex_image::morphology::{closing, StructuringElement};
/// // 3x3 white block with a hole in center
/// let data = vec![255u8, 255, 255, 255, 0, 255, 255, 255, 255];
/// let img = Image::from_raw(data, 3, 3, PixelFormat::Gray).unwrap();
/// let se = StructuringElement::Rect(3, 3);
/// let closed = closing(&img, &se).unwrap();
/// assert_eq!(closed.get_pixel(1, 1).unwrap(), vec![255]); // hole filled
/// ```
pub fn closing<T: Scalar + Ord>(img: &Image<T>, se: &StructuringElement) -> Result<Image<T>> {
    let dilated = dilate(img, se)?;
    erode(&dilated, se)
}

/// Morphological gradient: dilation minus erosion.
///
/// Highlights the boundaries of objects.
///
/// # Examples
///
/// ```
/// # use scivex_image::{Image, PixelFormat};
/// # use scivex_image::morphology::{morphological_gradient, StructuringElement};
/// let data = vec![
///     0, 0, 0, 0, 0,
///     0, 255, 255, 255, 0,
///     0, 255, 255, 255, 0,
///     0, 255, 255, 255, 0,
///     0, 0, 0, 0, 0u8,
/// ];
/// let img = Image::from_raw(data, 5, 5, PixelFormat::Gray).unwrap();
/// let se = StructuringElement::Rect(3, 3);
/// let grad = morphological_gradient(&img, &se).unwrap();
/// assert_eq!(grad.get_pixel(2, 2).unwrap(), vec![0]); // interior = 0
/// ```
pub fn morphological_gradient<T: Scalar + Ord>(
    img: &Image<T>,
    se: &StructuringElement,
) -> Result<Image<T>> {
    let dilated = dilate(img, se)?;
    let eroded = erode(img, se)?;
    let d_slice = dilated.as_slice();
    let e_slice = eroded.as_slice();
    let (w, h) = img.dimensions();

    let out: Vec<T> = d_slice
        .iter()
        .zip(e_slice.iter())
        .map(|(&d, &e)| d - e)
        .collect();

    Image::from_raw(out, w, h, PixelFormat::Gray)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_5x5_gray() -> Image<u8> {
        #[rustfmt::skip]
        let data = vec![
            0,   0,   0,   0, 0,
            0, 255, 255, 255, 0,
            0, 255, 255, 255, 0,
            0, 255, 255, 255, 0,
            0,   0,   0,   0, 0,
        ];
        Image::from_raw(data, 5, 5, PixelFormat::Gray).unwrap()
    }

    #[test]
    fn test_erode_shrinks_region() {
        let img = make_5x5_gray();
        let se = StructuringElement::Rect(3, 3);
        let eroded = erode(&img, &se).unwrap();
        // Only the center pixel should survive
        assert_eq!(eroded.get_pixel(2, 2).unwrap(), vec![255]);
        // Border of the white square should be eroded
        assert_eq!(eroded.get_pixel(1, 1).unwrap(), vec![0]);
    }

    #[test]
    fn test_dilate_grows_region() {
        let mut img = Image::<u8>::new(5, 5, PixelFormat::Gray).unwrap();
        img.set_pixel(2, 2, &[255]).unwrap();
        let se = StructuringElement::Rect(3, 3);
        let dilated = dilate(&img, &se).unwrap();
        // Center should still be 255
        assert_eq!(dilated.get_pixel(2, 2).unwrap(), vec![255]);
        // Neighbors should now be 255
        assert_eq!(dilated.get_pixel(1, 2).unwrap(), vec![255]);
        assert_eq!(dilated.get_pixel(2, 1).unwrap(), vec![255]);
    }

    #[test]
    fn test_opening_removes_small_features() {
        // Single bright pixel on dark background — opening should remove it
        let mut img = Image::<u8>::new(5, 5, PixelFormat::Gray).unwrap();
        img.set_pixel(2, 2, &[255]).unwrap();
        let se = StructuringElement::Rect(3, 3);
        let opened = opening(&img, &se).unwrap();
        assert_eq!(opened.get_pixel(2, 2).unwrap(), vec![0]);
    }

    #[test]
    fn test_closing_fills_small_holes() {
        let img = make_5x5_gray();
        // Poke a hole in the center
        let mut holed = img.clone();
        holed.set_pixel(2, 2, &[0]).unwrap();
        let se = StructuringElement::Rect(3, 3);
        let closed = closing(&holed, &se).unwrap();
        // The hole should be filled
        assert_eq!(closed.get_pixel(2, 2).unwrap(), vec![255]);
    }

    #[test]
    fn test_morphological_gradient() {
        let img = make_5x5_gray();
        let se = StructuringElement::Rect(3, 3);
        let grad = morphological_gradient(&img, &se).unwrap();
        // Interior pixel (2,2): dilate=255, erode=255, gradient=0
        assert_eq!(grad.get_pixel(2, 2).unwrap(), vec![0]);
        // Edge pixel (1,1): dilate=255, erode=0, gradient=255
        assert_eq!(grad.get_pixel(1, 1).unwrap(), vec![255]);
    }

    #[test]
    fn test_structuring_element_cross() {
        let se = StructuringElement::Cross(1);
        assert_eq!(se.size(), (3, 3));
        // Center and arms are active
        assert!(se.active(1, 1));
        assert!(se.active(0, 1));
        assert!(se.active(1, 0));
        // Corners are not active
        assert!(!se.active(0, 0));
        assert!(!se.active(2, 2));
    }

    #[test]
    fn test_structuring_element_disk() {
        let se = StructuringElement::Disk(1);
        assert_eq!(se.size(), (3, 3));
        // Center and cardinal neighbors are active
        assert!(se.active(1, 1));
        assert!(se.active(0, 1));
        assert!(se.active(1, 0));
        // Corners: distance^2 = 2 > 1 = radius^2, so not active
        assert!(!se.active(0, 0));
    }

    #[test]
    fn test_erode_rejects_rgb() {
        let img = Image::<u8>::new(3, 3, PixelFormat::Rgb).unwrap();
        let se = StructuringElement::Rect(3, 3);
        assert!(erode(&img, &se).is_err());
    }
}
