use scivex_core::{Scalar, Tensor};

use crate::error::{ImageError, Result};

/// Pixel format describing how channels are interpreted.
///
/// # Examples
///
/// ```
/// # use scivex_image::prelude::*;
/// assert_eq!(PixelFormat::Rgb.channels(), 3);
/// assert_eq!(PixelFormat::Gray.channels(), 1);
/// ```
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    /// Single-channel grayscale.
    Gray,
    /// Grayscale with alpha.
    GrayAlpha,
    /// Three-channel red/green/blue.
    Rgb,
    /// Red/green/blue with alpha.
    Rgba,
}

impl PixelFormat {
    /// Number of channels for this format.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_image::prelude::*;
    /// assert_eq!(PixelFormat::Rgba.channels(), 4);
    /// ```
    #[inline]
    pub fn channels(self) -> usize {
        match self {
            Self::Gray => 1,
            Self::GrayAlpha => 2,
            Self::Rgb => 3,
            Self::Rgba => 4,
        }
    }
}

/// A 2-D image backed by a [`Tensor<T>`] with shape `[height, width, channels]`.
///
/// # Examples
///
/// ```
/// # use scivex_image::prelude::*;
/// let img = Image::<u8>::new(10, 10, PixelFormat::Rgb).unwrap();
/// assert_eq!(img.dimensions(), (10, 10));
/// ```
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct Image<T: Scalar> {
    data: Tensor<T>,
    format: PixelFormat,
}

impl<T: Scalar> Image<T> {
    /// Create a new zero-filled image.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_image::prelude::*;
    /// let img = Image::<u8>::new(32, 24, PixelFormat::Gray).unwrap();
    /// assert_eq!(img.width(), 32);
    /// assert_eq!(img.height(), 24);
    /// ```
    pub fn new(width: usize, height: usize, format: PixelFormat) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(ImageError::InvalidDimensions { width, height });
        }
        let c = format.channels();
        let data = Tensor::zeros(vec![height, width, c]);
        Ok(Self { data, format })
    }

    /// Create an image from raw pixel data in row-major order.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_image::prelude::*;
    /// let data = vec![255u8; 2 * 3 * 3]; // 3x2 RGB
    /// let img = Image::from_raw(data, 3, 2, PixelFormat::Rgb).unwrap();
    /// assert_eq!(img.width(), 3);
    /// ```
    pub fn from_raw(
        data: Vec<T>,
        width: usize,
        height: usize,
        format: PixelFormat,
    ) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(ImageError::InvalidDimensions { width, height });
        }
        let c = format.channels();
        let expected = height * width * c;
        if data.len() != expected {
            return Err(ImageError::DataLengthMismatch {
                expected,
                got: data.len(),
            });
        }
        let tensor = Tensor::from_vec(data, vec![height, width, c])?;
        Ok(Self {
            data: tensor,
            format,
        })
    }

    /// Wrap an existing tensor as an image.
    ///
    /// The tensor must have shape `[height, width, channels]` where channels
    /// matches the given format.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_image::prelude::*;
    /// # use scivex_core::Tensor;
    /// let t = Tensor::from_vec(vec![1u8; 12], vec![2, 2, 3]).unwrap();
    /// let img = Image::from_tensor(t, PixelFormat::Rgb).unwrap();
    /// assert_eq!(img.dimensions(), (2, 2));
    /// ```
    pub fn from_tensor(tensor: Tensor<T>, format: PixelFormat) -> Result<Self> {
        let shape = tensor.shape();
        if shape.len() != 3 {
            return Err(ImageError::InvalidDimensions {
                width: 0,
                height: 0,
            });
        }
        let c = format.channels();
        if shape[2] != c {
            return Err(ImageError::UnsupportedChannels { channels: shape[2] });
        }
        if shape[0] == 0 || shape[1] == 0 {
            return Err(ImageError::InvalidDimensions {
                width: shape[1],
                height: shape[0],
            });
        }
        Ok(Self {
            data: tensor,
            format,
        })
    }

    /// Image height in pixels.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_image::prelude::*;
    /// let img = Image::<u8>::new(10, 5, PixelFormat::Gray).unwrap();
    /// assert_eq!(img.height(), 5);
    /// ```
    #[inline]
    pub fn height(&self) -> usize {
        self.data.shape()[0]
    }

    /// Image width in pixels.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_image::prelude::*;
    /// let img = Image::<u8>::new(10, 5, PixelFormat::Gray).unwrap();
    /// assert_eq!(img.width(), 10);
    /// ```
    #[inline]
    pub fn width(&self) -> usize {
        self.data.shape()[1]
    }

    /// Number of channels.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_image::prelude::*;
    /// let img = Image::<u8>::new(4, 4, PixelFormat::Rgba).unwrap();
    /// assert_eq!(img.channels(), 4);
    /// ```
    #[inline]
    pub fn channels(&self) -> usize {
        self.data.shape()[2]
    }

    /// The pixel format.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_image::prelude::*;
    /// let img = Image::<u8>::new(4, 4, PixelFormat::Rgb).unwrap();
    /// assert_eq!(img.format(), PixelFormat::Rgb);
    /// ```
    #[inline]
    pub fn format(&self) -> PixelFormat {
        self.format
    }

    /// Returns `(width, height)`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_image::prelude::*;
    /// let img = Image::<u8>::new(8, 6, PixelFormat::Rgb).unwrap();
    /// assert_eq!(img.dimensions(), (8, 6));
    /// ```
    #[inline]
    pub fn dimensions(&self) -> (usize, usize) {
        (self.width(), self.height())
    }

    /// Get pixel values at `(row, col)`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_image::prelude::*;
    /// let img = Image::from_raw(vec![10u8, 20, 30], 1, 1, PixelFormat::Rgb).unwrap();
    /// assert_eq!(img.get_pixel(0, 0).unwrap(), vec![10, 20, 30]);
    /// ```
    pub fn get_pixel(&self, row: usize, col: usize) -> Result<Vec<T>> {
        let c = self.channels();
        let mut pixel = Vec::with_capacity(c);
        for ch in 0..c {
            pixel.push(*self.data.get(&[row, col, ch])?);
        }
        Ok(pixel)
    }

    /// Set pixel values at `(row, col)`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_image::prelude::*;
    /// let mut img = Image::<u8>::new(4, 4, PixelFormat::Rgb).unwrap();
    /// img.set_pixel(0, 0, &[255, 0, 0]).unwrap();
    /// assert_eq!(img.get_pixel(0, 0).unwrap(), vec![255, 0, 0]);
    /// ```
    pub fn set_pixel(&mut self, row: usize, col: usize, values: &[T]) -> Result<()> {
        let c = self.channels();
        if values.len() != c {
            return Err(ImageError::DataLengthMismatch {
                expected: c,
                got: values.len(),
            });
        }
        for (ch, &val) in values.iter().enumerate() {
            self.data.set(&[row, col, ch], val)?;
        }
        Ok(())
    }

    /// Borrow the underlying tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_image::prelude::*;
    /// let img = Image::<u8>::new(2, 3, PixelFormat::Rgb).unwrap();
    /// assert_eq!(img.as_tensor().shape(), &[3, 2, 3]);
    /// ```
    #[inline]
    pub fn as_tensor(&self) -> &Tensor<T> {
        &self.data
    }

    /// Consume the image and return the underlying tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_image::prelude::*;
    /// let img = Image::from_raw(vec![1u8, 2, 3], 1, 1, PixelFormat::Rgb).unwrap();
    /// let t = img.into_tensor();
    /// assert_eq!(t.as_slice(), &[1, 2, 3]);
    /// ```
    #[inline]
    pub fn into_tensor(self) -> Tensor<T> {
        self.data
    }

    /// Flat slice of all pixel data.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_image::prelude::*;
    /// let img = Image::from_raw(vec![10u8, 20, 30], 1, 1, PixelFormat::Rgb).unwrap();
    /// assert_eq!(img.as_slice(), &[10, 20, 30]);
    /// ```
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self.data.as_slice()
    }

    /// Mutable flat slice of all pixel data.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_image::prelude::*;
    /// let mut img = Image::<u8>::new(2, 2, PixelFormat::Gray).unwrap();
    /// img.as_mut_slice()[0] = 42;
    /// assert_eq!(img.get_pixel(0, 0).unwrap(), vec![42]);
    /// ```
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.data.as_mut_slice()
    }

    /// Apply a function to every element, producing a new image.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_image::prelude::*;
    /// let img = Image::from_raw(vec![10u8, 20, 30], 1, 1, PixelFormat::Rgb).unwrap();
    /// let doubled = img.map_pixels(|v| v.saturating_mul(2));
    /// assert_eq!(doubled.get_pixel(0, 0).unwrap(), vec![20, 40, 60]);
    /// ```
    pub fn map_pixels<F: Fn(T) -> T>(&self, f: F) -> Self {
        Self {
            data: self.data.map(f),
            format: self.format,
        }
    }
}

impl Image<u8> {
    /// Convert a `u8` image to `f32`, scaling `[0, 255]` to `[0.0, 1.0]`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_image::prelude::*;
    /// let img = Image::from_raw(vec![0u8, 255], 1, 1, PixelFormat::GrayAlpha).unwrap();
    /// let f = img.to_f32();
    /// assert!((f.as_slice()[1] - 1.0).abs() < 1e-6);
    /// ```
    pub fn to_f32(&self) -> Image<f32> {
        let src = self.data.as_slice();
        let dst: Vec<f32> = src.iter().map(|&v| f32::from(v) / 255.0).collect();
        let shape = self.data.shape().to_vec();
        Image {
            data: Tensor::from_vec(dst, shape).expect("shape unchanged"),
            format: self.format,
        }
    }
}

impl Image<f32> {
    /// Convert an `f32` image to `u8`, clamping `[0.0, 1.0]` and scaling to `[0, 255]`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_image::prelude::*;
    /// let img = Image::from_raw(vec![0.0f32, 0.5, 1.0], 1, 1, PixelFormat::Rgb).unwrap();
    /// let u8_img = img.to_u8();
    /// assert_eq!(u8_img.get_pixel(0, 0).unwrap(), vec![0, 128, 255]);
    /// ```
    pub fn to_u8(&self) -> Image<u8> {
        let src = self.data.as_slice();
        let dst: Vec<u8> = src
            .iter()
            .map(|&v| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8)
            .collect();
        let shape = self.data.shape().to_vec();
        Image {
            data: Tensor::from_vec(dst, shape).expect("shape unchanged"),
            format: self.format,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_image() {
        let img = Image::<u8>::new(10, 20, PixelFormat::Rgb).unwrap();
        assert_eq!(img.width(), 10);
        assert_eq!(img.height(), 20);
        assert_eq!(img.channels(), 3);
        assert_eq!(img.format(), PixelFormat::Rgb);
        assert_eq!(img.dimensions(), (10, 20));
    }

    #[test]
    fn test_new_zero_dims() {
        assert!(Image::<u8>::new(0, 10, PixelFormat::Gray).is_err());
        assert!(Image::<u8>::new(10, 0, PixelFormat::Gray).is_err());
    }

    #[test]
    fn test_from_raw() {
        let data = vec![255u8; 2 * 3 * 3];
        let img = Image::from_raw(data, 3, 2, PixelFormat::Rgb).unwrap();
        assert_eq!(img.width(), 3);
        assert_eq!(img.height(), 2);
        let pixel = img.get_pixel(0, 0).unwrap();
        assert_eq!(pixel, vec![255, 255, 255]);
    }

    #[test]
    fn test_from_raw_bad_len() {
        let data = vec![0u8; 10];
        assert!(Image::from_raw(data, 3, 2, PixelFormat::Rgb).is_err());
    }

    #[test]
    fn test_get_set_pixel() {
        let mut img = Image::<u8>::new(4, 4, PixelFormat::Rgb).unwrap();
        img.set_pixel(1, 2, &[10, 20, 30]).unwrap();
        let pixel = img.get_pixel(1, 2).unwrap();
        assert_eq!(pixel, vec![10, 20, 30]);
    }

    #[test]
    fn test_to_f32_and_back() {
        let data = vec![0, 128, 255, 0, 64, 192];
        let img = Image::from_raw(data, 2, 1, PixelFormat::Rgb).unwrap();
        let f32_img = img.to_f32();
        let roundtrip = f32_img.to_u8();
        assert_eq!(roundtrip.get_pixel(0, 0).unwrap(), vec![0, 128, 255]);
        assert_eq!(roundtrip.get_pixel(0, 1).unwrap(), vec![0, 64, 192]);
    }

    #[test]
    fn test_map_pixels() {
        let data = vec![10u8, 20, 30, 40, 50, 60];
        let img = Image::from_raw(data, 2, 1, PixelFormat::Rgb).unwrap();
        let doubled = img.map_pixels(|v| v.saturating_mul(2));
        assert_eq!(doubled.get_pixel(0, 0).unwrap(), vec![20, 40, 60]);
    }

    #[test]
    fn test_from_tensor() {
        let t = Tensor::from_vec(vec![1u8; 12], vec![2, 2, 3]).unwrap();
        let img = Image::from_tensor(t, PixelFormat::Rgb).unwrap();
        assert_eq!(img.width(), 2);
        assert_eq!(img.height(), 2);
    }

    #[test]
    fn test_from_tensor_bad_channels() {
        let t = Tensor::from_vec(vec![1u8; 8], vec![2, 2, 2]).unwrap();
        assert!(Image::from_tensor(t, PixelFormat::Rgb).is_err());
    }

    #[test]
    fn test_from_raw_zero_width() {
        let data: Vec<u8> = vec![];
        assert!(Image::from_raw(data, 0, 5, PixelFormat::Gray).is_err());
    }

    #[test]
    fn test_from_raw_zero_height() {
        let data: Vec<u8> = vec![];
        assert!(Image::from_raw(data, 5, 0, PixelFormat::Rgb).is_err());
    }

    #[test]
    fn test_set_pixel_wrong_channel_count() {
        let mut img = Image::<u8>::new(2, 2, PixelFormat::Rgb).unwrap();
        // Providing 2 values for a 3-channel image should fail.
        assert!(img.set_pixel(0, 0, &[10, 20]).is_err());
    }

    #[test]
    fn test_from_tensor_wrong_ndim() {
        // A 2D tensor should be rejected (needs 3D).
        let t = Tensor::from_vec(vec![1u8; 4], vec![2, 2]).unwrap();
        assert!(Image::from_tensor(t, PixelFormat::Gray).is_err());
    }

    #[test]
    fn test_from_tensor_zero_height() {
        // 3D tensor with height=0 should be rejected.
        let t = Tensor::from_vec(Vec::<u8>::new(), vec![0, 2, 3]).unwrap();
        assert!(Image::from_tensor(t, PixelFormat::Rgb).is_err());
    }

    #[test]
    fn test_1x1_gray_image() {
        let img = Image::<u8>::new(1, 1, PixelFormat::Gray).unwrap();
        assert_eq!(img.width(), 1);
        assert_eq!(img.height(), 1);
        assert_eq!(img.channels(), 1);
        assert_eq!(img.get_pixel(0, 0).unwrap(), vec![0]);
    }

    #[test]
    fn test_as_mut_slice() {
        let mut img = Image::<u8>::new(2, 2, PixelFormat::Gray).unwrap();
        let slice = img.as_mut_slice();
        slice[0] = 42;
        assert_eq!(img.get_pixel(0, 0).unwrap(), vec![42]);
    }

    #[test]
    fn test_into_tensor() {
        let data = vec![1u8, 2, 3, 4, 5, 6];
        let img = Image::from_raw(data.clone(), 2, 1, PixelFormat::Rgb).unwrap();
        let tensor = img.into_tensor();
        assert_eq!(tensor.shape(), &[1, 2, 3]);
        assert_eq!(tensor.as_slice(), &data);
    }

    #[test]
    fn test_pixel_format_channels() {
        assert_eq!(PixelFormat::Gray.channels(), 1);
        assert_eq!(PixelFormat::GrayAlpha.channels(), 2);
        assert_eq!(PixelFormat::Rgb.channels(), 3);
        assert_eq!(PixelFormat::Rgba.channels(), 4);
    }

    #[test]
    fn test_to_f32_clamp_roundtrip() {
        // Values at extremes should survive the roundtrip.
        let data = vec![0u8, 255];
        let img = Image::from_raw(data, 1, 1, PixelFormat::GrayAlpha).unwrap();
        let f = img.to_f32();
        let p = f.as_slice();
        assert!((p[0] - 0.0).abs() < 1e-6);
        assert!((p[1] - 1.0).abs() < 1e-6);
        let back = f.to_u8();
        assert_eq!(back.as_slice(), &[0, 255]);
    }

    #[test]
    fn test_to_u8_clamps_out_of_range() {
        // f32 values outside [0,1] should be clamped.
        let data = vec![-0.5f32, 1.5, 0.5];
        let img = Image::from_raw(data, 1, 1, PixelFormat::Rgb).unwrap();
        let u8img = img.to_u8();
        let p = u8img.get_pixel(0, 0).unwrap();
        assert_eq!(p[0], 0);
        assert_eq!(p[1], 255);
        assert_eq!(p[2], 128);
    }

    #[test]
    fn test_1x1_rgb_image() {
        let img = Image::<u8>::new(1, 1, PixelFormat::Rgb).unwrap();
        assert_eq!(img.width(), 1);
        assert_eq!(img.height(), 1);
        assert_eq!(img.channels(), 3);
        assert_eq!(img.as_slice(), &[0, 0, 0]);
    }

    #[test]
    fn test_2x2_rgba_image() {
        let img = Image::<u8>::new(2, 2, PixelFormat::Rgba).unwrap();
        assert_eq!(img.width(), 2);
        assert_eq!(img.height(), 2);
        assert_eq!(img.channels(), 4);
        assert_eq!(img.as_slice().len(), 2 * 2 * 4);
        // All zeros
        assert!(img.as_slice().iter().all(|&v| v == 0));
    }

    #[test]
    fn test_image_all_same_pixel_values() {
        let data = vec![42u8; 3 * 3 * 3]; // 3x3 RGB, all 42
        let img = Image::from_raw(data, 3, 3, PixelFormat::Rgb).unwrap();
        for row in 0..3 {
            for col in 0..3 {
                assert_eq!(img.get_pixel(row, col).unwrap(), vec![42, 42, 42]);
            }
        }
    }

    #[test]
    fn test_new_zero_width_and_zero_height() {
        assert!(Image::<u8>::new(0, 0, PixelFormat::Rgb).is_err());
    }

    #[test]
    fn test_from_raw_zero_dims_both() {
        let data: Vec<u8> = vec![];
        assert!(Image::from_raw(data, 0, 0, PixelFormat::Gray).is_err());
    }

    #[test]
    fn test_map_pixels_identity() {
        let data = vec![10u8, 20, 30];
        let img = Image::from_raw(data.clone(), 1, 1, PixelFormat::Rgb).unwrap();
        let same = img.map_pixels(|v| v);
        assert_eq!(same.as_slice(), &data);
    }
}
