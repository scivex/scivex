//! JPEG image read/write support.
//!
//! Requires the `jpeg` feature flag. Uses `jpeg-decoder` for decoding and
//! `jpeg-encoder` for encoding.

use std::io::{Read, Write};

use crate::error::{ImageError, Result};
use crate::image::{Image, PixelFormat};

/// Read a JPEG image from any reader.
///
/// JPEG images are always decoded to RGB or Grayscale.
pub fn read_jpeg<R: Read>(reader: R) -> Result<Image<u8>> {
    let mut decoder = jpeg_decoder::Decoder::new(reader);
    let pixels = decoder.decode().map_err(|e| jpeg_decode_err(&e))?;
    let info = decoder
        .info()
        .ok_or_else(|| ImageError::UnsupportedFormat {
            format: "JPEG: no image info available".into(),
        })?;

    let width = info.width as usize;
    let height = info.height as usize;

    let format = match info.pixel_format {
        jpeg_decoder::PixelFormat::L8 => PixelFormat::Gray,
        jpeg_decoder::PixelFormat::RGB24 => PixelFormat::Rgb,
        other => {
            return Err(ImageError::UnsupportedFormat {
                format: format!("JPEG pixel format: {other:?}"),
            });
        }
    };

    Image::from_raw(pixels, width, height, format)
}

/// Write an image as JPEG to any writer.
///
/// Only RGB and Grayscale images are supported. Quality is fixed at 90.
pub fn write_jpeg<W: Write>(img: &Image<u8>, writer: &mut W) -> Result<()> {
    let color_type = match img.format() {
        PixelFormat::Gray => jpeg_encoder::ColorType::Luma,
        PixelFormat::Rgb => jpeg_encoder::ColorType::Rgb,
        other => {
            return Err(ImageError::UnsupportedChannels {
                channels: other.channels(),
            });
        }
    };

    let encoder = jpeg_encoder::Encoder::new(writer, 90);
    encoder
        .encode(
            img.as_slice(),
            img.width() as u16,
            img.height() as u16,
            color_type,
        )
        .map_err(|e| jpeg_encode_err(&e))?;

    Ok(())
}

fn jpeg_decode_err(e: &jpeg_decoder::Error) -> ImageError {
    ImageError::IoError(std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        e.to_string(),
    ))
}

fn jpeg_encode_err(e: &jpeg_encoder::EncodingError) -> ImageError {
    ImageError::IoError(std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        e.to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jpeg_roundtrip_rgb() {
        // Create a simple gradient image
        let w = 8;
        let h = 8;
        let mut data = Vec::with_capacity(w * h * 3);
        for y in 0..h {
            for x in 0..w {
                data.push((x * 32) as u8); // R
                data.push((y * 32) as u8); // G
                data.push(128); // B
            }
        }
        let img = Image::from_raw(data.clone(), w, h, PixelFormat::Rgb).unwrap();

        let mut buf = Vec::new();
        write_jpeg(&img, &mut buf).unwrap();

        let loaded = read_jpeg(buf.as_slice()).unwrap();
        assert_eq!(loaded.width(), w);
        assert_eq!(loaded.height(), h);
        assert_eq!(loaded.format(), PixelFormat::Rgb);

        // JPEG is lossy, so check approximate match
        let orig = &data;
        let result = loaded.as_slice();
        let max_diff: u8 = orig
            .iter()
            .zip(result.iter())
            .map(|(&a, &b)| a.abs_diff(b))
            .max()
            .unwrap_or(0);
        assert!(
            max_diff < 30,
            "JPEG roundtrip max pixel diff too large: {max_diff}"
        );
    }

    #[test]
    fn jpeg_roundtrip_gray() {
        let w = 4;
        let h = 4;
        let data: Vec<u8> = (0..w * h).map(|i| (i * 16) as u8).collect();
        let img = Image::from_raw(data.clone(), w, h, PixelFormat::Gray).unwrap();

        let mut buf = Vec::new();
        write_jpeg(&img, &mut buf).unwrap();

        let loaded = read_jpeg(buf.as_slice()).unwrap();
        assert_eq!(loaded.format(), PixelFormat::Gray);
        assert_eq!(loaded.width(), w);
        assert_eq!(loaded.height(), h);

        let max_diff: u8 = data
            .iter()
            .zip(loaded.as_slice().iter())
            .map(|(&a, &b)| a.abs_diff(b))
            .max()
            .unwrap_or(0);
        assert!(max_diff < 30, "JPEG gray max diff: {max_diff}");
    }

    #[test]
    fn jpeg_invalid_data() {
        let bad = b"not a jpeg file";
        assert!(read_jpeg(bad.as_slice()).is_err());
    }

    #[test]
    fn jpeg_rgba_unsupported() {
        let data = vec![0u8; 2 * 2 * 4];
        let img = Image::from_raw(data, 2, 2, PixelFormat::Rgba).unwrap();
        let mut buf = Vec::new();
        assert!(write_jpeg(&img, &mut buf).is_err());
    }

    #[test]
    fn jpeg_solid_color() {
        let w = 16;
        let h = 16;
        let data = vec![200u8; w * h * 3];
        let img = Image::from_raw(data, w, h, PixelFormat::Rgb).unwrap();

        let mut buf = Vec::new();
        write_jpeg(&img, &mut buf).unwrap();

        let loaded = read_jpeg(buf.as_slice()).unwrap();
        // Solid color should survive JPEG well
        let max_diff: u8 = loaded
            .as_slice()
            .iter()
            .map(|&v| v.abs_diff(200))
            .max()
            .unwrap_or(0);
        assert!(max_diff < 10, "solid color max diff: {max_diff}");
    }
}
