//! PNG image read/write support.
//!
//! Requires the `png` feature flag. Uses the `png` crate for encoding and
//! decoding.

use std::io::{Read, Write};

use crate::error::{ImageError, Result};
use crate::image::{Image, PixelFormat};

/// Read a PNG image from any reader.
pub fn read_png<R: Read>(reader: R) -> Result<Image<u8>> {
    let decoder = png::Decoder::new(reader);
    let mut png_reader = decoder.read_info().map_err(png_err)?;

    let info = png_reader.info();
    let width = info.width as usize;
    let height = info.height as usize;

    let format = match info.color_type {
        png::ColorType::Grayscale => PixelFormat::Gray,
        png::ColorType::GrayscaleAlpha => PixelFormat::GrayAlpha,
        png::ColorType::Rgb => PixelFormat::Rgb,
        png::ColorType::Rgba => PixelFormat::Rgba,
        other @ png::ColorType::Indexed => {
            return Err(ImageError::UnsupportedFormat {
                format: format!("PNG color type: {other:?}"),
            });
        }
    };

    // Handle bit-depth: the png crate can transform to 8-bit for us.
    let bit_depth = info.bit_depth;
    if bit_depth != png::BitDepth::Eight {
        return Err(ImageError::UnsupportedFormat {
            format: format!("PNG bit depth {bit_depth:?} not supported, only 8-bit"),
        });
    }

    let channels = format.channels();
    let mut data = vec![0u8; width * height * channels];
    let output_info = png_reader.next_frame(&mut data).map_err(png_err)?;

    // Trim to actual output size (should match, but be safe)
    data.truncate(output_info.buffer_size());

    Image::from_raw(data, width, height, format)
}

/// Write an image as PNG to any writer.
pub fn write_png<W: Write>(img: &Image<u8>, writer: W) -> Result<()> {
    let width = img.width() as u32;
    let height = img.height() as u32;

    let color_type = match img.format() {
        PixelFormat::Gray => png::ColorType::Grayscale,
        PixelFormat::GrayAlpha => png::ColorType::GrayscaleAlpha,
        PixelFormat::Rgb => png::ColorType::Rgb,
        PixelFormat::Rgba => png::ColorType::Rgba,
    };

    let mut encoder = png::Encoder::new(writer, width, height);
    encoder.set_color(color_type);
    encoder.set_depth(png::BitDepth::Eight);

    let mut png_writer = encoder.write_header().map_err(png_encoding_err)?;
    png_writer
        .write_image_data(img.as_slice())
        .map_err(png_encoding_err)?;

    Ok(())
}

fn png_err(e: png::DecodingError) -> ImageError {
    ImageError::IoError(std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

fn png_encoding_err(e: png::EncodingError) -> ImageError {
    ImageError::IoError(std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn png_roundtrip_rgb() {
        let data = vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 128, 128, 128];
        let img = Image::from_raw(data.clone(), 2, 2, PixelFormat::Rgb).unwrap();

        let mut buf = Vec::new();
        write_png(&img, &mut buf).unwrap();

        let loaded = read_png(buf.as_slice()).unwrap();
        assert_eq!(loaded.width(), 2);
        assert_eq!(loaded.height(), 2);
        assert_eq!(loaded.format(), PixelFormat::Rgb);
        assert_eq!(loaded.as_slice(), data.as_slice());
    }

    #[test]
    fn png_roundtrip_gray() {
        let data = vec![0, 64, 128, 255];
        let img = Image::from_raw(data.clone(), 2, 2, PixelFormat::Gray).unwrap();

        let mut buf = Vec::new();
        write_png(&img, &mut buf).unwrap();

        let loaded = read_png(buf.as_slice()).unwrap();
        assert_eq!(loaded.format(), PixelFormat::Gray);
        assert_eq!(loaded.as_slice(), data.as_slice());
    }

    #[test]
    fn png_roundtrip_rgba() {
        let data = vec![255, 0, 0, 128, 0, 255, 0, 255];
        let img = Image::from_raw(data.clone(), 2, 1, PixelFormat::Rgba).unwrap();

        let mut buf = Vec::new();
        write_png(&img, &mut buf).unwrap();

        let loaded = read_png(buf.as_slice()).unwrap();
        assert_eq!(loaded.format(), PixelFormat::Rgba);
        assert_eq!(loaded.as_slice(), data.as_slice());
    }

    #[test]
    fn png_roundtrip_gray_alpha() {
        let data = vec![100, 200, 50, 150];
        let img = Image::from_raw(data.clone(), 2, 1, PixelFormat::GrayAlpha).unwrap();

        let mut buf = Vec::new();
        write_png(&img, &mut buf).unwrap();

        let loaded = read_png(buf.as_slice()).unwrap();
        assert_eq!(loaded.format(), PixelFormat::GrayAlpha);
        assert_eq!(loaded.as_slice(), data.as_slice());
    }

    #[test]
    fn png_invalid_data() {
        let bad = b"not a png file at all";
        assert!(read_png(bad.as_slice()).is_err());
    }

    #[test]
    fn png_1x1_image() {
        let data = vec![42, 43, 44];
        let img = Image::from_raw(data.clone(), 1, 1, PixelFormat::Rgb).unwrap();

        let mut buf = Vec::new();
        write_png(&img, &mut buf).unwrap();

        let loaded = read_png(buf.as_slice()).unwrap();
        assert_eq!(loaded.as_slice(), data.as_slice());
    }

    #[test]
    fn png_large_image() {
        let w = 100;
        let h = 80;
        let data: Vec<u8> = (0..w * h * 3).map(|i| (i % 256) as u8).collect();
        let img = Image::from_raw(data.clone(), w, h, PixelFormat::Rgb).unwrap();

        let mut buf = Vec::new();
        write_png(&img, &mut buf).unwrap();

        let loaded = read_png(buf.as_slice()).unwrap();
        assert_eq!(loaded.width(), w);
        assert_eq!(loaded.height(), h);
        assert_eq!(loaded.as_slice(), data.as_slice());
    }
}
