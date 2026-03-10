use std::io::{Read, Write};

use crate::error::{ImageError, Result};
use crate::image::{Image, PixelFormat};

/// Read a 24-bit uncompressed BMP image.
pub fn read_bmp<R: Read>(mut reader: R) -> Result<Image<u8>> {
    // BMP file header (14 bytes)
    let mut header = [0u8; 14];
    reader.read_exact(&mut header)?;

    if header[0] != b'B' || header[1] != b'M' {
        return Err(ImageError::UnsupportedFormat {
            format: "not a BMP file".into(),
        });
    }

    let pixel_offset = u32_le(&header[10..14]) as usize;

    // DIB header (at least 40 bytes for BITMAPINFOHEADER)
    let mut dib_size_buf = [0u8; 4];
    reader.read_exact(&mut dib_size_buf)?;
    let dib_size = u32_le(&dib_size_buf) as usize;

    if dib_size < 40 {
        return Err(ImageError::UnsupportedFormat {
            format: "unsupported BMP DIB header size".into(),
        });
    }

    let mut dib = vec![0u8; dib_size - 4];
    reader.read_exact(&mut dib)?;

    let width = i32_le(&dib[0..4]) as usize;
    let raw_height = i32_le(&dib[4..8]);
    let top_down = raw_height < 0;
    let height = raw_height.unsigned_abs() as usize;
    let bits_per_pixel = u16_le(&dib[10..12]);
    let compression = u32_le(&dib[12..16]);

    if bits_per_pixel != 24 {
        return Err(ImageError::UnsupportedFormat {
            format: format!("{bits_per_pixel}-bit BMP not supported, only 24-bit"),
        });
    }
    if compression != 0 {
        return Err(ImageError::UnsupportedFormat {
            format: "compressed BMP not supported".into(),
        });
    }

    // Skip to pixel data
    let header_total = 14 + dib_size;
    if pixel_offset > header_total {
        let mut skip = vec![0u8; pixel_offset - header_total];
        reader.read_exact(&mut skip)?;
    }

    // Row stride: each row is padded to 4-byte boundary
    let row_bytes = width * 3;
    let row_stride = (row_bytes + 3) & !3;
    let padding = row_stride - row_bytes;

    let mut data = vec![0u8; width * height * 3];
    let mut row_buf = vec![0u8; row_stride];

    for row_idx in 0..height {
        reader.read_exact(&mut row_buf)?;
        // BMP stores pixels as BGR
        let dest_row = if top_down {
            row_idx
        } else {
            height - 1 - row_idx
        };
        let offset = dest_row * width * 3;
        for x in 0..width {
            let src = x * 3;
            let dst = offset + x * 3;
            data[dst] = row_buf[src + 2]; // R
            data[dst + 1] = row_buf[src + 1]; // G
            data[dst + 2] = row_buf[src]; // B
        }
    }

    // Discard any remaining padding bytes
    let _ = padding;

    Image::from_raw(data, width, height, PixelFormat::Rgb)
}

/// Write a 24-bit uncompressed BMP image (bottom-up).
#[allow(clippy::cast_possible_wrap)]
pub fn write_bmp<W: Write>(img: &Image<u8>, writer: &mut W) -> Result<()> {
    if img.format() != PixelFormat::Rgb {
        return Err(ImageError::UnsupportedChannels {
            channels: img.channels(),
        });
    }

    let width = img.width();
    let height = img.height();
    let row_bytes = width * 3;
    let row_stride = (row_bytes + 3) & !3;
    let padding = row_stride - row_bytes;
    let pixel_data_size = row_stride * height;
    let file_size = 14 + 40 + pixel_data_size;

    // File header (14 bytes)
    writer.write_all(b"BM")?;
    writer.write_all(&(file_size as u32).to_le_bytes())?;
    writer.write_all(&[0u8; 4])?; // reserved
    writer.write_all(&54u32.to_le_bytes())?; // pixel offset

    // DIB header (40 bytes - BITMAPINFOHEADER)
    writer.write_all(&40u32.to_le_bytes())?;
    writer.write_all(&(width as i32).to_le_bytes())?;
    writer.write_all(&(height as i32).to_le_bytes())?; // bottom-up
    writer.write_all(&1u16.to_le_bytes())?; // planes
    writer.write_all(&24u16.to_le_bytes())?; // bits per pixel
    writer.write_all(&0u32.to_le_bytes())?; // compression (none)
    writer.write_all(&(pixel_data_size as u32).to_le_bytes())?;
    writer.write_all(&2835u32.to_le_bytes())?; // h resolution (72 DPI)
    writer.write_all(&2835u32.to_le_bytes())?; // v resolution
    writer.write_all(&0u32.to_le_bytes())?; // palette colors
    writer.write_all(&0u32.to_le_bytes())?; // important colors

    let pad_bytes = [0u8; 3];
    let src = img.as_slice();

    // Write rows bottom-up, converting RGB to BGR
    for row in (0..height).rev() {
        let offset = row * width * 3;
        for x in 0..width {
            let i = offset + x * 3;
            writer.write_all(&[src[i + 2], src[i + 1], src[i]])?; // BGR
        }
        if padding > 0 {
            writer.write_all(&pad_bytes[..padding])?;
        }
    }

    Ok(())
}

fn u16_le(b: &[u8]) -> u16 {
    u16::from_le_bytes([b[0], b[1]])
}

fn u32_le(b: &[u8]) -> u32 {
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

fn i32_le(b: &[u8]) -> i32 {
    i32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bmp_roundtrip() {
        let data = vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 128, 128, 128];
        let img = Image::from_raw(data.clone(), 2, 2, PixelFormat::Rgb).unwrap();
        let mut buf = Vec::new();
        write_bmp(&img, &mut buf).unwrap();

        let loaded = read_bmp(buf.as_slice()).unwrap();
        assert_eq!(loaded.width(), 2);
        assert_eq!(loaded.height(), 2);
        assert_eq!(loaded.as_slice(), data.as_slice());
    }

    #[test]
    fn test_bmp_odd_width() {
        // Width 3 means row_bytes=9, padded to 12
        let data = vec![0u8; 3 * 2 * 3]; // 3x2 RGB
        let img = Image::from_raw(data.clone(), 3, 2, PixelFormat::Rgb).unwrap();
        let mut buf = Vec::new();
        write_bmp(&img, &mut buf).unwrap();

        let loaded = read_bmp(buf.as_slice()).unwrap();
        assert_eq!(loaded.width(), 3);
        assert_eq!(loaded.height(), 2);
        assert_eq!(loaded.as_slice(), data.as_slice());
    }
}
