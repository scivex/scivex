use std::io::{BufRead, BufReader, Read, Write};

use crate::error::{ImageError, Result};
use crate::image::{Image, PixelFormat};

/// Read a PPM (P6 binary) or PPM (P3 ASCII) image, or PGM (P5/P2).
pub fn read_ppm<R: Read>(reader: R) -> Result<Image<u8>> {
    let mut br = BufReader::new(reader);

    // Read magic number
    let mut magic = String::new();
    read_token(&mut br, &mut magic)?;

    match magic.as_str() {
        "P2" => read_pgm_ascii(&mut br),
        "P3" => read_ppm_ascii(&mut br),
        "P5" => read_pgm_binary(&mut br),
        "P6" => read_ppm_binary(&mut br),
        _ => Err(ImageError::UnsupportedFormat {
            format: format!("PPM magic: {magic}"),
        }),
    }
}

fn read_ppm_binary<R: BufRead>(reader: &mut R) -> Result<Image<u8>> {
    let mut token = String::new();

    read_token(reader, &mut token)?;
    let width: usize = token.parse().map_err(|_| ImageError::UnsupportedFormat {
        format: "invalid PPM width".into(),
    })?;

    token.clear();
    read_token(reader, &mut token)?;
    let height: usize = token.parse().map_err(|_| ImageError::UnsupportedFormat {
        format: "invalid PPM height".into(),
    })?;

    token.clear();
    read_token(reader, &mut token)?;
    let _max_val: u32 = token.parse().map_err(|_| ImageError::UnsupportedFormat {
        format: "invalid PPM maxval".into(),
    })?;

    // Read binary pixel data
    let mut data = vec![0u8; width * height * 3];
    reader.read_exact(&mut data)?;

    Image::from_raw(data, width, height, PixelFormat::Rgb)
}

fn read_ppm_ascii<R: BufRead>(reader: &mut R) -> Result<Image<u8>> {
    let mut token = String::new();

    read_token(reader, &mut token)?;
    let width: usize = parse_token(&token, "width")?;

    token.clear();
    read_token(reader, &mut token)?;
    let height: usize = parse_token(&token, "height")?;

    token.clear();
    read_token(reader, &mut token)?;
    let _max_val: u32 = parse_token(&token, "maxval")?;

    let count = width * height * 3;
    let mut data = Vec::with_capacity(count);
    for _ in 0..count {
        token.clear();
        read_token(reader, &mut token)?;
        let val: u8 = parse_token(&token, "pixel value")?;
        data.push(val);
    }

    Image::from_raw(data, width, height, PixelFormat::Rgb)
}

fn read_pgm_binary<R: BufRead>(reader: &mut R) -> Result<Image<u8>> {
    let mut token = String::new();

    read_token(reader, &mut token)?;
    let width: usize = parse_token(&token, "width")?;

    token.clear();
    read_token(reader, &mut token)?;
    let height: usize = parse_token(&token, "height")?;

    token.clear();
    read_token(reader, &mut token)?;
    let _max_val: u32 = parse_token(&token, "maxval")?;

    let mut data = vec![0u8; width * height];
    reader.read_exact(&mut data)?;

    Image::from_raw(data, width, height, PixelFormat::Gray)
}

fn read_pgm_ascii<R: BufRead>(reader: &mut R) -> Result<Image<u8>> {
    let mut token = String::new();

    read_token(reader, &mut token)?;
    let width: usize = parse_token(&token, "width")?;

    token.clear();
    read_token(reader, &mut token)?;
    let height: usize = parse_token(&token, "height")?;

    token.clear();
    read_token(reader, &mut token)?;
    let _max_val: u32 = parse_token(&token, "maxval")?;

    let count = width * height;
    let mut data = Vec::with_capacity(count);
    for _ in 0..count {
        token.clear();
        read_token(reader, &mut token)?;
        let val: u8 = parse_token(&token, "pixel value")?;
        data.push(val);
    }

    Image::from_raw(data, width, height, PixelFormat::Gray)
}

/// Write a PPM P6 (binary) image.
pub fn write_ppm<W: Write>(img: &Image<u8>, writer: &mut W) -> Result<()> {
    if img.format() != PixelFormat::Rgb {
        return Err(ImageError::UnsupportedChannels {
            channels: img.channels(),
        });
    }
    let header = format!("P6\n{} {}\n255\n", img.width(), img.height());
    writer.write_all(header.as_bytes())?;
    writer.write_all(img.as_slice())?;
    Ok(())
}

/// Write a PGM P5 (binary) image.
pub fn write_pgm<W: Write>(img: &Image<u8>, writer: &mut W) -> Result<()> {
    if img.format() != PixelFormat::Gray {
        return Err(ImageError::UnsupportedChannels {
            channels: img.channels(),
        });
    }
    let header = format!("P5\n{} {}\n255\n", img.width(), img.height());
    writer.write_all(header.as_bytes())?;
    writer.write_all(img.as_slice())?;
    Ok(())
}

/// Read whitespace-delimited tokens, skipping `#` comments.
fn read_token<R: BufRead>(reader: &mut R, buf: &mut String) -> Result<()> {
    buf.clear();
    loop {
        let mut byte = [0u8; 1];
        let n = reader.read(&mut byte)?;
        if n == 0 {
            if buf.is_empty() {
                return Err(ImageError::UnsupportedFormat {
                    format: "unexpected end of file".into(),
                });
            }
            return Ok(());
        }
        let ch = byte[0] as char;
        if ch == '#' {
            // Skip rest of line
            let mut comment = String::new();
            reader.read_line(&mut comment)?;
            continue;
        }
        if ch.is_ascii_whitespace() {
            if buf.is_empty() {
                continue;
            }
            return Ok(());
        }
        buf.push(ch);
    }
}

fn parse_token<T: std::str::FromStr>(token: &str, field: &str) -> Result<T> {
    token.parse().map_err(|_| ImageError::UnsupportedFormat {
        format: format!("invalid PPM {field}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ppm_roundtrip() {
        let data = vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 128, 128, 128];
        let img = Image::from_raw(data.clone(), 2, 2, PixelFormat::Rgb).unwrap();
        let mut buf = Vec::new();
        write_ppm(&img, &mut buf).unwrap();

        let loaded = read_ppm(buf.as_slice()).unwrap();
        assert_eq!(loaded.width(), 2);
        assert_eq!(loaded.height(), 2);
        assert_eq!(loaded.as_slice(), data.as_slice());
    }

    #[test]
    fn test_pgm_roundtrip() {
        let data = vec![0, 64, 128, 255];
        let img = Image::from_raw(data.clone(), 2, 2, PixelFormat::Gray).unwrap();
        let mut buf = Vec::new();
        write_pgm(&img, &mut buf).unwrap();

        let loaded = read_ppm(buf.as_slice()).unwrap();
        assert_eq!(loaded.format(), PixelFormat::Gray);
        assert_eq!(loaded.as_slice(), data.as_slice());
    }

    #[test]
    fn test_ppm_ascii() {
        let ascii = b"P3\n2 1\n255\n255 0 0 0 255 0\n";
        let img = read_ppm(ascii.as_slice()).unwrap();
        assert_eq!(img.width(), 2);
        assert_eq!(img.height(), 1);
        assert_eq!(img.get_pixel(0, 0).unwrap(), vec![255, 0, 0]);
        assert_eq!(img.get_pixel(0, 1).unwrap(), vec![0, 255, 0]);
    }
}
