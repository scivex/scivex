pub mod bmp;
pub mod ppm;

use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use crate::error::{ImageError, Result};
use crate::image::Image;

/// Supported image file formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    /// PPM (Portable Pixmap) — color images.
    Ppm,
    /// PGM (Portable Graymap) — grayscale images.
    Pgm,
    /// BMP (Bitmap) — 24-bit uncompressed.
    Bmp,
}

/// Detect image format from a file extension.
fn detect_format(path: &Path) -> Result<Format> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_lowercase);

    match ext.as_deref() {
        Some("ppm") => Ok(Format::Ppm),
        Some("pgm") => Ok(Format::Pgm),
        Some("bmp") => Ok(Format::Bmp),
        Some(other) => Err(ImageError::UnsupportedFormat {
            format: other.to_string(),
        }),
        None => Err(ImageError::UnsupportedFormat {
            format: "unknown (no extension)".into(),
        }),
    }
}

/// Load an image from a file, detecting the format from the extension.
pub fn load<P: AsRef<Path>>(path: P) -> Result<Image<u8>> {
    let path = path.as_ref();
    let format = detect_format(path)?;
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    match format {
        Format::Ppm | Format::Pgm => ppm::read_ppm(reader),
        Format::Bmp => bmp::read_bmp(reader),
    }
}

/// Save an image to a file, detecting the format from the extension.
pub fn save<P: AsRef<Path>>(img: &Image<u8>, path: P) -> Result<()> {
    let path = path.as_ref();
    let format = detect_format(path)?;
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    match format {
        Format::Ppm => ppm::write_ppm(img, &mut writer),
        Format::Pgm => ppm::write_pgm(img, &mut writer),
        Format::Bmp => bmp::write_bmp(img, &mut writer),
    }
}
