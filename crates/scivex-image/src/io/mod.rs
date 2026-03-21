pub mod bmp;
#[cfg(feature = "jpeg")]
pub mod jpeg;
#[cfg(feature = "png")]
pub mod png;
pub mod ppm;

use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use crate::error::{ImageError, Result};
use crate::image::Image;

/// Supported image file formats.
///
/// # Examples
///
/// ```
/// # use scivex_image::io::Format;
/// let fmt = Format::Ppm;
/// assert_eq!(fmt, Format::Ppm);
/// ```
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    /// PPM (Portable Pixmap) — color images.
    Ppm,
    /// PGM (Portable Graymap) — grayscale images.
    Pgm,
    /// BMP (Bitmap) — 24-bit uncompressed.
    Bmp,
    /// PNG (Portable Network Graphics).
    Png,
    /// JPEG / JPG.
    Jpeg,
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
        Some("png") => Ok(Format::Png),
        Some("jpg" | "jpeg") => Ok(Format::Jpeg),
        Some(other) => Err(ImageError::UnsupportedFormat {
            format: other.to_string(),
        }),
        None => Err(ImageError::UnsupportedFormat {
            format: "unknown (no extension)".into(),
        }),
    }
}

/// Load an image from a file, detecting the format from the extension.
///
/// # Examples
///
/// ```ignore
/// # use scivex_image::io;
/// // Requires a real file on disk; shown for documentation only.
/// let img = io::load("/tmp/image.ppm").unwrap();
/// assert!(img.width() > 0);
/// ```
pub fn load<P: AsRef<Path>>(path: P) -> Result<Image<u8>> {
    let path = path.as_ref();
    let format = detect_format(path)?;
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    match format {
        Format::Ppm | Format::Pgm => ppm::read_ppm(reader),
        Format::Bmp => bmp::read_bmp(reader),
        #[cfg(feature = "png")]
        Format::Png => png::read_png(reader),
        #[cfg(not(feature = "png"))]
        Format::Png => Err(ImageError::UnsupportedFormat {
            format: "PNG support requires the `png` feature".into(),
        }),
        #[cfg(feature = "jpeg")]
        Format::Jpeg => jpeg::read_jpeg(reader),
        #[cfg(not(feature = "jpeg"))]
        Format::Jpeg => Err(ImageError::UnsupportedFormat {
            format: "JPEG support requires the `jpeg` feature".into(),
        }),
    }
}

/// Save an image to a file, detecting the format from the extension.
///
/// # Examples
///
/// ```ignore
/// # use scivex_image::{Image, PixelFormat, io};
/// let img = Image::<u8>::new(4, 4, PixelFormat::Rgb).unwrap();
/// io::save(&img, "/tmp/out.ppm").unwrap();
/// ```
pub fn save<P: AsRef<Path>>(img: &Image<u8>, path: P) -> Result<()> {
    let path = path.as_ref();
    let format = detect_format(path)?;
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    match format {
        Format::Ppm => ppm::write_ppm(img, &mut writer),
        Format::Pgm => ppm::write_pgm(img, &mut writer),
        Format::Bmp => bmp::write_bmp(img, &mut writer),
        #[cfg(feature = "png")]
        Format::Png => png::write_png(img, writer),
        #[cfg(not(feature = "png"))]
        Format::Png => Err(ImageError::UnsupportedFormat {
            format: "PNG support requires the `png` feature".into(),
        }),
        #[cfg(feature = "jpeg")]
        Format::Jpeg => jpeg::write_jpeg(img, &mut writer),
        #[cfg(not(feature = "jpeg"))]
        Format::Jpeg => Err(ImageError::UnsupportedFormat {
            format: "JPEG support requires the `jpeg` feature".into(),
        }),
    }
}
