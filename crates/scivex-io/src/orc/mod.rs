//! Apache ORC file format reader and writer.
//!
//! This module provides a from-scratch implementation of the Apache ORC
//! (Optimized Row Columnar) file format. It supports reading and writing
//! [`DataFrame`](scivex_frame::DataFrame) values with ORC primitive types:
//! boolean, int, long, float, double, and string.
//!
//! # ORC File Layout
//!
//! ```text
//! [Header "ORC"][Stripe 1]...[Stripe N][Footer][PostScript][PS Length (1 byte)]
//! ```
//!
//! The PostScript is at the end of the file; its length is stored in the
//! very last byte. The Footer (potentially compressed) contains type info,
//! stripe locations, and row counts.
//!
//! # Example
//!
//! ```no_run
//! use std::path::Path;
//! # fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
//! // let df = read_orc("data.orc")?;
//! // write_orc(&df, "output.orc")?;
//! # Ok(())
//! # }
//! ```

pub mod encoding;
pub mod proto;
pub mod stripe;

use std::path::Path;

use scivex_frame::{AnySeries, DataFrame, Series, StringSeries};

use crate::error::{IoError, Result};

use proto::{
    ColumnEncoding, ColumnEncodingKind, CompressionKind, Footer, OrcType, PostScript, Stream,
    StripeFooter, StripeInformation, TypeKind,
};
use stripe::ColumnData;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// ORC magic bytes at the start of the file.
const ORC_MAGIC: &[u8] = b"ORC";

/// Default maximum stripe size in rows (used when writing).
const DEFAULT_STRIPE_ROW_LIMIT: usize = 10_000;

// ---------------------------------------------------------------------------
// Public compression enum
// ---------------------------------------------------------------------------

/// Compression codec for ORC files.
///
/// # Examples
///
/// ```
/// use scivex_io::orc::OrcCompression;
/// assert_eq!(OrcCompression::None, OrcCompression::None);
/// assert_ne!(OrcCompression::None, OrcCompression::Zlib);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrcCompression {
    /// No compression.
    None,
    /// Zlib (deflate) compression.
    Zlib,
}

impl OrcCompression {
    fn to_proto(self) -> CompressionKind {
        match self {
            Self::None => CompressionKind::None,
            Self::Zlib => CompressionKind::Zlib,
        }
    }
}

// ---------------------------------------------------------------------------
// Public API — read
// ---------------------------------------------------------------------------

/// Read an ORC file from the given path into a [`DataFrame`].
///
/// # Examples
///
/// ```ignore
/// use scivex_io::orc::read_orc;
/// let df = read_orc("data.orc").unwrap();
/// assert!(df.nrows() > 0);
/// ```
pub fn read_orc(path: impl AsRef<Path>) -> Result<DataFrame> {
    let data = std::fs::read(path.as_ref())?;
    read_orc_bytes(&data)
}

/// Read an ORC file from an in-memory byte slice into a [`DataFrame`].
///
/// # Examples
///
/// ```
/// use scivex_io::orc::{write_orc_to_bytes, read_orc_bytes, OrcCompression};
/// use scivex_frame::{DataFrame, Series, AnySeries};
///
/// let col: Box<dyn AnySeries> = Box::new(Series::new("x", vec![1_i64, 2, 3]));
/// let df = DataFrame::new(vec![col]).unwrap();
/// let bytes = write_orc_to_bytes(&df, OrcCompression::None, 10_000).unwrap();
/// let df2 = read_orc_bytes(&bytes).unwrap();
/// assert_eq!(df2.nrows(), 3);
/// ```
pub fn read_orc_bytes(data: &[u8]) -> Result<DataFrame> {
    if data.len() < 4 {
        return Err(IoError::FormatError("ORC file too small".into()));
    }

    // Validate magic header
    if &data[..3] != ORC_MAGIC {
        return Err(IoError::FormatError(
            "invalid ORC magic bytes (expected 'ORC')".into(),
        ));
    }

    // PostScript length is the last byte
    let ps_len = *data.last().unwrap() as usize;
    if ps_len == 0 || data.len() < 4 + ps_len + 1 {
        return Err(IoError::FormatError(
            "ORC file too small for PostScript".into(),
        ));
    }

    // Parse PostScript
    let ps_start = data.len() - 1 - ps_len;
    let ps_bytes = &data[ps_start..data.len() - 1];
    let postscript = PostScript::parse(ps_bytes)?;

    // Validate PostScript magic
    if !postscript.magic.is_empty() && postscript.magic != "ORC" {
        return Err(IoError::FormatError(format!(
            "PostScript magic mismatch: expected 'ORC', got '{}'",
            postscript.magic
        )));
    }

    let compression = postscript.compression;

    // Parse Footer
    #[allow(clippy::cast_possible_truncation)]
    let footer_len = postscript.footer_length as usize;
    #[allow(clippy::cast_possible_truncation)]
    let metadata_len = postscript.metadata_length as usize;

    let footer_end = ps_start - metadata_len;
    if footer_len > footer_end {
        return Err(IoError::FormatError("footer exceeds file bounds".into()));
    }
    let footer_start = footer_end - footer_len;
    let footer_raw = &data[footer_start..footer_end];

    let footer_bytes = decompress_if_needed(footer_raw, compression)?;
    let footer = Footer::parse(&footer_bytes)?;

    if footer.types.is_empty() {
        return Ok(DataFrame::empty());
    }

    // Read stripes
    let _num_cols = footer.types[0].subtypes.len();
    let mut all_columns: Vec<Vec<ColumnData>> = Vec::new();

    for stripe_info in &footer.stripes {
        let stripe_columns = read_stripe(data, stripe_info, &footer.types, compression)?;
        all_columns.push(stripe_columns);
    }

    // Merge stripe column data into Series
    build_dataframe(&footer, &all_columns)
}

/// Read a single stripe and return its decoded columns.
fn read_stripe(
    file_data: &[u8],
    stripe_info: &StripeInformation,
    types: &[OrcType],
    compression: CompressionKind,
) -> Result<Vec<ColumnData>> {
    #[allow(clippy::cast_possible_truncation)]
    let offset = stripe_info.offset as usize;
    #[allow(clippy::cast_possible_truncation)]
    let index_len = stripe_info.index_length as usize;
    #[allow(clippy::cast_possible_truncation)]
    let data_len = stripe_info.data_length as usize;
    #[allow(clippy::cast_possible_truncation)]
    let footer_len = stripe_info.footer_length as usize;
    #[allow(clippy::cast_possible_truncation)]
    let num_rows = stripe_info.number_of_rows as usize;

    let total_len = index_len + data_len + footer_len;
    if offset + total_len > file_data.len() {
        return Err(IoError::FormatError(
            "stripe extends beyond file bounds".into(),
        ));
    }

    // Parse stripe footer
    let sf_start = offset + index_len + data_len;
    let sf_raw = &file_data[sf_start..sf_start + footer_len];
    let sf_bytes = decompress_if_needed(sf_raw, compression)?;
    let stripe_footer = StripeFooter::parse(&sf_bytes)?;

    // The stripe data includes index + data streams (footer is separate)
    let stripe_data = &file_data[offset..offset + index_len + data_len];

    stripe::read_stripe_columns(stripe_data, &stripe_footer, types, num_rows, compression)
}

/// Decompress bytes if compression is enabled.
fn decompress_if_needed(data: &[u8], compression: CompressionKind) -> Result<Vec<u8>> {
    match compression {
        CompressionKind::None => Ok(data.to_vec()),
        CompressionKind::Zlib => {
            use std::io::Read as _;
            // ORC zlib block format
            let mut result = Vec::new();
            let mut pos = 0;
            while pos < data.len() {
                if pos + 3 > data.len() {
                    return Err(IoError::FormatError("zlib: truncated block header".into()));
                }
                let b0 = u32::from(data[pos]);
                let b1 = u32::from(data[pos + 1]);
                let b2 = u32::from(data[pos + 2]);
                pos += 3;

                let is_original = (b0 & 1) != 0;
                let block_len = ((b0 >> 1) | (b1 << 7) | (b2 << 15)) as usize;

                if pos + block_len > data.len() {
                    return Err(IoError::FormatError(
                        "zlib: block exceeds available data".into(),
                    ));
                }

                let block_data = &data[pos..pos + block_len];
                pos += block_len;

                if is_original {
                    result.extend_from_slice(block_data);
                } else {
                    let mut decoder = flate2::read::DeflateDecoder::new(block_data);
                    decoder.read_to_end(&mut result).map_err(|e| {
                        IoError::FormatError(format!("zlib decompression failed: {e}"))
                    })?;
                }
            }
            Ok(result)
        }
        other => Err(IoError::FormatError(format!(
            "unsupported compression: {other:?}"
        ))),
    }
}

/// Build a DataFrame from decoded stripe column data.
fn build_dataframe(footer: &Footer, all_stripes: &[Vec<ColumnData>]) -> Result<DataFrame> {
    if footer.types.is_empty() {
        return Ok(DataFrame::empty());
    }

    let root = &footer.types[0];
    let num_cols = root.subtypes.len();

    if num_cols == 0 {
        return Ok(DataFrame::empty());
    }

    let mut series_vec: Vec<Box<dyn AnySeries>> = Vec::with_capacity(num_cols);

    for ci in 0..num_cols {
        let col_type_id = root.subtypes[ci] as usize;
        let type_info = &footer.types[col_type_id];
        let col_name = if ci < root.field_names.len() {
            root.field_names[ci].clone()
        } else {
            format!("col_{ci}")
        };

        let series = merge_column_data(&col_name, type_info, all_stripes, ci)?;
        series_vec.push(series);
    }

    if series_vec.is_empty() {
        return Ok(DataFrame::empty());
    }
    Ok(DataFrame::new(series_vec)?)
}

/// Merge column data from multiple stripes into a single Series.
fn merge_column_data(
    name: &str,
    type_info: &OrcType,
    all_stripes: &[Vec<ColumnData>],
    col_index: usize,
) -> Result<Box<dyn AnySeries>> {
    match type_info.kind {
        TypeKind::Boolean => {
            let mut all_vals: Vec<u8> = Vec::new();
            for stripe_cols in all_stripes {
                if let ColumnData::Boolean(ref vals) = stripe_cols[col_index] {
                    all_vals.extend(vals.iter().map(|&b| u8::from(b)));
                }
            }
            Ok(Box::new(Series::new(name, all_vals)))
        }
        TypeKind::Int | TypeKind::Short | TypeKind::Byte => {
            let mut all_vals = Vec::new();
            for stripe_cols in all_stripes {
                if let ColumnData::Int(ref vals) = stripe_cols[col_index] {
                    all_vals.extend_from_slice(vals);
                }
            }
            Ok(Box::new(Series::new(name, all_vals)))
        }
        TypeKind::Long => {
            let mut all_vals = Vec::new();
            for stripe_cols in all_stripes {
                if let ColumnData::Long(ref vals) = stripe_cols[col_index] {
                    all_vals.extend_from_slice(vals);
                }
            }
            Ok(Box::new(Series::new(name, all_vals)))
        }
        TypeKind::Float => {
            let mut all_vals = Vec::new();
            for stripe_cols in all_stripes {
                if let ColumnData::Float(ref vals) = stripe_cols[col_index] {
                    all_vals.extend_from_slice(vals);
                }
            }
            Ok(Box::new(Series::new(name, all_vals)))
        }
        TypeKind::Double => {
            let mut all_vals = Vec::new();
            for stripe_cols in all_stripes {
                if let ColumnData::Double(ref vals) = stripe_cols[col_index] {
                    all_vals.extend_from_slice(vals);
                }
            }
            Ok(Box::new(Series::new(name, all_vals)))
        }
        TypeKind::String | TypeKind::Varchar | TypeKind::Char => {
            let mut all_vals = Vec::new();
            for stripe_cols in all_stripes {
                if let ColumnData::Str(ref vals) = stripe_cols[col_index] {
                    all_vals.extend(vals.iter().cloned());
                }
            }
            Ok(Box::new(StringSeries::new(name, all_vals)))
        }
        other => Err(IoError::FormatError(format!(
            "unsupported ORC type for column '{name}': {other:?}"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Public API — write
// ---------------------------------------------------------------------------

/// Write a [`DataFrame`] to an ORC file at the given path.
///
/// Uses `OrcCompression::None` by default. For zlib compression, use
/// [`write_orc_with_options`].
///
/// # Examples
///
/// ```ignore
/// use scivex_io::orc::write_orc;
/// use scivex_frame::{DataFrame, Series, AnySeries};
/// let col: Box<dyn AnySeries> = Box::new(Series::new("v", vec![1_i64, 2]));
/// let df = DataFrame::new(vec![col]).unwrap();
/// write_orc(&df, "out.orc").unwrap();
/// ```
pub fn write_orc(df: &DataFrame, path: impl AsRef<Path>) -> Result<()> {
    write_orc_with_options(df, path, OrcCompression::None, DEFAULT_STRIPE_ROW_LIMIT)
}

/// Write a [`DataFrame`] to an ORC file with explicit compression and stripe
/// size settings.
///
/// # Examples
///
/// ```ignore
/// use scivex_io::orc::{write_orc_with_options, OrcCompression};
/// use scivex_frame::{DataFrame, Series, AnySeries};
/// let col: Box<dyn AnySeries> = Box::new(Series::new("v", vec![1_i64, 2]));
/// let df = DataFrame::new(vec![col]).unwrap();
/// write_orc_with_options(&df, "out.orc", OrcCompression::Zlib, 5_000).unwrap();
/// ```
pub fn write_orc_with_options(
    df: &DataFrame,
    path: impl AsRef<Path>,
    compression: OrcCompression,
    stripe_row_limit: usize,
) -> Result<()> {
    let bytes = write_orc_to_bytes(df, compression, stripe_row_limit)?;
    std::fs::write(path.as_ref(), &bytes)?;
    Ok(())
}

/// Write a [`DataFrame`] to ORC format in memory, returning the bytes.
///
/// # Examples
///
/// ```
/// use scivex_io::orc::{write_orc_to_bytes, OrcCompression};
/// use scivex_frame::{DataFrame, Series, AnySeries};
///
/// let col: Box<dyn AnySeries> = Box::new(Series::new("n", vec![10_i64, 20]));
/// let df = DataFrame::new(vec![col]).unwrap();
/// let bytes = write_orc_to_bytes(&df, OrcCompression::None, 10_000).unwrap();
/// assert!(bytes.starts_with(b"ORC"));
/// ```
#[allow(clippy::too_many_lines)]
pub fn write_orc_to_bytes(
    df: &DataFrame,
    compression: OrcCompression,
    stripe_row_limit: usize,
) -> Result<Vec<u8>> {
    let comp = compression.to_proto();
    let columns = df.columns();
    let nrows = df.nrows();
    let ncols = columns.len();

    // Build the type tree: type 0 = struct root, types 1..=N = leaf columns.
    let mut types = Vec::with_capacity(1 + ncols);

    // Root struct type
    let subtypes: Vec<u32> = (1..=ncols as u32).collect();
    let field_names: Vec<String> = columns.iter().map(|c| c.name().to_string()).collect();
    types.push(OrcType {
        kind: TypeKind::Struct,
        subtypes,
        field_names,
    });

    // Leaf column types
    for col in columns {
        let tk = stripe::dtype_to_orc_type(col.dtype());
        types.push(OrcType {
            kind: tk,
            subtypes: Vec::new(),
            field_names: Vec::new(),
        });
    }

    let mut file_buf = Vec::new();

    // Write magic header
    file_buf.extend_from_slice(ORC_MAGIC);

    // Write stripes
    let mut stripe_infos = Vec::new();
    let stripe_limit = if stripe_row_limit == 0 {
        DEFAULT_STRIPE_ROW_LIMIT
    } else {
        stripe_row_limit
    };

    let mut row_offset = 0;
    while row_offset < nrows || (nrows == 0 && stripe_infos.is_empty()) {
        let rows_in_stripe = if nrows == 0 {
            0
        } else {
            (nrows - row_offset).min(stripe_limit)
        };

        let stripe_start = file_buf.len();

        // Encode each column's slice for this stripe
        let mut encoded_cols = Vec::with_capacity(ncols);
        for (ci, col) in columns.iter().enumerate() {
            let col_slice: Box<dyn AnySeries> = if nrows == 0 {
                col.slice(0, 0)
            } else {
                col.slice(row_offset, rows_in_stripe)
            };
            let ec = stripe::encode_column(&*col_slice, &types[ci + 1], comp)?;
            encoded_cols.push(ec);
        }

        // Build stripe footer
        let mut stripe_streams = Vec::new();
        let mut stripe_encodings = Vec::new();

        // Column 0 (struct root) has no streams but needs a placeholder encoding.
        stripe_encodings.push(ColumnEncoding {
            kind: ColumnEncodingKind::Direct,
            dictionary_size: 0,
        });

        // Collect all stream data in order
        let mut all_stream_data = Vec::new();

        for (ci, ec) in encoded_cols.iter().enumerate() {
            let col_id = (ci + 1) as u32;
            for (stream_kind, stream_data) in &ec.streams {
                stripe_streams.push(Stream {
                    kind: *stream_kind,
                    column: col_id,
                    length: stream_data.len() as u64,
                });
                all_stream_data.extend_from_slice(stream_data);
            }
            stripe_encodings.push(ec.encoding.clone());
        }

        let sf = StripeFooter {
            streams: stripe_streams,
            columns: stripe_encodings,
        };

        let sf_bytes = sf.encode();
        let sf_compressed = compress_for_write(&sf_bytes, comp)?;

        // Write stripe data (index streams: none, data streams, stripe footer)
        let data_length = all_stream_data.len() as u64;
        file_buf.extend_from_slice(&all_stream_data);
        file_buf.extend_from_slice(&sf_compressed);

        let stripe_info = StripeInformation {
            offset: stripe_start as u64,
            index_length: 0,
            data_length,
            footer_length: sf_compressed.len() as u64,
            number_of_rows: rows_in_stripe as u64,
        };
        stripe_infos.push(stripe_info);

        row_offset += rows_in_stripe;
        if nrows == 0 {
            break;
        }
    }

    // Build and write footer
    let footer = Footer {
        header_length: ORC_MAGIC.len() as u64,
        content_length: (file_buf.len() - ORC_MAGIC.len()) as u64,
        stripes: stripe_infos,
        types,
        number_of_rows: nrows as u64,
        row_index_stride: 0,
    };

    let footer_bytes = footer.encode();
    let footer_compressed = compress_for_write(&footer_bytes, comp)?;
    file_buf.extend_from_slice(&footer_compressed);

    // Build and write PostScript
    let postscript = PostScript {
        footer_length: footer_compressed.len() as u64,
        compression: comp,
        compression_block_size: 262_144,
        metadata_length: 0,
        version: vec![0, 12], // ORC version 0.12
        magic: "ORC".into(),
    };

    let ps_bytes = postscript.encode();
    if ps_bytes.len() > 255 {
        return Err(IoError::FormatError(
            "PostScript too large (> 255 bytes)".into(),
        ));
    }
    file_buf.extend_from_slice(&ps_bytes);
    file_buf.push(ps_bytes.len() as u8);

    Ok(file_buf)
}

/// Compress bytes for writing using the specified compression kind.
fn compress_for_write(data: &[u8], compression: CompressionKind) -> Result<Vec<u8>> {
    match compression {
        CompressionKind::None => Ok(data.to_vec()),
        CompressionKind::Zlib => {
            use std::io::Write as _;

            let mut compressed = Vec::new();
            {
                let mut encoder = flate2::write::DeflateEncoder::new(
                    &mut compressed,
                    flate2::Compression::default(),
                );
                encoder
                    .write_all(data)
                    .map_err(|e| IoError::FormatError(format!("zlib compression failed: {e}")))?;
                encoder.finish().map_err(|e| {
                    IoError::FormatError(format!("zlib compression finish failed: {e}"))
                })?;
            }

            let mut result = Vec::new();
            if compressed.len() < data.len() {
                let block_len = compressed.len() as u32;
                let header = block_len << 1;
                result.push((header & 0xFF) as u8);
                result.push(((header >> 8) & 0xFF) as u8);
                result.push(((header >> 16) & 0xFF) as u8);
                result.extend_from_slice(&compressed);
            } else {
                let block_len = data.len() as u32;
                let header = (block_len << 1) | 1;
                result.push((header & 0xFF) as u8);
                result.push(((header >> 8) & 0xFF) as u8);
                result.push(((header >> 16) & 0xFF) as u8);
                result.extend_from_slice(data);
            }
            Ok(result)
        }
        other => Err(IoError::FormatError(format!(
            "unsupported compression for writing: {other:?}"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scivex_frame::{AnySeries, DataFrame, Series, StringSeries};

    #[test]
    fn test_orc_roundtrip_integer() {
        let col: Box<dyn AnySeries> = Box::new(Series::new("ids", vec![10_i32, 20, 30, -5, 0]));
        let df = DataFrame::new(vec![col]).unwrap();

        let bytes = write_orc_to_bytes(&df, OrcCompression::None, 10_000).unwrap();
        let df2 = read_orc_bytes(&bytes).unwrap();

        assert_eq!(df2.nrows(), 5);
        assert_eq!(df2.ncols(), 1);
        let col2 = df2.column("ids").unwrap();
        let typed = col2.as_any().downcast_ref::<Series<i32>>().unwrap();
        assert_eq!(typed.as_slice(), &[10, 20, 30, -5, 0]);
    }

    #[test]
    fn test_orc_roundtrip_long() {
        let col: Box<dyn AnySeries> = Box::new(Series::new(
            "longs",
            vec![100_i64, -200, i64::MAX, i64::MIN, 0],
        ));
        let df = DataFrame::new(vec![col]).unwrap();

        let bytes = write_orc_to_bytes(&df, OrcCompression::None, 10_000).unwrap();
        let df2 = read_orc_bytes(&bytes).unwrap();

        let col2 = df2.column("longs").unwrap();
        let typed = col2.as_any().downcast_ref::<Series<i64>>().unwrap();
        assert_eq!(typed.as_slice(), &[100, -200, i64::MAX, i64::MIN, 0]);
    }

    #[test]
    fn test_orc_roundtrip_float_double() {
        let f32_col: Box<dyn AnySeries> =
            Box::new(Series::new("floats", vec![1.5_f32, -2.25, 0.0, 2.78]));
        let f64_col: Box<dyn AnySeries> = Box::new(Series::new(
            "doubles",
            vec![100.0_f64, -0.001, 1e10, std::f64::consts::PI],
        ));
        let df = DataFrame::new(vec![f32_col, f64_col]).unwrap();

        let bytes = write_orc_to_bytes(&df, OrcCompression::None, 10_000).unwrap();
        let df2 = read_orc_bytes(&bytes).unwrap();

        assert_eq!(df2.nrows(), 4);
        assert_eq!(df2.ncols(), 2);

        let f32_read = df2.column("floats").unwrap();
        let f32_typed = f32_read.as_any().downcast_ref::<Series<f32>>().unwrap();
        assert_eq!(f32_typed.as_slice(), &[1.5_f32, -2.25, 0.0, 2.78]);

        let f64_read = df2.column("doubles").unwrap();
        let f64_typed = f64_read.as_any().downcast_ref::<Series<f64>>().unwrap();
        assert_eq!(
            f64_typed.as_slice(),
            &[100.0_f64, -0.001, 1e10, std::f64::consts::PI]
        );
    }

    #[test]
    fn test_orc_roundtrip_u8() {
        // u8 columns map to ORC Int type
        let col: Box<dyn AnySeries> = Box::new(Series::new("flags", vec![1_u8, 0, 1, 1, 0]));
        let df = DataFrame::new(vec![col]).unwrap();

        let bytes = write_orc_to_bytes(&df, OrcCompression::None, 10_000).unwrap();
        let df2 = read_orc_bytes(&bytes).unwrap();

        // u8 gets widened to i32 (ORC Int type) on read
        let col2 = df2.column("flags").unwrap();
        let typed = col2.as_any().downcast_ref::<Series<i32>>().unwrap();
        assert_eq!(typed.as_slice(), &[1_i32, 0, 1, 1, 0]);
    }

    #[test]
    fn test_orc_roundtrip_string() {
        let col: Box<dyn AnySeries> = Box::new(StringSeries::from_strs(
            "names",
            &["Alice", "Bob", "", "Carol", "Dave"],
        ));
        let df = DataFrame::new(vec![col]).unwrap();

        let bytes = write_orc_to_bytes(&df, OrcCompression::None, 10_000).unwrap();
        let df2 = read_orc_bytes(&bytes).unwrap();

        let col2 = df2.column("names").unwrap();
        let typed = col2.as_any().downcast_ref::<StringSeries>().unwrap();
        assert_eq!(typed.get(0), Some("Alice"));
        assert_eq!(typed.get(1), Some("Bob"));
        assert_eq!(typed.get(2), Some(""));
        assert_eq!(typed.get(3), Some("Carol"));
        assert_eq!(typed.get(4), Some("Dave"));
    }

    #[test]
    fn test_orc_roundtrip_empty() {
        let df = DataFrame::empty();

        let bytes = write_orc_to_bytes(&df, OrcCompression::None, 10_000).unwrap();
        let df2 = read_orc_bytes(&bytes).unwrap();

        assert_eq!(df2.nrows(), 0);
        assert_eq!(df2.ncols(), 0);
    }

    #[test]
    fn test_orc_multi_stripe() {
        // Write enough rows to trigger multiple stripes with a small limit.
        let n = 250;
        let vals: Vec<i64> = (0..n).collect();
        let col: Box<dyn AnySeries> = Box::new(Series::new("x", vals.clone()));
        let df = DataFrame::new(vec![col]).unwrap();

        // Use a stripe limit of 100, so we get 3 stripes (100 + 100 + 50).
        let bytes = write_orc_to_bytes(&df, OrcCompression::None, 100).unwrap();
        let df2 = read_orc_bytes(&bytes).unwrap();

        assert_eq!(df2.nrows(), 250);
        let col2 = df2.column("x").unwrap();
        let typed = col2.as_any().downcast_ref::<Series<i64>>().unwrap();
        assert_eq!(typed.as_slice(), &vals[..]);
    }

    #[test]
    fn test_orc_roundtrip_mixed_types() {
        let bool_col: Box<dyn AnySeries> = Box::new(Series::new("flag", vec![1_u8, 0, 1]));
        let int_col: Box<dyn AnySeries> = Box::new(Series::new("id", vec![1_i32, 2, 3]));
        let long_col: Box<dyn AnySeries> = Box::new(Series::new("big", vec![100_i64, 200, 300]));
        let f32_col: Box<dyn AnySeries> = Box::new(Series::new("score", vec![1.5_f32, 2.5, 3.5]));
        let f64_col: Box<dyn AnySeries> =
            Box::new(Series::new("amount", vec![10.0_f64, 20.0, 30.0]));
        let str_col: Box<dyn AnySeries> =
            Box::new(StringSeries::from_strs("name", &["a", "b", "c"]));

        let df =
            DataFrame::new(vec![bool_col, int_col, long_col, f32_col, f64_col, str_col]).unwrap();

        let bytes = write_orc_to_bytes(&df, OrcCompression::None, 10_000).unwrap();
        let df2 = read_orc_bytes(&bytes).unwrap();

        assert_eq!(df2.nrows(), 3);
        assert_eq!(df2.ncols(), 6);
        assert_eq!(
            df2.column_names(),
            vec!["flag", "id", "big", "score", "amount", "name"]
        );

        // Verify each column type
        let flag = df2.column("flag").unwrap();
        assert_eq!(
            flag.as_any()
                .downcast_ref::<Series<i32>>()
                .unwrap()
                .as_slice(),
            &[1_i32, 0, 1]
        );

        let id = df2.column("id").unwrap();
        assert_eq!(
            id.as_any()
                .downcast_ref::<Series<i32>>()
                .unwrap()
                .as_slice(),
            &[1, 2, 3]
        );

        let big = df2.column("big").unwrap();
        assert_eq!(
            big.as_any()
                .downcast_ref::<Series<i64>>()
                .unwrap()
                .as_slice(),
            &[100, 200, 300]
        );

        let score = df2.column("score").unwrap();
        assert_eq!(
            score
                .as_any()
                .downcast_ref::<Series<f32>>()
                .unwrap()
                .as_slice(),
            &[1.5_f32, 2.5, 3.5]
        );

        let amount = df2.column("amount").unwrap();
        assert_eq!(
            amount
                .as_any()
                .downcast_ref::<Series<f64>>()
                .unwrap()
                .as_slice(),
            &[10.0_f64, 20.0, 30.0]
        );

        let name = df2.column("name").unwrap();
        let name_typed = name.as_any().downcast_ref::<StringSeries>().unwrap();
        assert_eq!(name_typed.get(0), Some("a"));
        assert_eq!(name_typed.get(1), Some("b"));
        assert_eq!(name_typed.get(2), Some("c"));
    }

    #[test]
    fn test_orc_roundtrip_zlib_compression() {
        let col: Box<dyn AnySeries> = Box::new(Series::new("vals", vec![1_i64, 2, 3, 4, 5]));
        let str_col: Box<dyn AnySeries> = Box::new(StringSeries::from_strs(
            "words",
            &["hello", "world", "foo", "bar", "baz"],
        ));
        let df = DataFrame::new(vec![col, str_col]).unwrap();

        let bytes = write_orc_to_bytes(&df, OrcCompression::Zlib, 10_000).unwrap();
        let df2 = read_orc_bytes(&bytes).unwrap();

        assert_eq!(df2.nrows(), 5);
        let vals = df2.column("vals").unwrap();
        let typed = vals.as_any().downcast_ref::<Series<i64>>().unwrap();
        assert_eq!(typed.as_slice(), &[1, 2, 3, 4, 5]);

        let words = df2.column("words").unwrap();
        let words_typed = words.as_any().downcast_ref::<StringSeries>().unwrap();
        assert_eq!(words_typed.get(0), Some("hello"));
        assert_eq!(words_typed.get(4), Some("baz"));
    }
}
