//! ORC stripe reading and writing.
//!
//! Each ORC stripe contains column data encoded with RLE, optionally
//! compressed (zlib). This module handles decoding stripe data into
//! column vectors and encoding column vectors into stripe bytes.

use scivex_frame::{AnySeries, DType, Series, StringSeries};

use crate::error::{IoError, Result};

use super::encoding;
use super::proto::{
    ColumnEncoding, ColumnEncodingKind, CompressionKind, OrcType, StreamKind, StripeFooter,
    TypeKind,
};

// ---------------------------------------------------------------------------
// Column value container
// ---------------------------------------------------------------------------

/// Decoded column data from a stripe.
#[derive(Debug)]
pub(crate) enum ColumnData {
    Boolean(Vec<bool>),
    Int(Vec<i32>),
    Long(Vec<i64>),
    Float(Vec<f32>),
    Double(Vec<f64>),
    Str(Vec<String>),
}

// ---------------------------------------------------------------------------
// Stripe reading
// ---------------------------------------------------------------------------

/// Read column data from a stripe's raw bytes.
///
/// `stripe_data` is the raw bytes of the stripe (index + data regions).
/// `stripe_footer` is the parsed stripe footer.
/// `types` is the full type tree from the file footer.
/// `num_rows` is the number of rows in this stripe.
/// `compression` is the file-level compression setting.
#[allow(clippy::too_many_lines)]
pub(crate) fn read_stripe_columns(
    stripe_data: &[u8],
    stripe_footer: &StripeFooter,
    types: &[OrcType],
    num_rows: usize,
    compression: CompressionKind,
) -> Result<Vec<ColumnData>> {
    // The first type is the struct root; child columns are types[1..].
    if types.is_empty() {
        return Ok(Vec::new());
    }
    let root = &types[0];
    let num_cols = root.subtypes.len();
    let mut columns = Vec::with_capacity(num_cols);

    // Build a map of (column_id, stream_kind) -> byte range in stripe_data.
    // Streams are laid out sequentially in the order listed in the stripe footer.
    let mut stream_map: std::collections::HashMap<(u32, StreamKind), (usize, usize)> =
        std::collections::HashMap::new();
    let mut offset = 0;
    for stream in &stripe_footer.streams {
        #[allow(clippy::cast_possible_truncation)]
        let len = stream.length as usize;
        stream_map.insert((stream.column, stream.kind), (offset, len));
        offset += len;
    }

    for (ci, &col_type_id) in root.subtypes.iter().enumerate() {
        let col_id = col_type_id;
        let type_info = types
            .get(col_id as usize)
            .ok_or_else(|| IoError::FormatError(format!("missing type for column {col_id}")))?;

        let col_encoding = stripe_footer.columns.get(col_id as usize).ok_or_else(|| {
            IoError::FormatError(format!("missing column encoding for column {col_id}"))
        })?;

        let col_data = decode_column(
            stripe_data,
            &stream_map,
            col_id,
            type_info,
            col_encoding,
            num_rows,
            compression,
        )
        .map_err(|e| {
            IoError::FormatError(format!(
                "error decoding column {ci} (type_id={col_id}): {e}"
            ))
        })?;

        columns.push(col_data);
    }

    Ok(columns)
}

/// Decompress a stream's raw bytes if needed.
fn decompress_stream(raw: &[u8], compression: CompressionKind) -> Result<Vec<u8>> {
    match compression {
        CompressionKind::None => Ok(raw.to_vec()),
        CompressionKind::Zlib => decompress_orc_zlib(raw),
        other => Err(IoError::FormatError(format!(
            "unsupported compression: {other:?}"
        ))),
    }
}

/// ORC zlib compression uses a block format: each block starts with a 3-byte
/// header. Bit 0 of the first byte indicates if the block is uncompressed (1)
/// or compressed (0). The remaining 23 bits (across the 3 bytes, little-endian)
/// give the block length.
fn decompress_orc_zlib(data: &[u8]) -> Result<Vec<u8>> {
    use std::io::Read as _;

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
            // Uncompressed block
            result.extend_from_slice(block_data);
        } else {
            // Zlib-compressed block
            let mut decoder = flate2::read::DeflateDecoder::new(block_data);
            decoder
                .read_to_end(&mut result)
                .map_err(|e| IoError::FormatError(format!("zlib decompression failed: {e}")))?;
        }
    }

    Ok(result)
}

/// Get stream bytes from the stripe data, applying decompression if needed.
fn get_stream_bytes(
    stripe_data: &[u8],
    stream_map: &std::collections::HashMap<(u32, StreamKind), (usize, usize)>,
    col_id: u32,
    kind: StreamKind,
    compression: CompressionKind,
) -> Result<Option<Vec<u8>>> {
    let entry = stream_map.get(&(col_id, kind));
    match entry {
        Some(&(offset, len)) => {
            if offset + len > stripe_data.len() {
                return Err(IoError::FormatError(format!(
                    "stream ({col_id}, {kind:?}) exceeds stripe data bounds"
                )));
            }
            let raw = &stripe_data[offset..offset + len];
            let decompressed = decompress_stream(raw, compression)?;
            Ok(Some(decompressed))
        }
        None => Ok(None),
    }
}

/// Decode a single column from its streams.
#[allow(clippy::too_many_lines)]
fn decode_column(
    stripe_data: &[u8],
    stream_map: &std::collections::HashMap<(u32, StreamKind), (usize, usize)>,
    col_id: u32,
    type_info: &OrcType,
    col_encoding: &ColumnEncoding,
    num_rows: usize,
    compression: CompressionKind,
) -> Result<ColumnData> {
    match type_info.kind {
        TypeKind::Boolean => {
            let data_bytes = get_stream_bytes(
                stripe_data,
                stream_map,
                col_id,
                StreamKind::Data,
                compression,
            )?
            .unwrap_or_default();
            let bools = encoding::decode_booleans(&data_bytes, num_rows)?;
            Ok(ColumnData::Boolean(bools))
        }
        TypeKind::Int | TypeKind::Short | TypeKind::Byte => {
            let data_bytes = get_stream_bytes(
                stripe_data,
                stream_map,
                col_id,
                StreamKind::Data,
                compression,
            )?
            .unwrap_or_default();
            let ints = encoding::decode_int_rle_v1(&data_bytes, num_rows, true)?;
            #[allow(clippy::cast_possible_truncation)]
            let values: Vec<i32> = ints.into_iter().map(|v| v as i32).collect();
            Ok(ColumnData::Int(values))
        }
        TypeKind::Long => {
            let data_bytes = get_stream_bytes(
                stripe_data,
                stream_map,
                col_id,
                StreamKind::Data,
                compression,
            )?
            .unwrap_or_default();
            let longs = encoding::decode_int_rle_v1(&data_bytes, num_rows, true)?;
            Ok(ColumnData::Long(longs))
        }
        TypeKind::Float => {
            let data_bytes = get_stream_bytes(
                stripe_data,
                stream_map,
                col_id,
                StreamKind::Data,
                compression,
            )?
            .unwrap_or_default();
            let decompressed = &data_bytes;
            let mut values = Vec::with_capacity(num_rows);
            for i in 0..num_rows {
                let offset = i * 4;
                if offset + 4 > decompressed.len() {
                    return Err(IoError::FormatError("float data truncated".into()));
                }
                let bytes: [u8; 4] = [
                    decompressed[offset],
                    decompressed[offset + 1],
                    decompressed[offset + 2],
                    decompressed[offset + 3],
                ];
                values.push(f32::from_le_bytes(bytes));
            }
            Ok(ColumnData::Float(values))
        }
        TypeKind::Double => {
            let data_bytes = get_stream_bytes(
                stripe_data,
                stream_map,
                col_id,
                StreamKind::Data,
                compression,
            )?
            .unwrap_or_default();
            let decompressed = &data_bytes;
            let mut values = Vec::with_capacity(num_rows);
            for i in 0..num_rows {
                let offset = i * 8;
                if offset + 8 > decompressed.len() {
                    return Err(IoError::FormatError("double data truncated".into()));
                }
                let bytes: [u8; 8] = [
                    decompressed[offset],
                    decompressed[offset + 1],
                    decompressed[offset + 2],
                    decompressed[offset + 3],
                    decompressed[offset + 4],
                    decompressed[offset + 5],
                    decompressed[offset + 6],
                    decompressed[offset + 7],
                ];
                values.push(f64::from_le_bytes(bytes));
            }
            Ok(ColumnData::Double(values))
        }
        TypeKind::String | TypeKind::Varchar | TypeKind::Char => decode_string_column(
            stripe_data,
            stream_map,
            col_id,
            col_encoding,
            num_rows,
            compression,
        ),
        other => Err(IoError::FormatError(format!(
            "unsupported ORC type kind: {other:?}"
        ))),
    }
}

/// Decode a string column, handling both direct and dictionary encoding.
fn decode_string_column(
    stripe_data: &[u8],
    stream_map: &std::collections::HashMap<(u32, StreamKind), (usize, usize)>,
    col_id: u32,
    col_encoding: &ColumnEncoding,
    num_rows: usize,
    compression: CompressionKind,
) -> Result<ColumnData> {
    match col_encoding.kind {
        ColumnEncodingKind::Direct | ColumnEncodingKind::DirectV2 => {
            // Direct encoding: DATA stream has raw string bytes,
            // LENGTH stream has the lengths of each string.
            let data_bytes = get_stream_bytes(
                stripe_data,
                stream_map,
                col_id,
                StreamKind::Data,
                compression,
            )?
            .unwrap_or_default();
            let length_bytes = get_stream_bytes(
                stripe_data,
                stream_map,
                col_id,
                StreamKind::Length,
                compression,
            )?
            .unwrap_or_default();

            let lengths = encoding::decode_int_rle_v1(&length_bytes, num_rows, false)?;
            let strings = encoding::decode_string_dictionary(&data_bytes, &lengths)?;
            Ok(ColumnData::Str(strings))
        }
        ColumnEncodingKind::Dictionary | ColumnEncodingKind::DictionaryV2 => {
            // Dictionary encoding:
            // DICTIONARY_DATA has the dictionary strings concatenated.
            // LENGTH stream has the lengths of each dictionary entry.
            // DATA stream has the indices into the dictionary.
            let dict_data_bytes = get_stream_bytes(
                stripe_data,
                stream_map,
                col_id,
                StreamKind::DictionaryData,
                compression,
            )?
            .unwrap_or_default();
            let length_bytes = get_stream_bytes(
                stripe_data,
                stream_map,
                col_id,
                StreamKind::Length,
                compression,
            )?
            .unwrap_or_default();
            let data_bytes = get_stream_bytes(
                stripe_data,
                stream_map,
                col_id,
                StreamKind::Data,
                compression,
            )?
            .unwrap_or_default();

            let dict_size = col_encoding.dictionary_size as usize;
            let dict_lengths = encoding::decode_int_rle_v1(&length_bytes, dict_size, false)?;
            let dictionary = encoding::decode_string_dictionary(&dict_data_bytes, &dict_lengths)?;

            let indices = encoding::decode_int_rle_v1(&data_bytes, num_rows, false)?;
            let mut strings = Vec::with_capacity(num_rows);
            for &idx in &indices {
                #[allow(clippy::cast_sign_loss)]
                let idx = idx as usize;
                if idx >= dictionary.len() {
                    return Err(IoError::FormatError(format!(
                        "dictionary index {idx} out of range (dict size {})",
                        dictionary.len()
                    )));
                }
                strings.push(dictionary[idx].clone());
            }
            Ok(ColumnData::Str(strings))
        }
    }
}

// ---------------------------------------------------------------------------
// Stripe writing
// ---------------------------------------------------------------------------

/// Encoded data for a single column within a stripe.
pub(crate) struct EncodedColumn {
    /// The streams produced for this column.
    pub streams: Vec<(StreamKind, Vec<u8>)>,
    /// The column encoding used.
    pub encoding: ColumnEncoding,
}

/// Encode a DataFrame column (given as a trait object) for a stripe.
pub(crate) fn encode_column(
    col: &dyn AnySeries,
    col_type: &OrcType,
    compression: CompressionKind,
) -> Result<EncodedColumn> {
    match col_type.kind {
        TypeKind::Boolean => encode_boolean_column(col, compression),
        TypeKind::Int => encode_int_column(col, compression),
        TypeKind::Long => encode_long_column(col, compression),
        TypeKind::Float => encode_float_column(col, compression),
        TypeKind::Double => encode_double_column(col, compression),
        TypeKind::String => encode_string_column(col, compression),
        other => Err(IoError::FormatError(format!(
            "cannot encode ORC type: {other:?}"
        ))),
    }
}

fn compress_stream(data: &[u8], compression: CompressionKind) -> Result<Vec<u8>> {
    match compression {
        CompressionKind::None => Ok(data.to_vec()),
        CompressionKind::Zlib => compress_orc_zlib(data),
        other => Err(IoError::FormatError(format!(
            "unsupported compression for writing: {other:?}"
        ))),
    }
}

/// Compress data using ORC's zlib block format.
fn compress_orc_zlib(data: &[u8]) -> Result<Vec<u8>> {
    use std::io::Write as _;

    // Try to compress; if it doesn't shrink, store uncompressed.
    let mut compressed = Vec::new();
    {
        let mut encoder =
            flate2::write::DeflateEncoder::new(&mut compressed, flate2::Compression::default());
        encoder
            .write_all(data)
            .map_err(|e| IoError::FormatError(format!("zlib compression failed: {e}")))?;
        encoder
            .finish()
            .map_err(|e| IoError::FormatError(format!("zlib compression finish failed: {e}")))?;
    }

    let mut result = Vec::new();
    if compressed.len() < data.len() {
        // Write compressed block
        let block_len = compressed.len() as u32;
        // is_original = 0 (compressed), block_len in remaining 23 bits
        let header = block_len << 1; // bit 0 = 0 means compressed
        result.push((header & 0xFF) as u8);
        result.push(((header >> 8) & 0xFF) as u8);
        result.push(((header >> 16) & 0xFF) as u8);
        result.extend_from_slice(&compressed);
    } else {
        // Store uncompressed
        let block_len = data.len() as u32;
        let header = (block_len << 1) | 1; // bit 0 = 1 means original
        result.push((header & 0xFF) as u8);
        result.push(((header >> 8) & 0xFF) as u8);
        result.push(((header >> 16) & 0xFF) as u8);
        result.extend_from_slice(data);
    }

    Ok(result)
}

fn encode_boolean_column(
    col: &dyn AnySeries,
    compression: CompressionKind,
) -> Result<EncodedColumn> {
    let nrows = col.len();
    let mut bools = Vec::with_capacity(nrows);

    // Boolean columns stored as Series<u8> (0=false, 1=true)
    if let Some(typed) = col.as_any().downcast_ref::<Series<u8>>() {
        for &v in typed.as_slice() {
            bools.push(v != 0);
        }
    } else {
        return Err(IoError::FormatError(format!(
            "cannot encode column '{}' (dtype {:?}) as boolean",
            col.name(),
            col.dtype()
        )));
    }

    let encoded = encoding::encode_booleans(&bools);
    let compressed = compress_stream(&encoded, compression)?;

    Ok(EncodedColumn {
        streams: vec![(StreamKind::Data, compressed)],
        encoding: ColumnEncoding {
            kind: ColumnEncodingKind::Direct,
            dictionary_size: 0,
        },
    })
}

fn encode_int_column(col: &dyn AnySeries, compression: CompressionKind) -> Result<EncodedColumn> {
    let nrows = col.len();
    let mut values = Vec::with_capacity(nrows);

    if let Some(typed) = col.as_any().downcast_ref::<Series<u8>>() {
        for &v in typed.as_slice() {
            values.push(i64::from(v));
        }
    } else if let Some(typed) = col.as_any().downcast_ref::<Series<i32>>() {
        for &v in typed.as_slice() {
            values.push(i64::from(v));
        }
    } else if let Some(typed) = col.as_any().downcast_ref::<Series<i64>>() {
        for &v in typed.as_slice() {
            values.push(v);
        }
    } else {
        return Err(IoError::FormatError(format!(
            "cannot encode column '{}' (dtype {:?}) as int",
            col.name(),
            col.dtype()
        )));
    }

    let encoded = encoding::encode_int_rle_v1(&values, true);
    let compressed = compress_stream(&encoded, compression)?;

    Ok(EncodedColumn {
        streams: vec![(StreamKind::Data, compressed)],
        encoding: ColumnEncoding {
            kind: ColumnEncodingKind::Direct,
            dictionary_size: 0,
        },
    })
}

fn encode_long_column(col: &dyn AnySeries, compression: CompressionKind) -> Result<EncodedColumn> {
    let nrows = col.len();
    let mut values = Vec::with_capacity(nrows);

    if let Some(typed) = col.as_any().downcast_ref::<Series<i64>>() {
        for &v in typed.as_slice() {
            values.push(v);
        }
    } else if let Some(typed) = col.as_any().downcast_ref::<Series<i32>>() {
        for &v in typed.as_slice() {
            values.push(i64::from(v));
        }
    } else {
        return Err(IoError::FormatError(format!(
            "cannot encode column '{}' (dtype {:?}) as long",
            col.name(),
            col.dtype()
        )));
    }

    let encoded = encoding::encode_int_rle_v1(&values, true);
    let compressed = compress_stream(&encoded, compression)?;

    Ok(EncodedColumn {
        streams: vec![(StreamKind::Data, compressed)],
        encoding: ColumnEncoding {
            kind: ColumnEncodingKind::Direct,
            dictionary_size: 0,
        },
    })
}

fn encode_float_column(col: &dyn AnySeries, compression: CompressionKind) -> Result<EncodedColumn> {
    let nrows = col.len();
    let mut data_bytes = Vec::with_capacity(nrows * 4);

    if let Some(typed) = col.as_any().downcast_ref::<Series<f32>>() {
        for &v in typed.as_slice() {
            data_bytes.extend_from_slice(&v.to_le_bytes());
        }
    } else {
        return Err(IoError::FormatError(format!(
            "cannot encode column '{}' (dtype {:?}) as float",
            col.name(),
            col.dtype()
        )));
    }

    let compressed = compress_stream(&data_bytes, compression)?;

    Ok(EncodedColumn {
        streams: vec![(StreamKind::Data, compressed)],
        encoding: ColumnEncoding {
            kind: ColumnEncodingKind::Direct,
            dictionary_size: 0,
        },
    })
}

fn encode_double_column(
    col: &dyn AnySeries,
    compression: CompressionKind,
) -> Result<EncodedColumn> {
    let nrows = col.len();
    let mut data_bytes = Vec::with_capacity(nrows * 8);

    if let Some(typed) = col.as_any().downcast_ref::<Series<f64>>() {
        for &v in typed.as_slice() {
            data_bytes.extend_from_slice(&v.to_le_bytes());
        }
    } else {
        return Err(IoError::FormatError(format!(
            "cannot encode column '{}' (dtype {:?}) as double",
            col.name(),
            col.dtype()
        )));
    }

    let compressed = compress_stream(&data_bytes, compression)?;

    Ok(EncodedColumn {
        streams: vec![(StreamKind::Data, compressed)],
        encoding: ColumnEncoding {
            kind: ColumnEncodingKind::Direct,
            dictionary_size: 0,
        },
    })
}

fn encode_string_column(
    col: &dyn AnySeries,
    compression: CompressionKind,
) -> Result<EncodedColumn> {
    let nrows = col.len();

    if let Some(typed) = col.as_any().downcast_ref::<StringSeries>() {
        // Use direct encoding: DATA = concatenated string bytes, LENGTH = per-string lengths.
        let mut data_buf = Vec::new();
        let mut lengths = Vec::with_capacity(nrows);

        for i in 0..nrows {
            let s = typed.get(i).unwrap_or("");
            data_buf.extend_from_slice(s.as_bytes());
            #[allow(clippy::cast_possible_wrap)]
            lengths.push(s.len() as i64);
        }

        let data_encoded = compress_stream(&data_buf, compression)?;
        let length_rle = encoding::encode_int_rle_v1(&lengths, false);
        let length_encoded = compress_stream(&length_rle, compression)?;

        Ok(EncodedColumn {
            streams: vec![
                (StreamKind::Data, data_encoded),
                (StreamKind::Length, length_encoded),
            ],
            encoding: ColumnEncoding {
                kind: ColumnEncodingKind::Direct,
                dictionary_size: 0,
            },
        })
    } else {
        Err(IoError::FormatError(format!(
            "cannot encode column '{}' (dtype {:?}) as string",
            col.name(),
            col.dtype()
        )))
    }
}

/// Map a DataFrame DType to an ORC TypeKind.
pub(crate) fn dtype_to_orc_type(dtype: DType) -> TypeKind {
    match dtype {
        DType::Bool => TypeKind::Boolean,
        DType::I8 | DType::I16 | DType::I32 | DType::U8 | DType::U16 => TypeKind::Int,
        DType::I64 | DType::U32 | DType::U64 => TypeKind::Long,
        DType::F32 => TypeKind::Float,
        DType::F64 => TypeKind::Double,
        DType::Str | DType::Categorical | DType::DateTime => TypeKind::String,
    }
}
