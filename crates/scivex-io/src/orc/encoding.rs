//! ORC run-length encoding and decoding.
//!
//! This module implements:
//! - Byte RLE (used for boolean/presence streams)
//! - Integer RLE v1 (used for integer columns)
//! - String dictionary helpers

use crate::error::{IoError, Result};

// ---------------------------------------------------------------------------
// Byte RLE — used for boolean (bit-packed) and presence streams
// ---------------------------------------------------------------------------

/// Decode a byte RLE stream.
///
/// The encoding is: read a signed byte `header`.
/// - If header >= 0: run of (header + 3) copies of the next byte
/// - If header < 0: literal run of (-header) distinct bytes
pub fn decode_byte_rle(data: &[u8], num_values: usize) -> Result<Vec<u8>> {
    let mut result = Vec::with_capacity(num_values);
    let mut pos = 0;

    while result.len() < num_values && pos < data.len() {
        #[allow(clippy::cast_possible_wrap)]
        let header = data[pos] as i8;
        pos += 1;

        if header >= 0 {
            // Run: repeat the next byte (header + 3) times
            let run_len = (header as usize) + 3;
            if pos >= data.len() {
                return Err(IoError::FormatError(
                    "byte RLE: unexpected end of run data".into(),
                ));
            }
            let val = data[pos];
            pos += 1;
            let count = run_len.min(num_values - result.len());
            result.extend(std::iter::repeat_n(val, count));
        } else {
            // Literal: read (-header) bytes
            let lit_len = (-header) as usize;
            if pos + lit_len > data.len() {
                return Err(IoError::FormatError(
                    "byte RLE: unexpected end of literal data".into(),
                ));
            }
            let count = lit_len.min(num_values - result.len());
            result.extend_from_slice(&data[pos..pos + count]);
            pos += lit_len;
        }
    }

    Ok(result)
}

/// Encode a byte stream using byte RLE.
pub fn encode_byte_rle(data: &[u8]) -> Vec<u8> {
    let mut result = Vec::new();
    let mut i = 0;

    while i < data.len() {
        // Try to find a run
        let val = data[i];
        let mut run_len = 1;
        while i + run_len < data.len() && data[i + run_len] == val && run_len < 130 {
            run_len += 1;
        }

        if run_len >= 3 {
            // Encode as run: header = run_len - 3
            #[allow(clippy::cast_possible_truncation)]
            result.push((run_len - 3) as u8);
            result.push(val);
            i += run_len;
        } else {
            // Encode as literal run — gather up to 128 literal bytes
            let start = i;
            let mut lit_len = 0;
            while i + lit_len < data.len() && lit_len < 128 {
                // Check if the next position starts a run of >= 3
                let next_val = data[i + lit_len];
                let mut ahead = 1;
                while i + lit_len + ahead < data.len()
                    && data[i + lit_len + ahead] == next_val
                    && ahead < 3
                {
                    ahead += 1;
                }
                if ahead >= 3 && lit_len > 0 {
                    break;
                }
                lit_len += 1;
            }
            // header = -(lit_len) as i8
            #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
            let header = -(lit_len as i8);
            result.push(header as u8);
            result.extend_from_slice(&data[start..start + lit_len]);
            i = start + lit_len;
        }
    }

    result
}

// ---------------------------------------------------------------------------
// Boolean streams — bit-packed inside byte RLE
// ---------------------------------------------------------------------------

/// Decode a boolean stream: first byte-RLE decode, then unpack bits.
/// Booleans are packed MSB-first: bit 7 of first byte is first value.
pub fn decode_booleans(data: &[u8], num_values: usize) -> Result<Vec<bool>> {
    let num_bytes = num_values.div_ceil(8);
    let decoded_bytes = decode_byte_rle(data, num_bytes)?;
    let mut result = Vec::with_capacity(num_values);

    for (byte_idx, &byte_val) in decoded_bytes.iter().enumerate() {
        for bit in (0..8).rev() {
            if byte_idx * 8 + (7 - bit) >= num_values {
                break;
            }
            result.push((byte_val >> bit) & 1 != 0);
        }
    }

    result.truncate(num_values);
    Ok(result)
}

/// Encode booleans to a byte-RLE encoded stream.
/// Booleans are packed MSB-first.
pub fn encode_booleans(values: &[bool]) -> Vec<u8> {
    let num_bytes = values.len().div_ceil(8);
    let mut bytes = vec![0u8; num_bytes];

    for (i, &val) in values.iter().enumerate() {
        if val {
            let byte_idx = i / 8;
            let bit_idx = 7 - (i % 8); // MSB-first
            bytes[byte_idx] |= 1 << bit_idx;
        }
    }

    encode_byte_rle(&bytes)
}

// ---------------------------------------------------------------------------
// Integer RLE v1 — used for integer columns
// ---------------------------------------------------------------------------

/// Decode an integer RLE v1 stream.
///
/// The encoding:
/// - Read a signed byte `header`.
/// - If header >= 0: run of (header + 3) integers.
///   Read a signed byte `delta` and a base varint. Values are
///   base, base+delta, base+2*delta, ...
/// - If header < 0: literal run of (-header) varints.
///
/// The varints are unsigned base-128, but represent signed values via the
/// caller's interpretation.
pub fn decode_int_rle_v1(data: &[u8], num_values: usize, signed: bool) -> Result<Vec<i64>> {
    let mut result = Vec::with_capacity(num_values);
    let mut pos = 0;

    while result.len() < num_values && pos < data.len() {
        #[allow(clippy::cast_possible_wrap)]
        let header = data[pos] as i8;
        pos += 1;

        if header >= 0 {
            // Run encoding
            let run_len = (header as usize) + 3;
            if pos >= data.len() {
                return Err(IoError::FormatError(
                    "int RLE v1: unexpected end of run data".into(),
                ));
            }
            #[allow(clippy::cast_possible_wrap)]
            let delta = data[pos] as i8;
            pos += 1;

            // Read base varint
            let (base_raw, n) = decode_varint_from_slice(&data[pos..])?;
            pos += n;

            let base = if signed {
                super::proto::zigzag_decode(base_raw)
            } else {
                #[allow(clippy::cast_possible_wrap)]
                let v = base_raw as i64;
                v
            };

            let count = run_len.min(num_values - result.len());
            for j in 0..count {
                #[allow(clippy::cast_possible_wrap)]
                let val = base + (j as i64) * i64::from(delta);
                result.push(val);
            }
        } else {
            // Literal encoding
            let lit_len = (-header) as usize;
            let count = lit_len.min(num_values - result.len());
            for _ in 0..count {
                let (raw, n) = decode_varint_from_slice(&data[pos..])?;
                pos += n;
                let val = if signed {
                    super::proto::zigzag_decode(raw)
                } else {
                    #[allow(clippy::cast_possible_wrap)]
                    let v = raw as i64;
                    v
                };
                result.push(val);
            }
        }
    }

    Ok(result)
}

/// Encode integers using RLE v1.
pub fn encode_int_rle_v1(values: &[i64], signed: bool) -> Vec<u8> {
    let mut result = Vec::new();
    let mut i = 0;

    while i < values.len() {
        // Try to detect a run (same delta between consecutive values)
        if i + 2 < values.len() {
            let delta = values[i + 1].wrapping_sub(values[i]);
            let mut run_len = 2;
            while i + run_len < values.len()
                && run_len < 130
                && values[i + run_len].wrapping_sub(values[i + run_len - 1]) == delta
            {
                run_len += 1;
            }

            if run_len >= 3 {
                // Encode as run
                #[allow(clippy::cast_possible_truncation)]
                let header = (run_len - 3) as u8;
                result.push(header);
                // Delta as signed byte
                #[allow(clippy::cast_possible_truncation)]
                let delta_byte = delta as i8;
                result.push(delta_byte as u8);
                // Base value as varint
                if signed {
                    encode_varint_to_vec(&mut result, zigzag_encode_i64(values[i]));
                } else {
                    #[allow(clippy::cast_sign_loss)]
                    encode_varint_to_vec(&mut result, values[i] as u64);
                }
                i += run_len;
                continue;
            }
        }

        // Literal run — collect up to 128 values
        let start = i;
        let mut lit_len = 0;
        while i + lit_len < values.len() && lit_len < 128 {
            // Check if starting a run of 3+ at this position
            if i + lit_len + 2 < values.len() && lit_len > 0 {
                let d = values[i + lit_len + 1].wrapping_sub(values[i + lit_len]);
                if values[i + lit_len + 2].wrapping_sub(values[i + lit_len + 1]) == d {
                    break;
                }
            }
            lit_len += 1;
        }
        if lit_len == 0 {
            lit_len = 1;
        }

        // header = -(lit_len) as i8
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let header = -(lit_len as i8);
        result.push(header as u8);
        for j in 0..lit_len {
            if signed {
                encode_varint_to_vec(&mut result, zigzag_encode_i64(values[start + j]));
            } else {
                #[allow(clippy::cast_sign_loss)]
                encode_varint_to_vec(&mut result, values[start + j] as u64);
            }
        }
        i = start + lit_len;
    }

    result
}

// ---------------------------------------------------------------------------
// Varint helpers
// ---------------------------------------------------------------------------

/// Decode a varint from a slice.
fn decode_varint_from_slice(data: &[u8]) -> Result<(u64, usize)> {
    super::proto::decode_varint(data)
}

/// Zigzag encode an i64 to u64.
fn zigzag_encode_i64(n: i64) -> u64 {
    ((n << 1) ^ (n >> 63)) as u64
}

/// Encode a varint and append to vec.
fn encode_varint_to_vec(buf: &mut Vec<u8>, mut val: u64) {
    loop {
        if val < 0x80 {
            buf.push(val as u8);
            break;
        }
        buf.push((val as u8 & 0x7F) | 0x80);
        val >>= 7;
    }
}

// ---------------------------------------------------------------------------
// String dictionary helpers
// ---------------------------------------------------------------------------

/// Decode string dictionary data: a contiguous buffer of UTF-8 bytes split
/// by the given lengths.
pub fn decode_string_dictionary(data: &[u8], lengths: &[i64]) -> Result<Vec<String>> {
    let mut dict = Vec::with_capacity(lengths.len());
    let mut pos = 0;
    for &len in lengths {
        #[allow(clippy::cast_sign_loss)]
        let len = len as usize;
        if pos + len > data.len() {
            return Err(IoError::FormatError(
                "string dictionary: unexpected end of data".into(),
            ));
        }
        let s = String::from_utf8_lossy(&data[pos..pos + len]).into_owned();
        pos += len;
        dict.push(s);
    }
    Ok(dict)
}
