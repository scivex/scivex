//! Minimal protobuf wire-format parser.
//!
//! Only the subset of protobuf features used by the ONNX spec is implemented:
//! varint, length-delimited, fixed32, and fixed64 wire types.

use crate::error::{NnError, Result};

// -----------------------------------------------------------------------
// Wire types
// -----------------------------------------------------------------------

/// Protobuf wire types we need.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WireType {
    Varint,
    Fixed64,
    LengthDelimited,
    Fixed32,
}

impl WireType {
    fn from_u8(v: u8) -> Result<Self> {
        match v {
            0 => Ok(Self::Varint),
            1 => Ok(Self::Fixed64),
            2 => Ok(Self::LengthDelimited),
            5 => Ok(Self::Fixed32),
            other => Err(NnError::OnnxError(format!(
                "unsupported protobuf wire type: {other}"
            ))),
        }
    }
}

// -----------------------------------------------------------------------
// A parsed protobuf field
// -----------------------------------------------------------------------

/// A single protobuf field parsed from the wire format.
#[derive(Debug, Clone)]
pub(crate) enum FieldValue<'a> {
    Varint(u64),
    Fixed64(u64),
    Fixed32(u32),
    Bytes(&'a [u8]),
}

/// A field tag + value pair.
#[derive(Debug, Clone)]
pub(crate) struct Field<'a> {
    pub field_number: u32,
    pub value: FieldValue<'a>,
}

// -----------------------------------------------------------------------
// Decoding primitives
// -----------------------------------------------------------------------

/// Decode a varint from `buf` starting at `pos`.
/// Returns `(value, new_pos)`.
pub(crate) fn decode_varint(buf: &[u8], pos: usize) -> Result<(u64, usize)> {
    let mut result: u64 = 0;
    let mut shift: u32 = 0;
    let mut i = pos;
    loop {
        if i >= buf.len() {
            return Err(NnError::OnnxError(
                "unexpected end of buffer while decoding varint".into(),
            ));
        }
        let byte = buf[i];
        result |= u64::from(byte & 0x7F) << shift;
        i += 1;
        if byte & 0x80 == 0 {
            return Ok((result, i));
        }
        shift += 7;
        if shift >= 64 {
            return Err(NnError::OnnxError("varint too long".into()));
        }
    }
}

/// Encode a varint into a `Vec<u8>`.
#[cfg(test)]
pub(crate) fn encode_varint(mut value: u64) -> Vec<u8> {
    let mut out = Vec::new();
    loop {
        let mut byte = (value & 0x7F) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        out.push(byte);
        if value == 0 {
            break;
        }
    }
    out
}

/// Read a fixed-width 32-bit little-endian value.
fn read_fixed32(buf: &[u8], pos: usize) -> Result<(u32, usize)> {
    if pos + 4 > buf.len() {
        return Err(NnError::OnnxError(
            "unexpected end of buffer reading fixed32".into(),
        ));
    }
    let val = u32::from_le_bytes([buf[pos], buf[pos + 1], buf[pos + 2], buf[pos + 3]]);
    Ok((val, pos + 4))
}

/// Read a fixed-width 64-bit little-endian value.
fn read_fixed64(buf: &[u8], pos: usize) -> Result<(u64, usize)> {
    if pos + 8 > buf.len() {
        return Err(NnError::OnnxError(
            "unexpected end of buffer reading fixed64".into(),
        ));
    }
    let val = u64::from_le_bytes([
        buf[pos],
        buf[pos + 1],
        buf[pos + 2],
        buf[pos + 3],
        buf[pos + 4],
        buf[pos + 5],
        buf[pos + 6],
        buf[pos + 7],
    ]);
    Ok((val, pos + 8))
}

// -----------------------------------------------------------------------
// Field-level parser
// -----------------------------------------------------------------------

/// Parse all top-level fields from a protobuf message.
pub(crate) fn parse_fields(buf: &[u8]) -> Result<Vec<Field<'_>>> {
    let mut fields = Vec::new();
    let mut pos = 0;
    while pos < buf.len() {
        let (tag, new_pos) = decode_varint(buf, pos)?;
        pos = new_pos;

        #[allow(clippy::cast_possible_truncation)]
        let field_number = (tag >> 3) as u32;
        #[allow(clippy::cast_possible_truncation)]
        let wire_type = WireType::from_u8((tag & 0x07) as u8)?;

        let value;
        match wire_type {
            WireType::Varint => {
                let (v, np) = decode_varint(buf, pos)?;
                pos = np;
                value = FieldValue::Varint(v);
            }
            WireType::Fixed64 => {
                let (v, np) = read_fixed64(buf, pos)?;
                pos = np;
                value = FieldValue::Fixed64(v);
            }
            WireType::Fixed32 => {
                let (v, np) = read_fixed32(buf, pos)?;
                pos = np;
                value = FieldValue::Fixed32(v);
            }
            WireType::LengthDelimited => {
                let (len, np) = decode_varint(buf, pos)?;
                pos = np;
                #[allow(clippy::cast_possible_truncation)]
                let len = len as usize;
                if pos + len > buf.len() {
                    return Err(NnError::OnnxError(
                        "length-delimited field extends past buffer".into(),
                    ));
                }
                value = FieldValue::Bytes(&buf[pos..pos + len]);
                pos += len;
            }
        }

        fields.push(Field {
            field_number,
            value,
        });
    }
    Ok(fields)
}

/// Extract the varint value for a given field number (first occurrence).
pub(crate) fn get_varint(fields: &[Field<'_>], number: u32) -> Option<u64> {
    for f in fields {
        #[allow(clippy::collapsible_if)]
        if f.field_number == number {
            if let FieldValue::Varint(v) = f.value {
                return Some(v);
            }
        }
    }
    None
}

/// Extract the bytes for a given field number (first occurrence).
pub(crate) fn get_bytes<'a>(fields: &[Field<'a>], number: u32) -> Option<&'a [u8]> {
    for f in fields {
        #[allow(clippy::collapsible_if)]
        if f.field_number == number {
            if let FieldValue::Bytes(b) = f.value {
                return Some(b);
            }
        }
    }
    None
}

/// Collect all bytes fields for a given field number (repeated).
pub(crate) fn get_all_bytes<'a>(fields: &[Field<'a>], number: u32) -> Vec<&'a [u8]> {
    let mut result = Vec::new();
    for f in fields {
        #[allow(clippy::collapsible_if)]
        if f.field_number == number {
            if let FieldValue::Bytes(b) = f.value {
                result.push(b);
            }
        }
    }
    result
}

/// Decode a packed repeated field of varints.
pub(crate) fn decode_packed_varints(buf: &[u8]) -> Result<Vec<u64>> {
    let mut values = Vec::new();
    let mut pos = 0;
    while pos < buf.len() {
        let (v, np) = decode_varint(buf, pos)?;
        pos = np;
        values.push(v);
    }
    Ok(values)
}

/// Decode a packed repeated field of fixed32 values.
#[allow(dead_code)]
pub(crate) fn decode_packed_fixed32(buf: &[u8]) -> Result<Vec<u32>> {
    #[allow(clippy::manual_is_multiple_of)]
    if buf.len() % 4 != 0 {
        return Err(NnError::OnnxError(
            "packed fixed32 field length not a multiple of 4".into(),
        ));
    }
    let mut values = Vec::with_capacity(buf.len() / 4);
    let mut pos = 0;
    while pos < buf.len() {
        let (v, np) = read_fixed32(buf, pos)?;
        pos = np;
        values.push(v);
    }
    Ok(values)
}

/// Decode a packed repeated field of fixed64 values.
#[allow(dead_code)]
pub(crate) fn decode_packed_fixed64(buf: &[u8]) -> Result<Vec<u64>> {
    #[allow(clippy::manual_is_multiple_of)]
    if buf.len() % 8 != 0 {
        return Err(NnError::OnnxError(
            "packed fixed64 field length not a multiple of 8".into(),
        ));
    }
    let mut values = Vec::with_capacity(buf.len() / 8);
    let mut pos = 0;
    while pos < buf.len() {
        let (v, np) = read_fixed64(buf, pos)?;
        pos = np;
        values.push(v);
    }
    Ok(values)
}

/// Get a string from a bytes field.
pub(crate) fn get_string(fields: &[Field<'_>], number: u32) -> Option<String> {
    get_bytes(fields, number).map(|b| String::from_utf8_lossy(b).into_owned())
}

/// Collect all string fields for a given field number (repeated).
pub(crate) fn get_all_strings(fields: &[Field<'_>], number: u32) -> Vec<String> {
    get_all_bytes(fields, number)
        .into_iter()
        .map(|b| String::from_utf8_lossy(b).into_owned())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_varint_roundtrip() {
        for &val in &[0u64, 1, 127, 128, 300, 16384, u64::MAX >> 1] {
            let encoded = encode_varint(val);
            let (decoded, end) = decode_varint(&encoded, 0).unwrap();
            assert_eq!(decoded, val);
            assert_eq!(end, encoded.len());
        }
    }

    #[test]
    fn test_parse_length_delimited() {
        // Build a single length-delimited field: field_number=2, wire_type=2
        // tag = (2 << 3) | 2 = 18
        let mut buf = Vec::new();
        buf.extend_from_slice(&encode_varint(18)); // tag
        let payload = b"hello";
        buf.extend_from_slice(&encode_varint(payload.len() as u64));
        buf.extend_from_slice(payload);

        let fields = parse_fields(&buf).unwrap();
        assert_eq!(fields.len(), 1);
        assert_eq!(fields[0].field_number, 2);
        if let FieldValue::Bytes(b) = fields[0].value {
            assert_eq!(b, b"hello");
        } else {
            panic!("expected bytes field");
        }
    }
}
