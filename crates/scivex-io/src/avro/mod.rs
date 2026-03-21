//! Apache Avro container file reading and writing.
//!
//! This module provides a from-scratch implementation of the Avro Object
//! Container File format (uncompressed / "null" codec only). It supports
//! reading and writing [`DataFrame`](scivex_frame::DataFrame) values with
//! primitive Avro types: null, boolean, int, long, float, double, string,
//! and bytes.

use std::io::{Read, Write};

use scivex_frame::{AnySeries, DType, DataFrame, Series, StringSeries};

use crate::error::{IoError, Result};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Avro magic bytes: `O`, `b`, `j`, 0x01.
const AVRO_MAGIC: [u8; 4] = [b'O', b'b', b'j', 0x01];

/// Length of the sync marker in bytes.
const SYNC_MARKER_LEN: usize = 16;

// ---------------------------------------------------------------------------
// Avro schema types (simplified)
// ---------------------------------------------------------------------------

/// Subset of Avro types we support.
///
/// # Examples
///
/// ```
/// use scivex_io::avro::AvroType;
/// let t = AvroType::Long;
/// assert_eq!(t, AvroType::Long);
/// assert_ne!(t, AvroType::Double);
///
/// let nullable = AvroType::Union(Box::new(AvroType::String));
/// assert!(matches!(nullable, AvroType::Union(_)));
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AvroType {
    Null,
    Boolean,
    Int,
    Long,
    Float,
    Double,
    String,
    Bytes,
    /// A union of exactly [null, T] — represents a nullable field.
    Union(Box<AvroType>),
}

/// A single field in an Avro record schema.
///
/// # Examples
///
/// ```
/// use scivex_io::avro::{AvroField, AvroType};
/// let field = AvroField {
///     name: "age".to_string(),
///     avro_type: AvroType::Long,
/// };
/// assert_eq!(field.name, "age");
/// assert_eq!(field.avro_type, AvroType::Long);
/// ```
#[derive(Debug, Clone)]
pub struct AvroField {
    /// Field name.
    pub name: std::string::String,
    /// Field type.
    pub avro_type: AvroType,
}

/// Parsed Avro record schema (top-level schema must be a record).
///
/// # Examples
///
/// ```
/// use scivex_io::avro::{AvroSchema, AvroField, AvroType};
/// let schema = AvroSchema {
///     name: "MyRecord".to_string(),
///     fields: vec![
///         AvroField { name: "id".to_string(), avro_type: AvroType::Long },
///         AvroField { name: "score".to_string(), avro_type: AvroType::Double },
///     ],
/// };
/// assert_eq!(schema.name, "MyRecord");
/// assert_eq!(schema.fields.len(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct AvroSchema {
    /// Record name.
    pub name: std::string::String,
    /// Record fields.
    pub fields: Vec<AvroField>,
}

// ---------------------------------------------------------------------------
// Zigzag varint encoding / decoding
// ---------------------------------------------------------------------------

/// Encode a signed 64-bit integer using zigzag encoding into varint bytes.
///
/// # Examples
///
/// ```
/// use scivex_io::avro::zigzag_encode_i64;
/// let encoded = zigzag_encode_i64(0_i64);
/// assert_eq!(encoded, vec![0]);
/// let encoded = zigzag_encode_i64(1_i64);
/// assert_eq!(encoded, vec![2]);
/// let encoded = zigzag_encode_i64(-1_i64);
/// assert_eq!(encoded, vec![1]);
/// ```
pub fn zigzag_encode_i64(n: i64) -> Vec<u8> {
    let mut z = ((n << 1) ^ (n >> 63)) as u64;
    let mut buf = Vec::with_capacity(10);
    loop {
        if z & !0x7F == 0 {
            buf.push(z as u8);
            break;
        }
        buf.push((z as u8 & 0x7F) | 0x80);
        z >>= 7;
    }
    buf
}

/// Decode a zigzag-encoded varint from a reader, returning the signed i64.
///
/// # Examples
///
/// ```
/// use std::io::Cursor;
/// use scivex_io::avro::{zigzag_encode_i64, zigzag_decode_i64};
/// let encoded = zigzag_encode_i64(42_i64);
/// let mut cursor = Cursor::new(encoded);
/// let decoded = zigzag_decode_i64(&mut cursor).unwrap();
/// assert_eq!(decoded, 42_i64);
/// ```
pub fn zigzag_decode_i64<R: Read>(reader: &mut R) -> Result<i64> {
    let mut result: u64 = 0;
    let mut shift: u32 = 0;
    let mut buf = [0u8; 1];
    loop {
        reader
            .read_exact(&mut buf)
            .map_err(|e| IoError::FormatError(format!("failed to read varint byte: {e}")))?;
        let b = buf[0];
        result |= u64::from(b & 0x7F) << shift;
        if b & 0x80 == 0 {
            break;
        }
        shift += 7;
        if shift >= 64 {
            return Err(IoError::FormatError("varint too long".into()));
        }
    }
    // Zigzag decode: (n >>> 1) ^ -(n & 1)
    #[allow(clippy::cast_possible_wrap)]
    let decoded = ((result >> 1) as i64) ^ (-((result & 1) as i64));
    Ok(decoded)
}

// ---------------------------------------------------------------------------
// Schema JSON parsing (minimal, no external JSON library)
// ---------------------------------------------------------------------------

/// Parse an Avro schema from its JSON representation.
///
/// We implement a minimal JSON parser sufficient for Avro record schemas with
/// primitive field types.
///
/// # Examples
///
/// ```
/// use scivex_io::avro::{parse_schema, AvroType};
/// let json = r#"{"type":"record","name":"User","fields":[{"name":"id","type":"long"}]}"#;
/// let schema = parse_schema(json).unwrap();
/// assert_eq!(schema.name, "User");
/// assert_eq!(schema.fields[0].avro_type, AvroType::Long);
/// ```
pub fn parse_schema(json: &str) -> Result<AvroSchema> {
    let json = json.trim();
    let obj = parse_json_object(json)?;

    // Expect "type": "record"
    let record_type = obj
        .get("type")
        .ok_or_else(|| IoError::FormatError("schema missing 'type'".into()))?;
    // Strip surrounding quotes from JSON string values
    let record_type_unquoted = record_type.trim_matches('"');
    if record_type_unquoted != "record" {
        return Err(IoError::FormatError(format!(
            "expected top-level type 'record', got '{record_type}'"
        )));
    }

    let name = obj
        .get("name")
        .map_or_else(|| "Record".into(), |s| s.trim_matches('"').to_string());

    // Parse fields array
    let fields_str = obj
        .get("fields")
        .ok_or_else(|| IoError::FormatError("schema missing 'fields'".into()))?;
    let fields = parse_fields_array(fields_str)?;

    Ok(AvroSchema { name, fields })
}

/// A very minimal JSON object parser. Returns key-value pairs where values
/// are raw JSON strings (not recursively parsed).
fn parse_json_object(
    s: &str,
) -> Result<std::collections::HashMap<std::string::String, std::string::String>> {
    let s = s.trim();
    if !s.starts_with('{') || !s.ends_with('}') {
        return Err(IoError::FormatError("expected JSON object".into()));
    }
    let inner = &s[1..s.len() - 1];
    let mut map = std::collections::HashMap::new();

    let pairs = split_json_top_level(inner, ',');
    for pair in pairs {
        let pair = pair.trim();
        if pair.is_empty() {
            continue;
        }
        let colon_pos = find_colon(pair)?;
        let key = pair[..colon_pos].trim();
        let key = unquote_json_string(key)?;
        let value = pair[colon_pos + 1..].trim().to_string();
        map.insert(key, value);
    }
    Ok(map)
}

/// Find the position of the first colon that is not inside a string.
fn find_colon(s: &str) -> Result<usize> {
    let mut in_string = false;
    let mut escape_next = false;
    for (i, ch) in s.char_indices() {
        if escape_next {
            escape_next = false;
            continue;
        }
        if ch == '\\' && in_string {
            escape_next = true;
            continue;
        }
        if ch == '"' {
            in_string = !in_string;
            continue;
        }
        if ch == ':' && !in_string {
            return Ok(i);
        }
    }
    Err(IoError::FormatError("no colon found in JSON pair".into()))
}

/// Split a JSON string at top-level commas (not nested in objects/arrays/strings).
fn split_json_top_level(s: &str, sep: char) -> Vec<std::string::String> {
    let mut parts = Vec::new();
    let mut current = std::string::String::new();
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escape_next = false;

    for ch in s.chars() {
        if escape_next {
            current.push(ch);
            escape_next = false;
            continue;
        }
        if ch == '\\' && in_string {
            current.push(ch);
            escape_next = true;
            continue;
        }
        if ch == '"' {
            in_string = !in_string;
            current.push(ch);
            continue;
        }
        if !in_string {
            if ch == '{' || ch == '[' {
                depth += 1;
            } else if ch == '}' || ch == ']' {
                depth -= 1;
            } else if ch == sep && depth == 0 {
                parts.push(std::mem::take(&mut current));
                continue;
            }
        }
        current.push(ch);
    }
    if !current.trim().is_empty() {
        parts.push(current);
    }
    parts
}

/// Remove surrounding double-quotes and unescape basic JSON strings.
fn unquote_json_string(s: &str) -> Result<std::string::String> {
    let s = s.trim();
    if s.len() < 2 || !s.starts_with('"') || !s.ends_with('"') {
        return Err(IoError::FormatError(format!(
            "expected quoted JSON string, got: {s}"
        )));
    }
    let inner = &s[1..s.len() - 1];
    let mut result = std::string::String::new();
    let mut chars = inner.chars();
    while let Some(ch) = chars.next() {
        if ch == '\\' {
            match chars.next() {
                Some('"') => result.push('"'),
                Some('\\') | None => result.push('\\'),
                Some('/') => result.push('/'),
                Some('n') => result.push('\n'),
                Some('t') => result.push('\t'),
                Some('r') => result.push('\r'),
                Some(other) => {
                    result.push('\\');
                    result.push(other);
                }
            }
        } else {
            result.push(ch);
        }
    }
    Ok(result)
}

/// Parse the JSON array of field objects.
fn parse_fields_array(s: &str) -> Result<Vec<AvroField>> {
    let s = s.trim();
    if !s.starts_with('[') || !s.ends_with(']') {
        return Err(IoError::FormatError(
            "expected JSON array for fields".into(),
        ));
    }
    let inner = &s[1..s.len() - 1];
    let items = split_json_top_level(inner, ',');
    let mut fields = Vec::new();
    for item in items {
        let item = item.trim();
        if item.is_empty() {
            continue;
        }
        let obj = parse_json_object(item)?;
        let name = obj
            .get("name")
            .ok_or_else(|| IoError::FormatError("field missing 'name'".into()))?
            .clone();
        let name = unquote_json_string(&name)?;
        let type_str = obj
            .get("type")
            .ok_or_else(|| IoError::FormatError("field missing 'type'".into()))?;
        let avro_type = parse_avro_type(type_str.trim())?;
        fields.push(AvroField { name, avro_type });
    }
    Ok(fields)
}

/// Parse an Avro type from its JSON representation.
fn parse_avro_type(s: &str) -> Result<AvroType> {
    let s = s.trim();
    // Simple quoted string type
    if s.starts_with('"') && s.ends_with('"') {
        let type_name = unquote_json_string(s)?;
        return match type_name.as_str() {
            "null" => Ok(AvroType::Null),
            "boolean" => Ok(AvroType::Boolean),
            "int" => Ok(AvroType::Int),
            "long" => Ok(AvroType::Long),
            "float" => Ok(AvroType::Float),
            "double" => Ok(AvroType::Double),
            "string" => Ok(AvroType::String),
            "bytes" => Ok(AvroType::Bytes),
            other => Err(IoError::FormatError(format!(
                "unsupported Avro type: {other}"
            ))),
        };
    }
    // Array type = union
    if s.starts_with('[') && s.ends_with(']') {
        let inner = &s[1..s.len() - 1];
        let items = split_json_top_level(inner, ',');
        if items.len() == 2 {
            let first = items[0].trim();
            let second = items[1].trim();
            // ["null", T] or [T, "null"]
            if first == "\"null\"" {
                let inner_type = parse_avro_type(second)?;
                return Ok(AvroType::Union(Box::new(inner_type)));
            } else if second == "\"null\"" {
                let inner_type = parse_avro_type(first)?;
                return Ok(AvroType::Union(Box::new(inner_type)));
            }
        }
        return Err(IoError::FormatError(
            "only [\"null\", T] unions are supported".into(),
        ));
    }
    Err(IoError::FormatError(format!(
        "unsupported Avro type expression: {s}"
    )))
}

// ---------------------------------------------------------------------------
// Schema to JSON serialization (for writing)
// ---------------------------------------------------------------------------

fn avro_type_to_json(ty: &AvroType) -> std::string::String {
    match ty {
        AvroType::Null => "\"null\"".into(),
        AvroType::Boolean => "\"boolean\"".into(),
        AvroType::Int => "\"int\"".into(),
        AvroType::Long => "\"long\"".into(),
        AvroType::Float => "\"float\"".into(),
        AvroType::Double => "\"double\"".into(),
        AvroType::String => "\"string\"".into(),
        AvroType::Bytes => "\"bytes\"".into(),
        AvroType::Union(inner) => {
            format!("[\"null\",{}]", avro_type_to_json(inner))
        }
    }
}

fn schema_to_json(schema: &AvroSchema) -> std::string::String {
    let mut s = std::string::String::new();
    s.push_str("{\"type\":\"record\",\"name\":\"");
    s.push_str(&schema.name);
    s.push_str("\",\"fields\":[");
    for (i, field) in schema.fields.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str("{\"name\":\"");
        s.push_str(&field.name);
        s.push_str("\",\"type\":");
        s.push_str(&avro_type_to_json(&field.avro_type));
        s.push('}');
    }
    s.push_str("]}");
    s
}

// ---------------------------------------------------------------------------
// Binary reading helpers
// ---------------------------------------------------------------------------

fn read_exact<R: Read>(reader: &mut R, len: usize) -> Result<Vec<u8>> {
    let mut buf = vec![0u8; len];
    reader
        .read_exact(&mut buf)
        .map_err(|e| IoError::FormatError(format!("unexpected EOF reading {len} bytes: {e}")))?;
    Ok(buf)
}

fn read_avro_string<R: Read>(reader: &mut R) -> Result<std::string::String> {
    let len = zigzag_decode_i64(reader)?;
    if len < 0 {
        return Err(IoError::FormatError("negative string length".into()));
    }
    let bytes = read_exact(reader, len as usize)?;
    std::string::String::from_utf8(bytes)
        .map_err(|e| IoError::FormatError(format!("invalid UTF-8 in Avro string: {e}")))
}

fn read_avro_bytes<R: Read>(reader: &mut R) -> Result<Vec<u8>> {
    let len = zigzag_decode_i64(reader)?;
    if len < 0 {
        return Err(IoError::FormatError("negative bytes length".into()));
    }
    read_exact(reader, len as usize)
}

fn read_f32<R: Read>(reader: &mut R) -> Result<f32> {
    let mut buf = [0u8; 4];
    reader
        .read_exact(&mut buf)
        .map_err(|e| IoError::FormatError(format!("failed to read f32: {e}")))?;
    Ok(f32::from_le_bytes(buf))
}

fn read_f64<R: Read>(reader: &mut R) -> Result<f64> {
    let mut buf = [0u8; 8];
    reader
        .read_exact(&mut buf)
        .map_err(|e| IoError::FormatError(format!("failed to read f64: {e}")))?;
    Ok(f64::from_le_bytes(buf))
}

// ---------------------------------------------------------------------------
// Avro container file reader
// ---------------------------------------------------------------------------

/// Parsed Avro file header.
///
/// Obtain via [`read_avro_header`] after writing an Avro file with
/// [`write_avro`].
///
/// # Examples
///
/// ```
/// use std::io::Cursor;
/// use scivex_io::avro::{write_avro, read_avro_header};
/// use scivex_frame::{DataFrame, Series, AnySeries};
///
/// let col: Box<dyn AnySeries> = Box::new(Series::new("x", vec![1_i64, 2]));
/// let df = DataFrame::new(vec![col]).unwrap();
/// let mut buf = Vec::new();
/// write_avro(&mut buf, &df).unwrap();
///
/// let mut cursor = Cursor::new(&buf);
/// let header = read_avro_header(&mut cursor).unwrap();
/// assert_eq!(header.codec, "null");
/// assert_eq!(header.schema.fields[0].name, "x");
/// ```
#[derive(Debug, Clone)]
pub struct AvroHeader {
    /// The record schema.
    pub schema: AvroSchema,
    /// The 16-byte sync marker.
    pub sync_marker: [u8; SYNC_MARKER_LEN],
    /// The codec (only "null" is supported).
    pub codec: std::string::String,
}

/// Read the Avro container file header from a reader.
///
/// # Examples
///
/// ```
/// use std::io::Cursor;
/// use scivex_io::avro::{write_avro, read_avro_header};
/// use scivex_frame::{DataFrame, Series, AnySeries};
///
/// let col: Box<dyn AnySeries> = Box::new(Series::new("n", vec![10_i64]));
/// let df = DataFrame::new(vec![col]).unwrap();
/// let mut buf = Vec::new();
/// write_avro(&mut buf, &df).unwrap();
/// let mut cursor = Cursor::new(&buf);
/// let header = read_avro_header(&mut cursor).unwrap();
/// assert_eq!(header.schema.name, "DataFrame");
/// ```
pub fn read_avro_header<R: Read>(reader: &mut R) -> Result<AvroHeader> {
    // 1. Magic bytes
    let magic = read_exact(reader, 4)?;
    if magic != AVRO_MAGIC {
        return Err(IoError::FormatError(
            "invalid Avro magic bytes (expected Obj\\x01)".into(),
        ));
    }

    // 2. File metadata — stored as Avro map: { count, [key, value]*, 0 }
    let mut metadata = std::collections::HashMap::new();
    loop {
        let count = zigzag_decode_i64(reader)?;
        if count == 0 {
            break;
        }
        let abs_count = count.unsigned_abs();
        if count < 0 {
            // Negative count means the next long is the byte size of the block;
            // we just skip that length value and read entries.
            let _byte_size = zigzag_decode_i64(reader)?;
        }
        for _ in 0..abs_count {
            let key = read_avro_string(reader)?;
            let value = read_avro_bytes(reader)?;
            metadata.insert(key, value);
        }
    }

    // 3. Sync marker
    let sync_bytes = read_exact(reader, SYNC_MARKER_LEN)?;
    let mut sync_marker = [0u8; SYNC_MARKER_LEN];
    sync_marker.copy_from_slice(&sync_bytes);

    // 4. Parse schema from metadata
    let schema_bytes = metadata
        .get("avro.schema")
        .ok_or_else(|| IoError::FormatError("missing avro.schema in metadata".into()))?;
    let schema_str = std::str::from_utf8(schema_bytes)
        .map_err(|e| IoError::FormatError(format!("avro.schema is not valid UTF-8: {e}")))?;
    let schema = parse_schema(schema_str)?;

    let codec = metadata.get("avro.codec").map_or_else(
        || "null".into(),
        |b| std::string::String::from_utf8_lossy(b).into_owned(),
    );

    if codec != "null" {
        return Err(IoError::FormatError(format!(
            "unsupported Avro codec: {codec} (only 'null' is supported)"
        )));
    }

    Ok(AvroHeader {
        schema,
        sync_marker,
        codec,
    })
}

/// An intermediate row value for a single Avro datum field.
#[derive(Debug, Clone)]
enum AvroValue {
    Null,
    Boolean(bool),
    Int(i32),
    Long(i64),
    Float(f32),
    Double(f64),
    String(std::string::String),
    Bytes(Vec<u8>),
}

/// Read a single datum value for the given Avro type.
fn read_avro_value<R: Read>(reader: &mut R, avro_type: &AvroType) -> Result<AvroValue> {
    match avro_type {
        AvroType::Null => Ok(AvroValue::Null),
        AvroType::Boolean => {
            let mut buf = [0u8; 1];
            reader
                .read_exact(&mut buf)
                .map_err(|e| IoError::FormatError(format!("failed to read boolean: {e}")))?;
            Ok(AvroValue::Boolean(buf[0] != 0))
        }
        AvroType::Int => {
            let v = zigzag_decode_i64(reader)?;
            Ok(AvroValue::Int(v as i32))
        }
        AvroType::Long => {
            let v = zigzag_decode_i64(reader)?;
            Ok(AvroValue::Long(v))
        }
        AvroType::Float => {
            let v = read_f32(reader)?;
            Ok(AvroValue::Float(v))
        }
        AvroType::Double => {
            let v = read_f64(reader)?;
            Ok(AvroValue::Double(v))
        }
        AvroType::String => {
            let s = read_avro_string(reader)?;
            Ok(AvroValue::String(s))
        }
        AvroType::Bytes => {
            let b = read_avro_bytes(reader)?;
            Ok(AvroValue::Bytes(b))
        }
        AvroType::Union(inner) => {
            // Union index: 0 = first branch (null), 1 = second branch
            let idx = zigzag_decode_i64(reader)?;
            if idx == 0 {
                Ok(AvroValue::Null)
            } else if idx == 1 {
                read_avro_value(reader, inner)
            } else {
                Err(IoError::FormatError(format!(
                    "unexpected union index: {idx}"
                )))
            }
        }
    }
}

/// Read an Avro container file into a [`DataFrame`].
///
/// # Examples
///
/// ```
/// use std::io::Cursor;
/// use scivex_io::avro::{write_avro, read_avro};
/// use scivex_frame::{DataFrame, Series, AnySeries};
///
/// let col: Box<dyn AnySeries> = Box::new(Series::new("id", vec![1_i64, 2, 3]));
/// let df = DataFrame::new(vec![col]).unwrap();
/// let mut buf = Vec::new();
/// write_avro(&mut buf, &df).unwrap();
///
/// let mut cursor = Cursor::new(&buf);
/// let df2 = read_avro(&mut cursor).unwrap();
/// assert_eq!(df2.nrows(), 3);
/// assert_eq!(df2.ncols(), 1);
/// ```
pub fn read_avro<R: Read>(reader: &mut R) -> Result<DataFrame> {
    let header = read_avro_header(reader)?;
    let schema = &header.schema;
    let num_fields = schema.fields.len();

    // Accumulate column data
    let mut columns: Vec<Vec<AvroValue>> = vec![Vec::new(); num_fields];

    // Read data blocks
    while let Ok(row_count) = zigzag_decode_i64(reader) {
        if row_count == 0 {
            break;
        }
        #[allow(clippy::cast_sign_loss)]
        let row_count = row_count as usize;

        // Block byte size (we ignore this since codec is "null")
        let _block_bytes = zigzag_decode_i64(reader)?;

        // Read rows
        for _ in 0..row_count {
            for (fi, field) in schema.fields.iter().enumerate() {
                let value = read_avro_value(reader, &field.avro_type)?;
                columns[fi].push(value);
            }
        }

        // Sync marker
        let sync = read_exact(reader, SYNC_MARKER_LEN)?;
        if sync != header.sync_marker {
            return Err(IoError::FormatError("sync marker mismatch".into()));
        }
    }

    // Convert columns to Series
    build_dataframe_from_avro_columns(schema, &columns)
}

/// Convert accumulated Avro column data into a DataFrame.
#[allow(clippy::too_many_lines)]
fn build_dataframe_from_avro_columns(
    schema: &AvroSchema,
    columns: &[Vec<AvroValue>],
) -> Result<DataFrame> {
    let mut series_vec: Vec<Box<dyn AnySeries>> = Vec::with_capacity(schema.fields.len());

    for (fi, field) in schema.fields.iter().enumerate() {
        let col_data = &columns[fi];
        let base_type = match &field.avro_type {
            AvroType::Union(inner) => inner.as_ref(),
            other => other,
        };
        let is_nullable = matches!(field.avro_type, AvroType::Union(_));

        let series: Box<dyn AnySeries> = match base_type {
            AvroType::Boolean => {
                let mut data = Vec::with_capacity(col_data.len());
                let mut null_mask = Vec::with_capacity(col_data.len());
                let mut has_nulls = false;
                for v in col_data {
                    match v {
                        AvroValue::Boolean(b) => {
                            data.push(u8::from(*b));
                            null_mask.push(false);
                        }
                        AvroValue::Null => {
                            data.push(0);
                            null_mask.push(true);
                            has_nulls = true;
                        }
                        _ => {
                            return Err(IoError::FormatError(format!(
                                "unexpected value type for boolean field '{}'",
                                field.name
                            )));
                        }
                    }
                }
                if has_nulls || is_nullable {
                    Box::new(
                        Series::with_nulls(&field.name, data, null_mask)
                            .map_err(|e| IoError::FormatError(format!("series error: {e}")))?,
                    )
                } else {
                    Box::new(Series::new(&field.name, data))
                }
            }
            AvroType::Int | AvroType::Long => {
                let mut data = Vec::with_capacity(col_data.len());
                let mut null_mask = Vec::with_capacity(col_data.len());
                let mut has_nulls = false;
                for v in col_data {
                    match v {
                        AvroValue::Int(n) => {
                            data.push(i64::from(*n));
                            null_mask.push(false);
                        }
                        AvroValue::Long(n) => {
                            data.push(*n);
                            null_mask.push(false);
                        }
                        AvroValue::Null => {
                            data.push(0);
                            null_mask.push(true);
                            has_nulls = true;
                        }
                        _ => {
                            return Err(IoError::FormatError(format!(
                                "unexpected value type for int/long field '{}'",
                                field.name
                            )));
                        }
                    }
                }
                if has_nulls || is_nullable {
                    Box::new(
                        Series::with_nulls(&field.name, data, null_mask)
                            .map_err(|e| IoError::FormatError(format!("series error: {e}")))?,
                    )
                } else {
                    Box::new(Series::new(&field.name, data))
                }
            }
            AvroType::Float | AvroType::Double => {
                let mut data = Vec::with_capacity(col_data.len());
                let mut null_mask = Vec::with_capacity(col_data.len());
                let mut has_nulls = false;
                for v in col_data {
                    match v {
                        AvroValue::Float(f) => {
                            data.push(f64::from(*f));
                            null_mask.push(false);
                        }
                        AvroValue::Double(f) => {
                            data.push(*f);
                            null_mask.push(false);
                        }
                        AvroValue::Null => {
                            data.push(0.0);
                            null_mask.push(true);
                            has_nulls = true;
                        }
                        _ => {
                            return Err(IoError::FormatError(format!(
                                "unexpected value type for float/double field '{}'",
                                field.name
                            )));
                        }
                    }
                }
                if has_nulls || is_nullable {
                    Box::new(
                        Series::with_nulls(&field.name, data, null_mask)
                            .map_err(|e| IoError::FormatError(format!("series error: {e}")))?,
                    )
                } else {
                    Box::new(Series::new(&field.name, data))
                }
            }
            AvroType::String | AvroType::Bytes => {
                let mut data = Vec::with_capacity(col_data.len());
                let mut null_mask = Vec::with_capacity(col_data.len());
                let mut has_nulls = false;
                for v in col_data {
                    match v {
                        AvroValue::String(s) => {
                            data.push(s.clone());
                            null_mask.push(false);
                        }
                        AvroValue::Bytes(b) => {
                            data.push(std::string::String::from_utf8_lossy(b).into_owned());
                            null_mask.push(false);
                        }
                        AvroValue::Null => {
                            data.push(std::string::String::new());
                            null_mask.push(true);
                            has_nulls = true;
                        }
                        _ => {
                            return Err(IoError::FormatError(format!(
                                "unexpected value type for string/bytes field '{}'",
                                field.name
                            )));
                        }
                    }
                }
                if has_nulls || is_nullable {
                    Box::new(
                        StringSeries::with_nulls(&field.name, data, null_mask)
                            .map_err(|e| IoError::FormatError(format!("series error: {e}")))?,
                    )
                } else {
                    Box::new(StringSeries::new(&field.name, data))
                }
            }
            AvroType::Null => {
                // All-null column — store as i64 with all nulls
                let null_mask = vec![true; col_data.len()];
                let data = vec![0i64; col_data.len()];
                Box::new(
                    Series::with_nulls(&field.name, data, null_mask)
                        .map_err(|e| IoError::FormatError(format!("series error: {e}")))?,
                )
            }
            AvroType::Union(_) => {
                // Should not happen since we unwrap unions above
                return Err(IoError::FormatError("nested unions not supported".into()));
            }
        };
        series_vec.push(series);
    }

    if series_vec.is_empty() {
        return Ok(DataFrame::empty());
    }
    Ok(DataFrame::new(series_vec)?)
}

// ---------------------------------------------------------------------------
// Avro container file writer
// ---------------------------------------------------------------------------

/// Write a [`DataFrame`] to Avro container file format.
///
/// # Examples
///
/// ```
/// use scivex_io::avro::write_avro;
/// use scivex_frame::{DataFrame, Series, AnySeries};
///
/// let col: Box<dyn AnySeries> = Box::new(Series::new("val", vec![10_i64, 20]));
/// let df = DataFrame::new(vec![col]).unwrap();
/// let mut buf = Vec::new();
/// write_avro(&mut buf, &df).unwrap();
/// // Avro files start with "Obj\x01"
/// assert_eq!(&buf[..3], b"Obj");
/// ```
pub fn write_avro<W: Write>(writer: &mut W, df: &DataFrame) -> Result<()> {
    let schema = dataframe_to_schema(df);

    // Generate a random sync marker (simple deterministic for reproducibility
    // isn't required; use column count + row count based hash).
    let mut sync_marker = [0u8; SYNC_MARKER_LEN];
    let seed = (df.ncols() as u64)
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(df.nrows() as u64)
        .wrapping_mul(1_442_695_040_888_963_407);
    for (i, b) in sync_marker.iter_mut().enumerate() {
        *b = ((seed.wrapping_mul((i as u64).wrapping_add(1))) >> (i % 8 * 8)) as u8;
    }

    write_header(writer, &schema, &sync_marker)?;

    let nrows = df.nrows();
    if nrows == 0 {
        return Ok(());
    }

    // Write all rows in a single block
    let mut block_buf: Vec<u8> = Vec::new();
    let columns = df.columns();

    for row in 0..nrows {
        for (fi, field) in schema.fields.iter().enumerate() {
            write_avro_value_for_column(&mut block_buf, &*columns[fi], row, &field.avro_type)?;
        }
    }

    // Block header: row count, byte size
    #[allow(clippy::cast_possible_wrap)]
    let nrows_i64 = nrows as i64;
    writer.write_all(&zigzag_encode_i64(nrows_i64))?;
    #[allow(clippy::cast_possible_wrap)]
    let block_len_i64 = block_buf.len() as i64;
    writer.write_all(&zigzag_encode_i64(block_len_i64))?;
    writer.write_all(&block_buf)?;
    writer.write_all(&sync_marker)?;

    Ok(())
}

/// Infer an Avro schema from a DataFrame.
fn dataframe_to_schema(df: &DataFrame) -> AvroSchema {
    let columns = df.columns();
    let mut fields = Vec::with_capacity(columns.len());

    for col in columns {
        let avro_type = dtype_to_avro_type(col.dtype());
        fields.push(AvroField {
            name: col.name().to_string(),
            avro_type,
        });
    }

    AvroSchema {
        name: "DataFrame".into(),
        fields,
    }
}

/// Map a DataFrame DType to an Avro type.
fn dtype_to_avro_type(dtype: DType) -> AvroType {
    match dtype {
        DType::I64 | DType::I32 | DType::I16 | DType::I8 | DType::U64 | DType::U32 | DType::U16 => {
            AvroType::Long
        }
        DType::U8 | DType::Bool => AvroType::Boolean,
        DType::F64 => AvroType::Double,
        DType::F32 => AvroType::Float,
        DType::Str | DType::Categorical | DType::DateTime => AvroType::String,
    }
}

/// Write the Avro container file header.
fn write_header<W: Write>(
    writer: &mut W,
    schema: &AvroSchema,
    sync_marker: &[u8; SYNC_MARKER_LEN],
) -> Result<()> {
    // Magic
    writer.write_all(&AVRO_MAGIC)?;

    // Metadata map — we write two entries: avro.schema and avro.codec
    let schema_json = schema_to_json(schema);
    let schema_bytes = schema_json.as_bytes();
    let codec_bytes = b"null";

    // Map block count = 2
    writer.write_all(&zigzag_encode_i64(2))?;

    // Entry 1: avro.schema
    write_avro_string_to(writer, "avro.schema")?;
    write_avro_bytes_to(writer, schema_bytes)?;

    // Entry 2: avro.codec
    write_avro_string_to(writer, "avro.codec")?;
    write_avro_bytes_to(writer, codec_bytes)?;

    // End of map
    writer.write_all(&zigzag_encode_i64(0))?;

    // Sync marker
    writer.write_all(sync_marker)?;

    Ok(())
}

fn write_avro_string_to<W: Write>(writer: &mut W, s: &str) -> std::io::Result<()> {
    #[allow(clippy::cast_possible_wrap)]
    let len = s.len() as i64;
    writer.write_all(&zigzag_encode_i64(len))?;
    writer.write_all(s.as_bytes())
}

fn write_avro_bytes_to<W: Write>(writer: &mut W, b: &[u8]) -> std::io::Result<()> {
    #[allow(clippy::cast_possible_wrap)]
    let len = b.len() as i64;
    writer.write_all(&zigzag_encode_i64(len))?;
    writer.write_all(b)
}

/// Write a single value from a column at a given row.
fn write_avro_value_for_column<W: Write>(
    writer: &mut W,
    col: &dyn AnySeries,
    row: usize,
    avro_type: &AvroType,
) -> Result<()> {
    match avro_type {
        AvroType::Boolean => {
            // Bool columns stored as Series<u8>
            let typed = col.as_any().downcast_ref::<Series<u8>>().ok_or_else(|| {
                IoError::FormatError(format!(
                    "expected u8 series for boolean column '{}'",
                    col.name()
                ))
            })?;
            let val = typed.as_slice()[row];
            writer.write_all(&[u8::from(val != 0)])?;
        }
        AvroType::Int => {
            // u8 columns -> int
            if let Some(typed) = col.as_any().downcast_ref::<Series<u8>>() {
                writer.write_all(&zigzag_encode_i64(i64::from(typed.as_slice()[row])))?;
            } else {
                let val = col.display_value(row);
                let n: i64 = val
                    .parse()
                    .map_err(|_| IoError::FormatError(format!("cannot parse '{val}' as int")))?;
                writer.write_all(&zigzag_encode_i64(n))?;
            }
        }
        AvroType::Long => {
            if let Some(typed) = col.as_any().downcast_ref::<Series<i64>>() {
                writer.write_all(&zigzag_encode_i64(typed.as_slice()[row]))?;
            } else if let Some(typed) = col.as_any().downcast_ref::<Series<i32>>() {
                writer.write_all(&zigzag_encode_i64(i64::from(typed.as_slice()[row])))?;
            } else if let Some(typed) = col.as_any().downcast_ref::<Series<i16>>() {
                writer.write_all(&zigzag_encode_i64(i64::from(typed.as_slice()[row])))?;
            } else if let Some(typed) = col.as_any().downcast_ref::<Series<i8>>() {
                writer.write_all(&zigzag_encode_i64(i64::from(typed.as_slice()[row])))?;
            } else if let Some(typed) = col.as_any().downcast_ref::<Series<u64>>() {
                #[allow(clippy::cast_possible_wrap)]
                let val = typed.as_slice()[row] as i64;
                writer.write_all(&zigzag_encode_i64(val))?;
            } else if let Some(typed) = col.as_any().downcast_ref::<Series<u32>>() {
                writer.write_all(&zigzag_encode_i64(i64::from(typed.as_slice()[row])))?;
            } else if let Some(typed) = col.as_any().downcast_ref::<Series<u16>>() {
                writer.write_all(&zigzag_encode_i64(i64::from(typed.as_slice()[row])))?;
            } else {
                let val = col.display_value(row);
                let n: i64 = val
                    .parse()
                    .map_err(|_| IoError::FormatError(format!("cannot parse '{val}' as long")))?;
                writer.write_all(&zigzag_encode_i64(n))?;
            }
        }
        AvroType::Float => {
            if let Some(typed) = col.as_any().downcast_ref::<Series<f32>>() {
                writer.write_all(&typed.as_slice()[row].to_le_bytes())?;
            } else {
                let val = col.display_value(row);
                let f: f32 = val
                    .parse()
                    .map_err(|_| IoError::FormatError(format!("cannot parse '{val}' as float")))?;
                writer.write_all(&f.to_le_bytes())?;
            }
        }
        AvroType::Double => {
            if let Some(typed) = col.as_any().downcast_ref::<Series<f64>>() {
                writer.write_all(&typed.as_slice()[row].to_le_bytes())?;
            } else {
                let val = col.display_value(row);
                let f: f64 = val
                    .parse()
                    .map_err(|_| IoError::FormatError(format!("cannot parse '{val}' as double")))?;
                writer.write_all(&f.to_le_bytes())?;
            }
        }
        AvroType::String => {
            if let Some(typed) = col.as_any().downcast_ref::<StringSeries>() {
                let s = typed.get(row).unwrap_or("");
                write_avro_string_to(writer, s)?;
            } else {
                let val = col.display_value(row);
                write_avro_string_to(writer, &val)?;
            }
        }
        AvroType::Bytes => {
            let val = col.display_value(row);
            write_avro_bytes_to(writer, val.as_bytes())?;
        }
        AvroType::Null => {
            // Nothing to write for null type
        }
        AvroType::Union(_inner) => {
            // We don't write unions from DataFrame (no nullable columns in output)
            // This shouldn't be reached from dataframe_to_schema which only
            // produces non-union types.
            return Err(IoError::FormatError(
                "writing union types from DataFrame is not supported".into(),
            ));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn make_test_df() -> DataFrame {
        let id: Box<dyn AnySeries> = Box::new(Series::new("id", vec![1_i64, 2, 3]));
        let score: Box<dyn AnySeries> = Box::new(Series::new("score", vec![1.5_f64, 2.5, 3.5]));
        let name: Box<dyn AnySeries> =
            Box::new(StringSeries::from_strs("name", &["Alice", "Bob", "Carol"]));
        DataFrame::new(vec![id, score, name]).unwrap()
    }

    #[test]
    fn test_avro_roundtrip_basic() {
        let df = make_test_df();
        let mut buf = Vec::new();
        write_avro(&mut buf, &df).unwrap();

        let mut cursor = Cursor::new(&buf);
        let df2 = read_avro(&mut cursor).unwrap();

        assert_eq!(df2.nrows(), 3);
        assert_eq!(df2.ncols(), 3);
        assert_eq!(df2.column_names(), vec!["id", "score", "name"]);

        // Check int column
        let id_col = df2.column("id").unwrap();
        let id_typed = id_col.as_any().downcast_ref::<Series<i64>>().unwrap();
        assert_eq!(id_typed.as_slice(), &[1, 2, 3]);

        // Check float column
        let score_col = df2.column("score").unwrap();
        let score_typed = score_col.as_any().downcast_ref::<Series<f64>>().unwrap();
        assert_eq!(score_typed.as_slice(), &[1.5, 2.5, 3.5]);

        // Check string column
        let name_col = df2.column("name").unwrap();
        let name_typed = name_col.as_any().downcast_ref::<StringSeries>().unwrap();
        assert_eq!(name_typed.get(0), Some("Alice"));
        assert_eq!(name_typed.get(1), Some("Bob"));
        assert_eq!(name_typed.get(2), Some("Carol"));
    }

    #[test]
    fn test_zigzag_varint_roundtrip() {
        let test_values: Vec<i64> = vec![
            0,
            1,
            -1,
            42,
            -42,
            127,
            -128,
            1000,
            -1000,
            i64::MAX,
            i64::MIN,
        ];
        for &val in &test_values {
            let encoded = zigzag_encode_i64(val);
            let mut cursor = Cursor::new(&encoded);
            let decoded = zigzag_decode_i64(&mut cursor).unwrap();
            assert_eq!(val, decoded, "zigzag roundtrip failed for {val}");
        }
    }

    #[test]
    fn test_avro_boolean_column() {
        let bools: Box<dyn AnySeries> = Box::new(Series::new("flag", vec![1_u8, 0, 1, 0]));
        let df = DataFrame::new(vec![bools]).unwrap();

        let mut buf = Vec::new();
        write_avro(&mut buf, &df).unwrap();

        let mut cursor = Cursor::new(&buf);
        let df2 = read_avro(&mut cursor).unwrap();

        assert_eq!(df2.nrows(), 4);
        let col = df2.column("flag").unwrap();
        assert_eq!(col.dtype(), DType::U8);
        let typed = col.as_any().downcast_ref::<Series<u8>>().unwrap();
        assert_eq!(typed.as_slice(), &[1, 0, 1, 0]);
    }

    #[test]
    fn test_avro_empty_dataframe() {
        let id: Box<dyn AnySeries> = Box::new(Series::<i64>::new("id", vec![]));
        let name: Box<dyn AnySeries> = Box::new(StringSeries::new("name", vec![]));
        let df = DataFrame::new(vec![id, name]).unwrap();

        let mut buf = Vec::new();
        write_avro(&mut buf, &df).unwrap();

        let mut cursor = Cursor::new(&buf);
        let df2 = read_avro(&mut cursor).unwrap();

        assert_eq!(df2.nrows(), 0);
        // Schema should still have columns
        assert_eq!(df2.ncols(), 2);
        assert_eq!(df2.column_names(), vec!["id", "name"]);
    }

    #[test]
    fn test_avro_schema_extraction() {
        let df = make_test_df();
        let mut buf = Vec::new();
        write_avro(&mut buf, &df).unwrap();

        let mut cursor = Cursor::new(&buf);
        let header = read_avro_header(&mut cursor).unwrap();

        assert_eq!(header.schema.name, "DataFrame");
        assert_eq!(header.schema.fields.len(), 3);
        assert_eq!(header.schema.fields[0].name, "id");
        assert_eq!(header.schema.fields[0].avro_type, AvroType::Long);
        assert_eq!(header.schema.fields[1].name, "score");
        assert_eq!(header.schema.fields[1].avro_type, AvroType::Double);
        assert_eq!(header.schema.fields[2].name, "name");
        assert_eq!(header.schema.fields[2].avro_type, AvroType::String);
        assert_eq!(header.codec, "null");
    }

    #[test]
    fn test_avro_large_dataframe() {
        let n = 1000;
        let ids: Vec<i64> = (0..n).collect();
        let scores: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
        let names: Vec<std::string::String> = (0..n).map(|i| format!("row_{i}")).collect();

        let id_col: Box<dyn AnySeries> = Box::new(Series::new("id", ids.clone()));
        let score_col: Box<dyn AnySeries> = Box::new(Series::new("score", scores.clone()));
        let name_col: Box<dyn AnySeries> = Box::new(StringSeries::new("name", names.clone()));
        let df = DataFrame::new(vec![id_col, score_col, name_col]).unwrap();

        let mut buf = Vec::new();
        write_avro(&mut buf, &df).unwrap();

        let mut cursor = Cursor::new(&buf);
        let df2 = read_avro(&mut cursor).unwrap();

        assert_eq!(df2.nrows(), n as usize);
        assert_eq!(df2.ncols(), 3);

        let id_typed = df2
            .column("id")
            .unwrap()
            .as_any()
            .downcast_ref::<Series<i64>>()
            .unwrap();
        assert_eq!(id_typed.as_slice(), &ids);

        let score_typed = df2
            .column("score")
            .unwrap()
            .as_any()
            .downcast_ref::<Series<f64>>()
            .unwrap();
        assert_eq!(score_typed.as_slice(), &scores);

        let name_typed = df2
            .column("name")
            .unwrap()
            .as_any()
            .downcast_ref::<StringSeries>()
            .unwrap();
        for (i, name) in names.iter().enumerate().take(n as usize) {
            assert_eq!(name_typed.get(i), Some(name.as_str()));
        }
    }

    #[test]
    fn test_avro_invalid_magic() {
        let data = vec![0x00, 0x01, 0x02, 0x03];
        let mut cursor = Cursor::new(&data);
        let result = read_avro(&mut cursor);
        assert!(result.is_err());
    }

    #[test]
    fn test_schema_parse_roundtrip() {
        let schema = AvroSchema {
            name: "TestRecord".into(),
            fields: vec![
                AvroField {
                    name: "x".into(),
                    avro_type: AvroType::Long,
                },
                AvroField {
                    name: "y".into(),
                    avro_type: AvroType::Double,
                },
                AvroField {
                    name: "s".into(),
                    avro_type: AvroType::String,
                },
            ],
        };
        let json = schema_to_json(&schema);
        let parsed = parse_schema(&json).unwrap();
        assert_eq!(parsed.name, "TestRecord");
        assert_eq!(parsed.fields.len(), 3);
        assert_eq!(parsed.fields[0].avro_type, AvroType::Long);
        assert_eq!(parsed.fields[1].avro_type, AvroType::Double);
        assert_eq!(parsed.fields[2].avro_type, AvroType::String);
    }

    #[test]
    fn test_avro_nullable_union_schema() {
        let schema_json =
            r#"{"type":"record","name":"Test","fields":[{"name":"x","type":["null","long"]}]}"#;
        let schema = parse_schema(schema_json).unwrap();
        assert_eq!(schema.fields.len(), 1);
        assert_eq!(
            schema.fields[0].avro_type,
            AvroType::Union(Box::new(AvroType::Long))
        );
    }
}
