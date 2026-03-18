//! Hand-rolled protobuf wire format decoder for ORC metadata.
//!
//! ORC stores its metadata (PostScript, Footer, StripeFooter, etc.) as
//! Protocol Buffer messages. We decode them from scratch rather than
//! depending on an external protobuf crate.

use crate::error::{IoError, Result};

// ---------------------------------------------------------------------------
// Wire format primitives
// ---------------------------------------------------------------------------

/// Protobuf wire types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WireType {
    Varint,          // 0
    Fixed64,         // 1
    LengthDelimited, // 2
    Fixed32,         // 5
}

impl WireType {
    fn from_u64(val: u64) -> Result<Self> {
        match val {
            0 => Ok(Self::Varint),
            1 => Ok(Self::Fixed64),
            2 => Ok(Self::LengthDelimited),
            5 => Ok(Self::Fixed32),
            other => Err(IoError::FormatError(format!(
                "unknown protobuf wire type: {other}"
            ))),
        }
    }
}

/// A parsed protobuf field tag: (field_number, wire_type).
#[derive(Debug, Clone, Copy)]
pub(crate) struct FieldTag {
    pub field_number: u32,
    pub wire_type: WireType,
}

/// Decode a varint from a byte slice, returning (value, bytes_consumed).
pub(crate) fn decode_varint(data: &[u8]) -> Result<(u64, usize)> {
    let mut result: u64 = 0;
    let mut shift: u32 = 0;
    for (i, &b) in data.iter().enumerate() {
        result |= u64::from(b & 0x7F) << shift;
        if b & 0x80 == 0 {
            return Ok((result, i + 1));
        }
        shift += 7;
        if shift >= 64 {
            return Err(IoError::FormatError("varint too long".into()));
        }
    }
    Err(IoError::FormatError("unexpected end of varint".into()))
}

/// Zigzag-decode a u64 into an i64 (used by ORC for signed integers in some
/// contexts, though most ORC integer metadata uses plain varints).
#[allow(clippy::cast_possible_wrap)]
pub(crate) fn zigzag_decode(n: u64) -> i64 {
    ((n >> 1) as i64) ^ (-((n & 1) as i64))
}

/// Parse a field tag from the byte slice.
pub(crate) fn decode_tag(data: &[u8]) -> Result<(FieldTag, usize)> {
    let (val, consumed) = decode_varint(data)?;
    let wire_type = WireType::from_u64(val & 0x07)?;
    #[allow(clippy::cast_possible_truncation)]
    let field_number = (val >> 3) as u32;
    Ok((
        FieldTag {
            field_number,
            wire_type,
        },
        consumed,
    ))
}

/// Skip a field value of the given wire type, returning bytes consumed.
pub(crate) fn skip_field(data: &[u8], wire_type: WireType) -> Result<usize> {
    match wire_type {
        WireType::Varint => {
            let (_, n) = decode_varint(data)?;
            Ok(n)
        }
        WireType::Fixed64 => {
            if data.len() < 8 {
                return Err(IoError::FormatError("truncated fixed64".into()));
            }
            Ok(8)
        }
        WireType::Fixed32 => {
            if data.len() < 4 {
                return Err(IoError::FormatError("truncated fixed32".into()));
            }
            Ok(4)
        }
        WireType::LengthDelimited => {
            let (len, n) = decode_varint(data)?;
            #[allow(clippy::cast_possible_truncation)]
            let len = len as usize;
            if data.len() < n + len {
                return Err(IoError::FormatError(
                    "truncated length-delimited field".into(),
                ));
            }
            Ok(n + len)
        }
    }
}

// ---------------------------------------------------------------------------
// ORC metadata structures
// ---------------------------------------------------------------------------

/// ORC compression kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionKind {
    None = 0,
    Zlib = 1,
    Snappy = 2,
    Lzo = 3,
    Lz4 = 4,
    Zstd = 5,
}

impl CompressionKind {
    pub(crate) fn from_u64(val: u64) -> Result<Self> {
        match val {
            0 => Ok(Self::None),
            1 => Ok(Self::Zlib),
            2 => Ok(Self::Snappy),
            3 => Ok(Self::Lzo),
            4 => Ok(Self::Lz4),
            5 => Ok(Self::Zstd),
            other => Err(IoError::FormatError(format!(
                "unknown ORC compression kind: {other}"
            ))),
        }
    }
}

/// ORC type kind (from the Type message).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeKind {
    Boolean = 0,
    Byte = 1,
    Short = 2,
    Int = 3,
    Long = 4,
    Float = 5,
    Double = 6,
    String = 7,
    Binary = 8,
    Timestamp = 9,
    List = 10,
    Map = 11,
    Struct = 12,
    Union = 13,
    Decimal = 14,
    Date = 15,
    Varchar = 16,
    Char = 17,
}

impl TypeKind {
    pub(crate) fn from_u64(val: u64) -> Result<Self> {
        match val {
            0 => Ok(Self::Boolean),
            1 => Ok(Self::Byte),
            2 => Ok(Self::Short),
            3 => Ok(Self::Int),
            4 => Ok(Self::Long),
            5 => Ok(Self::Float),
            6 => Ok(Self::Double),
            7 => Ok(Self::String),
            8 => Ok(Self::Binary),
            9 => Ok(Self::Timestamp),
            10 => Ok(Self::List),
            11 => Ok(Self::Map),
            12 => Ok(Self::Struct),
            13 => Ok(Self::Union),
            14 => Ok(Self::Decimal),
            15 => Ok(Self::Date),
            16 => Ok(Self::Varchar),
            17 => Ok(Self::Char),
            other => Err(IoError::FormatError(format!(
                "unknown ORC type kind: {other}"
            ))),
        }
    }
}

/// ORC column encoding kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnEncodingKind {
    Direct = 0,
    Dictionary = 1,
    DirectV2 = 2,
    DictionaryV2 = 3,
}

impl ColumnEncodingKind {
    pub(crate) fn from_u64(val: u64) -> Result<Self> {
        match val {
            0 => Ok(Self::Direct),
            1 => Ok(Self::Dictionary),
            2 => Ok(Self::DirectV2),
            3 => Ok(Self::DictionaryV2),
            other => Err(IoError::FormatError(format!(
                "unknown ORC column encoding kind: {other}"
            ))),
        }
    }
}

/// ORC stream kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StreamKind {
    Present = 0,
    Data = 1,
    Length = 2,
    DictionaryData = 3,
    DictionaryCount = 4,
    Secondary = 5,
    RowIndex = 6,
}

impl StreamKind {
    pub(crate) fn from_u64(val: u64) -> Result<Self> {
        match val {
            0 => Ok(Self::Present),
            1 => Ok(Self::Data),
            2 => Ok(Self::Length),
            3 => Ok(Self::DictionaryData),
            4 => Ok(Self::DictionaryCount),
            5 => Ok(Self::Secondary),
            6 => Ok(Self::RowIndex),
            other => Err(IoError::FormatError(format!(
                "unknown ORC stream kind: {other}"
            ))),
        }
    }
}

// ---------------------------------------------------------------------------
// PostScript message (field numbers from ORC spec)
// ---------------------------------------------------------------------------

/// ORC PostScript — located at the end of the file.
///
/// Protobuf fields:
///   1: footer_length (uint64)
///   2: compression (CompressionKind enum)
///   3: compression_block_size (uint64)
///   4: version (repeated uint32) — major, minor
///   5: metadata_length (uint64)
///   6: writer_version (uint32)
///   7: magic (string, should be "ORC")
#[derive(Debug, Clone)]
pub struct PostScript {
    pub footer_length: u64,
    pub compression: CompressionKind,
    pub compression_block_size: u64,
    pub metadata_length: u64,
    pub version: Vec<u32>,
    pub magic: String,
}

impl PostScript {
    pub(crate) fn parse(data: &[u8]) -> Result<Self> {
        let mut footer_length: u64 = 0;
        let mut compression = CompressionKind::None;
        let mut compression_block_size: u64 = 262_144; // default 256k
        let mut metadata_length: u64 = 0;
        let mut version = Vec::new();
        let mut magic = String::new();

        let mut pos = 0;
        while pos < data.len() {
            let (tag, tag_len) = decode_tag(&data[pos..])?;
            pos += tag_len;
            match tag.field_number {
                1 => {
                    let (val, n) = decode_varint(&data[pos..])?;
                    pos += n;
                    footer_length = val;
                }
                2 => {
                    let (val, n) = decode_varint(&data[pos..])?;
                    pos += n;
                    compression = CompressionKind::from_u64(val)?;
                }
                3 => {
                    let (val, n) = decode_varint(&data[pos..])?;
                    pos += n;
                    compression_block_size = val;
                }
                4 => {
                    // Repeated uint32 — packed or individual
                    if tag.wire_type == WireType::LengthDelimited {
                        let (len, n) = decode_varint(&data[pos..])?;
                        pos += n;
                        #[allow(clippy::cast_possible_truncation)]
                        let end = pos + len as usize;
                        while pos < end {
                            let (val, n2) = decode_varint(&data[pos..])?;
                            pos += n2;
                            #[allow(clippy::cast_possible_truncation)]
                            version.push(val as u32);
                        }
                    } else {
                        let (val, n) = decode_varint(&data[pos..])?;
                        pos += n;
                        #[allow(clippy::cast_possible_truncation)]
                        version.push(val as u32);
                    }
                }
                5 => {
                    let (val, n) = decode_varint(&data[pos..])?;
                    pos += n;
                    metadata_length = val;
                }
                6 => {
                    // writer_version — skip
                    let n = skip_field(&data[pos..], tag.wire_type)?;
                    pos += n;
                }
                7 => {
                    // magic string (length-delimited)
                    let (len, n) = decode_varint(&data[pos..])?;
                    pos += n;
                    #[allow(clippy::cast_possible_truncation)]
                    let len = len as usize;
                    if pos + len > data.len() {
                        return Err(IoError::FormatError("truncated magic string".into()));
                    }
                    magic = String::from_utf8_lossy(&data[pos..pos + len]).into_owned();
                    pos += len;
                }
                _ => {
                    let n = skip_field(&data[pos..], tag.wire_type)?;
                    pos += n;
                }
            }
        }

        Ok(Self {
            footer_length,
            compression,
            compression_block_size,
            metadata_length,
            version,
            magic,
        })
    }
}

// ---------------------------------------------------------------------------
// StripeInformation
// ---------------------------------------------------------------------------

/// Information about a single stripe in the ORC file.
///
/// Protobuf fields:
///   1: offset (uint64)
///   2: index_length (uint64)
///   3: data_length (uint64)
///   4: footer_length (uint64)
///   5: number_of_rows (uint64)
#[derive(Debug, Clone)]
pub struct StripeInformation {
    pub offset: u64,
    pub index_length: u64,
    pub data_length: u64,
    pub footer_length: u64,
    pub number_of_rows: u64,
}

impl StripeInformation {
    pub(crate) fn parse(data: &[u8]) -> Result<Self> {
        let mut offset: u64 = 0;
        let mut index_length: u64 = 0;
        let mut data_length: u64 = 0;
        let mut footer_length: u64 = 0;
        let mut number_of_rows: u64 = 0;

        let mut pos = 0;
        while pos < data.len() {
            let (tag, tag_len) = decode_tag(&data[pos..])?;
            pos += tag_len;
            match tag.field_number {
                1 => {
                    let (val, n) = decode_varint(&data[pos..])?;
                    pos += n;
                    offset = val;
                }
                2 => {
                    let (val, n) = decode_varint(&data[pos..])?;
                    pos += n;
                    index_length = val;
                }
                3 => {
                    let (val, n) = decode_varint(&data[pos..])?;
                    pos += n;
                    data_length = val;
                }
                4 => {
                    let (val, n) = decode_varint(&data[pos..])?;
                    pos += n;
                    footer_length = val;
                }
                5 => {
                    let (val, n) = decode_varint(&data[pos..])?;
                    pos += n;
                    number_of_rows = val;
                }
                _ => {
                    let n = skip_field(&data[pos..], tag.wire_type)?;
                    pos += n;
                }
            }
        }

        Ok(Self {
            offset,
            index_length,
            data_length,
            footer_length,
            number_of_rows,
        })
    }
}

// ---------------------------------------------------------------------------
// Type
// ---------------------------------------------------------------------------

/// An ORC type descriptor.
///
/// Protobuf fields:
///   1: kind (TypeKind enum)
///   2: subtypes (repeated uint32) — child type IDs
///   3: field_names (repeated string) — for struct types
#[derive(Debug, Clone)]
pub struct OrcType {
    pub kind: TypeKind,
    pub subtypes: Vec<u32>,
    pub field_names: Vec<String>,
}

impl OrcType {
    pub(crate) fn parse(data: &[u8]) -> Result<Self> {
        let mut kind = TypeKind::Boolean; // will be overwritten
        let mut subtypes = Vec::new();
        let mut field_names = Vec::new();

        let mut pos = 0;
        while pos < data.len() {
            let (tag, tag_len) = decode_tag(&data[pos..])?;
            pos += tag_len;
            match tag.field_number {
                1 => {
                    let (val, n) = decode_varint(&data[pos..])?;
                    pos += n;
                    kind = TypeKind::from_u64(val)?;
                }
                2 => {
                    if tag.wire_type == WireType::LengthDelimited {
                        let (len, n) = decode_varint(&data[pos..])?;
                        pos += n;
                        #[allow(clippy::cast_possible_truncation)]
                        let end = pos + len as usize;
                        while pos < end {
                            let (val, n2) = decode_varint(&data[pos..])?;
                            pos += n2;
                            #[allow(clippy::cast_possible_truncation)]
                            subtypes.push(val as u32);
                        }
                    } else {
                        let (val, n) = decode_varint(&data[pos..])?;
                        pos += n;
                        #[allow(clippy::cast_possible_truncation)]
                        subtypes.push(val as u32);
                    }
                }
                3 => {
                    let (len, n) = decode_varint(&data[pos..])?;
                    pos += n;
                    #[allow(clippy::cast_possible_truncation)]
                    let len = len as usize;
                    if pos + len > data.len() {
                        return Err(IoError::FormatError("truncated field name".into()));
                    }
                    let name = String::from_utf8_lossy(&data[pos..pos + len]).into_owned();
                    pos += len;
                    field_names.push(name);
                }
                _ => {
                    let n = skip_field(&data[pos..], tag.wire_type)?;
                    pos += n;
                }
            }
        }

        Ok(Self {
            kind,
            subtypes,
            field_names,
        })
    }
}

// ---------------------------------------------------------------------------
// Footer
// ---------------------------------------------------------------------------

/// ORC file Footer.
///
/// Protobuf fields:
///   1: header_length (uint64)
///   2: content_length (uint64)
///   3: stripes (repeated StripeInformation)
///   4: types (repeated Type)
///   5: metadata (repeated UserMetadataItem) — skipped
///   6: number_of_rows (uint64)
///   7: statistics (repeated ColumnStatistics) — skipped
///   8: row_index_stride (uint32)
#[derive(Debug, Clone)]
pub struct Footer {
    pub header_length: u64,
    pub content_length: u64,
    pub stripes: Vec<StripeInformation>,
    pub types: Vec<OrcType>,
    pub number_of_rows: u64,
    pub row_index_stride: u32,
}

impl Footer {
    pub(crate) fn parse(data: &[u8]) -> Result<Self> {
        let mut header_length: u64 = 0;
        let mut content_length: u64 = 0;
        let mut stripes = Vec::new();
        let mut types = Vec::new();
        let mut number_of_rows: u64 = 0;
        let mut row_index_stride: u32 = 0;

        let mut pos = 0;
        while pos < data.len() {
            let (tag, tag_len) = decode_tag(&data[pos..])?;
            pos += tag_len;
            match tag.field_number {
                1 => {
                    let (val, n) = decode_varint(&data[pos..])?;
                    pos += n;
                    header_length = val;
                }
                2 => {
                    let (val, n) = decode_varint(&data[pos..])?;
                    pos += n;
                    content_length = val;
                }
                3 => {
                    // length-delimited StripeInformation
                    let (len, n) = decode_varint(&data[pos..])?;
                    pos += n;
                    #[allow(clippy::cast_possible_truncation)]
                    let len = len as usize;
                    let stripe = StripeInformation::parse(&data[pos..pos + len])?;
                    pos += len;
                    stripes.push(stripe);
                }
                4 => {
                    // length-delimited Type
                    let (len, n) = decode_varint(&data[pos..])?;
                    pos += n;
                    #[allow(clippy::cast_possible_truncation)]
                    let len = len as usize;
                    let typ = OrcType::parse(&data[pos..pos + len])?;
                    pos += len;
                    types.push(typ);
                }
                5 | 7 => {
                    // metadata or statistics — skip
                    let n = skip_field(&data[pos..], tag.wire_type)?;
                    pos += n;
                }
                6 => {
                    let (val, n) = decode_varint(&data[pos..])?;
                    pos += n;
                    number_of_rows = val;
                }
                8 => {
                    let (val, n) = decode_varint(&data[pos..])?;
                    pos += n;
                    #[allow(clippy::cast_possible_truncation)]
                    {
                        row_index_stride = val as u32;
                    }
                }
                _ => {
                    let n = skip_field(&data[pos..], tag.wire_type)?;
                    pos += n;
                }
            }
        }

        Ok(Self {
            header_length,
            content_length,
            stripes,
            types,
            number_of_rows,
            row_index_stride,
        })
    }
}

// ---------------------------------------------------------------------------
// Stream (inside StripeFooter)
// ---------------------------------------------------------------------------

/// An ORC stream descriptor (inside a stripe footer).
///
/// Protobuf fields:
///   1: kind (StreamKind enum)
///   2: column (uint32)
///   3: length (uint64)
#[derive(Debug, Clone)]
pub struct Stream {
    pub kind: StreamKind,
    pub column: u32,
    pub length: u64,
}

impl Stream {
    pub(crate) fn parse(data: &[u8]) -> Result<Self> {
        let mut kind = StreamKind::Data;
        let mut column: u32 = 0;
        let mut length: u64 = 0;

        let mut pos = 0;
        while pos < data.len() {
            let (tag, tag_len) = decode_tag(&data[pos..])?;
            pos += tag_len;
            match tag.field_number {
                1 => {
                    let (val, n) = decode_varint(&data[pos..])?;
                    pos += n;
                    kind = StreamKind::from_u64(val)?;
                }
                2 => {
                    let (val, n) = decode_varint(&data[pos..])?;
                    pos += n;
                    #[allow(clippy::cast_possible_truncation)]
                    {
                        column = val as u32;
                    }
                }
                3 => {
                    let (val, n) = decode_varint(&data[pos..])?;
                    pos += n;
                    length = val;
                }
                _ => {
                    let n = skip_field(&data[pos..], tag.wire_type)?;
                    pos += n;
                }
            }
        }

        Ok(Self {
            kind,
            column,
            length,
        })
    }
}

// ---------------------------------------------------------------------------
// ColumnEncoding (inside StripeFooter)
// ---------------------------------------------------------------------------

/// Column encoding descriptor.
///
/// Protobuf fields:
///   1: kind (ColumnEncodingKind enum)
///   2: dictionary_size (uint32)
#[derive(Debug, Clone)]
pub struct ColumnEncoding {
    pub kind: ColumnEncodingKind,
    pub dictionary_size: u32,
}

impl ColumnEncoding {
    pub(crate) fn parse(data: &[u8]) -> Result<Self> {
        let mut kind = ColumnEncodingKind::Direct;
        let mut dictionary_size: u32 = 0;

        let mut pos = 0;
        while pos < data.len() {
            let (tag, tag_len) = decode_tag(&data[pos..])?;
            pos += tag_len;
            match tag.field_number {
                1 => {
                    let (val, n) = decode_varint(&data[pos..])?;
                    pos += n;
                    kind = ColumnEncodingKind::from_u64(val)?;
                }
                2 => {
                    let (val, n) = decode_varint(&data[pos..])?;
                    pos += n;
                    #[allow(clippy::cast_possible_truncation)]
                    {
                        dictionary_size = val as u32;
                    }
                }
                _ => {
                    let n = skip_field(&data[pos..], tag.wire_type)?;
                    pos += n;
                }
            }
        }

        Ok(Self {
            kind,
            dictionary_size,
        })
    }
}

// ---------------------------------------------------------------------------
// StripeFooter
// ---------------------------------------------------------------------------

/// ORC Stripe Footer.
///
/// Protobuf fields:
///   1: streams (repeated Stream)
///   2: columns (repeated ColumnEncoding)
#[derive(Debug, Clone)]
pub struct StripeFooter {
    pub streams: Vec<Stream>,
    pub columns: Vec<ColumnEncoding>,
}

impl StripeFooter {
    pub(crate) fn parse(data: &[u8]) -> Result<Self> {
        let mut streams = Vec::new();
        let mut columns = Vec::new();

        let mut pos = 0;
        while pos < data.len() {
            let (tag, tag_len) = decode_tag(&data[pos..])?;
            pos += tag_len;
            match tag.field_number {
                1 => {
                    let (len, n) = decode_varint(&data[pos..])?;
                    pos += n;
                    #[allow(clippy::cast_possible_truncation)]
                    let len = len as usize;
                    let stream = Stream::parse(&data[pos..pos + len])?;
                    pos += len;
                    streams.push(stream);
                }
                2 => {
                    let (len, n) = decode_varint(&data[pos..])?;
                    pos += n;
                    #[allow(clippy::cast_possible_truncation)]
                    let len = len as usize;
                    let encoding = ColumnEncoding::parse(&data[pos..pos + len])?;
                    pos += len;
                    columns.push(encoding);
                }
                _ => {
                    let n = skip_field(&data[pos..], tag.wire_type)?;
                    pos += n;
                }
            }
        }

        Ok(Self { streams, columns })
    }
}

// ---------------------------------------------------------------------------
// Protobuf encoding helpers (for writing)
// ---------------------------------------------------------------------------

/// Encode a varint into the buffer.
pub(crate) fn encode_varint(buf: &mut Vec<u8>, mut val: u64) {
    loop {
        if val < 0x80 {
            buf.push(val as u8);
            break;
        }
        buf.push((val as u8 & 0x7F) | 0x80);
        val >>= 7;
    }
}

/// Encode a field tag (field_number << 3 | wire_type).
pub(crate) fn encode_tag(buf: &mut Vec<u8>, field_number: u32, wire_type: u32) {
    encode_varint(buf, u64::from(field_number) << 3 | u64::from(wire_type));
}

/// Encode a varint field: tag + varint value.
pub(crate) fn encode_varint_field(buf: &mut Vec<u8>, field_number: u32, val: u64) {
    if val != 0 {
        encode_tag(buf, field_number, 0); // wire type 0 = varint
        encode_varint(buf, val);
    }
}

/// Encode a varint field even if the value is 0.
pub(crate) fn encode_varint_field_always(buf: &mut Vec<u8>, field_number: u32, val: u64) {
    encode_tag(buf, field_number, 0);
    encode_varint(buf, val);
}

/// Encode a length-delimited field: tag + length + bytes.
pub(crate) fn encode_bytes_field(buf: &mut Vec<u8>, field_number: u32, data: &[u8]) {
    encode_tag(buf, field_number, 2); // wire type 2 = length-delimited
    encode_varint(buf, data.len() as u64);
    buf.extend_from_slice(data);
}

/// Encode a string field (same as bytes).
pub(crate) fn encode_string_field(buf: &mut Vec<u8>, field_number: u32, s: &str) {
    encode_bytes_field(buf, field_number, s.as_bytes());
}

// ---------------------------------------------------------------------------
// Serialization helpers for each message type
// ---------------------------------------------------------------------------

impl PostScript {
    /// Serialize to protobuf bytes.
    pub(crate) fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        encode_varint_field(&mut buf, 1, self.footer_length);
        encode_varint_field_always(&mut buf, 2, self.compression as u64);
        encode_varint_field(&mut buf, 3, self.compression_block_size);
        // version — write each element individually
        for &v in &self.version {
            encode_varint_field_always(&mut buf, 4, u64::from(v));
        }
        encode_varint_field(&mut buf, 5, self.metadata_length);
        if !self.magic.is_empty() {
            encode_string_field(&mut buf, 7, &self.magic);
        }
        buf
    }
}

impl StripeInformation {
    /// Serialize to protobuf bytes.
    pub(crate) fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        encode_varint_field(&mut buf, 1, self.offset);
        encode_varint_field(&mut buf, 2, self.index_length);
        encode_varint_field(&mut buf, 3, self.data_length);
        encode_varint_field(&mut buf, 4, self.footer_length);
        encode_varint_field(&mut buf, 5, self.number_of_rows);
        buf
    }
}

impl OrcType {
    /// Serialize to protobuf bytes.
    pub(crate) fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        encode_varint_field_always(&mut buf, 1, self.kind as u64);
        for &st in &self.subtypes {
            encode_varint_field_always(&mut buf, 2, u64::from(st));
        }
        for name in &self.field_names {
            encode_string_field(&mut buf, 3, name);
        }
        buf
    }
}

impl Stream {
    /// Serialize to protobuf bytes.
    pub(crate) fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        encode_varint_field_always(&mut buf, 1, self.kind as u64);
        encode_varint_field_always(&mut buf, 2, u64::from(self.column));
        encode_varint_field(&mut buf, 3, self.length);
        buf
    }
}

impl ColumnEncoding {
    /// Serialize to protobuf bytes.
    pub(crate) fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        encode_varint_field_always(&mut buf, 1, self.kind as u64);
        if self.dictionary_size > 0 {
            encode_varint_field(&mut buf, 2, u64::from(self.dictionary_size));
        }
        buf
    }
}

impl StripeFooter {
    /// Serialize to protobuf bytes.
    pub(crate) fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        for stream in &self.streams {
            let encoded = stream.encode();
            encode_bytes_field(&mut buf, 1, &encoded);
        }
        for col in &self.columns {
            let encoded = col.encode();
            encode_bytes_field(&mut buf, 2, &encoded);
        }
        buf
    }
}

impl Footer {
    /// Serialize to protobuf bytes.
    pub(crate) fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        encode_varint_field(&mut buf, 1, self.header_length);
        encode_varint_field(&mut buf, 2, self.content_length);
        for stripe in &self.stripes {
            let encoded = stripe.encode();
            encode_bytes_field(&mut buf, 3, &encoded);
        }
        for typ in &self.types {
            let encoded = typ.encode();
            encode_bytes_field(&mut buf, 4, &encoded);
        }
        encode_varint_field(&mut buf, 6, self.number_of_rows);
        if self.row_index_stride > 0 {
            encode_varint_field(&mut buf, 8, u64::from(self.row_index_stride));
        }
        buf
    }
}
