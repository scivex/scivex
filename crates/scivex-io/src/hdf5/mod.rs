//! HDF5 file format support (from-scratch binary parser/writer).
//!
//! This module implements a minimal subset of the HDF5 specification:
//! - Superblock v0
//! - B-tree v1 and local heap for group symbol table navigation
//! - Object headers with dataspace and datatype messages
//! - Contiguous data storage layout
//!
//! Supported element types: `f32`, `f64`, `i32`, `i64`.
//!
//! # Limitations
//!
//! - Only superblock version 0 is supported
//! - Only contiguous (non-chunked) storage
//! - No compression filters
//! - No variable-length types or compound types
//! - No virtual datasets

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom, Write};

use scivex_core::{Scalar, Tensor};

use crate::error::{IoError, Result};

// ---------------------------------------------------------------------------
// HDF5 Constants
// ---------------------------------------------------------------------------

/// HDF5 file signature (magic bytes).
const HDF5_SIGNATURE: [u8; 8] = [0x89, b'H', b'D', b'F', b'\r', b'\n', 0x1a, b'\n'];

/// Object header signature (version 1 has no explicit signature — just starts
/// with version byte).
const OBJECT_HEADER_V1: u8 = 1;

/// B-tree v1 signature.
const BTREE_SIGNATURE: [u8; 4] = *b"TREE";

/// Local heap signature.
const HEAP_SIGNATURE: [u8; 4] = *b"HEAP";

/// Symbol-table node signature.
const SNOD_SIGNATURE: [u8; 4] = *b"SNOD";

/// Dataspace message type.
const MSG_DATASPACE: u16 = 0x0001;

/// Datatype message type.
const MSG_DATATYPE: u16 = 0x0003;

/// Data layout message type.
const MSG_DATA_LAYOUT: u16 = 0x0008;

/// Symbol table message type.
const MSG_SYMBOL_TABLE: u16 = 0x0011;

// ---------------------------------------------------------------------------
// HDF5 data type class constants
// ---------------------------------------------------------------------------

const DTYPE_CLASS_FIXED_POINT: u8 = 0; // integer
const DTYPE_CLASS_FLOATING_POINT: u8 = 1; // float

// ---------------------------------------------------------------------------
// Internal helper trait for byte encoding
// ---------------------------------------------------------------------------

/// Trait for types that can be read/written as raw bytes in HDF5 datasets.
pub trait Hdf5Scalar: Scalar {
    /// HDF5 datatype class (0 = fixed-point/integer, 1 = floating-point).
    fn hdf5_class() -> u8;
    /// Size in bytes.
    fn hdf5_size() -> usize;
    /// Encode to little-endian bytes.
    fn to_le_bytes_vec(self) -> Vec<u8>;
    /// Decode from little-endian bytes. Panics if slice is wrong length.
    fn from_le_bytes_slice(bytes: &[u8]) -> Self;
    /// HDF5 datatype bit-field flags (byte-order, sign, etc.).
    fn hdf5_class_bits() -> u32;
}

impl Hdf5Scalar for f32 {
    fn hdf5_class() -> u8 {
        DTYPE_CLASS_FLOATING_POINT
    }
    fn hdf5_size() -> usize {
        4
    }
    fn to_le_bytes_vec(self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }
    fn from_le_bytes_slice(bytes: &[u8]) -> Self {
        let arr: [u8; 4] = [bytes[0], bytes[1], bytes[2], bytes[3]];
        f32::from_le_bytes(arr)
    }
    fn hdf5_class_bits() -> u32 {
        // byte-order=little-endian(0), padding=0, mantissa/exponent per IEEE 754
        0x00_00_00_00
    }
}

impl Hdf5Scalar for f64 {
    fn hdf5_class() -> u8 {
        DTYPE_CLASS_FLOATING_POINT
    }
    fn hdf5_size() -> usize {
        8
    }
    fn to_le_bytes_vec(self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }
    fn from_le_bytes_slice(bytes: &[u8]) -> Self {
        let mut arr = [0u8; 8];
        arr.copy_from_slice(&bytes[..8]);
        f64::from_le_bytes(arr)
    }
    fn hdf5_class_bits() -> u32 {
        0x00_00_00_00
    }
}

impl Hdf5Scalar for i32 {
    fn hdf5_class() -> u8 {
        DTYPE_CLASS_FIXED_POINT
    }
    fn hdf5_size() -> usize {
        4
    }
    fn to_le_bytes_vec(self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }
    fn from_le_bytes_slice(bytes: &[u8]) -> Self {
        let arr: [u8; 4] = [bytes[0], bytes[1], bytes[2], bytes[3]];
        i32::from_le_bytes(arr)
    }
    fn hdf5_class_bits() -> u32 {
        // byte-order=little-endian(0), signed(0x08)
        0x00_00_00_08
    }
}

impl Hdf5Scalar for i64 {
    fn hdf5_class() -> u8 {
        DTYPE_CLASS_FIXED_POINT
    }
    fn hdf5_size() -> usize {
        8
    }
    fn to_le_bytes_vec(self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }
    fn from_le_bytes_slice(bytes: &[u8]) -> Self {
        let mut arr = [0u8; 8];
        arr.copy_from_slice(&bytes[..8]);
        i64::from_le_bytes(arr)
    }
    fn hdf5_class_bits() -> u32 {
        0x00_00_00_08
    }
}

// ---------------------------------------------------------------------------
// Superblock
// ---------------------------------------------------------------------------

/// Parsed HDF5 superblock (version 0).
#[derive(Debug)]
struct Superblock {
    /// Offset size in bytes (4 or 8).
    size_of_offsets: u8,
    /// Length size in bytes (4 or 8).
    size_of_lengths: u8,
    /// Address of the root group object header.
    root_group_object_header_address: u64,
}

// ---------------------------------------------------------------------------
// Reading helpers
// ---------------------------------------------------------------------------

fn read_u8<R: Read>(r: &mut R) -> Result<u8> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn read_u16_le<R: Read>(r: &mut R) -> Result<u16> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf)?;
    Ok(u16::from_le_bytes(buf))
}

fn read_u32_le<R: Read>(r: &mut R) -> Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64_le<R: Read>(r: &mut R) -> Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

/// Read an offset of `size` bytes (4 or 8) as u64.
fn read_offset<R: Read>(r: &mut R, size: u8) -> Result<u64> {
    match size {
        4 => read_u32_le(r).map(u64::from),
        8 => read_u64_le(r),
        _ => Err(IoError::FormatError(format!(
            "unsupported offset size: {size}"
        ))),
    }
}

/// Read a length of `size` bytes (4 or 8) as u64.
fn read_length<R: Read>(r: &mut R, size: u8) -> Result<u64> {
    read_offset(r, size)
}

fn read_bytes<R: Read>(r: &mut R, n: usize) -> Result<Vec<u8>> {
    let mut buf = vec![0u8; n];
    r.read_exact(&mut buf)?;
    Ok(buf)
}

// ---------------------------------------------------------------------------
// Superblock parser
// ---------------------------------------------------------------------------

fn read_superblock<R: Read + Seek>(r: &mut R) -> Result<Superblock> {
    r.seek(SeekFrom::Start(0))?;

    // Signature
    let sig = read_bytes(r, 8)?;
    if sig != HDF5_SIGNATURE {
        return Err(IoError::FormatError(
            "not a valid HDF5 file (bad signature)".into(),
        ));
    }

    let version = read_u8(r)?;
    if version != 0 {
        return Err(IoError::FormatError(format!(
            "unsupported superblock version: {version} (only v0 supported)"
        )));
    }

    // Free-space storage version, root group symbol table entry version,
    // reserved byte
    let _free_space_version = read_u8(r)?;
    let _root_group_stev = read_u8(r)?;
    let _reserved = read_u8(r)?;

    // Shared header message format version, size of offsets, size of lengths
    let _shared_header_version = read_u8(r)?;
    let size_of_offsets = read_u8(r)?;
    let size_of_lengths = read_u8(r)?;

    // Reserved byte
    let _reserved2 = read_u8(r)?;

    // Group leaf node K, group internal node K
    let _group_leaf_k = read_u16_le(r)?;
    let _group_internal_k = read_u16_le(r)?;

    // Consistency flags
    let _consistency_flags = read_u32_le(r)?;

    // Base address, address of file free-space info, end-of-file address,
    // driver information block address
    let _base_address = read_offset(r, size_of_offsets)?;
    let _freespace_address = read_offset(r, size_of_offsets)?;
    let _eof_address = read_offset(r, size_of_offsets)?;
    let _driver_info_address = read_offset(r, size_of_offsets)?;

    // Root group symbol table entry
    // Link name offset (in group's local heap)
    let _link_name_offset = read_offset(r, size_of_offsets)?;
    // Object header address
    let root_group_object_header_address = read_offset(r, size_of_offsets)?;
    // Cache type and reserved
    let _cache_type = read_u32_le(r)?;
    let _reserved3 = read_u32_le(r)?;
    // Scratch-pad space (16 bytes for group entries)
    // B-tree address and local heap address
    let _scratch = read_bytes(r, 16)?;

    Ok(Superblock {
        size_of_offsets,
        size_of_lengths,
        root_group_object_header_address,
    })
}

// ---------------------------------------------------------------------------
// Object header parser
// ---------------------------------------------------------------------------

/// Parsed information from an object header.
#[derive(Debug, Default)]
struct ObjectHeaderInfo {
    /// Dataset dimensions (from dataspace message).
    dimensions: Vec<u64>,
    /// Datatype class (0=integer, 1=float).
    datatype_class: u8,
    /// Datatype size in bytes.
    datatype_size: u32,
    /// Data address (contiguous layout).
    data_address: u64,
    /// Data size in bytes.
    data_size: u64,
    /// Symbol table B-tree address (for groups).
    btree_address: Option<u64>,
    /// Symbol table local heap address (for groups).
    local_heap_address: Option<u64>,
    /// Datatype class bit-field (for sign info).
    datatype_class_bits: u32,
}

fn parse_object_header<R: Read + Seek>(
    r: &mut R,
    addr: u64,
    sb: &Superblock,
) -> Result<ObjectHeaderInfo> {
    r.seek(SeekFrom::Start(addr))?;

    let version = read_u8(r)?;
    if version != OBJECT_HEADER_V1 {
        return Err(IoError::FormatError(format!(
            "unsupported object header version: {version}"
        )));
    }

    // Reserved byte
    let _reserved = read_u8(r)?;
    // Number of header messages
    let num_messages = read_u16_le(r)?;
    // Object reference count
    let _ref_count = read_u32_le(r)?;
    // Object header size (total size of all header messages)
    let _header_size = read_u32_le(r)?;

    let mut info = ObjectHeaderInfo::default();

    for _ in 0..num_messages {
        let msg_type = read_u16_le(r)?;
        let msg_size = read_u16_le(r)?;
        let msg_flags = read_u8(r)?;
        let _ = msg_flags;
        // 3 reserved bytes
        let _reserved = read_bytes(r, 3)?;

        let msg_start = r.stream_position()?;

        match msg_type {
            MSG_DATASPACE => {
                parse_dataspace_msg(r, &mut info)?;
            }
            MSG_DATATYPE => {
                parse_datatype_msg(r, &mut info)?;
            }
            MSG_DATA_LAYOUT => {
                parse_data_layout_msg(r, &mut info, sb)?;
            }
            MSG_SYMBOL_TABLE => {
                let btree_addr = read_offset(r, sb.size_of_offsets)?;
                let heap_addr = read_offset(r, sb.size_of_offsets)?;
                info.btree_address = Some(btree_addr);
                info.local_heap_address = Some(heap_addr);
            }
            _ => {
                // Skip unknown messages
            }
        }

        // Seek to end of this message
        r.seek(SeekFrom::Start(msg_start + u64::from(msg_size)))?;
    }

    Ok(info)
}

fn parse_dataspace_msg<R: Read>(r: &mut R, info: &mut ObjectHeaderInfo) -> Result<()> {
    let version = read_u8(r)?;
    let ndims = read_u8(r)?;
    let flags = read_u8(r)?;

    if version == 1 {
        // 5 reserved bytes
        let _reserved = read_bytes(r, 5)?;
    }

    let mut dims = Vec::with_capacity(ndims as usize);
    for _ in 0..ndims {
        dims.push(read_u64_le(r)?);
    }

    // If max-dims flag set, read max dims but discard
    if version == 1 || (flags & 0x01) != 0 {
        for _ in 0..ndims {
            let _max = read_u64_le(r)?;
        }
    }

    info.dimensions = dims;
    Ok(())
}

fn parse_datatype_msg<R: Read>(r: &mut R, info: &mut ObjectHeaderInfo) -> Result<()> {
    // Class and version are packed in first 4 bytes
    let class_and_version = read_u32_le(r)?;
    let class = (class_and_version & 0x0F) as u8;
    let class_bits = class_and_version >> 8;
    let size = read_u32_le(r)?;

    info.datatype_class = class;
    info.datatype_size = size;
    info.datatype_class_bits = class_bits;

    Ok(())
}

fn parse_data_layout_msg<R: Read>(
    r: &mut R,
    info: &mut ObjectHeaderInfo,
    sb: &Superblock,
) -> Result<()> {
    let version = read_u8(r)?;

    // We only support contiguous layout (class 1) or compact (class 0) in v3,
    // but v1/v2 format is different.
    if version <= 2 {
        let ndims = read_u8(r)?;
        let layout_class = read_u8(r)?;

        // reserved bytes
        let _reserved = read_bytes(r, 5)?;

        if layout_class == 1 {
            // Contiguous: data address
            let data_address = read_offset(r, sb.size_of_offsets)?;
            info.data_address = data_address;

            // dimension sizes (ndims - 1 dimensions, each 4 bytes) — these
            // encode the data size as a product
            let mut total: u64 = 1;
            for _ in 0..(ndims.saturating_sub(1)) {
                total = total.saturating_mul(u64::from(read_u32_le(r)?));
            }
            info.data_size = total;
        }
    } else if version == 3 {
        // v3 format: version(1) + layout_class(1) + class-specific (no ndims)
        let layout_class = read_u8(r)?;

        match layout_class {
            1 => {
                // Contiguous storage (v3)
                let data_address = read_offset(r, sb.size_of_offsets)?;
                let data_size = read_length(r, sb.size_of_lengths)?;
                info.data_address = data_address;
                info.data_size = data_size;
            }
            _ => {
                return Err(IoError::FormatError(format!(
                    "unsupported data layout class: {layout_class}"
                )));
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Group navigation (B-tree v1 + local heap + SNOD)
// ---------------------------------------------------------------------------

/// An entry in a symbol table (group member).
#[derive(Debug)]
struct SymbolTableEntry {
    name: String,
    object_header_address: u64,
}

/// Read all entries from a group given its B-tree and local heap addresses.
fn read_group_entries<R: Read + Seek>(
    r: &mut R,
    btree_addr: u64,
    heap_addr: u64,
    sb: &Superblock,
) -> Result<Vec<SymbolTableEntry>> {
    // First read the local heap so we can resolve name offsets
    let heap_data = read_local_heap(r, heap_addr, sb)?;

    // Then traverse the B-tree to find SNOD nodes
    let mut entries = Vec::new();
    read_btree_node(r, btree_addr, sb, &heap_data, &mut entries)?;

    Ok(entries)
}

/// Read the local heap data segment.
fn read_local_heap<R: Read + Seek>(r: &mut R, addr: u64, sb: &Superblock) -> Result<Vec<u8>> {
    r.seek(SeekFrom::Start(addr))?;

    let sig = read_bytes(r, 4)?;
    if sig != HEAP_SIGNATURE {
        return Err(IoError::FormatError("invalid local heap signature".into()));
    }

    let _version = read_u8(r)?;
    // 3 reserved bytes
    let _reserved = read_bytes(r, 3)?;

    let data_segment_size = read_length(r, sb.size_of_lengths)?;
    let _free_list_offset = read_length(r, sb.size_of_lengths)?;
    let data_segment_address = read_offset(r, sb.size_of_offsets)?;

    r.seek(SeekFrom::Start(data_segment_address))?;
    #[allow(clippy::cast_possible_truncation)]
    let size = data_segment_size as usize;
    read_bytes(r, size)
}

/// Read a name from the local heap data at the given offset.
fn read_heap_name(heap_data: &[u8], offset: usize) -> String {
    let end = heap_data[offset..]
        .iter()
        .position(|&b| b == 0)
        .unwrap_or(heap_data.len() - offset);
    String::from_utf8_lossy(&heap_data[offset..offset + end]).into_owned()
}

/// Recursively read a B-tree v1 node.
fn read_btree_node<R: Read + Seek>(
    r: &mut R,
    addr: u64,
    sb: &Superblock,
    heap_data: &[u8],
    entries: &mut Vec<SymbolTableEntry>,
) -> Result<()> {
    r.seek(SeekFrom::Start(addr))?;

    let sig = read_bytes(r, 4)?;
    if sig != BTREE_SIGNATURE {
        return Err(IoError::FormatError("invalid B-tree signature".into()));
    }

    let node_type = read_u8(r)?;
    let node_level = read_u8(r)?;
    let entries_used = read_u16_le(r)?;

    let _left_sibling = read_offset(r, sb.size_of_offsets)?;
    let _right_sibling = read_offset(r, sb.size_of_offsets)?;

    if node_type != 0 {
        return Err(IoError::FormatError(format!(
            "unsupported B-tree node type: {node_type}"
        )));
    }

    if node_level == 0 {
        // Leaf node: children are SNOD addresses
        for _ in 0..entries_used {
            // Key: object header address size worth of bytes
            let _key = read_offset(r, sb.size_of_offsets)?;
            let child_addr = read_offset(r, sb.size_of_offsets)?;

            // Save position in B-tree before seeking to SNOD
            let btree_pos = r.stream_position()?;
            read_snod(r, child_addr, sb, heap_data, entries)?;
            // Restore position in B-tree
            r.seek(SeekFrom::Start(btree_pos))?;
        }
        // One extra key at the end
        let _last_key = read_offset(r, sb.size_of_offsets)?;
    } else {
        // Internal node: recurse into children
        let mut child_addrs = Vec::with_capacity(entries_used as usize);
        for _ in 0..entries_used {
            let _key = read_offset(r, sb.size_of_offsets)?;
            let child = read_offset(r, sb.size_of_offsets)?;
            child_addrs.push(child);
        }
        let _last_key = read_offset(r, sb.size_of_offsets)?;

        for child in child_addrs {
            read_btree_node(r, child, sb, heap_data, entries)?;
        }
    }

    Ok(())
}

/// Read a symbol-table node (SNOD).
fn read_snod<R: Read + Seek>(
    r: &mut R,
    addr: u64,
    sb: &Superblock,
    heap_data: &[u8],
    entries: &mut Vec<SymbolTableEntry>,
) -> Result<()> {
    r.seek(SeekFrom::Start(addr))?;

    let sig = read_bytes(r, 4)?;
    if sig != SNOD_SIGNATURE {
        return Err(IoError::FormatError("invalid SNOD signature".into()));
    }

    let _version = read_u8(r)?;
    let _reserved = read_u8(r)?;
    let num_symbols = read_u16_le(r)?;

    for _ in 0..num_symbols {
        let link_name_offset = read_offset(r, sb.size_of_offsets)?;
        let obj_header_addr = read_offset(r, sb.size_of_offsets)?;
        let _cache_type = read_u32_le(r)?;
        let _reserved2 = read_u32_le(r)?;
        let _scratch_pad = read_bytes(r, 16)?;

        #[allow(clippy::cast_possible_truncation)]
        let name = read_heap_name(heap_data, link_name_offset as usize);
        if !name.is_empty() {
            entries.push(SymbolTableEntry {
                name,
                object_header_address: obj_header_addr,
            });
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Dataset resolution: navigate path to find the dataset object header
// ---------------------------------------------------------------------------

fn resolve_dataset_path<R: Read + Seek>(
    r: &mut R,
    sb: &Superblock,
    root_addr: u64,
    path: &str,
) -> Result<u64> {
    let parts: Vec<&str> = path
        .trim_start_matches('/')
        .split('/')
        .filter(|s| !s.is_empty())
        .collect();

    if parts.is_empty() {
        return Err(IoError::FormatError("empty dataset path".into()));
    }

    let mut current_addr = root_addr;

    for (i, part) in parts.iter().enumerate() {
        let info = parse_object_header(r, current_addr, sb)?;

        let (Some(btree_addr), Some(heap_addr)) = (info.btree_address, info.local_heap_address)
        else {
            return Err(IoError::FormatError(format!(
                "path component '{part}' is not a group"
            )));
        };

        let group_entries = read_group_entries(r, btree_addr, heap_addr, sb)?;

        let target = group_entries.iter().find(|e| e.name == *part);

        match target {
            Some(entry) => {
                if i == parts.len() - 1 {
                    return Ok(entry.object_header_address);
                }
                current_addr = entry.object_header_address;
            }
            None => {
                return Err(IoError::FormatError(format!(
                    "dataset or group '{part}' not found"
                )));
            }
        }
    }

    Err(IoError::FormatError("empty dataset path".into()))
}

// ---------------------------------------------------------------------------
// Read API
// ---------------------------------------------------------------------------

/// Read a dataset from an HDF5 file as a [`Tensor<T>`].
///
/// The `dataset_path` uses `/`-separated components, e.g., `"my_dataset"` or
/// `"group1/nested_dataset"`.
///
/// Only contiguous storage with fixed-size numeric types (`f32`, `f64`, `i32`,
/// `i64`) is supported.
///
/// # Errors
///
/// Returns [`IoError::FormatError`] if the file is not valid HDF5, the path
/// does not exist, or the datatype cannot be decoded to `T`.
pub fn read_hdf5_dataset<T: Hdf5Scalar, R: Read + Seek>(
    reader: &mut R,
    dataset_path: &str,
) -> Result<Tensor<T>> {
    let sb = read_superblock(reader)?;
    let dataset_addr = resolve_dataset_path(
        reader,
        &sb,
        sb.root_group_object_header_address,
        dataset_path,
    )?;
    let info = parse_object_header(reader, dataset_addr, &sb)?;

    // Validate datatype compatibility
    validate_datatype::<T>(&info)?;

    let shape: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
    let numel: usize = shape.iter().product();

    if info.data_address == 0 || info.data_address == u64::MAX {
        return Err(IoError::FormatError(
            "dataset has no contiguous data address".into(),
        ));
    }

    reader.seek(SeekFrom::Start(info.data_address))?;

    let elem_size = T::hdf5_size();
    let raw = read_bytes(reader, numel * elem_size)?;

    let mut data = Vec::with_capacity(numel);
    for i in 0..numel {
        let start = i * elem_size;
        data.push(T::from_le_bytes_slice(&raw[start..start + elem_size]));
    }

    let tensor = Tensor::from_vec(data, shape)
        .map_err(|e| IoError::FormatError(format!("failed to create tensor: {e}")))?;

    Ok(tensor)
}

fn validate_datatype<T: Hdf5Scalar>(info: &ObjectHeaderInfo) -> Result<()> {
    if info.datatype_class != T::hdf5_class() {
        return Err(IoError::FormatError(format!(
            "datatype class mismatch: file has class {}, expected {}",
            info.datatype_class,
            T::hdf5_class()
        )));
    }
    if info.datatype_size != T::hdf5_size() as u32 {
        return Err(IoError::FormatError(format!(
            "datatype size mismatch: file has {} bytes, expected {}",
            info.datatype_size,
            T::hdf5_size()
        )));
    }
    Ok(())
}

/// List all dataset paths in an HDF5 file.
///
/// Returns paths relative to the root group, e.g. `["dataset1", "group/ds2"]`.
pub fn list_hdf5_datasets<R: Read + Seek>(reader: &mut R) -> Result<Vec<String>> {
    let sb = read_superblock(reader)?;
    let mut datasets = Vec::new();
    collect_datasets(
        reader,
        &sb,
        sb.root_group_object_header_address,
        String::new(),
        &mut datasets,
    )?;
    Ok(datasets)
}

fn collect_datasets<R: Read + Seek>(
    r: &mut R,
    sb: &Superblock,
    addr: u64,
    prefix: String,
    datasets: &mut Vec<String>,
) -> Result<()> {
    let info = parse_object_header(r, addr, sb)?;

    match (info.btree_address, info.local_heap_address) {
        (Some(btree), Some(heap)) => {
            // This is a group — enumerate its children
            let entries = read_group_entries(r, btree, heap, sb)?;
            for entry in entries {
                let path = if prefix.is_empty() {
                    entry.name.clone()
                } else {
                    format!("{prefix}/{}", entry.name)
                };
                collect_datasets(r, sb, entry.object_header_address, path, datasets)?;
            }
        }
        _ => {
            // This is a dataset (or other object) — add it if it has data
            if !prefix.is_empty() {
                datasets.push(prefix);
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Write API
// ---------------------------------------------------------------------------

/// Write a [`Tensor<T>`] as a dataset in a new HDF5 file.
///
/// Creates a minimal valid HDF5 file with a root group containing a single
/// dataset at the given `dataset_path`. If `dataset_path` contains `/`,
/// intermediate groups are **not** created — only a single-level dataset name
/// is supported for writing (e.g., `"my_data"`).
///
/// Multiple datasets can be written with [`write_hdf5_datasets`].
///
/// # Errors
///
/// Returns [`IoError::FormatError`] if the dataset path contains `/`
/// separators (nested groups are not supported for writing).
pub fn write_hdf5_dataset<T: Hdf5Scalar, W: Write + Seek>(
    writer: &mut W,
    dataset_path: &str,
    tensor: &Tensor<T>,
) -> Result<()> {
    let mut map = HashMap::new();
    map.insert(dataset_path.to_string(), tensor);
    write_hdf5_datasets(writer, &map)
}

/// Write multiple tensors of the same type as datasets in a new HDF5 file.
#[allow(clippy::too_many_lines)]
pub fn write_hdf5_datasets<T: Hdf5Scalar, W: Write + Seek, S: ::std::hash::BuildHasher>(
    writer: &mut W,
    datasets: &HashMap<String, &Tensor<T>, S>,
) -> Result<()> {
    for name in datasets.keys() {
        if name.contains('/') {
            return Err(IoError::FormatError(
                "nested dataset paths (with '/') are not supported for writing".into(),
            ));
        }
    }

    // We need deterministic ordering
    let mut names: Vec<&String> = datasets.keys().collect();
    names.sort();

    // -----------------------------------------------------------------------
    // Layout plan:
    //   [superblock]
    //   [root group object header]
    //   [dataset object headers...]
    //   [dataset raw data...]
    //   [B-tree for root group]
    //   [SNOD for root group]
    //   [Local heap for root group]
    //
    // We use 8-byte offsets/lengths throughout for simplicity.
    // -----------------------------------------------------------------------

    let offset_size: u8 = 8;
    let length_size: u8 = 8;

    // Phase 1: Write superblock placeholder (we'll come back to fill root addr)
    let superblock_size: u64 = 8 + 4 + 4 + 4 + 4 + 4 * 8 + 8 + 8 + 4 + 4 + 16; // 96 bytes
    let superblock_bytes = vec![0u8; superblock_size as usize];
    writer.write_all(&superblock_bytes)?;

    let root_obj_header_addr = writer.stream_position()?;

    // Phase 2: Write dataset object headers and collect their metadata.
    // But first we need to know data addresses. We'll write object headers
    // with placeholder data addresses, then patch them.

    // For simplicity: write root group obj header first, then dataset obj
    // headers, then data, then group metadata.

    // Skip root group header for now (we'll write it after we know everything).
    // Reserve space: version(1) + reserved(1) + nmessages(2) + refcount(4) +
    // header_size(4) + symbol_table_msg(8 + 2*offset_size)
    let root_header_msg_size: u64 = 2 * u64::from(offset_size); // symbol table message payload
    let root_header_total: u64 = 12 + (8 + root_header_msg_size); // obj header prefix + 1 message
    writer.seek(SeekFrom::Start(root_obj_header_addr + root_header_total))?;

    // Write each dataset's object header
    #[allow(dead_code, clippy::items_after_statements)]
    struct DatasetLayout {
        name: String,
        obj_header_addr: u64,
        data_addr_patch_offset: u64, // where to patch the data address
        data_byte_size: u64,
        shape: Vec<usize>,
    }

    let mut layouts: Vec<DatasetLayout> = Vec::new();

    for name in &names {
        let tensor = datasets[*name];
        let obj_header_addr = writer.stream_position()?;

        let shape = tensor.shape();
        #[allow(clippy::cast_possible_truncation)]
        let ndims = shape.len() as u8;
        let numel: usize = shape.iter().product();
        let elem_size = T::hdf5_size();
        #[allow(clippy::cast_lossless)]
        let data_byte_size = (numel * elem_size) as u64;

        // --- Object header prefix ---
        writer.write_all(&[OBJECT_HEADER_V1])?; // version
        writer.write_all(&[0u8])?; // reserved

        // 3 messages: dataspace, datatype, data layout
        writer.write_all(&3u16.to_le_bytes())?; // num messages
        writer.write_all(&1u32.to_le_bytes())?; // ref count

        // We'll patch header_size after writing all messages
        let header_size_offset = writer.stream_position()?;
        writer.write_all(&0u32.to_le_bytes())?; // placeholder

        let messages_start = writer.stream_position()?;

        // --- Dataspace message ---
        let dataspace_payload_size: u16 =
            1 + 1 + 1 + 5 + u16::from(ndims) * 8 + u16::from(ndims) * 8;
        writer.write_all(&MSG_DATASPACE.to_le_bytes())?;
        writer.write_all(&dataspace_payload_size.to_le_bytes())?;
        writer.write_all(&[0u8; 4])?; // flags + reserved

        writer.write_all(&[1u8])?; // version 1
        writer.write_all(&[ndims])?; // dimensionality
        writer.write_all(&[0u8])?; // flags
        writer.write_all(&[0u8; 5])?; // reserved (v1)

        for &dim in shape {
            #[allow(clippy::cast_lossless)]
            let dim_u64 = dim as u64;
            writer.write_all(&dim_u64.to_le_bytes())?;
        }
        // Max dimensions (same as current)
        for &dim in shape {
            #[allow(clippy::cast_lossless)]
            let dim_u64 = dim as u64;
            writer.write_all(&dim_u64.to_le_bytes())?;
        }

        // --- Datatype message ---
        let datatype_payload_size: u16 = 4 + 4 + dtype_properties_size::<T>();
        writer.write_all(&MSG_DATATYPE.to_le_bytes())?;
        writer.write_all(&datatype_payload_size.to_le_bytes())?;
        writer.write_all(&[0u8; 4])?; // flags + reserved

        write_datatype_msg::<T, _>(writer)?;

        // --- Data layout message (version 3, contiguous) ---
        let layout_payload_size: u16 = 1 + u16::from(offset_size) + u16::from(length_size);
        writer.write_all(&MSG_DATA_LAYOUT.to_le_bytes())?;
        writer.write_all(&layout_payload_size.to_le_bytes())?;
        writer.write_all(&[0u8; 4])?; // flags + reserved

        writer.write_all(&[3u8])?; // version 3
        // We do NOT write ndims or layout_class for v3 contiguous — v3 format
        // is: version(1), layout_class(1), then class-specific fields.
        // Actually let me re-check the HDF5 spec for data layout v3:
        // v3: version(1) + layout_class(1) + class-specific
        // For contiguous: address(offset_size) + size(length_size)

        // We already wrote version=3. Need to write layout_class=1.
        // But we computed layout_payload_size wrong. Let me fix.

        // Actually, we already wrote version byte as part of the payload.
        // The payload = version(1) + layout_class(1) + address(8) + size(8) = 18
        // We need to go back and fix the payload size.
        let current = writer.stream_position()?;
        let correct_layout_size: u16 = 1 + 1 + u16::from(offset_size) + u16::from(length_size);
        // We already wrote version. Let's seek back to fix the size.
        // The size was written at current - 1 (version byte) - 4 (flags+reserved) - 2 (size) - 2 (type)
        let layout_msg_header_start = current - 1 - 4 - 2 - 2;
        writer.seek(SeekFrom::Start(layout_msg_header_start + 2))?;
        writer.write_all(&correct_layout_size.to_le_bytes())?;
        writer.seek(SeekFrom::Start(current))?;

        writer.write_all(&[1u8])?; // layout class: contiguous

        let data_addr_patch_offset = writer.stream_position()?;
        // Placeholder data address
        writer.write_all(&0u64.to_le_bytes())?;
        writer.write_all(&data_byte_size.to_le_bytes())?;

        let messages_end = writer.stream_position()?;
        #[allow(clippy::cast_possible_truncation)]
        let header_size = (messages_end - messages_start) as u32;

        // Patch header_size
        writer.seek(SeekFrom::Start(header_size_offset))?;
        writer.write_all(&header_size.to_le_bytes())?;
        writer.seek(SeekFrom::Start(messages_end))?;

        layouts.push(DatasetLayout {
            name: (*name).clone(),
            obj_header_addr,
            data_addr_patch_offset,
            data_byte_size,
            shape: shape.to_vec(),
        });
    }

    // Phase 3: Write raw data for each dataset and patch addresses
    for layout in &layouts {
        let data_addr = writer.stream_position()?;

        // Patch the data address in the object header
        let current = writer.stream_position()?;
        writer.seek(SeekFrom::Start(layout.data_addr_patch_offset))?;
        writer.write_all(&data_addr.to_le_bytes())?;
        writer.seek(SeekFrom::Start(current))?;

        let tensor = datasets[&layout.name];
        for &val in tensor.as_slice() {
            writer.write_all(&val.to_le_bytes_vec())?;
        }
    }

    // Phase 4: Write B-tree, SNOD, and local heap for root group

    // --- Local heap ---
    // Build the heap data: concatenation of null-terminated name strings
    let mut heap_data = vec![0u8]; // first byte is empty string (for root ".")
    let mut name_offsets: Vec<(String, usize)> = Vec::new();
    for layout in &layouts {
        let offset = heap_data.len();
        name_offsets.push((layout.name.clone(), offset));
        heap_data.extend_from_slice(layout.name.as_bytes());
        heap_data.push(0); // null terminator
    }

    // Align heap data to 8 bytes
    #[allow(clippy::manual_is_multiple_of)]
    while heap_data.len() % 8 != 0 {
        heap_data.push(0);
    }

    // --- SNOD (Symbol Table Node) ---
    let snod_addr = writer.stream_position()?;

    writer.write_all(&SNOD_SIGNATURE)?;
    writer.write_all(&[1u8])?; // version
    writer.write_all(&[0u8])?; // reserved
    #[allow(clippy::cast_possible_truncation)]
    let num_layouts = layouts.len() as u16;
    writer.write_all(&num_layouts.to_le_bytes())?;

    for layout in &layouts {
        let name_offset = name_offsets
            .iter()
            .find(|(n, _)| n == &layout.name)
            .map_or(0, |(_, o)| *o);

        // Symbol table entry
        #[allow(clippy::cast_lossless)]
        let name_offset_u64 = name_offset as u64;
        write_u64_le(writer, name_offset_u64)?; // link name offset
        write_u64_le(writer, layout.obj_header_addr)?; // object header address
        writer.write_all(&0u32.to_le_bytes())?; // cache type (0 = no cache)
        writer.write_all(&0u32.to_le_bytes())?; // reserved
        writer.write_all(&[0u8; 16])?; // scratch-pad space
    }

    // --- B-tree (leaf node pointing to SNOD) ---
    let btree_addr = writer.stream_position()?;

    writer.write_all(&BTREE_SIGNATURE)?;
    writer.write_all(&[0u8])?; // node type: 0 = group
    writer.write_all(&[0u8])?; // node level: 0 = leaf
    // All entries are in a single SNOD, so one B-tree entry
    writer.write_all(&1u16.to_le_bytes())?; // entries used

    // Left/right sibling: undefined
    write_u64_le(writer, u64::MAX)?;
    write_u64_le(writer, u64::MAX)?;

    // Single key-child pair + trailing key
    write_u64_le(writer, 0)?; // key
    write_u64_le(writer, snod_addr)?; // child = SNOD address
    write_u64_le(writer, 0)?; // trailing key

    // --- Local heap header ---
    let heap_addr = writer.stream_position()?;

    writer.write_all(&HEAP_SIGNATURE)?;
    writer.write_all(&[0u8])?; // version
    writer.write_all(&[0u8; 3])?; // reserved

    #[allow(clippy::cast_lossless)]
    let heap_data_len = heap_data.len() as u64;
    write_u64_le(writer, heap_data_len)?; // data segment size
    write_u64_le(writer, u64::MAX)?; // free list head offset (none)

    let heap_data_addr = writer.stream_position()? + 8; // after the data segment address
    write_u64_le(writer, heap_data_addr)?; // data segment address
    writer.write_all(&heap_data)?;

    // Phase 5: Write root group object header
    writer.seek(SeekFrom::Start(root_obj_header_addr))?;

    writer.write_all(&[OBJECT_HEADER_V1])?; // version
    writer.write_all(&[0u8])?; // reserved
    writer.write_all(&1u16.to_le_bytes())?; // 1 message (symbol table)
    writer.write_all(&1u32.to_le_bytes())?; // ref count

    let msg_payload_size = 2 * u32::from(offset_size);
    let root_msg_total = 8 + msg_payload_size; // msg header(8) + payload
    writer.write_all(&root_msg_total.to_le_bytes())?; // header data size

    // Symbol table message
    writer.write_all(&MSG_SYMBOL_TABLE.to_le_bytes())?;
    writer.write_all(&(2 * u16::from(offset_size)).to_le_bytes())?;
    writer.write_all(&[0u8; 4])?; // flags + reserved
    write_u64_le(writer, btree_addr)?;
    write_u64_le(writer, heap_addr)?;

    // Phase 6: Write the real superblock
    writer.seek(SeekFrom::Start(0))?;

    let eof_addr = {
        let current = writer.stream_position()?;
        writer.seek(SeekFrom::End(0))?;
        let eof = writer.stream_position()?;
        writer.seek(SeekFrom::Start(current))?;
        eof
    };

    // Signature
    writer.write_all(&HDF5_SIGNATURE)?;
    // Version
    writer.write_all(&[0u8])?; // superblock version 0
    // Free-space storage version
    writer.write_all(&[0u8])?;
    // Root group symbol table entry version
    writer.write_all(&[0u8])?;
    // Reserved
    writer.write_all(&[0u8])?;
    // Shared header message format version
    writer.write_all(&[0u8])?;
    // Size of offsets, size of lengths
    writer.write_all(&[offset_size])?;
    writer.write_all(&[length_size])?;
    // Reserved
    writer.write_all(&[0u8])?;
    // Group leaf node K, group internal node K
    writer.write_all(&4u16.to_le_bytes())?;
    writer.write_all(&16u16.to_le_bytes())?;
    // Consistency flags
    writer.write_all(&0u32.to_le_bytes())?;

    // Base address
    write_u64_le(writer, 0)?;
    // Free-space info address (undefined)
    write_u64_le(writer, u64::MAX)?;
    // End-of-file address
    write_u64_le(writer, eof_addr)?;
    // Driver info block address (undefined)
    write_u64_le(writer, u64::MAX)?;

    // Root group symbol table entry
    write_u64_le(writer, 0)?; // link name offset (empty string in heap)
    write_u64_le(writer, root_obj_header_addr)?; // object header address
    writer.write_all(&1u32.to_le_bytes())?; // cache type: 1 = symbol table
    writer.write_all(&0u32.to_le_bytes())?; // reserved

    // Scratch-pad: B-tree address and heap address
    let mut scratch = [0u8; 16];
    scratch[..8].copy_from_slice(&btree_addr.to_le_bytes());
    scratch[8..16].copy_from_slice(&heap_addr.to_le_bytes());
    writer.write_all(&scratch)?;

    writer.flush()?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Datatype writing helpers
// ---------------------------------------------------------------------------

fn dtype_properties_size<T: Hdf5Scalar>() -> u16 {
    match T::hdf5_class() {
        DTYPE_CLASS_FIXED_POINT => 4 + 4, // bit offset + bit precision for integers
        DTYPE_CLASS_FLOATING_POINT => 2 + 2 + 1 + 1 + 1 + 1 + 4, // bit offset(2) + precision(2) + exp_loc(1) + exp_size(1) + mant_loc(1) + mant_size(1) + exp_bias(4)
        _ => 0,
    }
}

fn write_datatype_msg<T: Hdf5Scalar, W: Write>(w: &mut W) -> Result<()> {
    let class = T::hdf5_class();
    let class_bits = T::hdf5_class_bits();
    let size = T::hdf5_size() as u32;

    // Class byte with version in high nibble (version 1 = 0x10)
    let class_and_version: u32 = u32::from(class) | (1 << 4) | (class_bits << 8);
    w.write_all(&class_and_version.to_le_bytes())?;
    w.write_all(&size.to_le_bytes())?;

    match class {
        DTYPE_CLASS_FIXED_POINT => {
            // Bit offset
            w.write_all(&0u16.to_le_bytes())?;
            w.write_all(&0u16.to_le_bytes())?;
            // Bit precision
            let bits = (size * 8) as u16;
            w.write_all(&bits.to_le_bytes())?;
            w.write_all(&0u16.to_le_bytes())?;
        }
        DTYPE_CLASS_FLOATING_POINT => {
            // Floating point properties per HDF5 spec
            match size {
                4 => {
                    // f32 IEEE 754
                    w.write_all(&0u16.to_le_bytes())?; // bit offset
                    w.write_all(&32u16.to_le_bytes())?; // bit precision
                    w.write_all(&23u8.to_le_bytes())?; // exponent location
                    w.write_all(&8u8.to_le_bytes())?; // exponent size
                    w.write_all(&0u8.to_le_bytes())?; // mantissa location
                    w.write_all(&23u8.to_le_bytes())?; // mantissa size (actually stored in next fields)
                    w.write_all(&127u32.to_le_bytes())?; // exponent bias
                }
                8 => {
                    // f64 IEEE 754
                    w.write_all(&0u16.to_le_bytes())?; // bit offset
                    w.write_all(&64u16.to_le_bytes())?; // bit precision
                    w.write_all(&52u8.to_le_bytes())?; // exponent location
                    w.write_all(&11u8.to_le_bytes())?; // exponent size
                    w.write_all(&0u8.to_le_bytes())?; // mantissa location
                    w.write_all(&52u8.to_le_bytes())?; // mantissa size
                    w.write_all(&1023u32.to_le_bytes())?; // exponent bias
                }
                _ => {}
            }
        }
        _ => {}
    }

    Ok(())
}

fn write_u64_le<W: Write>(w: &mut W, val: u64) -> Result<()> {
    w.write_all(&val.to_le_bytes())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// Helper: round-trip write then read.
    fn roundtrip_tensor<T: Hdf5Scalar + PartialEq + std::fmt::Debug>(
        tensor: &Tensor<T>,
        name: &str,
    ) -> Tensor<T> {
        let mut buf = Cursor::new(Vec::new());
        write_hdf5_dataset(&mut buf, name, tensor).expect("write failed");
        buf.seek(SeekFrom::Start(0)).unwrap();
        read_hdf5_dataset::<T, _>(&mut buf, name).expect("read failed")
    }

    #[test]
    fn test_roundtrip_1d_f64() {
        let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let tensor = Tensor::from_vec(data.clone(), vec![5]).unwrap();
        let result = roundtrip_tensor(&tensor, "data");
        assert_eq!(result.shape(), &[5]);
        assert_eq!(result.as_slice(), data.as_slice());
    }

    #[test]
    fn test_roundtrip_2d_f32() {
        let data: Vec<f32> = (0..12).map(|i| i as f32 * 0.5).collect();
        let tensor = Tensor::from_vec(data.clone(), vec![3, 4]).unwrap();
        let result = roundtrip_tensor(&tensor, "matrix");
        assert_eq!(result.shape(), &[3, 4]);
        assert_eq!(result.as_slice(), data.as_slice());
    }

    #[test]
    fn test_multiple_datasets_and_listing() {
        let t1 = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0], vec![3]).unwrap();
        let t2 = Tensor::from_vec(vec![4.0_f64, 5.0, 6.0, 7.0], vec![2, 2]).unwrap();

        let mut datasets = HashMap::new();
        datasets.insert("alpha".to_string(), &t1);
        datasets.insert("beta".to_string(), &t2);

        let mut buf = Cursor::new(Vec::new());
        write_hdf5_datasets(&mut buf, &datasets).expect("write failed");

        buf.seek(SeekFrom::Start(0)).unwrap();
        let mut listing = list_hdf5_datasets(&mut buf).expect("list failed");
        listing.sort();
        assert_eq!(listing, vec!["alpha", "beta"]);

        // Read each dataset back
        buf.seek(SeekFrom::Start(0)).unwrap();
        let r1: Tensor<f64> = read_hdf5_dataset(&mut buf, "alpha").expect("read alpha");
        assert_eq!(r1.as_slice(), &[1.0, 2.0, 3.0]);

        buf.seek(SeekFrom::Start(0)).unwrap();
        let r2: Tensor<f64> = read_hdf5_dataset(&mut buf, "beta").expect("read beta");
        assert_eq!(r2.shape(), &[2, 2]);
        assert_eq!(r2.as_slice(), &[4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn test_invalid_file_errors() {
        // Empty file
        let mut buf = Cursor::new(Vec::<u8>::new());
        let result = read_hdf5_dataset::<f64, _>(&mut buf, "x");
        assert!(result.is_err());

        // Random garbage
        let mut buf = Cursor::new(vec![0u8; 64]);
        let result = read_hdf5_dataset::<f64, _>(&mut buf, "x");
        assert!(result.is_err());

        // Truncated HDF5 signature
        let mut buf = Cursor::new(HDF5_SIGNATURE[..4].to_vec());
        let result = read_hdf5_dataset::<f64, _>(&mut buf, "x");
        assert!(result.is_err());
    }

    #[test]
    fn test_shape_preservation() {
        let data: Vec<f64> = (0..12).map(f64::from).collect();
        let tensor = Tensor::from_vec(data, vec![3, 4]).unwrap();
        let result = roundtrip_tensor(&tensor, "shaped");
        assert_eq!(result.shape(), &[3, 4]);
        assert_eq!(result.numel(), 12);
    }

    #[test]
    fn test_large_dataset() {
        let data: Vec<f64> = (0..5000).map(|i| f64::from(i) * 0.001).collect();
        let tensor = Tensor::from_vec(data.clone(), vec![100, 50]).unwrap();
        let result = roundtrip_tensor(&tensor, "big");
        assert_eq!(result.shape(), &[100, 50]);
        assert_eq!(result.as_slice(), data.as_slice());
    }

    #[test]
    fn test_roundtrip_i32() {
        let data: Vec<i32> = (0..10).collect();
        let tensor = Tensor::from_vec(data.clone(), vec![10]).unwrap();
        let result = roundtrip_tensor(&tensor, "integers");
        assert_eq!(result.shape(), &[10]);
        assert_eq!(result.as_slice(), data.as_slice());
    }

    #[test]
    fn test_roundtrip_i64() {
        let data: Vec<i64> = (0..6).map(|i| i * 1_000_000).collect();
        let tensor = Tensor::from_vec(data.clone(), vec![2, 3]).unwrap();
        let result = roundtrip_tensor(&tensor, "big_ints");
        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(result.as_slice(), data.as_slice());
    }

    #[test]
    fn test_nonexistent_dataset_error() {
        let tensor = Tensor::from_vec(vec![1.0_f64], vec![1]).unwrap();
        let mut buf = Cursor::new(Vec::new());
        write_hdf5_dataset(&mut buf, "exists", &tensor).unwrap();
        buf.seek(SeekFrom::Start(0)).unwrap();

        let result = read_hdf5_dataset::<f64, _>(&mut buf, "does_not_exist");
        assert!(result.is_err());
    }

    #[test]
    fn test_list_empty_file_with_no_datasets() {
        // Write a file with one dataset, then check listing works
        let tensor = Tensor::from_vec(vec![42.0_f64], vec![1]).unwrap();
        let mut buf = Cursor::new(Vec::new());
        write_hdf5_dataset(&mut buf, "solo", &tensor).unwrap();
        buf.seek(SeekFrom::Start(0)).unwrap();

        let listing = list_hdf5_datasets(&mut buf).unwrap();
        assert_eq!(listing, vec!["solo"]);
    }
}
