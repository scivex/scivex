//! GGUF format — llama.cpp's standard for quantized model storage.
//!
//! ## Format
//!
//! ```text
//! [4 bytes]  Magic: "GGUF"
//! [4 bytes]  Version (little-endian u32, currently 3)
//! [8 bytes]  Tensor count (little-endian u64)
//! [8 bytes]  Metadata KV count (little-endian u64)
//! [...]      Metadata key-value pairs
//! [...]      Tensor info entries
//! [padding]  Alignment to 32 bytes
//! [...]      Tensor data (each tensor aligned to 32 bytes)
//! ```
//!
//! Supports F16 (read as F32), F32, and F64 tensor types.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};

use scivex_core::{Float, Tensor};

use crate::error::{NnError, Result};

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const GGUF_VERSION: u32 = 3;
const ALIGNMENT: usize = 32;

// GGUF type tags for metadata values
const GGUF_TYPE_UINT32: u32 = 4;
const GGUF_TYPE_INT32: u32 = 5;
const GGUF_TYPE_FLOAT32: u32 = 6;
const GGUF_TYPE_STRING: u32 = 8;
const GGUF_TYPE_UINT64: u32 = 10;

// GGUF tensor type IDs
const GGML_TYPE_F32: u32 = 0;
const GGML_TYPE_F16: u32 = 1;
const GGML_TYPE_F64: u32 = 28;

fn ser_err(msg: &str) -> NnError {
    NnError::SerializeError(msg.to_string())
}

fn io_err() -> NnError {
    NnError::SerializeError("I/O error during GGUF persistence".to_string())
}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A value in a GGUF metadata key-value store.
///
/// # Examples
///
/// ```
/// # use scivex_nn::serialize::GgufValue;
/// let v = GgufValue::Uint32(42);
/// if let GgufValue::Uint32(n) = v {
///     assert_eq!(n, 42);
/// }
/// let s = GgufValue::String("hello".to_string());
/// if let GgufValue::String(text) = s {
///     assert_eq!(text, "hello");
/// }
/// ```
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone, PartialEq)]
pub enum GgufValue {
    /// Unsigned 32-bit integer.
    Uint32(u32),
    /// Signed 32-bit integer.
    Int32(i32),
    /// 32-bit float.
    Float32(f32),
    /// UTF-8 string.
    String(String),
    /// Unsigned 64-bit integer.
    Uint64(u64),
}

/// A GGUF file containing metadata and tensors.
///
/// # Examples
///
/// ```
/// # use scivex_nn::serialize::{GgufFile, GgufValue};
/// # use scivex_core::Tensor;
/// let t = Tensor::<f64>::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
/// let file: GgufFile<f64> = GgufFile {
///     metadata: vec![("layers".to_string(), GgufValue::Uint32(12))],
///     tensors: vec![("weight".to_string(), t)],
/// };
/// assert_eq!(file.tensors.len(), 1);
/// assert_eq!(file.metadata.len(), 1);
/// ```
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct GgufFile<T: Float> {
    /// Metadata key-value pairs.
    pub metadata: Vec<(String, GgufValue)>,
    /// Named tensors.
    pub tensors: Vec<(String, Tensor<T>)>,
}

// ---------------------------------------------------------------------------
// Read helpers
// ---------------------------------------------------------------------------

fn read_u32(r: &mut impl Read) -> Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf).map_err(|_| io_err())?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64(r: &mut impl Read) -> Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf).map_err(|_| io_err())?;
    Ok(u64::from_le_bytes(buf))
}

fn read_i32(r: &mut impl Read) -> Result<i32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf).map_err(|_| io_err())?;
    Ok(i32::from_le_bytes(buf))
}

fn read_f32(r: &mut impl Read) -> Result<f32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf).map_err(|_| io_err())?;
    Ok(f32::from_le_bytes(buf))
}

fn read_gguf_string(r: &mut impl Read) -> Result<String> {
    let len = read_u64(r)? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf).map_err(|_| io_err())?;
    String::from_utf8(buf).map_err(|_| ser_err("invalid UTF-8 in GGUF string"))
}

fn read_gguf_value(r: &mut impl Read) -> Result<GgufValue> {
    let type_id = read_u32(r)?;
    match type_id {
        GGUF_TYPE_UINT32 => Ok(GgufValue::Uint32(read_u32(r)?)),
        GGUF_TYPE_INT32 => Ok(GgufValue::Int32(read_i32(r)?)),
        GGUF_TYPE_FLOAT32 => Ok(GgufValue::Float32(read_f32(r)?)),
        GGUF_TYPE_STRING => Ok(GgufValue::String(read_gguf_string(r)?)),
        GGUF_TYPE_UINT64 => Ok(GgufValue::Uint64(read_u64(r)?)),
        _ => Err(ser_err(&format!("unsupported GGUF value type: {type_id}"))),
    }
}

// ---------------------------------------------------------------------------
// Write helpers
// ---------------------------------------------------------------------------

fn write_u32(w: &mut impl Write, v: u32) -> Result<()> {
    w.write_all(&v.to_le_bytes()).map_err(|_| io_err())
}

fn write_u64(w: &mut impl Write, v: u64) -> Result<()> {
    w.write_all(&v.to_le_bytes()).map_err(|_| io_err())
}

fn write_i32(w: &mut impl Write, v: i32) -> Result<()> {
    w.write_all(&v.to_le_bytes()).map_err(|_| io_err())
}

fn write_f32(w: &mut impl Write, v: f32) -> Result<()> {
    w.write_all(&v.to_le_bytes()).map_err(|_| io_err())
}

fn write_gguf_string(w: &mut impl Write, s: &str) -> Result<()> {
    write_u64(w, s.len() as u64)?;
    w.write_all(s.as_bytes()).map_err(|_| io_err())
}

fn write_gguf_value(w: &mut impl Write, val: &GgufValue) -> Result<()> {
    match val {
        GgufValue::Uint32(v) => {
            write_u32(w, GGUF_TYPE_UINT32)?;
            write_u32(w, *v)
        }
        GgufValue::Int32(v) => {
            write_u32(w, GGUF_TYPE_INT32)?;
            write_i32(w, *v)
        }
        GgufValue::Float32(v) => {
            write_u32(w, GGUF_TYPE_FLOAT32)?;
            write_f32(w, *v)
        }
        GgufValue::String(v) => {
            write_u32(w, GGUF_TYPE_STRING)?;
            write_gguf_string(w, v)
        }
        GgufValue::Uint64(v) => {
            write_u32(w, GGUF_TYPE_UINT64)?;
            write_u64(w, *v)
        }
    }
}

/// Determine the GGML type ID for type `T`.
fn ggml_type_for<T: Float>() -> u32 {
    let size = std::mem::size_of::<T>();
    match size {
        4 => GGML_TYPE_F32,
        _ => GGML_TYPE_F64, // 8 bytes or fallback
    }
}

/// Return the byte size per element for a GGML type.
fn ggml_type_size(type_id: u32) -> Result<usize> {
    match type_id {
        GGML_TYPE_F16 => Ok(2),
        GGML_TYPE_F32 => Ok(4),
        GGML_TYPE_F64 => Ok(8),
        _ => Err(ser_err(&format!("unsupported GGML tensor type: {type_id}"))),
    }
}

fn align_offset(offset: usize) -> usize {
    let remainder = offset % ALIGNMENT;
    if remainder == 0 {
        offset
    } else {
        offset + (ALIGNMENT - remainder)
    }
}

/// Half-precision (F16) to f32 conversion following IEEE 754.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from((bits >> 15) & 1);
    let exponent = u32::from((bits >> 10) & 0x1F);
    let mantissa = u32::from(bits & 0x3FF);

    if exponent == 0 {
        // Subnormal or zero
        if mantissa == 0 {
            return f32::from_bits(sign << 31);
        }
        // Subnormal: convert to normalized f32
        let mut m = mantissa;
        let mut e: i32 = -14; // bias difference: 127 - 15 - (10 mantissa bits normalisation)
        // Normalize
        while m & 0x400 == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x3FF; // remove implicit bit
        #[allow(clippy::cast_sign_loss)]
        let f32_exp = (e + 127) as u32;
        let f32_bits = (sign << 31) | (f32_exp << 23) | (m << 13);
        return f32::from_bits(f32_bits);
    }

    if exponent == 31 {
        // Inf or NaN
        let f32_bits = (sign << 31) | (0xFF << 23) | (mantissa << 13);
        return f32::from_bits(f32_bits);
    }

    // Normal
    let f32_exp = exponent + 112; // 127 - 15
    let f32_bits = (sign << 31) | (f32_exp << 23) | (mantissa << 13);
    f32::from_bits(f32_bits)
}

// ---------------------------------------------------------------------------
// Tensor info for reading
// ---------------------------------------------------------------------------

struct TensorInfo {
    name: String,
    dims: Vec<usize>,
    type_id: u32,
    offset: u64,
}

fn read_tensor_info(r: &mut impl Read) -> Result<TensorInfo> {
    let name = read_gguf_string(r)?;
    let n_dims = read_u32(r)? as usize;
    let mut dims = Vec::with_capacity(n_dims);
    for _ in 0..n_dims {
        dims.push(read_u64(r)? as usize);
    }
    let type_id = read_u32(r)?;
    let offset = read_u64(r)?;
    Ok(TensorInfo {
        name,
        dims,
        type_id,
        offset,
    })
}

fn read_tensor_data<T: Float>(data: &[u8], info: &TensorInfo) -> Result<Tensor<T>> {
    let numel: usize = if info.dims.is_empty() {
        1
    } else {
        info.dims.iter().product()
    };
    let elem_size = ggml_type_size(info.type_id)?;
    let byte_len = numel * elem_size;
    let start = info.offset as usize;
    let end = start + byte_len;

    if end > data.len() {
        return Err(ser_err("tensor data out of bounds in GGUF"));
    }

    let slice = &data[start..end];
    let mut values = Vec::with_capacity(numel);

    match info.type_id {
        GGML_TYPE_F16 => {
            for i in 0..numel {
                let off = i * 2;
                let bits = u16::from_le_bytes([slice[off], slice[off + 1]]);
                let val = f16_to_f32(bits);
                values.push(T::from_f64(f64::from(val)));
            }
        }
        GGML_TYPE_F32 => {
            for i in 0..numel {
                let off = i * 4;
                let val = f32::from_le_bytes([
                    slice[off],
                    slice[off + 1],
                    slice[off + 2],
                    slice[off + 3],
                ]);
                values.push(T::from_f64(f64::from(val)));
            }
        }
        GGML_TYPE_F64 => {
            for i in 0..numel {
                let off = i * 8;
                let val = f64::from_le_bytes([
                    slice[off],
                    slice[off + 1],
                    slice[off + 2],
                    slice[off + 3],
                    slice[off + 4],
                    slice[off + 5],
                    slice[off + 6],
                    slice[off + 7],
                ]);
                values.push(T::from_f64(val));
            }
        }
        _ => {
            return Err(ser_err(&format!(
                "unsupported tensor type {}",
                info.type_id
            )));
        }
    }

    let shape = if info.dims.is_empty() {
        vec![1]
    } else {
        info.dims.clone()
    };

    Tensor::from_vec(values, shape).map_err(|e| ser_err(&format!("tensor creation: {e}")))
}

// ---------------------------------------------------------------------------
// Size calculation helpers for writing
// ---------------------------------------------------------------------------

fn gguf_string_wire_size(s: &str) -> usize {
    8 + s.len() // u64 length + bytes
}

fn gguf_value_wire_size(val: &GgufValue) -> usize {
    let type_tag = 4usize; // u32 type tag
    type_tag
        + match val {
            GgufValue::Uint32(_) | GgufValue::Int32(_) | GgufValue::Float32(_) => 4,
            GgufValue::String(s) => gguf_string_wire_size(s),
            GgufValue::Uint64(_) => 8,
        }
}

fn gguf_kv_wire_size(key: &str, val: &GgufValue) -> usize {
    gguf_string_wire_size(key) + gguf_value_wire_size(val)
}

fn tensor_info_wire_size(name: &str, n_dims: usize) -> usize {
    gguf_string_wire_size(name)  // name
        + 4                       // n_dims (u32)
        + n_dims * 8              // dims (u64 each)
        + 4                       // type (u32)
        + 8 // offset (u64)
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Load a GGUF file from disk.
///
/// # Examples
///
/// ```ignore
/// # use scivex_nn::serialize::gguf::load_gguf;
/// let file = load_gguf::<f64>("/path/to/model.gguf").unwrap();
/// println!("tensors: {}", file.tensors.len());
/// ```
pub fn load_gguf<T: Float>(path: &str) -> Result<GgufFile<T>> {
    let f = File::open(path).map_err(|_| ser_err("cannot open GGUF file"))?;
    let mut r = BufReader::new(f);

    // Magic
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic).map_err(|_| io_err())?;
    if &magic != GGUF_MAGIC {
        return Err(ser_err("not a valid GGUF file (bad magic)"));
    }

    // Version
    let version = read_u32(&mut r)?;
    #[allow(clippy::manual_range_contains)]
    if version < 2 || version > 3 {
        return Err(ser_err(&format!("unsupported GGUF version: {version}")));
    }

    // Counts
    let tensor_count = read_u64(&mut r)? as usize;
    let kv_count = read_u64(&mut r)? as usize;

    // Read metadata
    let mut metadata = Vec::with_capacity(kv_count);
    for _ in 0..kv_count {
        let key = read_gguf_string(&mut r)?;
        let val = read_gguf_value(&mut r)?;
        metadata.push((key, val));
    }

    // Read tensor infos
    let mut tensor_infos = Vec::with_capacity(tensor_count);
    for _ in 0..tensor_count {
        tensor_infos.push(read_tensor_info(&mut r)?);
    }

    // Read remaining data (after alignment)
    let mut all_data = Vec::new();
    r.read_to_end(&mut all_data).map_err(|_| io_err())?;

    // The offsets in tensor info are relative to the start of the data section.
    // We need to figure out where the data section starts within `all_data`.
    // The data we just read includes alignment padding + tensor data.
    // The tensor offsets are relative to the data section start (after alignment).
    // We need to find the first aligned position in `all_data`.
    //
    // Actually, offsets in GGUF are relative to the start of the tensor data
    // region. The tensor data region starts at the next ALIGNMENT boundary
    // after the header + metadata + tensor-info section.
    //
    // Since we've already consumed everything up to where `read_to_end` starts,
    // the `all_data` buffer starts right after the last tensor info. We need to
    // skip alignment padding from our current position.
    //
    // Current position in stream = 4 (magic) + 4 (version) + 8 (tc) + 8 (kvc)
    //   + metadata bytes + tensor_info bytes
    //
    // We compute header_bytes_consumed and find how much padding is in all_data.

    let mut header_bytes: usize = 4 + 4 + 8 + 8;
    for (k, v) in &metadata {
        header_bytes += gguf_kv_wire_size(k, v);
    }
    for info in &tensor_infos {
        header_bytes += tensor_info_wire_size(&info.name, info.dims.len());
    }

    let data_start_aligned = align_offset(header_bytes);
    let padding = data_start_aligned - header_bytes;

    // `all_data` starts right after the tensor info section, so padding bytes
    // are at the beginning of `all_data`.
    let data_section = if padding <= all_data.len() {
        &all_data[padding..]
    } else {
        &all_data
    };

    // Build tensors
    let mut tensors = Vec::with_capacity(tensor_count);
    for info in &tensor_infos {
        let tensor = read_tensor_data(data_section, info)?;
        tensors.push((info.name.clone(), tensor));
    }

    Ok(GgufFile { metadata, tensors })
}

/// Save a `GgufFile` to disk.
///
/// # Examples
///
/// ```ignore
/// # use scivex_nn::serialize::gguf::{GgufFile, save_gguf};
/// # use scivex_core::Tensor;
/// let t = Tensor::<f64>::from_vec(vec![1.0, 2.0], vec![2]).unwrap();
/// let file = GgufFile { metadata: vec![], tensors: vec![("w".to_string(), t)] };
/// save_gguf("/tmp/model.gguf", &file).unwrap();
/// ```
pub fn save_gguf<T: Float>(path: &str, file: &GgufFile<T>) -> Result<()> {
    let f = File::create(path).map_err(|_| ser_err("cannot create GGUF file"))?;
    let mut w = BufWriter::new(f);

    // Magic + version
    w.write_all(GGUF_MAGIC).map_err(|_| io_err())?;
    write_u32(&mut w, GGUF_VERSION)?;

    // Counts
    write_u64(&mut w, file.tensors.len() as u64)?;
    write_u64(&mut w, file.metadata.len() as u64)?;

    // Write metadata KV pairs
    for (key, val) in &file.metadata {
        write_gguf_string(&mut w, key)?;
        write_gguf_value(&mut w, val)?;
    }

    // Compute header size (up to this point + tensor info)
    let ggml_type = ggml_type_for::<T>();
    let elem_size = ggml_type_size(ggml_type)?;

    let mut header_size: usize = 4 + 4 + 8 + 8;
    for (k, v) in &file.metadata {
        header_size += gguf_kv_wire_size(k, v);
    }
    for (name, tensor) in &file.tensors {
        header_size += tensor_info_wire_size(name, tensor.ndim());
    }
    let data_section_start = align_offset(header_size);

    // Write tensor info entries
    let mut offset: u64 = 0;
    let mut tensor_offsets = Vec::with_capacity(file.tensors.len());
    for (name, tensor) in &file.tensors {
        write_gguf_string(&mut w, name)?;
        let ndim = tensor.ndim();
        write_u32(&mut w, ndim as u32)?;
        for &dim in tensor.shape() {
            write_u64(&mut w, dim as u64)?;
        }
        write_u32(&mut w, ggml_type)?;
        write_u64(&mut w, offset)?;
        tensor_offsets.push(offset);

        let numel: usize = tensor.shape().iter().product();
        let byte_len = numel * elem_size;
        let next = align_offset(offset as usize + byte_len);
        offset = next as u64;
    }

    // Write alignment padding between header and data
    let padding = data_section_start - header_size;
    if padding > 0 {
        let pad = vec![0u8; padding];
        w.write_all(&pad).map_err(|_| io_err())?;
    }

    // Write tensor data (with alignment padding between tensors)
    for (i, (_name, tensor)) in file.tensors.iter().enumerate() {
        let values = tensor.as_slice();
        let expected_offset = tensor_offsets[i] as usize;
        let current_data_offset = if i == 0 {
            0
        } else {
            let prev_tensor = &file.tensors[i - 1].1;
            let prev_numel: usize = prev_tensor.shape().iter().product();
            tensor_offsets[i - 1] as usize + prev_numel * elem_size
        };
        // Write padding if needed
        if expected_offset > current_data_offset {
            let pad = vec![0u8; expected_offset - current_data_offset];
            w.write_all(&pad).map_err(|_| io_err())?;
        }

        match elem_size {
            4 => {
                for &v in values {
                    #[allow(clippy::cast_possible_truncation)]
                    let f = v.to_f64() as f32;
                    w.write_all(&f.to_le_bytes()).map_err(|_| io_err())?;
                }
            }
            8 => {
                for &v in values {
                    let f = v.to_f64();
                    w.write_all(&f.to_le_bytes()).map_err(|_| io_err())?;
                }
            }
            _ => return Err(ser_err("unsupported element size")),
        }
    }

    w.flush().map_err(|_| io_err())?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_path(name: &str) -> String {
        let dir = std::env::temp_dir();
        format!(
            "{}/scivex_gguf_test_{name}_{}.gguf",
            dir.display(),
            std::process::id()
        )
    }

    #[test]
    fn test_gguf_single_tensor() {
        let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = Tensor::from_vec(data.clone(), vec![2, 3]).unwrap();
        let file = GgufFile {
            metadata: vec![],
            tensors: vec![("weight".to_string(), t)],
        };
        let path = temp_path("single");
        save_gguf(&path, &file).unwrap();
        let loaded: GgufFile<f64> = load_gguf(&path).unwrap();
        assert_eq!(loaded.tensors.len(), 1);
        assert_eq!(loaded.tensors[0].0, "weight");
        assert_eq!(loaded.tensors[0].1.shape(), &[2, 3]);
        let loaded_data = loaded.tensors[0].1.as_slice();
        for (a, b) in data.iter().zip(loaded_data.iter()) {
            assert!((*a - *b).abs() < 1e-12);
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_gguf_with_metadata() {
        let t = Tensor::from_vec(vec![1.0_f64, 2.0], vec![2]).unwrap();
        let file = GgufFile {
            metadata: vec![
                (
                    "model.name".to_string(),
                    GgufValue::String("test-model".to_string()),
                ),
                ("model.layers".to_string(), GgufValue::Uint32(12)),
                ("model.version".to_string(), GgufValue::Float32(1.5)),
            ],
            tensors: vec![("bias".to_string(), t)],
        };
        let path = temp_path("meta");
        save_gguf(&path, &file).unwrap();
        let loaded: GgufFile<f64> = load_gguf(&path).unwrap();
        assert_eq!(loaded.metadata.len(), 3);
        assert_eq!(loaded.tensors.len(), 1);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_gguf_multiple_tensors() {
        let t1 = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0], vec![3]).unwrap();
        let t2 = Tensor::from_vec(vec![4.0_f64, 5.0, 6.0, 7.0], vec![2, 2]).unwrap();
        let t3 = Tensor::from_vec(vec![8.0_f64, 9.0], vec![1, 2]).unwrap();
        let file = GgufFile {
            metadata: vec![],
            tensors: vec![
                ("layer1.weight".to_string(), t1),
                ("layer1.bias".to_string(), t2),
                ("layer2.weight".to_string(), t3),
            ],
        };
        let path = temp_path("multi");
        save_gguf(&path, &file).unwrap();
        let loaded: GgufFile<f64> = load_gguf(&path).unwrap();
        assert_eq!(loaded.tensors.len(), 3);
        // Verify data
        assert_eq!(loaded.tensors[0].1.as_slice(), &[1.0, 2.0, 3.0]);
        assert_eq!(loaded.tensors[1].1.as_slice(), &[4.0, 5.0, 6.0, 7.0]);
        assert_eq!(loaded.tensors[2].1.as_slice(), &[8.0, 9.0]);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    #[allow(clippy::approx_constant)]
    fn test_gguf_metadata_roundtrip() {
        let t = Tensor::from_vec(vec![0.0_f64], vec![1]).unwrap();
        let file = GgufFile {
            metadata: vec![
                ("key.uint32".to_string(), GgufValue::Uint32(42)),
                ("key.int32".to_string(), GgufValue::Int32(-7)),
                ("key.float32".to_string(), GgufValue::Float32(3.14)),
                (
                    "key.string".to_string(),
                    GgufValue::String("hello world".to_string()),
                ),
                ("key.uint64".to_string(), GgufValue::Uint64(1_000_000)),
            ],
            tensors: vec![("dummy".to_string(), t)],
        };
        let path = temp_path("kv_roundtrip");
        save_gguf(&path, &file).unwrap();
        let loaded: GgufFile<f64> = load_gguf(&path).unwrap();

        assert_eq!(loaded.metadata.len(), 5);
        assert_eq!(
            loaded.metadata[0],
            ("key.uint32".to_string(), GgufValue::Uint32(42))
        );
        assert_eq!(
            loaded.metadata[1],
            ("key.int32".to_string(), GgufValue::Int32(-7))
        );
        // Float comparison
        if let GgufValue::Float32(v) = loaded.metadata[2].1 {
            assert!((v - 3.14).abs() < 1e-5);
        } else {
            panic!("expected Float32");
        }
        assert_eq!(
            loaded.metadata[3],
            (
                "key.string".to_string(),
                GgufValue::String("hello world".to_string())
            )
        );
        assert_eq!(
            loaded.metadata[4],
            ("key.uint64".to_string(), GgufValue::Uint64(1_000_000))
        );

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_gguf_large_tensor() {
        let mut data = Vec::with_capacity(10_000);
        for i in 0..10_000 {
            data.push(f64::from(i) * 0.001);
        }
        let t = Tensor::from_vec(data.clone(), vec![100, 100]).unwrap();
        let file = GgufFile {
            metadata: vec![],
            tensors: vec![("big".to_string(), t)],
        };
        let path = temp_path("large");
        save_gguf(&path, &file).unwrap();
        let loaded: GgufFile<f64> = load_gguf(&path).unwrap();
        assert_eq!(loaded.tensors.len(), 1);
        assert_eq!(loaded.tensors[0].1.shape(), &[100, 100]);
        let loaded_data = loaded.tensors[0].1.as_slice();
        for (a, b) in data.iter().zip(loaded_data.iter()) {
            assert!((*a - *b).abs() < 1e-12);
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_f16_to_f32_conversion() {
        // 1.0 in f16: sign=0, exp=15 (0b01111), mant=0 => 0x3C00
        let val = f16_to_f32(0x3C00);
        assert!((val - 1.0).abs() < 1e-6);

        // 0.0 in f16
        let val = f16_to_f32(0x0000);
        assert!(val == 0.0);

        // -1.0 in f16: 0xBC00
        let val = f16_to_f32(0xBC00);
        assert!((val - (-1.0)).abs() < 1e-6);
    }
}
