//! Write [`Tensor<f64>`] to `.npy` and `.npz` files.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Seek, Write};
use std::path::Path;

use scivex_core::Tensor;

use crate::error::Result;

/// NPY magic bytes.
const NPY_MAGIC: &[u8; 6] = b"\x93NUMPY";

/// Write a `Tensor<f64>` to a writer in `.npy` format.
///
/// Always writes in C-order (row-major), little-endian, f64 (`<f8`).
///
/// # Examples
///
/// ```
/// use scivex_io::npy::write_npy;
/// use scivex_core::Tensor;
///
/// let tensor = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0], vec![3]).unwrap();
/// let mut buf = Vec::new();
/// write_npy(&mut buf, &tensor).unwrap();
/// // NPY files start with magic bytes \x93NUMPY
/// assert_eq!(&buf[..6], b"\x93NUMPY");
/// ```
pub fn write_npy<W: Write>(mut writer: W, tensor: &Tensor<f64>) -> Result<()> {
    let header_bytes = build_npy_header(tensor.shape());
    writer.write_all(&header_bytes)?;

    // Write raw f64 data in little-endian
    for &val in tensor.as_slice() {
        writer.write_all(&val.to_le_bytes())?;
    }

    Ok(())
}

/// Write a `Tensor<f64>` to a file path in `.npy` format.
///
/// # Examples
///
/// ```ignore
/// use scivex_io::npy::write_npy_path;
/// use scivex_core::Tensor;
/// let tensor = Tensor::from_vec(vec![1.0_f64, 2.0], vec![2]).unwrap();
/// write_npy_path("out.npy", &tensor).unwrap();
/// ```
pub fn write_npy_path<P: AsRef<Path>>(path: P, tensor: &Tensor<f64>) -> Result<()> {
    let file = File::create(path)?;
    write_npy(BufWriter::new(file), tensor)
}

/// Write multiple named tensors to a writer in `.npz` format (uncompressed ZIP).
///
/// # Examples
///
/// ```
/// use std::io::Cursor;
/// use std::collections::HashMap;
/// use scivex_io::npy::write_npz;
/// use scivex_core::Tensor;
///
/// let t = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0], vec![3]).unwrap();
/// let mut tensors = HashMap::new();
/// tensors.insert("data".to_string(), t);
/// let mut buf = Cursor::new(Vec::new());
/// write_npz(&mut buf, &tensors).unwrap();
/// // ZIP files start with PK signature
/// assert_eq!(&buf.into_inner()[..2], b"PK");
/// ```
pub fn write_npz<W: Write + Seek, S: std::hash::BuildHasher>(
    mut writer: W,
    tensors: &HashMap<String, Tensor<f64>, S>,
) -> Result<()> {
    let mut entries: Vec<ZipLocalEntry> = Vec::with_capacity(tensors.len());

    // Sort keys for deterministic output
    let mut keys: Vec<&String> = tensors.keys().collect();
    keys.sort();

    // Write local file headers + data
    for key in &keys {
        let tensor = &tensors[*key];
        let name = format!("{key}.npy");

        let mut npy_data = Vec::new();
        write_npy(&mut npy_data, tensor)?;

        let offset = writer.stream_position()? as u32;

        write_zip_local_header(&mut writer, &name, npy_data.len() as u32)?;
        writer.write_all(&npy_data)?;

        entries.push(ZipLocalEntry {
            name,
            offset,
            size: npy_data.len() as u32,
            crc32: crc32_simple(&npy_data),
        });
    }

    // Write central directory
    let cd_offset = writer.stream_position()? as u32;
    for entry in &entries {
        write_zip_cd_entry(&mut writer, entry)?;
    }
    let cd_size = writer.stream_position()? as u32 - cd_offset;

    // Write EOCD
    write_zip_eocd(&mut writer, entries.len() as u16, cd_size, cd_offset)?;

    Ok(())
}

/// Write multiple named tensors to a file path in `.npz` format.
///
/// # Examples
///
/// ```ignore
/// use std::collections::HashMap;
/// use scivex_io::npy::write_npz_path;
/// use scivex_core::Tensor;
/// let t = Tensor::from_vec(vec![1.0_f64], vec![1]).unwrap();
/// let mut map = HashMap::new();
/// map.insert("x".to_string(), t);
/// write_npz_path("out.npz", &map).unwrap();
/// ```
pub fn write_npz_path<P: AsRef<Path>, S: std::hash::BuildHasher>(
    path: P,
    tensors: &HashMap<String, Tensor<f64>, S>,
) -> Result<()> {
    let file = File::create(path)?;
    write_npz(BufWriter::new(file), tensors)
}

/// Build the NPY header bytes (magic + version + header string).
fn build_npy_header(shape: &[usize]) -> Vec<u8> {
    let shape_str = if shape.len() == 1 {
        format!("({},)", shape[0])
    } else {
        let parts: Vec<String> = shape.iter().map(std::string::ToString::to_string).collect();
        format!("({})", parts.join(", "))
    };

    let header_str = format!("{{'descr': '<f8', 'fortran_order': False, 'shape': {shape_str}, }}");

    // Pad to align (magic + version + header_len + header) to 64 bytes
    let prefix_len = 6 + 2 + 2; // magic(6) + version(2) + header_len(2) for v1
    let mut padded = header_str;
    while (prefix_len + padded.len()) % 64 != 63 {
        padded.push(' ');
    }
    padded.push('\n');

    let header_len = padded.len() as u16;

    let mut buf = Vec::with_capacity(prefix_len + padded.len());
    buf.extend_from_slice(NPY_MAGIC);
    buf.push(1); // major version
    buf.push(0); // minor version
    buf.extend_from_slice(&header_len.to_le_bytes());
    buf.extend_from_slice(padded.as_bytes());
    buf
}

// --- Minimal ZIP writer ---

struct ZipLocalEntry {
    name: String,
    offset: u32,
    size: u32,
    crc32: u32,
}

fn write_zip_local_header<W: Write>(writer: &mut W, name: &str, size: u32) -> Result<()> {
    let name_bytes = name.as_bytes();
    writer.write_all(&[0x50, 0x4b, 0x03, 0x04])?; // local file header signature
    writer.write_all(&20u16.to_le_bytes())?; // version needed (2.0)
    writer.write_all(&0u16.to_le_bytes())?; // general purpose flags
    writer.write_all(&0u16.to_le_bytes())?; // compression: stored
    writer.write_all(&0u16.to_le_bytes())?; // mod time
    writer.write_all(&0u16.to_le_bytes())?; // mod date
    writer.write_all(&0u32.to_le_bytes())?; // crc32 (filled later in CD)
    writer.write_all(&size.to_le_bytes())?; // compressed size
    writer.write_all(&size.to_le_bytes())?; // uncompressed size
    writer.write_all(&(name_bytes.len() as u16).to_le_bytes())?; // file name length
    writer.write_all(&0u16.to_le_bytes())?; // extra field length
    writer.write_all(name_bytes)?; // file name
    Ok(())
}

fn write_zip_cd_entry<W: Write>(writer: &mut W, entry: &ZipLocalEntry) -> Result<()> {
    let name_bytes = entry.name.as_bytes();
    writer.write_all(&[0x50, 0x4b, 0x01, 0x02])?; // central directory signature
    writer.write_all(&20u16.to_le_bytes())?; // version made by
    writer.write_all(&20u16.to_le_bytes())?; // version needed
    writer.write_all(&0u16.to_le_bytes())?; // flags
    writer.write_all(&0u16.to_le_bytes())?; // compression: stored
    writer.write_all(&0u16.to_le_bytes())?; // mod time
    writer.write_all(&0u16.to_le_bytes())?; // mod date
    writer.write_all(&entry.crc32.to_le_bytes())?; // crc32
    writer.write_all(&entry.size.to_le_bytes())?; // compressed size
    writer.write_all(&entry.size.to_le_bytes())?; // uncompressed size
    writer.write_all(&(name_bytes.len() as u16).to_le_bytes())?; // file name length
    writer.write_all(&0u16.to_le_bytes())?; // extra field length
    writer.write_all(&0u16.to_le_bytes())?; // comment length
    writer.write_all(&0u16.to_le_bytes())?; // disk number start
    writer.write_all(&0u16.to_le_bytes())?; // internal file attrs
    writer.write_all(&0u32.to_le_bytes())?; // external file attrs
    writer.write_all(&entry.offset.to_le_bytes())?; // local header offset
    writer.write_all(name_bytes)?; // file name
    Ok(())
}

fn write_zip_eocd<W: Write>(
    writer: &mut W,
    num_entries: u16,
    cd_size: u32,
    cd_offset: u32,
) -> Result<()> {
    writer.write_all(&[0x50, 0x4b, 0x05, 0x06])?; // EOCD signature
    writer.write_all(&0u16.to_le_bytes())?; // disk number
    writer.write_all(&0u16.to_le_bytes())?; // disk with CD
    writer.write_all(&num_entries.to_le_bytes())?; // entries on disk
    writer.write_all(&num_entries.to_le_bytes())?; // total entries
    writer.write_all(&cd_size.to_le_bytes())?; // CD size
    writer.write_all(&cd_offset.to_le_bytes())?; // CD offset
    writer.write_all(&0u16.to_le_bytes())?; // comment length
    Ok(())
}

/// Simple CRC-32 (ISO 3309 / ITU-T V.42) without lookup table.
fn crc32_simple(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in data {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB8_8320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::npy::read_npy;
    use crate::npy::read_npz;
    use std::io::Cursor;

    #[test]
    fn test_write_read_npy_roundtrip() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_vec(data.clone(), vec![2, 3]).unwrap();

        let mut buf = Vec::new();
        write_npy(&mut buf, &tensor).unwrap();

        let result = read_npy(&buf[..]).unwrap();
        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(result.as_slice(), &data[..]);
    }

    #[test]
    fn test_write_read_npy_1d() {
        let data = vec![10.0, 20.0, 30.0];
        let tensor = Tensor::from_vec(data.clone(), vec![3]).unwrap();

        let mut buf = Vec::new();
        write_npy(&mut buf, &tensor).unwrap();

        let result = read_npy(&buf[..]).unwrap();
        assert_eq!(result.shape(), &[3]);
        assert_eq!(result.as_slice(), &data[..]);
    }

    #[test]
    fn test_write_read_npy_3d() {
        let data: Vec<f64> = (0..24).map(f64::from).collect();
        let tensor = Tensor::from_vec(data.clone(), vec![2, 3, 4]).unwrap();

        let mut buf = Vec::new();
        write_npy(&mut buf, &tensor).unwrap();

        let result = read_npy(&buf[..]).unwrap();
        assert_eq!(result.shape(), &[2, 3, 4]);
        assert_eq!(result.as_slice(), &data[..]);
    }

    #[test]
    fn test_write_read_npz_roundtrip() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let t2 = Tensor::from_vec(vec![4.0, 5.0, 6.0, 7.0], vec![2, 2]).unwrap();

        let mut tensors = HashMap::new();
        tensors.insert("arr_0".to_string(), t1);
        tensors.insert("arr_1".to_string(), t2);

        let mut buf = Cursor::new(Vec::new());
        write_npz(&mut buf, &tensors).unwrap();

        buf.set_position(0);
        let result = read_npz(buf).unwrap();

        assert_eq!(result.len(), 2);
        assert_eq!(result["arr_0"].as_slice(), &[1.0, 2.0, 3.0]);
        assert_eq!(result["arr_0"].shape(), &[3]);
        assert_eq!(result["arr_1"].as_slice(), &[4.0, 5.0, 6.0, 7.0]);
        assert_eq!(result["arr_1"].shape(), &[2, 2]);
    }

    #[test]
    fn test_write_read_npy_path_roundtrip() {
        let data = vec![1.5, 2.5, 3.5];
        let tensor = Tensor::from_vec(data.clone(), vec![3]).unwrap();

        let dir = std::env::temp_dir();
        let path = dir.join("scivex_test_roundtrip.npy");

        write_npy_path(&path, &tensor).unwrap();
        let result = crate::npy::read_npy_path(&path).unwrap();

        assert_eq!(result.as_slice(), &data[..]);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_write_read_npz_path_roundtrip() {
        let mut tensors = HashMap::new();
        tensors.insert(
            "x".to_string(),
            Tensor::from_vec(vec![1.0, 2.0], vec![2]).unwrap(),
        );

        let dir = std::env::temp_dir();
        let path = dir.join("scivex_test_roundtrip.npz");

        write_npz_path(&path, &tensors).unwrap();
        let result = crate::npy::read_npz_path(&path).unwrap();

        assert_eq!(result["x"].as_slice(), &[1.0, 2.0]);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_crc32_simple() {
        // Known CRC-32 values
        assert_eq!(crc32_simple(b""), 0x0000_0000);
        assert_eq!(crc32_simple(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn test_npy_header_alignment() {
        let header = build_npy_header(&[3, 4]);
        // Total length should be a multiple of 64
        assert_eq!(header.len() % 64, 0, "header len = {}", header.len());
    }
}
