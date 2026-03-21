//! Read `.npy` and `.npz` files into [`Tensor<f64>`].

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek};
use std::path::Path;

use scivex_core::Tensor;

use crate::error::{IoError, Result};

/// NPY magic bytes: `\x93NUMPY`.
const NPY_MAGIC: &[u8; 6] = b"\x93NUMPY";

/// Parsed NPY header fields.
struct NpyHeader {
    descr: String,
    fortran_order: bool,
    shape: Vec<usize>,
}

/// Read a `.npy` file from a reader into a `Tensor<f64>`.
///
/// Supports `<f4`, `<f8`, `<i4`, `<i8`, `>f4`, `>f8`, `>i4`, `>i8` dtypes.
/// Non-f64 data is converted to f64.
///
/// # Examples
///
/// ```
/// use scivex_io::npy::{read_npy, write_npy};
/// use scivex_core::Tensor;
///
/// let tensor = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0], vec![3]).unwrap();
/// let mut buf = Vec::new();
/// write_npy(&mut buf, &tensor).unwrap();
/// let result = read_npy(&buf[..]).unwrap();
/// assert_eq!(result.as_slice(), &[1.0_f64, 2.0, 3.0]);
/// ```
pub fn read_npy<R: Read>(mut reader: R) -> Result<Tensor<f64>> {
    let header = read_npy_header(&mut reader)?;
    read_npy_data(&mut reader, &header)
}

/// Read a `.npy` file from a path into a `Tensor<f64>`.
///
/// # Examples
///
/// ```ignore
/// use scivex_io::npy::read_npy_path;
/// let tensor = read_npy_path("array.npy").unwrap();
/// assert_eq!(tensor.shape(), &[3, 4]);
/// ```
pub fn read_npy_path<P: AsRef<Path>>(path: P) -> Result<Tensor<f64>> {
    let file = File::open(path)?;
    read_npy(BufReader::new(file))
}

/// Read a `.npz` file from a reader into a map of named `Tensor<f64>`.
///
/// Each entry in the ZIP archive is expected to be a `.npy` file.
///
/// # Examples
///
/// ```
/// use std::io::Cursor;
/// use std::collections::HashMap;
/// use scivex_io::npy::{read_npz, write_npz};
/// use scivex_core::Tensor;
///
/// let t = Tensor::from_vec(vec![1.0_f64, 2.0], vec![2]).unwrap();
/// let mut map = HashMap::new();
/// map.insert("arr".to_string(), t);
/// let mut buf = Cursor::new(Vec::new());
/// write_npz(&mut buf, &map).unwrap();
/// buf.set_position(0);
/// let result = read_npz(buf).unwrap();
/// assert_eq!(result["arr"].as_slice(), &[1.0_f64, 2.0]);
/// ```
pub fn read_npz<R: Read + Seek>(mut reader: R) -> Result<HashMap<String, Tensor<f64>>> {
    let mut result = HashMap::new();

    // Read ZIP directory from end of file
    let entries = read_zip_directory(&mut reader)?;

    for entry in &entries {
        reader
            .seek(std::io::SeekFrom::Start(entry.data_offset))
            .map_err(IoError::Io)?;

        let mut data = vec![0u8; entry.uncompressed_size as usize];
        reader.read_exact(&mut data)?;

        let tensor = read_npy(&data[..])?;

        // Strip .npy extension from name
        let name = entry
            .name
            .strip_suffix(".npy")
            .unwrap_or(&entry.name)
            .to_string();
        result.insert(name, tensor);
    }

    Ok(result)
}

/// Read a `.npz` file from a path into a map of named `Tensor<f64>`.
///
/// # Examples
///
/// ```ignore
/// use scivex_io::npy::read_npz_path;
/// let tensors = read_npz_path("arrays.npz").unwrap();
/// assert!(!tensors.is_empty());
/// ```
pub fn read_npz_path<P: AsRef<Path>>(path: P) -> Result<HashMap<String, Tensor<f64>>> {
    let file = File::open(path)?;
    read_npz(BufReader::new(file))
}

/// Read and parse the NPY header from a reader.
fn read_npy_header<R: Read>(reader: &mut R) -> Result<NpyHeader> {
    // Read magic bytes
    let mut magic = [0u8; 6];
    reader.read_exact(&mut magic)?;
    if &magic != NPY_MAGIC {
        return Err(IoError::InvalidHeader {
            reason: "not a valid .npy file (bad magic bytes)".to_string(),
        });
    }

    // Read version
    let mut version = [0u8; 2];
    reader.read_exact(&mut version)?;
    let major = version[0];

    // Read header length
    let header_len = match major {
        1 => {
            let mut buf = [0u8; 2];
            reader.read_exact(&mut buf)?;
            u16::from_le_bytes(buf) as usize
        }
        2 | 3 => {
            let mut buf = [0u8; 4];
            reader.read_exact(&mut buf)?;
            u32::from_le_bytes(buf) as usize
        }
        _ => {
            return Err(IoError::InvalidHeader {
                reason: format!("unsupported .npy version {major}"),
            });
        }
    };

    // Read header string
    let mut header_bytes = vec![0u8; header_len];
    reader.read_exact(&mut header_bytes)?;
    let header_str = String::from_utf8_lossy(&header_bytes);

    parse_npy_header(&header_str)
}

/// Parse the Python dict header string.
///
/// Expected format: `{'descr': '<f8', 'fortran_order': False, 'shape': (3, 4), }`
fn parse_npy_header(header: &str) -> Result<NpyHeader> {
    let header = header.trim().trim_matches(|c| c == '{' || c == '}');

    let mut descr = String::new();
    let mut fortran_order = false;
    let mut shape = Vec::new();

    for part in split_header_fields(header) {
        let part = part.trim();
        if let Some((key, value)) = part.split_once(':') {
            let key = key.trim().trim_matches('\'').trim_matches('"');
            let value = value.trim();

            match key {
                "descr" => {
                    descr = value.trim_matches('\'').trim_matches('"').to_string();
                }
                "fortran_order" => {
                    fortran_order = value == "True";
                }
                "shape" => {
                    let shape_str = value.trim_matches(|c| c == '(' || c == ')');
                    shape = shape_str
                        .split(',')
                        .filter_map(|s| {
                            let s = s.trim();
                            if s.is_empty() {
                                None
                            } else {
                                Some(s.parse::<usize>().map_err(|_| IoError::InvalidHeader {
                                    reason: format!("invalid shape dimension: {s}"),
                                }))
                            }
                        })
                        .collect::<Result<Vec<_>>>()?;
                }
                _ => {} // Ignore unknown keys
            }
        }
    }

    if descr.is_empty() {
        return Err(IoError::InvalidHeader {
            reason: "missing 'descr' in .npy header".to_string(),
        });
    }

    Ok(NpyHeader {
        descr,
        fortran_order,
        shape,
    })
}

/// Split header fields by commas, respecting parentheses for shape tuples.
fn split_header_fields(header: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut current = String::new();
    let mut paren_depth = 0;

    for ch in header.chars() {
        match ch {
            '(' => {
                paren_depth += 1;
                current.push(ch);
            }
            ')' => {
                paren_depth -= 1;
                current.push(ch);
            }
            ',' if paren_depth == 0 => {
                fields.push(current.clone());
                current.clear();
            }
            _ => current.push(ch),
        }
    }
    if !current.trim().is_empty() {
        fields.push(current);
    }
    fields
}

/// Read raw data from a reader and build a Tensor<f64>.
#[allow(clippy::cast_precision_loss)] // i64→f64 precision loss is expected for NumPy interop
fn read_npy_data<R: Read>(reader: &mut R, header: &NpyHeader) -> Result<Tensor<f64>> {
    let numel: usize = if header.shape.is_empty() {
        1 // scalar
    } else {
        header.shape.iter().product()
    };

    let data = match header.descr.as_str() {
        "<f8" | "=f8" => read_f64_le(reader, numel)?,
        ">f8" => read_f64_be(reader, numel)?,
        "<f4" | "=f4" => {
            let f32_data = read_f32_le(reader, numel)?;
            f32_data.into_iter().map(f64::from).collect()
        }
        ">f4" => {
            let f32_data = read_f32_be(reader, numel)?;
            f32_data.into_iter().map(f64::from).collect()
        }
        "<i8" | "=i8" => {
            let i64_data = read_i64_le(reader, numel)?;
            i64_data.into_iter().map(|v| v as f64).collect()
        }
        ">i8" => {
            let i64_data = read_i64_be(reader, numel)?;
            i64_data.into_iter().map(|v| v as f64).collect()
        }
        "<i4" | "=i4" => {
            let i32_data = read_i32_le(reader, numel)?;
            i32_data.into_iter().map(f64::from).collect()
        }
        ">i4" => {
            let i32_data = read_i32_be(reader, numel)?;
            i32_data.into_iter().map(f64::from).collect()
        }
        other => {
            return Err(IoError::InvalidHeader {
                reason: format!("unsupported dtype: {other}"),
            });
        }
    };

    let mut result_data = data;

    // Handle Fortran order by transposing to C order
    if header.fortran_order && header.shape.len() >= 2 {
        result_data = fortran_to_c_order(&result_data, &header.shape);
    }

    let shape = if header.shape.is_empty() {
        vec![1]
    } else {
        header.shape.clone()
    };

    Tensor::from_vec(result_data, shape).map_err(IoError::CoreError)
}

/// Convert Fortran-order (column-major) data to C-order (row-major).
fn fortran_to_c_order(data: &[f64], shape: &[usize]) -> Vec<f64> {
    let ndim = shape.len();
    let numel = data.len();
    let mut result = vec![0.0; numel];

    // Compute strides for C-order and Fortran-order
    let mut c_strides = vec![1usize; ndim];
    for i in (0..ndim - 1).rev() {
        c_strides[i] = c_strides[i + 1] * shape[i + 1];
    }

    let mut f_strides = vec![1usize; ndim];
    for i in 1..ndim {
        f_strides[i] = f_strides[i - 1] * shape[i - 1];
    }

    for (flat_c, dest) in result.iter_mut().enumerate() {
        // Decompose C-order flat index into multi-index
        let mut remaining = flat_c;
        let mut flat_f = 0;
        for d in 0..ndim {
            let idx = remaining / c_strides[d];
            remaining %= c_strides[d];
            flat_f += idx * f_strides[d];
        }
        *dest = data[flat_f];
    }

    result
}

// --- Raw binary readers ---

fn read_f64_le<R: Read>(reader: &mut R, n: usize) -> Result<Vec<f64>> {
    let mut buf = vec![0u8; n * 8];
    reader.read_exact(&mut buf)?;
    Ok((0..n)
        .map(|i| f64::from_le_bytes(buf[i * 8..(i + 1) * 8].try_into().unwrap()))
        .collect())
}

fn read_f64_be<R: Read>(reader: &mut R, n: usize) -> Result<Vec<f64>> {
    let mut buf = vec![0u8; n * 8];
    reader.read_exact(&mut buf)?;
    Ok((0..n)
        .map(|i| f64::from_be_bytes(buf[i * 8..(i + 1) * 8].try_into().unwrap()))
        .collect())
}

fn read_f32_le<R: Read>(reader: &mut R, n: usize) -> Result<Vec<f32>> {
    let mut buf = vec![0u8; n * 4];
    reader.read_exact(&mut buf)?;
    Ok((0..n)
        .map(|i| f32::from_le_bytes(buf[i * 4..(i + 1) * 4].try_into().unwrap()))
        .collect())
}

fn read_f32_be<R: Read>(reader: &mut R, n: usize) -> Result<Vec<f32>> {
    let mut buf = vec![0u8; n * 4];
    reader.read_exact(&mut buf)?;
    Ok((0..n)
        .map(|i| f32::from_be_bytes(buf[i * 4..(i + 1) * 4].try_into().unwrap()))
        .collect())
}

fn read_i64_le<R: Read>(reader: &mut R, n: usize) -> Result<Vec<i64>> {
    let mut buf = vec![0u8; n * 8];
    reader.read_exact(&mut buf)?;
    Ok((0..n)
        .map(|i| i64::from_le_bytes(buf[i * 8..(i + 1) * 8].try_into().unwrap()))
        .collect())
}

fn read_i64_be<R: Read>(reader: &mut R, n: usize) -> Result<Vec<i64>> {
    let mut buf = vec![0u8; n * 8];
    reader.read_exact(&mut buf)?;
    Ok((0..n)
        .map(|i| i64::from_be_bytes(buf[i * 8..(i + 1) * 8].try_into().unwrap()))
        .collect())
}

fn read_i32_le<R: Read>(reader: &mut R, n: usize) -> Result<Vec<i32>> {
    let mut buf = vec![0u8; n * 4];
    reader.read_exact(&mut buf)?;
    Ok((0..n)
        .map(|i| i32::from_le_bytes(buf[i * 4..(i + 1) * 4].try_into().unwrap()))
        .collect())
}

fn read_i32_be<R: Read>(reader: &mut R, n: usize) -> Result<Vec<i32>> {
    let mut buf = vec![0u8; n * 4];
    reader.read_exact(&mut buf)?;
    Ok((0..n)
        .map(|i| i32::from_be_bytes(buf[i * 4..(i + 1) * 4].try_into().unwrap()))
        .collect())
}

// --- Minimal ZIP reader ---

struct ZipEntry {
    name: String,
    data_offset: u64,
    uncompressed_size: u32,
}

/// Read the ZIP central directory to find entries.
///
/// Only supports uncompressed (stored) entries, which is what NumPy uses for .npz.
fn read_zip_directory<R: Read + Seek>(reader: &mut R) -> Result<Vec<ZipEntry>> {
    // Find the End of Central Directory record (EOCD).
    // It's at most 65535 + 22 bytes from the end.
    let file_len = reader.seek(std::io::SeekFrom::End(0))?;
    let search_start = file_len.saturating_sub(65557);
    reader.seek(std::io::SeekFrom::Start(search_start))?;

    let mut tail = Vec::new();
    reader.read_to_end(&mut tail)?;

    // Find EOCD signature: 0x06054b50
    let eocd_sig = [0x50, 0x4b, 0x05, 0x06];
    let eocd_pos = tail
        .windows(4)
        .rposition(|w| w == eocd_sig)
        .ok_or_else(|| IoError::InvalidHeader {
            reason: "not a valid .npz file (no ZIP EOCD found)".to_string(),
        })?;

    let eocd = &tail[eocd_pos..];
    if eocd.len() < 22 {
        return Err(IoError::InvalidHeader {
            reason: "truncated ZIP EOCD".to_string(),
        });
    }

    let num_entries = u16::from_le_bytes([eocd[10], eocd[11]]) as usize;
    let cd_offset = u64::from(u32::from_le_bytes([eocd[16], eocd[17], eocd[18], eocd[19]]));

    // Read central directory entries
    reader.seek(std::io::SeekFrom::Start(cd_offset))?;
    let mut entries = Vec::with_capacity(num_entries);

    for _ in 0..num_entries {
        let mut sig = [0u8; 4];
        reader.read_exact(&mut sig)?;
        if sig != [0x50, 0x4b, 0x01, 0x02] {
            return Err(IoError::InvalidHeader {
                reason: "invalid ZIP central directory entry".to_string(),
            });
        }

        let mut cd_header = [0u8; 42]; // bytes 4..46 of central dir entry
        reader.read_exact(&mut cd_header)?;

        let compression = u16::from_le_bytes([cd_header[6], cd_header[7]]);
        let uncompressed_size =
            u32::from_le_bytes([cd_header[20], cd_header[21], cd_header[22], cd_header[23]]);
        let name_len = u16::from_le_bytes([cd_header[24], cd_header[25]]) as usize;
        let extra_len = u16::from_le_bytes([cd_header[26], cd_header[27]]) as usize;
        let comment_len = u16::from_le_bytes([cd_header[28], cd_header[29]]) as usize;
        let local_header_offset = u64::from(u32::from_le_bytes([
            cd_header[38],
            cd_header[39],
            cd_header[40],
            cd_header[41],
        ]));

        let mut name_bytes = vec![0u8; name_len];
        reader.read_exact(&mut name_bytes)?;
        let name = String::from_utf8_lossy(&name_bytes).to_string();

        // Skip extra and comment fields
        let skip = extra_len + comment_len;
        if skip > 0 {
            let mut discard = vec![0u8; skip];
            reader.read_exact(&mut discard)?;
        }

        if compression != 0 {
            return Err(IoError::InvalidHeader {
                reason: format!("compressed .npz entries not supported (entry: {name})"),
            });
        }

        // The data offset is local_header_offset + 30 + name_len + extra_len_local.
        // We need to read the local file header to get the local extra field length.
        let saved_pos = reader.stream_position()?;
        reader.seek(std::io::SeekFrom::Start(local_header_offset))?;

        let mut local_sig = [0u8; 4];
        reader.read_exact(&mut local_sig)?;
        if local_sig != [0x50, 0x4b, 0x03, 0x04] {
            return Err(IoError::InvalidHeader {
                reason: "invalid ZIP local file header".to_string(),
            });
        }

        let mut local_header = [0u8; 26];
        reader.read_exact(&mut local_header)?;
        let local_name_len = u64::from(u16::from_le_bytes([local_header[22], local_header[23]]));
        let local_extra_len = u64::from(u16::from_le_bytes([local_header[24], local_header[25]]));

        let data_offset = local_header_offset + 30 + local_name_len + local_extra_len;

        reader.seek(std::io::SeekFrom::Start(saved_pos))?;

        entries.push(ZipEntry {
            name,
            data_offset,
            uncompressed_size,
        });
    }

    Ok(entries)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_npy_header_basic() {
        let header = "{'descr': '<f8', 'fortran_order': False, 'shape': (3, 4), }";
        let parsed = parse_npy_header(header).unwrap();
        assert_eq!(parsed.descr, "<f8");
        assert!(!parsed.fortran_order);
        assert_eq!(parsed.shape, vec![3, 4]);
    }

    #[test]
    fn test_parse_npy_header_1d() {
        let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (10,), }";
        let parsed = parse_npy_header(header).unwrap();
        assert_eq!(parsed.descr, "<f4");
        assert_eq!(parsed.shape, vec![10]);
    }

    #[test]
    fn test_parse_npy_header_scalar() {
        let header = "{'descr': '<f8', 'fortran_order': False, 'shape': (), }";
        let parsed = parse_npy_header(header).unwrap();
        assert_eq!(parsed.shape, Vec::<usize>::new());
    }

    #[test]
    fn test_parse_npy_header_fortran() {
        let header = "{'descr': '<f8', 'fortran_order': True, 'shape': (2, 3), }";
        let parsed = parse_npy_header(header).unwrap();
        assert!(parsed.fortran_order);
    }

    #[test]
    fn test_read_npy_f64_le() {
        // Build a valid .npy file in memory
        let shape = vec![2, 3];
        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let npy_bytes = build_test_npy("<f8", &shape, &data_to_bytes_f64_le(&data));

        let tensor = read_npy(&npy_bytes[..]).unwrap();
        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_read_npy_f32_le() {
        let shape = vec![4];
        let data: Vec<f32> = vec![1.0, 2.5, 3.0, 4.5];
        let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let npy_bytes = build_test_npy("<f4", &shape, &raw);

        let tensor = read_npy(&npy_bytes[..]).unwrap();
        assert_eq!(tensor.shape(), &[4]);
        assert!((tensor.as_slice()[0] - 1.0).abs() < 1e-6);
        assert!((tensor.as_slice()[1] - 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_read_npy_i32_le() {
        let shape = vec![3];
        let data: Vec<i32> = vec![10, 20, 30];
        let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let npy_bytes = build_test_npy("<i4", &shape, &raw);

        let tensor = read_npy(&npy_bytes[..]).unwrap();
        assert_eq!(tensor.as_slice(), &[10.0, 20.0, 30.0]);
    }

    #[test]
    fn test_read_npy_invalid_magic() {
        let data = b"NOT_NPY_FILE";
        let result = read_npy(&data[..]);
        assert!(result.is_err());
    }

    #[test]
    fn test_fortran_to_c_order() {
        // 2x3 matrix in Fortran order: column-major
        // Fortran: [1, 3, 5, 2, 4, 6] → C: [1, 2, 3, 4, 5, 6]
        // Wait, Fortran stores column-by-column:
        // col0=[1,4], col1=[2,5], col2=[3,6] for a 2x3 matrix
        // So Fortran flat: [1, 4, 2, 5, 3, 6]
        // C flat: [1, 2, 3, 4, 5, 6]
        let fortran_data = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let c_data = fortran_to_c_order(&fortran_data, &[2, 3]);
        assert_eq!(c_data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    // --- Test helpers ---

    fn build_test_npy(descr: &str, shape: &[usize], raw_data: &[u8]) -> Vec<u8> {
        let shape_str = if shape.len() == 1 {
            format!("({},)", shape[0])
        } else {
            let parts: Vec<String> = shape.iter().map(std::string::ToString::to_string).collect();
            format!("({})", parts.join(", "))
        };
        let header_str =
            format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape_str}, }}");

        // Pad header to align total (magic + version + header_len + header) to 64 bytes
        let prefix_len = 6 + 2 + 2; // magic + version + header_len (v1)
        let mut padded = header_str.clone();
        while (prefix_len + padded.len()) % 64 != 63 {
            padded.push(' ');
        }
        padded.push('\n');

        let header_len = padded.len() as u16;

        let mut buf = Vec::new();
        buf.extend_from_slice(NPY_MAGIC);
        buf.push(1); // major version
        buf.push(0); // minor version
        buf.extend_from_slice(&header_len.to_le_bytes());
        buf.extend_from_slice(padded.as_bytes());
        buf.extend_from_slice(raw_data);
        buf
    }

    fn data_to_bytes_f64_le(data: &[f64]) -> Vec<u8> {
        data.iter().flat_map(|v| v.to_le_bytes()).collect()
    }
}
