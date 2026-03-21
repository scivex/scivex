//! Memory-mapped tensor reader for `.npy` files.

use std::fs::File;
use std::path::Path;

use memmap2::Mmap;
use scivex_core::Tensor;

use crate::error::{IoError, Result};

/// A memory-mapped tensor reader that lazily accesses data from a `.npy` file.
///
/// The file is memory-mapped rather than fully loaded, so only the accessed
/// pages are brought into memory by the OS.
///
/// # Examples
///
/// ```ignore
/// use scivex_io::mmap::MmapTensorReader;
/// let reader = MmapTensorReader::open("large_array.npy").unwrap();
/// println!("shape: {:?}, elements: {}", reader.shape(), reader.numel());
/// let tensor = reader.read_tensor().unwrap();
/// ```
pub struct MmapTensorReader {
    mmap: Mmap,
    shape: Vec<usize>,
    data_offset: usize,
    numel: usize,
}

impl MmapTensorReader {
    /// Open a `.npy` file and memory-map it.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        // SAFETY: We only read from the memory-mapped region. The file must
        // not be modified while the mapping is live.
        let mmap = unsafe { Mmap::map(&file) }.map_err(IoError::Io)?;
        Self::from_mmap(mmap)
    }

    /// Create a reader from an existing `Mmap`.
    fn from_mmap(mmap: Mmap) -> Result<Self> {
        let data = &mmap[..];
        if data.len() < 10 {
            return Err(IoError::FormatError("file too small for npy".into()));
        }

        // Parse the NPY header to find shape and data offset.
        let (shape, data_offset) = parse_npy_header_for_mmap(data)?;
        let numel: usize = shape.iter().product();

        Ok(Self {
            mmap,
            shape,
            data_offset,
            numel,
        })
    }

    /// Shape of the stored tensor.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Number of elements.
    pub fn numel(&self) -> usize {
        self.numel
    }

    /// Read the full tensor into memory as `Tensor<f64>`.
    pub fn read_tensor(&self) -> Result<Tensor<f64>> {
        let bytes = &self.mmap[self.data_offset..];
        let expected_bytes = self.numel * 8;
        if bytes.len() < expected_bytes {
            return Err(IoError::FormatError(
                "npy file data section too small".into(),
            ));
        }

        let mut data = Vec::with_capacity(self.numel);
        for i in 0..self.numel {
            let offset = i * 8;
            let val = f64::from_le_bytes(
                bytes[offset..offset + 8]
                    .try_into()
                    .map_err(|_| IoError::FormatError("invalid f64 bytes".into()))?,
            );
            data.push(val);
        }

        Tensor::from_vec(data, self.shape.clone()).map_err(IoError::CoreError)
    }

    /// Read a slice of elements starting at `offset` with `count` elements.
    pub fn read_slice(&self, offset: usize, count: usize) -> Result<Vec<f64>> {
        if offset + count > self.numel {
            return Err(IoError::FormatError("slice out of bounds".into()));
        }

        let bytes = &self.mmap[self.data_offset..];
        let mut data = Vec::with_capacity(count);
        for i in offset..offset + count {
            let byte_off = i * 8;
            let val = f64::from_le_bytes(
                bytes[byte_off..byte_off + 8]
                    .try_into()
                    .map_err(|_| IoError::FormatError("invalid f64 bytes".into()))?,
            );
            data.push(val);
        }
        Ok(data)
    }
}

/// Memory-map a `.npy` file and read the tensor.
///
/// # Examples
///
/// ```ignore
/// use scivex_io::mmap::mmap_npy;
/// let tensor = mmap_npy("data.npy").unwrap();
/// assert_eq!(tensor.shape(), &[100, 200]);
/// ```
pub fn mmap_npy<P: AsRef<Path>>(path: P) -> Result<Tensor<f64>> {
    let reader = MmapTensorReader::open(path)?;
    reader.read_tensor()
}

/// Parse a npy header from raw bytes, returning (shape, data_offset).
fn parse_npy_header_for_mmap(data: &[u8]) -> Result<(Vec<usize>, usize)> {
    // Verify magic bytes
    if data.len() < 10 || &data[0..6] != b"\x93NUMPY" {
        return Err(IoError::FormatError("invalid npy magic bytes".into()));
    }

    let major = data[6];
    let header_len_offset = 8;
    let data_start;
    let header_str;

    if major == 1 {
        let header_len = u16::from_le_bytes([data[header_len_offset], data[header_len_offset + 1]]);
        let hdr_start = 10;
        let hdr_end = hdr_start + usize::from(header_len);
        if data.len() < hdr_end {
            return Err(IoError::FormatError("npy header truncated".into()));
        }
        header_str = std::str::from_utf8(&data[hdr_start..hdr_end])
            .map_err(|_| IoError::FormatError("npy header not utf8".into()))?;
        data_start = hdr_end;
    } else if major == 2 {
        if data.len() < 12 {
            return Err(IoError::FormatError("npy v2 header truncated".into()));
        }
        let header_len = u32::from_le_bytes([
            data[header_len_offset],
            data[header_len_offset + 1],
            data[header_len_offset + 2],
            data[header_len_offset + 3],
        ]);
        let hdr_start = 12;
        let hdr_end = hdr_start + header_len as usize;
        if data.len() < hdr_end {
            return Err(IoError::FormatError("npy v2 header truncated".into()));
        }
        header_str = std::str::from_utf8(&data[hdr_start..hdr_end])
            .map_err(|_| IoError::FormatError("npy header not utf8".into()))?;
        data_start = hdr_end;
    } else {
        return Err(IoError::FormatError(format!(
            "unsupported npy version {major}"
        )));
    }

    // Parse shape from header string like: {'descr': '<f8', 'fortran_order': False, 'shape': (3, 4), }
    let shape = parse_shape_from_header(header_str)?;

    Ok((shape, data_start))
}

/// Extract the shape tuple from a npy header string.
fn parse_shape_from_header(header: &str) -> Result<Vec<usize>> {
    let shape_key = "'shape':";
    let shape_start = header
        .find(shape_key)
        .ok_or_else(|| IoError::FormatError("no shape in npy header".into()))?;
    let after = &header[shape_start + shape_key.len()..];

    let paren_start = after
        .find('(')
        .ok_or_else(|| IoError::FormatError("no '(' in shape".into()))?;
    let paren_end = after
        .find(')')
        .ok_or_else(|| IoError::FormatError("no ')' in shape".into()))?;

    let shape_body = after[paren_start + 1..paren_end].trim();
    if shape_body.is_empty() {
        return Ok(vec![]);
    }

    let dims: Result<Vec<usize>> = shape_body
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse::<usize>()
                .map_err(|_| IoError::FormatError(format!("invalid shape dimension: {s}")))
        })
        .collect();
    dims
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_test_npy(data: &[f64], shape: &[usize]) -> std::path::PathBuf {
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "scivex_mmap_test_{}.npy",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));

        // Build a simple npy file
        let shape_str = if shape.len() == 1 {
            format!("({},)", shape[0])
        } else {
            let parts: Vec<String> = shape.iter().map(std::string::ToString::to_string).collect();
            format!("({})", parts.join(", "))
        };

        let header_str =
            format!("{{'descr': '<f8', 'fortran_order': False, 'shape': {shape_str}, }}");
        let prefix_len = 10; // magic(6) + version(2) + header_len(2)
        let mut padded = header_str;
        while (prefix_len + padded.len()) % 64 != 63 {
            padded.push(' ');
        }
        padded.push('\n');

        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(b"\x93NUMPY").unwrap();
        file.write_all(&[1u8, 0u8]).unwrap();
        file.write_all(&(padded.len() as u16).to_le_bytes())
            .unwrap();
        file.write_all(padded.as_bytes()).unwrap();
        for &v in data {
            file.write_all(&v.to_le_bytes()).unwrap();
        }

        path
    }

    #[test]
    fn test_mmap_npy_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let path = write_test_npy(&data, &[2, 3]);

        let tensor = mmap_npy(&path).unwrap();
        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.as_slice(), &data[..]);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_mmap_reader_shape() {
        let data = vec![1.0, 2.0, 3.0];
        let path = write_test_npy(&data, &[3]);

        let reader = MmapTensorReader::open(&path).unwrap();
        assert_eq!(reader.shape(), &[3]);
        assert_eq!(reader.numel(), 3);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_mmap_read_slice() {
        let data = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let path = write_test_npy(&data, &[5]);

        let reader = MmapTensorReader::open(&path).unwrap();
        let slice = reader.read_slice(1, 3).unwrap();
        assert_eq!(slice, vec![20.0, 30.0, 40.0]);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_mmap_read_slice_out_of_bounds() {
        let data = vec![1.0, 2.0, 3.0];
        let path = write_test_npy(&data, &[3]);

        let reader = MmapTensorReader::open(&path).unwrap();
        assert!(reader.read_slice(2, 5).is_err());

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_mmap_npy_1d() {
        let data = vec![42.0];
        let path = write_test_npy(&data, &[1]);

        let tensor = mmap_npy(&path).unwrap();
        assert_eq!(tensor.shape(), &[1]);
        assert_eq!(tensor.as_slice(), &[42.0]);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_parse_shape_from_header() {
        assert_eq!(
            parse_shape_from_header("'shape': (3, 4), ").unwrap(),
            vec![3, 4]
        );
        assert_eq!(parse_shape_from_header("'shape': (5,), ").unwrap(), vec![5]);
        assert_eq!(
            parse_shape_from_header("'shape': (), ").unwrap(),
            Vec::<usize>::new()
        );
    }
}
