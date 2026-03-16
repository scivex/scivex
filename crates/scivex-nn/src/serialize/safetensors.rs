//! SafeTensors format — HuggingFace's standard for storing model weights.
//!
//! ## Format
//!
//! ```text
//! [8 bytes]  Header size (little-endian u64)
//! [N bytes]  JSON header: {"tensor_name": {"dtype": "F32", "shape": [2,3], "data_offsets": [0, 24]}, ...}
//! [M bytes]  Concatenated raw tensor data
//! ```
//!
//! Supported dtypes: `F32`, `F64`, `I32`, `I64`.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};

use scivex_core::{Float, Tensor};

use crate::error::{NnError, Result};

// ---------------------------------------------------------------------------
// Dtype helpers
// ---------------------------------------------------------------------------

/// Return the dtype string and element size for the concrete float type `T`.
fn dtype_info<T: Float>() -> (&'static str, usize) {
    let size = std::mem::size_of::<T>();
    match size {
        4 => ("F32", 4),
        _ => ("F64", 8), // 8 bytes or fallback
    }
}

fn element_size_for_dtype(dtype: &str) -> Result<usize> {
    match dtype {
        "F32" | "I32" => Ok(4),
        "F64" | "I64" => Ok(8),
        _ => Err(ser_err(&format!("unsupported dtype: {dtype}"))),
    }
}

fn ser_err(msg: &str) -> NnError {
    NnError::SerializeError(msg.to_string())
}

// ---------------------------------------------------------------------------
// Minimal JSON generation
// ---------------------------------------------------------------------------

fn build_header_json<T: Float>(tensors: &[(&str, &Tensor<T>)]) -> (String, Vec<u64>) {
    let (dtype_str, elem_size) = dtype_info::<T>();
    let mut entries = Vec::with_capacity(tensors.len());
    let mut offset: u64 = 0;
    let mut offsets = Vec::with_capacity(tensors.len());

    for &(name, tensor) in tensors {
        let numel: usize = tensor.shape().iter().product();
        let byte_len = (numel * elem_size) as u64;
        let start = offset;
        let end = offset + byte_len;
        offsets.push(start);

        let shape_str = format!(
            "[{}]",
            tensor
                .shape()
                .iter()
                .map(std::string::ToString::to_string)
                .collect::<Vec<_>>()
                .join(",")
        );
        entries.push(format!(
            "\"{}\":{{\"dtype\":\"{}\",\"shape\":{},\"data_offsets\":[{},{}]}}",
            escape_json_string(name),
            dtype_str,
            shape_str,
            start,
            end
        ));
        offset = end;
    }

    let json = format!("{{{}}}", entries.join(","));
    (json, offsets)
}

fn escape_json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            _ => out.push(ch),
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Minimal JSON parsing (enough for safetensors header)
// ---------------------------------------------------------------------------

struct JsonParser<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> JsonParser<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn skip_ws(&mut self) {
        while self.pos < self.data.len() {
            match self.data[self.pos] {
                b' ' | b'\t' | b'\n' | b'\r' => self.pos += 1,
                _ => break,
            }
        }
    }

    fn peek(&self) -> Option<u8> {
        self.data.get(self.pos).copied()
    }

    fn consume(&mut self, expected: u8) -> Result<()> {
        self.skip_ws();
        if self.peek() == Some(expected) {
            self.pos += 1;
            Ok(())
        } else {
            Err(ser_err(&format!(
                "expected '{}' at position {}",
                expected as char, self.pos
            )))
        }
    }

    fn parse_string(&mut self) -> Result<String> {
        self.skip_ws();
        self.consume(b'"')?;
        let mut s = String::new();
        loop {
            if self.pos >= self.data.len() {
                return Err(ser_err("unterminated string"));
            }
            let b = self.data[self.pos];
            self.pos += 1;
            if b == b'"' {
                return Ok(s);
            }
            if b == b'\\' {
                if self.pos >= self.data.len() {
                    return Err(ser_err("unterminated escape"));
                }
                let esc = self.data[self.pos];
                self.pos += 1;
                match esc {
                    b'"' => s.push('"'),
                    b'\\' => s.push('\\'),
                    b'n' => s.push('\n'),
                    b'r' => s.push('\r'),
                    b't' => s.push('\t'),
                    _ => {
                        s.push('\\');
                        s.push(esc as char);
                    }
                }
            } else {
                s.push(b as char);
            }
        }
    }

    fn parse_number(&mut self) -> Result<u64> {
        self.skip_ws();
        let start = self.pos;
        while self.pos < self.data.len() && self.data[self.pos].is_ascii_digit() {
            self.pos += 1;
        }
        if self.pos == start {
            return Err(ser_err("expected number"));
        }
        let s = std::str::from_utf8(&self.data[start..self.pos])
            .map_err(|_| ser_err("invalid utf8 in number"))?;
        s.parse::<u64>().map_err(|_| ser_err("invalid number"))
    }

    fn parse_u64_array(&mut self) -> Result<Vec<u64>> {
        self.skip_ws();
        self.consume(b'[')?;
        let mut vals = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(vals);
        }
        loop {
            vals.push(self.parse_number()?);
            self.skip_ws();
            if self.peek() == Some(b',') {
                self.pos += 1;
            } else {
                break;
            }
        }
        self.consume(b']')?;
        Ok(vals)
    }

    /// Parse a tensor entry: `{"dtype": "F32", "shape": [...], "data_offsets": [start, end]}`
    fn parse_tensor_entry(&mut self) -> Result<(String, Vec<usize>, u64, u64)> {
        self.skip_ws();
        self.consume(b'{')?;
        let mut dtype = String::new();
        let mut shape: Vec<usize> = Vec::new();
        let mut data_start: u64 = 0;
        let mut data_end: u64 = 0;

        loop {
            self.skip_ws();
            if self.peek() == Some(b'}') {
                self.pos += 1;
                break;
            }
            let key = self.parse_string()?;
            self.skip_ws();
            self.consume(b':')?;
            match key.as_str() {
                "dtype" => {
                    dtype = self.parse_string()?;
                }
                "shape" => {
                    shape = self
                        .parse_u64_array()?
                        .into_iter()
                        .map(|v| v as usize)
                        .collect();
                }
                "data_offsets" => {
                    let offsets = self.parse_u64_array()?;
                    if offsets.len() != 2 {
                        return Err(ser_err("data_offsets must have exactly 2 elements"));
                    }
                    data_start = offsets[0];
                    data_end = offsets[1];
                }
                _ => {
                    // skip unknown value
                    self.skip_value()?;
                }
            }
            self.skip_ws();
            if self.peek() == Some(b',') {
                self.pos += 1;
            }
        }

        Ok((dtype, shape, data_start, data_end))
    }

    /// Skip an arbitrary JSON value (string, number, object, array, bool, null).
    fn skip_value(&mut self) -> Result<()> {
        self.skip_ws();
        match self.peek() {
            Some(b'"') => {
                self.parse_string()?;
            }
            Some(b'{') => {
                self.skip_nested(b'{', b'}')?;
            }
            Some(b'[') => {
                self.skip_nested(b'[', b']')?;
            }
            _ => {
                // number, bool, null — consume until delimiter
                while self.pos < self.data.len() {
                    match self.data[self.pos] {
                        b',' | b'}' | b']' => break,
                        _ => self.pos += 1,
                    }
                }
            }
        }
        Ok(())
    }

    fn skip_nested(&mut self, open: u8, close: u8) -> Result<()> {
        self.consume(open)?;
        let mut depth = 1u32;
        let mut in_string = false;
        while self.pos < self.data.len() && depth > 0 {
            let b = self.data[self.pos];
            self.pos += 1;
            if in_string {
                if b == b'\\' {
                    self.pos += 1; // skip escaped char
                } else if b == b'"' {
                    in_string = false;
                }
            } else if b == b'"' {
                in_string = true;
            } else if b == open {
                depth += 1;
            } else if b == close {
                depth -= 1;
            }
        }
        Ok(())
    }

    /// Parse the full header: `{ "name1": {...}, "name2": {...}, ... }`
    /// Returns vec of (name, dtype, shape, data_start, data_end).
    #[allow(clippy::type_complexity)]
    fn parse_header(&mut self) -> Result<Vec<(String, String, Vec<usize>, u64, u64)>> {
        self.skip_ws();
        self.consume(b'{')?;
        let mut entries = Vec::new();
        loop {
            self.skip_ws();
            if self.peek() == Some(b'}') {
                self.pos += 1;
                break;
            }
            let name = self.parse_string()?;
            self.skip_ws();
            self.consume(b':')?;

            // Check if this is "__metadata__" (skip it)
            self.skip_ws();
            if name == "__metadata__" {
                self.skip_value()?;
            } else {
                let (dtype, shape, start, end) = self.parse_tensor_entry()?;
                entries.push((name, dtype, shape, start, end));
            }

            self.skip_ws();
            if self.peek() == Some(b',') {
                self.pos += 1;
            }
        }
        Ok(entries)
    }
}

// ---------------------------------------------------------------------------
// Read helpers
// ---------------------------------------------------------------------------

fn read_tensor_data<T: Float>(
    raw: &[u8],
    dtype: &str,
    shape: &[usize],
    start: usize,
    end: usize,
) -> Result<Tensor<T>> {
    let numel: usize = shape.iter().product();
    let byte_len = end - start;
    let elem_sz = element_size_for_dtype(dtype)?;

    if byte_len != numel * elem_sz {
        return Err(ser_err("tensor data length mismatch"));
    }
    if end > raw.len() {
        return Err(ser_err("tensor data out of bounds"));
    }

    let slice = &raw[start..end];
    let mut data = Vec::with_capacity(numel);

    match dtype {
        "F32" => {
            for i in 0..numel {
                let off = i * 4;
                let val = f32::from_le_bytes([
                    slice[off],
                    slice[off + 1],
                    slice[off + 2],
                    slice[off + 3],
                ]);
                data.push(T::from_f64(f64::from(val)));
            }
        }
        "F64" => {
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
                data.push(T::from_f64(val));
            }
        }
        "I32" => {
            for i in 0..numel {
                let off = i * 4;
                let val = i32::from_le_bytes([
                    slice[off],
                    slice[off + 1],
                    slice[off + 2],
                    slice[off + 3],
                ]);
                data.push(T::from_f64(f64::from(val)));
            }
        }
        "I64" => {
            for i in 0..numel {
                let off = i * 8;
                let val = i64::from_le_bytes([
                    slice[off],
                    slice[off + 1],
                    slice[off + 2],
                    slice[off + 3],
                    slice[off + 4],
                    slice[off + 5],
                    slice[off + 6],
                    slice[off + 7],
                ]);
                #[allow(clippy::cast_precision_loss)]
                data.push(T::from_f64(val as f64));
            }
        }
        _ => return Err(ser_err(&format!("unsupported dtype: {dtype}"))),
    }

    Tensor::from_vec(data, shape.to_vec()).map_err(|e| ser_err(&format!("tensor creation: {e}")))
}

fn write_tensor_bytes<T: Float>(w: &mut impl Write, tensor: &Tensor<T>) -> Result<()> {
    let (_, elem_size) = dtype_info::<T>();
    let values = tensor.as_slice();
    match elem_size {
        4 => {
            for &v in values {
                #[allow(clippy::cast_possible_truncation)]
                let f = v.to_f64() as f32;
                w.write_all(&f.to_le_bytes())
                    .map_err(|_| ser_err("write error"))?;
            }
        }
        8 => {
            for &v in values {
                let f = v.to_f64();
                w.write_all(&f.to_le_bytes())
                    .map_err(|_| ser_err("write error"))?;
            }
        }
        _ => return Err(ser_err("unsupported element size")),
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Save tensors to a file in SafeTensors format.
pub fn save_safetensors<T: Float>(path: &str, tensors: &[(&str, &Tensor<T>)]) -> Result<()> {
    let f = File::create(path).map_err(|_| ser_err("cannot create file"))?;
    let mut w = BufWriter::new(f);
    save_safetensors_to_writer(&mut w, tensors)
}

/// Load tensors from a SafeTensors file.
pub fn load_safetensors<T: Float>(path: &str) -> Result<Vec<(String, Tensor<T>)>> {
    let f = File::open(path).map_err(|_| ser_err("cannot open file"))?;
    let mut r = BufReader::new(f);
    load_safetensors_from_reader(&mut r)
}

/// Save tensors to a writer in SafeTensors format.
pub fn save_safetensors_to_writer<T: Float>(
    writer: &mut impl Write,
    tensors: &[(&str, &Tensor<T>)],
) -> Result<()> {
    let (header_json, _offsets) = build_header_json(tensors);
    let header_bytes = header_json.as_bytes();
    let header_size = header_bytes.len() as u64;

    writer
        .write_all(&header_size.to_le_bytes())
        .map_err(|_| ser_err("write error"))?;
    writer
        .write_all(header_bytes)
        .map_err(|_| ser_err("write error"))?;

    for &(_name, tensor) in tensors {
        write_tensor_bytes(writer, tensor)?;
    }

    writer.flush().map_err(|_| ser_err("flush error"))?;
    Ok(())
}

/// Load tensors from a reader in SafeTensors format.
pub fn load_safetensors_from_reader<T: Float>(
    reader: &mut impl Read,
) -> Result<Vec<(String, Tensor<T>)>> {
    // Read header size
    let mut size_buf = [0u8; 8];
    reader
        .read_exact(&mut size_buf)
        .map_err(|_| ser_err("cannot read header size"))?;
    let header_size = u64::from_le_bytes(size_buf) as usize;

    // Read header JSON
    let mut header_buf = vec![0u8; header_size];
    reader
        .read_exact(&mut header_buf)
        .map_err(|_| ser_err("cannot read header"))?;

    // Parse header
    let mut parser = JsonParser::new(&header_buf);
    let entries = parser.parse_header()?;

    // Read all tensor data
    let total_data: usize = entries
        .iter()
        .map(|(_, _, _, _, end)| *end as usize)
        .max()
        .unwrap_or(0);
    let mut data_buf = vec![0u8; total_data];
    reader
        .read_exact(&mut data_buf)
        .map_err(|_| ser_err("cannot read tensor data"))?;

    // Build tensors
    let mut result = Vec::with_capacity(entries.len());
    for (name, dtype, shape, start, end) in entries {
        let tensor =
            read_tensor_data::<T>(&data_buf, &dtype, &shape, start as usize, end as usize)?;
        result.push((name, tensor));
    }

    Ok(result)
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
            "{}/scivex_st_test_{name}_{}.safetensors",
            dir.display(),
            std::process::id()
        )
    }

    #[test]
    fn test_safetensors_single_f64() {
        let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = Tensor::from_vec(data.clone(), vec![2, 3]).unwrap();
        let path = temp_path("single_f64");
        save_safetensors(&path, &[("weights", &t)]).unwrap();
        let loaded = load_safetensors::<f64>(&path).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].0, "weights");
        assert_eq!(loaded[0].1.shape(), &[2, 3]);
        let loaded_data = loaded[0].1.as_slice();
        for (a, b) in data.iter().zip(loaded_data.iter()) {
            assert!((*a - *b).abs() < 1e-12);
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_safetensors_multiple_tensors() {
        let t1 = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0], vec![3]).unwrap();
        let t2 = Tensor::from_vec(vec![4.0_f64, 5.0, 6.0, 7.0], vec![2, 2]).unwrap();
        let path = temp_path("multi");
        save_safetensors(&path, &[("bias", &t1), ("weight", &t2)]).unwrap();
        let loaded = load_safetensors::<f64>(&path).unwrap();
        assert_eq!(loaded.len(), 2);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_safetensors_names_preserved() {
        let t1 = Tensor::from_vec(vec![1.0_f64], vec![1]).unwrap();
        let t2 = Tensor::from_vec(vec![2.0_f64], vec![1]).unwrap();
        let t3 = Tensor::from_vec(vec![3.0_f64], vec![1]).unwrap();
        let path = temp_path("names");
        save_safetensors(&path, &[("alpha", &t1), ("beta", &t2), ("gamma", &t3)]).unwrap();
        let loaded = load_safetensors::<f64>(&path).unwrap();
        let names: Vec<&str> = loaded.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.contains(&"alpha"));
        assert!(names.contains(&"beta"));
        assert!(names.contains(&"gamma"));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_safetensors_shapes_preserved() {
        let t1 = Tensor::from_vec(vec![0.0_f64; 24], vec![2, 3, 4]).unwrap();
        let t2 = Tensor::from_vec(vec![0.0_f64; 5], vec![5]).unwrap();
        let path = temp_path("shapes");
        save_safetensors(&path, &[("a", &t1), ("b", &t2)]).unwrap();
        let loaded = load_safetensors::<f64>(&path).unwrap();
        for (name, tensor) in &loaded {
            match name.as_str() {
                "a" => assert_eq!(tensor.shape(), &[2, 3, 4]),
                "b" => assert_eq!(tensor.shape(), &[5]),
                _ => panic!("unexpected tensor name"),
            }
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_safetensors_f32_roundtrip() {
        let data = vec![1.0_f32, 2.5, 3.75, -1.0, 0.0, 100.0];
        let t = Tensor::from_vec(data.clone(), vec![2, 3]).unwrap();
        let path = temp_path("f32");
        save_safetensors(&path, &[("w", &t)]).unwrap();
        let loaded = load_safetensors::<f32>(&path).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].1.shape(), &[2, 3]);
        let loaded_data = loaded[0].1.as_slice();
        for (a, b) in data.iter().zip(loaded_data.iter()) {
            assert!((*a - *b).abs() < 1e-6);
        }
        std::fs::remove_file(&path).ok();
    }
}
