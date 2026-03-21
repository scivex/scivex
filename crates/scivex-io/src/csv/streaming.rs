//! Streaming (chunked) CSV reader for constant-memory processing.
//!
//! Reads a CSV file chunk-by-chunk, producing a sequence of [`DataFrame`]s
//! without loading the entire file into memory.

use std::io::{BufRead, BufReader, Read};
use std::path::Path;

use scivex_frame::DataFrame;

use super::parser::parse_record;
use crate::common::{build_series_from_strings, infer_column_type};
use crate::error::{IoError, Result};

/// A streaming CSV reader that yields chunks of rows as `DataFrame`s.
///
/// # Examples
///
/// ```
/// # use scivex_io::csv::CsvChunkReader;
/// let csv = "x,y\n1,2\n3,4\n5,6\n";
/// let mut reader = CsvChunkReader::from_reader(csv.as_bytes(), 2).unwrap();
/// let chunk1 = reader.next_chunk().unwrap().unwrap();
/// assert_eq!(chunk1.nrows(), 2);
/// let chunk2 = reader.next_chunk().unwrap().unwrap();
/// assert_eq!(chunk2.nrows(), 1);
/// assert!(reader.next_chunk().unwrap().is_none());
/// ```
pub struct CsvChunkReader<R: Read> {
    buf_reader: BufReader<R>,
    delimiter: u8,
    quote_char: u8,
    headers: Vec<String>,
    chunk_size: usize,
    done: bool,
}

impl CsvChunkReader<std::fs::File> {
    /// Open a CSV file for chunked reading.
    pub fn open<P: AsRef<Path>>(path: P, chunk_size: usize) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        Self::from_reader(file, chunk_size)
    }
}

impl<R: Read> CsvChunkReader<R> {
    /// Create a chunk reader from any `Read` source.
    pub fn from_reader(reader: R, chunk_size: usize) -> Result<Self> {
        let mut buf_reader = BufReader::new(reader);

        // Read the header line
        let mut header_line = String::new();
        buf_reader.read_line(&mut header_line)?;
        let headers = parse_record(header_line.trim_end(), b',', b'"');

        if headers.is_empty() {
            return Err(IoError::FormatError("CSV has no headers".into()));
        }

        Ok(Self {
            buf_reader,
            delimiter: b',',
            quote_char: b'"',
            headers,
            chunk_size,
            done: false,
        })
    }

    /// Column names from the header row.
    pub fn headers(&self) -> &[String] {
        &self.headers
    }

    /// Read the next chunk. Returns `Ok(None)` when the file is exhausted.
    pub fn next_chunk(&mut self) -> Result<Option<DataFrame>> {
        if self.done {
            return Ok(None);
        }

        let ncols = self.headers.len();
        let mut columns: Vec<Vec<String>> = vec![Vec::with_capacity(self.chunk_size); ncols];
        let mut rows_read = 0;
        let mut line = String::new();

        while rows_read < self.chunk_size {
            line.clear();
            let bytes_read = self.buf_reader.read_line(&mut line)?;
            if bytes_read == 0 {
                self.done = true;
                break;
            }

            let trimmed = line.trim_end();
            if trimmed.is_empty() {
                continue;
            }

            let fields = parse_record(trimmed, self.delimiter, self.quote_char);
            for (col_idx, field) in fields.iter().enumerate() {
                if col_idx < ncols {
                    columns[col_idx].push(field.clone());
                }
            }
            // Pad missing columns
            for col in columns.iter_mut().skip(fields.len()) {
                col.push(String::new());
            }
            rows_read += 1;
        }

        if rows_read == 0 {
            return Ok(None);
        }

        build_chunk_dataframe(&self.headers, columns)
    }
}

/// Build a `DataFrame` from column string data.
fn build_chunk_dataframe(
    headers: &[String],
    columns: Vec<Vec<String>>,
) -> Result<Option<DataFrame>> {
    let mut series_list: Vec<Box<dyn scivex_frame::AnySeries>> = Vec::with_capacity(headers.len());

    for (i, col_data) in columns.into_iter().enumerate() {
        let name = &headers[i];
        let refs: Vec<&str> = col_data.iter().map(String::as_str).collect();
        let dtype = infer_column_type(&refs);
        let series = build_series_from_strings(name, &col_data, dtype)?;
        series_list.push(series);
    }

    let df = DataFrame::new(series_list)?;
    Ok(Some(df))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_reader_basic() {
        let csv = "a,b,c\n1,2,3\n4,5,6\n7,8,9\n10,11,12\n";
        let mut reader = CsvChunkReader::from_reader(csv.as_bytes(), 2).unwrap();

        let chunk1 = reader.next_chunk().unwrap().unwrap();
        assert_eq!(chunk1.nrows(), 2);

        let chunk2 = reader.next_chunk().unwrap().unwrap();
        assert_eq!(chunk2.nrows(), 2);

        assert!(reader.next_chunk().unwrap().is_none());
    }

    #[test]
    fn test_chunk_reader_uneven() {
        let csv = "x,y\n1,2\n3,4\n5,6\n";
        let mut reader = CsvChunkReader::from_reader(csv.as_bytes(), 2).unwrap();

        let chunk1 = reader.next_chunk().unwrap().unwrap();
        assert_eq!(chunk1.nrows(), 2);

        let chunk2 = reader.next_chunk().unwrap().unwrap();
        assert_eq!(chunk2.nrows(), 1);

        assert!(reader.next_chunk().unwrap().is_none());
    }

    #[test]
    fn test_chunk_reader_single_chunk() {
        let csv = "a\n1\n2\n3\n";
        let mut reader = CsvChunkReader::from_reader(csv.as_bytes(), 100).unwrap();

        let chunk = reader.next_chunk().unwrap().unwrap();
        assert_eq!(chunk.nrows(), 3);

        assert!(reader.next_chunk().unwrap().is_none());
    }

    #[test]
    fn test_chunk_reader_headers() {
        let csv = "name,age\nAlice,30\n";
        let reader = CsvChunkReader::from_reader(csv.as_bytes(), 100).unwrap();
        assert_eq!(reader.headers(), &["name", "age"]);
    }

    #[test]
    fn test_chunk_reader_empty_body() {
        let csv = "a,b\n";
        let mut reader = CsvChunkReader::from_reader(csv.as_bytes(), 100).unwrap();
        assert!(reader.next_chunk().unwrap().is_none());
    }
}
