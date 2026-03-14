//! Parquet file reading and writing for DataFrames.
//!
//! Uses the Apache `parquet` and `arrow` crates for Parquet format support
//! with column pruning and compression options.

use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;

use parquet_crate::arrow::ArrowWriter;
use parquet_crate::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet_crate::basic::Compression;
use parquet_crate::file::properties::WriterProperties;

use scivex_frame::DataFrame;

use crate::arrow_conv::{dataframe_to_record_batch, record_batches_to_dataframe};
use crate::error::{IoError, Result};

/// Read a Parquet file at the given path into a [`DataFrame`].
///
/// All columns are read by default. Use [`ParquetReaderBuilder`] for
/// column selection and batch size control.
pub fn read_parquet(path: impl AsRef<Path>) -> Result<DataFrame> {
    ParquetReaderBuilder::new().read_path(path)
}

/// Write a [`DataFrame`] to a Parquet file at the given path.
///
/// Uses Snappy compression by default. Use [`ParquetWriterBuilder`] for
/// custom compression and writer settings.
pub fn write_parquet(df: &DataFrame, path: impl AsRef<Path>) -> Result<()> {
    ParquetWriterBuilder::new().write_path(df, path)
}

/// Builder for configuring Parquet reading.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct ParquetReaderBuilder {
    batch_size: usize,
    columns: Option<Vec<String>>,
}

impl Default for ParquetReaderBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ParquetReaderBuilder {
    /// Create a new reader builder with default settings.
    pub fn new() -> Self {
        Self {
            batch_size: 8192,
            columns: None,
        }
    }

    /// Set the batch size for reading (default: 8192 rows per batch).
    pub fn batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Select specific columns to read (column pruning).
    ///
    /// Only the named columns will be read from the file, improving performance
    /// for wide tables when only a few columns are needed.
    pub fn columns(mut self, cols: Vec<String>) -> Self {
        self.columns = Some(cols);
        self
    }

    /// Read a Parquet file from disk.
    pub fn read_path(&self, path: impl AsRef<Path>) -> Result<DataFrame> {
        let file = File::open(path)?;
        self.read_file(file)
    }

    /// Read Parquet data from a [`File`].
    pub fn read_file(&self, file: File) -> Result<DataFrame> {
        let mut builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .map_err(|e| IoError::ArrowError(e.to_string()))?
            .with_batch_size(self.batch_size);

        // Apply column projection if specified.
        if let Some(ref cols) = self.columns {
            let schema = builder.schema().clone();
            let indices: Vec<usize> = cols
                .iter()
                .filter_map(|name| schema.fields().iter().position(|f| f.name() == name))
                .collect();
            let mask =
                parquet_crate::arrow::ProjectionMask::roots(builder.parquet_schema(), indices);
            builder = builder.with_projection(mask);
        }

        let reader = builder
            .build()
            .map_err(|e| IoError::ArrowError(e.to_string()))?;

        let batches: Vec<_> = reader
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| IoError::ArrowError(e.to_string()))?;

        record_batches_to_dataframe(&batches)
    }

    /// Read Parquet data from an in-memory byte buffer.
    pub fn read_bytes(&self, data: bytes::Bytes) -> Result<DataFrame> {
        let mut builder = ParquetRecordBatchReaderBuilder::try_new(data)
            .map_err(|e| IoError::ArrowError(e.to_string()))?
            .with_batch_size(self.batch_size);

        if let Some(ref cols) = self.columns {
            let schema = builder.schema().clone();
            let indices: Vec<usize> = cols
                .iter()
                .filter_map(|name| schema.fields().iter().position(|f| f.name() == name))
                .collect();
            let mask =
                parquet_crate::arrow::ProjectionMask::roots(builder.parquet_schema(), indices);
            builder = builder.with_projection(mask);
        }

        let reader = builder
            .build()
            .map_err(|e| IoError::ArrowError(e.to_string()))?;

        let batches: Vec<_> = reader
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| IoError::ArrowError(e.to_string()))?;

        record_batches_to_dataframe(&batches)
    }
}

/// Compression codec for Parquet writing.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParquetCompression {
    /// No compression.
    None,
    /// Snappy compression (fast, moderate ratio).
    Snappy,
    /// Gzip compression (slower, better ratio).
    Gzip,
    /// Zstd compression (good speed/ratio balance).
    Zstd,
    /// LZ4 compression (very fast).
    Lz4,
}

impl ParquetCompression {
    fn to_parquet(self) -> Compression {
        match self {
            Self::None => Compression::UNCOMPRESSED,
            Self::Snappy => Compression::SNAPPY,
            Self::Gzip => Compression::GZIP(parquet_crate::basic::GzipLevel::default()),
            Self::Zstd => Compression::ZSTD(parquet_crate::basic::ZstdLevel::default()),
            Self::Lz4 => Compression::LZ4,
        }
    }
}

/// Builder for configuring Parquet writing.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct ParquetWriterBuilder {
    compression: ParquetCompression,
}

impl Default for ParquetWriterBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ParquetWriterBuilder {
    /// Create a new writer builder with default settings (Snappy compression).
    pub fn new() -> Self {
        Self {
            compression: ParquetCompression::Snappy,
        }
    }

    /// Set the compression codec.
    pub fn compression(mut self, codec: ParquetCompression) -> Self {
        self.compression = codec;
        self
    }

    /// Write a DataFrame to a file path.
    pub fn write_path(&self, df: &DataFrame, path: impl AsRef<Path>) -> Result<()> {
        let file = File::create(path)?;
        self.write(df, file)
    }

    /// Write a DataFrame to any [`Write`] destination.
    pub fn write<W: Write + Send>(&self, df: &DataFrame, writer: W) -> Result<()> {
        let batch = dataframe_to_record_batch(df)?;
        let schema = batch.schema();

        let props = WriterProperties::builder()
            .set_compression(self.compression.to_parquet())
            .build();

        let mut arrow_writer = ArrowWriter::try_new(writer, Arc::clone(&schema), Some(props))
            .map_err(|e| IoError::ArrowError(e.to_string()))?;

        arrow_writer
            .write(&batch)
            .map_err(|e| IoError::ArrowError(e.to_string()))?;

        arrow_writer
            .close()
            .map_err(|e| IoError::ArrowError(e.to_string()))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scivex_frame::{AnySeries, Series, StringSeries};

    fn sample_df() -> DataFrame {
        let ids: Box<dyn AnySeries> = Box::new(Series::new("id", vec![1_i64, 2, 3]));
        let names: Box<dyn AnySeries> = Box::new(StringSeries::from_strs(
            "name",
            &["alice", "bob", "charlie"],
        ));
        let scores: Box<dyn AnySeries> = Box::new(Series::new("score", vec![95.5_f64, 87.3, 92.1]));
        DataFrame::new(vec![ids, names, scores]).unwrap()
    }

    #[test]
    fn test_parquet_roundtrip_buffer() {
        let df = sample_df();

        // Write to in-memory buffer.
        let mut buf = Vec::new();
        ParquetWriterBuilder::new()
            .compression(ParquetCompression::None)
            .write(&df, &mut buf)
            .unwrap();

        assert!(!buf.is_empty());

        // Read back via bytes.
        let read_df = ParquetReaderBuilder::new()
            .read_bytes(bytes::Bytes::from(buf))
            .unwrap();

        assert_eq!(read_df.nrows(), 3);
        assert_eq!(read_df.ncols(), 3);

        let id_col = read_df.column_typed::<i64>("id").unwrap();
        assert_eq!(id_col.as_slice(), &[1, 2, 3]);

        let score_col = read_df.column_typed::<f64>("score").unwrap();
        assert!((score_col.as_slice()[0] - 95.5).abs() < 1e-10);
    }

    #[test]
    fn test_parquet_with_snappy() {
        let df = sample_df();
        let mut buf = Vec::new();
        ParquetWriterBuilder::new()
            .compression(ParquetCompression::Snappy)
            .write(&df, &mut buf)
            .unwrap();

        let read_df = ParquetReaderBuilder::new()
            .read_bytes(bytes::Bytes::from(buf))
            .unwrap();
        assert_eq!(read_df.nrows(), 3);
    }

    #[test]
    fn test_parquet_with_nulls() {
        let ids: Box<dyn AnySeries> = Box::new(
            Series::with_nulls("id", vec![1_i64, 0, 3], vec![false, true, false]).unwrap(),
        );
        let names: Box<dyn AnySeries> = Box::new(
            StringSeries::with_nulls(
                "name",
                vec!["alice".into(), String::new(), "charlie".into()],
                vec![false, true, false],
            )
            .unwrap(),
        );
        let df = DataFrame::new(vec![ids, names]).unwrap();

        let mut buf = Vec::new();
        ParquetWriterBuilder::new()
            .compression(ParquetCompression::None)
            .write(&df, &mut buf)
            .unwrap();

        let read_df = ParquetReaderBuilder::new()
            .read_bytes(bytes::Bytes::from(buf))
            .unwrap();

        assert_eq!(read_df.nrows(), 3);
        let id_col = read_df.column("id").unwrap();
        assert!(!id_col.is_null(0));
        assert!(id_col.is_null(1));
        assert!(!id_col.is_null(2));
    }

    #[test]
    fn test_parquet_empty_dataframe() {
        let df = DataFrame::new(vec![Box::new(Series::new("x", Vec::<i64>::new())) as _]).unwrap();

        let mut buf = Vec::new();
        ParquetWriterBuilder::new()
            .compression(ParquetCompression::None)
            .write(&df, &mut buf)
            .unwrap();

        let read_df = ParquetReaderBuilder::new()
            .read_bytes(bytes::Bytes::from(buf))
            .unwrap();
        assert_eq!(read_df.nrows(), 0);
    }

    #[test]
    fn test_parquet_file_roundtrip() {
        let df = sample_df();
        let dir = std::env::temp_dir();
        let path = dir.join("scivex_test_parquet.parquet");

        write_parquet(&df, &path).unwrap();
        let read_df = read_parquet(&path).unwrap();

        assert_eq!(read_df.nrows(), 3);
        assert_eq!(read_df.ncols(), 3);

        let _ = std::fs::remove_file(&path);
    }
}
