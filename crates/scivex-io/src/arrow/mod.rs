//! Arrow IPC file reading and writing for DataFrames.
//!
//! Supports both the Arrow IPC file format (random access, also known as
//! Feather v2) and the Arrow IPC stream format (sequential access).

use std::fs::File;
use std::io::{Read, Seek, Write};
use std::path::Path;

use arrow_crate::ipc::reader::{FileReader as IpcFileReader, StreamReader as IpcStreamReader};
use arrow_crate::ipc::writer::{FileWriter as IpcFileWriter, StreamWriter as IpcStreamWriter};

use scivex_frame::DataFrame;

use crate::arrow_conv::{dataframe_to_record_batch, record_batches_to_dataframe};
use crate::error::{IoError, Result};

/// Read an Arrow IPC file into a [`DataFrame`].
pub fn read_arrow(path: impl AsRef<Path>) -> Result<DataFrame> {
    let file = File::open(path)?;
    read_arrow_from(file)
}

/// Read an Arrow IPC file from any readable + seekable source.
pub fn read_arrow_from<R: Read + Seek>(reader: R) -> Result<DataFrame> {
    let ipc_reader =
        IpcFileReader::try_new(reader, None).map_err(|e| IoError::ArrowError(e.to_string()))?;

    let batches: Vec<_> = ipc_reader
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|e| IoError::ArrowError(e.to_string()))?;

    record_batches_to_dataframe(&batches)
}

/// Write a [`DataFrame`] to an Arrow IPC file.
pub fn write_arrow(df: &DataFrame, path: impl AsRef<Path>) -> Result<()> {
    let file = File::create(path)?;
    write_arrow_to(df, file)
}

/// Write a [`DataFrame`] to any writable destination in Arrow IPC file format.
pub fn write_arrow_to<W: Write>(df: &DataFrame, writer: W) -> Result<()> {
    let batch = dataframe_to_record_batch(df)?;
    let schema = batch.schema();

    let mut ipc_writer =
        IpcFileWriter::try_new(writer, &schema).map_err(|e| IoError::ArrowError(e.to_string()))?;

    ipc_writer
        .write(&batch)
        .map_err(|e| IoError::ArrowError(e.to_string()))?;

    ipc_writer
        .finish()
        .map_err(|e| IoError::ArrowError(e.to_string()))?;

    Ok(())
}

/// Read an Arrow IPC stream into a [`DataFrame`].
pub fn read_arrow_stream<R: Read>(reader: R) -> Result<DataFrame> {
    let stream_reader =
        IpcStreamReader::try_new(reader, None).map_err(|e| IoError::ArrowError(e.to_string()))?;

    let batches: Vec<_> = stream_reader
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|e| IoError::ArrowError(e.to_string()))?;

    record_batches_to_dataframe(&batches)
}

/// Write a [`DataFrame`] to an Arrow IPC stream.
pub fn write_arrow_stream<W: Write>(df: &DataFrame, writer: W) -> Result<()> {
    let batch = dataframe_to_record_batch(df)?;
    let schema = batch.schema();

    let mut stream_writer = IpcStreamWriter::try_new(writer, &schema)
        .map_err(|e| IoError::ArrowError(e.to_string()))?;

    stream_writer
        .write(&batch)
        .map_err(|e| IoError::ArrowError(e.to_string()))?;

    stream_writer
        .finish()
        .map_err(|e| IoError::ArrowError(e.to_string()))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use scivex_frame::{AnySeries, Series, StringSeries};
    use std::io::Cursor;

    fn sample_df() -> DataFrame {
        let ids: Box<dyn AnySeries> = Box::new(Series::new("id", vec![10_i64, 20, 30]));
        let values: Box<dyn AnySeries> = Box::new(Series::new("value", vec![1.1_f64, 2.2, 3.3]));
        let labels: Box<dyn AnySeries> =
            Box::new(StringSeries::from_strs("label", &["a", "b", "c"]));
        DataFrame::new(vec![ids, values, labels]).unwrap()
    }

    #[test]
    fn test_arrow_ipc_file_roundtrip() {
        let df = sample_df();
        let dir = std::env::temp_dir();
        let path = dir.join("scivex_test_arrow.arrow");

        write_arrow(&df, &path).unwrap();
        let read_df = read_arrow(&path).unwrap();

        assert_eq!(read_df.nrows(), 3);
        assert_eq!(read_df.ncols(), 3);

        let id_col = read_df.column_typed::<i64>("id").unwrap();
        assert_eq!(id_col.as_slice(), &[10, 20, 30]);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_arrow_ipc_stream_roundtrip() {
        let df = sample_df();

        let mut buf = Vec::new();
        write_arrow_stream(&df, &mut buf).unwrap();
        assert!(!buf.is_empty());

        let cursor = Cursor::new(buf);
        let read_df = read_arrow_stream(cursor).unwrap();

        assert_eq!(read_df.nrows(), 3);
        assert_eq!(read_df.ncols(), 3);

        let val_col = read_df.column_typed::<f64>("value").unwrap();
        assert!((val_col.as_slice()[0] - 1.1).abs() < 1e-10);
    }

    #[test]
    fn test_arrow_with_nulls() {
        let ids: Box<dyn AnySeries> = Box::new(
            Series::with_nulls("id", vec![1_i64, 0, 3], vec![false, true, false]).unwrap(),
        );
        let df = DataFrame::new(vec![ids]).unwrap();

        let mut buf = Vec::new();
        write_arrow_stream(&df, &mut buf).unwrap();

        let cursor = Cursor::new(buf);
        let read_df = read_arrow_stream(cursor).unwrap();

        assert_eq!(read_df.nrows(), 3);
        let col = read_df.column("id").unwrap();
        assert!(!col.is_null(0));
        assert!(col.is_null(1));
        assert!(!col.is_null(2));
    }

    #[test]
    fn test_arrow_empty() {
        let df = DataFrame::new(vec![Box::new(Series::new("x", Vec::<f64>::new())) as _]).unwrap();

        let mut buf = Vec::new();
        write_arrow_stream(&df, &mut buf).unwrap();

        let cursor = Cursor::new(buf);
        let read_df = read_arrow_stream(cursor).unwrap();
        assert_eq!(read_df.nrows(), 0);
    }

    #[test]
    fn test_arrow_multiple_types() {
        let col_i32: Box<dyn AnySeries> = Box::new(Series::new("i32_col", vec![1_i32, 2, 3]));
        let col_u8: Box<dyn AnySeries> = Box::new(Series::new("u8_col", vec![10_u8, 20, 30]));
        let col_f32: Box<dyn AnySeries> = Box::new(Series::new("f32_col", vec![0.1_f32, 0.2, 0.3]));
        let df = DataFrame::new(vec![col_i32, col_u8, col_f32]).unwrap();

        let mut buf = Vec::new();
        write_arrow_stream(&df, &mut buf).unwrap();

        let cursor = Cursor::new(buf);
        let read_df = read_arrow_stream(cursor).unwrap();

        assert_eq!(read_df.nrows(), 3);
        assert_eq!(read_df.ncols(), 3);

        let i32_col = read_df.column_typed::<i32>("i32_col").unwrap();
        assert_eq!(i32_col.as_slice(), &[1, 2, 3]);
    }
}
