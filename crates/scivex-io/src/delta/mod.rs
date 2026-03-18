//! Delta Lake table reader.
//!
//! Reads Delta Lake tables by replaying the `_delta_log/` transaction log to
//! determine the set of active Parquet data files, then reads and concatenates
//! them into a single [`DataFrame`].
//!
//! No external Delta-specific dependencies are used — the JSON transaction log
//! is parsed with a minimal hand-rolled parser (see [`log`]).

pub mod log;

use std::path::Path;

use scivex_frame::DataFrame;

use crate::arrow_conv::{dataframe_to_record_batch, record_batches_to_dataframe};
use crate::error::Result;
use crate::parquet::read_parquet;

// ---------------------------------------------------------------------------
// DeltaSnapshot
// ---------------------------------------------------------------------------

/// A snapshot of a Delta Lake table at a given version.
///
/// The snapshot records the table version and the list of active Parquet data
/// files as determined by replaying the transaction log.
#[derive(Debug, Clone)]
pub struct DeltaSnapshot {
    version: u64,
    files: Vec<String>,
}

impl DeltaSnapshot {
    /// Load the latest snapshot of the Delta table at `table_path`.
    pub fn load(table_path: impl AsRef<Path>) -> Result<Self> {
        let table_path = table_path.as_ref();
        let delta_log = log::log_dir(table_path)?;
        let (version, files) = log::replay_logs(&delta_log, None)?;
        Ok(Self { version, files })
    }

    /// Load a snapshot at a specific version.
    pub fn load_version(table_path: impl AsRef<Path>, version: u64) -> Result<Self> {
        let table_path = table_path.as_ref();
        let delta_log = log::log_dir(table_path)?;
        let (actual_version, files) = log::replay_logs(&delta_log, Some(version))?;
        Ok(Self {
            version: actual_version,
            files,
        })
    }

    /// The table version this snapshot represents.
    pub fn version(&self) -> u64 {
        self.version
    }

    /// The active data file paths (relative to the table root).
    pub fn files(&self) -> &[String] {
        &self.files
    }

    /// Read all active data files and combine them into a single [`DataFrame`].
    pub fn to_dataframe(&self, table_path: impl AsRef<Path>) -> Result<DataFrame> {
        let table_path = table_path.as_ref();

        if self.files.is_empty() {
            return Ok(DataFrame::empty());
        }

        // Read each parquet file and collect the DataFrames.
        let mut frames: Vec<DataFrame> = Vec::with_capacity(self.files.len());
        for file in &self.files {
            let file_path = table_path.join(file);
            let df = read_parquet(&file_path)?;
            frames.push(df);
        }

        // Concatenate all DataFrames vertically.
        concat_dataframes(frames)
    }
}

// ---------------------------------------------------------------------------
// Public convenience function
// ---------------------------------------------------------------------------

/// Read the latest snapshot of a Delta Lake table into a [`DataFrame`].
///
/// This is a shortcut for `DeltaSnapshot::load(path)?.to_dataframe(path)?`.
pub fn read_delta(path: impl AsRef<Path>) -> Result<DataFrame> {
    let path = path.as_ref();
    let snapshot = DeltaSnapshot::load(path)?;
    snapshot.to_dataframe(path)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Concatenate multiple DataFrames with the same schema vertically.
///
/// Converts each DataFrame to an Arrow RecordBatch and uses the existing
/// `record_batches_to_dataframe` utility which handles column concatenation.
fn concat_dataframes(frames: Vec<DataFrame>) -> Result<DataFrame> {
    if frames.is_empty() {
        return Ok(DataFrame::empty());
    }

    if frames.len() == 1 {
        return Ok(frames.into_iter().next().expect("checked len == 1"));
    }

    let batches: Vec<_> = frames
        .iter()
        .map(dataframe_to_record_batch)
        .collect::<Result<Vec<_>>>()?;

    record_batches_to_dataframe(&batches)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parquet::write_parquet;
    use scivex_frame::{AnySeries, Series};
    use std::fs;

    /// Create a minimal Delta table on disk for testing.
    fn create_test_delta_table(dir: &Path) {
        let log_dir = dir.join("_delta_log");
        fs::create_dir_all(&log_dir).unwrap();

        // Write a parquet file.
        let df = DataFrame::new(vec![
            Box::new(Series::new("id", vec![1_i64, 2, 3])) as Box<dyn AnySeries>,
            Box::new(Series::new("value", vec![10.0_f64, 20.0, 30.0])) as Box<dyn AnySeries>,
        ])
        .unwrap();
        write_parquet(&df, dir.join("part-00000.parquet")).unwrap();

        // Write transaction log.
        fs::write(
            log_dir.join("00000000000000000000.json"),
            r#"{"metaData":{"id":"test-table","format":{"provider":"parquet"},"schemaString":"{}","partitionColumns":[]}}
{"add":{"path":"part-00000.parquet","size":1000,"modificationTime":1000,"dataChange":true}}
"#,
        )
        .unwrap();
    }

    #[test]
    fn test_read_delta_table() {
        let dir = std::env::temp_dir().join("scivex_delta_read_test");
        let _ = fs::remove_dir_all(&dir);
        create_test_delta_table(&dir);

        let df = read_delta(&dir).unwrap();
        assert_eq!(df.nrows(), 3);
        assert_eq!(df.ncols(), 2);

        let ids = df.column_typed::<i64>("id").unwrap();
        assert_eq!(ids.as_slice(), &[1, 2, 3]);

        let values = df.column_typed::<f64>("value").unwrap();
        assert!((values.as_slice()[0] - 10.0).abs() < 1e-10);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_delta_snapshot_version() {
        let dir = std::env::temp_dir().join("scivex_delta_version_test");
        let _ = fs::remove_dir_all(&dir);
        let log_dir = dir.join("_delta_log");
        fs::create_dir_all(&log_dir).unwrap();

        // Version 0: add file A.
        let df_a = DataFrame::new(vec![
            Box::new(Series::new("x", vec![1_i64, 2])) as Box<dyn AnySeries>
        ])
        .unwrap();
        write_parquet(&df_a, dir.join("a.parquet")).unwrap();
        fs::write(
            log_dir.join("00000000000000000000.json"),
            r#"{"add":{"path":"a.parquet","size":100,"modificationTime":1000,"dataChange":true}}"#,
        )
        .unwrap();

        // Version 1: add file B.
        let df_b = DataFrame::new(vec![
            Box::new(Series::new("x", vec![3_i64, 4])) as Box<dyn AnySeries>
        ])
        .unwrap();
        write_parquet(&df_b, dir.join("b.parquet")).unwrap();
        fs::write(
            log_dir.join("00000000000000000001.json"),
            r#"{"add":{"path":"b.parquet","size":100,"modificationTime":2000,"dataChange":true}}"#,
        )
        .unwrap();

        // Load version 0 — should only contain file A.
        let snap_v0 = DeltaSnapshot::load_version(&dir, 0).unwrap();
        assert_eq!(snap_v0.version(), 0);
        assert_eq!(snap_v0.files(), &["a.parquet"]);
        let df_v0 = snap_v0.to_dataframe(&dir).unwrap();
        assert_eq!(df_v0.nrows(), 2);

        // Load latest — should contain both files.
        let snap_latest = DeltaSnapshot::load(&dir).unwrap();
        assert_eq!(snap_latest.version(), 1);
        assert_eq!(snap_latest.files().len(), 2);
        let df_latest = snap_latest.to_dataframe(&dir).unwrap();
        assert_eq!(df_latest.nrows(), 4);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_delta_empty_table() {
        let dir = std::env::temp_dir().join("scivex_delta_empty_table_test");
        let _ = fs::remove_dir_all(&dir);
        let log_dir = dir.join("_delta_log");
        fs::create_dir_all(&log_dir).unwrap();

        // Version 0: add a file, then version 1: remove it.
        let df_a = DataFrame::new(vec![
            Box::new(Series::new("x", vec![1_i64])) as Box<dyn AnySeries>
        ])
        .unwrap();
        write_parquet(&df_a, dir.join("a.parquet")).unwrap();
        fs::write(
            log_dir.join("00000000000000000000.json"),
            r#"{"add":{"path":"a.parquet","size":100,"modificationTime":1000,"dataChange":true}}"#,
        )
        .unwrap();
        fs::write(
            log_dir.join("00000000000000000001.json"),
            r#"{"remove":{"path":"a.parquet","deletionTimestamp":2000,"dataChange":true}}"#,
        )
        .unwrap();

        let snap = DeltaSnapshot::load(&dir).unwrap();
        assert!(snap.files().is_empty());
        let df = snap.to_dataframe(&dir).unwrap();
        assert!(df.is_empty());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_delta_missing_log_dir() {
        let dir = std::env::temp_dir().join("scivex_delta_no_log_test");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();

        let result = read_delta(&dir);
        assert!(result.is_err());

        let _ = fs::remove_dir_all(&dir);
    }
}
