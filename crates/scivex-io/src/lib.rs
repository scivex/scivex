//! # scivex-io
//!
//! CSV, JSON, SQL, Parquet, and Arrow I/O for [`scivex_frame::DataFrame`].
//!
//! ## Feature Flags
//!
//! | Feature | Enables |
//! |---------|---------|
//! | `csv` *(default)* | CSV reading and writing |
//! | `json` | JSON reading and writing (requires `serde_json`) |
//! | `sqlite` | SQLite database connectivity (via `rusqlite`) |
//! | `postgres` | PostgreSQL database connectivity (via `postgres`) |
//! | `mysql` | MySQL database connectivity (via `mysql`) |
//! | `mssql` | SQL Server database connectivity (via `tiberius`) |
//! | `sql` | All SQL backends |
//! | `parquet` | Parquet file reading and writing |
//! | `arrow` | Arrow IPC file and stream reading and writing |
//! | `full` | All I/O formats |

/// Shared I/O helpers and type inference.
pub mod common;
/// I/O-specific error types.
pub mod error;

/// CSV reading and writing.
#[cfg(feature = "csv")]
pub mod csv;

/// JSON reading and writing (requires `serde_json`).
#[cfg(feature = "json")]
pub mod json;

/// SQL database connectivity (SQLite, PostgreSQL, MySQL, SQL Server).
#[cfg(any(
    feature = "sqlite",
    feature = "postgres",
    feature = "mysql",
    feature = "mssql"
))]
pub mod sql;

/// Shared Arrow conversion utilities used by Parquet and Arrow IPC modules.
#[cfg(any(feature = "parquet", feature = "arrow"))]
pub(crate) mod arrow_conv;

/// Parquet file reading and writing.
#[cfg(feature = "parquet")]
pub mod parquet;

/// Arrow IPC file and stream reading and writing.
#[cfg(feature = "arrow")]
pub mod arrow;

/// NumPy `.npy` and `.npz` file format support.
#[cfg(feature = "npy")]
pub mod npy;

/// Excel (.xlsx) reading and writing.
#[cfg(feature = "excel")]
pub mod excel;

/// Memory-mapped file I/O for tensors.
#[cfg(feature = "mmap")]
pub mod mmap;

/// Tensor ↔ Arrow array conversions.
#[cfg(feature = "arrow")]
pub mod tensor_conv;

pub use error::{IoError, Result};

/// Glob-import convenience: `use scivex_io::prelude::*;`
pub mod prelude {
    pub use crate::error::{IoError, Result};

    #[cfg(feature = "csv")]
    pub use crate::csv::{
        CsvChunkReader, CsvReaderBuilder, CsvWriterBuilder, QuoteStyle, read_csv, read_csv_path,
        write_csv,
    };

    #[cfg(feature = "json")]
    pub use crate::json::{
        JsonOrientation, JsonReaderBuilder, JsonWriterBuilder, read_json, read_json_path,
        write_json,
    };

    #[cfg(any(
        feature = "sqlite",
        feature = "postgres",
        feature = "mysql",
        feature = "mssql"
    ))]
    pub use crate::sql::IfExists;

    #[cfg(feature = "sqlite")]
    pub use crate::sql::sqlite::SqliteConnection;

    #[cfg(feature = "postgres")]
    pub use crate::sql::postgres::PostgresConnection;

    #[cfg(feature = "mysql")]
    pub use crate::sql::mysql::MysqlConnection;

    #[cfg(feature = "mssql")]
    pub use crate::sql::mssql::{MssqlConfig, MssqlConnection};

    #[cfg(feature = "parquet")]
    pub use crate::parquet::{
        ParquetCompression, ParquetReaderBuilder, ParquetWriterBuilder, read_parquet, write_parquet,
    };

    #[cfg(feature = "arrow")]
    pub use crate::arrow::{read_arrow, read_arrow_stream, write_arrow, write_arrow_stream};

    #[cfg(feature = "npy")]
    pub use crate::npy::{
        read_npy, read_npy_path, read_npz, read_npz_path, write_npy, write_npy_path, write_npz,
        write_npz_path,
    };

    #[cfg(feature = "excel")]
    pub use crate::excel::{ExcelReaderBuilder, ExcelWriterBuilder, read_excel, write_excel};

    #[cfg(feature = "mmap")]
    pub use crate::mmap::{MmapTensorReader, mmap_npy};

    #[cfg(feature = "arrow")]
    pub use crate::tensor_conv::{
        any_arrow_to_tensor_f64, arrow_f32_to_tensor, arrow_f64_to_tensor, record_batch_to_tensor,
        tensor_f32_to_arrow, tensor_f64_to_arrow, tensor_to_record_batch,
        tensor_to_record_batch_named,
    };
}
