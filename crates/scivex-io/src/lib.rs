//! # scivex-io
//!
//! CSV and JSON I/O for [`scivex_frame::DataFrame`].
//!
//! ## Feature Flags
//!
//! | Feature | Enables |
//! |---------|---------|
//! | `csv` *(default)* | CSV reading and writing |
//! | `json` | JSON reading and writing (requires `serde_json`) |
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

pub use error::{IoError, Result};

/// Glob-import convenience: `use scivex_io::prelude::*;`
pub mod prelude {
    pub use crate::error::{IoError, Result};

    #[cfg(feature = "csv")]
    pub use crate::csv::{
        CsvReaderBuilder, CsvWriterBuilder, QuoteStyle, read_csv, read_csv_path, write_csv,
    };

    #[cfg(feature = "json")]
    pub use crate::json::{
        JsonOrientation, JsonReaderBuilder, JsonWriterBuilder, read_json, read_json_path,
        write_json,
    };
}
