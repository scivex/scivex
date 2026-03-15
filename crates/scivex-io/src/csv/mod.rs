//! CSV reading and writing for [`DataFrame`](scivex_frame::DataFrame).

mod parser;
pub mod reader;
pub mod streaming;
pub mod writer;

pub use reader::{CsvReaderBuilder, read_csv, read_csv_path};
pub use streaming::CsvChunkReader;
pub use writer::{CsvWriterBuilder, QuoteStyle, write_csv};
