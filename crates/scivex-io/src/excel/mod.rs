//! Excel (.xlsx) reading and writing for [`DataFrame`](scivex_frame::DataFrame).
//!
//! Uses `calamine` for reading and `rust_xlsxwriter` for writing.

mod reader;
mod writer;

pub use reader::{ExcelReaderBuilder, read_excel};
pub use writer::{ExcelWriterBuilder, write_excel};
