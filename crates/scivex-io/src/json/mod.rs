//! JSON reading and writing for [`DataFrame`](scivex_frame::DataFrame).

pub mod reader;
pub mod writer;

pub use reader::{JsonOrientation, JsonReaderBuilder, read_json, read_json_path};
pub use writer::{JsonWriterBuilder, write_json};
