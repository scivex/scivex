//! Model serialization formats: SafeTensors and GGUF.
//!
//! This module provides support for reading and writing neural network weights
//! in industry-standard formats:
//!
//! - **SafeTensors** — HuggingFace's format for safe, fast tensor storage
//! - **GGUF** — llama.cpp's format for (optionally quantized) model storage

mod gguf;
mod safetensors;

pub use gguf::{GgufFile, GgufValue, load_gguf, save_gguf};
pub use safetensors::{
    load_safetensors, load_safetensors_from_reader, save_safetensors, save_safetensors_to_writer,
};
