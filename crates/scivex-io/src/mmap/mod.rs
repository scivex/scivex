//! Memory-mapped file I/O for tensors and data.
//!
//! Provides memory-mapped file access for reading large files without loading
//! them entirely into memory. Useful for datasets that are too large to fit
//! in RAM.
//!
//! # Example
//!
//! ```no_run
//! use scivex_io::mmap::{MmapTensorReader, mmap_npy};
//!
//! // Memory-map a .npy file and read the tensor
//! let tensor = mmap_npy("/path/to/large_array.npy").unwrap();
//! ```

mod tensor_mmap;

pub use tensor_mmap::{MmapTensorReader, mmap_npy};
