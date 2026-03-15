//! NumPy `.npy` and `.npz` file format support.
//!
//! - `.npy`: Single tensor in NumPy's binary array format.
//! - `.npz`: ZIP archive containing multiple `.npy` files.
//!
//! Supports `f32`, `f64`, `i32`, `i64` dtypes in little-endian and big-endian
//! byte order. Both C-order (row-major) and Fortran-order (column-major) are
//! handled on read; writes always use C-order.

mod reader;
mod writer;

pub use reader::{read_npy, read_npy_path, read_npz, read_npz_path};
pub use writer::{write_npy, write_npy_path, write_npz, write_npz_path};
