//! C FFI bindings for Scivex.
//!
//! This crate exposes Scivex's core functionality through a C-compatible ABI,
//! enabling integration with Python (via `ctypes`/`cffi`), Julia, R, and any
//! language that supports C interop.
//!
//! # Memory Management
//!
//! All opaque types are heap-allocated and returned as raw pointers. The caller
//! **must** free them via the corresponding `scivex_*_free` function. Failing
//! to do so leaks memory.
//!
//! # Error Handling
//!
//! Functions that can fail return a status code (`0` = success, `-1` = error)
//! and write the result through an out-pointer. The last error message can be
//! retrieved with [`scivex_last_error`].
//!
//! # Thread Safety
//!
//! The error message is stored in thread-local storage — each thread has its
//! own independent error state.

mod error;
mod ml;
mod stats;
mod tensor;

pub use error::*;
pub use ml::*;
pub use stats::*;
pub use tensor::*;
