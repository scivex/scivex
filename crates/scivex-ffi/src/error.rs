//! Thread-local error handling for the FFI layer.

use std::cell::RefCell;
use std::ffi::CString;
use std::os::raw::c_char;

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

/// Store an error message in thread-local storage.
pub(crate) fn set_error(msg: &str) {
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = CString::new(msg).ok();
    });
}

/// Return a pointer to the last error message, or null if none.
///
/// The returned string is valid until the next FFI call on the same thread.
/// The caller must **not** free this pointer.
#[unsafe(no_mangle)]
pub extern "C" fn scivex_last_error() -> *const c_char {
    LAST_ERROR.with(|e| e.borrow().as_ref().map_or(std::ptr::null(), |s| s.as_ptr()))
}

/// Clear the last error.
#[unsafe(no_mangle)]
pub extern "C" fn scivex_clear_error() {
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = None;
    });
}
