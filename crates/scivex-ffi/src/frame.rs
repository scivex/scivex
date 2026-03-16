//! C FFI bindings for DataFrame operations.

use std::ffi::CStr;
use std::os::raw::{c_char, c_int};
use std::slice;

use scivex_frame::series::string::StringSeries;
use scivex_frame::{DataFrame, Series};

use crate::error::set_error;

/// Opaque handle to a `DataFrame`.
pub struct ScivexDataFrame {
    inner: DataFrame,
}

// ---------------------------------------------------------------------------
// Construction & destruction
// ---------------------------------------------------------------------------

/// Create an empty DataFrame.
#[unsafe(no_mangle)]
pub extern "C" fn scivex_df_new() -> *mut ScivexDataFrame {
    Box::into_raw(Box::new(ScivexDataFrame {
        inner: DataFrame::empty(),
    }))
}

/// Add an f64 column to the DataFrame.
/// Returns 0 on success, -1 on error.
///
/// # Safety
///
/// `df` must be a valid pointer. `name` must be a valid null-terminated C string.
/// `data` must point to `len` valid f64 values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn scivex_df_add_column(
    df: *mut ScivexDataFrame,
    name: *const c_char,
    data: *const f64,
    len: usize,
) -> c_int {
    if df.is_null() || name.is_null() || data.is_null() {
        set_error("null pointer passed to scivex_df_add_column");
        return -1;
    }
    let name_str = match unsafe { CStr::from_ptr(name) }.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_error(&e.to_string());
            return -1;
        }
    };
    let data_slice = unsafe { slice::from_raw_parts(data, len) };
    let series = Series::new(name_str, data_slice.to_vec());
    let df_ref = unsafe { &mut *df };
    match df_ref
        .inner
        .add_column(Box::new(series) as Box<dyn scivex_frame::series::AnySeries>)
    {
        Ok(()) => 0,
        Err(e) => {
            set_error(&e.to_string());
            -1
        }
    }
}

/// Add a string column to the DataFrame.
/// Returns 0 on success, -1 on error.
///
/// # Safety
///
/// `df` must be a valid pointer. `name` must be a valid null-terminated C string.
/// `strings` must point to `count` valid null-terminated C string pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn scivex_df_add_string_column(
    df: *mut ScivexDataFrame,
    name: *const c_char,
    strings: *const *const c_char,
    count: usize,
) -> c_int {
    if df.is_null() || name.is_null() || strings.is_null() {
        set_error("null pointer passed to scivex_df_add_string_column");
        return -1;
    }
    let name_str = match unsafe { CStr::from_ptr(name) }.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_error(&e.to_string());
            return -1;
        }
    };
    let ptrs = unsafe { slice::from_raw_parts(strings, count) };
    let mut str_vec = Vec::with_capacity(count);
    for &p in ptrs {
        if p.is_null() {
            set_error("null string pointer in array");
            return -1;
        }
        match unsafe { CStr::from_ptr(p) }.to_str() {
            Ok(s) => str_vec.push(s.to_string()),
            Err(e) => {
                set_error(&e.to_string());
                return -1;
            }
        }
    }
    let series = StringSeries::new(name_str, str_vec);
    let df_ref = unsafe { &mut *df };
    match df_ref
        .inner
        .add_column(Box::new(series) as Box<dyn scivex_frame::series::AnySeries>)
    {
        Ok(()) => 0,
        Err(e) => {
            set_error(&e.to_string());
            -1
        }
    }
}

/// Get the number of rows.
///
/// # Safety
///
/// `df` must be a valid, non-null pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn scivex_df_nrows(df: *const ScivexDataFrame) -> usize {
    unsafe { (*df).inner.nrows() }
}

/// Get the number of columns.
///
/// # Safety
///
/// `df` must be a valid, non-null pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn scivex_df_ncols(df: *const ScivexDataFrame) -> usize {
    unsafe { (*df).inner.ncols() }
}

/// Copy f64 column data into the caller's buffer.
/// Returns 0 on success, -1 on error.
///
/// # Safety
///
/// `df` must be a valid pointer. `name` must be a valid null-terminated C string.
/// `out` must point to a buffer of at least `out_len` f64 values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn scivex_df_column_data(
    df: *const ScivexDataFrame,
    name: *const c_char,
    out: *mut f64,
    out_len: usize,
) -> c_int {
    if df.is_null() || name.is_null() || out.is_null() {
        set_error("null pointer passed to scivex_df_column_data");
        return -1;
    }
    let name_str = match unsafe { CStr::from_ptr(name) }.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_error(&e.to_string());
            return -1;
        }
    };
    let df_ref = unsafe { &*df };
    match df_ref.inner.column_typed::<f64>(name_str) {
        Ok(series) => {
            let data = series.as_slice();
            if out_len < data.len() {
                set_error("output buffer too small for column data");
                return -1;
            }
            let out_slice = unsafe { slice::from_raw_parts_mut(out, data.len()) };
            out_slice.copy_from_slice(data);
            0
        }
        Err(e) => {
            set_error(&e.to_string());
            -1
        }
    }
}

/// Get column names as a newline-separated, null-terminated string.
/// Returns the number of bytes needed (including null terminator).
/// If `buf_len` is large enough, the string is written to `out_buf`.
///
/// # Safety
///
/// `df` must be a valid pointer. `out_buf` may be null (to query size only)
/// or must point to `buf_len` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn scivex_df_column_names(
    df: *const ScivexDataFrame,
    out_buf: *mut c_char,
    buf_len: usize,
) -> usize {
    let df_ref = unsafe { &*df };
    let names = df_ref.inner.column_names();
    let joined = names.join("\n");
    let needed = joined.len() + 1; // +1 for null terminator
    if !out_buf.is_null() && buf_len >= needed {
        let out_slice = unsafe { slice::from_raw_parts_mut(out_buf.cast::<u8>(), needed) };
        out_slice[..joined.len()].copy_from_slice(joined.as_bytes());
        out_slice[joined.len()] = 0;
    }
    needed
}

/// Return a new DataFrame with the first `n` rows.
///
/// # Safety
///
/// `df` must be a valid, non-null pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn scivex_df_head(
    df: *const ScivexDataFrame,
    n: usize,
) -> *mut ScivexDataFrame {
    let df_ref = unsafe { &*df };
    let result = df_ref.inner.head(n);
    Box::into_raw(Box::new(ScivexDataFrame { inner: result }))
}

/// Return a new DataFrame with selected columns.
///
/// # Safety
///
/// `df` must be a valid pointer. `col_names` must point to `n_cols` valid
/// null-terminated C string pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn scivex_df_select(
    df: *const ScivexDataFrame,
    col_names: *const *const c_char,
    n_cols: usize,
) -> *mut ScivexDataFrame {
    if df.is_null() || col_names.is_null() {
        set_error("null pointer passed to scivex_df_select");
        return std::ptr::null_mut();
    }
    let ptrs = unsafe { slice::from_raw_parts(col_names, n_cols) };
    let mut names = Vec::with_capacity(n_cols);
    for &p in ptrs {
        if p.is_null() {
            set_error("null string pointer in column names array");
            return std::ptr::null_mut();
        }
        match unsafe { CStr::from_ptr(p) }.to_str() {
            Ok(s) => names.push(s),
            Err(e) => {
                set_error(&e.to_string());
                return std::ptr::null_mut();
            }
        }
    }
    let df_ref = unsafe { &*df };
    match df_ref.inner.select(&names) {
        Ok(result) => Box::into_raw(Box::new(ScivexDataFrame { inner: result })),
        Err(e) => {
            set_error(&e.to_string());
            std::ptr::null_mut()
        }
    }
}

/// Filter rows where a column value is greater than `threshold`.
/// Only works on f64 columns. Returns a new DataFrame or null on error.
///
/// # Safety
///
/// `df` must be a valid pointer. `col_name` must be a valid null-terminated C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn scivex_df_filter_gt(
    df: *const ScivexDataFrame,
    col_name: *const c_char,
    threshold: f64,
) -> *mut ScivexDataFrame {
    if df.is_null() || col_name.is_null() {
        set_error("null pointer passed to scivex_df_filter_gt");
        return std::ptr::null_mut();
    }
    let name_str = match unsafe { CStr::from_ptr(col_name) }.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_error(&e.to_string());
            return std::ptr::null_mut();
        }
    };
    let df_ref = unsafe { &*df };
    let series = match df_ref.inner.column_typed::<f64>(name_str) {
        Ok(s) => s,
        Err(e) => {
            set_error(&e.to_string());
            return std::ptr::null_mut();
        }
    };
    let mask: Vec<bool> = series.as_slice().iter().map(|&v| v > threshold).collect();
    match df_ref.inner.filter(&mask) {
        Ok(result) => Box::into_raw(Box::new(ScivexDataFrame { inner: result })),
        Err(e) => {
            set_error(&e.to_string());
            std::ptr::null_mut()
        }
    }
}

/// Sort the DataFrame by a column.
/// Returns a new DataFrame or null on error.
///
/// # Safety
///
/// `df` must be a valid pointer. `col_name` must be a valid null-terminated C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn scivex_df_sort_by(
    df: *const ScivexDataFrame,
    col_name: *const c_char,
    ascending: c_int,
) -> *mut ScivexDataFrame {
    if df.is_null() || col_name.is_null() {
        set_error("null pointer passed to scivex_df_sort_by");
        return std::ptr::null_mut();
    }
    let name_str = match unsafe { CStr::from_ptr(col_name) }.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_error(&e.to_string());
            return std::ptr::null_mut();
        }
    };
    let df_ref = unsafe { &*df };
    match df_ref.inner.sort_by(name_str, ascending != 0) {
        Ok(result) => Box::into_raw(Box::new(ScivexDataFrame { inner: result })),
        Err(e) => {
            set_error(&e.to_string());
            std::ptr::null_mut()
        }
    }
}

/// Write a summary description of the DataFrame into the caller's buffer.
/// Returns the number of bytes needed (including null terminator).
/// If `buf_len` is large enough, the string is written to `out_buf`.
///
/// # Safety
///
/// `df` must be a valid pointer. `out_buf` may be null (to query size only)
/// or must point to `buf_len` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn scivex_df_describe(
    df: *const ScivexDataFrame,
    out_buf: *mut c_char,
    buf_len: usize,
) -> usize {
    let df_ref = unsafe { &*df };
    let desc = format!("{}", df_ref.inner);
    let needed = desc.len() + 1;
    if !out_buf.is_null() && buf_len >= needed {
        let out_slice = unsafe { slice::from_raw_parts_mut(out_buf.cast::<u8>(), needed) };
        out_slice[..desc.len()].copy_from_slice(desc.as_bytes());
        out_slice[desc.len()] = 0;
    }
    needed
}

/// Free a DataFrame. Passing null is a no-op.
///
/// # Safety
///
/// `df` must be a valid pointer returned by a `scivex_df_*` constructor, or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn scivex_df_free(df: *mut ScivexDataFrame) {
    if !df.is_null() {
        drop(unsafe { Box::from_raw(df) });
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp, clippy::borrow_as_ptr)]
mod tests {
    use super::*;
    use std::ffi::CString;

    #[test]
    fn test_ffi_df_roundtrip() {
        let df = scivex_df_new();
        assert!(!df.is_null());

        let name = CString::new("x").unwrap();
        let data = [1.0, 2.0, 3.0];
        let rc = unsafe { scivex_df_add_column(df, name.as_ptr(), data.as_ptr(), 3) };
        assert_eq!(rc, 0);

        assert_eq!(unsafe { scivex_df_nrows(df) }, 3);
        assert_eq!(unsafe { scivex_df_ncols(df) }, 1);

        // Read column data back
        let mut out = [0.0f64; 3];
        let rc = unsafe { scivex_df_column_data(df, name.as_ptr(), out.as_mut_ptr(), 3) };
        assert_eq!(rc, 0);
        assert_eq!(out, [1.0, 2.0, 3.0]);

        unsafe { scivex_df_free(df) };
    }

    #[test]
    fn test_ffi_df_add_columns() {
        let df = scivex_df_new();

        let name_x = CString::new("x").unwrap();
        let name_y = CString::new("y").unwrap();
        let data_x = [10.0, 20.0];
        let data_y = [30.0, 40.0];

        let rc = unsafe { scivex_df_add_column(df, name_x.as_ptr(), data_x.as_ptr(), 2) };
        assert_eq!(rc, 0);
        let rc = unsafe { scivex_df_add_column(df, name_y.as_ptr(), data_y.as_ptr(), 2) };
        assert_eq!(rc, 0);

        assert_eq!(unsafe { scivex_df_nrows(df) }, 2);
        assert_eq!(unsafe { scivex_df_ncols(df) }, 2);

        unsafe { scivex_df_free(df) };
    }

    #[test]
    fn test_ffi_df_string_column() {
        let df = scivex_df_new();

        let name = CString::new("names").unwrap();
        let s1 = CString::new("alice").unwrap();
        let s2 = CString::new("bob").unwrap();
        let ptrs = [s1.as_ptr(), s2.as_ptr()];

        let rc =
            unsafe { scivex_df_add_string_column(df, name.as_ptr(), ptrs.as_ptr(), ptrs.len()) };
        assert_eq!(rc, 0);
        assert_eq!(unsafe { scivex_df_nrows(df) }, 2);

        unsafe { scivex_df_free(df) };
    }

    #[test]
    fn test_ffi_df_head() {
        let df = scivex_df_new();
        let name = CString::new("x").unwrap();
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        unsafe { scivex_df_add_column(df, name.as_ptr(), data.as_ptr(), 5) };

        let head = unsafe { scivex_df_head(df, 3) };
        assert!(!head.is_null());
        assert_eq!(unsafe { scivex_df_nrows(head) }, 3);

        let mut out = [0.0f64; 3];
        let rc = unsafe { scivex_df_column_data(head, name.as_ptr(), out.as_mut_ptr(), 3) };
        assert_eq!(rc, 0);
        assert_eq!(out, [1.0, 2.0, 3.0]);

        unsafe {
            scivex_df_free(head);
            scivex_df_free(df);
        }
    }

    #[test]
    fn test_ffi_df_filter_gt() {
        let df = scivex_df_new();
        let name = CString::new("val").unwrap();
        let data = [1.0, 5.0, 3.0, 7.0, 2.0];
        unsafe { scivex_df_add_column(df, name.as_ptr(), data.as_ptr(), 5) };

        let filtered = unsafe { scivex_df_filter_gt(df, name.as_ptr(), 3.0) };
        assert!(!filtered.is_null());
        assert_eq!(unsafe { scivex_df_nrows(filtered) }, 2);

        let mut out = [0.0f64; 2];
        let rc = unsafe { scivex_df_column_data(filtered, name.as_ptr(), out.as_mut_ptr(), 2) };
        assert_eq!(rc, 0);
        assert_eq!(out, [5.0, 7.0]);

        unsafe {
            scivex_df_free(filtered);
            scivex_df_free(df);
        }
    }

    #[test]
    fn test_ffi_df_sort_by() {
        let df = scivex_df_new();
        let name = CString::new("val").unwrap();
        let data = [3.0, 1.0, 2.0];
        unsafe { scivex_df_add_column(df, name.as_ptr(), data.as_ptr(), 3) };

        // Ascending
        let sorted = unsafe { scivex_df_sort_by(df, name.as_ptr(), 1) };
        assert!(!sorted.is_null());

        let mut out = [0.0f64; 3];
        let rc = unsafe { scivex_df_column_data(sorted, name.as_ptr(), out.as_mut_ptr(), 3) };
        assert_eq!(rc, 0);
        assert_eq!(out, [1.0, 2.0, 3.0]);

        // Descending
        let sorted_desc = unsafe { scivex_df_sort_by(df, name.as_ptr(), 0) };
        assert!(!sorted_desc.is_null());
        let rc = unsafe { scivex_df_column_data(sorted_desc, name.as_ptr(), out.as_mut_ptr(), 3) };
        assert_eq!(rc, 0);
        assert_eq!(out, [3.0, 2.0, 1.0]);

        unsafe {
            scivex_df_free(sorted);
            scivex_df_free(sorted_desc);
            scivex_df_free(df);
        }
    }

    #[test]
    fn test_ffi_df_select() {
        let df = scivex_df_new();
        let name_a = CString::new("a").unwrap();
        let name_b = CString::new("b").unwrap();
        let data_a = [1.0, 2.0];
        let data_b = [3.0, 4.0];
        unsafe {
            scivex_df_add_column(df, name_a.as_ptr(), data_a.as_ptr(), 2);
            scivex_df_add_column(df, name_b.as_ptr(), data_b.as_ptr(), 2);
        }

        let col_names = [name_b.as_ptr()];
        let selected = unsafe { scivex_df_select(df, col_names.as_ptr(), 1) };
        assert!(!selected.is_null());
        assert_eq!(unsafe { scivex_df_ncols(selected) }, 1);

        let mut out = [0.0f64; 2];
        let rc = unsafe { scivex_df_column_data(selected, name_b.as_ptr(), out.as_mut_ptr(), 2) };
        assert_eq!(rc, 0);
        assert_eq!(out, [3.0, 4.0]);

        unsafe {
            scivex_df_free(selected);
            scivex_df_free(df);
        }
    }

    #[test]
    fn test_ffi_df_column_names() {
        let df = scivex_df_new();
        let name_a = CString::new("alpha").unwrap();
        let name_b = CString::new("beta").unwrap();
        let data = [1.0];
        unsafe {
            scivex_df_add_column(df, name_a.as_ptr(), data.as_ptr(), 1);
            scivex_df_add_column(df, name_b.as_ptr(), data.as_ptr(), 1);
        }

        // Query size
        let needed = unsafe { scivex_df_column_names(df, std::ptr::null_mut(), 0) };
        assert!(needed > 0);

        // Read names
        let mut buf = vec![0u8; needed];
        let written =
            unsafe { scivex_df_column_names(df, buf.as_mut_ptr().cast::<c_char>(), needed) };
        assert_eq!(written, needed);
        let s = std::str::from_utf8(&buf[..needed - 1]).unwrap();
        assert_eq!(s, "alpha\nbeta");

        unsafe { scivex_df_free(df) };
    }

    #[test]
    fn test_ffi_df_describe() {
        let df = scivex_df_new();
        let name = CString::new("x").unwrap();
        let data = [1.0, 2.0, 3.0];
        unsafe { scivex_df_add_column(df, name.as_ptr(), data.as_ptr(), 3) };

        let needed = unsafe { scivex_df_describe(df, std::ptr::null_mut(), 0) };
        assert!(needed > 0);

        let mut buf = vec![0u8; needed];
        unsafe { scivex_df_describe(df, buf.as_mut_ptr().cast::<c_char>(), needed) };
        let s = std::str::from_utf8(&buf[..needed - 1]).unwrap();
        assert!(s.contains('x'));

        unsafe { scivex_df_free(df) };
    }

    #[test]
    fn test_ffi_df_null_safety() {
        unsafe { scivex_df_free(std::ptr::null_mut()) };
    }
}
