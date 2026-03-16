//! C FFI bindings for visualization.

use std::ffi::CStr;
use std::os::raw::{c_char, c_int};
use std::slice;

use scivex_viz::{Axes, BarPlot, Figure, LinePlot, ScatterPlot};

use crate::error::set_error;

/// Opaque handle to a `Figure` with its associated `Axes`.
///
/// The FFI figure holds a single `Axes` that accumulates plots and labels.
/// On render, the axes are placed into a fresh `Figure`.
pub struct ScivexFigure {
    axes: Axes,
    title: Option<String>,
    x_label: Option<String>,
    y_label: Option<String>,
}

/// Helper: optionally apply a label from a C string pointer to a plot.
/// Returns the parsed label string if the pointer is non-null and valid UTF-8.
fn parse_optional_label(label: *const c_char) -> Option<String> {
    if label.is_null() {
        return None;
    }
    unsafe { CStr::from_ptr(label) }
        .to_str()
        .ok()
        .map(String::from)
}

// ---------------------------------------------------------------------------
// Construction & destruction
// ---------------------------------------------------------------------------

/// Create a new figure.
#[unsafe(no_mangle)]
pub extern "C" fn scivex_figure_new() -> *mut ScivexFigure {
    Box::into_raw(Box::new(ScivexFigure {
        axes: Axes::new(),
        title: None,
        x_label: None,
        y_label: None,
    }))
}

/// Free a figure. Passing null is a no-op.
///
/// # Safety
///
/// `fig` must be a valid pointer from `scivex_figure_new`, or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn scivex_figure_free(fig: *mut ScivexFigure) {
    if !fig.is_null() {
        drop(unsafe { Box::from_raw(fig) });
    }
}

// ---------------------------------------------------------------------------
// Plot builders
// ---------------------------------------------------------------------------

/// Add a line plot to the figure.
/// Returns 0 on success, -1 on error.
///
/// # Safety
///
/// `fig` must be a valid pointer. `x` and `y` must each point to `len` f64 values.
/// `label` may be null (no label) or a valid null-terminated C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn scivex_figure_line_plot(
    fig: *mut ScivexFigure,
    x: *const f64,
    y: *const f64,
    len: usize,
    label: *const c_char,
) -> c_int {
    if fig.is_null() || x.is_null() || y.is_null() {
        set_error("null pointer passed to scivex_figure_line_plot");
        return -1;
    }
    let x_data = unsafe { slice::from_raw_parts(x, len) }.to_vec();
    let y_data = unsafe { slice::from_raw_parts(y, len) }.to_vec();
    let mut plot = LinePlot::new(x_data, y_data);
    if let Some(s) = parse_optional_label(label) {
        plot = plot.label(&s);
    }
    let fig_ref = unsafe { &mut *fig };
    let axes = std::mem::replace(&mut fig_ref.axes, Axes::new());
    fig_ref.axes = axes.add_plot(plot);
    0
}

/// Add a scatter plot to the figure.
/// Returns 0 on success, -1 on error.
///
/// # Safety
///
/// `fig` must be a valid pointer. `x` and `y` must each point to `len` f64 values.
/// `label` may be null or a valid null-terminated C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn scivex_figure_scatter_plot(
    fig: *mut ScivexFigure,
    x: *const f64,
    y: *const f64,
    len: usize,
    label: *const c_char,
) -> c_int {
    if fig.is_null() || x.is_null() || y.is_null() {
        set_error("null pointer passed to scivex_figure_scatter_plot");
        return -1;
    }
    let x_data = unsafe { slice::from_raw_parts(x, len) }.to_vec();
    let y_data = unsafe { slice::from_raw_parts(y, len) }.to_vec();
    let mut plot = ScatterPlot::new(x_data, y_data);
    if let Some(s) = parse_optional_label(label) {
        plot = plot.label(&s);
    }
    let fig_ref = unsafe { &mut *fig };
    let axes = std::mem::replace(&mut fig_ref.axes, Axes::new());
    fig_ref.axes = axes.add_plot(plot);
    0
}

/// Add a bar plot to the figure.
/// Returns 0 on success, -1 on error.
///
/// # Safety
///
/// `fig` must be a valid pointer. `labels_ptr` must point to `len` valid
/// null-terminated C string pointers. `values` must point to `len` f64 values.
/// `label` may be null or a valid null-terminated C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn scivex_figure_bar_plot(
    fig: *mut ScivexFigure,
    labels_ptr: *const *const c_char,
    values: *const f64,
    len: usize,
    label: *const c_char,
) -> c_int {
    if fig.is_null() || labels_ptr.is_null() || values.is_null() {
        set_error("null pointer passed to scivex_figure_bar_plot");
        return -1;
    }
    let label_ptrs = unsafe { slice::from_raw_parts(labels_ptr, len) };
    let mut categories = Vec::with_capacity(len);
    for &p in label_ptrs {
        if p.is_null() {
            set_error("null string pointer in bar labels array");
            return -1;
        }
        match unsafe { CStr::from_ptr(p) }.to_str() {
            Ok(s) => categories.push(s.to_string()),
            Err(e) => {
                set_error(&e.to_string());
                return -1;
            }
        }
    }
    let vals = unsafe { slice::from_raw_parts(values, len) }.to_vec();
    let mut plot = BarPlot::new(categories, vals);
    if let Some(s) = parse_optional_label(label) {
        plot = plot.label(&s);
    }
    let fig_ref = unsafe { &mut *fig };
    let axes = std::mem::replace(&mut fig_ref.axes, Axes::new());
    fig_ref.axes = axes.add_plot(plot);
    0
}

// ---------------------------------------------------------------------------
// Labels
// ---------------------------------------------------------------------------

/// Set the figure title.
///
/// # Safety
///
/// `fig` must be a valid pointer. `title` must be a valid null-terminated C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn scivex_figure_title(
    fig: *mut ScivexFigure,
    title: *const c_char,
) -> c_int {
    if fig.is_null() || title.is_null() {
        set_error("null pointer passed to scivex_figure_title");
        return -1;
    }
    match unsafe { CStr::from_ptr(title) }.to_str() {
        Ok(s) => {
            let fig_ref = unsafe { &mut *fig };
            fig_ref.title = Some(s.to_string());
            0
        }
        Err(e) => {
            set_error(&e.to_string());
            -1
        }
    }
}

/// Set the x-axis label.
///
/// # Safety
///
/// `fig` must be a valid pointer. `label` must be a valid null-terminated C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn scivex_figure_x_label(
    fig: *mut ScivexFigure,
    label: *const c_char,
) -> c_int {
    if fig.is_null() || label.is_null() {
        set_error("null pointer passed to scivex_figure_x_label");
        return -1;
    }
    match unsafe { CStr::from_ptr(label) }.to_str() {
        Ok(s) => {
            let fig_ref = unsafe { &mut *fig };
            fig_ref.x_label = Some(s.to_string());
            0
        }
        Err(e) => {
            set_error(&e.to_string());
            -1
        }
    }
}

/// Set the y-axis label.
///
/// # Safety
///
/// `fig` must be a valid pointer. `label` must be a valid null-terminated C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn scivex_figure_y_label(
    fig: *mut ScivexFigure,
    label: *const c_char,
) -> c_int {
    if fig.is_null() || label.is_null() {
        set_error("null pointer passed to scivex_figure_y_label");
        return -1;
    }
    match unsafe { CStr::from_ptr(label) }.to_str() {
        Ok(s) => {
            let fig_ref = unsafe { &mut *fig };
            fig_ref.y_label = Some(s.to_string());
            0
        }
        Err(e) => {
            set_error(&e.to_string());
            -1
        }
    }
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

/// Build the axes with labels applied, consuming the stored axes.
fn finalize_axes(fig: &mut ScivexFigure) -> Axes {
    let mut axes = std::mem::replace(&mut fig.axes, Axes::new());
    if let Some(ref t) = fig.title {
        axes = axes.title(t);
    }
    if let Some(ref l) = fig.x_label {
        axes = axes.x_label(l);
    }
    if let Some(ref l) = fig.y_label {
        axes = axes.y_label(l);
    }
    axes
}

/// Render the figure to an SVG string. Returns the number of bytes needed
/// (including null terminator). If `buf_len` is large enough, writes
/// the SVG string to `out_buf`.
///
/// **Note:** This function consumes the accumulated plots. After calling it,
/// the figure's plot list is empty (labels are preserved). Add plots again
/// before rendering a second time.
///
/// # Safety
///
/// `fig` must be a valid pointer. `out_buf` may be null (to query size only)
/// or must point to `buf_len` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn scivex_figure_to_svg(
    fig: *mut ScivexFigure,
    out_buf: *mut c_char,
    buf_len: usize,
) -> usize {
    if fig.is_null() {
        set_error("null pointer passed to scivex_figure_to_svg");
        return 0;
    }
    let fig_ref = unsafe { &mut *fig };
    let axes = finalize_axes(fig_ref);
    let figure = Figure::new().plot(axes);

    match figure.to_svg() {
        Ok(svg) => {
            let needed = svg.len() + 1;
            if !out_buf.is_null() && buf_len >= needed {
                let out_slice = unsafe { slice::from_raw_parts_mut(out_buf.cast::<u8>(), needed) };
                out_slice[..svg.len()].copy_from_slice(svg.as_bytes());
                out_slice[svg.len()] = 0;
            }
            needed
        }
        Err(e) => {
            set_error(&e.to_string());
            0
        }
    }
}

/// Save the figure as an SVG file.
/// Returns 0 on success, -1 on error.
///
/// **Note:** This function consumes the accumulated plots (same as
/// `scivex_figure_to_svg`).
///
/// # Safety
///
/// `fig` must be a valid pointer. `path` must be a valid null-terminated C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn scivex_figure_to_svg_file(
    fig: *mut ScivexFigure,
    path: *const c_char,
) -> c_int {
    if fig.is_null() || path.is_null() {
        set_error("null pointer passed to scivex_figure_to_svg_file");
        return -1;
    }
    let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
        Ok(s) => s.to_string(),
        Err(e) => {
            set_error(&e.to_string());
            return -1;
        }
    };
    let fig_ref = unsafe { &mut *fig };
    let axes = finalize_axes(fig_ref);
    let figure = Figure::new().plot(axes);

    match figure.save_svg(&path_str) {
        Ok(()) => 0,
        Err(e) => {
            set_error(&e.to_string());
            -1
        }
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp, clippy::borrow_as_ptr)]
mod tests {
    use super::*;
    use std::ffi::CString;

    #[test]
    fn test_ffi_figure_create_free() {
        let fig = scivex_figure_new();
        assert!(!fig.is_null());
        unsafe { scivex_figure_free(fig) };
    }

    #[test]
    fn test_ffi_figure_null_free() {
        unsafe { scivex_figure_free(std::ptr::null_mut()) };
    }

    #[test]
    fn test_ffi_figure_line_plot() {
        let fig = scivex_figure_new();
        let x = [0.0, 1.0, 2.0];
        let y = [0.0, 1.0, 0.5];
        let label = CString::new("test line").unwrap();

        let rc = unsafe { scivex_figure_line_plot(fig, x.as_ptr(), y.as_ptr(), 3, label.as_ptr()) };
        assert_eq!(rc, 0);

        unsafe { scivex_figure_free(fig) };
    }

    #[test]
    fn test_ffi_figure_scatter_plot() {
        let fig = scivex_figure_new();
        let x = [1.0, 2.0, 3.0];
        let y = [4.0, 5.0, 6.0];

        let rc =
            unsafe { scivex_figure_scatter_plot(fig, x.as_ptr(), y.as_ptr(), 3, std::ptr::null()) };
        assert_eq!(rc, 0);

        unsafe { scivex_figure_free(fig) };
    }

    #[test]
    fn test_ffi_figure_bar_plot() {
        let fig = scivex_figure_new();
        let cat_a = CString::new("A").unwrap();
        let cat_b = CString::new("B").unwrap();
        let cat_c = CString::new("C").unwrap();
        let cats = [cat_a.as_ptr(), cat_b.as_ptr(), cat_c.as_ptr()];
        let vals = [10.0, 20.0, 30.0];

        let rc = unsafe {
            scivex_figure_bar_plot(fig, cats.as_ptr(), vals.as_ptr(), 3, std::ptr::null())
        };
        assert_eq!(rc, 0);

        unsafe { scivex_figure_free(fig) };
    }

    #[test]
    fn test_ffi_figure_labels() {
        let fig = scivex_figure_new();
        let title = CString::new("My Plot").unwrap();
        let xlabel = CString::new("X axis").unwrap();
        let ylabel = CString::new("Y axis").unwrap();

        assert_eq!(unsafe { scivex_figure_title(fig, title.as_ptr()) }, 0);
        assert_eq!(unsafe { scivex_figure_x_label(fig, xlabel.as_ptr()) }, 0);
        assert_eq!(unsafe { scivex_figure_y_label(fig, ylabel.as_ptr()) }, 0);

        unsafe { scivex_figure_free(fig) };
    }

    #[test]
    fn test_ffi_figure_render_svg() {
        let fig = scivex_figure_new();
        let x = [0.0, 1.0, 2.0];
        let y = [0.0, 1.0, 0.5];
        let title = CString::new("Test").unwrap();

        unsafe {
            scivex_figure_line_plot(fig, x.as_ptr(), y.as_ptr(), 3, std::ptr::null());
            scivex_figure_title(fig, title.as_ptr());
        }

        // Query SVG size (consumes plots)
        let needed = unsafe { scivex_figure_to_svg(fig, std::ptr::null_mut(), 0) };
        assert!(needed > 0, "SVG rendering should produce output");

        unsafe { scivex_figure_free(fig) };
    }

    #[test]
    fn test_ffi_figure_render_svg_to_buffer() {
        let fig = scivex_figure_new();
        let x = [0.0, 1.0];
        let y = [0.0, 1.0];
        unsafe {
            scivex_figure_line_plot(fig, x.as_ptr(), y.as_ptr(), 2, std::ptr::null());
        }

        // Render to get needed size and write to buffer in one call.
        // We need a big enough buffer — allocate generously.
        let mut buf = vec![0u8; 65536];
        let needed =
            unsafe { scivex_figure_to_svg(fig, buf.as_mut_ptr().cast::<c_char>(), buf.len()) };
        assert!(needed > 0);
        assert!(needed <= buf.len());

        let svg = std::str::from_utf8(&buf[..needed - 1]).unwrap();
        assert!(svg.contains("<svg"));

        unsafe { scivex_figure_free(fig) };
    }

    #[test]
    fn test_ffi_figure_to_svg_file() {
        let fig = scivex_figure_new();
        let x = [0.0, 1.0, 2.0];
        let y = [0.0, 1.0, 0.5];
        unsafe {
            scivex_figure_line_plot(fig, x.as_ptr(), y.as_ptr(), 3, std::ptr::null());
        }

        let tmp = std::env::temp_dir().join("scivex_ffi_test.svg");
        let path = CString::new(tmp.to_str().unwrap()).unwrap();
        let rc = unsafe { scivex_figure_to_svg_file(fig, path.as_ptr()) };
        assert_eq!(rc, 0);

        let contents = std::fs::read_to_string(&tmp).unwrap();
        assert!(contents.contains("<svg"));

        // Cleanup
        let _ = std::fs::remove_file(&tmp);

        unsafe { scivex_figure_free(fig) };
    }
}
