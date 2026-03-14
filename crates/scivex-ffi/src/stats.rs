//! C FFI bindings for statistical functions.

use std::os::raw::c_int;
use std::slice;

use crate::error::set_error;

// ---------------------------------------------------------------------------
// Descriptive statistics (operate on raw f64 arrays)
// ---------------------------------------------------------------------------

/// Compute the mean of a slice.
///
/// # Safety
///
/// `data` must point to `len` valid f64 values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn scivex_stats_mean(data: *const f64, len: usize) -> f64 {
    let s = unsafe { slice::from_raw_parts(data, len) };
    scivex_stats::descriptive::mean(s).unwrap_or(f64::NAN)
}

/// Compute the variance of a slice.
///
/// # Safety
///
/// `data` must point to `len` valid f64 values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn scivex_stats_variance(data: *const f64, len: usize) -> f64 {
    let s = unsafe { slice::from_raw_parts(data, len) };
    scivex_stats::descriptive::variance(s).unwrap_or(f64::NAN)
}

/// Compute the standard deviation of a slice.
///
/// # Safety
///
/// `data` must point to `len` valid f64 values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn scivex_stats_std_dev(data: *const f64, len: usize) -> f64 {
    let s = unsafe { slice::from_raw_parts(data, len) };
    scivex_stats::descriptive::std_dev(s).unwrap_or(f64::NAN)
}

/// Compute the median of a slice.
///
/// # Safety
///
/// `data` must point to `len` valid f64 values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn scivex_stats_median(data: *const f64, len: usize) -> f64 {
    let s = unsafe { slice::from_raw_parts(data, len) };
    scivex_stats::descriptive::median(s).unwrap_or(f64::NAN)
}

/// Compute Pearson correlation between two slices.
/// Returns 0 on success, -1 on error.
///
/// # Safety
///
/// `x` and `y` must each point to `len` valid f64 values. `out` must be non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn scivex_stats_pearson(
    x: *const f64,
    y: *const f64,
    len: usize,
    out: *mut f64,
) -> c_int {
    let xs = unsafe { slice::from_raw_parts(x, len) };
    let ys = unsafe { slice::from_raw_parts(y, len) };
    match scivex_stats::correlation::pearson(xs, ys) {
        Ok(v) => {
            unsafe { *out = v };
            0
        }
        Err(e) => {
            set_error(&e.to_string());
            -1
        }
    }
}

// ---------------------------------------------------------------------------
// Normal distribution
// ---------------------------------------------------------------------------

/// Opaque handle to a Normal distribution.
pub struct ScivexNormal {
    inner: scivex_stats::distributions::Normal<f64>,
}

/// Create a Normal distribution with the given mean and standard deviation.
/// Returns null on invalid parameters (std_dev <= 0).
#[unsafe(no_mangle)]
pub extern "C" fn scivex_normal_new(mean: f64, std_dev: f64) -> *mut ScivexNormal {
    match scivex_stats::distributions::Normal::new(mean, std_dev) {
        Ok(n) => Box::into_raw(Box::new(ScivexNormal { inner: n })),
        Err(e) => {
            set_error(&e.to_string());
            std::ptr::null_mut()
        }
    }
}

/// Free a Normal distribution handle.
///
/// # Safety
///
/// `n` must be a valid pointer from `scivex_normal_new`, or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn scivex_normal_free(n: *mut ScivexNormal) {
    if !n.is_null() {
        drop(unsafe { Box::from_raw(n) });
    }
}

/// Evaluate the PDF at `x`.
///
/// # Safety
///
/// `n` must be a valid, non-null normal distribution pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn scivex_normal_pdf(n: *const ScivexNormal, x: f64) -> f64 {
    use scivex_stats::distributions::Distribution;
    unsafe { (*n).inner.pdf(x) }
}

/// Evaluate the CDF at `x`.
///
/// # Safety
///
/// `n` must be a valid, non-null normal distribution pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn scivex_normal_cdf(n: *const ScivexNormal, x: f64) -> f64 {
    use scivex_stats::distributions::Distribution;
    unsafe { (*n).inner.cdf(x) }
}

/// Evaluate the inverse CDF (PPF) at probability `p`.
/// Returns 0 on success, -1 on error.
///
/// # Safety
///
/// `n` must be a valid normal distribution pointer. `out` must be non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn scivex_normal_ppf(n: *const ScivexNormal, p: f64, out: *mut f64) -> c_int {
    use scivex_stats::distributions::Distribution;
    match unsafe { (*n).inner.ppf(p) } {
        Ok(v) => {
            unsafe { *out = v };
            0
        }
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

    #[test]
    fn test_ffi_stats_descriptive() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let m = unsafe { scivex_stats_mean(data.as_ptr(), 5) };
        assert!((m - 3.0).abs() < 1e-10);

        let v = unsafe { scivex_stats_variance(data.as_ptr(), 5) };
        assert!((v - 2.5).abs() < 1e-10); // sample variance (ddof=1)

        let med = unsafe { scivex_stats_median(data.as_ptr(), 5) };
        assert!((med - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_ffi_stats_pearson() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [2.0, 4.0, 6.0, 8.0, 10.0];
        let mut out = 0.0;
        let rc = unsafe { scivex_stats_pearson(x.as_ptr(), y.as_ptr(), 5, &mut out) };
        assert_eq!(rc, 0);
        assert!((out - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_ffi_normal_distribution() {
        let n = scivex_normal_new(0.0, 1.0);
        assert!(!n.is_null());

        let pdf = unsafe { scivex_normal_pdf(n, 0.0) };
        assert!((pdf - 0.398_942_28).abs() < 1e-5);

        let cdf = unsafe { scivex_normal_cdf(n, 0.0) };
        assert!((cdf - 0.5).abs() < 1e-6);

        let mut ppf_out = 0.0;
        let rc = unsafe { scivex_normal_ppf(n, 0.5, &mut ppf_out) };
        assert_eq!(rc, 0);
        assert!(ppf_out.abs() < 1e-5);

        unsafe { scivex_normal_free(n) };
    }

    #[test]
    fn test_ffi_normal_invalid() {
        let n = scivex_normal_new(0.0, -1.0);
        assert!(n.is_null());
    }
}
