//! C FFI bindings for machine learning.

use std::os::raw::c_int;

use crate::error::set_error;
use crate::tensor::ScivexTensor;

// ---------------------------------------------------------------------------
// Linear Regression
// ---------------------------------------------------------------------------

/// Opaque handle to a `LinearRegression<f64>`.
pub struct ScivexLinearRegression {
    inner: scivex_ml::linear::LinearRegression<f64>,
}

/// Create a new (unfitted) linear regression model.
#[unsafe(no_mangle)]
pub extern "C" fn scivex_linreg_new() -> *mut ScivexLinearRegression {
    Box::into_raw(Box::new(ScivexLinearRegression {
        inner: scivex_ml::linear::LinearRegression::new(),
    }))
}

/// Free a linear regression model.
///
/// # Safety
///
/// `m` must be a valid pointer from `scivex_linreg_new`, or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn scivex_linreg_free(m: *mut ScivexLinearRegression) {
    if !m.is_null() {
        drop(unsafe { Box::from_raw(m) });
    }
}

/// Fit the model. `x` has shape [n_samples, n_features], `y` has shape [n_samples].
/// Returns 0 on success, -1 on error.
///
/// # Safety
///
/// All pointers must be valid and non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn scivex_linreg_fit(
    m: *mut ScivexLinearRegression,
    x: *const ScivexTensor,
    y: *const ScivexTensor,
) -> c_int {
    use scivex_ml::traits::Predictor;
    match unsafe { (*m).inner.fit((*x).inner(), (*y).inner()) } {
        Ok(()) => 0,
        Err(e) => {
            set_error(&e.to_string());
            -1
        }
    }
}

/// Predict target values for `x`. Returns a new tensor or null on error.
///
/// # Safety
///
/// `m` and `x` must be valid, non-null pointers. Model must be fitted.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn scivex_linreg_predict(
    m: *const ScivexLinearRegression,
    x: *const ScivexTensor,
) -> *mut ScivexTensor {
    use scivex_ml::traits::Predictor;
    match unsafe { (*m).inner.predict((*x).inner()) } {
        Ok(t) => Box::into_raw(Box::new(ScivexTensor::from_inner(t))),
        Err(e) => {
            set_error(&e.to_string());
            std::ptr::null_mut()
        }
    }
}

/// Get fitted weights. Writes up to `out_len` values into `out`.
/// Returns the number of weights, or -1 on error.
///
/// # Safety
///
/// `m` must be a fitted model. `out` must point to `out_len` f64 values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn scivex_linreg_weights(
    m: *const ScivexLinearRegression,
    out: *mut f64,
    out_len: usize,
) -> c_int {
    if let Some(w) = unsafe { (*m).inner.weights() } {
        let n = w.len().min(out_len);
        let out_slice = unsafe { std::slice::from_raw_parts_mut(out, n) };
        out_slice.copy_from_slice(&w[..n]);
        #[allow(clippy::cast_possible_wrap)]
        let len = w.len() as c_int;
        len
    } else {
        set_error("model not fitted");
        -1
    }
}

/// Get fitted bias (intercept). Returns 0 on success, -1 if not fitted.
///
/// # Safety
///
/// `m` must be a valid model pointer. `out` must be non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn scivex_linreg_bias(
    m: *const ScivexLinearRegression,
    out: *mut f64,
) -> c_int {
    if let Some(b) = unsafe { (*m).inner.bias() } {
        unsafe { *out = b };
        0
    } else {
        set_error("model not fitted");
        -1
    }
}

// ---------------------------------------------------------------------------
// K-Means
// ---------------------------------------------------------------------------

/// Opaque handle to a `KMeans<f64>`.
pub struct ScivexKMeans {
    inner: scivex_ml::cluster::KMeans<f64>,
}

/// Create a new KMeans model.
/// Returns null on invalid parameters.
#[unsafe(no_mangle)]
pub extern "C" fn scivex_kmeans_new(
    n_clusters: usize,
    max_iter: usize,
    n_init: usize,
    seed: u64,
) -> *mut ScivexKMeans {
    match scivex_ml::cluster::KMeans::new(n_clusters, max_iter, 1e-8, n_init, seed) {
        Ok(inner) => Box::into_raw(Box::new(ScivexKMeans { inner })),
        Err(e) => {
            set_error(&e.to_string());
            std::ptr::null_mut()
        }
    }
}

/// Free a KMeans model.
///
/// # Safety
///
/// `m` must be a valid pointer from `scivex_kmeans_new`, or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn scivex_kmeans_free(m: *mut ScivexKMeans) {
    if !m.is_null() {
        drop(unsafe { Box::from_raw(m) });
    }
}

/// Fit the KMeans model. `x` has shape [n_samples, n_features].
/// Returns 0 on success, -1 on error.
///
/// # Safety
///
/// `m` and `x` must be valid, non-null pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn scivex_kmeans_fit(m: *mut ScivexKMeans, x: *const ScivexTensor) -> c_int {
    match unsafe { (*m).inner.fit((*x).inner()) } {
        Ok(()) => 0,
        Err(e) => {
            set_error(&e.to_string());
            -1
        }
    }
}

/// Predict cluster labels. Returns a new tensor or null on error.
///
/// # Safety
///
/// `m` and `x` must be valid, non-null pointers. Model must be fitted.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn scivex_kmeans_predict(
    m: *const ScivexKMeans,
    x: *const ScivexTensor,
) -> *mut ScivexTensor {
    match unsafe { (*m).inner.predict((*x).inner()) } {
        Ok(t) => Box::into_raw(Box::new(ScivexTensor::from_inner(t))),
        Err(e) => {
            set_error(&e.to_string());
            std::ptr::null_mut()
        }
    }
}

/// Get the inertia. Returns the value, or NaN if not fitted.
///
/// # Safety
///
/// `m` must be a valid, non-null pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn scivex_kmeans_inertia(m: *const ScivexKMeans) -> f64 {
    unsafe { (*m).inner.inertia().unwrap_or(f64::NAN) }
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

/// Accuracy score. Returns 0 on success, -1 on error.
///
/// # Safety
///
/// `y_true` and `y_pred` must each point to `len` valid f64 values.
/// `out` must be non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn scivex_metrics_accuracy(
    y_true: *const f64,
    y_pred: *const f64,
    len: usize,
    out: *mut f64,
) -> c_int {
    let yt = unsafe { std::slice::from_raw_parts(y_true, len) };
    let yp = unsafe { std::slice::from_raw_parts(y_pred, len) };
    match scivex_ml::metrics::accuracy(yt, yp) {
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

/// Mean squared error. Returns 0 on success, -1 on error.
///
/// # Safety
///
/// `y_true` and `y_pred` must each point to `len` valid f64 values.
/// `out` must be non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn scivex_metrics_mse(
    y_true: *const f64,
    y_pred: *const f64,
    len: usize,
    out: *mut f64,
) -> c_int {
    let yt = unsafe { std::slice::from_raw_parts(y_true, len) };
    let yp = unsafe { std::slice::from_raw_parts(y_pred, len) };
    match scivex_ml::metrics::mse(yt, yp) {
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
    use crate::tensor::{scivex_tensor_data, scivex_tensor_free, scivex_tensor_from_array};

    #[test]
    fn test_ffi_linreg_fit_predict() {
        // y = 2*x1 + 3*x2 + 1
        let x_data = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0];
        let x_shape = [4usize, 2];
        let y_data = [3.0, 4.0, 6.0, 8.0];
        let y_shape = [4usize];

        let x = unsafe { scivex_tensor_from_array(x_data.as_ptr(), 8, x_shape.as_ptr(), 2) };
        let y = unsafe { scivex_tensor_from_array(y_data.as_ptr(), 4, y_shape.as_ptr(), 1) };

        let m = scivex_linreg_new();
        let rc = unsafe { scivex_linreg_fit(m, x, y) };
        assert_eq!(rc, 0);

        // Predict
        let pred = unsafe { scivex_linreg_predict(m, x) };
        assert!(!pred.is_null());

        let mut pred_data = [0.0f64; 4];
        unsafe { scivex_tensor_data(pred, pred_data.as_mut_ptr(), 4) };
        // Check predictions are close to y
        for (p, &actual) in pred_data.iter().zip(y_data.iter()) {
            assert!((p - actual).abs() < 0.5, "pred={p}, actual={actual}");
        }

        // Get bias
        let mut bias = 0.0;
        let rc = unsafe { scivex_linreg_bias(m, &mut bias) };
        assert_eq!(rc, 0);

        unsafe {
            scivex_tensor_free(x);
            scivex_tensor_free(y);
            scivex_tensor_free(pred);
            scivex_linreg_free(m);
        }
    }

    #[test]
    fn test_ffi_kmeans() {
        let data = [0.0, 0.0, 0.1, 0.1, 10.0, 10.0, 10.1, 10.1];
        let shape = [4usize, 2];
        let x = unsafe { scivex_tensor_from_array(data.as_ptr(), 8, shape.as_ptr(), 2) };

        let m = scivex_kmeans_new(2, 100, 3, 42);
        assert!(!m.is_null());

        let rc = unsafe { scivex_kmeans_fit(m, x) };
        assert_eq!(rc, 0);

        let labels = unsafe { scivex_kmeans_predict(m, x) };
        assert!(!labels.is_null());

        let mut label_data = [0.0f64; 4];
        unsafe { scivex_tensor_data(labels, label_data.as_mut_ptr(), 4) };
        // Points 0,1 should have same label; points 2,3 should have same label
        assert_eq!(label_data[0], label_data[1]);
        assert_eq!(label_data[2], label_data[3]);
        assert_ne!(label_data[0], label_data[2]);

        let inertia = unsafe { scivex_kmeans_inertia(m) };
        assert!(inertia > 0.0);

        unsafe {
            scivex_tensor_free(x);
            scivex_tensor_free(labels);
            scivex_kmeans_free(m);
        }
    }

    #[test]
    fn test_ffi_metrics() {
        let yt = [1.0, 0.0, 1.0, 1.0];
        let yp = [1.0, 0.0, 0.0, 1.0];
        let mut out = 0.0;

        let rc = unsafe { scivex_metrics_accuracy(yt.as_ptr(), yp.as_ptr(), 4, &mut out) };
        assert_eq!(rc, 0);
        assert!((out - 0.75).abs() < 1e-10);

        let yt2 = [1.0, 2.0, 3.0];
        let yp2 = [1.0, 2.0, 3.0];
        let rc = unsafe { scivex_metrics_mse(yt2.as_ptr(), yp2.as_ptr(), 3, &mut out) };
        assert_eq!(rc, 0);
        assert!(out.abs() < 1e-10);
    }
}
