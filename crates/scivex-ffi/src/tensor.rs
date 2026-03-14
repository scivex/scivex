//! C FFI bindings for Tensor operations.

use std::os::raw::c_int;
use std::slice;

use scivex_core::Tensor;

use crate::error::set_error;

/// Opaque handle to a `Tensor<f64>`.
pub struct ScivexTensor {
    inner: Tensor<f64>,
}

// ---------------------------------------------------------------------------
// Construction & destruction
// ---------------------------------------------------------------------------

/// Create a tensor from a flat array and shape.
///
/// Returns a pointer to the tensor, or null on failure.
/// The caller takes ownership and must call `scivex_tensor_free`.
///
/// # Safety
///
/// `data` must point to `data_len` valid f64 values.
/// `shape` must point to `shape_len` valid usize values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn scivex_tensor_from_array(
    data: *const f64,
    data_len: usize,
    shape: *const usize,
    shape_len: usize,
) -> *mut ScivexTensor {
    if data.is_null() || shape.is_null() {
        set_error("null pointer passed to scivex_tensor_from_array");
        return std::ptr::null_mut();
    }
    let data_slice = unsafe { slice::from_raw_parts(data, data_len) };
    let shape_slice = unsafe { slice::from_raw_parts(shape, shape_len) };

    match Tensor::from_vec(data_slice.to_vec(), shape_slice.to_vec()) {
        Ok(t) => Box::into_raw(Box::new(ScivexTensor { inner: t })),
        Err(e) => {
            set_error(&e.to_string());
            std::ptr::null_mut()
        }
    }
}

/// Create a tensor of zeros with the given shape.
///
/// # Safety
///
/// `shape` must point to `shape_len` valid usize values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn scivex_tensor_zeros(
    shape: *const usize,
    shape_len: usize,
) -> *mut ScivexTensor {
    if shape.is_null() {
        set_error("null pointer passed to scivex_tensor_zeros");
        return std::ptr::null_mut();
    }
    let shape_slice = unsafe { slice::from_raw_parts(shape, shape_len) };
    let t = Tensor::zeros(shape_slice.to_vec());
    Box::into_raw(Box::new(ScivexTensor { inner: t }))
}

/// Create a tensor of ones with the given shape.
///
/// # Safety
///
/// `shape` must point to `shape_len` valid usize values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn scivex_tensor_ones(
    shape: *const usize,
    shape_len: usize,
) -> *mut ScivexTensor {
    if shape.is_null() {
        set_error("null pointer passed to scivex_tensor_ones");
        return std::ptr::null_mut();
    }
    let shape_slice = unsafe { slice::from_raw_parts(shape, shape_len) };
    let t = Tensor::ones(shape_slice.to_vec());
    Box::into_raw(Box::new(ScivexTensor { inner: t }))
}

/// Create an identity matrix of size n×n.
#[unsafe(no_mangle)]
pub extern "C" fn scivex_tensor_eye(n: usize) -> *mut ScivexTensor {
    let t = Tensor::eye(n);
    Box::into_raw(Box::new(ScivexTensor { inner: t }))
}

/// Free a tensor. Passing null is a no-op.
///
/// # Safety
///
/// `t` must be a valid pointer returned by a `scivex_tensor_*` constructor,
/// or null. After this call, `t` is invalid and must not be used.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn scivex_tensor_free(t: *mut ScivexTensor) {
    if !t.is_null() {
        drop(unsafe { Box::from_raw(t) });
    }
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

/// Get the number of dimensions.
///
/// # Safety
///
/// `t` must be a valid, non-null tensor pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn scivex_tensor_ndim(t: *const ScivexTensor) -> usize {
    unsafe { (*t).inner.shape().len() }
}

/// Get the total number of elements.
///
/// # Safety
///
/// `t` must be a valid, non-null tensor pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn scivex_tensor_numel(t: *const ScivexTensor) -> usize {
    unsafe { (*t).inner.as_slice().len() }
}

/// Copy the shape into the caller's buffer.
///
/// Returns 0 on success, -1 if the buffer is too small.
///
/// # Safety
///
/// `t` must be a valid tensor pointer. `out` must point to a buffer of at
/// least `out_len` usizes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn scivex_tensor_shape(
    t: *const ScivexTensor,
    out: *mut usize,
    out_len: usize,
) -> c_int {
    let shape = unsafe { (*t).inner.shape() };
    if out_len < shape.len() {
        set_error("output buffer too small for shape");
        return -1;
    }
    let out_slice = unsafe { slice::from_raw_parts_mut(out, shape.len()) };
    out_slice.copy_from_slice(shape);
    0
}

/// Copy the tensor data into the caller's buffer.
///
/// Returns 0 on success, -1 if the buffer is too small.
///
/// # Safety
///
/// `t` must be a valid tensor pointer. `out` must point to a buffer of at
/// least `out_len` f64 values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn scivex_tensor_data(
    t: *const ScivexTensor,
    out: *mut f64,
    out_len: usize,
) -> c_int {
    let data = unsafe { (*t).inner.as_slice() };
    if out_len < data.len() {
        set_error("output buffer too small for data");
        return -1;
    }
    let out_slice = unsafe { slice::from_raw_parts_mut(out, data.len()) };
    out_slice.copy_from_slice(data);
    0
}

/// Get a pointer to the tensor's internal data (read-only, no copy).
///
/// The pointer is valid as long as the tensor is alive and not mutated.
///
/// # Safety
///
/// `t` must be a valid tensor pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn scivex_tensor_data_ptr(t: *const ScivexTensor) -> *const f64 {
    unsafe { (*t).inner.as_slice().as_ptr() }
}

// ---------------------------------------------------------------------------
// Arithmetic
// ---------------------------------------------------------------------------

/// Element-wise addition: `a + b`. Returns a new tensor or null on error.
///
/// # Safety
///
/// Both `a` and `b` must be valid, non-null tensor pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn scivex_tensor_add(
    a: *const ScivexTensor,
    b: *const ScivexTensor,
) -> *mut ScivexTensor {
    match unsafe { (*a).inner.add_checked(&(*b).inner) } {
        Ok(t) => Box::into_raw(Box::new(ScivexTensor { inner: t })),
        Err(e) => {
            set_error(&e.to_string());
            std::ptr::null_mut()
        }
    }
}

/// Element-wise subtraction: `a - b`. Returns a new tensor or null on error.
///
/// # Safety
///
/// Both `a` and `b` must be valid, non-null tensor pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn scivex_tensor_sub(
    a: *const ScivexTensor,
    b: *const ScivexTensor,
) -> *mut ScivexTensor {
    match unsafe { (*a).inner.sub_checked(&(*b).inner) } {
        Ok(t) => Box::into_raw(Box::new(ScivexTensor { inner: t })),
        Err(e) => {
            set_error(&e.to_string());
            std::ptr::null_mut()
        }
    }
}

/// Element-wise multiplication: `a * b`. Returns a new tensor or null on error.
///
/// # Safety
///
/// Both `a` and `b` must be valid, non-null tensor pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn scivex_tensor_mul(
    a: *const ScivexTensor,
    b: *const ScivexTensor,
) -> *mut ScivexTensor {
    match unsafe { (*a).inner.zip_map(&(*b).inner, |x, y| x * y) } {
        Ok(t) => Box::into_raw(Box::new(ScivexTensor { inner: t })),
        Err(e) => {
            set_error(&e.to_string());
            std::ptr::null_mut()
        }
    }
}

/// Matrix multiplication: `a @ b`. Returns a new tensor or null on error.
///
/// # Safety
///
/// Both `a` and `b` must be valid, non-null tensor pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn scivex_tensor_matmul(
    a: *const ScivexTensor,
    b: *const ScivexTensor,
) -> *mut ScivexTensor {
    match unsafe { (*a).inner.matmul(&(*b).inner) } {
        Ok(t) => Box::into_raw(Box::new(ScivexTensor { inner: t })),
        Err(e) => {
            set_error(&e.to_string());
            std::ptr::null_mut()
        }
    }
}

/// Dot product of two 1-D tensors. Returns 0 on success, -1 on error.
///
/// # Safety
///
/// Both `a` and `b` must be valid tensor pointers. `out` must be non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn scivex_tensor_dot(
    a: *const ScivexTensor,
    b: *const ScivexTensor,
    out: *mut f64,
) -> c_int {
    match unsafe { (*a).inner.dot(&(*b).inner) } {
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

/// Transpose a 2-D tensor. Returns a new tensor or null on error.
///
/// # Safety
///
/// `t` must be a valid, non-null tensor pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn scivex_tensor_transpose(t: *const ScivexTensor) -> *mut ScivexTensor {
    match unsafe { (*t).inner.transpose() } {
        Ok(r) => Box::into_raw(Box::new(ScivexTensor { inner: r })),
        Err(e) => {
            set_error(&e.to_string());
            std::ptr::null_mut()
        }
    }
}

/// Add a scalar to every element. Returns a new tensor.
///
/// # Safety
///
/// `t` must be a valid, non-null tensor pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn scivex_tensor_add_scalar(
    t: *const ScivexTensor,
    val: f64,
) -> *mut ScivexTensor {
    let result = unsafe { (*t).inner.map(|x| x + val) };
    Box::into_raw(Box::new(ScivexTensor { inner: result }))
}

/// Multiply every element by a scalar. Returns a new tensor.
///
/// # Safety
///
/// `t` must be a valid, non-null tensor pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn scivex_tensor_mul_scalar(
    t: *const ScivexTensor,
    val: f64,
) -> *mut ScivexTensor {
    let result = unsafe { (*t).inner.map(|x| x * val) };
    Box::into_raw(Box::new(ScivexTensor { inner: result }))
}

// ---------------------------------------------------------------------------
// Reductions
// ---------------------------------------------------------------------------

/// Sum of all elements.
///
/// # Safety
///
/// `t` must be a valid, non-null tensor pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn scivex_tensor_sum(t: *const ScivexTensor) -> f64 {
    unsafe { (*t).inner.as_slice().iter().sum() }
}

/// Mean of all elements.
///
/// # Safety
///
/// `t` must be a valid, non-null tensor pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn scivex_tensor_mean(t: *const ScivexTensor) -> f64 {
    let s = unsafe { (*t).inner.as_slice() };
    if s.is_empty() {
        return 0.0;
    }
    s.iter().sum::<f64>() / s.len() as f64
}

/// Determinant of a square matrix. Returns 0 on success, -1 on error.
///
/// # Safety
///
/// `t` must be a valid tensor pointer. `out` must be non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn scivex_tensor_det(t: *const ScivexTensor, out: *mut f64) -> c_int {
    match scivex_core::linalg::det(unsafe { &(*t).inner }) {
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
// Reshape
// ---------------------------------------------------------------------------

/// Reshape a tensor (returns a new tensor, does not modify the original).
///
/// # Safety
///
/// `t` must be a valid tensor pointer. `shape` must point to `shape_len` usizes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn scivex_tensor_reshape(
    t: *const ScivexTensor,
    shape: *const usize,
    shape_len: usize,
) -> *mut ScivexTensor {
    let shape_slice = unsafe { slice::from_raw_parts(shape, shape_len) };
    match unsafe { (*t).inner.reshaped(shape_slice.to_vec()) } {
        Ok(r) => Box::into_raw(Box::new(ScivexTensor { inner: r })),
        Err(e) => {
            set_error(&e.to_string());
            std::ptr::null_mut()
        }
    }
}

// ---------------------------------------------------------------------------
// Linear algebra
// ---------------------------------------------------------------------------

/// Solve a linear system Ax = b. Returns x or null on error.
///
/// # Safety
///
/// Both `a` and `b` must be valid tensor pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn scivex_tensor_solve(
    a: *const ScivexTensor,
    b: *const ScivexTensor,
) -> *mut ScivexTensor {
    match unsafe { (*a).inner.solve(&(*b).inner) } {
        Ok(x) => Box::into_raw(Box::new(ScivexTensor { inner: x })),
        Err(e) => {
            set_error(&e.to_string());
            std::ptr::null_mut()
        }
    }
}

/// Matrix inverse. Returns the inverse or null on error.
///
/// # Safety
///
/// `t` must be a valid tensor pointer to a square matrix.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn scivex_tensor_inv(t: *const ScivexTensor) -> *mut ScivexTensor {
    match unsafe { (*t).inner.inv() } {
        Ok(r) => Box::into_raw(Box::new(ScivexTensor { inner: r })),
        Err(e) => {
            set_error(&e.to_string());
            std::ptr::null_mut()
        }
    }
}

// ---------------------------------------------------------------------------
// Internal access for other FFI modules
// ---------------------------------------------------------------------------

impl ScivexTensor {
    pub(crate) fn inner(&self) -> &Tensor<f64> {
        &self.inner
    }

    pub(crate) fn from_inner(inner: Tensor<f64>) -> Self {
        Self { inner }
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp, clippy::borrow_as_ptr)]
mod tests {
    use super::*;

    #[test]
    fn test_ffi_tensor_roundtrip() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = [2usize, 3];
        let t = unsafe {
            scivex_tensor_from_array(data.as_ptr(), data.len(), shape.as_ptr(), shape.len())
        };
        assert!(!t.is_null());

        assert_eq!(unsafe { scivex_tensor_ndim(t) }, 2);
        assert_eq!(unsafe { scivex_tensor_numel(t) }, 6);

        let mut out_shape = [0usize; 2];
        let rc = unsafe { scivex_tensor_shape(t, out_shape.as_mut_ptr(), 2) };
        assert_eq!(rc, 0);
        assert_eq!(out_shape, [2, 3]);

        let mut out_data = [0.0f64; 6];
        let rc = unsafe { scivex_tensor_data(t, out_data.as_mut_ptr(), 6) };
        assert_eq!(rc, 0);
        assert_eq!(out_data, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        unsafe { scivex_tensor_free(t) };
    }

    #[test]
    fn test_ffi_tensor_arithmetic() {
        let data_a = [1.0, 2.0, 3.0];
        let data_b = [4.0, 5.0, 6.0];
        let shape = [3usize];

        let a = unsafe { scivex_tensor_from_array(data_a.as_ptr(), 3, shape.as_ptr(), 1) };
        let b = unsafe { scivex_tensor_from_array(data_b.as_ptr(), 3, shape.as_ptr(), 1) };

        // add
        let c = unsafe { scivex_tensor_add(a, b) };
        assert!(!c.is_null());
        let mut out = [0.0f64; 3];
        unsafe { scivex_tensor_data(c, out.as_mut_ptr(), 3) };
        assert_eq!(out, [5.0, 7.0, 9.0]);

        // sum
        let s = unsafe { scivex_tensor_sum(a) };
        assert!((s - 6.0).abs() < 1e-10);

        // mean
        let m = unsafe { scivex_tensor_mean(a) };
        assert!((m - 2.0).abs() < 1e-10);

        // scalar multiply
        let d = unsafe { scivex_tensor_mul_scalar(a, 10.0) };
        unsafe { scivex_tensor_data(d, out.as_mut_ptr(), 3) };
        assert_eq!(out, [10.0, 20.0, 30.0]);

        unsafe {
            scivex_tensor_free(a);
            scivex_tensor_free(b);
            scivex_tensor_free(c);
            scivex_tensor_free(d);
        }
    }

    #[test]
    fn test_ffi_tensor_matmul() {
        let a_data = [1.0, 2.0, 3.0, 4.0];
        let a_shape = [2usize, 2];
        let b_data = [5.0, 6.0, 7.0, 8.0];

        let a = unsafe { scivex_tensor_from_array(a_data.as_ptr(), 4, a_shape.as_ptr(), 2) };
        let b = unsafe { scivex_tensor_from_array(b_data.as_ptr(), 4, a_shape.as_ptr(), 2) };

        let c = unsafe { scivex_tensor_matmul(a, b) };
        assert!(!c.is_null());

        let mut out = [0.0f64; 4];
        unsafe { scivex_tensor_data(c, out.as_mut_ptr(), 4) };
        assert_eq!(out, [19.0, 22.0, 43.0, 50.0]);

        unsafe {
            scivex_tensor_free(a);
            scivex_tensor_free(b);
            scivex_tensor_free(c);
        }
    }

    #[test]
    fn test_ffi_tensor_null_safety() {
        let t = unsafe { scivex_tensor_from_array(std::ptr::null(), 0, std::ptr::null(), 0) };
        assert!(t.is_null());

        // scivex_tensor_free should handle null
        unsafe { scivex_tensor_free(std::ptr::null_mut()) };
    }

    #[test]
    fn test_ffi_tensor_shape_mismatch() {
        let data_a = [1.0, 2.0, 3.0];
        let data_b = [4.0, 5.0];
        let shape_3 = [3usize];
        let shape_2 = [2usize];

        let a = unsafe { scivex_tensor_from_array(data_a.as_ptr(), 3, shape_3.as_ptr(), 1) };
        let b = unsafe { scivex_tensor_from_array(data_b.as_ptr(), 2, shape_2.as_ptr(), 1) };

        let c = unsafe { scivex_tensor_add(a, b) };
        assert!(c.is_null());

        // Check error message
        let err = crate::error::scivex_last_error();
        assert!(!err.is_null());

        unsafe {
            scivex_tensor_free(a);
            scivex_tensor_free(b);
        }
    }
}
