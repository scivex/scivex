//! BLAS Level 1–3 operations on [`Tensor`].
//!
//! All functions operate on tensors and validate shapes, returning
//! [`Result`] on dimension mismatches.
//!
//! When the `blas-backend` feature is enabled, GEMM dispatches to the
//! system BLAS (Accelerate on macOS, OpenBLAS on Linux) for f64/f32.

use crate::error::{CoreError, Result};
use crate::tensor::Tensor;
use crate::{Float, Scalar};

// ======================================================================
// System BLAS FFI (behind blas-backend feature)
// ======================================================================

#[cfg(feature = "blas-backend")]
mod blas_ffi {
    use libc::c_int;

    // CBLAS row-major enum value
    pub const CBLAS_ROW_MAJOR: c_int = 101;
    pub const CBLAS_NO_TRANS: c_int = 111;

    unsafe extern "C" {
        pub fn cblas_dgemm(
            order: c_int,
            transa: c_int,
            transb: c_int,
            m: c_int,
            n: c_int,
            k: c_int,
            alpha: f64,
            a: *const f64,
            lda: c_int,
            b: *const f64,
            ldb: c_int,
            beta: f64,
            c: *mut f64,
            ldc: c_int,
        );

        pub fn cblas_sgemm(
            order: c_int,
            transa: c_int,
            transb: c_int,
            m: c_int,
            n: c_int,
            k: c_int,
            alpha: f32,
            a: *const f32,
            lda: c_int,
            b: *const f32,
            ldb: c_int,
            beta: f32,
            c: *mut f32,
            ldc: c_int,
        );

        pub fn cblas_dgemv(
            order: c_int,
            trans: c_int,
            m: c_int,
            n: c_int,
            alpha: f64,
            a: *const f64,
            lda: c_int,
            x: *const f64,
            incx: c_int,
            beta: f64,
            y: *mut f64,
            incy: c_int,
        );

        pub fn cblas_sgemv(
            order: c_int,
            trans: c_int,
            m: c_int,
            n: c_int,
            alpha: f32,
            a: *const f32,
            lda: c_int,
            x: *const f32,
            incx: c_int,
            beta: f32,
            y: *mut f32,
            incy: c_int,
        );
    }
}

// ======================================================================
// BLAS Level 1 — vector operations, O(n)
// ======================================================================

/// Inner (dot) product of two 1-D tensors: `sum(x_i * y_i)`.
///
/// Both tensors must be 1-D with the same length.
///
/// ```
/// # use scivex_core::tensor::Tensor;
/// # use scivex_core::linalg::dot;
/// let x = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0], vec![3]).unwrap();
/// let y = Tensor::from_vec(vec![4.0_f64, 5.0, 6.0], vec![3]).unwrap();
/// let d = dot(&x, &y).unwrap();
/// assert!((d - 32.0).abs() < 1e-10);
/// ```
pub fn dot<T: Scalar>(x: &Tensor<T>, y: &Tensor<T>) -> Result<T> {
    check_vectors(x, y, "dot")?;
    Ok(dot_slice(x.as_slice(), y.as_slice()))
}

/// Inner dot product on raw slices, dispatching to SIMD when available.
fn dot_slice<T: Scalar>(a: &[T], b: &[T]) -> T {
    #[cfg(feature = "simd")]
    {
        use crate::simd;
        use std::any::TypeId;
        if TypeId::of::<T>() == TypeId::of::<f64>() {
            // SAFETY: T is f64 confirmed by TypeId.
            let result =
                unsafe { simd::f64_ops::dot_f64(simd::slice_as_f64(a), simd::slice_as_f64(b)) };
            return unsafe { simd::f64_to_t(result) };
        }
        if TypeId::of::<T>() == TypeId::of::<f32>() {
            // SAFETY: T is f32 confirmed by TypeId.
            let result =
                unsafe { simd::f32_ops::dot_f32(simd::slice_as_f32(a), simd::slice_as_f32(b)) };
            return unsafe { simd::f32_to_t(result) };
        }
    }
    a.iter()
        .zip(b.iter())
        .fold(T::zero(), |acc, (&x, &y)| acc + x * y)
}

/// `y = alpha * x + y` (in-place update of `y`).
///
/// Both tensors must be 1-D with the same length.
///
/// ```
/// # use scivex_core::tensor::Tensor;
/// # use scivex_core::linalg::axpy;
/// let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
/// let mut y = Tensor::from_vec(vec![10.0, 20.0, 30.0], vec![3]).unwrap();
/// axpy(2.0, &x, &mut y).unwrap();
/// assert_eq!(y.as_slice(), &[12.0, 24.0, 36.0]);
/// ```
pub fn axpy<T: Scalar>(alpha: T, x: &Tensor<T>, y: &mut Tensor<T>) -> Result<()> {
    check_vectors(x, y, "axpy")?;
    axpy_slice(alpha, x.as_slice(), y.as_mut_slice());
    Ok(())
}

/// In-place axpy on raw slices, dispatching to SIMD when available.
pub(crate) fn axpy_slice<T: Scalar>(alpha: T, x: &[T], y: &mut [T]) {
    #[cfg(feature = "simd")]
    {
        use crate::simd;
        use std::any::TypeId;
        if TypeId::of::<T>() == TypeId::of::<f64>() {
            // SAFETY: T is f64 confirmed by TypeId.
            unsafe {
                simd::f64_ops::axpy_f64(
                    simd::t_to_f64(alpha),
                    simd::slice_as_f64(x),
                    simd::slice_as_f64_mut(y),
                );
            }
            return;
        }
        if TypeId::of::<T>() == TypeId::of::<f32>() {
            // SAFETY: T is f32 confirmed by TypeId.
            unsafe {
                simd::f32_ops::axpy_f32(
                    simd::t_to_f32(alpha),
                    simd::slice_as_f32(x),
                    simd::slice_as_f32_mut(y),
                );
            }
            return;
        }
    }
    for (yi, &xi) in y.iter_mut().zip(x.iter()) {
        *yi += alpha * xi;
    }
}

/// Euclidean norm (L2 norm) of a 1-D tensor: `sqrt(sum(x_i^2))`.
///
/// ```
/// # use scivex_core::tensor::Tensor;
/// # use scivex_core::linalg::nrm2;
/// let x = Tensor::from_vec(vec![3.0_f64, 4.0], vec![2]).unwrap();
/// let n = nrm2(&x).unwrap();
/// assert!((n - 5.0).abs() < 1e-10);
/// ```
pub fn nrm2<T: Float>(x: &Tensor<T>) -> Result<T> {
    check_vector(x, "nrm2")?;
    #[cfg(feature = "simd")]
    {
        use crate::simd;
        use std::any::TypeId;
        if TypeId::of::<T>() == TypeId::of::<f64>() {
            // SAFETY: T is f64 confirmed by TypeId.
            let result =
                unsafe { simd::f64_ops::sum_sq_f64(simd::slice_as_f64(x.as_slice())).sqrt() };
            return Ok(unsafe { simd::f64_to_t(result) });
        }
        if TypeId::of::<T>() == TypeId::of::<f32>() {
            // SAFETY: T is f32 confirmed by TypeId.
            let result =
                unsafe { simd::f32_ops::sum_sq_f32(simd::slice_as_f32(x.as_slice())).sqrt() };
            return Ok(unsafe { simd::f32_to_t(result) });
        }
    }
    let sum_sq = x.as_slice().iter().fold(T::zero(), |acc, &v| acc + v * v);
    Ok(sum_sq.sqrt())
}

/// Sum of absolute values (L1 norm) of a 1-D tensor: `sum(|x_i|)`.
///
/// ```
/// # use scivex_core::tensor::Tensor;
/// # use scivex_core::linalg::asum;
/// let x = Tensor::from_vec(vec![-1.0_f64, 2.0, -3.0], vec![3]).unwrap();
/// let s = asum(&x).unwrap();
/// assert!((s - 6.0).abs() < 1e-10);
/// ```
pub fn asum<T: Float>(x: &Tensor<T>) -> Result<T> {
    check_vector(x, "asum")?;
    #[cfg(feature = "simd")]
    {
        use crate::simd;
        use std::any::TypeId;
        if TypeId::of::<T>() == TypeId::of::<f64>() {
            // SAFETY: T is f64 confirmed by TypeId.
            let result = unsafe { simd::f64_ops::asum_f64(simd::slice_as_f64(x.as_slice())) };
            return Ok(unsafe { simd::f64_to_t(result) });
        }
        if TypeId::of::<T>() == TypeId::of::<f32>() {
            // SAFETY: T is f32 confirmed by TypeId.
            let result = unsafe { simd::f32_ops::asum_f32(simd::slice_as_f32(x.as_slice())) };
            return Ok(unsafe { simd::f32_to_t(result) });
        }
    }
    let result = x.as_slice().iter().fold(T::zero(), |acc, &v| acc + v.abs());
    Ok(result)
}

/// Scale a vector in place: `x = alpha * x`.
///
/// ```
/// # use scivex_core::tensor::Tensor;
/// # use scivex_core::linalg::scal;
/// let mut x = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
/// scal(10.0, &mut x).unwrap();
/// assert_eq!(x.as_slice(), &[10.0, 20.0, 30.0]);
/// ```
pub fn scal<T: Scalar>(alpha: T, x: &mut Tensor<T>) -> Result<()> {
    check_vector(x, "scal")?;
    #[cfg(feature = "simd")]
    {
        use crate::simd;
        use std::any::TypeId;
        if TypeId::of::<T>() == TypeId::of::<f64>() {
            // SAFETY: T is f64 confirmed by TypeId.
            unsafe {
                simd::f64_ops::scal_f64(
                    simd::t_to_f64(alpha),
                    simd::slice_as_f64_mut(x.as_mut_slice()),
                );
            }
            return Ok(());
        }
        if TypeId::of::<T>() == TypeId::of::<f32>() {
            // SAFETY: T is f32 confirmed by TypeId.
            unsafe {
                simd::f32_ops::scal_f32(
                    simd::t_to_f32(alpha),
                    simd::slice_as_f32_mut(x.as_mut_slice()),
                );
            }
            return Ok(());
        }
    }
    for v in x.as_mut_slice() {
        *v *= alpha;
    }
    Ok(())
}

/// Index of the element with the largest absolute value.
///
/// Returns `None` for empty tensors.
///
/// ```
/// # use scivex_core::tensor::Tensor;
/// # use scivex_core::linalg::iamax;
/// let x = Tensor::from_vec(vec![1.0_f64, -5.0, 3.0], vec![3]).unwrap();
/// assert_eq!(iamax(&x).unwrap(), Some(1));
/// ```
pub fn iamax<T: Float>(x: &Tensor<T>) -> Result<Option<usize>> {
    check_vector(x, "iamax")?;
    if x.is_empty() {
        return Ok(None);
    }
    let mut max_idx = 0;
    let mut max_val = x.as_slice()[0].abs();
    for (i, &v) in x.as_slice().iter().enumerate().skip(1) {
        let av = v.abs();
        if av > max_val {
            max_val = av;
            max_idx = i;
        }
    }
    Ok(Some(max_idx))
}

// ======================================================================
// BLAS Level 2 — matrix-vector operations, O(n^2)
// ======================================================================

/// General matrix-vector multiply: `y = alpha * A * x + beta * y`.
///
/// - `a` must be 2-D with shape `[m, n]`.
/// - `x` must be 1-D with length `n`.
/// - `y` must be 1-D with length `m`.
///
/// If `beta` is zero, `y` is overwritten (not read).
///
/// ```
/// # use scivex_core::tensor::Tensor;
/// # use scivex_core::linalg::gemv;
/// // A = [[1, 2], [3, 4]], x = [5, 6]
/// let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
/// let x = Tensor::from_vec(vec![5.0, 6.0], vec![2]).unwrap();
/// let mut y = Tensor::<f64>::zeros(vec![2]);
/// gemv(1.0, &a, &x, 0.0, &mut y).unwrap();
/// assert_eq!(y.as_slice(), &[17.0, 39.0]);
/// ```
#[allow(clippy::many_single_char_names, clippy::cast_possible_wrap)]
pub fn gemv<T: Scalar>(
    alpha: T,
    a: &Tensor<T>,
    x: &Tensor<T>,
    beta: T,
    y: &mut Tensor<T>,
) -> Result<()> {
    if a.ndim() != 2 {
        return Err(CoreError::InvalidArgument {
            reason: "gemv: `a` must be a 2-D tensor (matrix)",
        });
    }
    if x.ndim() != 1 {
        return Err(CoreError::InvalidArgument {
            reason: "gemv: `x` must be a 1-D tensor (vector)",
        });
    }
    if y.ndim() != 1 {
        return Err(CoreError::InvalidArgument {
            reason: "gemv: `y` must be a 1-D tensor (vector)",
        });
    }

    let m = a.shape()[0];
    let n = a.shape()[1];

    if x.numel() != n {
        return Err(CoreError::DimensionMismatch {
            expected: vec![n],
            got: x.shape().to_vec(),
        });
    }
    if y.numel() != m {
        return Err(CoreError::DimensionMismatch {
            expected: vec![m],
            got: y.shape().to_vec(),
        });
    }

    // Dispatch to system BLAS when available.
    #[cfg(feature = "blas-backend")]
    {
        use std::any::TypeId;
        if TypeId::of::<T>() == TypeId::of::<f64>() {
            unsafe {
                let alpha_f64: f64 = core::mem::transmute_copy(&alpha);
                let beta_f64: f64 = core::mem::transmute_copy(&beta);
                blas_ffi::cblas_dgemv(
                    blas_ffi::CBLAS_ROW_MAJOR,
                    blas_ffi::CBLAS_NO_TRANS,
                    m as libc::c_int,
                    n as libc::c_int,
                    alpha_f64,
                    a.as_slice().as_ptr().cast::<f64>(),
                    n as libc::c_int,
                    x.as_slice().as_ptr().cast::<f64>(),
                    1,
                    beta_f64,
                    y.as_mut_slice().as_mut_ptr().cast::<f64>(),
                    1,
                );
            }
            return Ok(());
        }
        if TypeId::of::<T>() == TypeId::of::<f32>() {
            unsafe {
                let alpha_f32: f32 = core::mem::transmute_copy(&alpha);
                let beta_f32: f32 = core::mem::transmute_copy(&beta);
                blas_ffi::cblas_sgemv(
                    blas_ffi::CBLAS_ROW_MAJOR,
                    blas_ffi::CBLAS_NO_TRANS,
                    m as libc::c_int,
                    n as libc::c_int,
                    alpha_f32,
                    a.as_slice().as_ptr().cast::<f32>(),
                    n as libc::c_int,
                    x.as_slice().as_ptr().cast::<f32>(),
                    1,
                    beta_f32,
                    y.as_mut_slice().as_mut_ptr().cast::<f32>(),
                    1,
                );
            }
            return Ok(());
        }
    }

    let a_data = a.as_slice();
    let x_data = x.as_slice();
    let y_data = y.as_mut_slice();

    for (i, yi) in y_data.iter_mut().enumerate().take(m) {
        let row_offset = i * n;
        let row = &a_data[row_offset..row_offset + n];
        let sum = dot_slice(row, x_data);
        *yi = alpha * sum + beta * *yi;
    }

    Ok(())
}

// ======================================================================
// BLAS Level 3 — matrix-matrix operations, O(n^3)
// ======================================================================

/// General matrix-matrix multiply: `C = alpha * A * B + beta * C`.
///
/// - `a` must be 2-D with shape `[m, k]`.
/// - `b` must be 2-D with shape `[k, n]`.
/// - `c` must be 2-D with shape `[m, n]`.
///
/// If `beta` is zero, `c` is overwritten (not read).
///
/// ```
/// # use scivex_core::tensor::Tensor;
/// # use scivex_core::linalg::gemm;
/// let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
/// let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
/// let mut c = Tensor::<f64>::zeros(vec![2, 2]);
/// gemm(1.0, &a, &b, 0.0, &mut c).unwrap();
/// assert_eq!(c.as_slice(), &[19.0, 22.0, 43.0, 50.0]);
/// ```
#[allow(
    clippy::many_single_char_names,
    clippy::too_many_lines,
    clippy::cast_possible_wrap
)]
pub fn gemm<T: Scalar>(
    alpha: T,
    a: &Tensor<T>,
    b: &Tensor<T>,
    beta: T,
    c: &mut Tensor<T>,
) -> Result<()> {
    // Tile sizes for blocked GEMM — keep working sets in L1/L2 cache.
    const MC: usize = 64; // row block size for A/C
    const KC: usize = 256; // reduction-dimension block size
    const NC: usize = 256; // column block size for B/C

    if a.ndim() != 2 || b.ndim() != 2 || c.ndim() != 2 {
        return Err(CoreError::InvalidArgument {
            reason: "gemm: all arguments must be 2-D tensors (matrices)",
        });
    }

    let m = a.shape()[0];
    let k = a.shape()[1];
    let n = b.shape()[1];

    if b.shape()[0] != k {
        return Err(CoreError::DimensionMismatch {
            expected: vec![k, n],
            got: b.shape().to_vec(),
        });
    }
    if c.shape()[0] != m || c.shape()[1] != n {
        return Err(CoreError::DimensionMismatch {
            expected: vec![m, n],
            got: c.shape().to_vec(),
        });
    }

    // Dispatch to system BLAS when available (Accelerate/MKL/OpenBLAS).
    #[cfg(feature = "blas-backend")]
    {
        use std::any::TypeId;
        if TypeId::of::<T>() == TypeId::of::<f64>() {
            // SAFETY: T is f64 confirmed by TypeId; transmute alpha/beta, cast pointers.
            unsafe {
                let alpha_f64: f64 = core::mem::transmute_copy(&alpha);
                let beta_f64: f64 = core::mem::transmute_copy(&beta);
                blas_ffi::cblas_dgemm(
                    blas_ffi::CBLAS_ROW_MAJOR,
                    blas_ffi::CBLAS_NO_TRANS,
                    blas_ffi::CBLAS_NO_TRANS,
                    m as libc::c_int,
                    n as libc::c_int,
                    k as libc::c_int,
                    alpha_f64,
                    a.as_slice().as_ptr().cast::<f64>(),
                    k as libc::c_int,
                    b.as_slice().as_ptr().cast::<f64>(),
                    n as libc::c_int,
                    beta_f64,
                    c.as_mut_slice().as_mut_ptr().cast::<f64>(),
                    n as libc::c_int,
                );
            }
            return Ok(());
        }
        if TypeId::of::<T>() == TypeId::of::<f32>() {
            unsafe {
                let alpha_f32: f32 = core::mem::transmute_copy(&alpha);
                let beta_f32: f32 = core::mem::transmute_copy(&beta);
                blas_ffi::cblas_sgemm(
                    blas_ffi::CBLAS_ROW_MAJOR,
                    blas_ffi::CBLAS_NO_TRANS,
                    blas_ffi::CBLAS_NO_TRANS,
                    m as libc::c_int,
                    n as libc::c_int,
                    k as libc::c_int,
                    alpha_f32,
                    a.as_slice().as_ptr().cast::<f32>(),
                    k as libc::c_int,
                    b.as_slice().as_ptr().cast::<f32>(),
                    n as libc::c_int,
                    beta_f32,
                    c.as_mut_slice().as_mut_ptr().cast::<f32>(),
                    n as libc::c_int,
                );
            }
            return Ok(());
        }
        // Non-float types fall through to the manual implementation.
    }

    let a_data = a.as_slice();
    let b_data = b.as_slice();
    let c_data = c.as_mut_slice();

    // Scale C by beta first (or zero it).
    if beta == T::zero() {
        for v in c_data.iter_mut() {
            *v = T::zero();
        }
    } else if beta != T::one() {
        for v in c_data.iter_mut() {
            *v *= beta;
        }
    }

    // Blocked GEMM with cache-aware tiling and panel packing.
    // Packing copies A and B panels into contiguous buffers so the
    // micro-kernel accesses sequential memory, eliminating TLB misses.

    // Packing buffers (allocated once, reused across tiles).
    #[cfg(all(target_arch = "aarch64", feature = "simd"))]
    let (is_f64, is_f32) = {
        use std::any::TypeId;
        (
            TypeId::of::<T>() == TypeId::of::<f64>(),
            TypeId::of::<T>() == TypeId::of::<f32>(),
        )
    };
    #[cfg(all(target_arch = "aarch64", feature = "simd"))]
    let mut pack_b_f64: Vec<f64> = if is_f64 {
        vec![0.0; KC * NC]
    } else {
        Vec::new()
    };
    #[cfg(all(target_arch = "aarch64", feature = "simd"))]
    let mut pack_a_f64: Vec<f64> = if is_f64 {
        vec![0.0; MC * KC]
    } else {
        Vec::new()
    };
    #[cfg(all(target_arch = "aarch64", feature = "simd"))]
    let mut pack_b_f32: Vec<f32> = if is_f32 {
        vec![0.0; KC * NC]
    } else {
        Vec::new()
    };
    #[cfg(all(target_arch = "aarch64", feature = "simd"))]
    let mut pack_a_f32: Vec<f32> = if is_f32 {
        vec![0.0; MC * KC]
    } else {
        Vec::new()
    };

    // Loop over K-dimension blocks
    for pk in (0..k).step_by(KC) {
        let kb = KC.min(k - pk);

        // Loop over column blocks of B / C (pack B once per pk×pj)
        for pj in (0..n).step_by(NC) {
            let nb = NC.min(n - pj);

            // Pack B panel: B[pk..pk+kb, pj..pj+nb] → contiguous buffer
            #[cfg(all(target_arch = "aarch64", feature = "simd"))]
            if is_f64 {
                unsafe {
                    let b_f64 = b_data.as_ptr().cast::<f64>();
                    for p in 0..kb {
                        let src_row = b_f64.add((pk + p) * n + pj);
                        let dst_row = pack_b_f64.as_mut_ptr().add(p * nb);
                        core::ptr::copy_nonoverlapping(src_row, dst_row, nb);
                    }
                }
            }
            #[cfg(all(target_arch = "aarch64", feature = "simd"))]
            if is_f32 {
                unsafe {
                    let b_f32 = b_data.as_ptr().cast::<f32>();
                    for p in 0..kb {
                        let src_row = b_f32.add((pk + p) * n + pj);
                        let dst_row = pack_b_f32.as_mut_ptr().add(p * nb);
                        core::ptr::copy_nonoverlapping(src_row, dst_row, nb);
                    }
                }
            }

            // Loop over row blocks of A / C
            for pi in (0..m).step_by(MC) {
                let mb = MC.min(m - pi);

                #[cfg(all(target_arch = "aarch64", feature = "simd"))]
                if is_f64 {
                    unsafe {
                        let a_f64 = a_data.as_ptr().cast::<f64>();
                        let c_f64 = c_data.as_mut_ptr().cast::<f64>();
                        let alpha_f64 = crate::simd::t_to_f64(alpha);

                        // Pack A panel: A[pi..pi+mb, pk..pk+kb] → contiguous
                        for i in 0..mb {
                            let src_row = a_f64.add((pi + i) * k + pk);
                            let dst_row = pack_a_f64.as_mut_ptr().add(i * kb);
                            core::ptr::copy_nonoverlapping(src_row, dst_row, kb);
                        }

                        let pa = pack_a_f64.as_ptr();
                        let pb = pack_b_f64.as_ptr();
                        let j4 = nb / 4 * 4;

                        // Process 8-row blocks with 8x4 micro-kernel
                        let i8 = mb / 8 * 8;
                        for i in (0..i8).step_by(8) {
                            for j in (0..j4).step_by(4) {
                                crate::simd::neon_f64_ops::gemm_8x4_f64_neon(
                                    pa.add(i * kb),
                                    pb.add(j),
                                    c_f64.add((pi + i) * n + (pj + j)),
                                    alpha_f64,
                                    kb,
                                    kb, // packed A row stride = kb
                                    nb, // packed B row stride = nb
                                    n,  // C row stride = n
                                );
                            }
                            // Remainder columns
                            if j4 < nb {
                                for ii in 0..8 {
                                    let row_c = (pi + i + ii) * n + pj + j4;
                                    for p in 0..kb {
                                        let scale_f64 = alpha_f64 * *pa.add((i + ii) * kb + p);
                                        for jj in 0..(nb - j4) {
                                            *c_f64.add(row_c + jj) +=
                                                scale_f64 * *pb.add(p * nb + j4 + jj);
                                        }
                                    }
                                }
                            }
                        }
                        // Remaining 4-row block with 4x4 micro-kernel
                        let i4_start = i8;
                        let i4_end = i4_start + (mb - i8) / 4 * 4;
                        for i in (i4_start..i4_end).step_by(4) {
                            for j in (0..j4).step_by(4) {
                                crate::simd::neon_f64_ops::gemm_4x4_f64_neon(
                                    pa.add(i * kb),
                                    pb.add(j),
                                    c_f64.add((pi + i) * n + (pj + j)),
                                    alpha_f64,
                                    kb,
                                    kb,
                                    nb,
                                    n,
                                );
                            }
                            if j4 < nb {
                                for ii in 0..4 {
                                    let row_c = (pi + i + ii) * n + pj + j4;
                                    for p in 0..kb {
                                        let scale_f64 = alpha_f64 * *pa.add((i + ii) * kb + p);
                                        for jj in 0..(nb - j4) {
                                            *c_f64.add(row_c + jj) +=
                                                scale_f64 * *pb.add(p * nb + j4 + jj);
                                        }
                                    }
                                }
                            }
                        }
                        // Scalar remainder rows
                        for i in i4_end..mb {
                            let row_c = (pi + i) * n + pj;
                            for p in 0..kb {
                                let scale = alpha * a_data[(pi + i) * k + pk + p];
                                let b_off = (pk + p) * n + pj;
                                let b_row = &b_data[b_off..b_off + nb];
                                let c_slice = &mut c_data[row_c..row_c + nb];
                                axpy_slice(scale, b_row, c_slice);
                            }
                        }
                    }
                    continue;
                }

                // NEON f32 micro-kernel path with panel packing.
                #[cfg(all(target_arch = "aarch64", feature = "simd"))]
                if is_f32 {
                    unsafe {
                        let a_f32 = a_data.as_ptr().cast::<f32>();
                        let c_f32 = c_data.as_mut_ptr().cast::<f32>();
                        let alpha_f32 = crate::simd::t_to_f32(alpha);

                        // Pack A panel
                        for i in 0..mb {
                            let src_row = a_f32.add((pi + i) * k + pk);
                            let dst_row = pack_a_f32.as_mut_ptr().add(i * kb);
                            core::ptr::copy_nonoverlapping(src_row, dst_row, kb);
                        }

                        let pa = pack_a_f32.as_ptr();
                        let pb = pack_b_f32.as_ptr();
                        let j4 = nb / 4 * 4;

                        // Process 8-row blocks with 8x4 micro-kernel
                        let i8 = mb / 8 * 8;
                        for i in (0..i8).step_by(8) {
                            for j in (0..j4).step_by(4) {
                                crate::simd::neon_f32_ops::gemm_8x4_f32_neon(
                                    pa.add(i * kb),
                                    pb.add(j),
                                    c_f32.add((pi + i) * n + (pj + j)),
                                    alpha_f32,
                                    kb,
                                    kb,
                                    nb,
                                    n,
                                );
                            }
                            if j4 < nb {
                                for ii in 0..8 {
                                    let row_c = (pi + i + ii) * n + pj + j4;
                                    for p in 0..kb {
                                        let scale_f32 = alpha_f32 * *pa.add((i + ii) * kb + p);
                                        for jj in 0..(nb - j4) {
                                            *c_f32.add(row_c + jj) +=
                                                scale_f32 * *pb.add(p * nb + j4 + jj);
                                        }
                                    }
                                }
                            }
                        }
                        // Remaining 4-row block with 4x4 micro-kernel
                        let i4_start = i8;
                        let i4_end = i4_start + (mb - i8) / 4 * 4;
                        for i in (i4_start..i4_end).step_by(4) {
                            for j in (0..j4).step_by(4) {
                                crate::simd::neon_f32_ops::gemm_4x4_f32_neon(
                                    pa.add(i * kb),
                                    pb.add(j),
                                    c_f32.add((pi + i) * n + (pj + j)),
                                    alpha_f32,
                                    kb,
                                    kb,
                                    nb,
                                    n,
                                );
                            }
                            if j4 < nb {
                                for ii in 0..4 {
                                    let row_c = (pi + i + ii) * n + pj + j4;
                                    for p in 0..kb {
                                        let scale_f32 = alpha_f32 * *pa.add((i + ii) * kb + p);
                                        for jj in 0..(nb - j4) {
                                            *c_f32.add(row_c + jj) +=
                                                scale_f32 * *pb.add(p * nb + j4 + jj);
                                        }
                                    }
                                }
                            }
                        }
                        // Scalar remainder rows
                        for i in i4_end..mb {
                            let row_c = (pi + i) * n + pj;
                            for p in 0..kb {
                                let scale = alpha * a_data[(pi + i) * k + pk + p];
                                let b_off = (pk + p) * n + pj;
                                let b_row = &b_data[b_off..b_off + nb];
                                let c_slice = &mut c_data[row_c..row_c + nb];
                                axpy_slice(scale, b_row, c_slice);
                            }
                        }
                    }
                    continue;
                }

                // Generic fallback (non-f64/f32 or non-aarch64)
                for i in 0..mb {
                    let row_a = (pi + i) * k + pk;
                    let row_c = (pi + i) * n + pj;
                    for p in 0..kb {
                        let scale = alpha * a_data[row_a + p];
                        let b_off = (pk + p) * n + pj;
                        let b_row = &b_data[b_off..b_off + nb];
                        let c_slice = &mut c_data[row_c..row_c + nb];
                        axpy_slice(scale, b_row, c_slice);
                    }
                }
            }
        }
    }

    Ok(())
}

// ======================================================================
// Convenience methods on Tensor
// ======================================================================

impl<T: Scalar> Tensor<T> {
    /// Matrix-vector multiply: returns `A @ x` as a new 1-D tensor.
    ///
    /// `self` must be 2-D `[m, n]`, `x` must be 1-D `[n]`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_core::tensor::Tensor;
    /// let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    /// let x = Tensor::from_vec(vec![5.0, 6.0], vec![2]).unwrap();
    /// let y = a.matvec(&x).unwrap();
    /// assert_eq!(y.as_slice(), &[17.0, 39.0]);
    /// ```
    pub fn matvec(&self, x: &Tensor<T>) -> Result<Tensor<T>> {
        let m = self.shape().first().copied().unwrap_or(0);
        let mut y = Tensor::zeros(vec![m]);
        gemv(T::one(), self, x, T::zero(), &mut y)?;
        Ok(y)
    }

    /// Matrix-matrix multiply: returns `self @ other` as a new 2-D tensor.
    ///
    /// `self` must be 2-D `[m, k]`, `other` must be 2-D `[k, n]`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_core::tensor::Tensor;
    /// let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    /// let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
    /// let c = a.matmul(&b).unwrap();
    /// assert_eq!(c.as_slice(), &[19.0, 22.0, 43.0, 50.0]);
    /// ```
    pub fn matmul(&self, other: &Tensor<T>) -> Result<Tensor<T>> {
        let m = self.shape().first().copied().unwrap_or(0);
        let n = other.shape().get(1).copied().unwrap_or(0);
        let mut c = Tensor::zeros(vec![m, n]);
        gemm(T::one(), self, other, T::zero(), &mut c)?;
        Ok(c)
    }

    /// Dot product with another 1-D tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_core::tensor::Tensor;
    /// let x = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0], vec![3]).unwrap();
    /// let y = Tensor::from_vec(vec![4.0, 5.0, 6.0], vec![3]).unwrap();
    /// assert_eq!(x.dot(&y).unwrap(), 32.0);
    /// ```
    pub fn dot(&self, other: &Tensor<T>) -> Result<T> {
        dot(self, other)
    }
}

impl<T: Float> Tensor<T> {
    /// Euclidean (L2) norm of a 1-D tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_core::tensor::Tensor;
    /// let x = Tensor::from_vec(vec![3.0_f64, 4.0], vec![2]).unwrap();
    /// assert!((x.norm().unwrap() - 5.0).abs() < 1e-10);
    /// ```
    pub fn norm(&self) -> Result<T> {
        nrm2(self)
    }

    /// Solve the linear system `self * x = b` for a square matrix `self`.
    ///
    /// Uses LU decomposition with partial pivoting.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_core::tensor::Tensor;
    /// let a = Tensor::from_vec(vec![2.0_f64, 1.0, 1.0, 4.0], vec![2, 2]).unwrap();
    /// let b = Tensor::from_vec(vec![5.0_f64, 6.0], vec![2]).unwrap();
    /// let x = a.solve(&b).unwrap();
    /// assert!((x.as_slice()[0] - 2.0).abs() < 1e-10);
    /// assert!((x.as_slice()[1] - 1.0).abs() < 1e-10);
    /// ```
    pub fn solve(&self, b: &Tensor<T>) -> Result<Tensor<T>> {
        crate::linalg::solve(self, b)
    }

    /// Compute the inverse of a square matrix.
    ///
    /// Uses LU decomposition with partial pivoting.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_core::tensor::Tensor;
    /// let a = Tensor::from_vec(vec![2.0_f64, 1.0, 1.0, 4.0], vec![2, 2]).unwrap();
    /// let inv = a.inv().unwrap();
    /// let eye = a.matmul(&inv).unwrap();
    /// assert!((eye.as_slice()[0] - 1.0).abs() < 1e-10);
    /// ```
    pub fn inv(&self) -> Result<Tensor<T>> {
        crate::linalg::inv(self)
    }

    /// Compute the determinant of a square matrix.
    ///
    /// Uses LU decomposition with partial pivoting.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_core::tensor::Tensor;
    /// let a = Tensor::from_vec(vec![2.0_f64, 1.0, 1.0, 4.0], vec![2, 2]).unwrap();
    /// assert!((a.det().unwrap() - 7.0).abs() < 1e-10);
    /// ```
    pub fn det(&self) -> Result<T> {
        crate::linalg::det(self)
    }

    /// Solve the least-squares problem `min ||self * x - b||_2`.
    ///
    /// Uses QR decomposition with Householder reflections.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_core::tensor::Tensor;
    /// // Overdetermined system: 2x = [2, 4, 6] => x ≈ [1, 2, 3] / something
    /// let a = Tensor::from_vec(vec![1.0_f64, 1.0, 1.0], vec![3, 1]).unwrap();
    /// let b = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0], vec![3]).unwrap();
    /// let x = a.lstsq(&b).unwrap();
    /// assert_eq!(x.shape(), &[1]);
    /// ```
    pub fn lstsq(&self, b: &Tensor<T>) -> Result<Tensor<T>> {
        crate::linalg::lstsq(self, b)
    }
}

// ======================================================================
// Internal helpers
// ======================================================================

fn check_vector<T: Scalar>(x: &Tensor<T>, name: &'static str) -> Result<()> {
    if x.ndim() != 1 {
        return Err(CoreError::InvalidArgument {
            reason: match name {
                "nrm2" => "nrm2: expected a 1-D tensor",
                "asum" => "asum: expected a 1-D tensor",
                "scal" => "scal: expected a 1-D tensor",
                "iamax" => "iamax: expected a 1-D tensor",
                _ => "expected a 1-D tensor",
            },
        });
    }
    Ok(())
}

fn check_vectors<T: Scalar>(x: &Tensor<T>, y: &Tensor<T>, name: &'static str) -> Result<()> {
    if x.ndim() != 1 || y.ndim() != 1 {
        return Err(CoreError::InvalidArgument {
            reason: match name {
                "dot" => "dot: both arguments must be 1-D tensors",
                "axpy" => "axpy: both arguments must be 1-D tensors",
                _ => "both arguments must be 1-D tensors",
            },
        });
    }
    if x.numel() != y.numel() {
        return Err(CoreError::DimensionMismatch {
            expected: x.shape().to_vec(),
            got: y.shape().to_vec(),
        });
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    fn vec_f64(data: &[f64]) -> Tensor<f64> {
        Tensor::from_vec(data.to_vec(), vec![data.len()]).unwrap()
    }

    fn mat_f64(data: &[f64], rows: usize, cols: usize) -> Tensor<f64> {
        Tensor::from_vec(data.to_vec(), vec![rows, cols]).unwrap()
    }

    // ------------------------------------------------------------------
    // BLAS L1
    // ------------------------------------------------------------------

    #[test]
    fn test_dot_basic() {
        let x = vec_f64(&[1.0, 2.0, 3.0]);
        let y = vec_f64(&[4.0, 5.0, 6.0]);
        assert_eq!(dot(&x, &y).unwrap(), 32.0);
    }

    #[test]
    fn test_dot_single() {
        let x = vec_f64(&[3.0]);
        let y = vec_f64(&[7.0]);
        assert_eq!(dot(&x, &y).unwrap(), 21.0);
    }

    #[test]
    fn test_dot_length_mismatch() {
        let x = vec_f64(&[1.0, 2.0]);
        let y = vec_f64(&[1.0, 2.0, 3.0]);
        assert!(dot(&x, &y).is_err());
    }

    #[test]
    fn test_dot_not_1d() {
        let x = mat_f64(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let y = vec_f64(&[1.0, 2.0]);
        assert!(dot(&x, &y).is_err());
    }

    #[test]
    fn test_axpy() {
        let x = vec_f64(&[1.0, 2.0, 3.0]);
        let mut y = vec_f64(&[10.0, 20.0, 30.0]);
        axpy(2.0, &x, &mut y).unwrap();
        assert_eq!(y.as_slice(), &[12.0, 24.0, 36.0]);
    }

    #[test]
    fn test_axpy_zero_alpha() {
        let x = vec_f64(&[1.0, 2.0, 3.0]);
        let mut y = vec_f64(&[10.0, 20.0, 30.0]);
        axpy(0.0, &x, &mut y).unwrap();
        assert_eq!(y.as_slice(), &[10.0, 20.0, 30.0]);
    }

    #[test]
    fn test_nrm2() {
        let x = vec_f64(&[3.0, 4.0]);
        assert!((nrm2(&x).unwrap() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_nrm2_single() {
        let x = vec_f64(&[-7.0]);
        assert!((nrm2(&x).unwrap() - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_asum() {
        let x = vec_f64(&[-1.0, 2.0, -3.0, 4.0]);
        assert!((asum(&x).unwrap() - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_scal() {
        let mut x = vec_f64(&[1.0, 2.0, 3.0]);
        scal(10.0, &mut x).unwrap();
        assert_eq!(x.as_slice(), &[10.0, 20.0, 30.0]);
    }

    #[test]
    fn test_scal_zero() {
        let mut x = vec_f64(&[1.0, 2.0, 3.0]);
        scal(0.0, &mut x).unwrap();
        assert_eq!(x.as_slice(), &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_iamax() {
        let x = vec_f64(&[1.0, -5.0, 3.0, -2.0]);
        assert_eq!(iamax(&x).unwrap(), Some(1));
    }

    #[test]
    fn test_iamax_first_is_max() {
        let x = vec_f64(&[100.0, 1.0, 2.0]);
        assert_eq!(iamax(&x).unwrap(), Some(0));
    }

    #[test]
    fn test_iamax_empty() {
        let x = Tensor::<f64>::zeros(vec![0]);
        assert_eq!(iamax(&x).unwrap(), None);
    }

    // ------------------------------------------------------------------
    // BLAS L2
    // ------------------------------------------------------------------

    #[test]
    fn test_gemv_basic() {
        // A = [[1, 2], [3, 4]], x = [5, 6]
        // y = A @ x = [17, 39]
        let a = mat_f64(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let x = vec_f64(&[5.0, 6.0]);
        let mut y = Tensor::<f64>::zeros(vec![2]);
        gemv(1.0, &a, &x, 0.0, &mut y).unwrap();
        assert_eq!(y.as_slice(), &[17.0, 39.0]);
    }

    #[test]
    fn test_gemv_with_alpha_beta() {
        // y = 2 * A @ x + 3 * y
        let a = mat_f64(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let x = vec_f64(&[1.0, 1.0]);
        let mut y = vec_f64(&[10.0, 10.0]);
        gemv(2.0, &a, &x, 3.0, &mut y).unwrap();
        // A @ x = [3, 7], 2*[3,7] + 3*[10,10] = [6+30, 14+30] = [36, 44]
        assert_eq!(y.as_slice(), &[36.0, 44.0]);
    }

    #[test]
    fn test_gemv_rectangular() {
        // A = [[1, 2, 3], [4, 5, 6]]  (2x3)
        // x = [1, 0, 1]  (3)
        // y = A @ x = [4, 10]
        let a = mat_f64(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let x = vec_f64(&[1.0, 0.0, 1.0]);
        let mut y = Tensor::<f64>::zeros(vec![2]);
        gemv(1.0, &a, &x, 0.0, &mut y).unwrap();
        assert_eq!(y.as_slice(), &[4.0, 10.0]);
    }

    #[test]
    fn test_gemv_dimension_mismatch() {
        let a = mat_f64(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let x = vec_f64(&[1.0, 2.0, 3.0]);
        let mut y = Tensor::<f64>::zeros(vec![2]);
        assert!(gemv(1.0, &a, &x, 0.0, &mut y).is_err());
    }

    #[test]
    fn test_gemv_y_dimension_mismatch() {
        let a = mat_f64(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let x = vec_f64(&[1.0, 2.0]);
        let mut y = Tensor::<f64>::zeros(vec![3]);
        assert!(gemv(1.0, &a, &x, 0.0, &mut y).is_err());
    }

    // ------------------------------------------------------------------
    // BLAS L3
    // ------------------------------------------------------------------

    #[test]
    fn test_gemm_square() {
        // A = [[1, 2], [3, 4]]
        // B = [[5, 6], [7, 8]]
        // C = A @ B = [[19, 22], [43, 50]]
        let a = mat_f64(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let b = mat_f64(&[5.0, 6.0, 7.0, 8.0], 2, 2);
        let mut c = Tensor::<f64>::zeros(vec![2, 2]);
        gemm(1.0, &a, &b, 0.0, &mut c).unwrap();
        assert_eq!(c.as_slice(), &[19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_gemm_rectangular() {
        // A (2x3) @ B (3x2) = C (2x2)
        let a = mat_f64(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let b = mat_f64(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], 3, 2);
        let mut c = Tensor::<f64>::zeros(vec![2, 2]);
        gemm(1.0, &a, &b, 0.0, &mut c).unwrap();
        // Row 0: 1*7+2*9+3*11 = 7+18+33 = 58, 1*8+2*10+3*12 = 8+20+36 = 64
        // Row 1: 4*7+5*9+6*11 = 28+45+66 = 139, 4*8+5*10+6*12 = 32+50+72 = 154
        assert_eq!(c.as_slice(), &[58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_gemm_with_alpha_beta() {
        // C = 2 * A @ B + 3 * C
        let a = mat_f64(&[1.0, 0.0, 0.0, 1.0], 2, 2); // identity
        let b = mat_f64(&[5.0, 6.0, 7.0, 8.0], 2, 2);
        let mut c = mat_f64(&[1.0, 1.0, 1.0, 1.0], 2, 2);
        gemm(2.0, &a, &b, 3.0, &mut c).unwrap();
        // 2*B + 3*ones = [10+3, 12+3, 14+3, 16+3] = [13, 15, 17, 19]
        assert_eq!(c.as_slice(), &[13.0, 15.0, 17.0, 19.0]);
    }

    #[test]
    fn test_gemm_identity() {
        let a = mat_f64(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 3, 3);
        let eye = Tensor::<f64>::eye(3);
        let mut c = Tensor::<f64>::zeros(vec![3, 3]);
        gemm(1.0, &a, &eye, 0.0, &mut c).unwrap();
        assert_eq!(c.as_slice(), a.as_slice());
    }

    #[test]
    fn test_gemm_single_element() {
        let a = mat_f64(&[3.0], 1, 1);
        let b = mat_f64(&[7.0], 1, 1);
        let mut c = Tensor::<f64>::zeros(vec![1, 1]);
        gemm(1.0, &a, &b, 0.0, &mut c).unwrap();
        assert_eq!(c.as_slice(), &[21.0]);
    }

    #[test]
    fn test_gemm_dimension_mismatch() {
        let a = mat_f64(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let b = mat_f64(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
        let mut c = Tensor::<f64>::zeros(vec![2, 2]);
        assert!(gemm(1.0, &a, &b, 0.0, &mut c).is_err());
    }

    #[test]
    fn test_gemm_c_shape_mismatch() {
        let a = mat_f64(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let b = mat_f64(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let mut c = Tensor::<f64>::zeros(vec![3, 3]);
        assert!(gemm(1.0, &a, &b, 0.0, &mut c).is_err());
    }

    // ------------------------------------------------------------------
    // Convenience methods
    // ------------------------------------------------------------------

    #[test]
    fn test_matvec() {
        let a = mat_f64(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let x = vec_f64(&[5.0, 6.0]);
        let y = a.matvec(&x).unwrap();
        assert_eq!(y.as_slice(), &[17.0, 39.0]);
    }

    #[test]
    fn test_matmul() {
        let a = mat_f64(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let b = mat_f64(&[5.0, 6.0, 7.0, 8.0], 2, 2);
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.as_slice(), &[19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_tensor_dot() {
        let x = vec_f64(&[1.0, 2.0, 3.0]);
        let y = vec_f64(&[4.0, 5.0, 6.0]);
        assert_eq!(x.dot(&y).unwrap(), 32.0);
    }

    #[test]
    fn test_tensor_norm() {
        let x = vec_f64(&[3.0, 4.0]);
        assert!((x.norm().unwrap() - 5.0).abs() < 1e-10);
    }

    // ------------------------------------------------------------------
    // NumPy reference values
    // ------------------------------------------------------------------

    #[test]
    fn test_gemm_numpy_reference() {
        // >>> import numpy as np
        // >>> a = np.array([[1,2,3],[4,5,6]], dtype=np.float64)
        // >>> b = np.array([[7,8],[9,10],[11,12]], dtype=np.float64)
        // >>> a @ b
        // array([[ 58.,  64.],
        //        [139., 154.]])
        let a = mat_f64(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let b = mat_f64(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], 3, 2);
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.as_slice(), &[58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_gemv_numpy_reference() {
        // >>> a = np.array([[1,2,3],[4,5,6]], dtype=np.float64)
        // >>> x = np.array([1,1,1], dtype=np.float64)
        // >>> a @ x
        // array([ 6., 15.])
        let a = mat_f64(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let x = vec_f64(&[1.0, 1.0, 1.0]);
        let y = a.matvec(&x).unwrap();
        assert_eq!(y.as_slice(), &[6.0, 15.0]);
    }

    #[test]
    fn test_dot_numpy_reference() {
        // >>> np.dot([1,2,3,4,5], [5,4,3,2,1])
        // 35
        let x = vec_f64(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = vec_f64(&[5.0, 4.0, 3.0, 2.0, 1.0]);
        assert_eq!(dot(&x, &y).unwrap(), 35.0);
    }

    #[test]
    fn test_nrm2_numpy_reference() {
        // >>> np.linalg.norm([1, 2, 3, 4, 5])
        // 7.416198487095663
        let x = vec_f64(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let n = nrm2(&x).unwrap();
        assert!((n - 7.416_198_487_095_663).abs() < 1e-12);
    }

    #[test]
    fn test_gemm_f32() {
        // Test f32 GEMM (exercises NEON f32 micro-kernels on aarch64)
        let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let b = Tensor::from_vec(vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]).unwrap();
        let mut c = Tensor::<f32>::zeros(vec![2, 2]);
        gemm(1.0f32, &a, &b, 0.0f32, &mut c).unwrap();
        assert_eq!(c.as_slice(), &[58.0f32, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_gemm_f32_large() {
        // 16x16 f32 GEMM — exercises 8x4 and 4x4 micro-kernels + remainder handling
        let n = 16;
        let a_data: Vec<f32> = (0..n * n).map(|i| (i % 7) as f32 + 1.0).collect();
        let b_data: Vec<f32> = (0..n * n).map(|i| ((i + 3) % 5) as f32 + 1.0).collect();
        let a = Tensor::from_vec(a_data.clone(), vec![n, n]).unwrap();
        let b = Tensor::from_vec(b_data.clone(), vec![n, n]).unwrap();
        let mut c = Tensor::<f32>::zeros(vec![n, n]);
        gemm(1.0f32, &a, &b, 0.0f32, &mut c).unwrap();

        // Verify against naive multiplication
        for i in 0..n {
            for j in 0..n {
                let mut expected = 0.0f32;
                for k in 0..n {
                    expected += a_data[i * n + k] * b_data[k * n + j];
                }
                let actual = c.as_slice()[i * n + j];
                assert!(
                    (actual - expected).abs() < 1e-3,
                    "mismatch at [{i},{j}]: expected {expected}, got {actual}"
                );
            }
        }
    }
}
