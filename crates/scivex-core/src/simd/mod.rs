//! SIMD-accelerated kernels for core numerical operations.
//!
//! This module provides hand-tuned SIMD implementations for:
//! - Element-wise operations (add, mul, fma)
//! - Reductions (sum, dot product, min, max)
//! - BLAS Level 1 operations (axpy, scal, nrm2, asum)
//!
//! All kernels use runtime feature detection (`is_x86_feature_detected!` on
//! x86_64) and fall back to scalar loops on unsupported platforms.
//!
//! # Safety
//!
//! SIMD intrinsic functions are `unsafe`. Each call site has a `// SAFETY:`
//! comment documenting the invariant (feature detection, alignment, bounds).

pub mod f32_ops;
pub mod f64_ops;

/// Dispatch a SIMD kernel with AVX fallback on x86_64.
///
/// Usage: `dispatch_f64!(avx_fn, fallback_fn, args...)`
macro_rules! dispatch_f64 {
    ($avx_fn:expr, $fallback_fn:expr, $($arg:expr),* $(,)?) => {{
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx") {
                // SAFETY: we just confirmed AVX is available at runtime.
                unsafe { $avx_fn($($arg),*) }
            } else {
                $fallback_fn($($arg),*)
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            $fallback_fn($($arg),*)
        }
    }};
}

/// Dispatch a SIMD kernel with AVX fallback on x86_64 (f32 variant).
macro_rules! dispatch_f32 {
    ($avx_fn:expr, $fallback_fn:expr, $($arg:expr),* $(,)?) => {{
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx") {
                // SAFETY: we just confirmed AVX is available at runtime.
                unsafe { $avx_fn($($arg),*) }
            } else {
                $fallback_fn($($arg),*)
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            $fallback_fn($($arg),*)
        }
    }};
}

pub(crate) use dispatch_f32;
pub(crate) use dispatch_f64;

// ---------------------------------------------------------------------------
// Type-punning helpers (used by blas.rs and ops.rs for generic→concrete dispatch)
// ---------------------------------------------------------------------------

/// Reinterpret a `&[T]` as `&[f64]`.
///
/// # Safety
/// Caller must ensure `T` is actually `f64` (verified via `TypeId`).
#[inline]
pub(crate) unsafe fn slice_as_f64<T>(s: &[T]) -> &[f64] {
    // SAFETY: T is f64 (caller invariant). Same size, same alignment.
    unsafe { core::slice::from_raw_parts(s.as_ptr().cast::<f64>(), s.len()) }
}

/// Reinterpret a `&[T]` as `&[f32]`.
///
/// # Safety
/// Caller must ensure `T` is actually `f32` (verified via `TypeId`).
#[inline]
pub(crate) unsafe fn slice_as_f32<T>(s: &[T]) -> &[f32] {
    // SAFETY: T is f32 (caller invariant). Same size, same alignment.
    unsafe { core::slice::from_raw_parts(s.as_ptr().cast::<f32>(), s.len()) }
}

/// Reinterpret a `&mut [T]` as `&mut [f64]`.
///
/// # Safety
/// Caller must ensure `T` is actually `f64` (verified via `TypeId`).
#[inline]
pub(crate) unsafe fn slice_as_f64_mut<T>(s: &mut [T]) -> &mut [f64] {
    // SAFETY: T is f64 (caller invariant). Same size, same alignment.
    unsafe { core::slice::from_raw_parts_mut(s.as_mut_ptr().cast::<f64>(), s.len()) }
}

/// Reinterpret a `&mut [T]` as `&mut [f32]`.
///
/// # Safety
/// Caller must ensure `T` is actually `f32` (verified via `TypeId`).
#[inline]
pub(crate) unsafe fn slice_as_f32_mut<T>(s: &mut [T]) -> &mut [f32] {
    // SAFETY: T is f32 (caller invariant). Same size, same alignment.
    unsafe { core::slice::from_raw_parts_mut(s.as_mut_ptr().cast::<f32>(), s.len()) }
}

/// Transmute an `f64` value to `T`.
///
/// # Safety
/// Caller must ensure `T` is actually `f64`.
#[inline]
pub(crate) unsafe fn f64_to_t<T: Copy>(v: f64) -> T {
    // SAFETY: T is f64 (caller invariant). Same size.
    unsafe { *(&raw const v).cast::<T>() }
}

/// Transmute an `f32` value to `T`.
///
/// # Safety
/// Caller must ensure `T` is actually `f32`.
#[inline]
pub(crate) unsafe fn f32_to_t<T: Copy>(v: f32) -> T {
    // SAFETY: T is f32 (caller invariant). Same size.
    unsafe { *(&raw const v).cast::<T>() }
}

/// Transmute a `T` value to `f64`.
///
/// # Safety
/// Caller must ensure `T` is actually `f64`.
#[inline]
pub(crate) unsafe fn t_to_f64<T: Copy>(v: T) -> f64 {
    // SAFETY: T is f64 (caller invariant). Same size.
    unsafe { *(&raw const v).cast::<f64>() }
}

/// Transmute a `T` value to `f32`.
///
/// # Safety
/// Caller must ensure `T` is actually `f32`.
#[inline]
pub(crate) unsafe fn t_to_f32<T: Copy>(v: T) -> f32 {
    // SAFETY: T is f32 (caller invariant). Same size.
    unsafe { *(&raw const v).cast::<f32>() }
}
