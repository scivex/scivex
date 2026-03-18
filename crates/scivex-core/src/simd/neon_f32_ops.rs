//! NEON-accelerated f32 kernels (4-wide `float32x4_t`) for aarch64.

use core::arch::aarch64::*;

/// NEON dot product of two f32 slices.
///
/// # Safety
/// Caller must ensure this runs on aarch64 (NEON always available).
#[inline]
pub(crate) unsafe fn dot_f32_neon(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    // SAFETY: NEON is always available on aarch64.
    let mut acc = vdupq_n_f32(0.0);
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let va = vld1q_f32(a_ptr.add(i * 4));
        let vb = vld1q_f32(b_ptr.add(i * 4));
        acc = vfmaq_f32(acc, va, vb);
    }

    let mut result = vaddvq_f32(acc);
    let tail = chunks * 4;
    for j in 0..remainder {
        result += *a.get_unchecked(tail + j) * *b.get_unchecked(tail + j);
    }
    result
}

/// NEON sum of an f32 slice.
///
/// # Safety
/// Caller must ensure this runs on aarch64.
#[inline]
pub(crate) unsafe fn sum_f32_neon(a: &[f32]) -> f32 {
    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let mut acc = vdupq_n_f32(0.0);
    let ptr = a.as_ptr();

    for i in 0..chunks {
        let va = vld1q_f32(ptr.add(i * 4));
        acc = vaddq_f32(acc, va);
    }

    let mut result = vaddvq_f32(acc);
    let tail = chunks * 4;
    for j in 0..remainder {
        result += *a.get_unchecked(tail + j);
    }
    result
}

/// NEON element-wise add for f32.
///
/// # Safety
/// Caller must ensure this runs on aarch64 and slices have equal length.
#[inline]
pub(crate) unsafe fn add_f32_neon(a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let o_ptr = out.as_mut_ptr();

    for i in 0..chunks {
        let off = i * 4;
        let va = vld1q_f32(a_ptr.add(off));
        let vb = vld1q_f32(b_ptr.add(off));
        vst1q_f32(o_ptr.add(off), vaddq_f32(va, vb));
    }

    let tail = chunks * 4;
    for j in 0..remainder {
        *out.get_unchecked_mut(tail + j) = *a.get_unchecked(tail + j) + *b.get_unchecked(tail + j);
    }
}

/// NEON element-wise mul for f32.
///
/// # Safety
/// Caller must ensure this runs on aarch64 and slices have equal length.
#[inline]
pub(crate) unsafe fn mul_f32_neon(a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let o_ptr = out.as_mut_ptr();

    for i in 0..chunks {
        let off = i * 4;
        let va = vld1q_f32(a_ptr.add(off));
        let vb = vld1q_f32(b_ptr.add(off));
        vst1q_f32(o_ptr.add(off), vmulq_f32(va, vb));
    }

    let tail = chunks * 4;
    for j in 0..remainder {
        *out.get_unchecked_mut(tail + j) = *a.get_unchecked(tail + j) * *b.get_unchecked(tail + j);
    }
}

/// NEON axpy for f32: `y[i] += alpha * x[i]`.
///
/// # Safety
/// Caller must ensure this runs on aarch64 and slices have equal length.
#[inline]
pub(crate) unsafe fn axpy_f32_neon(alpha: f32, x: &[f32], y: &mut [f32]) {
    let n = x.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let valpha = vdupq_n_f32(alpha);
    let x_ptr = x.as_ptr();
    let y_ptr = y.as_mut_ptr();

    for i in 0..chunks {
        let off = i * 4;
        let vx = vld1q_f32(x_ptr.add(off));
        let vy = vld1q_f32(y_ptr.add(off));
        vst1q_f32(y_ptr.add(off), vfmaq_f32(vy, valpha, vx));
    }

    let tail = chunks * 4;
    for j in 0..remainder {
        *y.get_unchecked_mut(tail + j) += alpha * *x.get_unchecked(tail + j);
    }
}

/// NEON scal for f32: `x[i] *= alpha`.
///
/// # Safety
/// Caller must ensure this runs on aarch64.
#[inline]
pub(crate) unsafe fn scal_f32_neon(alpha: f32, x: &mut [f32]) {
    let n = x.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let valpha = vdupq_n_f32(alpha);
    let ptr = x.as_mut_ptr();

    for i in 0..chunks {
        let off = i * 4;
        let vx = vld1q_f32(ptr.add(off));
        vst1q_f32(ptr.add(off), vmulq_f32(valpha, vx));
    }

    let tail = chunks * 4;
    for j in 0..remainder {
        *x.get_unchecked_mut(tail + j) *= alpha;
    }
}

/// NEON sum of squares for f32.
///
/// # Safety
/// Caller must ensure this runs on aarch64.
#[inline]
pub(crate) unsafe fn sum_sq_f32_neon(a: &[f32]) -> f32 {
    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let mut acc = vdupq_n_f32(0.0);
    let ptr = a.as_ptr();

    for i in 0..chunks {
        let va = vld1q_f32(ptr.add(i * 4));
        acc = vfmaq_f32(acc, va, va);
    }

    let mut result = vaddvq_f32(acc);
    let tail = chunks * 4;
    for j in 0..remainder {
        let v = *a.get_unchecked(tail + j);
        result += v * v;
    }
    result
}

/// NEON sum of absolute values for f32.
///
/// # Safety
/// Caller must ensure this runs on aarch64.
#[inline]
pub(crate) unsafe fn asum_f32_neon(a: &[f32]) -> f32 {
    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let mut acc = vdupq_n_f32(0.0);
    let ptr = a.as_ptr();

    for i in 0..chunks {
        let va = vld1q_f32(ptr.add(i * 4));
        acc = vaddq_f32(acc, vabsq_f32(va));
    }

    let mut result = vaddvq_f32(acc);
    let tail = chunks * 4;
    for j in 0..remainder {
        result += (*a.get_unchecked(tail + j)).abs();
    }
    result
}

/// NEON min of an f32 slice.
///
/// # Safety
/// Caller must ensure this runs on aarch64 and slice is non-empty.
#[inline]
pub(crate) unsafe fn min_f32_neon(a: &[f32]) -> f32 {
    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let mut vmin = vdupq_n_f32(f32::INFINITY);
    let ptr = a.as_ptr();

    for i in 0..chunks {
        let va = vld1q_f32(ptr.add(i * 4));
        vmin = vminq_f32(vmin, va);
    }

    let mut result = vminvq_f32(vmin);
    let tail = chunks * 4;
    for j in 0..remainder {
        result = result.min(*a.get_unchecked(tail + j));
    }
    result
}

/// NEON max of an f32 slice.
///
/// # Safety
/// Caller must ensure this runs on aarch64 and slice is non-empty.
#[inline]
pub(crate) unsafe fn max_f32_neon(a: &[f32]) -> f32 {
    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let mut vmax = vdupq_n_f32(f32::NEG_INFINITY);
    let ptr = a.as_ptr();

    for i in 0..chunks {
        let va = vld1q_f32(ptr.add(i * 4));
        vmax = vmaxq_f32(vmax, va);
    }

    let mut result = vmaxvq_f32(vmax);
    let tail = chunks * 4;
    for j in 0..remainder {
        result = result.max(*a.get_unchecked(tail + j));
    }
    result
}

/// NEON mean of an f32 slice.
///
/// # Safety
/// Caller must ensure this runs on aarch64 and slice is non-empty.
#[inline]
pub(crate) unsafe fn mean_f32_neon(a: &[f32]) -> f32 {
    sum_f32_neon(a) / a.len() as f32
}
