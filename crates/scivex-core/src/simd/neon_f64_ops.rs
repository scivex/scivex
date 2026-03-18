//! NEON-accelerated f64 kernels (2-wide `float64x2_t`) for aarch64.

use core::arch::aarch64::*;

/// Horizontal sum of a `float64x2_t` (2 lanes).
///
/// # Safety
/// Caller must ensure this runs on aarch64.
#[inline(always)]
unsafe fn hsum_f64x2(v: float64x2_t) -> f64 {
    vaddvq_f64(v)
}

/// NEON dot product of two f64 slices.
///
/// # Safety
/// Caller must ensure this runs on aarch64.
#[inline]
pub(crate) unsafe fn dot_f64_neon(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    let chunks = n / 2;
    let remainder = n % 2;

    let mut acc = vdupq_n_f64(0.0);
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let va = vld1q_f64(a_ptr.add(i * 2));
        let vb = vld1q_f64(b_ptr.add(i * 2));
        acc = vfmaq_f64(acc, va, vb);
    }

    let mut result = hsum_f64x2(acc);
    let tail = chunks * 2;
    for j in 0..remainder {
        result += *a.get_unchecked(tail + j) * *b.get_unchecked(tail + j);
    }
    result
}

/// NEON sum of an f64 slice.
///
/// # Safety
/// Caller must ensure this runs on aarch64.
#[inline]
pub(crate) unsafe fn sum_f64_neon(a: &[f64]) -> f64 {
    let n = a.len();
    let chunks = n / 2;
    let remainder = n % 2;

    let mut acc = vdupq_n_f64(0.0);
    let ptr = a.as_ptr();

    for i in 0..chunks {
        let va = vld1q_f64(ptr.add(i * 2));
        acc = vaddq_f64(acc, va);
    }

    let mut result = hsum_f64x2(acc);
    let tail = chunks * 2;
    for j in 0..remainder {
        result += *a.get_unchecked(tail + j);
    }
    result
}

/// NEON element-wise add for f64.
///
/// # Safety
/// Caller must ensure this runs on aarch64 and slices have equal length.
#[inline]
pub(crate) unsafe fn add_f64_neon(a: &[f64], b: &[f64], out: &mut [f64]) {
    let n = a.len();
    let chunks = n / 2;
    let remainder = n % 2;

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let o_ptr = out.as_mut_ptr();

    for i in 0..chunks {
        let off = i * 2;
        let va = vld1q_f64(a_ptr.add(off));
        let vb = vld1q_f64(b_ptr.add(off));
        vst1q_f64(o_ptr.add(off), vaddq_f64(va, vb));
    }

    let tail = chunks * 2;
    for j in 0..remainder {
        *out.get_unchecked_mut(tail + j) = *a.get_unchecked(tail + j) + *b.get_unchecked(tail + j);
    }
}

/// NEON element-wise mul for f64.
///
/// # Safety
/// Caller must ensure this runs on aarch64 and slices have equal length.
#[inline]
pub(crate) unsafe fn mul_f64_neon(a: &[f64], b: &[f64], out: &mut [f64]) {
    let n = a.len();
    let chunks = n / 2;
    let remainder = n % 2;

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let o_ptr = out.as_mut_ptr();

    for i in 0..chunks {
        let off = i * 2;
        let va = vld1q_f64(a_ptr.add(off));
        let vb = vld1q_f64(b_ptr.add(off));
        vst1q_f64(o_ptr.add(off), vmulq_f64(va, vb));
    }

    let tail = chunks * 2;
    for j in 0..remainder {
        *out.get_unchecked_mut(tail + j) = *a.get_unchecked(tail + j) * *b.get_unchecked(tail + j);
    }
}

/// NEON axpy for f64: `y[i] += alpha * x[i]`.
///
/// # Safety
/// Caller must ensure this runs on aarch64 and slices have equal length.
#[inline]
pub(crate) unsafe fn axpy_f64_neon(alpha: f64, x: &[f64], y: &mut [f64]) {
    let n = x.len();
    let chunks = n / 2;
    let remainder = n % 2;

    let valpha = vdupq_n_f64(alpha);
    let x_ptr = x.as_ptr();
    let y_ptr = y.as_mut_ptr();

    for i in 0..chunks {
        let off = i * 2;
        let vx = vld1q_f64(x_ptr.add(off));
        let vy = vld1q_f64(y_ptr.add(off));
        vst1q_f64(y_ptr.add(off), vfmaq_f64(vy, valpha, vx));
    }

    let tail = chunks * 2;
    for j in 0..remainder {
        *y.get_unchecked_mut(tail + j) += alpha * *x.get_unchecked(tail + j);
    }
}

/// NEON scal for f64: `x[i] *= alpha`.
///
/// # Safety
/// Caller must ensure this runs on aarch64.
#[inline]
pub(crate) unsafe fn scal_f64_neon(alpha: f64, x: &mut [f64]) {
    let n = x.len();
    let chunks = n / 2;
    let remainder = n % 2;

    let valpha = vdupq_n_f64(alpha);
    let ptr = x.as_mut_ptr();

    for i in 0..chunks {
        let off = i * 2;
        let vx = vld1q_f64(ptr.add(off));
        vst1q_f64(ptr.add(off), vmulq_f64(valpha, vx));
    }

    let tail = chunks * 2;
    for j in 0..remainder {
        *x.get_unchecked_mut(tail + j) *= alpha;
    }
}

/// NEON sum of squares for f64.
///
/// # Safety
/// Caller must ensure this runs on aarch64.
#[inline]
pub(crate) unsafe fn sum_sq_f64_neon(a: &[f64]) -> f64 {
    let n = a.len();
    let chunks = n / 2;
    let remainder = n % 2;

    let mut acc = vdupq_n_f64(0.0);
    let ptr = a.as_ptr();

    for i in 0..chunks {
        let va = vld1q_f64(ptr.add(i * 2));
        acc = vfmaq_f64(acc, va, va);
    }

    let mut result = hsum_f64x2(acc);
    let tail = chunks * 2;
    for j in 0..remainder {
        let v = *a.get_unchecked(tail + j);
        result += v * v;
    }
    result
}

/// NEON sum of absolute values for f64.
///
/// # Safety
/// Caller must ensure this runs on aarch64.
#[inline]
pub(crate) unsafe fn asum_f64_neon(a: &[f64]) -> f64 {
    let n = a.len();
    let chunks = n / 2;
    let remainder = n % 2;

    let mut acc = vdupq_n_f64(0.0);
    let ptr = a.as_ptr();

    for i in 0..chunks {
        let va = vld1q_f64(ptr.add(i * 2));
        acc = vaddq_f64(acc, vabsq_f64(va));
    }

    let mut result = hsum_f64x2(acc);
    let tail = chunks * 2;
    for j in 0..remainder {
        result += (*a.get_unchecked(tail + j)).abs();
    }
    result
}

/// NEON min of an f64 slice.
///
/// # Safety
/// Caller must ensure this runs on aarch64 and slice is non-empty.
#[inline]
pub(crate) unsafe fn min_f64_neon(a: &[f64]) -> f64 {
    let n = a.len();
    let chunks = n / 2;
    let remainder = n % 2;

    let mut vmin = vdupq_n_f64(f64::INFINITY);
    let ptr = a.as_ptr();

    for i in 0..chunks {
        let va = vld1q_f64(ptr.add(i * 2));
        vmin = vminq_f64(vmin, va);
    }

    let mut result = vminvq_f64(vmin);
    let tail = chunks * 2;
    for j in 0..remainder {
        result = result.min(*a.get_unchecked(tail + j));
    }
    result
}

/// NEON max of an f64 slice.
///
/// # Safety
/// Caller must ensure this runs on aarch64 and slice is non-empty.
#[inline]
pub(crate) unsafe fn max_f64_neon(a: &[f64]) -> f64 {
    let n = a.len();
    let chunks = n / 2;
    let remainder = n % 2;

    let mut vmax = vdupq_n_f64(f64::NEG_INFINITY);
    let ptr = a.as_ptr();

    for i in 0..chunks {
        let va = vld1q_f64(ptr.add(i * 2));
        vmax = vmaxq_f64(vmax, va);
    }

    let mut result = vmaxvq_f64(vmax);
    let tail = chunks * 2;
    for j in 0..remainder {
        result = result.max(*a.get_unchecked(tail + j));
    }
    result
}

/// NEON mean of an f64 slice.
///
/// # Safety
/// Caller must ensure this runs on aarch64 and slice is non-empty.
#[inline]
pub(crate) unsafe fn mean_f64_neon(a: &[f64]) -> f64 {
    sum_f64_neon(a) / a.len() as f64
}
