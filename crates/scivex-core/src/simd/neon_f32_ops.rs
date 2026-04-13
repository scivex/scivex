//! NEON-accelerated f32 kernels (4-wide `float32x4_t`) for aarch64.
#![allow(clippy::wildcard_imports)]

use core::arch::aarch64::*;

/// NEON dot product of two f32 slices.
///
/// # Safety
/// Caller must ensure this runs on aarch64 (NEON always available).
#[inline]
pub(crate) unsafe fn dot_f32_neon(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    // 4 accumulators × 4 lanes = 16 f32 FMAs per unrolled iteration.
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);

    let chunks16 = n / 16;
    for i in 0..chunks16 {
        let base = i * 16;
        acc0 = vfmaq_f32(acc0, vld1q_f32(a_ptr.add(base)), vld1q_f32(b_ptr.add(base)));
        acc1 = vfmaq_f32(
            acc1,
            vld1q_f32(a_ptr.add(base + 4)),
            vld1q_f32(b_ptr.add(base + 4)),
        );
        acc2 = vfmaq_f32(
            acc2,
            vld1q_f32(a_ptr.add(base + 8)),
            vld1q_f32(b_ptr.add(base + 8)),
        );
        acc3 = vfmaq_f32(
            acc3,
            vld1q_f32(a_ptr.add(base + 12)),
            vld1q_f32(b_ptr.add(base + 12)),
        );
    }

    acc0 = vaddq_f32(acc0, acc1);
    acc2 = vaddq_f32(acc2, acc3);
    acc0 = vaddq_f32(acc0, acc2);
    let mut result = vaddvq_f32(acc0);

    let tail = chunks16 * 16;
    for j in tail..n {
        result += *a.get_unchecked(j) * *b.get_unchecked(j);
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
    let ptr = a.as_ptr();

    // 4 accumulators × 4 lanes = 16 f32s per unrolled iteration.
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);

    let chunks16 = n / 16;
    for i in 0..chunks16 {
        let base = i * 16;
        acc0 = vaddq_f32(acc0, vld1q_f32(ptr.add(base)));
        acc1 = vaddq_f32(acc1, vld1q_f32(ptr.add(base + 4)));
        acc2 = vaddq_f32(acc2, vld1q_f32(ptr.add(base + 8)));
        acc3 = vaddq_f32(acc3, vld1q_f32(ptr.add(base + 12)));
    }

    acc0 = vaddq_f32(acc0, acc1);
    acc2 = vaddq_f32(acc2, acc3);
    acc0 = vaddq_f32(acc0, acc2);
    let mut result = vaddvq_f32(acc0);

    let tail = chunks16 * 16;
    for j in tail..n {
        result += *a.get_unchecked(j);
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
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let o_ptr = out.as_mut_ptr();

    let chunks16 = n / 16;
    for i in 0..chunks16 {
        let base = i * 16;
        vst1q_f32(
            o_ptr.add(base),
            vaddq_f32(vld1q_f32(a_ptr.add(base)), vld1q_f32(b_ptr.add(base))),
        );
        vst1q_f32(
            o_ptr.add(base + 4),
            vaddq_f32(
                vld1q_f32(a_ptr.add(base + 4)),
                vld1q_f32(b_ptr.add(base + 4)),
            ),
        );
        vst1q_f32(
            o_ptr.add(base + 8),
            vaddq_f32(
                vld1q_f32(a_ptr.add(base + 8)),
                vld1q_f32(b_ptr.add(base + 8)),
            ),
        );
        vst1q_f32(
            o_ptr.add(base + 12),
            vaddq_f32(
                vld1q_f32(a_ptr.add(base + 12)),
                vld1q_f32(b_ptr.add(base + 12)),
            ),
        );
    }

    let tail = chunks16 * 16;
    for j in tail..n {
        *out.get_unchecked_mut(j) = *a.get_unchecked(j) + *b.get_unchecked(j);
    }
}

/// NEON element-wise sub for f32.
///
/// # Safety
/// Caller must ensure this runs on aarch64 and slices have equal length.
#[inline]
pub(crate) unsafe fn sub_f32_neon(a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = a.len();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let o_ptr = out.as_mut_ptr();

    let chunks16 = n / 16;
    for i in 0..chunks16 {
        let base = i * 16;
        vst1q_f32(
            o_ptr.add(base),
            vsubq_f32(vld1q_f32(a_ptr.add(base)), vld1q_f32(b_ptr.add(base))),
        );
        vst1q_f32(
            o_ptr.add(base + 4),
            vsubq_f32(
                vld1q_f32(a_ptr.add(base + 4)),
                vld1q_f32(b_ptr.add(base + 4)),
            ),
        );
        vst1q_f32(
            o_ptr.add(base + 8),
            vsubq_f32(
                vld1q_f32(a_ptr.add(base + 8)),
                vld1q_f32(b_ptr.add(base + 8)),
            ),
        );
        vst1q_f32(
            o_ptr.add(base + 12),
            vsubq_f32(
                vld1q_f32(a_ptr.add(base + 12)),
                vld1q_f32(b_ptr.add(base + 12)),
            ),
        );
    }

    let tail = chunks16 * 16;
    for j in tail..n {
        *out.get_unchecked_mut(j) = *a.get_unchecked(j) - *b.get_unchecked(j);
    }
}

/// NEON element-wise mul for f32.
///
/// # Safety
/// Caller must ensure this runs on aarch64 and slices have equal length.
#[inline]
pub(crate) unsafe fn mul_f32_neon(a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = a.len();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let o_ptr = out.as_mut_ptr();

    let chunks16 = n / 16;
    for i in 0..chunks16 {
        let base = i * 16;
        vst1q_f32(
            o_ptr.add(base),
            vmulq_f32(vld1q_f32(a_ptr.add(base)), vld1q_f32(b_ptr.add(base))),
        );
        vst1q_f32(
            o_ptr.add(base + 4),
            vmulq_f32(
                vld1q_f32(a_ptr.add(base + 4)),
                vld1q_f32(b_ptr.add(base + 4)),
            ),
        );
        vst1q_f32(
            o_ptr.add(base + 8),
            vmulq_f32(
                vld1q_f32(a_ptr.add(base + 8)),
                vld1q_f32(b_ptr.add(base + 8)),
            ),
        );
        vst1q_f32(
            o_ptr.add(base + 12),
            vmulq_f32(
                vld1q_f32(a_ptr.add(base + 12)),
                vld1q_f32(b_ptr.add(base + 12)),
            ),
        );
    }

    let tail = chunks16 * 16;
    for j in tail..n {
        *out.get_unchecked_mut(j) = *a.get_unchecked(j) * *b.get_unchecked(j);
    }
}

/// NEON element-wise div for f32.
///
/// # Safety
/// Caller must ensure this runs on aarch64 and slices have equal length.
#[inline]
pub(crate) unsafe fn div_f32_neon(a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = a.len();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let o_ptr = out.as_mut_ptr();

    let chunks16 = n / 16;
    for i in 0..chunks16 {
        let base = i * 16;
        vst1q_f32(
            o_ptr.add(base),
            vdivq_f32(vld1q_f32(a_ptr.add(base)), vld1q_f32(b_ptr.add(base))),
        );
        vst1q_f32(
            o_ptr.add(base + 4),
            vdivq_f32(
                vld1q_f32(a_ptr.add(base + 4)),
                vld1q_f32(b_ptr.add(base + 4)),
            ),
        );
        vst1q_f32(
            o_ptr.add(base + 8),
            vdivq_f32(
                vld1q_f32(a_ptr.add(base + 8)),
                vld1q_f32(b_ptr.add(base + 8)),
            ),
        );
        vst1q_f32(
            o_ptr.add(base + 12),
            vdivq_f32(
                vld1q_f32(a_ptr.add(base + 12)),
                vld1q_f32(b_ptr.add(base + 12)),
            ),
        );
    }

    let tail = chunks16 * 16;
    for j in tail..n {
        *out.get_unchecked_mut(j) = *a.get_unchecked(j) / *b.get_unchecked(j);
    }
}

/// NEON axpy for f32: `y[i] += alpha * x[i]`.
///
/// # Safety
/// Caller must ensure this runs on aarch64 and slices have equal length.
#[inline]
pub(crate) unsafe fn axpy_f32_neon(alpha: f32, x: &[f32], y: &mut [f32]) {
    let n = x.len();
    let valpha = vdupq_n_f32(alpha);
    let x_ptr = x.as_ptr();
    let y_ptr = y.as_mut_ptr();

    let chunks16 = n / 16;
    for i in 0..chunks16 {
        let base = i * 16;
        let vy0 = vld1q_f32(y_ptr.add(base));
        let vy1 = vld1q_f32(y_ptr.add(base + 4));
        let vy2 = vld1q_f32(y_ptr.add(base + 8));
        let vy3 = vld1q_f32(y_ptr.add(base + 12));
        vst1q_f32(
            y_ptr.add(base),
            vfmaq_f32(vy0, valpha, vld1q_f32(x_ptr.add(base))),
        );
        vst1q_f32(
            y_ptr.add(base + 4),
            vfmaq_f32(vy1, valpha, vld1q_f32(x_ptr.add(base + 4))),
        );
        vst1q_f32(
            y_ptr.add(base + 8),
            vfmaq_f32(vy2, valpha, vld1q_f32(x_ptr.add(base + 8))),
        );
        vst1q_f32(
            y_ptr.add(base + 12),
            vfmaq_f32(vy3, valpha, vld1q_f32(x_ptr.add(base + 12))),
        );
    }

    let tail = chunks16 * 16;
    for j in tail..n {
        *y.get_unchecked_mut(j) += alpha * *x.get_unchecked(j);
    }
}

/// NEON scal for f32: `x[i] *= alpha`.
///
/// # Safety
/// Caller must ensure this runs on aarch64.
#[inline]
pub(crate) unsafe fn scal_f32_neon(alpha: f32, x: &mut [f32]) {
    let n = x.len();
    let valpha = vdupq_n_f32(alpha);
    let ptr = x.as_mut_ptr();

    let chunks16 = n / 16;
    for i in 0..chunks16 {
        let base = i * 16;
        vst1q_f32(ptr.add(base), vmulq_f32(valpha, vld1q_f32(ptr.add(base))));
        vst1q_f32(
            ptr.add(base + 4),
            vmulq_f32(valpha, vld1q_f32(ptr.add(base + 4))),
        );
        vst1q_f32(
            ptr.add(base + 8),
            vmulq_f32(valpha, vld1q_f32(ptr.add(base + 8))),
        );
        vst1q_f32(
            ptr.add(base + 12),
            vmulq_f32(valpha, vld1q_f32(ptr.add(base + 12))),
        );
    }

    let tail = chunks16 * 16;
    for j in tail..n {
        *x.get_unchecked_mut(j) *= alpha;
    }
}

/// NEON sum of squares for f32.
///
/// # Safety
/// Caller must ensure this runs on aarch64.
#[inline]
pub(crate) unsafe fn sum_sq_f32_neon(a: &[f32]) -> f32 {
    let n = a.len();
    let ptr = a.as_ptr();

    // 4 accumulators × 4 lanes = 16 f32s per unrolled iteration.
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);

    let chunks16 = n / 16;
    for i in 0..chunks16 {
        let base = i * 16;
        let v0 = vld1q_f32(ptr.add(base));
        let v1 = vld1q_f32(ptr.add(base + 4));
        let v2 = vld1q_f32(ptr.add(base + 8));
        let v3 = vld1q_f32(ptr.add(base + 12));
        acc0 = vfmaq_f32(acc0, v0, v0);
        acc1 = vfmaq_f32(acc1, v1, v1);
        acc2 = vfmaq_f32(acc2, v2, v2);
        acc3 = vfmaq_f32(acc3, v3, v3);
    }

    acc0 = vaddq_f32(acc0, acc1);
    acc2 = vaddq_f32(acc2, acc3);
    acc0 = vaddq_f32(acc0, acc2);
    let mut result = vaddvq_f32(acc0);

    let tail = chunks16 * 16;
    for j in tail..n {
        let v = *a.get_unchecked(j);
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
    let ptr = a.as_ptr();

    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);

    let chunks16 = n / 16;
    for i in 0..chunks16 {
        let base = i * 16;
        acc0 = vaddq_f32(acc0, vabsq_f32(vld1q_f32(ptr.add(base))));
        acc1 = vaddq_f32(acc1, vabsq_f32(vld1q_f32(ptr.add(base + 4))));
        acc2 = vaddq_f32(acc2, vabsq_f32(vld1q_f32(ptr.add(base + 8))));
        acc3 = vaddq_f32(acc3, vabsq_f32(vld1q_f32(ptr.add(base + 12))));
    }

    acc0 = vaddq_f32(acc0, acc1);
    acc2 = vaddq_f32(acc2, acc3);
    acc0 = vaddq_f32(acc0, acc2);
    let mut result = vaddvq_f32(acc0);

    let tail = chunks16 * 16;
    for j in tail..n {
        result += (*a.get_unchecked(j)).abs();
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
    let ptr = a.as_ptr();

    let mut vmin0 = vdupq_n_f32(f32::INFINITY);
    let mut vmin1 = vdupq_n_f32(f32::INFINITY);
    let mut vmin2 = vdupq_n_f32(f32::INFINITY);
    let mut vmin3 = vdupq_n_f32(f32::INFINITY);

    let chunks16 = n / 16;
    for i in 0..chunks16 {
        let base = i * 16;
        vmin0 = vminq_f32(vmin0, vld1q_f32(ptr.add(base)));
        vmin1 = vminq_f32(vmin1, vld1q_f32(ptr.add(base + 4)));
        vmin2 = vminq_f32(vmin2, vld1q_f32(ptr.add(base + 8)));
        vmin3 = vminq_f32(vmin3, vld1q_f32(ptr.add(base + 12)));
    }

    vmin0 = vminq_f32(vmin0, vmin1);
    vmin2 = vminq_f32(vmin2, vmin3);
    vmin0 = vminq_f32(vmin0, vmin2);
    let mut result = vminvq_f32(vmin0);

    let tail = chunks16 * 16;
    for j in tail..n {
        result = result.min(*a.get_unchecked(j));
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
    let ptr = a.as_ptr();

    let mut vmax0 = vdupq_n_f32(f32::NEG_INFINITY);
    let mut vmax1 = vdupq_n_f32(f32::NEG_INFINITY);
    let mut vmax2 = vdupq_n_f32(f32::NEG_INFINITY);
    let mut vmax3 = vdupq_n_f32(f32::NEG_INFINITY);

    let chunks16 = n / 16;
    for i in 0..chunks16 {
        let base = i * 16;
        vmax0 = vmaxq_f32(vmax0, vld1q_f32(ptr.add(base)));
        vmax1 = vmaxq_f32(vmax1, vld1q_f32(ptr.add(base + 4)));
        vmax2 = vmaxq_f32(vmax2, vld1q_f32(ptr.add(base + 8)));
        vmax3 = vmaxq_f32(vmax3, vld1q_f32(ptr.add(base + 12)));
    }

    vmax0 = vmaxq_f32(vmax0, vmax1);
    vmax2 = vmaxq_f32(vmax2, vmax3);
    vmax0 = vmaxq_f32(vmax0, vmax2);
    let mut result = vmaxvq_f32(vmax0);

    let tail = chunks16 * 16;
    for j in tail..n {
        result = result.max(*a.get_unchecked(j));
    }
    result
}

/// NEON ReLU (max(0, x)) for f32.
///
/// # Safety
/// Caller must ensure this runs on aarch64 and slices have equal length.
#[inline]
pub(crate) unsafe fn relu_f32_neon(a: &[f32], out: &mut [f32]) {
    let n = a.len();
    let a_ptr = a.as_ptr();
    let o_ptr = out.as_mut_ptr();
    let vzero = vdupq_n_f32(0.0);

    let chunks16 = n / 16;
    for i in 0..chunks16 {
        let base = i * 16;
        vst1q_f32(
            o_ptr.add(base),
            vmaxq_f32(vzero, vld1q_f32(a_ptr.add(base))),
        );
        vst1q_f32(
            o_ptr.add(base + 4),
            vmaxq_f32(vzero, vld1q_f32(a_ptr.add(base + 4))),
        );
        vst1q_f32(
            o_ptr.add(base + 8),
            vmaxq_f32(vzero, vld1q_f32(a_ptr.add(base + 8))),
        );
        vst1q_f32(
            o_ptr.add(base + 12),
            vmaxq_f32(vzero, vld1q_f32(a_ptr.add(base + 12))),
        );
    }

    let tail = chunks16 * 16;
    for j in tail..n {
        let v = *a.get_unchecked(j);
        *out.get_unchecked_mut(j) = if v > 0.0 { v } else { 0.0 };
    }
}

/// NEON mean of an f32 slice.
///
/// # Safety
/// Caller must ensure this runs on aarch64 and slice is non-empty.
#[inline]
pub(crate) unsafe fn mean_f32_neon(a: &[f32]) -> f32 {
    if a.is_empty() {
        return 0.0;
    }
    sum_f32_neon(a) / a.len() as f32
}

/// NEON scalar broadcast add for f32: `out[i] = a[i] + s`.
#[inline]
pub(crate) unsafe fn add_scalar_f32_neon(a: &[f32], s: f32, out: &mut [f32]) {
    let n = a.len();
    let vs = vdupq_n_f32(s);
    let a_ptr = a.as_ptr();
    let o_ptr = out.as_mut_ptr();
    let chunks16 = n / 16;
    for i in 0..chunks16 {
        let base = i * 16;
        vst1q_f32(o_ptr.add(base), vaddq_f32(vld1q_f32(a_ptr.add(base)), vs));
        vst1q_f32(
            o_ptr.add(base + 4),
            vaddq_f32(vld1q_f32(a_ptr.add(base + 4)), vs),
        );
        vst1q_f32(
            o_ptr.add(base + 8),
            vaddq_f32(vld1q_f32(a_ptr.add(base + 8)), vs),
        );
        vst1q_f32(
            o_ptr.add(base + 12),
            vaddq_f32(vld1q_f32(a_ptr.add(base + 12)), vs),
        );
    }
    let tail = chunks16 * 16;
    for j in tail..n {
        *out.get_unchecked_mut(j) = *a.get_unchecked(j) + s;
    }
}

/// NEON scalar broadcast sub for f32: `out[i] = a[i] - s`.
#[inline]
pub(crate) unsafe fn sub_scalar_f32_neon(a: &[f32], s: f32, out: &mut [f32]) {
    let n = a.len();
    let vs = vdupq_n_f32(s);
    let a_ptr = a.as_ptr();
    let o_ptr = out.as_mut_ptr();
    let chunks16 = n / 16;
    for i in 0..chunks16 {
        let base = i * 16;
        vst1q_f32(o_ptr.add(base), vsubq_f32(vld1q_f32(a_ptr.add(base)), vs));
        vst1q_f32(
            o_ptr.add(base + 4),
            vsubq_f32(vld1q_f32(a_ptr.add(base + 4)), vs),
        );
        vst1q_f32(
            o_ptr.add(base + 8),
            vsubq_f32(vld1q_f32(a_ptr.add(base + 8)), vs),
        );
        vst1q_f32(
            o_ptr.add(base + 12),
            vsubq_f32(vld1q_f32(a_ptr.add(base + 12)), vs),
        );
    }
    let tail = chunks16 * 16;
    for j in tail..n {
        *out.get_unchecked_mut(j) = *a.get_unchecked(j) - s;
    }
}

/// NEON scalar broadcast mul for f32: `out[i] = a[i] * s`.
#[inline]
pub(crate) unsafe fn mul_scalar_f32_neon(a: &[f32], s: f32, out: &mut [f32]) {
    let n = a.len();
    let vs = vdupq_n_f32(s);
    let a_ptr = a.as_ptr();
    let o_ptr = out.as_mut_ptr();
    let chunks16 = n / 16;
    for i in 0..chunks16 {
        let base = i * 16;
        vst1q_f32(o_ptr.add(base), vmulq_f32(vld1q_f32(a_ptr.add(base)), vs));
        vst1q_f32(
            o_ptr.add(base + 4),
            vmulq_f32(vld1q_f32(a_ptr.add(base + 4)), vs),
        );
        vst1q_f32(
            o_ptr.add(base + 8),
            vmulq_f32(vld1q_f32(a_ptr.add(base + 8)), vs),
        );
        vst1q_f32(
            o_ptr.add(base + 12),
            vmulq_f32(vld1q_f32(a_ptr.add(base + 12)), vs),
        );
    }
    let tail = chunks16 * 16;
    for j in tail..n {
        *out.get_unchecked_mut(j) = *a.get_unchecked(j) * s;
    }
}

/// NEON scalar broadcast div for f32: `out[i] = a[i] / s`.
#[inline]
pub(crate) unsafe fn div_scalar_f32_neon(a: &[f32], s: f32, out: &mut [f32]) {
    let n = a.len();
    let vs = vdupq_n_f32(s);
    let a_ptr = a.as_ptr();
    let o_ptr = out.as_mut_ptr();
    let chunks16 = n / 16;
    for i in 0..chunks16 {
        let base = i * 16;
        vst1q_f32(o_ptr.add(base), vdivq_f32(vld1q_f32(a_ptr.add(base)), vs));
        vst1q_f32(
            o_ptr.add(base + 4),
            vdivq_f32(vld1q_f32(a_ptr.add(base + 4)), vs),
        );
        vst1q_f32(
            o_ptr.add(base + 8),
            vdivq_f32(vld1q_f32(a_ptr.add(base + 8)), vs),
        );
        vst1q_f32(
            o_ptr.add(base + 12),
            vdivq_f32(vld1q_f32(a_ptr.add(base + 12)), vs),
        );
    }
    let tail = chunks16 * 16;
    for j in tail..n {
        *out.get_unchecked_mut(j) = *a.get_unchecked(j) / s;
    }
}

/// NEON negate for f32: `out[i] = -a[i]`.
#[inline]
pub(crate) unsafe fn neg_f32_neon(a: &[f32], out: &mut [f32]) {
    let n = a.len();
    let a_ptr = a.as_ptr();
    let o_ptr = out.as_mut_ptr();
    let chunks16 = n / 16;
    for i in 0..chunks16 {
        let base = i * 16;
        vst1q_f32(o_ptr.add(base), vnegq_f32(vld1q_f32(a_ptr.add(base))));
        vst1q_f32(
            o_ptr.add(base + 4),
            vnegq_f32(vld1q_f32(a_ptr.add(base + 4))),
        );
        vst1q_f32(
            o_ptr.add(base + 8),
            vnegq_f32(vld1q_f32(a_ptr.add(base + 8))),
        );
        vst1q_f32(
            o_ptr.add(base + 12),
            vnegq_f32(vld1q_f32(a_ptr.add(base + 12))),
        );
    }
    let tail = chunks16 * 16;
    for j in tail..n {
        *out.get_unchecked_mut(j) = -*a.get_unchecked(j);
    }
}

// ---------------------------------------------------------------------------
// GEMM micro-kernels for f32
// ---------------------------------------------------------------------------

/// NEON 8x4 micro-kernel for f32 GEMM.
///
/// Computes `C[0..8, 0..4] += alpha * A[0..8, 0..kb] * B[0..kb, 0..4]`.
/// Each row of C is a single `float32x4_t` (4 f32 lanes), so 8 accumulator
/// registers cover the full 8×4 tile.
///
/// # Safety
/// Caller must ensure valid pointers and aarch64 target.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn gemm_8x4_f32_neon(
    a_ptr: *const f32,
    b_ptr: *const f32,
    c_ptr: *mut f32,
    alpha: f32,
    kb: usize,
    a_rs: usize, // packed A row stride
    b_rs: usize, // packed B row stride
    c_rs: usize, // C row stride
) {
    // 8 accumulator registers, one per row, each holding 4 columns.
    let mut c0 = vdupq_n_f32(0.0);
    let mut c1 = vdupq_n_f32(0.0);
    let mut c2 = vdupq_n_f32(0.0);
    let mut c3 = vdupq_n_f32(0.0);
    let mut c4 = vdupq_n_f32(0.0);
    let mut c5 = vdupq_n_f32(0.0);
    let mut c6 = vdupq_n_f32(0.0);
    let mut c7 = vdupq_n_f32(0.0);

    for p in 0..kb {
        let b_row = b_ptr.add(p * b_rs);
        let bv = vld1q_f32(b_row); // B[p, 0..4]

        c0 = vfmaq_f32(c0, vdupq_n_f32(*a_ptr.add(p)), bv);
        c1 = vfmaq_f32(c1, vdupq_n_f32(*a_ptr.add(a_rs + p)), bv);
        c2 = vfmaq_f32(c2, vdupq_n_f32(*a_ptr.add(2 * a_rs + p)), bv);
        c3 = vfmaq_f32(c3, vdupq_n_f32(*a_ptr.add(3 * a_rs + p)), bv);
        c4 = vfmaq_f32(c4, vdupq_n_f32(*a_ptr.add(4 * a_rs + p)), bv);
        c5 = vfmaq_f32(c5, vdupq_n_f32(*a_ptr.add(5 * a_rs + p)), bv);
        c6 = vfmaq_f32(c6, vdupq_n_f32(*a_ptr.add(6 * a_rs + p)), bv);
        c7 = vfmaq_f32(c7, vdupq_n_f32(*a_ptr.add(7 * a_rs + p)), bv);
    }

    let valpha = vdupq_n_f32(alpha);
    c0 = vmulq_f32(c0, valpha);
    c1 = vmulq_f32(c1, valpha);
    c2 = vmulq_f32(c2, valpha);
    c3 = vmulq_f32(c3, valpha);
    c4 = vmulq_f32(c4, valpha);
    c5 = vmulq_f32(c5, valpha);
    c6 = vmulq_f32(c6, valpha);
    c7 = vmulq_f32(c7, valpha);

    // Accumulate into C.
    let cp0 = c_ptr;
    let cp1 = c_ptr.add(c_rs);
    let cp2 = c_ptr.add(2 * c_rs);
    let cp3 = c_ptr.add(3 * c_rs);
    let cp4 = c_ptr.add(4 * c_rs);
    let cp5 = c_ptr.add(5 * c_rs);
    let cp6 = c_ptr.add(6 * c_rs);
    let cp7 = c_ptr.add(7 * c_rs);

    vst1q_f32(cp0, vaddq_f32(vld1q_f32(cp0), c0));
    vst1q_f32(cp1, vaddq_f32(vld1q_f32(cp1), c1));
    vst1q_f32(cp2, vaddq_f32(vld1q_f32(cp2), c2));
    vst1q_f32(cp3, vaddq_f32(vld1q_f32(cp3), c3));
    vst1q_f32(cp4, vaddq_f32(vld1q_f32(cp4), c4));
    vst1q_f32(cp5, vaddq_f32(vld1q_f32(cp5), c5));
    vst1q_f32(cp6, vaddq_f32(vld1q_f32(cp6), c6));
    vst1q_f32(cp7, vaddq_f32(vld1q_f32(cp7), c7));
}

/// NEON 4x4 micro-kernel for f32 GEMM.
///
/// Computes `C[0..4, 0..4] += alpha * A[0..4, 0..kb] * B[0..kb, 0..4]`.
///
/// # Safety
/// Caller must ensure valid pointers and aarch64 target.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn gemm_4x4_f32_neon(
    a_ptr: *const f32,
    b_ptr: *const f32,
    c_ptr: *mut f32,
    alpha: f32,
    kb: usize,
    a_rs: usize,
    b_rs: usize,
    c_rs: usize,
) {
    let mut c0 = vdupq_n_f32(0.0);
    let mut c1 = vdupq_n_f32(0.0);
    let mut c2 = vdupq_n_f32(0.0);
    let mut c3 = vdupq_n_f32(0.0);

    for p in 0..kb {
        let bv = vld1q_f32(b_ptr.add(p * b_rs));
        c0 = vfmaq_f32(c0, vdupq_n_f32(*a_ptr.add(p)), bv);
        c1 = vfmaq_f32(c1, vdupq_n_f32(*a_ptr.add(a_rs + p)), bv);
        c2 = vfmaq_f32(c2, vdupq_n_f32(*a_ptr.add(2 * a_rs + p)), bv);
        c3 = vfmaq_f32(c3, vdupq_n_f32(*a_ptr.add(3 * a_rs + p)), bv);
    }

    let valpha = vdupq_n_f32(alpha);
    c0 = vmulq_f32(c0, valpha);
    c1 = vmulq_f32(c1, valpha);
    c2 = vmulq_f32(c2, valpha);
    c3 = vmulq_f32(c3, valpha);

    let cp0 = c_ptr;
    let cp1 = c_ptr.add(c_rs);
    let cp2 = c_ptr.add(2 * c_rs);
    let cp3 = c_ptr.add(3 * c_rs);

    vst1q_f32(cp0, vaddq_f32(vld1q_f32(cp0), c0));
    vst1q_f32(cp1, vaddq_f32(vld1q_f32(cp1), c1));
    vst1q_f32(cp2, vaddq_f32(vld1q_f32(cp2), c2));
    vst1q_f32(cp3, vaddq_f32(vld1q_f32(cp3), c3));
}
