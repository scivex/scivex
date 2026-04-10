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
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    // 4 accumulators × 2 lanes = 8 f64 FMAs per unrolled iteration.
    let mut acc0 = vdupq_n_f64(0.0);
    let mut acc1 = vdupq_n_f64(0.0);
    let mut acc2 = vdupq_n_f64(0.0);
    let mut acc3 = vdupq_n_f64(0.0);

    let chunks8 = n / 8;
    for i in 0..chunks8 {
        let base = i * 8;
        acc0 = vfmaq_f64(acc0, vld1q_f64(a_ptr.add(base)), vld1q_f64(b_ptr.add(base)));
        acc1 = vfmaq_f64(acc1, vld1q_f64(a_ptr.add(base + 2)), vld1q_f64(b_ptr.add(base + 2)));
        acc2 = vfmaq_f64(acc2, vld1q_f64(a_ptr.add(base + 4)), vld1q_f64(b_ptr.add(base + 4)));
        acc3 = vfmaq_f64(acc3, vld1q_f64(a_ptr.add(base + 6)), vld1q_f64(b_ptr.add(base + 6)));
    }

    acc0 = vaddq_f64(acc0, acc1);
    acc2 = vaddq_f64(acc2, acc3);
    acc0 = vaddq_f64(acc0, acc2);
    let mut result = hsum_f64x2(acc0);

    let tail = chunks8 * 8;
    for j in tail..n {
        result += *a.get_unchecked(j) * *b.get_unchecked(j);
    }
    result
}

/// NEON sum of an f64 slice using 4 independent accumulators to hide ADD latency.
///
/// # Safety
/// Caller must ensure this runs on aarch64.
#[inline]
pub(crate) unsafe fn sum_f64_neon(a: &[f64]) -> f64 {
    let n = a.len();
    let ptr = a.as_ptr();

    // 4 accumulators × 2 lanes = 8 f64s per unrolled iteration.
    let mut acc0 = vdupq_n_f64(0.0);
    let mut acc1 = vdupq_n_f64(0.0);
    let mut acc2 = vdupq_n_f64(0.0);
    let mut acc3 = vdupq_n_f64(0.0);

    let chunks8 = n / 8;
    for i in 0..chunks8 {
        let base = i * 8;
        acc0 = vaddq_f64(acc0, vld1q_f64(ptr.add(base)));
        acc1 = vaddq_f64(acc1, vld1q_f64(ptr.add(base + 2)));
        acc2 = vaddq_f64(acc2, vld1q_f64(ptr.add(base + 4)));
        acc3 = vaddq_f64(acc3, vld1q_f64(ptr.add(base + 6)));
    }

    // Reduce accumulators.
    acc0 = vaddq_f64(acc0, acc1);
    acc2 = vaddq_f64(acc2, acc3);
    acc0 = vaddq_f64(acc0, acc2);
    let mut result = hsum_f64x2(acc0);

    // Handle remaining elements.
    let tail = chunks8 * 8;
    for j in tail..n {
        result += *a.get_unchecked(j);
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
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let o_ptr = out.as_mut_ptr();

    // 4-way unroll: 8 f64s per iteration.
    let chunks8 = n / 8;
    for i in 0..chunks8 {
        let base = i * 8;
        vst1q_f64(o_ptr.add(base), vaddq_f64(vld1q_f64(a_ptr.add(base)), vld1q_f64(b_ptr.add(base))));
        vst1q_f64(o_ptr.add(base + 2), vaddq_f64(vld1q_f64(a_ptr.add(base + 2)), vld1q_f64(b_ptr.add(base + 2))));
        vst1q_f64(o_ptr.add(base + 4), vaddq_f64(vld1q_f64(a_ptr.add(base + 4)), vld1q_f64(b_ptr.add(base + 4))));
        vst1q_f64(o_ptr.add(base + 6), vaddq_f64(vld1q_f64(a_ptr.add(base + 6)), vld1q_f64(b_ptr.add(base + 6))));
    }

    let tail = chunks8 * 8;
    for j in tail..n {
        *out.get_unchecked_mut(j) = *a.get_unchecked(j) + *b.get_unchecked(j);
    }
}

/// NEON element-wise sub for f64.
///
/// # Safety
/// Caller must ensure this runs on aarch64 and slices have equal length.
#[inline]
pub(crate) unsafe fn sub_f64_neon(a: &[f64], b: &[f64], out: &mut [f64]) {
    let n = a.len();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let o_ptr = out.as_mut_ptr();

    let chunks8 = n / 8;
    for i in 0..chunks8 {
        let base = i * 8;
        vst1q_f64(o_ptr.add(base), vsubq_f64(vld1q_f64(a_ptr.add(base)), vld1q_f64(b_ptr.add(base))));
        vst1q_f64(o_ptr.add(base + 2), vsubq_f64(vld1q_f64(a_ptr.add(base + 2)), vld1q_f64(b_ptr.add(base + 2))));
        vst1q_f64(o_ptr.add(base + 4), vsubq_f64(vld1q_f64(a_ptr.add(base + 4)), vld1q_f64(b_ptr.add(base + 4))));
        vst1q_f64(o_ptr.add(base + 6), vsubq_f64(vld1q_f64(a_ptr.add(base + 6)), vld1q_f64(b_ptr.add(base + 6))));
    }

    let tail = chunks8 * 8;
    for j in tail..n {
        *out.get_unchecked_mut(j) = *a.get_unchecked(j) - *b.get_unchecked(j);
    }
}

/// NEON element-wise mul for f64.
///
/// # Safety
/// Caller must ensure this runs on aarch64 and slices have equal length.
#[inline]
pub(crate) unsafe fn mul_f64_neon(a: &[f64], b: &[f64], out: &mut [f64]) {
    let n = a.len();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let o_ptr = out.as_mut_ptr();

    let chunks8 = n / 8;
    for i in 0..chunks8 {
        let base = i * 8;
        vst1q_f64(o_ptr.add(base), vmulq_f64(vld1q_f64(a_ptr.add(base)), vld1q_f64(b_ptr.add(base))));
        vst1q_f64(o_ptr.add(base + 2), vmulq_f64(vld1q_f64(a_ptr.add(base + 2)), vld1q_f64(b_ptr.add(base + 2))));
        vst1q_f64(o_ptr.add(base + 4), vmulq_f64(vld1q_f64(a_ptr.add(base + 4)), vld1q_f64(b_ptr.add(base + 4))));
        vst1q_f64(o_ptr.add(base + 6), vmulq_f64(vld1q_f64(a_ptr.add(base + 6)), vld1q_f64(b_ptr.add(base + 6))));
    }

    let tail = chunks8 * 8;
    for j in tail..n {
        *out.get_unchecked_mut(j) = *a.get_unchecked(j) * *b.get_unchecked(j);
    }
}

/// NEON element-wise div for f64.
///
/// # Safety
/// Caller must ensure this runs on aarch64 and slices have equal length.
#[inline]
pub(crate) unsafe fn div_f64_neon(a: &[f64], b: &[f64], out: &mut [f64]) {
    let n = a.len();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let o_ptr = out.as_mut_ptr();

    let chunks8 = n / 8;
    for i in 0..chunks8 {
        let base = i * 8;
        vst1q_f64(o_ptr.add(base), vdivq_f64(vld1q_f64(a_ptr.add(base)), vld1q_f64(b_ptr.add(base))));
        vst1q_f64(o_ptr.add(base + 2), vdivq_f64(vld1q_f64(a_ptr.add(base + 2)), vld1q_f64(b_ptr.add(base + 2))));
        vst1q_f64(o_ptr.add(base + 4), vdivq_f64(vld1q_f64(a_ptr.add(base + 4)), vld1q_f64(b_ptr.add(base + 4))));
        vst1q_f64(o_ptr.add(base + 6), vdivq_f64(vld1q_f64(a_ptr.add(base + 6)), vld1q_f64(b_ptr.add(base + 6))));
    }

    let tail = chunks8 * 8;
    for j in tail..n {
        *out.get_unchecked_mut(j) = *a.get_unchecked(j) / *b.get_unchecked(j);
    }
}

/// NEON axpy for f64: `y[i] += alpha * x[i]`.
///
/// # Safety
/// Caller must ensure this runs on aarch64 and slices have equal length.
#[inline]
pub(crate) unsafe fn axpy_f64_neon(alpha: f64, x: &[f64], y: &mut [f64]) {
    let n = x.len();
    let valpha = vdupq_n_f64(alpha);
    let x_ptr = x.as_ptr();
    let y_ptr = y.as_mut_ptr();

    let chunks8 = n / 8;
    for i in 0..chunks8 {
        let base = i * 8;
        let vy0 = vld1q_f64(y_ptr.add(base));
        let vy1 = vld1q_f64(y_ptr.add(base + 2));
        let vy2 = vld1q_f64(y_ptr.add(base + 4));
        let vy3 = vld1q_f64(y_ptr.add(base + 6));
        vst1q_f64(y_ptr.add(base), vfmaq_f64(vy0, valpha, vld1q_f64(x_ptr.add(base))));
        vst1q_f64(y_ptr.add(base + 2), vfmaq_f64(vy1, valpha, vld1q_f64(x_ptr.add(base + 2))));
        vst1q_f64(y_ptr.add(base + 4), vfmaq_f64(vy2, valpha, vld1q_f64(x_ptr.add(base + 4))));
        vst1q_f64(y_ptr.add(base + 6), vfmaq_f64(vy3, valpha, vld1q_f64(x_ptr.add(base + 6))));
    }

    let tail = chunks8 * 8;
    for j in tail..n {
        *y.get_unchecked_mut(j) += alpha * *x.get_unchecked(j);
    }
}

/// NEON scal for f64: `x[i] *= alpha`.
///
/// # Safety
/// Caller must ensure this runs on aarch64.
#[inline]
pub(crate) unsafe fn scal_f64_neon(alpha: f64, x: &mut [f64]) {
    let n = x.len();
    let valpha = vdupq_n_f64(alpha);
    let ptr = x.as_mut_ptr();

    let chunks8 = n / 8;
    for i in 0..chunks8 {
        let base = i * 8;
        vst1q_f64(ptr.add(base), vmulq_f64(valpha, vld1q_f64(ptr.add(base))));
        vst1q_f64(ptr.add(base + 2), vmulq_f64(valpha, vld1q_f64(ptr.add(base + 2))));
        vst1q_f64(ptr.add(base + 4), vmulq_f64(valpha, vld1q_f64(ptr.add(base + 4))));
        vst1q_f64(ptr.add(base + 6), vmulq_f64(valpha, vld1q_f64(ptr.add(base + 6))));
    }

    let tail = chunks8 * 8;
    for j in tail..n {
        *x.get_unchecked_mut(j) *= alpha;
    }
}

/// NEON sum of squares for f64.
///
/// # Safety
/// Caller must ensure this runs on aarch64.
#[inline]
pub(crate) unsafe fn sum_sq_f64_neon(a: &[f64]) -> f64 {
    let n = a.len();
    let ptr = a.as_ptr();

    // 4 accumulators × 2 lanes = 8 f64s per unrolled iteration.
    let mut acc0 = vdupq_n_f64(0.0);
    let mut acc1 = vdupq_n_f64(0.0);
    let mut acc2 = vdupq_n_f64(0.0);
    let mut acc3 = vdupq_n_f64(0.0);

    let chunks8 = n / 8;
    for i in 0..chunks8 {
        let base = i * 8;
        let v0 = vld1q_f64(ptr.add(base));
        let v1 = vld1q_f64(ptr.add(base + 2));
        let v2 = vld1q_f64(ptr.add(base + 4));
        let v3 = vld1q_f64(ptr.add(base + 6));
        acc0 = vfmaq_f64(acc0, v0, v0);
        acc1 = vfmaq_f64(acc1, v1, v1);
        acc2 = vfmaq_f64(acc2, v2, v2);
        acc3 = vfmaq_f64(acc3, v3, v3);
    }

    acc0 = vaddq_f64(acc0, acc1);
    acc2 = vaddq_f64(acc2, acc3);
    acc0 = vaddq_f64(acc0, acc2);
    let mut result = hsum_f64x2(acc0);

    let tail = chunks8 * 8;
    for j in tail..n {
        let v = *a.get_unchecked(j);
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
    let ptr = a.as_ptr();

    let mut acc0 = vdupq_n_f64(0.0);
    let mut acc1 = vdupq_n_f64(0.0);
    let mut acc2 = vdupq_n_f64(0.0);
    let mut acc3 = vdupq_n_f64(0.0);

    let chunks8 = n / 8;
    for i in 0..chunks8 {
        let base = i * 8;
        acc0 = vaddq_f64(acc0, vabsq_f64(vld1q_f64(ptr.add(base))));
        acc1 = vaddq_f64(acc1, vabsq_f64(vld1q_f64(ptr.add(base + 2))));
        acc2 = vaddq_f64(acc2, vabsq_f64(vld1q_f64(ptr.add(base + 4))));
        acc3 = vaddq_f64(acc3, vabsq_f64(vld1q_f64(ptr.add(base + 6))));
    }

    acc0 = vaddq_f64(acc0, acc1);
    acc2 = vaddq_f64(acc2, acc3);
    acc0 = vaddq_f64(acc0, acc2);
    let mut result = hsum_f64x2(acc0);

    let tail = chunks8 * 8;
    for j in tail..n {
        result += (*a.get_unchecked(j)).abs();
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
    let ptr = a.as_ptr();

    let mut vmin0 = vdupq_n_f64(f64::INFINITY);
    let mut vmin1 = vdupq_n_f64(f64::INFINITY);
    let mut vmin2 = vdupq_n_f64(f64::INFINITY);
    let mut vmin3 = vdupq_n_f64(f64::INFINITY);

    let chunks8 = n / 8;
    for i in 0..chunks8 {
        let base = i * 8;
        vmin0 = vminq_f64(vmin0, vld1q_f64(ptr.add(base)));
        vmin1 = vminq_f64(vmin1, vld1q_f64(ptr.add(base + 2)));
        vmin2 = vminq_f64(vmin2, vld1q_f64(ptr.add(base + 4)));
        vmin3 = vminq_f64(vmin3, vld1q_f64(ptr.add(base + 6)));
    }

    vmin0 = vminq_f64(vmin0, vmin1);
    vmin2 = vminq_f64(vmin2, vmin3);
    vmin0 = vminq_f64(vmin0, vmin2);
    let mut result = vminvq_f64(vmin0);

    let tail = chunks8 * 8;
    for j in tail..n {
        result = result.min(*a.get_unchecked(j));
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
    let ptr = a.as_ptr();

    let mut vmax0 = vdupq_n_f64(f64::NEG_INFINITY);
    let mut vmax1 = vdupq_n_f64(f64::NEG_INFINITY);
    let mut vmax2 = vdupq_n_f64(f64::NEG_INFINITY);
    let mut vmax3 = vdupq_n_f64(f64::NEG_INFINITY);

    let chunks8 = n / 8;
    for i in 0..chunks8 {
        let base = i * 8;
        vmax0 = vmaxq_f64(vmax0, vld1q_f64(ptr.add(base)));
        vmax1 = vmaxq_f64(vmax1, vld1q_f64(ptr.add(base + 2)));
        vmax2 = vmaxq_f64(vmax2, vld1q_f64(ptr.add(base + 4)));
        vmax3 = vmaxq_f64(vmax3, vld1q_f64(ptr.add(base + 6)));
    }

    vmax0 = vmaxq_f64(vmax0, vmax1);
    vmax2 = vmaxq_f64(vmax2, vmax3);
    vmax0 = vmaxq_f64(vmax0, vmax2);
    let mut result = vmaxvq_f64(vmax0);

    let tail = chunks8 * 8;
    for j in tail..n {
        result = result.max(*a.get_unchecked(j));
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

/// NEON ReLU (max(0, x)) for f64.
///
/// # Safety
/// Caller must ensure this runs on aarch64 and slices have equal length.
#[inline]
pub(crate) unsafe fn relu_f64_neon(a: &[f64], out: &mut [f64]) {
    let n = a.len();
    let a_ptr = a.as_ptr();
    let o_ptr = out.as_mut_ptr();
    let vzero = vdupq_n_f64(0.0);

    let chunks8 = n / 8;
    for i in 0..chunks8 {
        let base = i * 8;
        vst1q_f64(o_ptr.add(base), vmaxq_f64(vzero, vld1q_f64(a_ptr.add(base))));
        vst1q_f64(o_ptr.add(base + 2), vmaxq_f64(vzero, vld1q_f64(a_ptr.add(base + 2))));
        vst1q_f64(o_ptr.add(base + 4), vmaxq_f64(vzero, vld1q_f64(a_ptr.add(base + 4))));
        vst1q_f64(o_ptr.add(base + 6), vmaxq_f64(vzero, vld1q_f64(a_ptr.add(base + 6))));
    }

    let tail = chunks8 * 8;
    for j in tail..n {
        let v = *a.get_unchecked(j);
        *out.get_unchecked_mut(j) = if v > 0.0 { v } else { 0.0 };
    }
}

/// NEON 4x4 micro-kernel for GEMM: C[4x4] += alpha * A_panel[4xK] * B_panel[Kx4].
///
/// Computes a 4x4 block of C by accumulating K rank-1 updates using NEON FMA.
/// Uses 4 register pairs (8 float64x2_t accumulators) to keep the 4x4 tile in
/// registers throughout the K-loop.
///
/// # Parameters
/// - `a_ptr`: pointer to A sub-panel, row-major, stride `k` (i.e., a[i][p] = a_ptr[i*a_rs + p])
/// - `b_ptr`: pointer to B sub-panel, row-major, stride `n` (i.e., b[p][j] = b_ptr[p*b_rs + j])
/// - `c_ptr`: pointer to C sub-block, row-major, stride `c_rs`
/// - `alpha`: scalar multiplier
/// - `kb`: number of K-dimension iterations
/// - `a_rs`: row stride of A (number of columns in full A, = k)
/// - `b_rs`: row stride of B (number of columns in full B, = n)
/// - `c_rs`: row stride of C (number of columns in full C, = n)
///
/// # Safety
/// - Caller must ensure aarch64 with NEON.
/// - All pointers must be valid for the described access patterns.
/// - The 4x4 sub-block of C must be within bounds.
#[inline]
pub(crate) unsafe fn gemm_4x4_f64_neon(
    a_ptr: *const f64,
    b_ptr: *const f64,
    c_ptr: *mut f64,
    alpha: f64,
    kb: usize,
    a_rs: usize,
    b_rs: usize,
    c_rs: usize,
) {
    // 4x4 accumulator: c[i][j] stored as 4 pairs of float64x2_t (low 2, high 2)
    let mut c00 = vdupq_n_f64(0.0);
    let mut c01 = vdupq_n_f64(0.0);
    let mut c10 = vdupq_n_f64(0.0);
    let mut c11 = vdupq_n_f64(0.0);
    let mut c20 = vdupq_n_f64(0.0);
    let mut c21 = vdupq_n_f64(0.0);
    let mut c30 = vdupq_n_f64(0.0);
    let mut c31 = vdupq_n_f64(0.0);

    for p in 0..kb {
        let b_row = b_ptr.add(p * b_rs);
        let b0 = vld1q_f64(b_row);
        let b1 = vld1q_f64(b_row.add(2));

        let a0 = vdupq_n_f64(*a_ptr.add(0 * a_rs + p));
        c00 = vfmaq_f64(c00, a0, b0);
        c01 = vfmaq_f64(c01, a0, b1);

        let a1 = vdupq_n_f64(*a_ptr.add(1 * a_rs + p));
        c10 = vfmaq_f64(c10, a1, b0);
        c11 = vfmaq_f64(c11, a1, b1);

        let a2 = vdupq_n_f64(*a_ptr.add(2 * a_rs + p));
        c20 = vfmaq_f64(c20, a2, b0);
        c21 = vfmaq_f64(c21, a2, b1);

        let a3 = vdupq_n_f64(*a_ptr.add(3 * a_rs + p));
        c30 = vfmaq_f64(c30, a3, b0);
        c31 = vfmaq_f64(c31, a3, b1);
    }

    let valpha = vdupq_n_f64(alpha);
    c00 = vmulq_f64(c00, valpha);
    c01 = vmulq_f64(c01, valpha);
    c10 = vmulq_f64(c10, valpha);
    c11 = vmulq_f64(c11, valpha);
    c20 = vmulq_f64(c20, valpha);
    c21 = vmulq_f64(c21, valpha);
    c30 = vmulq_f64(c30, valpha);
    c31 = vmulq_f64(c31, valpha);

    let c0 = c_ptr;
    let c1 = c_ptr.add(c_rs);
    let c2 = c_ptr.add(2 * c_rs);
    let c3 = c_ptr.add(3 * c_rs);

    vst1q_f64(c0, vaddq_f64(vld1q_f64(c0), c00));
    vst1q_f64(c0.add(2), vaddq_f64(vld1q_f64(c0.add(2)), c01));
    vst1q_f64(c1, vaddq_f64(vld1q_f64(c1), c10));
    vst1q_f64(c1.add(2), vaddq_f64(vld1q_f64(c1.add(2)), c11));
    vst1q_f64(c2, vaddq_f64(vld1q_f64(c2), c20));
    vst1q_f64(c2.add(2), vaddq_f64(vld1q_f64(c2.add(2)), c21));
    vst1q_f64(c3, vaddq_f64(vld1q_f64(c3), c30));
    vst1q_f64(c3.add(2), vaddq_f64(vld1q_f64(c3.add(2)), c31));
}

/// NEON 8x4 micro-kernel for f64 GEMM (strided B, no packing).
///
/// Computes C[0..8, 0..4] += alpha * A[0..8, 0..kb] * B[0..kb, 0..4]
/// Uses 16 accumulator registers (8 rows × 2 float64x2_t).
///
/// # Safety
/// Caller must ensure valid pointers and aarch64 target.
#[inline]
pub(crate) unsafe fn gemm_8x4_f64_neon(
    a_ptr: *const f64,
    b_ptr: *const f64,
    c_ptr: *mut f64,
    alpha: f64,
    kb: usize,
    a_rs: usize,
    b_rs: usize,
    c_rs: usize,
) {
    let mut c00 = vdupq_n_f64(0.0); let mut c01 = vdupq_n_f64(0.0);
    let mut c10 = vdupq_n_f64(0.0); let mut c11 = vdupq_n_f64(0.0);
    let mut c20 = vdupq_n_f64(0.0); let mut c21 = vdupq_n_f64(0.0);
    let mut c30 = vdupq_n_f64(0.0); let mut c31 = vdupq_n_f64(0.0);
    let mut c40 = vdupq_n_f64(0.0); let mut c41 = vdupq_n_f64(0.0);
    let mut c50 = vdupq_n_f64(0.0); let mut c51 = vdupq_n_f64(0.0);
    let mut c60 = vdupq_n_f64(0.0); let mut c61 = vdupq_n_f64(0.0);
    let mut c70 = vdupq_n_f64(0.0); let mut c71 = vdupq_n_f64(0.0);

    for p in 0..kb {
        let b_row = b_ptr.add(p * b_rs);
        let b0 = vld1q_f64(b_row);
        let b1 = vld1q_f64(b_row.add(2));

        let a0 = vdupq_n_f64(*a_ptr.add(0 * a_rs + p));
        c00 = vfmaq_f64(c00, a0, b0); c01 = vfmaq_f64(c01, a0, b1);
        let a1 = vdupq_n_f64(*a_ptr.add(1 * a_rs + p));
        c10 = vfmaq_f64(c10, a1, b0); c11 = vfmaq_f64(c11, a1, b1);
        let a2 = vdupq_n_f64(*a_ptr.add(2 * a_rs + p));
        c20 = vfmaq_f64(c20, a2, b0); c21 = vfmaq_f64(c21, a2, b1);
        let a3 = vdupq_n_f64(*a_ptr.add(3 * a_rs + p));
        c30 = vfmaq_f64(c30, a3, b0); c31 = vfmaq_f64(c31, a3, b1);
        let a4 = vdupq_n_f64(*a_ptr.add(4 * a_rs + p));
        c40 = vfmaq_f64(c40, a4, b0); c41 = vfmaq_f64(c41, a4, b1);
        let a5 = vdupq_n_f64(*a_ptr.add(5 * a_rs + p));
        c50 = vfmaq_f64(c50, a5, b0); c51 = vfmaq_f64(c51, a5, b1);
        let a6 = vdupq_n_f64(*a_ptr.add(6 * a_rs + p));
        c60 = vfmaq_f64(c60, a6, b0); c61 = vfmaq_f64(c61, a6, b1);
        let a7 = vdupq_n_f64(*a_ptr.add(7 * a_rs + p));
        c70 = vfmaq_f64(c70, a7, b0); c71 = vfmaq_f64(c71, a7, b1);
    }

    let valpha = vdupq_n_f64(alpha);

    macro_rules! store_row {
        ($row:expr, $lo:ident, $hi:ident) => {{
            let cp = c_ptr.add($row * c_rs);
            vst1q_f64(cp, vaddq_f64(vld1q_f64(cp), vmulq_f64($lo, valpha)));
            vst1q_f64(cp.add(2), vaddq_f64(vld1q_f64(cp.add(2)), vmulq_f64($hi, valpha)));
        }};
    }
    store_row!(0, c00, c01);
    store_row!(1, c10, c11);
    store_row!(2, c20, c21);
    store_row!(3, c30, c31);
    store_row!(4, c40, c41);
    store_row!(5, c50, c51);
    store_row!(6, c60, c61);
    store_row!(7, c70, c71);
}
