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
    let mut c00 = vdupq_n_f64(0.0); // C[0][0..2]
    let mut c01 = vdupq_n_f64(0.0); // C[0][2..4]
    let mut c10 = vdupq_n_f64(0.0); // C[1][0..2]
    let mut c11 = vdupq_n_f64(0.0); // C[1][2..4]
    let mut c20 = vdupq_n_f64(0.0); // C[2][0..2]
    let mut c21 = vdupq_n_f64(0.0); // C[2][2..4]
    let mut c30 = vdupq_n_f64(0.0); // C[3][0..2]
    let mut c31 = vdupq_n_f64(0.0); // C[3][2..4]

    for p in 0..kb {
        // Load B[p][0..4] as two float64x2_t
        let b_row = b_ptr.add(p * b_rs);
        let b0 = vld1q_f64(b_row);
        let b1 = vld1q_f64(b_row.add(2));

        // Broadcast A[i][p] and FMA into C[i][0..4]
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

    // Scale by alpha
    let valpha = vdupq_n_f64(alpha);
    c00 = vmulq_f64(c00, valpha);
    c01 = vmulq_f64(c01, valpha);
    c10 = vmulq_f64(c10, valpha);
    c11 = vmulq_f64(c11, valpha);
    c20 = vmulq_f64(c20, valpha);
    c21 = vmulq_f64(c21, valpha);
    c30 = vmulq_f64(c30, valpha);
    c31 = vmulq_f64(c31, valpha);

    // Accumulate into C (C += micro-result)
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
