//! AVX-accelerated f64 kernels (4-wide `__m256d`).

// ---------------------------------------------------------------------------
// Scalar fallbacks (used on non-x86_64 or when AVX is unavailable)
// ---------------------------------------------------------------------------

/// Scalar dot product.
pub(crate) fn dot_f64_scalar(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .fold(0.0, |acc, (&x, &y)| acc + x * y)
}

/// Scalar sum.
pub(crate) fn sum_f64_scalar(a: &[f64]) -> f64 {
    a.iter().copied().sum()
}

/// Scalar element-wise add: `out[i] = a[i] + b[i]`.
pub(crate) fn add_f64_scalar(a: &[f64], b: &[f64], out: &mut [f64]) {
    for i in 0..a.len() {
        out[i] = a[i] + b[i];
    }
}

/// Scalar element-wise sub: `out[i] = a[i] - b[i]`.
pub(crate) fn sub_f64_scalar(a: &[f64], b: &[f64], out: &mut [f64]) {
    for i in 0..a.len() {
        out[i] = a[i] - b[i];
    }
}

/// Scalar element-wise mul: `out[i] = a[i] * b[i]`.
pub(crate) fn mul_f64_scalar(a: &[f64], b: &[f64], out: &mut [f64]) {
    for i in 0..a.len() {
        out[i] = a[i] * b[i];
    }
}

/// Scalar element-wise div: `out[i] = a[i] / b[i]`.
pub(crate) fn div_f64_scalar(a: &[f64], b: &[f64], out: &mut [f64]) {
    for i in 0..a.len() {
        out[i] = a[i] / b[i];
    }
}

/// Scalar axpy: `y[i] += alpha * x[i]`.
pub(crate) fn axpy_f64_scalar(alpha: f64, x: &[f64], y: &mut [f64]) {
    for i in 0..x.len() {
        y[i] = alpha.mul_add(x[i], y[i]);
    }
}

/// Scalar scal: `x[i] *= alpha`.
pub(crate) fn scal_f64_scalar(alpha: f64, x: &mut [f64]) {
    for v in x.iter_mut() {
        *v *= alpha;
    }
}

/// Scalar sum of squares (for nrm2).
pub(crate) fn sum_sq_f64_scalar(a: &[f64]) -> f64 {
    a.iter().fold(0.0, |acc, &v| v.mul_add(v, acc))
}

/// Scalar sum of absolute values (for asum).
pub(crate) fn asum_f64_scalar(a: &[f64]) -> f64 {
    a.iter().fold(0.0, |acc, &v| acc + v.abs())
}

/// Scalar min.
pub(crate) fn min_f64_scalar(a: &[f64]) -> f64 {
    a.iter().copied().fold(f64::INFINITY, f64::min)
}

/// Scalar max.
pub(crate) fn max_f64_scalar(a: &[f64]) -> f64 {
    a.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}

/// Scalar mean.
pub(crate) fn mean_f64_scalar(a: &[f64]) -> f64 {
    if a.is_empty() {
        return 0.0;
    }
    sum_f64_scalar(a) / a.len() as f64
}

/// Scalar broadcast add: `out[i] = a[i] + s`.
pub(crate) fn add_scalar_f64_scalar(a: &[f64], s: f64, out: &mut [f64]) {
    for i in 0..a.len() {
        out[i] = a[i] + s;
    }
}

/// Scalar broadcast sub: `out[i] = a[i] - s`.
pub(crate) fn sub_scalar_f64_scalar(a: &[f64], s: f64, out: &mut [f64]) {
    for i in 0..a.len() {
        out[i] = a[i] - s;
    }
}

/// Scalar broadcast mul: `out[i] = a[i] * s`.
pub(crate) fn mul_scalar_f64_scalar(a: &[f64], s: f64, out: &mut [f64]) {
    for i in 0..a.len() {
        out[i] = a[i] * s;
    }
}

/// Scalar broadcast div: `out[i] = a[i] / s`.
pub(crate) fn div_scalar_f64_scalar(a: &[f64], s: f64, out: &mut [f64]) {
    for i in 0..a.len() {
        out[i] = a[i] / s;
    }
}

/// Scalar negate: `out[i] = -a[i]`.
pub(crate) fn neg_f64_scalar(a: &[f64], out: &mut [f64]) {
    for i in 0..a.len() {
        out[i] = -a[i];
    }
}

/// Scalar ReLU: `out[i] = max(0, a[i])`.
pub(crate) fn relu_f64_scalar(a: &[f64], out: &mut [f64]) {
    for i in 0..a.len() {
        out[i] = if a[i] > 0.0 { a[i] } else { 0.0 };
    }
}

// ---------------------------------------------------------------------------
// AVX implementations (x86_64 only)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
mod avx {
    use core::arch::x86_64::*;

    /// Horizontal sum of a 256-bit f64 register (4 lanes).
    ///
    /// # Safety
    /// Caller must ensure AVX is available.
    #[inline(always)]
    unsafe fn hsum_256_f64(v: __m256d) -> f64 {
        // SAFETY: AVX guaranteed by caller.
        let hi = _mm256_extractf128_pd(v, 1); // upper 128
        let lo = _mm256_castpd256_pd128(v); // lower 128
        let sum128 = _mm_add_pd(lo, hi);
        let hi64 = _mm_unpackhi_pd(sum128, sum128);
        let result = _mm_add_sd(sum128, hi64);
        _mm_cvtsd_f64(result)
    }

    /// AVX dot product of two f64 slices.
    ///
    /// # Safety
    /// Caller must ensure AVX is available.
    #[target_feature(enable = "avx")]
    pub(crate) unsafe fn dot_f64_avx(a: &[f64], b: &[f64]) -> f64 {
        let n = a.len();
        let chunks = n / 4;
        let remainder = n % 4;

        // SAFETY: AVX is available (caller checked). Pointer arithmetic stays
        // within `a` and `b` bounds because `i * 4 + 3 < chunks * 4 <= n`.
        let mut acc = _mm256_setzero_pd();
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        for i in 0..chunks {
            let va = _mm256_loadu_pd(a_ptr.add(i * 4));
            let vb = _mm256_loadu_pd(b_ptr.add(i * 4));
            acc = _mm256_add_pd(acc, _mm256_mul_pd(va, vb));
        }

        let mut result = hsum_256_f64(acc);

        let tail = chunks * 4;
        for j in 0..remainder {
            result += a[tail + j] * b[tail + j];
        }
        result
    }

    /// AVX sum of an f64 slice.
    ///
    /// # Safety
    /// Caller must ensure AVX is available.
    #[target_feature(enable = "avx")]
    pub(crate) unsafe fn sum_f64_avx(a: &[f64]) -> f64 {
        let n = a.len();
        let chunks = n / 4;
        let remainder = n % 4;

        // SAFETY: AVX available, pointer stays in bounds.
        let mut acc = _mm256_setzero_pd();
        let ptr = a.as_ptr();

        for i in 0..chunks {
            let va = _mm256_loadu_pd(ptr.add(i * 4));
            acc = _mm256_add_pd(acc, va);
        }

        let mut result = hsum_256_f64(acc);

        let tail = chunks * 4;
        for j in 0..remainder {
            result += a[tail + j];
        }
        result
    }

    /// AVX element-wise add.
    ///
    /// # Safety
    /// Caller must ensure AVX is available and all slices have equal length.
    #[target_feature(enable = "avx")]
    pub(crate) unsafe fn add_f64_avx(a: &[f64], b: &[f64], out: &mut [f64]) {
        let n = a.len();
        let chunks = n / 4;
        let remainder = n % 4;

        // SAFETY: AVX available, all pointers within bounds.
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let o_ptr = out.as_mut_ptr();

        for i in 0..chunks {
            let off = i * 4;
            let va = _mm256_loadu_pd(a_ptr.add(off));
            let vb = _mm256_loadu_pd(b_ptr.add(off));
            _mm256_storeu_pd(o_ptr.add(off), _mm256_add_pd(va, vb));
        }

        let tail = chunks * 4;
        for j in 0..remainder {
            *out.get_unchecked_mut(tail + j) =
                *a.get_unchecked(tail + j) + *b.get_unchecked(tail + j);
        }
    }

    /// AVX element-wise sub.
    ///
    /// # Safety
    /// Caller must ensure AVX is available and all slices have equal length.
    #[target_feature(enable = "avx")]
    pub(crate) unsafe fn sub_f64_avx(a: &[f64], b: &[f64], out: &mut [f64]) {
        let n = a.len();
        let chunks = n / 4;
        let remainder = n % 4;

        // SAFETY: AVX available, all pointers within bounds.
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let o_ptr = out.as_mut_ptr();

        for i in 0..chunks {
            let off = i * 4;
            let va = _mm256_loadu_pd(a_ptr.add(off));
            let vb = _mm256_loadu_pd(b_ptr.add(off));
            _mm256_storeu_pd(o_ptr.add(off), _mm256_sub_pd(va, vb));
        }

        let tail = chunks * 4;
        for j in 0..remainder {
            *out.get_unchecked_mut(tail + j) =
                *a.get_unchecked(tail + j) - *b.get_unchecked(tail + j);
        }
    }

    /// AVX element-wise mul.
    ///
    /// # Safety
    /// Caller must ensure AVX is available and all slices have equal length.
    #[target_feature(enable = "avx")]
    pub(crate) unsafe fn mul_f64_avx(a: &[f64], b: &[f64], out: &mut [f64]) {
        let n = a.len();
        let chunks = n / 4;
        let remainder = n % 4;

        // SAFETY: AVX available, all pointers within bounds.
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let o_ptr = out.as_mut_ptr();

        for i in 0..chunks {
            let off = i * 4;
            let va = _mm256_loadu_pd(a_ptr.add(off));
            let vb = _mm256_loadu_pd(b_ptr.add(off));
            _mm256_storeu_pd(o_ptr.add(off), _mm256_mul_pd(va, vb));
        }

        let tail = chunks * 4;
        for j in 0..remainder {
            *out.get_unchecked_mut(tail + j) =
                *a.get_unchecked(tail + j) * *b.get_unchecked(tail + j);
        }
    }

    /// AVX element-wise div.
    ///
    /// # Safety
    /// Caller must ensure AVX is available and all slices have equal length.
    #[target_feature(enable = "avx")]
    pub(crate) unsafe fn div_f64_avx(a: &[f64], b: &[f64], out: &mut [f64]) {
        let n = a.len();
        let chunks = n / 4;
        let remainder = n % 4;

        // SAFETY: AVX available, all pointers within bounds.
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let o_ptr = out.as_mut_ptr();

        for i in 0..chunks {
            let off = i * 4;
            let va = _mm256_loadu_pd(a_ptr.add(off));
            let vb = _mm256_loadu_pd(b_ptr.add(off));
            _mm256_storeu_pd(o_ptr.add(off), _mm256_div_pd(va, vb));
        }

        let tail = chunks * 4;
        for j in 0..remainder {
            *out.get_unchecked_mut(tail + j) =
                *a.get_unchecked(tail + j) / *b.get_unchecked(tail + j);
        }
    }

    /// AVX axpy: `y[i] += alpha * x[i]`.
    ///
    /// # Safety
    /// Caller must ensure AVX is available and slices have equal length.
    #[target_feature(enable = "avx")]
    pub(crate) unsafe fn axpy_f64_avx(alpha: f64, x: &[f64], y: &mut [f64]) {
        let n = x.len();
        let chunks = n / 4;
        let remainder = n % 4;

        // SAFETY: AVX available, pointers within bounds.
        let valpha = _mm256_set1_pd(alpha);
        let x_ptr = x.as_ptr();
        let y_ptr = y.as_mut_ptr();

        for i in 0..chunks {
            let off = i * 4;
            let vx = _mm256_loadu_pd(x_ptr.add(off));
            let vy = _mm256_loadu_pd(y_ptr.add(off));
            let result = _mm256_add_pd(vy, _mm256_mul_pd(valpha, vx));
            _mm256_storeu_pd(y_ptr.add(off), result);
        }

        let tail = chunks * 4;
        for j in 0..remainder {
            *y.get_unchecked_mut(tail + j) += alpha * *x.get_unchecked(tail + j);
        }
    }

    /// AVX scal: `x[i] *= alpha`.
    ///
    /// # Safety
    /// Caller must ensure AVX is available.
    #[target_feature(enable = "avx")]
    pub(crate) unsafe fn scal_f64_avx(alpha: f64, x: &mut [f64]) {
        let n = x.len();
        let chunks = n / 4;
        let remainder = n % 4;

        // SAFETY: AVX available, pointers within bounds.
        let valpha = _mm256_set1_pd(alpha);
        let ptr = x.as_mut_ptr();

        for i in 0..chunks {
            let off = i * 4;
            let vx = _mm256_loadu_pd(ptr.add(off));
            _mm256_storeu_pd(ptr.add(off), _mm256_mul_pd(valpha, vx));
        }

        let tail = chunks * 4;
        for j in 0..remainder {
            *x.get_unchecked_mut(tail + j) *= alpha;
        }
    }

    /// AVX sum of squares.
    ///
    /// # Safety
    /// Caller must ensure AVX is available.
    #[target_feature(enable = "avx")]
    pub(crate) unsafe fn sum_sq_f64_avx(a: &[f64]) -> f64 {
        let n = a.len();
        let chunks = n / 4;
        let remainder = n % 4;

        // SAFETY: AVX available, pointers within bounds.
        let mut acc = _mm256_setzero_pd();
        let ptr = a.as_ptr();

        for i in 0..chunks {
            let va = _mm256_loadu_pd(ptr.add(i * 4));
            acc = _mm256_add_pd(acc, _mm256_mul_pd(va, va));
        }

        let mut result = hsum_256_f64(acc);

        let tail = chunks * 4;
        for j in 0..remainder {
            let v = a[tail + j];
            result += v * v;
        }
        result
    }

    /// AVX sum of absolute values.
    ///
    /// # Safety
    /// Caller must ensure AVX is available.
    #[target_feature(enable = "avx")]
    pub(crate) unsafe fn asum_f64_avx(a: &[f64]) -> f64 {
        let n = a.len();
        let chunks = n / 4;
        let remainder = n % 4;

        // SAFETY: AVX available, pointers within bounds.
        // Clear sign bit: AND with 0x7FFF_FFFF_FFFF_FFFF.
        let sign_mask = _mm256_set1_pd(f64::from_bits(0x7FFF_FFFF_FFFF_FFFF));
        let mut acc = _mm256_setzero_pd();
        let ptr = a.as_ptr();

        for i in 0..chunks {
            let va = _mm256_loadu_pd(ptr.add(i * 4));
            let abs_va = _mm256_and_pd(va, sign_mask);
            acc = _mm256_add_pd(acc, abs_va);
        }

        let mut result = hsum_256_f64(acc);

        let tail = chunks * 4;
        for j in 0..remainder {
            result += a[tail + j].abs();
        }
        result
    }

    /// AVX min of an f64 slice.
    ///
    /// # Safety
    /// Caller must ensure AVX is available and slice is non-empty.
    #[target_feature(enable = "avx")]
    pub(crate) unsafe fn min_f64_avx(a: &[f64]) -> f64 {
        let n = a.len();
        let chunks = n / 4;
        let remainder = n % 4;

        // SAFETY: AVX available, pointers within bounds.
        let mut vmin = _mm256_set1_pd(f64::INFINITY);
        let ptr = a.as_ptr();

        for i in 0..chunks {
            let va = _mm256_loadu_pd(ptr.add(i * 4));
            vmin = _mm256_min_pd(vmin, va);
        }

        // Horizontal min of 4 lanes.
        let hi = _mm256_extractf128_pd(vmin, 1);
        let lo = _mm256_castpd256_pd128(vmin);
        let min128 = _mm_min_pd(lo, hi);
        let hi64 = _mm_unpackhi_pd(min128, min128);
        let min64 = _mm_min_sd(min128, hi64);
        let mut result = _mm_cvtsd_f64(min64);

        let tail = chunks * 4;
        for j in 0..remainder {
            result = result.min(a[tail + j]);
        }
        result
    }

    /// AVX mean of an f64 slice.
    ///
    /// # Safety
    /// Caller must ensure AVX is available and slice is non-empty.
    #[target_feature(enable = "avx")]
    pub(crate) unsafe fn mean_f64_avx(a: &[f64]) -> f64 {
        sum_f64_avx(a) / a.len() as f64
    }

    /// AVX max of an f64 slice.
    ///
    /// # Safety
    /// Caller must ensure AVX is available and slice is non-empty.
    #[target_feature(enable = "avx")]
    pub(crate) unsafe fn max_f64_avx(a: &[f64]) -> f64 {
        let n = a.len();
        let chunks = n / 4;
        let remainder = n % 4;

        // SAFETY: AVX available, pointers within bounds.
        let mut vmax = _mm256_set1_pd(f64::NEG_INFINITY);
        let ptr = a.as_ptr();

        for i in 0..chunks {
            let va = _mm256_loadu_pd(ptr.add(i * 4));
            vmax = _mm256_max_pd(vmax, va);
        }

        // Horizontal max of 4 lanes.
        let hi = _mm256_extractf128_pd(vmax, 1);
        let lo = _mm256_castpd256_pd128(vmax);
        let max128 = _mm_max_pd(lo, hi);
        let hi64 = _mm_unpackhi_pd(max128, max128);
        let max64 = _mm_max_sd(max128, hi64);
        let mut result = _mm_cvtsd_f64(max64);

        let tail = chunks * 4;
        for j in 0..remainder {
            result = result.max(a[tail + j]);
        }
        result
    }

    /// AVX scalar broadcast add: `out[i] = a[i] + s`.
    #[target_feature(enable = "avx")]
    pub(crate) unsafe fn add_scalar_f64_avx(a: &[f64], s: f64, out: &mut [f64]) {
        let n = a.len();
        let chunks = n / 4;
        let vs = _mm256_set1_pd(s);
        let a_ptr = a.as_ptr();
        let o_ptr = out.as_mut_ptr();
        for i in 0..chunks {
            let off = i * 4;
            _mm256_storeu_pd(
                o_ptr.add(off),
                _mm256_add_pd(_mm256_loadu_pd(a_ptr.add(off)), vs),
            );
        }
        let tail = chunks * 4;
        for j in tail..n {
            *out.get_unchecked_mut(j) = *a.get_unchecked(j) + s;
        }
    }

    /// AVX scalar broadcast sub: `out[i] = a[i] - s`.
    #[target_feature(enable = "avx")]
    pub(crate) unsafe fn sub_scalar_f64_avx(a: &[f64], s: f64, out: &mut [f64]) {
        let n = a.len();
        let chunks = n / 4;
        let vs = _mm256_set1_pd(s);
        let a_ptr = a.as_ptr();
        let o_ptr = out.as_mut_ptr();
        for i in 0..chunks {
            let off = i * 4;
            _mm256_storeu_pd(
                o_ptr.add(off),
                _mm256_sub_pd(_mm256_loadu_pd(a_ptr.add(off)), vs),
            );
        }
        let tail = chunks * 4;
        for j in tail..n {
            *out.get_unchecked_mut(j) = *a.get_unchecked(j) - s;
        }
    }

    /// AVX scalar broadcast mul: `out[i] = a[i] * s`.
    #[target_feature(enable = "avx")]
    pub(crate) unsafe fn mul_scalar_f64_avx(a: &[f64], s: f64, out: &mut [f64]) {
        let n = a.len();
        let chunks = n / 4;
        let vs = _mm256_set1_pd(s);
        let a_ptr = a.as_ptr();
        let o_ptr = out.as_mut_ptr();
        for i in 0..chunks {
            let off = i * 4;
            _mm256_storeu_pd(
                o_ptr.add(off),
                _mm256_mul_pd(_mm256_loadu_pd(a_ptr.add(off)), vs),
            );
        }
        let tail = chunks * 4;
        for j in tail..n {
            *out.get_unchecked_mut(j) = *a.get_unchecked(j) * s;
        }
    }

    /// AVX scalar broadcast div: `out[i] = a[i] / s`.
    #[target_feature(enable = "avx")]
    pub(crate) unsafe fn div_scalar_f64_avx(a: &[f64], s: f64, out: &mut [f64]) {
        let n = a.len();
        let chunks = n / 4;
        let vs = _mm256_set1_pd(s);
        let a_ptr = a.as_ptr();
        let o_ptr = out.as_mut_ptr();
        for i in 0..chunks {
            let off = i * 4;
            _mm256_storeu_pd(
                o_ptr.add(off),
                _mm256_div_pd(_mm256_loadu_pd(a_ptr.add(off)), vs),
            );
        }
        let tail = chunks * 4;
        for j in tail..n {
            *out.get_unchecked_mut(j) = *a.get_unchecked(j) / s;
        }
    }

    /// AVX negate: `out[i] = -a[i]`.
    #[target_feature(enable = "avx")]
    pub(crate) unsafe fn neg_f64_avx(a: &[f64], out: &mut [f64]) {
        let n = a.len();
        let chunks = n / 4;
        let vneg = _mm256_set1_pd(-0.0); // sign bit mask
        let a_ptr = a.as_ptr();
        let o_ptr = out.as_mut_ptr();
        for i in 0..chunks {
            let off = i * 4;
            _mm256_storeu_pd(
                o_ptr.add(off),
                _mm256_xor_pd(_mm256_loadu_pd(a_ptr.add(off)), vneg),
            );
        }
        let tail = chunks * 4;
        for j in tail..n {
            *out.get_unchecked_mut(j) = -*a.get_unchecked(j);
        }
    }

    /// AVX ReLU: `out[i] = max(0, a[i])`.
    ///
    /// # Safety
    /// Caller must ensure AVX is available.
    #[target_feature(enable = "avx")]
    pub(crate) unsafe fn relu_f64_avx(a: &[f64], out: &mut [f64]) {
        let n = a.len();
        let chunks = n / 4;
        let remainder = n % 4;
        let a_ptr = a.as_ptr();
        let o_ptr = out.as_mut_ptr();
        // SAFETY: AVX available, pointers within bounds.
        let vzero = _mm256_setzero_pd();
        for i in 0..chunks {
            let va = _mm256_loadu_pd(a_ptr.add(i * 4));
            _mm256_storeu_pd(o_ptr.add(i * 4), _mm256_max_pd(vzero, va));
        }
        let tail = chunks * 4;
        for j in 0..remainder {
            let v = a[tail + j];
            out[tail + j] = if v > 0.0 { v } else { 0.0 };
        }
    }
}

// ---------------------------------------------------------------------------
// Public dispatch functions
// ---------------------------------------------------------------------------

/// SIMD-accelerated dot product for f64 slices.
pub(crate) fn dot_f64(a: &[f64], b: &[f64]) -> f64 {
    super::dispatch_f64!(
        avx::dot_f64_avx,
        super::neon_f64_ops::dot_f64_neon,
        dot_f64_scalar,
        a,
        b
    )
}

/// SIMD-accelerated sum for f64 slices.
pub(crate) fn sum_f64(a: &[f64]) -> f64 {
    super::dispatch_f64!(
        avx::sum_f64_avx,
        super::neon_f64_ops::sum_f64_neon,
        sum_f64_scalar,
        a
    )
}

/// SIMD-accelerated element-wise add for f64 slices.
pub(crate) fn add_f64(a: &[f64], b: &[f64], out: &mut [f64]) {
    super::dispatch_f64!(
        avx::add_f64_avx,
        super::neon_f64_ops::add_f64_neon,
        add_f64_scalar,
        a,
        b,
        out
    );
}

/// SIMD-accelerated element-wise sub for f64 slices.
pub(crate) fn sub_f64(a: &[f64], b: &[f64], out: &mut [f64]) {
    super::dispatch_f64!(
        avx::sub_f64_avx,
        super::neon_f64_ops::sub_f64_neon,
        sub_f64_scalar,
        a,
        b,
        out
    );
}

/// SIMD-accelerated element-wise mul for f64 slices.
pub(crate) fn mul_f64(a: &[f64], b: &[f64], out: &mut [f64]) {
    super::dispatch_f64!(
        avx::mul_f64_avx,
        super::neon_f64_ops::mul_f64_neon,
        mul_f64_scalar,
        a,
        b,
        out
    );
}

/// SIMD-accelerated element-wise div for f64 slices.
pub(crate) fn div_f64(a: &[f64], b: &[f64], out: &mut [f64]) {
    super::dispatch_f64!(
        avx::div_f64_avx,
        super::neon_f64_ops::div_f64_neon,
        div_f64_scalar,
        a,
        b,
        out
    );
}

/// SIMD-accelerated axpy for f64 slices.
pub(crate) fn axpy_f64(alpha: f64, x: &[f64], y: &mut [f64]) {
    super::dispatch_f64!(
        avx::axpy_f64_avx,
        super::neon_f64_ops::axpy_f64_neon,
        axpy_f64_scalar,
        alpha,
        x,
        y
    );
}

/// SIMD-accelerated scal for f64 slices.
pub(crate) fn scal_f64(alpha: f64, x: &mut [f64]) {
    super::dispatch_f64!(
        avx::scal_f64_avx,
        super::neon_f64_ops::scal_f64_neon,
        scal_f64_scalar,
        alpha,
        x
    );
}

/// SIMD-accelerated sum of squares for f64 slices.
pub(crate) fn sum_sq_f64(a: &[f64]) -> f64 {
    super::dispatch_f64!(
        avx::sum_sq_f64_avx,
        super::neon_f64_ops::sum_sq_f64_neon,
        sum_sq_f64_scalar,
        a
    )
}

/// SIMD-accelerated sum of absolute values for f64 slices.
pub(crate) fn asum_f64(a: &[f64]) -> f64 {
    super::dispatch_f64!(
        avx::asum_f64_avx,
        super::neon_f64_ops::asum_f64_neon,
        asum_f64_scalar,
        a
    )
}

/// SIMD-accelerated min for f64 slices.
pub(crate) fn min_f64(a: &[f64]) -> f64 {
    super::dispatch_f64!(
        avx::min_f64_avx,
        super::neon_f64_ops::min_f64_neon,
        min_f64_scalar,
        a
    )
}

/// SIMD-accelerated max for f64 slices.
pub(crate) fn max_f64(a: &[f64]) -> f64 {
    super::dispatch_f64!(
        avx::max_f64_avx,
        super::neon_f64_ops::max_f64_neon,
        max_f64_scalar,
        a
    )
}

/// SIMD-accelerated mean for f64 slices.
pub(crate) fn mean_f64(a: &[f64]) -> f64 {
    super::dispatch_f64!(
        avx::mean_f64_avx,
        super::neon_f64_ops::mean_f64_neon,
        mean_f64_scalar,
        a
    )
}

/// SIMD-accelerated ReLU for f64 slices: `out[i] = max(0, a[i])`.
pub(crate) fn relu_f64(a: &[f64], out: &mut [f64]) {
    super::dispatch_f64!(
        avx::relu_f64_avx,
        super::neon_f64_ops::relu_f64_neon,
        relu_f64_scalar,
        a,
        out
    );
}

/// SIMD-accelerated scalar broadcast add for f64: `out[i] = a[i] + s`.
pub(crate) fn add_scalar_f64(a: &[f64], s: f64, out: &mut [f64]) {
    super::dispatch_f64!(
        avx::add_scalar_f64_avx,
        super::neon_f64_ops::add_scalar_f64_neon,
        add_scalar_f64_scalar,
        a,
        s,
        out
    );
}

/// SIMD-accelerated scalar broadcast sub for f64: `out[i] = a[i] - s`.
pub(crate) fn sub_scalar_f64(a: &[f64], s: f64, out: &mut [f64]) {
    super::dispatch_f64!(
        avx::sub_scalar_f64_avx,
        super::neon_f64_ops::sub_scalar_f64_neon,
        sub_scalar_f64_scalar,
        a,
        s,
        out
    );
}

/// SIMD-accelerated scalar broadcast mul for f64: `out[i] = a[i] * s`.
pub(crate) fn mul_scalar_f64(a: &[f64], s: f64, out: &mut [f64]) {
    super::dispatch_f64!(
        avx::mul_scalar_f64_avx,
        super::neon_f64_ops::mul_scalar_f64_neon,
        mul_scalar_f64_scalar,
        a,
        s,
        out
    );
}

/// SIMD-accelerated scalar broadcast div for f64: `out[i] = a[i] / s`.
pub(crate) fn div_scalar_f64(a: &[f64], s: f64, out: &mut [f64]) {
    super::dispatch_f64!(
        avx::div_scalar_f64_avx,
        super::neon_f64_ops::div_scalar_f64_neon,
        div_scalar_f64_scalar,
        a,
        s,
        out
    );
}

/// SIMD-accelerated negate for f64: `out[i] = -a[i]`.
pub(crate) fn neg_f64(a: &[f64], out: &mut [f64]) {
    super::dispatch_f64!(
        avx::neg_f64_avx,
        super::neon_f64_ops::neg_f64_neon,
        neg_f64_scalar,
        a,
        out
    );
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_f64() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        assert!((dot_f64(&a, &b) - 35.0).abs() < 1e-10);
    }

    #[test]
    fn test_dot_f64_large() {
        let n = 1024;
        let a: Vec<f64> = (0..n).map(f64::from).collect();
        let b: Vec<f64> = (0..n).map(|i| f64::from(n - i)).collect();
        let expected = dot_f64_scalar(&a, &b);
        assert!((dot_f64(&a, &b) - expected).abs() < 1e-6);
    }

    #[test]
    fn test_sum_f64() {
        let a: Vec<f64> = (1..=100).map(f64::from).collect();
        assert!((sum_f64(&a) - 5050.0).abs() < 1e-10);
    }

    #[test]
    fn test_add_f64() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let mut out = vec![0.0; 5];
        add_f64(&a, &b, &mut out);
        assert_eq!(out, vec![11.0, 22.0, 33.0, 44.0, 55.0]);
    }

    #[test]
    fn test_mul_f64() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0, 3.0, 4.0, 5.0, 6.0];
        let mut out = vec![0.0; 5];
        mul_f64(&a, &b, &mut out);
        assert_eq!(out, vec![2.0, 6.0, 12.0, 20.0, 30.0]);
    }

    #[test]
    fn test_axpy_f64() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut y = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        axpy_f64(2.0, &x, &mut y);
        assert_eq!(y, vec![12.0, 24.0, 36.0, 48.0, 60.0]);
    }

    #[test]
    fn test_scal_f64() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        scal_f64(10.0, &mut x);
        assert_eq!(x, vec![10.0, 20.0, 30.0, 40.0, 50.0]);
    }

    #[test]
    fn test_sum_sq_f64() {
        let a = vec![3.0, 4.0];
        assert!((sum_sq_f64(&a) - 25.0).abs() < 1e-10);
    }

    #[test]
    fn test_asum_f64() {
        let a = vec![-1.0, 2.0, -3.0, 4.0, -5.0];
        assert!((asum_f64(&a) - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_min_f64() {
        let a = vec![3.0, 1.0, 4.0, 1.5, 9.0, 2.6, 5.3];
        assert!((min_f64(&a) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_max_f64() {
        let a = vec![3.0, 1.0, 4.0, 1.5, 9.0, 2.6, 5.3];
        assert!((max_f64(&a) - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_mean_f64() {
        let a: Vec<f64> = (1..=100).map(f64::from).collect();
        assert!((mean_f64(&a) - 50.5).abs() < 1e-10);
    }

    #[test]
    fn test_empty_slices() {
        assert_eq!(dot_f64(&[], &[]), 0.0);
        assert_eq!(sum_f64(&[]), 0.0);
        assert_eq!(mean_f64(&[]), 0.0);
    }

    #[test]
    fn test_single_element() {
        assert!((dot_f64(&[3.0], &[7.0]) - 21.0).abs() < 1e-10);
        assert!((sum_f64(&[42.0]) - 42.0).abs() < 1e-10);
    }

    #[test]
    fn test_exact_chunk_size() {
        // Exactly 4 elements — one full AVX chunk, no remainder.
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        assert!((dot_f64(&a, &b) - 70.0).abs() < 1e-10);
    }

    #[test]
    fn test_large_sum() {
        let n = 10_000;
        let a: Vec<f64> = (1..=n).map(f64::from).collect();
        let expected = f64::from(n * (n + 1) / 2);
        assert!((sum_f64(&a) - expected).abs() < 1e-4);
    }
}
