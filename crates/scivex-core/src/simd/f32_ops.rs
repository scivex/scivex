//! AVX-accelerated f32 kernels (8-wide `__m256`).

// ---------------------------------------------------------------------------
// Scalar fallbacks
// ---------------------------------------------------------------------------

/// Scalar dot product for f32.
pub(crate) fn dot_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .fold(0.0, |acc, (&x, &y)| acc + x * y)
}

/// Scalar sum for f32.
pub(crate) fn sum_f32_scalar(a: &[f32]) -> f32 {
    a.iter().copied().sum()
}

/// Scalar element-wise add for f32.
pub(crate) fn add_f32_scalar(a: &[f32], b: &[f32], out: &mut [f32]) {
    for i in 0..a.len() {
        out[i] = a[i] + b[i];
    }
}

/// Scalar element-wise sub for f32.
pub(crate) fn sub_f32_scalar(a: &[f32], b: &[f32], out: &mut [f32]) {
    for i in 0..a.len() {
        out[i] = a[i] - b[i];
    }
}

/// Scalar element-wise mul for f32.
pub(crate) fn mul_f32_scalar(a: &[f32], b: &[f32], out: &mut [f32]) {
    for i in 0..a.len() {
        out[i] = a[i] * b[i];
    }
}

/// Scalar element-wise div for f32.
pub(crate) fn div_f32_scalar(a: &[f32], b: &[f32], out: &mut [f32]) {
    for i in 0..a.len() {
        out[i] = a[i] / b[i];
    }
}

/// Scalar axpy for f32.
pub(crate) fn axpy_f32_scalar(alpha: f32, x: &[f32], y: &mut [f32]) {
    for i in 0..x.len() {
        y[i] += alpha * x[i];
    }
}

/// Scalar scal for f32.
pub(crate) fn scal_f32_scalar(alpha: f32, x: &mut [f32]) {
    for v in x.iter_mut() {
        *v *= alpha;
    }
}

/// Scalar sum of squares for f32.
pub(crate) fn sum_sq_f32_scalar(a: &[f32]) -> f32 {
    a.iter().fold(0.0, |acc, &v| acc + v * v)
}

/// Scalar sum of absolute values for f32.
pub(crate) fn asum_f32_scalar(a: &[f32]) -> f32 {
    a.iter().fold(0.0, |acc, &v| acc + v.abs())
}

/// Scalar min for f32.
pub(crate) fn min_f32_scalar(a: &[f32]) -> f32 {
    a.iter().copied().fold(f32::INFINITY, f32::min)
}

/// Scalar max for f32.
pub(crate) fn max_f32_scalar(a: &[f32]) -> f32 {
    a.iter().copied().fold(f32::NEG_INFINITY, f32::max)
}

/// Scalar mean for f32.
pub(crate) fn mean_f32_scalar(a: &[f32]) -> f32 {
    if a.is_empty() {
        return 0.0;
    }
    sum_f32_scalar(a) / a.len() as f32
}

/// Scalar ReLU for f32: `out[i] = max(0, a[i])`.
pub(crate) fn relu_f32_scalar(a: &[f32], out: &mut [f32]) {
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

    /// Horizontal sum of a 256-bit f32 register (8 lanes).
    ///
    /// # Safety
    /// Caller must ensure AVX is available.
    #[inline(always)]
    unsafe fn hsum_256_f32(v: __m256) -> f32 {
        // SAFETY: AVX guaranteed by caller.
        let hi = _mm256_extractf128_ps(v, 1);
        let lo = _mm256_castps256_ps128(v);
        let sum128 = _mm_add_ps(lo, hi); // [a+e, b+f, c+g, d+h]
        let shuf = _mm_movehdup_ps(sum128); // [b+f, b+f, d+h, d+h]
        let sums = _mm_add_ps(sum128, shuf); // [ab+ef, -, cd+gh, -]
        let hi32 = _mm_movehl_ps(sums, sums); // [cd+gh, -, -, -]
        let total = _mm_add_ss(sums, hi32);
        _mm_cvtss_f32(total)
    }

    /// AVX dot product of two f32 slices.
    ///
    /// # Safety
    /// Caller must ensure AVX is available.
    #[target_feature(enable = "avx")]
    pub(crate) unsafe fn dot_f32_avx(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len();
        let chunks = n / 8;
        let remainder = n % 8;

        // SAFETY: AVX available, pointer arithmetic within bounds.
        let mut acc = _mm256_setzero_ps();
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        for i in 0..chunks {
            let va = _mm256_loadu_ps(a_ptr.add(i * 8));
            let vb = _mm256_loadu_ps(b_ptr.add(i * 8));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(va, vb));
        }

        let mut result = hsum_256_f32(acc);

        let tail = chunks * 8;
        for j in 0..remainder {
            result += a[tail + j] * b[tail + j];
        }
        result
    }

    /// AVX sum of an f32 slice.
    ///
    /// # Safety
    /// Caller must ensure AVX is available.
    #[target_feature(enable = "avx")]
    pub(crate) unsafe fn sum_f32_avx(a: &[f32]) -> f32 {
        let n = a.len();
        let chunks = n / 8;
        let remainder = n % 8;

        // SAFETY: AVX available, pointer within bounds.
        let mut acc = _mm256_setzero_ps();
        let ptr = a.as_ptr();

        for i in 0..chunks {
            let va = _mm256_loadu_ps(ptr.add(i * 8));
            acc = _mm256_add_ps(acc, va);
        }

        let mut result = hsum_256_f32(acc);

        let tail = chunks * 8;
        for j in 0..remainder {
            result += a[tail + j];
        }
        result
    }

    /// AVX element-wise add for f32.
    ///
    /// # Safety
    /// Caller must ensure AVX is available and slices have equal length.
    #[target_feature(enable = "avx")]
    pub(crate) unsafe fn add_f32_avx(a: &[f32], b: &[f32], out: &mut [f32]) {
        let n = a.len();
        let chunks = n / 8;
        let remainder = n % 8;

        // SAFETY: AVX available, pointers within bounds.
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let o_ptr = out.as_mut_ptr();

        for i in 0..chunks {
            let off = i * 8;
            let va = _mm256_loadu_ps(a_ptr.add(off));
            let vb = _mm256_loadu_ps(b_ptr.add(off));
            _mm256_storeu_ps(o_ptr.add(off), _mm256_add_ps(va, vb));
        }

        let tail = chunks * 8;
        for j in 0..remainder {
            *out.get_unchecked_mut(tail + j) =
                *a.get_unchecked(tail + j) + *b.get_unchecked(tail + j);
        }
    }

    /// AVX element-wise sub for f32.
    ///
    /// # Safety
    /// Caller must ensure AVX is available and slices have equal length.
    #[target_feature(enable = "avx")]
    pub(crate) unsafe fn sub_f32_avx(a: &[f32], b: &[f32], out: &mut [f32]) {
        let n = a.len();
        let chunks = n / 8;
        let remainder = n % 8;

        // SAFETY: AVX available, pointers within bounds.
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let o_ptr = out.as_mut_ptr();

        for i in 0..chunks {
            let off = i * 8;
            let va = _mm256_loadu_ps(a_ptr.add(off));
            let vb = _mm256_loadu_ps(b_ptr.add(off));
            _mm256_storeu_ps(o_ptr.add(off), _mm256_sub_ps(va, vb));
        }

        let tail = chunks * 8;
        for j in 0..remainder {
            *out.get_unchecked_mut(tail + j) =
                *a.get_unchecked(tail + j) - *b.get_unchecked(tail + j);
        }
    }

    /// AVX element-wise mul for f32.
    ///
    /// # Safety
    /// Caller must ensure AVX is available and slices have equal length.
    #[target_feature(enable = "avx")]
    pub(crate) unsafe fn mul_f32_avx(a: &[f32], b: &[f32], out: &mut [f32]) {
        let n = a.len();
        let chunks = n / 8;
        let remainder = n % 8;

        // SAFETY: AVX available, pointers within bounds.
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let o_ptr = out.as_mut_ptr();

        for i in 0..chunks {
            let off = i * 8;
            let va = _mm256_loadu_ps(a_ptr.add(off));
            let vb = _mm256_loadu_ps(b_ptr.add(off));
            _mm256_storeu_ps(o_ptr.add(off), _mm256_mul_ps(va, vb));
        }

        let tail = chunks * 8;
        for j in 0..remainder {
            *out.get_unchecked_mut(tail + j) =
                *a.get_unchecked(tail + j) * *b.get_unchecked(tail + j);
        }
    }

    /// AVX element-wise div for f32.
    ///
    /// # Safety
    /// Caller must ensure AVX is available and slices have equal length.
    #[target_feature(enable = "avx")]
    pub(crate) unsafe fn div_f32_avx(a: &[f32], b: &[f32], out: &mut [f32]) {
        let n = a.len();
        let chunks = n / 8;
        let remainder = n % 8;

        // SAFETY: AVX available, pointers within bounds.
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let o_ptr = out.as_mut_ptr();

        for i in 0..chunks {
            let off = i * 8;
            let va = _mm256_loadu_ps(a_ptr.add(off));
            let vb = _mm256_loadu_ps(b_ptr.add(off));
            _mm256_storeu_ps(o_ptr.add(off), _mm256_div_ps(va, vb));
        }

        let tail = chunks * 8;
        for j in 0..remainder {
            *out.get_unchecked_mut(tail + j) =
                *a.get_unchecked(tail + j) / *b.get_unchecked(tail + j);
        }
    }

    /// AVX axpy for f32.
    ///
    /// # Safety
    /// Caller must ensure AVX is available and slices have equal length.
    #[target_feature(enable = "avx")]
    pub(crate) unsafe fn axpy_f32_avx(alpha: f32, x: &[f32], y: &mut [f32]) {
        let n = x.len();
        let chunks = n / 8;
        let remainder = n % 8;

        // SAFETY: AVX available, pointers within bounds.
        let valpha = _mm256_set1_ps(alpha);
        let x_ptr = x.as_ptr();
        let y_ptr = y.as_mut_ptr();

        for i in 0..chunks {
            let off = i * 8;
            let vx = _mm256_loadu_ps(x_ptr.add(off));
            let vy = _mm256_loadu_ps(y_ptr.add(off));
            let result = _mm256_add_ps(vy, _mm256_mul_ps(valpha, vx));
            _mm256_storeu_ps(y_ptr.add(off), result);
        }

        let tail = chunks * 8;
        for j in 0..remainder {
            *y.get_unchecked_mut(tail + j) += alpha * *x.get_unchecked(tail + j);
        }
    }

    /// AVX scal for f32.
    ///
    /// # Safety
    /// Caller must ensure AVX is available.
    #[target_feature(enable = "avx")]
    pub(crate) unsafe fn scal_f32_avx(alpha: f32, x: &mut [f32]) {
        let n = x.len();
        let chunks = n / 8;
        let remainder = n % 8;

        // SAFETY: AVX available, pointers within bounds.
        let valpha = _mm256_set1_ps(alpha);
        let ptr = x.as_mut_ptr();

        for i in 0..chunks {
            let off = i * 8;
            let vx = _mm256_loadu_ps(ptr.add(off));
            _mm256_storeu_ps(ptr.add(off), _mm256_mul_ps(valpha, vx));
        }

        let tail = chunks * 8;
        for j in 0..remainder {
            *x.get_unchecked_mut(tail + j) *= alpha;
        }
    }

    /// AVX sum of squares for f32.
    ///
    /// # Safety
    /// Caller must ensure AVX is available.
    #[target_feature(enable = "avx")]
    pub(crate) unsafe fn sum_sq_f32_avx(a: &[f32]) -> f32 {
        let n = a.len();
        let chunks = n / 8;
        let remainder = n % 8;

        // SAFETY: AVX available, pointers within bounds.
        let mut acc = _mm256_setzero_ps();
        let ptr = a.as_ptr();

        for i in 0..chunks {
            let va = _mm256_loadu_ps(ptr.add(i * 8));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(va, va));
        }

        let mut result = hsum_256_f32(acc);

        let tail = chunks * 8;
        for j in 0..remainder {
            let v = a[tail + j];
            result += v * v;
        }
        result
    }

    /// AVX sum of absolute values for f32.
    ///
    /// # Safety
    /// Caller must ensure AVX is available.
    #[target_feature(enable = "avx")]
    pub(crate) unsafe fn asum_f32_avx(a: &[f32]) -> f32 {
        let n = a.len();
        let chunks = n / 8;
        let remainder = n % 8;

        // SAFETY: AVX available, pointers within bounds.
        // Clear sign bit: AND with 0x7FFF_FFFF.
        let sign_mask = _mm256_set1_ps(f32::from_bits(0x7FFF_FFFF));
        let mut acc = _mm256_setzero_ps();
        let ptr = a.as_ptr();

        for i in 0..chunks {
            let va = _mm256_loadu_ps(ptr.add(i * 8));
            let abs_va = _mm256_and_ps(va, sign_mask);
            acc = _mm256_add_ps(acc, abs_va);
        }

        let mut result = hsum_256_f32(acc);

        let tail = chunks * 8;
        for j in 0..remainder {
            result += a[tail + j].abs();
        }
        result
    }

    /// AVX min of an f32 slice.
    ///
    /// # Safety
    /// Caller must ensure AVX is available and slice is non-empty.
    #[target_feature(enable = "avx")]
    pub(crate) unsafe fn min_f32_avx(a: &[f32]) -> f32 {
        let n = a.len();
        let chunks = n / 8;
        let remainder = n % 8;

        // SAFETY: AVX available, pointers within bounds.
        let mut vmin = _mm256_set1_ps(f32::INFINITY);
        let ptr = a.as_ptr();

        for i in 0..chunks {
            let va = _mm256_loadu_ps(ptr.add(i * 8));
            vmin = _mm256_min_ps(vmin, va);
        }

        // Horizontal min.
        let hi = _mm256_extractf128_ps(vmin, 1);
        let lo = _mm256_castps256_ps128(vmin);
        let min128 = _mm_min_ps(lo, hi);
        let shuf = _mm_movehdup_ps(min128);
        let mins = _mm_min_ps(min128, shuf);
        let hi32 = _mm_movehl_ps(mins, mins);
        let min_val = _mm_min_ss(mins, hi32);
        let mut result = _mm_cvtss_f32(min_val);

        let tail = chunks * 8;
        for j in 0..remainder {
            result = result.min(a[tail + j]);
        }
        result
    }

    /// AVX mean of an f32 slice.
    ///
    /// # Safety
    /// Caller must ensure AVX is available and slice is non-empty.
    #[target_feature(enable = "avx")]
    pub(crate) unsafe fn mean_f32_avx(a: &[f32]) -> f32 {
        sum_f32_avx(a) / a.len() as f32
    }

    /// AVX max of an f32 slice.
    ///
    /// # Safety
    /// Caller must ensure AVX is available and slice is non-empty.
    #[target_feature(enable = "avx")]
    pub(crate) unsafe fn max_f32_avx(a: &[f32]) -> f32 {
        let n = a.len();
        let chunks = n / 8;
        let remainder = n % 8;

        // SAFETY: AVX available, pointers within bounds.
        let mut vmax = _mm256_set1_ps(f32::NEG_INFINITY);
        let ptr = a.as_ptr();

        for i in 0..chunks {
            let va = _mm256_loadu_ps(ptr.add(i * 8));
            vmax = _mm256_max_ps(vmax, va);
        }

        // Horizontal max.
        let hi = _mm256_extractf128_ps(vmax, 1);
        let lo = _mm256_castps256_ps128(vmax);
        let max128 = _mm_max_ps(lo, hi);
        let shuf = _mm_movehdup_ps(max128);
        let maxs = _mm_max_ps(max128, shuf);
        let hi32 = _mm_movehl_ps(maxs, maxs);
        let max_val = _mm_max_ss(maxs, hi32);
        let mut result = _mm_cvtss_f32(max_val);

        let tail = chunks * 8;
        for j in 0..remainder {
            result = result.max(a[tail + j]);
        }
        result
    }

    /// AVX ReLU for f32: `out[i] = max(0, a[i])`.
    ///
    /// # Safety
    /// Caller must ensure AVX is available.
    #[target_feature(enable = "avx")]
    pub(crate) unsafe fn relu_f32_avx(a: &[f32], out: &mut [f32]) {
        let n = a.len();
        let chunks = n / 8;
        let remainder = n % 8;
        let a_ptr = a.as_ptr();
        let o_ptr = out.as_mut_ptr();
        // SAFETY: AVX available, pointers within bounds.
        let vzero = _mm256_setzero_ps();
        for i in 0..chunks {
            let va = _mm256_loadu_ps(a_ptr.add(i * 8));
            _mm256_storeu_ps(o_ptr.add(i * 8), _mm256_max_ps(vzero, va));
        }
        let tail = chunks * 8;
        for j in 0..remainder {
            let v = a[tail + j];
            out[tail + j] = if v > 0.0 { v } else { 0.0 };
        }
    }
}

// ---------------------------------------------------------------------------
// Public dispatch functions
// ---------------------------------------------------------------------------

/// SIMD-accelerated dot product for f32 slices.
pub(crate) fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    super::dispatch_f32!(
        avx::dot_f32_avx,
        super::neon_f32_ops::dot_f32_neon,
        dot_f32_scalar,
        a,
        b
    )
}

/// SIMD-accelerated sum for f32 slices.
pub(crate) fn sum_f32(a: &[f32]) -> f32 {
    super::dispatch_f32!(
        avx::sum_f32_avx,
        super::neon_f32_ops::sum_f32_neon,
        sum_f32_scalar,
        a
    )
}

/// SIMD-accelerated element-wise add for f32 slices.
pub(crate) fn add_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    super::dispatch_f32!(
        avx::add_f32_avx,
        super::neon_f32_ops::add_f32_neon,
        add_f32_scalar,
        a,
        b,
        out
    );
}

/// SIMD-accelerated element-wise sub for f32 slices.
pub(crate) fn sub_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    super::dispatch_f32!(
        avx::sub_f32_avx,
        super::neon_f32_ops::sub_f32_neon,
        sub_f32_scalar,
        a,
        b,
        out
    );
}

/// SIMD-accelerated element-wise mul for f32 slices.
pub(crate) fn mul_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    super::dispatch_f32!(
        avx::mul_f32_avx,
        super::neon_f32_ops::mul_f32_neon,
        mul_f32_scalar,
        a,
        b,
        out
    );
}

/// SIMD-accelerated element-wise div for f32 slices.
pub(crate) fn div_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    super::dispatch_f32!(
        avx::div_f32_avx,
        super::neon_f32_ops::div_f32_neon,
        div_f32_scalar,
        a,
        b,
        out
    );
}

/// SIMD-accelerated axpy for f32 slices.
pub(crate) fn axpy_f32(alpha: f32, x: &[f32], y: &mut [f32]) {
    super::dispatch_f32!(
        avx::axpy_f32_avx,
        super::neon_f32_ops::axpy_f32_neon,
        axpy_f32_scalar,
        alpha,
        x,
        y
    );
}

/// SIMD-accelerated scal for f32 slices.
pub(crate) fn scal_f32(alpha: f32, x: &mut [f32]) {
    super::dispatch_f32!(
        avx::scal_f32_avx,
        super::neon_f32_ops::scal_f32_neon,
        scal_f32_scalar,
        alpha,
        x
    );
}

/// SIMD-accelerated sum of squares for f32 slices.
pub(crate) fn sum_sq_f32(a: &[f32]) -> f32 {
    super::dispatch_f32!(
        avx::sum_sq_f32_avx,
        super::neon_f32_ops::sum_sq_f32_neon,
        sum_sq_f32_scalar,
        a
    )
}

/// SIMD-accelerated sum of absolute values for f32 slices.
pub(crate) fn asum_f32(a: &[f32]) -> f32 {
    super::dispatch_f32!(
        avx::asum_f32_avx,
        super::neon_f32_ops::asum_f32_neon,
        asum_f32_scalar,
        a
    )
}

/// SIMD-accelerated min for f32 slices.
pub(crate) fn min_f32(a: &[f32]) -> f32 {
    super::dispatch_f32!(
        avx::min_f32_avx,
        super::neon_f32_ops::min_f32_neon,
        min_f32_scalar,
        a
    )
}

/// SIMD-accelerated max for f32 slices.
pub(crate) fn max_f32(a: &[f32]) -> f32 {
    super::dispatch_f32!(
        avx::max_f32_avx,
        super::neon_f32_ops::max_f32_neon,
        max_f32_scalar,
        a
    )
}

/// SIMD-accelerated mean for f32 slices.
pub(crate) fn mean_f32(a: &[f32]) -> f32 {
    super::dispatch_f32!(
        avx::mean_f32_avx,
        super::neon_f32_ops::mean_f32_neon,
        mean_f32_scalar,
        a
    )
}

/// SIMD-accelerated ReLU for f32 slices: `out[i] = max(0, a[i])`.
pub(crate) fn relu_f32(a: &[f32], out: &mut [f32]) {
    super::dispatch_f32!(
        avx::relu_f32_avx,
        super::neon_f32_ops::relu_f32_neon,
        relu_f32_scalar,
        a,
        out
    );
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_f32() {
        let a = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let b = vec![9.0_f32, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let expected = dot_f32_scalar(&a, &b);
        assert!((dot_f32(&a, &b) - expected).abs() < 1e-4);
    }

    #[test]
    fn test_dot_f32_large() {
        let n = 1024;
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();
        let expected = dot_f32_scalar(&a, &b);
        assert!((dot_f32(&a, &b) - expected).abs() / expected.abs() < 1e-4);
    }

    #[test]
    fn test_sum_f32() {
        let a: Vec<f32> = (1..=100).map(|i| i as f32).collect();
        assert!((sum_f32(&a) - 5050.0).abs() < 1.0);
    }

    #[test]
    fn test_add_f32() {
        let a: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let b: Vec<f32> = (10..20).map(|i| i as f32).collect();
        let mut out = vec![0.0_f32; 10];
        add_f32(&a, &b, &mut out);
        for i in 0..10 {
            assert_eq!(out[i], a[i] + b[i]);
        }
    }

    #[test]
    fn test_mul_f32() {
        let a = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let b = vec![2.0_f32; 9];
        let mut out = vec![0.0_f32; 9];
        mul_f32(&a, &b, &mut out);
        for i in 0..9 {
            assert_eq!(out[i], a[i] * 2.0);
        }
    }

    #[test]
    fn test_axpy_f32() {
        let x = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mut y = vec![10.0_f32; 9];
        axpy_f32(2.0, &x, &mut y);
        for i in 0..9 {
            assert_eq!(y[i], 10.0 + 2.0 * x[i]);
        }
    }

    #[test]
    fn test_scal_f32() {
        let mut x = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let orig: Vec<f32> = x.clone();
        scal_f32(3.0, &mut x);
        for i in 0..9 {
            assert_eq!(x[i], orig[i] * 3.0);
        }
    }

    #[test]
    fn test_sum_sq_f32() {
        let a = vec![3.0_f32, 4.0];
        assert!((sum_sq_f32(&a) - 25.0).abs() < 1e-4);
    }

    #[test]
    fn test_asum_f32() {
        let a = vec![-1.0_f32, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0, -9.0];
        assert!((asum_f32(&a) - 45.0).abs() < 1e-4);
    }

    #[test]
    fn test_min_f32() {
        let a = vec![3.0_f32, 1.0, 4.0, 1.5, 9.0, 2.6, 5.3, 0.5, 7.0];
        assert!((min_f32(&a) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_max_f32() {
        let a = vec![3.0_f32, 1.0, 4.0, 1.5, 9.0, 2.6, 5.3, 0.5, 7.0];
        assert!((max_f32(&a) - 9.0).abs() < 1e-6);
    }

    #[test]
    fn test_mean_f32() {
        let a: Vec<f32> = (1..=100).map(|i| i as f32).collect();
        assert!((mean_f32(&a) - 50.5).abs() < 1e-2);
    }

    #[test]
    fn test_empty_slices_f32() {
        assert_eq!(dot_f32(&[], &[]), 0.0);
        assert_eq!(sum_f32(&[]), 0.0);
        assert_eq!(mean_f32(&[]), 0.0);
    }
}
