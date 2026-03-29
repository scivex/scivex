//! Fast Fourier Transform (FFT) module.
//!
//! Provides forward and inverse FFTs for complex and real-valued data.
//! Complex data is represented as tensors with a trailing dimension of size 2
//! (`[..., 2]`) where index 0 is real and index 1 is imaginary (interleaved
//! layout).
//!
//! # Algorithm
//!
//! Uses a mixed-radix Cooley-Tukey algorithm with radix-2/3/5/7 butterflies
//! for composite sizes. Arbitrary (including prime) lengths are handled via
//! Bluestein's chirp-z transform, which reduces to a power-of-2 convolution.
//!
//! # Examples
//!
//! ```
//! use scivex_core::tensor::Tensor;
//! use scivex_core::fft;
//!
//! // Real-to-complex FFT of a simple signal
//! let signal = Tensor::from_vec(vec![1.0, 0.0, -1.0, 0.0], vec![4]).unwrap();
//! let spectrum = fft::rfft(&signal).unwrap();
//! assert_eq!(spectrum.shape(), &[3, 2]); // N/2+1 complex bins
//! ```

use crate::Float;
use crate::error::{CoreError, Result};
use crate::tensor::Tensor;

// ---------------------------------------------------------------------------
// Internal helpers — factorisation & dispatch
// ---------------------------------------------------------------------------

/// Supported small-prime radices for mixed-radix FFT.
const SMALL_PRIMES: [usize; 4] = [2, 3, 5, 7];

/// Return the next power of two >= `n`. Returns 1 for `n == 0`.
fn next_power_of_two(n: usize) -> usize {
    n.next_power_of_two()
}

/// Factor `n` into small primes (2, 3, 5, 7). Returns the factors in
/// ascending order, or `None` if `n` has a prime factor > 7.
fn factor_small(mut n: usize) -> Option<Vec<usize>> {
    if n <= 1 {
        return Some(vec![]);
    }
    let mut factors = Vec::new();
    for &p in &SMALL_PRIMES {
        #[allow(clippy::manual_is_multiple_of)]
        while n % p == 0 {
            factors.push(p);
            n /= p;
        }
    }
    if n == 1 { Some(factors) } else { None }
}

/// General-purpose in-place FFT for arbitrary length.
///
/// Dispatches to radix-2, mixed-radix, or Bluestein's depending on `n`.
fn fft_general<T: Float>(re: &mut [T], im: &mut [T], inverse: bool) {
    let n = re.len();
    debug_assert_eq!(n, im.len());
    if n <= 1 {
        if inverse && n == 1 {
            // 1/N scaling is a no-op for N=1.
        }
        return;
    }

    if n.is_power_of_two() {
        fft_radix2(re, im, inverse);
    } else if let Some(factors) = factor_small(n) {
        fft_mixed_radix(re, im, &factors, inverse);
    } else {
        fft_bluestein(re, im, inverse);
    }
}

// ---------------------------------------------------------------------------
// Radix-2 Cooley-Tukey DIT
// ---------------------------------------------------------------------------

/// In-place bit-reversal permutation on parallel real/imaginary slices.
fn bit_reverse_permutation<T: Float>(re: &mut [T], im: &mut [T]) {
    let n = re.len();
    debug_assert!(n.is_power_of_two());
    let bits = n.trailing_zeros();
    for i in 0..n {
        let j = i.reverse_bits() >> (usize::BITS - bits);
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
}

/// In-place Cooley-Tukey radix-2 DIT FFT on separate real/imaginary slices.
///
/// `re` and `im` must have the same length which must be a power of two.
/// If `inverse` is true, computes the inverse FFT (with 1/N scaling).
fn fft_radix2<T: Float>(re: &mut [T], im: &mut [T], inverse: bool) {
    let n = re.len();
    debug_assert_eq!(n, im.len());
    debug_assert!(n.is_power_of_two());

    if n <= 1 {
        return;
    }

    bit_reverse_permutation(re, im);

    let sign = if inverse { T::one() } else { -T::one() };
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let angle_step = sign * T::pi() / T::from_usize(half);

        // Precompute twiddle factors for this stage to avoid redundant trig calls.
        let twiddles_re: Vec<T> = (0..half)
            .map(|k| (angle_step * T::from_usize(k)).cos())
            .collect();
        let twiddles_im: Vec<T> = (0..half)
            .map(|k| (angle_step * T::from_usize(k)).sin())
            .collect();

        let mut start = 0;
        while start < n {
            for k in 0..half {
                let wr = twiddles_re[k];
                let wi = twiddles_im[k];

                let even = start + k;
                let odd = start + k + half;

                let tr = wr * re[odd] - wi * im[odd];
                let ti = wr * im[odd] + wi * re[odd];

                re[odd] = re[even] - tr;
                im[odd] = im[even] - ti;
                re[even] += tr;
                im[even] += ti;
            }
            start += len;
        }
        len *= 2;
    }

    if inverse {
        let inv_n = T::one() / T::from_usize(n);
        for i in 0..n {
            re[i] *= inv_n;
            im[i] *= inv_n;
        }
    }
}

// ---------------------------------------------------------------------------
// Mixed-radix Cooley-Tukey (radix-2/3/5/7)
// ---------------------------------------------------------------------------

/// In-place mixed-radix Cooley-Tukey FFT.
///
/// `factors` lists the prime factorisation of `n` in ascending order.
/// Uses decimation-in-frequency: at each stage, splits by the smallest factor.
#[allow(clippy::needless_range_loop)]
fn fft_mixed_radix<T: Float>(re: &mut [T], im: &mut [T], factors: &[usize], inverse: bool) {
    let n = re.len();
    debug_assert_eq!(n, im.len());

    // We implement the Cooley-Tukey decomposition iteratively using
    // a stockham-style approach: at each stage, factor out the smallest
    // radix and perform the butterfly.
    //
    // N = r0 * r1 * ... * r_{k-1}
    //
    // Stage s processes radix r_s. Before stage s:
    //   product of already-processed radices = "processed"
    //   remaining = N / processed
    //
    // We use out-of-place stages with ping-pong buffers.

    let sign: T = if inverse { T::one() } else { -T::one() };

    let mut src_re = re.to_vec();
    let mut src_im = im.to_vec();
    let mut dst_re = vec![T::zero(); n];
    let mut dst_im = vec![T::zero(); n];

    let mut stride = n; // distance between elements in the same DFT at current stage

    for &radix in factors {
        stride /= radix;
        // At this stage we have `n / (radix * stride)` groups, each of size
        // `radix * stride`, and within each group we do `stride` butterflies.
        let num_groups = n / (radix * stride);

        for g in 0..num_groups {
            for s in 0..stride {
                // Gather the `radix` inputs for this butterfly.
                // Input index: g * radix * stride + r * stride + s
                // for r = 0..radix
                if radix == 2 {
                    // Specialised radix-2 butterfly (avoids allocation).
                    let i0 = g * 2 * stride + s;
                    let i1 = i0 + stride;
                    let a_re = src_re[i0];
                    let a_im = src_im[i0];
                    let b_re = src_re[i1];
                    let b_im = src_im[i1];

                    let angle = sign * T::from_f64(2.0) * T::pi() * T::from_usize(s * num_groups)
                        / T::from_usize(n);
                    let wr = angle.cos();
                    let wi = angle.sin();

                    let d_re = a_re - b_re;
                    let d_im = a_im - b_im;
                    let tw_re = d_re * wr - d_im * wi;
                    let tw_im = d_re * wi + d_im * wr;

                    let o0 = g * stride + s;
                    let o1 = (g + num_groups) * stride + s;
                    dst_re[o0] = a_re + b_re;
                    dst_im[o0] = a_im + b_im;
                    dst_re[o1] = tw_re;
                    dst_im[o1] = tw_im;
                } else {
                    // General radix-r butterfly
                    let mut inp_re = vec![T::zero(); radix];
                    let mut inp_im = vec![T::zero(); radix];
                    for r in 0..radix {
                        let idx = g * radix * stride + r * stride + s;
                        inp_re[r] = src_re[idx];
                        inp_im[r] = src_im[idx];
                    }

                    for o in 0..radix {
                        let mut sum_re = T::zero();
                        let mut sum_im = T::zero();
                        for r in 0..radix {
                            let angle_small =
                                sign * T::from_f64(2.0) * T::pi() * T::from_usize(o * r)
                                    / T::from_usize(radix);
                            let wr_s = angle_small.cos();
                            let wi_s = angle_small.sin();
                            sum_re += inp_re[r] * wr_s - inp_im[r] * wi_s;
                            sum_im += inp_re[r] * wi_s + inp_im[r] * wr_s;
                        }

                        if o > 0 {
                            let angle_tw = sign
                                * T::from_f64(2.0)
                                * T::pi()
                                * T::from_usize(s * num_groups * o)
                                / T::from_usize(n);
                            let wr_t = angle_tw.cos();
                            let wi_t = angle_tw.sin();
                            let tmp_re = sum_re * wr_t - sum_im * wi_t;
                            let tmp_im = sum_re * wi_t + sum_im * wr_t;
                            sum_re = tmp_re;
                            sum_im = tmp_im;
                        }

                        let out_idx = (g + num_groups * o) * stride + s;
                        dst_re[out_idx] = sum_re;
                        dst_im[out_idx] = sum_im;
                    }
                }
            }
        }

        // Swap src and dst.
        std::mem::swap(&mut src_re, &mut dst_re);
        std::mem::swap(&mut src_im, &mut dst_im);
    }

    // Copy result back.
    re.copy_from_slice(&src_re);
    im.copy_from_slice(&src_im);

    if inverse {
        let inv_n = T::one() / T::from_usize(n);
        for i in 0..n {
            re[i] *= inv_n;
            im[i] *= inv_n;
        }
    }
}

// ---------------------------------------------------------------------------
// Bluestein's chirp-z transform (arbitrary length)
// ---------------------------------------------------------------------------

/// In-place FFT for arbitrary length via Bluestein's algorithm.
///
/// Converts a length-N DFT into a circular convolution of length M (power of 2,
/// M >= 2N - 1), computed via radix-2 FFT.
fn fft_bluestein<T: Float>(re: &mut [T], im: &mut [T], inverse: bool) {
    let n = re.len();
    debug_assert_eq!(n, im.len());
    debug_assert!(n > 1);

    let sign: T = if inverse { T::one() } else { -T::one() };

    // Chirp sequence: w_k = exp(sign * i * pi * k^2 / N) for k = 0..N-1
    let mut chirp_re = vec![T::zero(); n];
    let mut chirp_im = vec![T::zero(); n];
    for k in 0..n {
        let angle = sign * T::pi() * T::from_usize(k * k) / T::from_usize(n);
        chirp_re[k] = angle.cos();
        chirp_im[k] = angle.sin();
    }

    // Convolution size: next power of 2 >= 2N - 1
    let m = next_power_of_two(2 * n - 1);

    // Sequence a: a_k = x_k * conj(chirp_k), zero-padded to m
    let mut a_re = vec![T::zero(); m];
    let mut a_im = vec![T::zero(); m];
    for k in 0..n {
        // conj(chirp) = (chirp_re, -chirp_im)
        // a = x * conj(chirp)
        a_re[k] = re[k] * chirp_re[k] + im[k] * chirp_im[k];
        a_im[k] = im[k] * chirp_re[k] - re[k] * chirp_im[k];
    }

    // Sequence b: b_k = chirp_k for k = 0..N-1,
    //             b_{m-k} = chirp_k for k = 1..N-1 (circular wrap),
    //             rest = 0
    let mut b_re = vec![T::zero(); m];
    let mut b_im = vec![T::zero(); m];
    b_re[0] = chirp_re[0];
    b_im[0] = chirp_im[0];
    for k in 1..n {
        b_re[k] = chirp_re[k];
        b_im[k] = chirp_im[k];
        b_re[m - k] = chirp_re[k];
        b_im[m - k] = chirp_im[k];
    }

    // Convolve a and b via FFT: result = IFFT(FFT(a) * FFT(b))
    fft_radix2(&mut a_re, &mut a_im, false);
    fft_radix2(&mut b_re, &mut b_im, false);

    // Pointwise complex multiply
    for i in 0..m {
        let pr = a_re[i] * b_re[i] - a_im[i] * b_im[i];
        let pi = a_re[i] * b_im[i] + a_im[i] * b_re[i];
        a_re[i] = pr;
        a_im[i] = pi;
    }

    fft_radix2(&mut a_re, &mut a_im, true);

    // Extract result: X_k = chirp_conj_k * conv_k
    for k in 0..n {
        // conj(chirp)
        let cr = chirp_re[k];
        let ci = -chirp_im[k];
        re[k] = a_re[k] * cr - a_im[k] * ci;
        im[k] = a_re[k] * ci + a_im[k] * cr;
    }

    if inverse {
        let inv_n = T::one() / T::from_usize(n);
        for k in 0..n {
            re[k] *= inv_n;
            im[k] *= inv_n;
        }
    }
}

// ---------------------------------------------------------------------------
// Packing helpers
// ---------------------------------------------------------------------------

/// Pack interleaved `[N, 2]` tensor data into separate real/imaginary `Vec`s,
/// zero-padding to `padded_n`.
fn unpack_complex<T: Float>(data: &[T], n: usize, padded_n: usize) -> (Vec<T>, Vec<T>) {
    let mut re = vec![T::zero(); padded_n];
    let mut im = vec![T::zero(); padded_n];
    for i in 0..n {
        re[i] = data[i * 2];
        im[i] = data[i * 2 + 1];
    }
    (re, im)
}

/// Pack separate real/imaginary slices into an interleaved `[N, 2]` tensor.
fn pack_complex<T: Float>(re: &[T], im: &[T], n: usize) -> Result<Tensor<T>> {
    let mut data = vec![T::zero(); n * 2];
    for i in 0..n {
        data[i * 2] = re[i];
        data[i * 2 + 1] = im[i];
    }
    Tensor::from_vec(data, vec![n, 2])
}

// ---------------------------------------------------------------------------
// Public API — 1-D transforms
// ---------------------------------------------------------------------------

/// Complex-to-complex forward FFT.
///
/// Input shape: `[N, 2]` (interleaved real/imaginary).
/// Output shape: `[N, 2]`. Supports arbitrary `N` (not just powers of two).
///
/// # Examples
///
/// ```
/// # use scivex_core::tensor::Tensor;
/// # use scivex_core::fft;
/// let input = Tensor::from_vec(vec![1.0, 0.0, 0.0, 0.0], vec![2, 2]).unwrap();
/// let spectrum = fft::fft(&input).unwrap();
/// assert_eq!(spectrum.shape(), &[2, 2]);
/// ```
pub fn fft<T: Float>(input: &Tensor<T>) -> Result<Tensor<T>> {
    if input.ndim() != 2 || input.shape()[1] != 2 {
        return Err(CoreError::InvalidArgument {
            reason: "fft expects input shape [N, 2] (complex interleaved)",
        });
    }
    let n = input.shape()[0];
    let (mut re, mut im) = unpack_complex(input.as_slice(), n, n);
    fft_general(&mut re, &mut im, false);
    pack_complex(&re, &im, n)
}

/// Complex-to-complex inverse FFT.
///
/// Input shape: `[N, 2]`. Output shape: `[N, 2]`.
///
/// # Examples
///
/// ```
/// # use scivex_core::tensor::Tensor;
/// # use scivex_core::fft;
/// let input = Tensor::from_vec(vec![2.0, 0.0, 0.0, 0.0], vec![2, 2]).unwrap();
/// let result = fft::ifft(&input).unwrap();
/// assert_eq!(result.shape(), &[2, 2]);
/// ```
pub fn ifft<T: Float>(input: &Tensor<T>) -> Result<Tensor<T>> {
    if input.ndim() != 2 || input.shape()[1] != 2 {
        return Err(CoreError::InvalidArgument {
            reason: "ifft expects input shape [N, 2] (complex interleaved)",
        });
    }
    let n = input.shape()[0];
    let (mut re, mut im) = unpack_complex(input.as_slice(), n, n);
    fft_general(&mut re, &mut im, true);
    pack_complex(&re, &im, n)
}

/// Real-to-complex forward FFT.
///
/// Input shape: `[N]` (real-valued signal).
/// Output shape: `[N/2 + 1, 2]` (complex spectrum, only non-negative
/// frequencies due to Hermitian symmetry).
///
/// # Examples
///
/// ```
/// # use scivex_core::tensor::Tensor;
/// # use scivex_core::fft;
/// let signal = Tensor::from_vec(vec![1.0, 0.0, -1.0, 0.0], vec![4]).unwrap();
/// let spectrum = fft::rfft(&signal).unwrap();
/// assert_eq!(spectrum.shape(), &[3, 2]); // N/2+1 complex bins
/// ```
pub fn rfft<T: Float>(input: &Tensor<T>) -> Result<Tensor<T>> {
    if input.ndim() != 1 {
        return Err(CoreError::InvalidArgument {
            reason: "rfft expects a 1-D real input",
        });
    }
    let n = input.shape()[0];
    let data = input.as_slice();
    let mut re = vec![T::zero(); n];
    let mut im = vec![T::zero(); n];
    for (i, &v) in data.iter().enumerate() {
        re[i] = v;
    }
    fft_general(&mut re, &mut im, false);

    let out_len = n / 2 + 1;
    pack_complex(&re[..out_len], &im[..out_len], out_len)
}

/// Complex-to-real inverse FFT.
///
/// Input shape: `[N/2+1, 2]` (non-negative frequency complex spectrum).
/// Output shape: `[n]` where `n` is the desired output length.
///
/// `input.shape()[0]` must equal `n/2 + 1`.
///
/// # Examples
///
/// ```
/// # use scivex_core::tensor::Tensor;
/// # use scivex_core::fft;
/// let spectrum = Tensor::from_vec(vec![10.0, 0.0, -2.0, 0.0, 2.0, 0.0], vec![3, 2]).unwrap();
/// let signal = fft::irfft(&spectrum, 4).unwrap();
/// assert_eq!(signal.shape(), &[4]);
/// ```
pub fn irfft<T: Float>(input: &Tensor<T>, n: usize) -> Result<Tensor<T>> {
    if input.ndim() != 2 || input.shape()[1] != 2 {
        return Err(CoreError::InvalidArgument {
            reason: "irfft expects input shape [N/2+1, 2]",
        });
    }
    let half_plus_1 = input.shape()[0];
    if half_plus_1 != n / 2 + 1 {
        return Err(CoreError::InvalidArgument {
            reason: "irfft: input length must equal n/2 + 1",
        });
    }

    let data = input.as_slice();
    let mut re = vec![T::zero(); n];
    let mut im = vec![T::zero(); n];

    // Fill non-negative frequencies.
    for i in 0..half_plus_1 {
        re[i] = data[i * 2];
        im[i] = data[i * 2 + 1];
    }
    // Reconstruct negative frequencies via Hermitian symmetry: X[k] = conj(X[N-k]).
    for k in half_plus_1..n {
        re[k] = re[n - k];
        im[k] = -im[n - k];
    }

    fft_general(&mut re, &mut im, true);

    Tensor::from_vec(re, vec![n])
}

// ---------------------------------------------------------------------------
// Public API — 2-D transforms
// ---------------------------------------------------------------------------

/// 2-D complex-to-complex forward FFT.
///
/// Input shape: `[M, N, 2]`. Output shape: `[M, N, 2]`.
///
/// # Examples
///
/// ```
/// # use scivex_core::tensor::Tensor;
/// # use scivex_core::fft;
/// let input = Tensor::from_vec(vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0], vec![2, 2, 2]).unwrap();
/// let spectrum = fft::fft2(&input).unwrap();
/// assert_eq!(spectrum.shape(), &[2, 2, 2]);
/// ```
pub fn fft2<T: Float>(input: &Tensor<T>) -> Result<Tensor<T>> {
    if input.ndim() != 3 || input.shape()[2] != 2 {
        return Err(CoreError::InvalidArgument {
            reason: "fft2 expects input shape [M, N, 2]",
        });
    }
    let rows = input.shape()[0];
    let cols = input.shape()[1];
    let data = input.as_slice();

    let mut re = vec![T::zero(); rows * cols];
    let mut im = vec![T::zero(); rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let src = (r * cols + c) * 2;
            re[r * cols + c] = data[src];
            im[r * cols + c] = data[src + 1];
        }
    }

    // FFT along each row.
    for r in 0..rows {
        let start = r * cols;
        let end = start + cols;
        fft_general(&mut re[start..end], &mut im[start..end], false);
    }

    // FFT along each column.
    let mut col_re = vec![T::zero(); rows];
    let mut col_im = vec![T::zero(); rows];
    for c in 0..cols {
        for r in 0..rows {
            col_re[r] = re[r * cols + c];
            col_im[r] = im[r * cols + c];
        }
        fft_general(&mut col_re, &mut col_im, false);
        for r in 0..rows {
            re[r * cols + c] = col_re[r];
            im[r * cols + c] = col_im[r];
        }
    }

    let mut out = vec![T::zero(); rows * cols * 2];
    for r in 0..rows {
        for c in 0..cols {
            let idx = (r * cols + c) * 2;
            out[idx] = re[r * cols + c];
            out[idx + 1] = im[r * cols + c];
        }
    }
    Tensor::from_vec(out, vec![rows, cols, 2])
}

/// 2-D complex-to-complex inverse FFT.
///
/// Input shape: `[M, N, 2]`. Output shape: `[M, N, 2]`.
///
/// # Examples
///
/// ```
/// # use scivex_core::tensor::Tensor;
/// # use scivex_core::fft;
/// let input = Tensor::from_vec(vec![10.0, 0.0, -2.0, 0.0, 2.0, 0.0, 0.0, 0.0], vec![2, 2, 2]).unwrap();
/// let result = fft::ifft2(&input).unwrap();
/// assert_eq!(result.shape(), &[2, 2, 2]);
/// ```
pub fn ifft2<T: Float>(input: &Tensor<T>) -> Result<Tensor<T>> {
    if input.ndim() != 3 || input.shape()[2] != 2 {
        return Err(CoreError::InvalidArgument {
            reason: "ifft2 expects input shape [M, N, 2]",
        });
    }
    let rows = input.shape()[0];
    let cols = input.shape()[1];
    let data = input.as_slice();

    let mut re = vec![T::zero(); rows * cols];
    let mut im = vec![T::zero(); rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let src = (r * cols + c) * 2;
            re[r * cols + c] = data[src];
            im[r * cols + c] = data[src + 1];
        }
    }

    // Inverse FFT along each row.
    for r in 0..rows {
        let start = r * cols;
        let end = start + cols;
        fft_general(&mut re[start..end], &mut im[start..end], true);
    }

    // Inverse FFT along each column.
    let mut col_re = vec![T::zero(); rows];
    let mut col_im = vec![T::zero(); rows];
    for c in 0..cols {
        for r in 0..rows {
            col_re[r] = re[r * cols + c];
            col_im[r] = im[r * cols + c];
        }
        fft_general(&mut col_re, &mut col_im, true);
        for r in 0..rows {
            re[r * cols + c] = col_re[r];
            im[r * cols + c] = col_im[r];
        }
    }

    let mut out = vec![T::zero(); rows * cols * 2];
    for r in 0..rows {
        for c in 0..cols {
            let idx = (r * cols + c) * 2;
            out[idx] = re[r * cols + c];
            out[idx + 1] = im[r * cols + c];
        }
    }
    Tensor::from_vec(out, vec![rows, cols, 2])
}

/// 2-D real-to-complex forward FFT.
///
/// Input shape: `[M, N]` (real matrix).
/// Output shape: `[M, N/2 + 1, 2]`.
///
/// # Examples
///
/// ```
/// # use scivex_core::tensor::Tensor;
/// # use scivex_core::fft;
/// let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
/// let spectrum = fft::rfft2(&input).unwrap();
/// assert_eq!(spectrum.shape(), &[2, 2, 2]); // [M, N/2+1, 2]
/// ```
pub fn rfft2<T: Float>(input: &Tensor<T>) -> Result<Tensor<T>> {
    if input.ndim() != 2 {
        return Err(CoreError::InvalidArgument {
            reason: "rfft2 expects a 2-D real input",
        });
    }
    let rows = input.shape()[0];
    let cols = input.shape()[1];
    let data = input.as_slice();

    let mut re = vec![T::zero(); rows * cols];
    let mut im = vec![T::zero(); rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            re[r * cols + c] = data[r * cols + c];
        }
    }

    // FFT along each row.
    for r in 0..rows {
        let start = r * cols;
        let end = start + cols;
        fft_general(&mut re[start..end], &mut im[start..end], false);
    }

    // FFT along each column.
    let mut col_re = vec![T::zero(); rows];
    let mut col_im = vec![T::zero(); rows];
    for c in 0..cols {
        for r in 0..rows {
            col_re[r] = re[r * cols + c];
            col_im[r] = im[r * cols + c];
        }
        fft_general(&mut col_re, &mut col_im, false);
        for r in 0..rows {
            re[r * cols + c] = col_re[r];
            im[r * cols + c] = col_im[r];
        }
    }

    // Only keep non-negative column frequencies.
    let out_cols = cols / 2 + 1;
    let mut out = vec![T::zero(); rows * out_cols * 2];
    for r in 0..rows {
        for c in 0..out_cols {
            let idx = (r * out_cols + c) * 2;
            out[idx] = re[r * cols + c];
            out[idx + 1] = im[r * cols + c];
        }
    }
    Tensor::from_vec(out, vec![rows, out_cols, 2])
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;
    const TOL_LOOSE: f64 = 1e-6;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < TOL
    }

    fn approx_eq_loose(a: f64, b: f64) -> bool {
        (a - b).abs() < TOL_LOOSE
    }

    // ---- fft / ifft round-trip ----

    #[test]
    fn test_fft_ifft_roundtrip() {
        let data = vec![1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0];
        let input = Tensor::from_vec(data.clone(), vec![4, 2]).unwrap();
        let spectrum = fft(&input).unwrap();
        let recovered = ifft(&spectrum).unwrap();
        assert_eq!(recovered.shape(), &[4, 2]);
        for (a, b) in recovered.as_slice().iter().zip(data.iter()) {
            assert!(approx_eq(*a, *b), "got {a}, expected {b}");
        }
    }

    #[test]
    fn test_fft_ifft_roundtrip_8() {
        let data: Vec<f64> = (0..16).map(|i| f64::from(i) * 0.1).collect();
        let input = Tensor::from_vec(data.clone(), vec![8, 2]).unwrap();
        let spectrum = fft(&input).unwrap();
        let recovered = ifft(&spectrum).unwrap();
        for (a, b) in recovered.as_slice().iter().zip(data.iter()) {
            assert!(approx_eq(*a, *b), "got {a}, expected {b}");
        }
    }

    // ---- rfft / irfft round-trip ----

    #[test]
    fn test_rfft_irfft_roundtrip() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let input = Tensor::from_vec(data.clone(), vec![4]).unwrap();
        let spectrum = rfft(&input).unwrap();
        assert_eq!(spectrum.shape(), &[3, 2]);
        let recovered = irfft(&spectrum, 4).unwrap();
        assert_eq!(recovered.shape(), &[4]);
        for (a, b) in recovered.as_slice().iter().zip(data.iter()) {
            assert!(approx_eq(*a, *b), "got {a}, expected {b}");
        }
    }

    #[test]
    fn test_rfft_irfft_roundtrip_8() {
        let data: Vec<f64> = (0..8).map(|i| f64::from(i).sin()).collect();
        let input = Tensor::from_vec(data.clone(), vec![8]).unwrap();
        let spectrum = rfft(&input).unwrap();
        assert_eq!(spectrum.shape(), &[5, 2]);
        let recovered = irfft(&spectrum, 8).unwrap();
        for (a, b) in recovered.as_slice().iter().zip(data.iter()) {
            assert!(approx_eq(*a, *b), "got {a}, expected {b}");
        }
    }

    // ---- Known signal: DC ----

    #[test]
    fn test_fft_dc_signal() {
        let n = 4;
        let data = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let input = Tensor::from_vec(data, vec![n, 2]).unwrap();
        let spectrum = fft(&input).unwrap();
        let s = spectrum.as_slice();
        assert!(approx_eq(s[0], 4.0));
        assert!(approx_eq(s[1], 0.0));
        for i in 1..n {
            assert!(approx_eq(s[i * 2], 0.0), "bin {i} real = {}", s[i * 2]);
            assert!(
                approx_eq(s[i * 2 + 1], 0.0),
                "bin {i} imag = {}",
                s[i * 2 + 1]
            );
        }
    }

    // ---- Known signal: impulse ----

    #[test]
    fn test_fft_impulse() {
        let n = 8;
        let mut data = vec![0.0; n * 2];
        data[0] = 1.0;
        let input = Tensor::from_vec(data, vec![n, 2]).unwrap();
        let spectrum = fft(&input).unwrap();
        let s = spectrum.as_slice();
        for i in 0..n {
            assert!(approx_eq(s[i * 2], 1.0), "bin {i} real = {}", s[i * 2]);
            assert!(
                approx_eq(s[i * 2 + 1], 0.0),
                "bin {i} imag = {}",
                s[i * 2 + 1]
            );
        }
    }

    // ---- Known signal: single frequency ----

    #[test]
    fn test_fft_single_frequency() {
        let n = 8;
        let mut data = vec![0.0; n * 2];
        for j in 0..n {
            let angle = 2.0 * std::f64::consts::PI * (j as f64) / (n as f64);
            data[j * 2] = angle.cos();
            data[j * 2 + 1] = angle.sin();
        }
        let input = Tensor::from_vec(data, vec![n, 2]).unwrap();
        let spectrum = fft(&input).unwrap();
        let s = spectrum.as_slice();
        assert!(approx_eq(s[2], n as f64), "bin 1 real = {}", s[2]);
        assert!(approx_eq(s[3], 0.0), "bin 1 imag = {}", s[3]);
        for i in 0..n {
            if i == 1 {
                continue;
            }
            assert!(
                s[i * 2].abs() < TOL,
                "bin {i} real = {} (expected ~0)",
                s[i * 2]
            );
            assert!(
                s[i * 2 + 1].abs() < TOL,
                "bin {i} imag = {} (expected ~0)",
                s[i * 2 + 1]
            );
        }
    }

    // ---- rfft of real cosine ----

    #[test]
    fn test_rfft_cosine() {
        let n = 8;
        let data: Vec<f64> = (0..n)
            .map(|j| (2.0 * std::f64::consts::PI * 2.0 * j as f64 / n as f64).cos())
            .collect();
        let input = Tensor::from_vec(data, vec![n]).unwrap();
        let spectrum = rfft(&input).unwrap();
        let s = spectrum.as_slice();
        assert!(approx_eq(s[4], 4.0), "bin 2 real = {}", s[4]);
        assert!(approx_eq(s[5], 0.0), "bin 2 imag = {}", s[5]);
    }

    // ---- Parseval's theorem ----

    #[test]
    fn test_parseval_theorem() {
        let data = vec![1.0, 0.5, 0.0, -0.5, -1.0, -0.5, 0.0, 0.5];
        let n = data.len();
        let time_energy: f64 = data.iter().map(|&x| x * x).sum();

        let input = Tensor::from_vec(data, vec![n]).unwrap();
        let mut cdata = vec![0.0; n * 2];
        for (i, &v) in input.as_slice().iter().enumerate() {
            cdata[i * 2] = v;
        }
        let cinput = Tensor::from_vec(cdata, vec![n, 2]).unwrap();
        let spectrum = fft(&cinput).unwrap();
        let s = spectrum.as_slice();
        let freq_energy: f64 = (0..n)
            .map(|i| s[i * 2] * s[i * 2] + s[i * 2 + 1] * s[i * 2 + 1])
            .sum();
        let freq_energy_scaled = freq_energy / n as f64;

        assert!(
            (time_energy - freq_energy_scaled).abs() < 1e-8,
            "Parseval: time={time_energy}, freq={freq_energy_scaled}"
        );
    }

    // ---- Non-power-of-2 with exact sizes (mixed-radix) ----

    #[test]
    fn test_fft_non_power_of_two_exact_size() {
        // N=6 = 2*3, should now return exact size 6 (not padded to 8).
        let data = vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0, 6.0, 0.0];
        let input = Tensor::from_vec(data, vec![6, 2]).unwrap();
        let spectrum = fft(&input).unwrap();
        assert_eq!(spectrum.shape(), &[6, 2]);
    }

    #[test]
    fn test_rfft_non_power_of_two_exact_size() {
        // N=7 (prime), should return exact 7/2+1 = 4.
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let input = Tensor::from_vec(data, vec![7]).unwrap();
        let spectrum = rfft(&input).unwrap();
        assert_eq!(spectrum.shape(), &[4, 2]);
    }

    // ---- Mixed-radix round-trip tests ----

    #[test]
    fn test_fft_ifft_roundtrip_n6() {
        // N=6 = 2*3 exercises mixed-radix path.
        let data = vec![1.0, 0.0, 2.0, 0.5, 3.0, -1.0, 0.0, 2.0, -1.0, 0.0, 0.5, 1.0];
        let input = Tensor::from_vec(data.clone(), vec![6, 2]).unwrap();
        let spectrum = fft(&input).unwrap();
        let recovered = ifft(&spectrum).unwrap();
        assert_eq!(recovered.shape(), &[6, 2]);
        for (a, b) in recovered.as_slice().iter().zip(data.iter()) {
            assert!(approx_eq_loose(*a, *b), "got {a}, expected {b}");
        }
    }

    #[test]
    fn test_fft_ifft_roundtrip_n12() {
        // N=12 = 2*2*3 exercises mixed-radix.
        let data: Vec<f64> = (0..24).map(|i| f64::from(i) * 0.1 - 1.0).collect();
        let input = Tensor::from_vec(data.clone(), vec![12, 2]).unwrap();
        let spectrum = fft(&input).unwrap();
        let recovered = ifft(&spectrum).unwrap();
        for (a, b) in recovered.as_slice().iter().zip(data.iter()) {
            assert!(approx_eq_loose(*a, *b), "got {a}, expected {b}");
        }
    }

    #[test]
    fn test_fft_ifft_roundtrip_n15() {
        // N=15 = 3*5 exercises mixed-radix with no factor of 2.
        let data: Vec<f64> = (0..30).map(|i| (f64::from(i) * 0.3).sin()).collect();
        let input = Tensor::from_vec(data.clone(), vec![15, 2]).unwrap();
        let spectrum = fft(&input).unwrap();
        let recovered = ifft(&spectrum).unwrap();
        for (a, b) in recovered.as_slice().iter().zip(data.iter()) {
            assert!(approx_eq_loose(*a, *b), "got {a}, expected {b}");
        }
    }

    #[test]
    fn test_fft_dc_n6() {
        // DC signal of length 6: all bins zero except bin 0 = 6.
        let n = 6;
        let mut data = vec![0.0; n * 2];
        for i in 0..n {
            data[i * 2] = 1.0;
        }
        let input = Tensor::from_vec(data, vec![n, 2]).unwrap();
        let spectrum = fft(&input).unwrap();
        let s = spectrum.as_slice();
        assert!(approx_eq_loose(s[0], 6.0), "DC real = {}", s[0]);
        assert!(approx_eq_loose(s[1], 0.0), "DC imag = {}", s[1]);
        for i in 1..n {
            assert!(s[i * 2].abs() < TOL_LOOSE, "bin {i} real = {}", s[i * 2]);
        }
    }

    // ---- Bluestein's tests (prime lengths) ----

    #[test]
    fn test_fft_ifft_roundtrip_n7() {
        // N=7 (prime) exercises Bluestein's path.
        let data: Vec<f64> = (0..14).map(|i| f64::from(i) * 0.2 - 0.5).collect();
        let input = Tensor::from_vec(data.clone(), vec![7, 2]).unwrap();
        let spectrum = fft(&input).unwrap();
        assert_eq!(spectrum.shape(), &[7, 2]);
        let recovered = ifft(&spectrum).unwrap();
        for (a, b) in recovered.as_slice().iter().zip(data.iter()) {
            assert!(approx_eq_loose(*a, *b), "got {a}, expected {b}");
        }
    }

    #[test]
    fn test_fft_ifft_roundtrip_n11() {
        // N=11 (prime) exercises Bluestein's path.
        let data: Vec<f64> = (0..22).map(|i| (f64::from(i) * 0.7).cos()).collect();
        let input = Tensor::from_vec(data.clone(), vec![11, 2]).unwrap();
        let spectrum = fft(&input).unwrap();
        let recovered = ifft(&spectrum).unwrap();
        for (a, b) in recovered.as_slice().iter().zip(data.iter()) {
            assert!(approx_eq_loose(*a, *b), "got {a}, expected {b}");
        }
    }

    #[test]
    fn test_fft_dc_n11() {
        let n = 11;
        let mut data = vec![0.0; n * 2];
        for i in 0..n {
            data[i * 2] = 1.0;
        }
        let input = Tensor::from_vec(data, vec![n, 2]).unwrap();
        let spectrum = fft(&input).unwrap();
        let s = spectrum.as_slice();
        assert!(approx_eq_loose(s[0], 11.0), "DC real = {}", s[0]);
        assert!(approx_eq_loose(s[1], 0.0), "DC imag = {}", s[1]);
        for i in 1..n {
            assert!(s[i * 2].abs() < TOL_LOOSE, "bin {i} real = {}", s[i * 2]);
        }
    }

    #[test]
    fn test_fft_impulse_n13() {
        // Impulse at index 0 => flat spectrum for any N.
        let n = 13;
        let mut data = vec![0.0; n * 2];
        data[0] = 1.0;
        let input = Tensor::from_vec(data, vec![n, 2]).unwrap();
        let spectrum = fft(&input).unwrap();
        let s = spectrum.as_slice();
        for i in 0..n {
            assert!(
                approx_eq_loose(s[i * 2], 1.0),
                "bin {i} real = {}",
                s[i * 2]
            );
            assert!(
                s[i * 2 + 1].abs() < TOL_LOOSE,
                "bin {i} imag = {}",
                s[i * 2 + 1]
            );
        }
    }

    // ---- rfft / irfft for non-power-of-2 ----

    #[test]
    fn test_rfft_irfft_roundtrip_n6() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input = Tensor::from_vec(data.clone(), vec![6]).unwrap();
        let spectrum = rfft(&input).unwrap();
        assert_eq!(spectrum.shape(), &[4, 2]); // 6/2+1 = 4
        let recovered = irfft(&spectrum, 6).unwrap();
        for (a, b) in recovered.as_slice().iter().zip(data.iter()) {
            assert!(approx_eq_loose(*a, *b), "got {a}, expected {b}");
        }
    }

    #[test]
    fn test_rfft_irfft_roundtrip_n7() {
        let data: Vec<f64> = (0..7).map(|i| f64::from(i).sin()).collect();
        let input = Tensor::from_vec(data.clone(), vec![7]).unwrap();
        let spectrum = rfft(&input).unwrap();
        assert_eq!(spectrum.shape(), &[4, 2]); // 7/2+1 = 4
        let recovered = irfft(&spectrum, 7).unwrap();
        for (a, b) in recovered.as_slice().iter().zip(data.iter()) {
            assert!(approx_eq_loose(*a, *b), "got {a}, expected {b}");
        }
    }

    // ---- Parseval's for non-power-of-2 ----

    #[test]
    fn test_parseval_n9() {
        let data: Vec<f64> = (0..9).map(|i| f64::from(i) * 0.3 - 1.0).collect();
        let n = data.len();
        let time_energy: f64 = data.iter().map(|&x| x * x).sum();

        let mut cdata = vec![0.0; n * 2];
        for (i, &v) in data.iter().enumerate() {
            cdata[i * 2] = v;
        }
        let cinput = Tensor::from_vec(cdata, vec![n, 2]).unwrap();
        let spectrum = fft(&cinput).unwrap();
        let s = spectrum.as_slice();
        let freq_energy: f64 = (0..n)
            .map(|i| s[i * 2] * s[i * 2] + s[i * 2 + 1] * s[i * 2 + 1])
            .sum();
        let freq_energy_scaled = freq_energy / n as f64;

        assert!(
            (time_energy - freq_energy_scaled).abs() < 1e-6,
            "Parseval: time={time_energy}, freq={freq_energy_scaled}"
        );
    }

    // ---- 2D non-power-of-2 ----

    #[test]
    fn test_fft2_ifft2_roundtrip() {
        let data = vec![
            1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0, 6.0, 0.0, 7.0, 0.0, 8.0, 0.0, 9.0,
            0.0, 10.0, 0.0, 11.0, 0.0, 12.0, 0.0, 13.0, 0.0, 14.0, 0.0, 15.0, 0.0, 16.0, 0.0,
        ];
        let input = Tensor::from_vec(data.clone(), vec![4, 4, 2]).unwrap();
        let spectrum = fft2(&input).unwrap();
        assert_eq!(spectrum.shape(), &[4, 4, 2]);
        let recovered = ifft2(&spectrum).unwrap();
        for (a, b) in recovered.as_slice().iter().zip(data.iter()) {
            assert!(approx_eq(*a, *b), "got {a}, expected {b}");
        }
    }

    #[test]
    fn test_fft2_ifft2_roundtrip_3x5() {
        // Non-power-of-2 2D: 3x5
        let n = 3 * 5;
        let data: Vec<f64> = (0..n * 2).map(|i| f64::from(i) * 0.1).collect();
        let input = Tensor::from_vec(data.clone(), vec![3, 5, 2]).unwrap();
        let spectrum = fft2(&input).unwrap();
        assert_eq!(spectrum.shape(), &[3, 5, 2]);
        let recovered = ifft2(&spectrum).unwrap();
        for (a, b) in recovered.as_slice().iter().zip(data.iter()) {
            assert!(approx_eq_loose(*a, *b), "got {a}, expected {b}");
        }
    }

    // ---- rfft2 basic ----

    #[test]
    fn test_rfft2_shape() {
        let data: Vec<f64> = (0..16).map(f64::from).collect();
        let input = Tensor::from_vec(data, vec![4, 4]).unwrap();
        let spectrum = rfft2(&input).unwrap();
        assert_eq!(spectrum.shape(), &[4, 3, 2]);
    }

    #[test]
    fn test_rfft2_dc() {
        let data = vec![1.0; 4 * 4];
        let input = Tensor::from_vec(data, vec![4, 4]).unwrap();
        let spectrum = rfft2(&input).unwrap();
        let s = spectrum.as_slice();
        assert!(approx_eq(s[0], 16.0), "DC real = {}", s[0]);
        assert!(approx_eq(s[1], 0.0), "DC imag = {}", s[1]);
    }

    // ---- factor_small tests ----

    #[test]
    fn test_factor_small() {
        assert_eq!(factor_small(1), Some(vec![]));
        assert_eq!(factor_small(2), Some(vec![2]));
        assert_eq!(factor_small(6), Some(vec![2, 3]));
        assert_eq!(factor_small(12), Some(vec![2, 2, 3]));
        assert_eq!(factor_small(30), Some(vec![2, 3, 5]));
        assert_eq!(factor_small(210), Some(vec![2, 3, 5, 7]));
        assert_eq!(factor_small(11), None); // prime > 7
        assert_eq!(factor_small(22), None); // has factor 11
    }

    // ---- Error cases ----

    #[test]
    fn test_fft_wrong_shape() {
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        assert!(fft(&input).is_err());
    }

    #[test]
    fn test_fft_wrong_ndim() {
        let input = Tensor::from_vec(vec![1.0; 24], vec![2, 3, 4]).unwrap();
        assert!(fft(&input).is_err());
    }

    #[test]
    fn test_ifft_wrong_shape() {
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        assert!(ifft(&input).is_err());
    }

    #[test]
    fn test_rfft_wrong_ndim() {
        let input = Tensor::from_vec(vec![1.0; 4], vec![2, 2]).unwrap();
        assert!(rfft(&input).is_err());
    }

    #[test]
    fn test_irfft_wrong_length() {
        let data = vec![0.0; 6];
        let input = Tensor::from_vec(data, vec![3, 2]).unwrap();
        assert!(irfft(&input, 8).is_err());
    }

    #[test]
    fn test_fft2_wrong_ndim() {
        let input = Tensor::from_vec(vec![1.0; 4], vec![2, 2]).unwrap();
        assert!(fft2(&input).is_err());
    }

    #[test]
    fn test_rfft2_wrong_ndim() {
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        assert!(rfft2(&input).is_err());
    }
}
