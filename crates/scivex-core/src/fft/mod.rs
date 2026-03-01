//! Fast Fourier Transform (FFT) module.
//!
//! Provides forward and inverse FFTs for complex and real-valued data.
//! Complex data is represented as tensors with a trailing dimension of size 2
//! (`[..., 2]`) where index 0 is real and index 1 is imaginary (interleaved
//! layout).
//!
//! # Algorithm
//!
//! Uses the Cooley-Tukey radix-2 decimation-in-time algorithm. Non-power-of-2
//! inputs are zero-padded to the next power of two.
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
// Internal helpers
// ---------------------------------------------------------------------------

/// Return the next power of two >= `n`. Returns 1 for `n == 0`.
fn next_power_of_two(n: usize) -> usize {
    n.next_power_of_two()
}

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
        // Iterate over each sub-DFT of size `len`.
        let mut start = 0;
        while start < n {
            for k in 0..half {
                let angle = angle_step * T::from_usize(k);
                let wr = angle.cos();
                let wi = angle.sin();

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

    // Scale by 1/N for inverse transform.
    if inverse {
        let inv_n = T::one() / T::from_usize(n);
        for i in 0..n {
            re[i] *= inv_n;
            im[i] *= inv_n;
        }
    }
}

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
// Public API
// ---------------------------------------------------------------------------

/// Complex-to-complex forward FFT.
///
/// Input shape: `[N, 2]` (interleaved real/imaginary).
/// Output shape: `[N_padded, 2]` where `N_padded` is the next power of two >= N.
///
/// If N is already a power of two, `N_padded == N`.
pub fn fft<T: Float>(input: &Tensor<T>) -> Result<Tensor<T>> {
    if input.ndim() != 2 || input.shape()[1] != 2 {
        return Err(CoreError::InvalidArgument {
            reason: "fft expects input shape [N, 2] (complex interleaved)",
        });
    }
    let n = input.shape()[0];
    let padded = next_power_of_two(n);
    let (mut re, mut im) = unpack_complex(input.as_slice(), n, padded);
    fft_radix2(&mut re, &mut im, false);
    pack_complex(&re, &im, padded)
}

/// Complex-to-complex inverse FFT.
///
/// Input shape: `[N, 2]`. Output shape: `[N_padded, 2]`.
pub fn ifft<T: Float>(input: &Tensor<T>) -> Result<Tensor<T>> {
    if input.ndim() != 2 || input.shape()[1] != 2 {
        return Err(CoreError::InvalidArgument {
            reason: "ifft expects input shape [N, 2] (complex interleaved)",
        });
    }
    let n = input.shape()[0];
    let padded = next_power_of_two(n);
    let (mut re, mut im) = unpack_complex(input.as_slice(), n, padded);
    fft_radix2(&mut re, &mut im, true);
    pack_complex(&re, &im, padded)
}

/// Real-to-complex forward FFT.
///
/// Input shape: `[N]` (real-valued signal).
/// Output shape: `[N_padded/2 + 1, 2]` (complex spectrum, only non-negative
/// frequencies due to Hermitian symmetry).
pub fn rfft<T: Float>(input: &Tensor<T>) -> Result<Tensor<T>> {
    if input.ndim() != 1 {
        return Err(CoreError::InvalidArgument {
            reason: "rfft expects a 1-D real input",
        });
    }
    let n = input.shape()[0];
    let padded = next_power_of_two(n);
    let data = input.as_slice();
    let mut re = vec![T::zero(); padded];
    let mut im = vec![T::zero(); padded];
    for (i, &v) in data.iter().enumerate() {
        re[i] = v;
    }
    fft_radix2(&mut re, &mut im, false);

    let out_len = padded / 2 + 1;
    pack_complex(&re[..out_len], &im[..out_len], out_len)
}

/// Complex-to-real inverse FFT.
///
/// Input shape: `[N/2+1, 2]` (non-negative frequency complex spectrum).
/// Output shape: `[n]` where `n` is the desired output length.
///
/// `n` must be a power of two and consistent with the input length
/// (`input.shape()[0] == n/2 + 1`).
pub fn irfft<T: Float>(input: &Tensor<T>, n: usize) -> Result<Tensor<T>> {
    if input.ndim() != 2 || input.shape()[1] != 2 {
        return Err(CoreError::InvalidArgument {
            reason: "irfft expects input shape [N/2+1, 2]",
        });
    }
    let half_plus_1 = input.shape()[0];
    let padded = next_power_of_two(n);
    if half_plus_1 != padded / 2 + 1 {
        return Err(CoreError::InvalidArgument {
            reason: "irfft: input length must equal n_padded/2 + 1",
        });
    }

    let data = input.as_slice();
    let mut re = vec![T::zero(); padded];
    let mut im = vec![T::zero(); padded];

    // Fill non-negative frequencies.
    for i in 0..half_plus_1 {
        re[i] = data[i * 2];
        im[i] = data[i * 2 + 1];
    }
    // Reconstruct negative frequencies via Hermitian symmetry.
    for i in 1..padded / 2 {
        re[padded - i] = re[i];
        im[padded - i] = -im[i];
    }

    fft_radix2(&mut re, &mut im, true);

    // Return only the first `n` real values.
    Tensor::from_vec(re[..n].to_vec(), vec![n])
}

/// 2-D complex-to-complex forward FFT.
///
/// Input shape: `[M, N, 2]`. Output shape: `[M_padded, N_padded, 2]`.
pub fn fft2<T: Float>(input: &Tensor<T>) -> Result<Tensor<T>> {
    if input.ndim() != 3 || input.shape()[2] != 2 {
        return Err(CoreError::InvalidArgument {
            reason: "fft2 expects input shape [M, N, 2]",
        });
    }
    let rows = input.shape()[0];
    let cols = input.shape()[1];
    let pr = next_power_of_two(rows);
    let pc = next_power_of_two(cols);
    let data = input.as_slice();

    // Build padded 2D arrays (row-major).
    let mut re = vec![T::zero(); pr * pc];
    let mut im = vec![T::zero(); pr * pc];
    for r in 0..rows {
        for c in 0..cols {
            let src = (r * cols + c) * 2;
            re[r * pc + c] = data[src];
            im[r * pc + c] = data[src + 1];
        }
    }

    // FFT along each row.
    for r in 0..pr {
        let start = r * pc;
        let end = start + pc;
        fft_radix2(&mut re[start..end], &mut im[start..end], false);
    }

    // FFT along each column (extract column, FFT, write back).
    let mut col_re = vec![T::zero(); pr];
    let mut col_im = vec![T::zero(); pr];
    for c in 0..pc {
        for r in 0..pr {
            col_re[r] = re[r * pc + c];
            col_im[r] = im[r * pc + c];
        }
        fft_radix2(&mut col_re, &mut col_im, false);
        for r in 0..pr {
            re[r * pc + c] = col_re[r];
            im[r * pc + c] = col_im[r];
        }
    }

    // Pack into [pr, pc, 2].
    let mut out = vec![T::zero(); pr * pc * 2];
    for r in 0..pr {
        for c in 0..pc {
            let idx = (r * pc + c) * 2;
            out[idx] = re[r * pc + c];
            out[idx + 1] = im[r * pc + c];
        }
    }
    Tensor::from_vec(out, vec![pr, pc, 2])
}

/// 2-D complex-to-complex inverse FFT.
///
/// Input shape: `[M, N, 2]`. Output shape: `[M_padded, N_padded, 2]`.
pub fn ifft2<T: Float>(input: &Tensor<T>) -> Result<Tensor<T>> {
    if input.ndim() != 3 || input.shape()[2] != 2 {
        return Err(CoreError::InvalidArgument {
            reason: "ifft2 expects input shape [M, N, 2]",
        });
    }
    let rows = input.shape()[0];
    let cols = input.shape()[1];
    let pr = next_power_of_two(rows);
    let pc = next_power_of_two(cols);
    let data = input.as_slice();

    let mut re = vec![T::zero(); pr * pc];
    let mut im = vec![T::zero(); pr * pc];
    for r in 0..rows {
        for c in 0..cols {
            let src = (r * cols + c) * 2;
            re[r * pc + c] = data[src];
            im[r * pc + c] = data[src + 1];
        }
    }

    // Inverse FFT along each row.
    for r in 0..pr {
        let start = r * pc;
        let end = start + pc;
        fft_radix2(&mut re[start..end], &mut im[start..end], true);
    }

    // Inverse FFT along each column.
    let mut col_re = vec![T::zero(); pr];
    let mut col_im = vec![T::zero(); pr];
    for c in 0..pc {
        for r in 0..pr {
            col_re[r] = re[r * pc + c];
            col_im[r] = im[r * pc + c];
        }
        fft_radix2(&mut col_re, &mut col_im, true);
        for r in 0..pr {
            re[r * pc + c] = col_re[r];
            im[r * pc + c] = col_im[r];
        }
    }

    let mut out = vec![T::zero(); pr * pc * 2];
    for r in 0..pr {
        for c in 0..pc {
            let idx = (r * pc + c) * 2;
            out[idx] = re[r * pc + c];
            out[idx + 1] = im[r * pc + c];
        }
    }
    Tensor::from_vec(out, vec![pr, pc, 2])
}

/// 2-D real-to-complex forward FFT.
///
/// Input shape: `[M, N]` (real matrix).
/// Output shape: `[M_padded, N_padded/2 + 1, 2]`.
pub fn rfft2<T: Float>(input: &Tensor<T>) -> Result<Tensor<T>> {
    if input.ndim() != 2 {
        return Err(CoreError::InvalidArgument {
            reason: "rfft2 expects a 2-D real input",
        });
    }
    let rows = input.shape()[0];
    let cols = input.shape()[1];
    let pr = next_power_of_two(rows);
    let pc = next_power_of_two(cols);
    let data = input.as_slice();

    // Build padded real 2D array.
    let mut re = vec![T::zero(); pr * pc];
    let mut im = vec![T::zero(); pr * pc];
    for r in 0..rows {
        for c in 0..cols {
            re[r * pc + c] = data[r * cols + c];
        }
    }

    // FFT along each row (full complex).
    for r in 0..pr {
        let start = r * pc;
        let end = start + pc;
        fft_radix2(&mut re[start..end], &mut im[start..end], false);
    }

    // FFT along each column.
    let mut col_re = vec![T::zero(); pr];
    let mut col_im = vec![T::zero(); pr];
    for c in 0..pc {
        for r in 0..pr {
            col_re[r] = re[r * pc + c];
            col_im[r] = im[r * pc + c];
        }
        fft_radix2(&mut col_re, &mut col_im, false);
        for r in 0..pr {
            re[r * pc + c] = col_re[r];
            im[r * pc + c] = col_im[r];
        }
    }

    // Only keep non-negative column frequencies: N/2+1.
    let out_cols = pc / 2 + 1;
    let mut out = vec![T::zero(); pr * out_cols * 2];
    for r in 0..pr {
        for c in 0..out_cols {
            let idx = (r * out_cols + c) * 2;
            out[idx] = re[r * pc + c];
            out[idx + 1] = im[r * pc + c];
        }
    }
    Tensor::from_vec(out, vec![pr, out_cols, 2])
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < TOL
    }

    // ---- fft / ifft round-trip ----

    #[test]
    fn test_fft_ifft_roundtrip() {
        // Complex input: [(1,0), (0,1), (-1,0), (0,-1)]
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
        assert_eq!(spectrum.shape(), &[3, 2]); // N/2+1 = 3
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
        // All-ones complex signal => DC bin = N, rest = 0.
        let n = 4;
        let data = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let input = Tensor::from_vec(data, vec![n, 2]).unwrap();
        let spectrum = fft(&input).unwrap();
        let s = spectrum.as_slice();
        // Bin 0 (DC): real = 4, imag = 0
        assert!(approx_eq(s[0], 4.0));
        assert!(approx_eq(s[1], 0.0));
        // All other bins should be zero.
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
        // Impulse at index 0 => flat spectrum (all bins = 1+0j).
        let n = 8;
        let mut data = vec![0.0; n * 2];
        data[0] = 1.0; // re[0] = 1
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
        // e^{2*pi*i*k*n/N} for k=1 => bin 1 = N, rest = 0.
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
        // Bin 1 should have magnitude N.
        assert!(approx_eq(s[2], n as f64), "bin 1 real = {}", s[2]);
        assert!(approx_eq(s[3], 0.0), "bin 1 imag = {}", s[3]);
        // Other bins should be ~0.
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
        // cos(2*pi*k/N) for k=2, N=8 => bins 2 and 6 get N/2.
        // rfft only returns bins 0..N/2, so bin 2 should have real = N/2 = 4.
        let n = 8;
        let data: Vec<f64> = (0..n)
            .map(|j| (2.0 * std::f64::consts::PI * 2.0 * j as f64 / n as f64).cos())
            .collect();
        let input = Tensor::from_vec(data, vec![n]).unwrap();
        let spectrum = rfft(&input).unwrap();
        let s = spectrum.as_slice();
        // Bin 2: real = N/2 = 4, imag = 0.
        assert!(approx_eq(s[4], 4.0), "bin 2 real = {}", s[4]);
        assert!(approx_eq(s[5], 0.0), "bin 2 imag = {}", s[5]);
    }

    // ---- Parseval's theorem ----

    #[test]
    fn test_parseval_theorem() {
        // Sum of |x[n]|^2 == (1/N) * Sum of |X[k]|^2
        let data = vec![1.0, 0.5, 0.0, -0.5, -1.0, -0.5, 0.0, 0.5];
        let n = data.len();
        let time_energy: f64 = data.iter().map(|&x| x * x).sum();

        let input = Tensor::from_vec(data, vec![n]).unwrap();
        // Use full complex FFT via rfft trick: make complex input.
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

    // ---- Non-power-of-2 input (auto zero-pad) ----

    #[test]
    fn test_fft_non_power_of_two() {
        // N=6 should be zero-padded to 8.
        let data = vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0, 6.0, 0.0];
        let input = Tensor::from_vec(data, vec![6, 2]).unwrap();
        let spectrum = fft(&input).unwrap();
        // Output should be padded to 8.
        assert_eq!(spectrum.shape(), &[8, 2]);
    }

    #[test]
    fn test_rfft_non_power_of_two() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let input = Tensor::from_vec(data, vec![7]).unwrap();
        let spectrum = rfft(&input).unwrap();
        // Padded to 8, so output is 8/2+1 = 5.
        assert_eq!(spectrum.shape(), &[5, 2]);
    }

    // ---- fft2 / ifft2 round-trip ----

    #[test]
    fn test_fft2_ifft2_roundtrip() {
        let data = vec![
            1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, // row 0
            5.0, 0.0, 6.0, 0.0, 7.0, 0.0, 8.0, 0.0, // row 1
            9.0, 0.0, 10.0, 0.0, 11.0, 0.0, 12.0, 0.0, // row 2
            13.0, 0.0, 14.0, 0.0, 15.0, 0.0, 16.0, 0.0, // row 3
        ];
        let input = Tensor::from_vec(data.clone(), vec![4, 4, 2]).unwrap();
        let spectrum = fft2(&input).unwrap();
        assert_eq!(spectrum.shape(), &[4, 4, 2]);
        let recovered = ifft2(&spectrum).unwrap();
        for (a, b) in recovered.as_slice().iter().zip(data.iter()) {
            assert!(approx_eq(*a, *b), "got {a}, expected {b}");
        }
    }

    // ---- rfft2 basic ----

    #[test]
    fn test_rfft2_shape() {
        let data: Vec<f64> = (0..16).map(f64::from).collect();
        let input = Tensor::from_vec(data, vec![4, 4]).unwrap();
        let spectrum = rfft2(&input).unwrap();
        // 4 rows, 4/2+1 = 3 columns, 2 (re/im).
        assert_eq!(spectrum.shape(), &[4, 3, 2]);
    }

    #[test]
    fn test_rfft2_dc() {
        // Constant 2D signal => only DC bin is non-zero.
        let data = vec![1.0; 4 * 4];
        let input = Tensor::from_vec(data, vec![4, 4]).unwrap();
        let spectrum = rfft2(&input).unwrap();
        let s = spectrum.as_slice();
        // DC bin at [0, 0] should be M*N = 16.
        assert!(approx_eq(s[0], 16.0), "DC real = {}", s[0]);
        assert!(approx_eq(s[1], 0.0), "DC imag = {}", s[1]);
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
        // Input [3, 2] expects n=4 (3 == 4/2+1), but we pass n=8.
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
