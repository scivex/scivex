//! 1-D convolution and cross-correlation.

use scivex_core::{Float, Tensor, fft};

use crate::error::{Result, SignalError};

/// Output mode for convolution and correlation.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvolveMode {
    /// Full convolution: output length = `len(a) + len(b) - 1`.
    Full,
    /// Output has the same length as the first input.
    Same,
    /// Only elements computed without zero-padding.
    /// Output length = `max(len(a), len(b)) - min(len(a), len(b)) + 1`.
    Valid,
}

/// 1-D linear convolution of two signals.
///
/// Both `a` and `b` must be 1-D tensors.
///
/// # Examples
///
/// ```
/// # use scivex_core::Tensor;
/// # use scivex_signal::convolution::{convolve, ConvolveMode};
/// let a = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0], vec![3]).unwrap();
/// let b = Tensor::from_vec(vec![1.0_f64, 1.0], vec![2]).unwrap();
/// let c = convolve(&a, &b, ConvolveMode::Full).unwrap();
/// assert_eq!(c.as_slice(), &[1.0, 3.0, 5.0, 3.0]);
/// ```
pub fn convolve<T: Float>(a: &Tensor<T>, b: &Tensor<T>, mode: ConvolveMode) -> Result<Tensor<T>> {
    if a.ndim() != 1 || b.ndim() != 1 {
        return Err(SignalError::InvalidParameter {
            name: "input",
            reason: "convolve expects 1-D tensors",
        });
    }
    let na = a.numel();
    let nb = b.numel();
    if na == 0 || nb == 0 {
        return Err(SignalError::EmptyInput);
    }

    // Choose FFT path when direct O(na*nb) exceeds FFT O(N log N).
    // Threshold: when the smaller signal exceeds 64 samples.
    let full_len = na + nb - 1;
    let full = if na.min(nb) > 64 {
        fftconvolve_inner(a.as_slice(), b.as_slice(), na, nb, full_len)?
    } else {
        direct_convolve(a.as_slice(), b.as_slice(), na, nb, full_len)
    };

    let out = match mode {
        ConvolveMode::Full => full,
        ConvolveMode::Same => {
            let start = (nb - 1) / 2;
            full[start..start + na].to_vec()
        }
        ConvolveMode::Valid => {
            let big = na.max(nb);
            let small = na.min(nb);
            let valid_len = big - small + 1;
            let start = small - 1;
            full[start..start + valid_len].to_vec()
        }
    };

    let len = out.len();
    Ok(Tensor::from_vec(out, vec![len])?)
}

/// 1-D cross-correlation: convolve `a` with time-reversed `b`.
///
/// # Examples
///
/// ```
/// # use scivex_core::Tensor;
/// # use scivex_signal::convolution::{correlate, ConvolveMode};
/// let a = Tensor::from_vec(vec![0.0_f64, 0.0, 1.0, 0.0, 0.0], vec![5]).unwrap();
/// let c = correlate(&a, &a, ConvolveMode::Full).unwrap();
/// // Auto-correlation peak is at the center.
/// let s = c.as_slice();
/// let max_idx = s.iter().enumerate()
///     .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
///     .unwrap().0;
/// assert_eq!(max_idx, 4);
/// ```
pub fn correlate<T: Float>(a: &Tensor<T>, b: &Tensor<T>, mode: ConvolveMode) -> Result<Tensor<T>> {
    if b.ndim() != 1 {
        return Err(SignalError::InvalidParameter {
            name: "b",
            reason: "correlate expects 1-D tensors",
        });
    }
    // Reverse b.
    let rev: Vec<T> = b.as_slice().iter().rev().copied().collect();
    let rev_len = rev.len();
    let b_rev = Tensor::from_vec(rev, vec![rev_len])?;
    convolve(a, &b_rev, mode)
}

/// Direct convolution with f64 FMA fast path.
fn direct_convolve<T: Float>(sa: &[T], sb: &[T], na: usize, nb: usize, full_len: usize) -> Vec<T> {
    let mut full = vec![T::zero(); full_len];
    if core::any::TypeId::of::<T>() == core::any::TypeId::of::<f64>() {
        // SAFETY: T is f64, pointer casts are valid.
        let sa_f64 = unsafe { &*(core::ptr::from_ref::<[T]>(sa) as *const [f64]) };
        let sb_f64 = unsafe { &*(core::ptr::from_ref::<[T]>(sb) as *const [f64]) };
        let full_f64 = unsafe { &mut *(core::ptr::from_mut::<[T]>(&mut full) as *mut [f64]) };
        for (i, &ai) in sa_f64.iter().enumerate().take(na) {
            let chunks4 = nb / 4;
            let rem = nb % 4;
            for jj in 0..chunks4 {
                let j = jj * 4;
                // SAFETY: i+j+3 < na+nb-1 = full_len, j+3 < nb
                unsafe {
                    let p0 = full_f64.get_unchecked_mut(i + j);
                    *p0 = ai.mul_add(*sb_f64.get_unchecked(j), *p0);
                    let p1 = full_f64.get_unchecked_mut(i + j + 1);
                    *p1 = ai.mul_add(*sb_f64.get_unchecked(j + 1), *p1);
                    let p2 = full_f64.get_unchecked_mut(i + j + 2);
                    *p2 = ai.mul_add(*sb_f64.get_unchecked(j + 2), *p2);
                    let p3 = full_f64.get_unchecked_mut(i + j + 3);
                    *p3 = ai.mul_add(*sb_f64.get_unchecked(j + 3), *p3);
                }
            }
            let tail = chunks4 * 4;
            for j in 0..rem {
                unsafe {
                    let p = full_f64.get_unchecked_mut(i + tail + j);
                    *p = ai.mul_add(*sb_f64.get_unchecked(tail + j), *p);
                }
            }
        }
    } else {
        for i in 0..na {
            for j in 0..nb {
                full[i + j] += sa[i] * sb[j];
            }
        }
    }
    full
}

/// FFT-based convolution for large inputs: O(N log N) vs O(na * nb).
///
/// Zero-pads both signals to `full_len`, takes rfft, multiplies spectra,
/// then takes irfft to get the result.
fn fftconvolve_inner<T: Float>(
    sa: &[T],
    sb: &[T],
    na: usize,
    nb: usize,
    full_len: usize,
) -> Result<Vec<T>> {
    // Pad both signals to full_len
    let mut a_pad = vec![T::zero(); full_len];
    a_pad[..na].copy_from_slice(sa);
    let mut b_pad = vec![T::zero(); full_len];
    b_pad[..nb].copy_from_slice(sb);

    let a_t = Tensor::from_vec(a_pad, vec![full_len])?;
    let b_t = Tensor::from_vec(b_pad, vec![full_len])?;

    // Forward FFT (real-to-complex)
    let fa = fft::rfft(&a_t)?;
    let fb = fft::rfft(&b_t)?;

    // Complex multiplication: (ar + ai*j) * (br + bi*j) = (ar*br - ai*bi) + (ar*bi + ai*br)*j
    let fa_s = fa.as_slice();
    let fb_s = fb.as_slice();
    let spec_len = fa.shape()[0]; // N/2 + 1
    let mut prod = vec![T::zero(); spec_len * 2];
    for k in 0..spec_len {
        let ar = fa_s[k * 2];
        let ai = fa_s[k * 2 + 1];
        let br = fb_s[k * 2];
        let bi = fb_s[k * 2 + 1];
        prod[k * 2] = ar * br - ai * bi;
        prod[k * 2 + 1] = ar * bi + ai * br;
    }

    let prod_t = Tensor::from_vec(prod, vec![spec_len, 2])?;
    let result = fft::irfft(&prod_t, full_len)?;
    Ok(result.as_slice().to_vec())
}

/// Public FFT-based convolution.
///
/// Always uses the FFT path regardless of input size. Useful when you know
/// both signals are large.
///
/// # Examples
///
/// ```
/// # use scivex_core::Tensor;
/// # use scivex_signal::convolution::{fftconvolve, ConvolveMode};
/// let a = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0], vec![3]).unwrap();
/// let b = Tensor::from_vec(vec![1.0_f64, 1.0], vec![2]).unwrap();
/// let c = fftconvolve(&a, &b, ConvolveMode::Full).unwrap();
/// let s = c.as_slice();
/// assert!((s[0] - 1.0).abs() < 1e-10);
/// assert!((s[1] - 3.0).abs() < 1e-10);
/// assert!((s[2] - 5.0).abs() < 1e-10);
/// assert!((s[3] - 3.0).abs() < 1e-10);
/// ```
pub fn fftconvolve<T: Float>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    mode: ConvolveMode,
) -> Result<Tensor<T>> {
    if a.ndim() != 1 || b.ndim() != 1 {
        return Err(SignalError::InvalidParameter {
            name: "input",
            reason: "fftconvolve expects 1-D tensors",
        });
    }
    let na = a.numel();
    let nb = b.numel();
    if na == 0 || nb == 0 {
        return Err(SignalError::EmptyInput);
    }

    let full_len = na + nb - 1;
    let full = fftconvolve_inner(a.as_slice(), b.as_slice(), na, nb, full_len)?;

    let out = match mode {
        ConvolveMode::Full => full,
        ConvolveMode::Same => {
            let start = (nb - 1) / 2;
            full[start..start + na].to_vec()
        }
        ConvolveMode::Valid => {
            let big = na.max(nb);
            let small = na.min(nb);
            let valid_len = big - small + 1;
            let start = small - 1;
            full[start..start + valid_len].to_vec()
        }
    };

    let len = out.len();
    Ok(Tensor::from_vec(out, vec![len])?)
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn test_convolve_full() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let b = Tensor::from_vec(vec![1.0, 1.0], vec![2]).unwrap();
        let c = convolve(&a, &b, ConvolveMode::Full).unwrap();
        assert_eq!(c.shape(), &[4]);
        assert_eq!(c.as_slice(), &[1.0, 3.0, 5.0, 3.0]);
    }

    #[test]
    fn test_convolve_same() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let b = Tensor::from_vec(vec![1.0, 1.0], vec![2]).unwrap();
        let c = convolve(&a, &b, ConvolveMode::Same).unwrap();
        assert_eq!(c.shape(), &[3]);
        assert_eq!(c.as_slice(), &[1.0, 3.0, 5.0]);
    }

    #[test]
    fn test_convolve_valid() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let b = Tensor::from_vec(vec![1.0, 1.0], vec![2]).unwrap();
        let c = convolve(&a, &b, ConvolveMode::Valid).unwrap();
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.as_slice(), &[3.0, 5.0]);
    }

    #[test]
    fn test_correlate_self_peak() {
        let a = Tensor::from_vec(vec![0.0, 0.0, 1.0, 0.0, 0.0], vec![5]).unwrap();
        let c = correlate(&a, &a, ConvolveMode::Full).unwrap();
        // Auto-correlation of impulse peaks at center.
        let s = c.as_slice();
        let max_idx = s
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(max_idx, 4); // center of length-9 output
    }

    #[test]
    fn test_convolve_empty_error() {
        let a = Tensor::from_vec(vec![1.0], vec![1]).unwrap();
        let b = Tensor::<f64>::from_vec(vec![], vec![0]).unwrap();
        assert!(convolve(&a, &b, ConvolveMode::Full).is_err());
    }

    #[test]
    fn test_fftconvolve_matches_direct() {
        let a = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0], vec![3]).unwrap();
        let b = Tensor::from_vec(vec![1.0, 1.0], vec![2]).unwrap();
        let c = fftconvolve(&a, &b, ConvolveMode::Full).unwrap();
        let s = c.as_slice();
        assert!((s[0] - 1.0).abs() < 1e-10);
        assert!((s[1] - 3.0).abs() < 1e-10);
        assert!((s[2] - 5.0).abs() < 1e-10);
        assert!((s[3] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_fftconvolve_same_mode() {
        let a = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
        let b = Tensor::from_vec(vec![1.0, 0.0, -1.0], vec![3]).unwrap();
        let direct = convolve(&a, &b, ConvolveMode::Same).unwrap();
        let fft_result = fftconvolve(&a, &b, ConvolveMode::Same).unwrap();
        for (d, f) in direct.as_slice().iter().zip(fft_result.as_slice()) {
            assert!((d - f).abs() < 1e-10, "direct={d}, fft={f}");
        }
    }

    #[test]
    fn test_fftconvolve_large_auto_dispatch() {
        // Large enough to trigger the FFT path in `convolve` (min > 64)
        let n = 128;
        let a_data: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
        let b_data: Vec<f64> = (0..n).map(|i| (i as f64 * 0.5).cos()).collect();
        let a = Tensor::from_vec(a_data.clone(), vec![n]).unwrap();
        let b = Tensor::from_vec(b_data.clone(), vec![n]).unwrap();

        let result = convolve(&a, &b, ConvolveMode::Full).unwrap();

        // Verify against explicit direct computation for a few points
        let direct = direct_convolve(&a_data, &b_data, n, n, 2 * n - 1);
        for (i, (&r, &d)) in result.as_slice().iter().zip(direct.iter()).enumerate() {
            assert!((r - d).abs() < 1e-6, "mismatch at {i}: {r} vs {d}");
        }
    }
}
