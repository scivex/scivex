//! Resampling: FFT-based resample, decimate, linear interpolation.

use scivex_core::{Float, Tensor, fft};

use crate::error::{Result, SignalError};
use crate::filter::{FirFilter, lfilter};

/// Resample a signal to `num_samples` using FFT-based method.
///
/// Zero-pads or truncates in the frequency domain, then applies inverse FFT.
pub fn resample<T: Float>(x: &Tensor<T>, num_samples: usize) -> Result<Tensor<T>> {
    if x.ndim() != 1 {
        return Err(SignalError::InvalidParameter {
            name: "x",
            reason: "resample expects a 1-D signal",
        });
    }
    if num_samples == 0 {
        return Err(SignalError::InvalidParameter {
            name: "num_samples",
            reason: "must be > 0",
        });
    }
    let n = x.numel();
    if n == 0 {
        return Err(SignalError::EmptyInput);
    }
    if n == num_samples {
        return Ok(x.clone());
    }

    // Compute full complex FFT.
    let xs = x.as_slice();
    let padded_in = n.next_power_of_two();
    let mut cdata = vec![T::zero(); padded_in * 2];
    for (i, &v) in xs.iter().enumerate() {
        cdata[i * 2] = v;
    }
    let cinput = Tensor::from_vec(cdata, vec![padded_in, 2])?;
    let spectrum = fft::fft(&cinput)?;
    let sd = spectrum.as_slice();
    let spec_len = spectrum.shape()[0];

    // Build new spectrum of target length (next power of two).
    let padded_out = num_samples.next_power_of_two();
    let mut new_spec = vec![T::zero(); padded_out * 2];

    // Copy positive frequencies.
    let half_min = (spec_len / 2).min(padded_out / 2);
    for i in 0..half_min {
        new_spec[i * 2] = sd[i * 2];
        new_spec[i * 2 + 1] = sd[i * 2 + 1];
    }
    // Copy negative frequencies from the end.
    for i in 1..half_min {
        let src = spec_len - i;
        let dst = padded_out - i;
        new_spec[dst * 2] = sd[src * 2];
        new_spec[dst * 2 + 1] = sd[src * 2 + 1];
    }

    let new_spec_tensor = Tensor::from_vec(new_spec, vec![padded_out, 2])?;
    let result = fft::ifft(&new_spec_tensor)?;
    let rs = result.as_slice();

    // Scale and take real parts.
    let scale = T::from_usize(padded_out) / T::from_usize(padded_in);
    let out: Vec<T> = (0..num_samples).map(|i| rs[i * 2] * scale).collect();

    Ok(Tensor::from_vec(out, vec![num_samples])?)
}

/// Downsample by integer factor with low-pass anti-aliasing filter.
///
/// `factor` must be >= 1.
pub fn decimate<T: Float>(x: &Tensor<T>, factor: usize) -> Result<Tensor<T>> {
    if x.ndim() != 1 {
        return Err(SignalError::InvalidParameter {
            name: "x",
            reason: "decimate expects a 1-D signal",
        });
    }
    if factor == 0 {
        return Err(SignalError::InvalidParameter {
            name: "factor",
            reason: "must be > 0",
        });
    }
    if factor == 1 {
        return Ok(x.clone());
    }
    let n = x.numel();
    if n == 0 {
        return Err(SignalError::EmptyInput);
    }

    // Design anti-aliasing LP filter at cutoff = 1/factor (Nyquist-normalized).
    let cutoff = T::one() / T::from_usize(factor);
    // Clamp to valid range.
    let cutoff = if cutoff >= T::one() {
        T::from_f64(0.99)
    } else {
        cutoff
    };
    let num_taps = 31.min(n).max(3);
    // Ensure odd.
    let num_taps = if num_taps % 2 == 0 {
        num_taps - 1
    } else {
        num_taps
    };

    let h = FirFilter::low_pass(cutoff, num_taps)?;
    let a = Tensor::from_vec(vec![T::one()], vec![1])?;
    let filtered = lfilter(&h, &a, x)?;

    // Downsample.
    let fs = filtered.as_slice();
    let out: Vec<T> = fs.iter().step_by(factor).copied().collect();
    let out_len = out.len();

    Ok(Tensor::from_vec(out, vec![out_len])?)
}

/// Upsample by integer factor with linear interpolation.
///
/// `factor` must be >= 1.
pub fn interpolate_linear<T: Float>(x: &Tensor<T>, factor: usize) -> Result<Tensor<T>> {
    if x.ndim() != 1 {
        return Err(SignalError::InvalidParameter {
            name: "x",
            reason: "interpolate_linear expects a 1-D signal",
        });
    }
    if factor == 0 {
        return Err(SignalError::InvalidParameter {
            name: "factor",
            reason: "must be > 0",
        });
    }
    if factor == 1 {
        return Ok(x.clone());
    }
    let n = x.numel();
    if n == 0 {
        return Err(SignalError::EmptyInput);
    }
    if n == 1 {
        return Ok(Tensor::full(vec![factor], x.as_slice()[0]));
    }

    let xs = x.as_slice();
    let out_len = (n - 1) * factor + 1;
    let mut out = vec![T::zero(); out_len];

    for i in 0..n - 1 {
        let a = xs[i];
        let b = xs[i + 1];
        for j in 0..factor {
            let t = T::from_usize(j) / T::from_usize(factor);
            out[i * factor + j] = a + (b - a) * t;
        }
    }
    out[out_len - 1] = xs[n - 1];

    Ok(Tensor::from_vec(out, vec![out_len])?)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 0.15;

    #[test]
    fn test_resample_identity() {
        let data = vec![1.0_f64, 2.0, 3.0, 4.0];
        let x = Tensor::from_vec(data.clone(), vec![4]).unwrap();
        let y = resample(&x, 4).unwrap();
        for (a, b) in y.as_slice().iter().zip(data.iter()) {
            assert!((a - b).abs() < TOL, "got {a}, expected {b}");
        }
    }

    #[test]
    fn test_decimate_factor_1() {
        let data = vec![1.0_f64, 2.0, 3.0, 4.0];
        let x = Tensor::from_vec(data.clone(), vec![4]).unwrap();
        let y = decimate(&x, 1).unwrap();
        assert_eq!(y.as_slice(), data.as_slice());
    }

    #[test]
    fn test_decimate_reduces_length() {
        let data: Vec<f64> = (0..100).map(|i| (f64::from(i) * 0.1).sin()).collect();
        let x = Tensor::from_vec(data, vec![100]).unwrap();
        let y = decimate(&x, 4).unwrap();
        assert_eq!(y.numel(), 25);
    }

    #[test]
    fn test_interpolate_linear_basic() {
        let data = vec![0.0_f64, 1.0, 2.0];
        let x = Tensor::from_vec(data, vec![3]).unwrap();
        let y = interpolate_linear(&x, 2).unwrap();
        assert_eq!(y.numel(), 5);
        let ys = y.as_slice();
        assert!((ys[0] - 0.0).abs() < 1e-10);
        assert!((ys[1] - 0.5).abs() < 1e-10);
        assert!((ys[2] - 1.0).abs() < 1e-10);
        assert!((ys[3] - 1.5).abs() < 1e-10);
        assert!((ys[4] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_interpolate_factor_1() {
        let data = vec![1.0_f64, 2.0, 3.0];
        let x = Tensor::from_vec(data.clone(), vec![3]).unwrap();
        let y = interpolate_linear(&x, 1).unwrap();
        assert_eq!(y.as_slice(), data.as_slice());
    }

    #[test]
    fn test_resample_zero_error() {
        let x = Tensor::from_vec(vec![1.0_f64], vec![1]).unwrap();
        assert!(resample(&x, 0).is_err());
    }

    #[test]
    fn test_decimate_zero_factor() {
        let x = Tensor::from_vec(vec![1.0_f64], vec![1]).unwrap();
        assert!(decimate(&x, 0).is_err());
    }
}
