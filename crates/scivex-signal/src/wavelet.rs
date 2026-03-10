//! Basic wavelet transforms: DWT and IDWT with Haar wavelet.

use scivex_core::{Float, Tensor};

use crate::error::{Result, SignalError};

/// Supported wavelet families.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Wavelet {
    /// Haar wavelet (Daubechies-1).
    /// LP coefficients: `[1/sqrt(2), 1/sqrt(2)]`
    /// HP coefficients: `[1/sqrt(2), -1/sqrt(2)]`
    Haar,
}

impl Wavelet {
    /// Low-pass (scaling) filter coefficients.
    fn low_pass<T: Float>(self) -> Vec<T> {
        match self {
            Self::Haar => {
                let s = T::one() / T::from_f64(2.0).sqrt();
                vec![s, s]
            }
        }
    }

    /// High-pass (wavelet) filter coefficients.
    fn high_pass<T: Float>(self) -> Vec<T> {
        match self {
            Self::Haar => {
                let s = T::one() / T::from_f64(2.0).sqrt();
                vec![s, -s]
            }
        }
    }
}

/// Single-level Discrete Wavelet Transform.
///
/// Returns `(approximation, detail)` coefficient tensors, each of length `ceil(n/2)`.
pub fn dwt<T: Float>(x: &Tensor<T>, wavelet: Wavelet) -> Result<(Tensor<T>, Tensor<T>)> {
    if x.ndim() != 1 {
        return Err(SignalError::InvalidParameter {
            name: "x",
            reason: "dwt expects a 1-D tensor",
        });
    }
    let n = x.numel();
    if n < 2 {
        return Err(SignalError::InvalidParameter {
            name: "x",
            reason: "signal must have at least 2 samples",
        });
    }

    let xs = x.as_slice();
    let lp = wavelet.low_pass::<T>();
    let hp = wavelet.high_pass::<T>();
    let filt_len = lp.len();

    // Output length: ceil(n / 2) for the downsampled convolution.
    let out_len = (n + filt_len - 1) / 2;
    let mut approx = vec![T::zero(); out_len];
    let mut detail = vec![T::zero(); out_len];

    for i in 0..out_len {
        let mut a = T::zero();
        let mut d = T::zero();
        for j in 0..filt_len {
            let idx = i * 2 + j;
            let val = if idx < n { xs[idx] } else { T::zero() };
            a += val * lp[j];
            d += val * hp[j];
        }
        approx[i] = a;
        detail[i] = d;
    }

    Ok((
        Tensor::from_vec(approx, vec![out_len])?,
        Tensor::from_vec(detail, vec![out_len])?,
    ))
}

/// Single-level Inverse Discrete Wavelet Transform.
///
/// Reconstructs the signal from approximation and detail coefficients.
pub fn idwt<T: Float>(
    approx: &Tensor<T>,
    detail: &Tensor<T>,
    wavelet: Wavelet,
) -> Result<Tensor<T>> {
    if approx.ndim() != 1 || detail.ndim() != 1 {
        return Err(SignalError::InvalidParameter {
            name: "input",
            reason: "idwt expects 1-D tensors",
        });
    }
    if approx.numel() != detail.numel() {
        return Err(SignalError::DimensionMismatch {
            expected: approx.numel(),
            got: detail.numel(),
        });
    }

    let coeff_len = approx.numel();
    let as_ = approx.as_slice();
    let ds = detail.as_slice();

    let lp = wavelet.low_pass::<T>();
    let hp = wavelet.high_pass::<T>();

    // Reconstruct: upsample by 2 and convolve with reconstruction filters.
    // For Haar, reconstruction filters are the same as analysis filters.
    let out_len = coeff_len * 2;
    let mut output = vec![T::zero(); out_len];

    for i in 0..coeff_len {
        for j in 0..lp.len() {
            let idx = i * 2 + j;
            if idx < out_len {
                output[idx] += as_[i] * lp[j] + ds[i] * hp[j];
            }
        }
    }

    Ok(Tensor::from_vec(output, vec![out_len])?)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;

    #[test]
    fn test_dwt_idwt_roundtrip() {
        let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let n = data.len();
        let x = Tensor::from_vec(data.clone(), vec![n]).unwrap();
        let (approx, detail) = dwt(&x, Wavelet::Haar).unwrap();
        let y = idwt(&approx, &detail, Wavelet::Haar).unwrap();
        assert_eq!(y.numel(), n);
        for (a, b) in y.as_slice().iter().zip(data.iter()) {
            assert!((a - b).abs() < TOL, "got {a}, expected {b}");
        }
    }

    #[test]
    fn test_haar_constant_signal() {
        // Constant signal: all detail coefficients should be zero.
        let data = vec![3.0_f64; 8];
        let x = Tensor::from_vec(data, vec![8]).unwrap();
        let (_, detail) = dwt(&x, Wavelet::Haar).unwrap();
        for &d in detail.as_slice() {
            assert!(d.abs() < TOL, "detail = {d}, expected ~0");
        }
    }

    #[test]
    fn test_haar_approximation_value() {
        // Haar approx of [a, b] = (a+b)/sqrt(2)
        let data = vec![1.0_f64, 3.0];
        let x = Tensor::from_vec(data, vec![2]).unwrap();
        let (approx, detail) = dwt(&x, Wavelet::Haar).unwrap();
        let expected_approx = (1.0 + 3.0) / 2.0_f64.sqrt();
        let expected_detail = (1.0 - 3.0) / 2.0_f64.sqrt();
        assert!((approx.as_slice()[0] - expected_approx).abs() < TOL);
        assert!((detail.as_slice()[0] - expected_detail).abs() < TOL);
    }

    #[test]
    fn test_dwt_short_signal() {
        let x = Tensor::from_vec(vec![1.0_f64], vec![1]).unwrap();
        assert!(dwt(&x, Wavelet::Haar).is_err());
    }

    #[test]
    fn test_idwt_mismatch() {
        let a = Tensor::from_vec(vec![1.0_f64, 2.0], vec![2]).unwrap();
        let d = Tensor::from_vec(vec![1.0_f64], vec![1]).unwrap();
        assert!(idwt(&a, &d, Wavelet::Haar).is_err());
    }
}
