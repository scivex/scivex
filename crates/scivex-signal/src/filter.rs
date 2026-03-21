//! Digital filter design and application.
//!
//! - `lfilter` — apply an IIR/FIR filter (direct form II transposed).
//! - `filtfilt` — zero-phase filtering (forward + backward pass).
//! - `FirFilter` — FIR filter design (low-pass, high-pass, band-pass).

use scivex_core::{Float, Tensor};

use crate::error::{Result, SignalError};
use crate::window;

/// Apply a digital IIR/FIR filter using Direct Form II Transposed.
///
/// `b` — numerator (feed-forward) coefficients, shape `[M]`.
/// `a` — denominator (feedback) coefficients, shape `[N]`. `a[0]` must be non-zero;
///        all coefficients are normalized by `a[0]`.
/// `x` — input signal, shape `[L]`.
///
/// Returns the filtered signal of the same length as `x`.
///
/// # Examples
///
/// ```
/// # use scivex_core::Tensor;
/// # use scivex_signal::filter::lfilter;
/// // Simple moving average filter (FIR, 3-tap)
/// let b = Tensor::from_vec(vec![1.0/3.0, 1.0/3.0, 1.0/3.0], vec![3]).unwrap();
/// let a = Tensor::from_vec(vec![1.0_f64], vec![1]).unwrap();
/// let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
/// let y = lfilter(&b, &a, &x).unwrap();
/// assert_eq!(y.shape(), &[5]);
/// ```
pub fn lfilter<T: Float>(b: &Tensor<T>, a: &Tensor<T>, x: &Tensor<T>) -> Result<Tensor<T>> {
    if b.ndim() != 1 || a.ndim() != 1 || x.ndim() != 1 {
        return Err(SignalError::InvalidParameter {
            name: "input",
            reason: "lfilter expects 1-D tensors",
        });
    }
    if b.is_empty() || a.is_empty() {
        return Err(SignalError::EmptyInput);
    }
    if x.is_empty() {
        let len = x.numel();
        return Ok(Tensor::from_vec(vec![], vec![len])?);
    }

    let bs = b.as_slice();
    let a_raw = a.as_slice();

    if a_raw[0] == T::zero() {
        return Err(SignalError::InvalidParameter {
            name: "a",
            reason: "a[0] must be non-zero",
        });
    }

    // Normalize by a[0].
    let a0 = a_raw[0];
    let bn: Vec<T> = bs.iter().map(|&v| v / a0).collect();
    let an: Vec<T> = a_raw.iter().map(|&v| v / a0).collect();

    let xs = x.as_slice();
    let n = xs.len();
    let nb = bn.len();
    let na = an.len();
    let order = nb.max(na);

    // State vector for direct form II transposed.
    let mut z = vec![T::zero(); order];
    let mut y = vec![T::zero(); n];

    for i in 0..n {
        y[i] = bn[0] * xs[i] + z[0];
        for j in 1..order {
            let b_val = if j < nb { bn[j] } else { T::zero() };
            let a_val = if j < na { an[j] } else { T::zero() };
            z[j - 1] = b_val * xs[i] - a_val * y[i] + if j + 1 < order { z[j] } else { T::zero() };
        }
    }

    Ok(Tensor::from_vec(y, vec![n])?)
}

/// Zero-phase digital filtering (forward-backward).
///
/// Applies `lfilter` forwards, then reverses the result and applies `lfilter`
/// again, giving zero phase distortion.
///
/// # Examples
///
/// ```
/// # use scivex_core::Tensor;
/// # use scivex_signal::filter::filtfilt;
/// let b = Tensor::from_vec(vec![1.0_f64], vec![1]).unwrap();
/// let a = Tensor::from_vec(vec![1.0], vec![1]).unwrap();
/// let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
/// let y = filtfilt(&b, &a, &x).unwrap();
/// assert_eq!(y.shape(), &[4]);
/// ```
pub fn filtfilt<T: Float>(b: &Tensor<T>, a: &Tensor<T>, x: &Tensor<T>) -> Result<Tensor<T>> {
    // Forward pass.
    let y1 = lfilter(b, a, x)?;

    // Reverse.
    let rev: Vec<T> = y1.as_slice().iter().rev().copied().collect();
    let rev_len = rev.len();
    let y1_rev = Tensor::from_vec(rev, vec![rev_len])?;

    // Backward pass.
    let y2 = lfilter(b, a, &y1_rev)?;

    // Reverse again.
    let result: Vec<T> = y2.as_slice().iter().rev().copied().collect();
    let result_len = result.len();
    Ok(Tensor::from_vec(result, vec![result_len])?)
}

/// FIR filter design utilities.
///
/// # Examples
///
/// ```
/// # use scivex_signal::filter::FirFilter;
/// let h = FirFilter::low_pass::<f64>(0.5, 31).unwrap();
/// assert_eq!(h.shape(), &[31]);
/// ```
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct FirFilter;

impl FirFilter {
    /// Design a low-pass FIR filter using the windowed-sinc method.
    ///
    /// `cutoff` — normalized cutoff frequency in `(0, 1)` where 1 = Nyquist.
    /// `num_taps` — number of filter taps (must be odd for type I FIR).
    ///
    /// Returns the filter coefficients as a 1-D tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_signal::filter::FirFilter;
    /// let h = FirFilter::low_pass::<f64>(0.5, 31).unwrap();
    /// let dc_gain: f64 = h.as_slice().iter().sum();
    /// assert!((dc_gain - 1.0).abs() < 0.01); // DC gain ≈ 1
    /// ```
    pub fn low_pass<T: Float>(cutoff: T, num_taps: usize) -> Result<Tensor<T>> {
        if num_taps == 0 {
            return Err(SignalError::InvalidParameter {
                name: "num_taps",
                reason: "must be > 0",
            });
        }
        if cutoff <= T::zero() || cutoff >= T::one() {
            return Err(SignalError::InvalidParameter {
                name: "cutoff",
                reason: "must be in (0, 1) where 1 = Nyquist",
            });
        }

        let wc = cutoff * T::pi(); // cutoff in radians (Nyquist-normalized)
        let mid = T::from_usize(num_taps - 1) / T::from_f64(2.0);

        // Ideal sinc filter.
        let mut h: Vec<T> = (0..num_taps)
            .map(|k| {
                let n = T::from_usize(k) - mid;
                if n.abs() < T::from_f64(1e-12) {
                    wc / T::pi()
                } else {
                    (wc * n).sin() / (T::pi() * n)
                }
            })
            .collect();

        // Apply Hamming window.
        let win = window::hamming::<T>(num_taps)?;
        let ws = win.as_slice();
        for (h_val, &w_val) in h.iter_mut().zip(ws.iter()) {
            *h_val *= w_val;
        }

        // Normalize so DC gain = 1.
        let sum: T = h.iter().copied().fold(T::zero(), |acc, v| acc + v);
        if sum.abs() > T::from_f64(1e-15) {
            for h_val in &mut h {
                *h_val /= sum;
            }
        }

        Ok(Tensor::from_vec(h, vec![num_taps])?)
    }

    /// Design a high-pass FIR filter via spectral inversion of a low-pass filter.
    ///
    /// `cutoff` — normalized cutoff frequency in `(0, 1)`.
    /// `num_taps` — must be odd.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_signal::filter::FirFilter;
    /// let h = FirFilter::high_pass::<f64>(0.5, 31).unwrap();
    /// let dc_gain: f64 = h.as_slice().iter().sum();
    /// assert!(dc_gain.abs() < 0.01); // rejects DC
    /// ```
    pub fn high_pass<T: Float>(cutoff: T, num_taps: usize) -> Result<Tensor<T>> {
        if num_taps % 2 == 0 {
            return Err(SignalError::InvalidParameter {
                name: "num_taps",
                reason: "high-pass FIR requires odd num_taps",
            });
        }
        let lp = Self::low_pass(cutoff, num_taps)?;
        let mid = num_taps / 2;
        let mut h: Vec<T> = lp.as_slice().iter().map(|&v| -v).collect();
        h[mid] += T::one();
        Ok(Tensor::from_vec(h, vec![num_taps])?)
    }

    /// Design a band-pass FIR filter: `LP(high) - LP(low)`.
    ///
    /// `low`, `high` — normalized cutoff frequencies in `(0, 1)`, `low < high`.
    /// `num_taps` — must be odd.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_signal::filter::FirFilter;
    /// let h = FirFilter::band_pass::<f64>(0.2, 0.4, 31).unwrap();
    /// let dc_gain: f64 = h.as_slice().iter().sum();
    /// assert!(dc_gain.abs() < 0.1); // band-pass rejects DC
    /// ```
    pub fn band_pass<T: Float>(low: T, high: T, num_taps: usize) -> Result<Tensor<T>> {
        if low >= high {
            return Err(SignalError::InvalidParameter {
                name: "low/high",
                reason: "low must be less than high",
            });
        }
        if num_taps % 2 == 0 {
            return Err(SignalError::InvalidParameter {
                name: "num_taps",
                reason: "band-pass FIR requires odd num_taps",
            });
        }
        let lp_high = Self::low_pass(high, num_taps)?;
        let lp_low = Self::low_pass(low, num_taps)?;
        lp_high.zip_map(&lp_low, |a, b| a - b).map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;

    #[test]
    fn test_lfilter_identity() {
        // b=[1], a=[1] => identity filter.
        let b = Tensor::from_vec(vec![1.0], vec![1]).unwrap();
        let a = Tensor::from_vec(vec![1.0], vec![1]).unwrap();
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let y = lfilter(&b, &a, &x).unwrap();
        assert_eq!(y.as_slice(), x.as_slice());
    }

    #[test]
    fn test_lfilter_fir_moving_average() {
        // b=[0.5, 0.5], a=[1] => 2-point moving average.
        let b = Tensor::from_vec(vec![0.5, 0.5], vec![2]).unwrap();
        let a = Tensor::from_vec(vec![1.0], vec![1]).unwrap();
        let x = Tensor::from_vec(vec![1.0, 3.0, 5.0, 7.0], vec![4]).unwrap();
        let y = lfilter(&b, &a, &x).unwrap();
        let ys = y.as_slice();
        assert!((ys[0] - 0.5).abs() < TOL); // 0.5*1 + 0.5*0
        assert!((ys[1] - 2.0).abs() < TOL); // 0.5*3 + 0.5*1
        assert!((ys[2] - 4.0).abs() < TOL); // 0.5*5 + 0.5*3
        assert!((ys[3] - 6.0).abs() < TOL); // 0.5*7 + 0.5*5
    }

    #[test]
    fn test_filtfilt_allpass() {
        // All-pass filter: b=[1], a=[1].
        let b = Tensor::from_vec(vec![1.0], vec![1]).unwrap();
        let a = Tensor::from_vec(vec![1.0], vec![1]).unwrap();
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let y = filtfilt(&b, &a, &x).unwrap();
        for (a_val, b_val) in y.as_slice().iter().zip(x.as_slice().iter()) {
            assert!((a_val - b_val).abs() < TOL, "got {a_val}, expected {b_val}");
        }
    }

    #[test]
    fn test_fir_lowpass_passes_dc() {
        let h = FirFilter::low_pass::<f64>(0.5, 31).unwrap();
        let sum: f64 = h.as_slice().iter().sum();
        // DC gain should be approximately 1.
        assert!((sum - 1.0).abs() < 0.01, "DC gain = {sum}");
    }

    #[test]
    fn test_fir_highpass() {
        let h = FirFilter::high_pass::<f64>(0.5, 31).unwrap();
        // Sum of high-pass coefficients should be ~0 (rejects DC).
        let sum: f64 = h.as_slice().iter().sum();
        assert!(sum.abs() < 0.01, "HP DC gain = {sum}");
    }

    #[test]
    fn test_fir_bandpass() {
        let h = FirFilter::band_pass::<f64>(0.2, 0.4, 31).unwrap();
        // Band-pass rejects DC.
        let sum: f64 = h.as_slice().iter().sum();
        assert!(sum.abs() < 0.1, "BP DC gain = {sum}");
    }

    #[test]
    fn test_lfilter_a0_zero() {
        let b = Tensor::from_vec(vec![1.0], vec![1]).unwrap();
        let a = Tensor::from_vec(vec![0.0], vec![1]).unwrap();
        let x = Tensor::from_vec(vec![1.0], vec![1]).unwrap();
        assert!(lfilter(&b, &a, &x).is_err());
    }
}
