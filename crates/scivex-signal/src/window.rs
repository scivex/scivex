//! Window functions for spectral analysis and filter design.

use scivex_core::{Float, Tensor};

use crate::error::{Result, SignalError};

/// Hann (raised-cosine) window: `0.5 * (1 - cos(2*pi*k / (N-1)))`.
///
/// # Examples
///
/// ```
/// # use scivex_signal::window;
/// let w = window::hann::<f64>(5).unwrap();
/// assert_eq!(w.shape(), &[5]);
/// assert!((w.as_slice()[0]).abs() < 1e-10); // first sample ≈ 0
/// ```
pub fn hann<T: Float>(n: usize) -> Result<Tensor<T>> {
    if n == 0 {
        return Err(SignalError::InvalidParameter {
            name: "n",
            reason: "window length must be > 0",
        });
    }
    if n == 1 {
        return Ok(Tensor::from_vec(vec![T::one()], vec![1])?);
    }
    let nm1 = T::from_usize(n - 1);
    let two_pi = T::from_f64(2.0) * T::pi();
    let half = T::from_f64(0.5);
    let data: Vec<T> = (0..n)
        .map(|k| half * (T::one() - (two_pi * T::from_usize(k) / nm1).cos()))
        .collect();
    Ok(Tensor::from_vec(data, vec![n])?)
}

/// Hamming window: `0.54 - 0.46 * cos(2*pi*k / (N-1))`.
///
/// # Examples
///
/// ```
/// # use scivex_signal::window;
/// let w = window::hamming::<f64>(5).unwrap();
/// assert_eq!(w.shape(), &[5]);
/// ```
pub fn hamming<T: Float>(n: usize) -> Result<Tensor<T>> {
    if n == 0 {
        return Err(SignalError::InvalidParameter {
            name: "n",
            reason: "window length must be > 0",
        });
    }
    if n == 1 {
        return Ok(Tensor::from_vec(vec![T::one()], vec![1])?);
    }
    let nm1 = T::from_usize(n - 1);
    let two_pi = T::from_f64(2.0) * T::pi();
    let a0 = T::from_f64(0.54);
    let a1 = T::from_f64(0.46);
    let data: Vec<T> = (0..n)
        .map(|k| a0 - a1 * (two_pi * T::from_usize(k) / nm1).cos())
        .collect();
    Ok(Tensor::from_vec(data, vec![n])?)
}

/// Blackman window: `0.42 - 0.5*cos(2*pi*k/(N-1)) + 0.08*cos(4*pi*k/(N-1))`.
///
/// # Examples
///
/// ```
/// # use scivex_signal::window;
/// let w = window::blackman::<f64>(5).unwrap();
/// assert_eq!(w.shape(), &[5]);
/// ```
pub fn blackman<T: Float>(n: usize) -> Result<Tensor<T>> {
    if n == 0 {
        return Err(SignalError::InvalidParameter {
            name: "n",
            reason: "window length must be > 0",
        });
    }
    if n == 1 {
        return Ok(Tensor::from_vec(vec![T::one()], vec![1])?);
    }
    let nm1 = T::from_usize(n - 1);
    let two_pi = T::from_f64(2.0) * T::pi();
    let four_pi = T::from_f64(4.0) * T::pi();
    let a0 = T::from_f64(0.42);
    let a1 = T::from_f64(0.5);
    let a2 = T::from_f64(0.08);
    let data: Vec<T> = (0..n)
        .map(|k| {
            let kf = T::from_usize(k);
            a0 - a1 * (two_pi * kf / nm1).cos() + a2 * (four_pi * kf / nm1).cos()
        })
        .collect();
    Ok(Tensor::from_vec(data, vec![n])?)
}

/// Bartlett (triangular) window.
///
/// # Examples
///
/// ```
/// # use scivex_signal::window;
/// let w = window::bartlett::<f64>(5).unwrap();
/// assert_eq!(w.shape(), &[5]);
/// assert!((w.as_slice()[2] - 1.0).abs() < 1e-10); // peak at center
/// ```
pub fn bartlett<T: Float>(n: usize) -> Result<Tensor<T>> {
    if n == 0 {
        return Err(SignalError::InvalidParameter {
            name: "n",
            reason: "window length must be > 0",
        });
    }
    if n == 1 {
        return Ok(Tensor::from_vec(vec![T::one()], vec![1])?);
    }
    let nm1 = T::from_usize(n - 1);
    let two = T::from_f64(2.0);
    let half_nm1 = nm1 / two;
    let data: Vec<T> = (0..n)
        .map(|k| T::one() - ((T::from_usize(k) - half_nm1) / half_nm1).abs())
        .collect();
    Ok(Tensor::from_vec(data, vec![n])?)
}

/// Rectangular (boxcar) window — all ones.
///
/// # Examples
///
/// ```
/// # use scivex_signal::window;
/// let w = window::rectangular::<f64>(4).unwrap();
/// assert!(w.as_slice().iter().all(|&x| (x - 1.0).abs() < 1e-10));
/// ```
pub fn rectangular<T: Float>(n: usize) -> Result<Tensor<T>> {
    if n == 0 {
        return Err(SignalError::InvalidParameter {
            name: "n",
            reason: "window length must be > 0",
        });
    }
    Ok(Tensor::ones(vec![n]))
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;

    #[test]
    fn test_hann_1() {
        let w = hann::<f64>(1).unwrap();
        assert_eq!(w.as_slice(), &[1.0]);
    }

    #[test]
    fn test_hann_symmetry() {
        let w = hann::<f64>(8).unwrap();
        let s = w.as_slice();
        for i in 0..4 {
            assert!((s[i] - s[7 - i]).abs() < TOL, "asymmetric at {i}");
        }
    }

    #[test]
    fn test_hann_endpoints() {
        let w = hann::<f64>(5).unwrap();
        let s = w.as_slice();
        assert!(s[0].abs() < TOL);
        assert!(s[4].abs() < TOL);
    }

    #[test]
    fn test_hamming_5() {
        let w = hamming::<f64>(5).unwrap();
        let s = w.as_slice();
        // Known values: hamming(5) = [0.08, 0.54, 1.0, 0.54, 0.08]
        assert!((s[0] - 0.08).abs() < TOL);
        assert!((s[2] - 1.0).abs() < TOL);
        assert!((s[4] - 0.08).abs() < TOL);
    }

    #[test]
    fn test_blackman_sum_positive() {
        let w = blackman::<f64>(32).unwrap();
        let sum: f64 = w.as_slice().iter().sum();
        assert!(sum > 0.0);
    }

    #[test]
    fn test_bartlett_peak() {
        let w = bartlett::<f64>(5).unwrap();
        let s = w.as_slice();
        assert!(s[0].abs() < TOL);
        assert!((s[2] - 1.0).abs() < TOL);
        assert!(s[4].abs() < TOL);
    }

    #[test]
    fn test_rectangular() {
        let w = rectangular::<f64>(4).unwrap();
        assert!(w.as_slice().iter().all(|&x| (x - 1.0).abs() < TOL));
    }

    #[test]
    fn test_zero_length_error() {
        assert!(hann::<f64>(0).is_err());
        assert!(hamming::<f64>(0).is_err());
        assert!(blackman::<f64>(0).is_err());
        assert!(bartlett::<f64>(0).is_err());
        assert!(rectangular::<f64>(0).is_err());
    }
}
