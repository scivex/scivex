//! 1-D convolution and cross-correlation.

use scivex_core::{Float, Tensor};

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

    let sa = a.as_slice();
    let sb = b.as_slice();

    // Full convolution length.
    let full_len = na + nb - 1;
    let mut full = vec![T::zero(); full_len];
    for i in 0..na {
        for j in 0..nb {
            full[i + j] += sa[i] * sb[j];
        }
    }

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
}
