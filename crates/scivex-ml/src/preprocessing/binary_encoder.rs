use scivex_core::{Float, Tensor};

use crate::error::{MlError, Result};
use crate::traits::Transformer;

/// Binary encoder: encodes categorical integer values as binary digit columns.
///
/// Each input column is treated as a categorical feature whose values are
/// rounded to the nearest integer. During `fit`, the encoder learns the sorted
/// set of unique categories per column and computes the number of bits needed
/// (`ceil(log2(n_categories))`). During `transform`, each category index is
/// expanded into its binary representation, producing `n_bits` output columns
/// per input column.
///
/// Unknown categories encountered during `transform` produce all-zero binary
/// columns.
///
/// # Examples
///
/// ```
/// # use scivex_ml::prelude::*;
/// # use scivex_core::prelude::*;
/// let x = Tensor::from_vec(vec![0.0_f64, 1.0, 2.0, 3.0], vec![4, 1]).unwrap();
/// let mut enc = BinaryEncoder::new();
/// let out = enc.fit_transform(&x).unwrap();
/// // 4 categories → ceil(log2(4)) = 2 bits per column
/// assert_eq!(out.shape(), &[4, 2]);
/// ```
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct BinaryEncoder<T: Float> {
    /// Per-column sorted unique categories.
    categories: Option<Vec<Vec<T>>>,
    /// Per-column number of bits required.
    n_bits: Option<Vec<usize>>,
}

impl<T: Float> BinaryEncoder<T> {
    /// Create a new unfitted encoder.
    pub fn new() -> Self {
        Self {
            categories: None,
            n_bits: None,
        }
    }

    /// Return the learned categories per column, or `None` if not fitted.
    pub fn categories(&self) -> Option<&[Vec<T>]> {
        self.categories.as_deref()
    }

    /// Return the number of binary bits per column, or `None` if not fitted.
    pub fn n_bits_per_column(&self) -> Option<&[usize]> {
        self.n_bits.as_deref()
    }

    /// Total number of output columns after encoding.
    pub fn n_output_features(&self) -> Option<usize> {
        self.n_bits.as_ref().map(|bits| bits.iter().sum())
    }
}

impl<T: Float> Default for BinaryEncoder<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute the number of bits needed to represent `n` distinct values.
/// Returns at least 1 bit even for n <= 1.
fn bits_needed(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    let mut bits = 0usize;
    let mut val = n - 1; // max index
    while val > 0 {
        bits += 1;
        val >>= 1;
    }
    bits
}

impl<T: Float> Transformer<T> for BinaryEncoder<T> {
    fn fit(&mut self, x: &Tensor<T>) -> Result<()> {
        let (n, p) = matrix_shape(x)?;
        let data = x.as_slice();

        let mut categories = Vec::with_capacity(p);
        let mut n_bits = Vec::with_capacity(p);

        for col in 0..p {
            let mut unique: Vec<T> = Vec::new();
            for row in 0..n {
                let val = round_val(data[row * p + col]);
                if !unique.iter().any(|&u| (u - val).abs() < T::from_f64(1e-9)) {
                    unique.push(val);
                }
            }
            unique.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            n_bits.push(bits_needed(unique.len()));
            categories.push(unique);
        }

        self.categories = Some(categories);
        self.n_bits = Some(n_bits);
        Ok(())
    }

    fn transform(&self, x: &Tensor<T>) -> Result<Tensor<T>> {
        let cats = self.categories.as_ref().ok_or(MlError::NotFitted)?;
        let bits = self.n_bits.as_ref().ok_or(MlError::NotFitted)?;
        let (n, p) = matrix_shape(x)?;
        if p != cats.len() {
            return Err(MlError::DimensionMismatch {
                expected: cats.len(),
                got: p,
            });
        }

        let total_cols: usize = bits.iter().sum();
        let data = x.as_slice();
        let mut out = vec![T::zero(); n * total_cols];

        for row in 0..n {
            let mut col_offset = 0;
            for col in 0..p {
                let val = round_val(data[row * p + col]);
                let idx = cats[col]
                    .iter()
                    .position(|&c| (c - val).abs() < T::from_f64(1e-9));

                if let Some(index) = idx {
                    // Write binary digits, most-significant bit first
                    let nb = bits[col];
                    for bit in (0..nb).rev() {
                        if index & (1 << bit) != 0 {
                            out[row * total_cols + col_offset + (nb - 1 - bit)] = T::one();
                        }
                    }
                }
                // Unknown categories → all zeros (already initialised)
                col_offset += bits[col];
            }
        }

        Tensor::from_vec(out, vec![n, total_cols]).map_err(MlError::from)
    }
}

/// Round to nearest integer (using floor(x + 0.5)).
fn round_val<T: Float>(v: T) -> T {
    (v + T::from_f64(0.5)).floor()
}

fn matrix_shape<T: Float>(x: &Tensor<T>) -> Result<(usize, usize)> {
    let s = x.shape();
    if s.len() != 2 {
        return Err(MlError::DimensionMismatch {
            expected: 2,
            got: s.len(),
        });
    }
    if s[0] == 0 {
        return Err(MlError::EmptyInput);
    }
    Ok((s[0], s[1]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_basic() {
        // 4 categories: {0,1,2,3} → 2 bits
        let x = Tensor::from_vec(vec![0.0_f64, 1.0, 2.0, 3.0], vec![4, 1]).unwrap();
        let mut enc = BinaryEncoder::new();
        let out = enc.fit_transform(&x).unwrap();

        assert_eq!(out.shape(), &[4, 2]);
        let d = out.as_slice();
        // 0 → 00, 1 → 01, 2 → 10, 3 → 11
        assert_eq!(&d[0..2], &[0.0, 0.0]);
        assert_eq!(&d[2..4], &[0.0, 1.0]);
        assert_eq!(&d[4..6], &[1.0, 0.0]);
        assert_eq!(&d[6..8], &[1.0, 1.0]);
    }

    #[test]
    fn test_binary_multi_column() {
        // col0: {0,1} → 1 bit, col1: {10,20,30} → 2 bits = 3 total cols
        let x = Tensor::from_vec(vec![0.0_f64, 10.0, 1.0, 20.0, 0.0, 30.0], vec![3, 2]).unwrap();
        let mut enc = BinaryEncoder::new();
        let out = enc.fit_transform(&x).unwrap();

        assert_eq!(out.shape(), &[3, 3]);
        let d = out.as_slice();
        // Row 0: col0=0→idx0→[0], col1=10→idx0→[0,0]
        assert_eq!(&d[0..3], &[0.0, 0.0, 0.0]);
        // Row 1: col0=1→idx1→[1], col1=20→idx1→[0,1]
        assert_eq!(&d[3..6], &[1.0, 0.0, 1.0]);
        // Row 2: col0=0→idx0→[0], col1=30→idx2→[1,0]
        assert_eq!(&d[6..9], &[0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_binary_unknown_category() {
        let x_train = Tensor::from_vec(vec![1.0_f64, 2.0], vec![2, 1]).unwrap();
        let mut enc = BinaryEncoder::new();
        enc.fit(&x_train).unwrap();

        let x_test = Tensor::from_vec(vec![1.0, 99.0], vec![2, 1]).unwrap();
        let out = enc.transform(&x_test).unwrap();
        // {1,2} → 1 bit; val=1→idx0→[0], val=99→unknown→[0]
        assert_eq!(out.as_slice(), &[0.0, 0.0]);
    }
}
