use scivex_core::{Float, Tensor};

use crate::error::{MlError, Result};
use crate::traits::Transformer;

/// Ordinal encoder: maps categorical values to ordered integer indices.
///
/// Each input column is treated as a categorical feature whose values are
/// rounded to the nearest integer. During `fit`, the encoder learns the sorted
/// set of unique categories per column. During `transform`, each value is
/// replaced with its zero-based ordinal index in that sorted set.
///
/// Unknown categories encountered during `transform` are mapped to the value
/// returned by [`OrdinalEncoder::unknown_value`] (defaults to `NaN`).
///
/// # Examples
///
/// ```
/// # use scivex_ml::prelude::*;
/// # use scivex_core::prelude::*;
/// let x = Tensor::from_vec(vec![3.0_f64, 1.0, 2.0], vec![3, 1]).unwrap();
/// let mut enc = OrdinalEncoder::new();
/// let out = enc.fit_transform(&x).unwrap();
/// // Sorted categories: [1, 2, 3] → ordinals [2, 0, 1]
/// assert_eq!(out.as_slice(), &[2.0, 0.0, 1.0]);
/// ```
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct OrdinalEncoder<T: Float> {
    /// Per-column sorted unique categories.
    categories: Option<Vec<Vec<T>>>,
    /// Value used for unknown categories during transform.
    unknown_value: T,
}

impl<T: Float> OrdinalEncoder<T> {
    /// Create a new unfitted encoder.
    ///
    /// Unknown categories are mapped to `NaN` by default.
    pub fn new() -> Self {
        Self {
            categories: None,
            unknown_value: T::nan(),
        }
    }

    /// Create a new encoder that maps unknown categories to `value`.
    pub fn with_unknown_value(value: T) -> Self {
        Self {
            categories: None,
            unknown_value: value,
        }
    }

    /// Return the learned categories per column, or `None` if not fitted.
    pub fn categories(&self) -> Option<&[Vec<T>]> {
        self.categories.as_deref()
    }

    /// Value used for unknown categories.
    pub fn unknown_value(&self) -> T {
        self.unknown_value
    }
}

impl<T: Float> Default for OrdinalEncoder<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float> Transformer<T> for OrdinalEncoder<T> {
    fn fit(&mut self, x: &Tensor<T>) -> Result<()> {
        let (n, p) = matrix_shape(x)?;
        let data = x.as_slice();

        let mut categories = Vec::with_capacity(p);
        for col in 0..p {
            let mut unique: Vec<T> = Vec::new();
            for row in 0..n {
                let val = round_val(data[row * p + col]);
                if !unique.iter().any(|&u| (u - val).abs() < T::from_f64(1e-9)) {
                    unique.push(val);
                }
            }
            unique.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            categories.push(unique);
        }

        self.categories = Some(categories);
        Ok(())
    }

    fn transform(&self, x: &Tensor<T>) -> Result<Tensor<T>> {
        let cats = self.categories.as_ref().ok_or(MlError::NotFitted)?;
        let (n, p) = matrix_shape(x)?;
        if p != cats.len() {
            return Err(MlError::DimensionMismatch {
                expected: cats.len(),
                got: p,
            });
        }

        let data = x.as_slice();
        let mut out = vec![T::zero(); n * p];

        for row in 0..n {
            for col in 0..p {
                let val = round_val(data[row * p + col]);
                let idx = cats[col]
                    .iter()
                    .position(|&c| (c - val).abs() < T::from_f64(1e-9));
                out[row * p + col] = match idx {
                    Some(i) => T::from_f64(i as f64),
                    None => self.unknown_value,
                };
            }
        }

        Tensor::from_vec(out, vec![n, p]).map_err(MlError::from)
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
    fn test_ordinal_basic() {
        // 1 feature with categories {3, 1, 2} → sorted [1, 2, 3] → ordinals [2, 0, 1]
        let x = Tensor::from_vec(vec![3.0_f64, 1.0, 2.0], vec![3, 1]).unwrap();
        let mut enc = OrdinalEncoder::new();
        let out = enc.fit_transform(&x).unwrap();

        assert_eq!(out.shape(), &[3, 1]);
        assert_eq!(out.as_slice(), &[2.0, 0.0, 1.0]);
    }

    #[test]
    fn test_ordinal_multi_column() {
        // 2 features: col0={0,1,2}, col1={10,20}
        let x = Tensor::from_vec(vec![0.0_f64, 10.0, 1.0, 20.0, 2.0, 10.0], vec![3, 2]).unwrap();
        let mut enc = OrdinalEncoder::new();
        let out = enc.fit_transform(&x).unwrap();

        assert_eq!(out.shape(), &[3, 2]);
        let d = out.as_slice();
        // col0 sorted: [0,1,2] → [0,1,2], col1 sorted: [10,20] → [0,1]
        assert_eq!(d, &[0.0, 0.0, 1.0, 1.0, 2.0, 0.0]);
    }

    #[test]
    fn test_ordinal_unknown_category() {
        let x_train = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0], vec![3, 1]).unwrap();
        let mut enc = OrdinalEncoder::with_unknown_value(-1.0_f64);
        enc.fit(&x_train).unwrap();

        let x_test = Tensor::from_vec(vec![1.0, 99.0], vec![2, 1]).unwrap();
        let out = enc.transform(&x_test).unwrap();
        assert_eq!(out.as_slice(), &[0.0, -1.0]);
    }
}
