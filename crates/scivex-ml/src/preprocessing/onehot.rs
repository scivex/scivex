use scivex_core::{Float, Tensor};

use crate::error::{MlError, Result};
use crate::traits::Transformer;

/// One-hot encoder: converts integer-valued categorical features into binary
/// indicator columns.
///
/// Each input column is treated as a categorical feature whose values are
/// rounded to the nearest integer. During `fit`, the encoder learns the set
/// of unique categories per column. During `transform`, each category is
/// expanded into a binary column.
///
/// # Example
///
/// Input (2 features, 3 categories each):
/// ```text
/// [[0, 1],
///  [1, 2],
///  [2, 0]]
/// ```
///
/// Output (3 + 3 = 6 columns):
/// ```text
/// [[1, 0, 0, 0, 1, 0],
///  [0, 1, 0, 0, 0, 1],
///  [0, 0, 1, 1, 0, 0]]
/// ```
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct OneHotEncoder<T: Float> {
    /// Per-column sorted unique categories.
    categories: Option<Vec<Vec<T>>>,
}

impl<T: Float> OneHotEncoder<T> {
    /// Create a new unfitted encoder.
    pub fn new() -> Self {
        Self { categories: None }
    }

    /// Return the learned categories per column, or `None` if not fitted.
    pub fn categories(&self) -> Option<&[Vec<T>]> {
        self.categories.as_deref()
    }

    /// Total number of output columns after encoding.
    pub fn n_output_features(&self) -> Option<usize> {
        self.categories
            .as_ref()
            .map(|cats| cats.iter().map(Vec::len).sum())
    }
}

impl<T: Float> Default for OneHotEncoder<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float> Transformer<T> for OneHotEncoder<T> {
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
            // Sort categories for deterministic output order
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

        let total_cols: usize = cats.iter().map(Vec::len).sum();
        let data = x.as_slice();
        let mut out = vec![T::zero(); n * total_cols];

        for row in 0..n {
            let mut col_offset = 0;
            for col in 0..p {
                let val = round_val(data[row * p + col]);
                for (k, &cat) in cats[col].iter().enumerate() {
                    if (cat - val).abs() < T::from_f64(1e-9) {
                        out[row * total_cols + col_offset + k] = T::one();
                        break;
                    }
                }
                col_offset += cats[col].len();
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
    fn test_onehot_basic() {
        // 2 features: col0 has {0,1,2}, col1 has {0,1}
        let x = Tensor::from_vec(vec![0.0_f64, 0.0, 1.0, 1.0, 2.0, 0.0], vec![3, 2]).unwrap();

        let mut enc = OneHotEncoder::new();
        let out = enc.fit_transform(&x).unwrap();

        // col0: 3 categories → 3 cols, col1: 2 categories → 2 cols = 5 total
        assert_eq!(out.shape(), &[3, 5]);
        let d = out.as_slice();

        // Row 0: val=(0,0) → [1,0,0, 1,0]
        assert_eq!(&d[0..5], &[1.0, 0.0, 0.0, 1.0, 0.0]);
        // Row 1: val=(1,1) → [0,1,0, 0,1]
        assert_eq!(&d[5..10], &[0.0, 1.0, 0.0, 0.0, 1.0]);
        // Row 2: val=(2,0) → [0,0,1, 1,0]
        assert_eq!(&d[10..15], &[0.0, 0.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn test_onehot_single_column() {
        let x = Tensor::from_vec(vec![3.0_f64, 1.0, 2.0, 1.0], vec![4, 1]).unwrap();
        let mut enc = OneHotEncoder::new();
        let out = enc.fit_transform(&x).unwrap();

        assert_eq!(out.shape(), &[4, 3]); // 3 unique: {1, 2, 3}
        let d = out.as_slice();
        // Sorted categories: [1, 2, 3]
        // val=3 → [0, 0, 1]
        assert_eq!(&d[0..3], &[0.0, 0.0, 1.0]);
        // val=1 → [1, 0, 0]
        assert_eq!(&d[3..6], &[1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_onehot_not_fitted() {
        let enc = OneHotEncoder::<f64>::new();
        let x = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]).unwrap();
        assert!(enc.transform(&x).is_err());
    }

    #[test]
    fn test_onehot_categories() {
        let x = Tensor::from_vec(vec![2.0_f64, 0.0, 1.0], vec![3, 1]).unwrap();
        let mut enc = OneHotEncoder::new();
        enc.fit(&x).unwrap();

        let cats = enc.categories().unwrap();
        assert_eq!(cats.len(), 1);
        assert_eq!(cats[0], vec![0.0, 1.0, 2.0]);
        assert_eq!(enc.n_output_features(), Some(3));
    }

    #[test]
    fn test_onehot_transform_mismatch_cols() {
        let x = Tensor::from_vec(vec![0.0_f64, 1.0], vec![2, 1]).unwrap();
        let mut enc = OneHotEncoder::new();
        enc.fit(&x).unwrap();

        // Try to transform with 2 columns instead of 1
        let x2 = Tensor::from_vec(vec![0.0, 1.0, 0.0, 1.0], vec![2, 2]).unwrap();
        assert!(enc.transform(&x2).is_err());
    }

    #[test]
    fn test_onehot_unknown_category() {
        // Unseen category during transform → all-zero for that feature
        let x_train = Tensor::from_vec(vec![0.0_f64, 1.0], vec![2, 1]).unwrap();
        let mut enc = OneHotEncoder::new();
        enc.fit(&x_train).unwrap();

        let x_test = Tensor::from_vec(vec![0.0, 1.0, 99.0], vec![3, 1]).unwrap();
        let out = enc.transform(&x_test).unwrap();
        // val=99 is unknown → [0, 0]
        assert_eq!(&out.as_slice()[4..6], &[0.0, 0.0]);
    }
}
