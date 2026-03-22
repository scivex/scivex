//! Select the *k* highest-scoring features.

use std::marker::PhantomData;

use scivex_core::{Float, Tensor};

use crate::error::{MlError, Result};
use crate::feature_selection::ScoringFunction;
use crate::feature_selection::scoring::{chi2, f_classif};

/// Univariate feature selector that keeps the *k* best features according to a
/// scoring function.
///
/// # Examples
///
/// ```
/// # use scivex_core::prelude::*;
/// # use scivex_ml::feature_selection::{SelectKBest, ScoringFunction};
/// let x = Tensor::from_vec(
///     vec![1.0_f64, 5.0, 2.0, 5.0, 10.0, 5.0, 11.0, 5.0],
///     vec![4, 2],
/// ).unwrap();
/// let y = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0], vec![4]).unwrap();
/// let mut sel = SelectKBest::<f64>::new(1, ScoringFunction::FClassif);
/// sel.fit(&x, &y).unwrap();
/// let x_new = sel.transform(&x).unwrap();
/// assert_eq!(x_new.shape(), &[4, 1]);
/// ```
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct SelectKBest<T: Float> {
    k: usize,
    scoring: ScoringFunction,
    scores: Option<Vec<T>>,
    selected_indices: Option<Vec<usize>>,
    _marker: PhantomData<T>,
}

impl<T: Float> SelectKBest<T> {
    /// Create a new `SelectKBest` that will keep `k` features scored by `scoring`.
    pub fn new(k: usize, scoring: ScoringFunction) -> Self {
        Self {
            k,
            scoring,
            scores: None,
            selected_indices: None,
            _marker: PhantomData,
        }
    }

    /// Compute feature scores and determine the top-k feature indices.
    pub fn fit(&mut self, x: &Tensor<T>, y: &Tensor<T>) -> Result<()> {
        let p = {
            let s = x.shape();
            if s.len() != 2 {
                return Err(MlError::DimensionMismatch {
                    expected: 2,
                    got: s.len(),
                });
            }
            s[1]
        };

        if self.k == 0 || self.k > p {
            return Err(MlError::InvalidParameter {
                name: "k",
                reason: "must be in [1, n_features]",
            });
        }

        let scores = match self.scoring {
            ScoringFunction::Chi2 => chi2(x, y)?,
            ScoringFunction::FClassif => f_classif(x, y)?,
        };

        // Find the k indices with the highest scores
        let mut indexed: Vec<(usize, T)> = scores.iter().copied().enumerate().collect();
        // Sort descending by score
        indexed.sort_by(|a, b| {
            b.1.to_f64()
                .partial_cmp(&a.1.to_f64())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut selected: Vec<usize> = indexed[..self.k].iter().map(|&(i, _)| i).collect();
        // Keep original column order
        selected.sort_unstable();

        self.scores = Some(scores);
        self.selected_indices = Some(selected);
        Ok(())
    }

    /// Project `x` onto the selected features.
    pub fn transform(&self, x: &Tensor<T>) -> Result<Tensor<T>> {
        let indices = self.selected_indices.as_ref().ok_or(MlError::NotFitted)?;
        let s = x.shape();
        if s.len() != 2 {
            return Err(MlError::DimensionMismatch {
                expected: 2,
                got: s.len(),
            });
        }
        let n = s[0];
        let p = s[1];
        let k = indices.len();
        let x_data = x.as_slice();

        let mut out = vec![T::zero(); n * k];
        for i in 0..n {
            for (out_j, &col) in indices.iter().enumerate() {
                out[i * k + out_j] = x_data[i * p + col];
            }
        }

        Ok(Tensor::from_vec(out, vec![n, k])?)
    }

    /// Fit and transform in one step.
    pub fn fit_transform(&mut self, x: &Tensor<T>, y: &Tensor<T>) -> Result<Tensor<T>> {
        self.fit(x, y)?;
        self.transform(x)
    }

    /// Per-feature scores computed during `fit`.
    pub fn scores(&self) -> Option<&[T]> {
        self.scores.as_deref()
    }

    /// Indices of the selected features (sorted in ascending order).
    pub fn selected_features(&self) -> Option<&[usize]> {
        self.selected_indices.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_select_k_best_f_classif() {
        // Feature 0 separates classes well, feature 1 does not
        let x = Tensor::from_vec(
            vec![1.0_f64, 5.0, 2.0, 5.0, 10.0, 5.0, 11.0, 5.0],
            vec![4, 2],
        )
        .unwrap();
        let y = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0], vec![4]).unwrap();

        let mut sel = SelectKBest::new(1, ScoringFunction::FClassif);
        let x_new = sel.fit_transform(&x, &y).unwrap();
        assert_eq!(x_new.shape(), &[4, 1]);
        // Should have selected feature 0
        assert_eq!(sel.selected_features().unwrap(), &[0]);
    }

    #[test]
    fn test_select_k_best_chi2() {
        let x =
            Tensor::from_vec(vec![1.0_f64, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0], vec![4, 2]).unwrap();
        let y = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0], vec![4]).unwrap();

        let mut sel = SelectKBest::new(1, ScoringFunction::Chi2);
        sel.fit(&x, &y).unwrap();
        let x_new = sel.transform(&x).unwrap();
        assert_eq!(x_new.shape(), &[4, 1]);
        assert!(sel.scores().is_some());
    }

    #[test]
    fn test_select_k_best_invalid_k() {
        let x = Tensor::from_vec(vec![1.0_f64, 2.0], vec![1, 2]).unwrap();
        let y = Tensor::from_vec(vec![0.0], vec![1]).unwrap();
        let mut sel = SelectKBest::<f64>::new(0, ScoringFunction::FClassif);
        assert!(sel.fit(&x, &y).is_err());
    }
}
