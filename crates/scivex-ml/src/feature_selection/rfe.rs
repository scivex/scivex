//! Recursive Feature Elimination (RFE).

use std::marker::PhantomData;

use scivex_core::{Float, Tensor};

use crate::error::{MlError, Result};
use crate::traits::{HasFeatureImportances, Predictor};

/// Recursive Feature Elimination.
///
/// Repeatedly trains an estimator and removes the least important feature(s)
/// until `n_features_to_select` remain.
///
/// # Examples
///
/// ```
/// # use scivex_core::prelude::*;
/// # use scivex_ml::feature_selection::RFE;
/// # use scivex_ml::linear::LinearRegression;
/// let x = Tensor::from_vec(
///     vec![1.0_f64, 9.0, 0.3,
///          2.0, 3.0, 0.7,
///          3.0, 7.0, 0.1,
///          4.0, 1.0, 0.9,
///          5.0, 5.0, 0.5,
///          6.0, 8.0, 0.2,
///          7.0, 2.0, 0.8],
///     vec![7, 3],
/// ).unwrap();
/// let y = Tensor::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0], vec![7]).unwrap();
/// let estimator = LinearRegression::<f64>::new();
/// let mut rfe = RFE::<f64>::new(1);
/// rfe.fit(&estimator, &x, &y).unwrap();
/// let x_new = rfe.transform(&x).unwrap();
/// assert_eq!(x_new.shape(), &[7, 1]);
/// ```
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct RFE<T: Float> {
    n_features_to_select: usize,
    step: usize,
    support: Option<Vec<bool>>,
    ranking: Option<Vec<usize>>,
    _marker: PhantomData<T>,
}

impl<T: Float> RFE<T> {
    /// Create a new RFE that will select `n_features_to_select` features,
    /// removing one feature per iteration.
    pub fn new(n_features_to_select: usize) -> Self {
        Self {
            n_features_to_select,
            step: 1,
            support: None,
            ranking: None,
            _marker: PhantomData,
        }
    }

    /// Set the number of features to remove per iteration.
    pub fn set_step(&mut self, step: usize) -> &mut Self {
        self.step = step;
        self
    }

    /// Recursively eliminate features using the estimator's feature importances.
    pub fn fit<M>(&mut self, estimator: &M, x: &Tensor<T>, y: &Tensor<T>) -> Result<()>
    where
        M: Predictor<T> + HasFeatureImportances<T> + Clone,
    {
        let s = x.shape();
        if s.len() != 2 {
            return Err(MlError::DimensionMismatch {
                expected: 2,
                got: s.len(),
            });
        }
        let n = s[0];
        let total_features = s[1];

        if self.n_features_to_select == 0 || self.n_features_to_select > total_features {
            return Err(MlError::InvalidParameter {
                name: "n_features_to_select",
                reason: "must be in [1, n_features]",
            });
        }
        if self.step == 0 {
            return Err(MlError::InvalidParameter {
                name: "step",
                reason: "must be at least 1",
            });
        }

        // Track which original features are still active
        let mut active: Vec<bool> = vec![true; total_features];
        let mut ranking: Vec<usize> = vec![1; total_features]; // 1 = selected

        loop {
            let active_indices: Vec<usize> = active
                .iter()
                .enumerate()
                .filter_map(|(i, &a)| if a { Some(i) } else { None })
                .collect();
            let n_active = active_indices.len();

            if n_active <= self.n_features_to_select {
                break;
            }

            // Build sub-matrix with only active features
            let x_sub = subset_columns(x, &active_indices, n)?;

            // Train and get importances
            let mut est = estimator.clone();
            est.fit(&x_sub, y)?;
            let importances = est.feature_importances()?;

            // Determine how many to remove this round
            let n_to_remove = self.step.min(n_active - self.n_features_to_select);

            // Find indices of the least important features (in the sub-matrix)
            let mut indexed: Vec<(usize, T)> = importances.iter().copied().enumerate().collect();
            // Sort ascending by importance
            indexed.sort_by(|a, b| {
                a.1.to_f64()
                    .partial_cmp(&b.1.to_f64())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let rank = total_features - n_active + 2;
            for &(sub_idx, _) in indexed.iter().take(n_to_remove) {
                let orig_idx = active_indices[sub_idx];
                active[orig_idx] = false;
                ranking[orig_idx] = rank;
            }
        }

        // Ensure all remaining active features get rank 1
        for (i, &a) in active.iter().enumerate() {
            if a {
                ranking[i] = 1;
            }
        }

        self.support = Some(active);
        self.ranking = Some(ranking);
        Ok(())
    }

    /// Project `x` onto the selected features.
    pub fn transform(&self, x: &Tensor<T>) -> Result<Tensor<T>> {
        let support = self.support.as_ref().ok_or(MlError::NotFitted)?;
        let selected: Vec<usize> = support
            .iter()
            .enumerate()
            .filter_map(|(i, &s)| if s { Some(i) } else { None })
            .collect();

        let s = x.shape();
        if s.len() != 2 {
            return Err(MlError::DimensionMismatch {
                expected: 2,
                got: s.len(),
            });
        }
        let n = s[0];

        subset_columns(x, &selected, n)
    }

    /// Boolean mask of selected features.
    pub fn support(&self) -> Option<&[bool]> {
        self.support.as_deref()
    }

    /// Feature ranking: 1 = selected, higher = eliminated earlier.
    pub fn ranking(&self) -> Option<&[usize]> {
        self.ranking.as_deref()
    }
}

/// Extract a subset of columns from a 2-D tensor.
fn subset_columns<T: Float>(x: &Tensor<T>, cols: &[usize], n: usize) -> Result<Tensor<T>> {
    let p = x.shape()[1];
    let k = cols.len();
    let x_data = x.as_slice();
    let mut out = vec![T::zero(); n * k];
    for i in 0..n {
        for (out_j, &col) in cols.iter().enumerate() {
            out[i * k + out_j] = x_data[i * p + col];
        }
    }
    Ok(Tensor::from_vec(out, vec![n, k])?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linear::LinearRegression;

    #[test]
    fn test_rfe_basic() {
        // y ≈ 2*x0; x1 is random noise, x2 is weak noise
        // Features are non-collinear so QR succeeds at every subset.
        #[rustfmt::skip]
        let x = Tensor::from_vec(
            vec![
                1.0_f64, 9.0, 0.3,
                2.0,     3.0, 0.7,
                3.0,     7.0, 0.1,
                4.0,     1.0, 0.9,
                5.0,     5.0, 0.5,
                6.0,     8.0, 0.2,
                7.0,     2.0, 0.8,
            ],
            vec![7, 3],
        )
        .unwrap();
        let y = Tensor::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0], vec![7]).unwrap();

        let estimator = LinearRegression::<f64>::new();
        let mut rfe = RFE::new(1);
        rfe.fit(&estimator, &x, &y).unwrap();

        let support = rfe.support().unwrap();
        assert_eq!(support.iter().filter(|&&s| s).count(), 1);

        let x_new = rfe.transform(&x).unwrap();
        assert_eq!(x_new.shape(), &[7, 1]);
    }

    #[test]
    fn test_rfe_select_two() {
        // y ≈ 2*x0 + 3*x2; x1 is random noise — all 3 features non-collinear
        #[rustfmt::skip]
        let x = Tensor::from_vec(
            vec![
                1.0_f64, 9.0, 0.5,
                2.0,     3.0, 1.5,
                3.0,     7.0, 0.2,
                4.0,     1.0, 2.0,
                5.0,     5.0, 1.0,
                6.0,     8.0, 3.0,
                7.0,     2.0, 0.8,
            ],
            vec![7, 3],
        )
        .unwrap();
        let y = Tensor::from_vec(vec![3.5, 8.5, 6.6, 14.0, 13.0, 21.0, 16.4], vec![7]).unwrap();

        let estimator = LinearRegression::<f64>::new();
        let mut rfe = RFE::new(2);
        rfe.fit(&estimator, &x, &y).unwrap();

        let support = rfe.support().unwrap();
        assert_eq!(support.iter().filter(|&&s| s).count(), 2);
    }

    #[test]
    fn test_rfe_transform_before_fit() {
        let rfe = RFE::<f64>::new(1);
        let x = Tensor::from_vec(vec![1.0_f64, 2.0], vec![1, 2]).unwrap();
        assert!(rfe.transform(&x).is_err());
    }
}
