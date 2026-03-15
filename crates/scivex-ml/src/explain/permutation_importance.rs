//! Permutation feature importance.

use scivex_core::{Float, Tensor, random::Rng};

use crate::error::{MlError, Result};
use crate::traits::Predictor;

/// Result of permutation importance computation.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct PermutationImportanceResult<T: Float> {
    /// Mean importance per feature `[n_features]`.
    pub importances_mean: Tensor<T>,
    /// Std deviation of importance per feature `[n_features]`.
    pub importances_std: Tensor<T>,
    /// Raw importance values `[n_features, n_repeats]`.
    pub importances_raw: Tensor<T>,
}

/// Compute permutation feature importance for any [`Predictor`].
///
/// For each feature, the column is shuffled `n_repeats` times and the
/// decrease in `scorer(y_true, y_pred)` is recorded.
pub fn permutation_importance<T, P>(
    model: &P,
    x: &Tensor<T>,
    y: &Tensor<T>,
    n_repeats: usize,
    scorer: fn(&Tensor<T>, &Tensor<T>) -> T,
    seed: u64,
) -> Result<PermutationImportanceResult<T>>
where
    T: Float,
    P: Predictor<T>,
{
    let shape = x.shape();
    if shape.len() != 2 {
        return Err(MlError::DimensionMismatch {
            expected: 2,
            got: shape.len(),
        });
    }
    let (n, p) = (shape[0], shape[1]);

    let baseline_pred = model.predict(x)?;
    let baseline_score = scorer(y, &baseline_pred);

    let mut rng = Rng::new(seed);
    let mut raw = vec![T::zero(); p * n_repeats];

    for feat in 0..p {
        for rep in 0..n_repeats {
            // Create a copy with feature `feat` shuffled
            let mut x_data = x.as_slice().to_vec();
            // Fisher-Yates shuffle on column `feat`
            for i in (1..n).rev() {
                let j = rng.next_u64() as usize % (i + 1);
                x_data.swap(i * p + feat, j * p + feat);
            }
            let x_perm = Tensor::from_vec(x_data, vec![n, p])?;
            let perm_pred = model.predict(&x_perm)?;
            let perm_score = scorer(y, &perm_pred);
            raw[feat * n_repeats + rep] = baseline_score - perm_score;
        }
    }

    // Compute mean and std per feature
    let mut means = vec![T::zero(); p];
    let mut stds = vec![T::zero(); p];
    for feat in 0..p {
        let slice = &raw[feat * n_repeats..(feat + 1) * n_repeats];
        let mean = slice.iter().copied().fold(T::zero(), |a, b| a + b)
            / T::from_usize(n_repeats);
        means[feat] = mean;
        let var = slice
            .iter()
            .map(|&v| {
                let d = v - mean;
                d * d
            })
            .fold(T::zero(), |a, b| a + b)
            / T::from_usize(n_repeats);
        stds[feat] = var.sqrt();
    }

    Ok(PermutationImportanceResult {
        importances_mean: Tensor::from_vec(means, vec![p])?,
        importances_std: Tensor::from_vec(stds, vec![p])?,
        importances_raw: Tensor::from_vec(raw, vec![p, n_repeats])?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linear::LinearRegression;

    fn r2_score<T: Float>(y_true: &Tensor<T>, y_pred: &Tensor<T>) -> T {
        let yt = y_true.as_slice();
        let yp = y_pred.as_slice();
        let mean = yt.iter().copied().fold(T::zero(), |a, b| a + b) / T::from_usize(yt.len());
        let ss_res = yt
            .iter()
            .zip(yp.iter())
            .map(|(&t, &p)| (t - p) * (t - p))
            .fold(T::zero(), |a, b| a + b);
        let ss_tot = yt
            .iter()
            .map(|&t| (t - mean) * (t - mean))
            .fold(T::zero(), |a, b| a + b);
        T::one() - ss_res / ss_tot
    }

    #[test]
    fn test_permutation_importance_relevant_feature() {
        // y = 3*x0 + noise, x1 is random noise
        let mut rng = Rng::new(42);
        let n = 100;
        let mut x_data = Vec::with_capacity(n * 2);
        let mut y_data = Vec::with_capacity(n);
        for _ in 0..n {
            let x0 = rng.next_f64() * 10.0;
            let x1 = rng.next_f64() * 10.0;
            x_data.push(x0);
            x_data.push(x1);
            y_data.push(3.0 * x0 + rng.next_f64() * 0.1);
        }
        let x = Tensor::from_vec(x_data, vec![n, 2]).unwrap();
        let y = Tensor::from_vec(y_data, vec![n]).unwrap();

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let result = permutation_importance(&model, &x, &y, 5, r2_score, 123).unwrap();
        let means = result.importances_mean.as_slice();
        // Feature 0 should be much more important than feature 1
        assert!(means[0] > means[1] * 5.0);
    }

    #[test]
    fn test_permutation_importance_shape() {
        let x = Tensor::from_vec(
            vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![4, 2],
        )
        .unwrap();
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();

        let mut model = crate::tree::DecisionTreeRegressor::new(Some(3), 1);
        model.fit(&x, &y).unwrap();

        let result = permutation_importance(&model, &x, &y, 3, r2_score, 0).unwrap();
        assert_eq!(result.importances_mean.shape(), &[2]);
        assert_eq!(result.importances_std.shape(), &[2]);
        assert_eq!(result.importances_raw.shape(), &[2, 3]);
    }
}
