//! Partial dependence plots (data generation).

use scivex_core::{Float, Tensor};

use crate::error::{MlError, Result};
use crate::traits::Predictor;

/// Partial dependence result for a single feature.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct PartialDependence<T: Float> {
    /// Grid values for the feature `[grid_resolution]`.
    pub feature_values: Tensor<T>,
    /// Average predictions at each grid point `[grid_resolution]`.
    pub average_predictions: Tensor<T>,
}

/// Compute partial dependence for a single feature.
///
/// For each grid point value, the feature column is replaced for all samples,
/// the model predicts, and the average prediction is returned.
pub fn partial_dependence<T, P>(
    model: &P,
    x: &Tensor<T>,
    feature_index: usize,
    grid_resolution: usize,
) -> Result<PartialDependence<T>>
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
    if feature_index >= p {
        return Err(MlError::DimensionMismatch {
            expected: p,
            got: feature_index + 1,
        });
    }
    if grid_resolution == 0 {
        return Err(MlError::InvalidParameter {
            name: "grid_resolution",
            reason: "must be at least 1",
        });
    }

    let data = x.as_slice();

    // Find min/max of the feature
    let mut fmin = data[feature_index];
    let mut fmax = data[feature_index];
    for i in 1..n {
        let v = data[i * p + feature_index];
        if v < fmin {
            fmin = v;
        }
        if v > fmax {
            fmax = v;
        }
    }

    // Build grid
    let grid: Vec<T> = if grid_resolution == 1 {
        vec![(fmin + fmax) / T::from_usize(2)]
    } else {
        let step = (fmax - fmin) / T::from_usize(grid_resolution - 1);
        (0..grid_resolution)
            .map(|i| fmin + step * T::from_usize(i))
            .collect()
    };

    // Compute average predictions at each grid point
    let mut avg_preds = Vec::with_capacity(grid_resolution);
    for &grid_val in &grid {
        let mut x_mod = data.to_vec();
        for i in 0..n {
            x_mod[i * p + feature_index] = grid_val;
        }
        let x_tensor = Tensor::from_vec(x_mod, vec![n, p])?;
        let preds = model.predict(&x_tensor)?;
        let pred_slice = preds.as_slice();
        let mean =
            pred_slice.iter().copied().fold(T::zero(), |a, b| a + b) / T::from_usize(n);
        avg_preds.push(mean);
    }

    Ok(PartialDependence {
        feature_values: Tensor::from_vec(grid, vec![grid_resolution])?,
        average_predictions: Tensor::from_vec(avg_preds, vec![grid_resolution])?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::DecisionTreeRegressor;

    #[test]
    fn test_pdp_shape() {
        let x = Tensor::from_vec(
            vec![1.0_f64, 10.0, 2.0, 20.0, 3.0, 15.0, 4.0, 40.0],
            vec![4, 2],
        )
        .unwrap();
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();

        let mut model = DecisionTreeRegressor::new(Some(3), 1);
        model.fit(&x, &y).unwrap();

        let pdp = partial_dependence(&model, &x, 0, 10).unwrap();
        assert_eq!(pdp.feature_values.shape(), &[10]);
        assert_eq!(pdp.average_predictions.shape(), &[10]);
    }

    #[test]
    fn test_pdp_monotonic() {
        // y ≈ x0, so PDP on feature 0 should be non-decreasing
        let x = Tensor::from_vec(
            vec![
                1.0_f64, 5.0, 2.0, 3.0, 3.0, 7.0, 4.0, 1.0, 5.0, 4.0,
            ],
            vec![5, 2],
        )
        .unwrap();
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();

        let mut model = DecisionTreeRegressor::new(Some(5), 1);
        model.fit(&x, &y).unwrap();

        let pdp = partial_dependence(&model, &x, 0, 5).unwrap();
        let preds = pdp.average_predictions.as_slice();
        // At minimum, first grid point prediction <= last
        assert!(preds[preds.len() - 1] >= preds[0] - 1e-6);
    }

    #[test]
    fn test_pdp_feature_out_of_bounds() {
        let x = Tensor::from_vec(
            vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![3, 2],
        )
        .unwrap();
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();

        let mut model = DecisionTreeRegressor::new(Some(3), 1);
        model.fit(&x, &y).unwrap();

        assert!(partial_dependence(&model, &x, 5, 10).is_err());
    }
}
