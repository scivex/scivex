//! LIME — Local Interpretable Model-agnostic Explanations.
//!
//! Perturbs each instance, weights perturbations by proximity to the original,
//! fits a local weighted linear model, and returns per-feature importance weights.

use scivex_core::{Float, Tensor, random::Rng};

use crate::error::{MlError, Result};
use crate::traits::Predictor;

/// Result of a LIME explanation for a single instance.
#[derive(Debug, Clone)]
pub struct LimeExplanation<T: Float> {
    /// Feature weights (importance scores) — positive means feature pushes prediction up.
    pub weights: Vec<T>,
    /// Intercept of the local linear model.
    pub intercept: T,
    /// Local model R² score (goodness of fit).
    pub score: T,
    /// The prediction for the explained instance.
    pub prediction: T,
}

/// Generate LIME explanations for instances.
///
/// Perturbs each instance, weights perturbations by proximity,
/// fits a local linear model, and returns feature weights.
///
/// # Arguments
/// - `model` — any fitted Predictor
/// - `x_train` — training data `[n_samples, n_features]` (used for feature statistics)
/// - `x_explain` — instances to explain `[n_explain, n_features]`
/// - `n_perturbations` — number of perturbed samples per instance
/// - `kernel_width` — width of exponential kernel (default: `sqrt(n_features) * 0.75`)
/// - `seed` — random seed
pub fn lime<T, P>(
    model: &P,
    x_train: &Tensor<T>,
    x_explain: &Tensor<T>,
    n_perturbations: usize,
    kernel_width: Option<T>,
    seed: u64,
) -> Result<Vec<LimeExplanation<T>>>
where
    T: Float,
    P: Predictor<T>,
{
    let train_shape = x_train.shape();
    let ex_shape = x_explain.shape();

    if train_shape.len() != 2 || ex_shape.len() != 2 {
        let bad_len = if train_shape.len() == 2 {
            ex_shape.len()
        } else {
            train_shape.len()
        };
        return Err(MlError::DimensionMismatch {
            expected: 2,
            got: bad_len,
        });
    }

    let n_features = train_shape[1];
    if n_features == 0 {
        return Err(MlError::EmptyInput);
    }
    if ex_shape[1] != n_features {
        return Err(MlError::DimensionMismatch {
            expected: n_features,
            got: ex_shape[1],
        });
    }

    let n_train = train_shape[0];
    if n_train == 0 {
        return Err(MlError::EmptyInput);
    }

    let n_explain = ex_shape[0];
    if n_explain == 0 {
        return Err(MlError::EmptyInput);
    }

    let train_data = x_train.as_slice();
    let ex_data = x_explain.as_slice();

    // Compute per-feature mean and std from training data.
    let (means, stds) = feature_stats(train_data, n_train, n_features);

    let sigma =
        kernel_width.unwrap_or_else(|| T::from_f64(0.75) * T::from_usize(n_features).sqrt());

    let mut rng = Rng::new(seed);
    let mut results = Vec::with_capacity(n_explain);

    for ei in 0..n_explain {
        let x_row = &ex_data[ei * n_features..(ei + 1) * n_features];

        // Get prediction for the original instance.
        let original_pred = {
            let t = Tensor::from_vec(x_row.to_vec(), vec![1, n_features])?;
            let pred = model.predict(&t)?;
            pred.as_slice()[0]
        };

        // Generate binary masks and perturbed samples.
        let masks = generate_binary_masks(&mut rng, n_perturbations, n_features);

        // Build perturbed dataset and get predictions.
        let mut perturbed = Vec::with_capacity(n_perturbations * n_features);
        for mi in 0..n_perturbations {
            let mask = &masks[mi * n_features..(mi + 1) * n_features];
            for j in 0..n_features {
                if mask[j] {
                    perturbed.push(x_row[j]);
                } else {
                    // Replace with random value from training distribution.
                    let noise = T::from_f64(rng.next_f64());
                    perturbed.push(means[j] + noise * stds[j]);
                }
            }
        }

        let perturbed_tensor = Tensor::from_vec(perturbed, vec![n_perturbations, n_features])?;
        let pred_tensor = model.predict(&perturbed_tensor)?;
        let preds = pred_tensor.as_slice();

        // Compute kernel weights from Hamming distances.
        let kernel_weights = exponential_kernel(&masks, n_perturbations, n_features, sigma);

        // Fit weighted linear regression on binary mask matrix (with intercept).
        // Design matrix: [1, mask_0, mask_1, ..., mask_{p-1}]  (n_perturbations x (p+1))
        let (beta, r2) = weighted_least_squares_with_intercept(
            &masks,
            preds,
            &kernel_weights,
            n_perturbations,
            n_features,
        );

        // beta[0] is intercept, beta[1..] are feature weights.
        let intercept = beta[0];
        let weights = beta[1..].to_vec();

        results.push(LimeExplanation {
            weights,
            intercept,
            score: r2,
            prediction: original_pred,
        });
    }

    Ok(results)
}

/// Compute per-feature mean and standard deviation.
fn feature_stats<T: Float>(data: &[T], n: usize, p: usize) -> (Vec<T>, Vec<T>) {
    let n_t = T::from_usize(n);
    let mut means = vec![T::zero(); p];
    let mut vars = vec![T::zero(); p];

    for i in 0..n {
        for j in 0..p {
            means[j] += data[i * p + j];
        }
    }
    for mean in means.iter_mut().take(p) {
        *mean /= n_t;
    }

    for i in 0..n {
        for j in 0..p {
            let diff = data[i * p + j] - means[j];
            vars[j] += diff * diff;
        }
    }

    let stds: Vec<T> = vars
        .iter()
        .map(|&v| {
            let s = (v / n_t).sqrt();
            // Avoid zero std (would collapse perturbations).
            if s < T::epsilon() { T::one() } else { s }
        })
        .collect();

    (means, stds)
}

/// Generate random binary masks: each feature is on (true) or off (false).
fn generate_binary_masks(rng: &mut Rng, n: usize, p: usize) -> Vec<bool> {
    let mut masks = Vec::with_capacity(n * p);
    for _ in 0..n {
        for _ in 0..p {
            masks.push(rng.next_f64() > 0.5);
        }
    }
    masks
}

/// Compute exponential kernel weights: exp(-d² / σ²) where d is Hamming distance.
fn exponential_kernel<T: Float>(masks: &[bool], n: usize, p: usize, sigma: T) -> Vec<T> {
    let sigma_sq = sigma * sigma;
    let mut weights = Vec::with_capacity(n);
    for i in 0..n {
        let mut dist = T::zero();
        for j in 0..p {
            if !masks[i * p + j] {
                dist += T::one();
            }
        }
        let w = (-(dist * dist) / sigma_sq).exp();
        weights.push(w);
    }
    weights
}

/// Weighted least squares with intercept column.
///
/// Design matrix X is [1 | mask_binary], shape (n, p+1).
/// Solves (X^T W X) β = X^T W y, then computes R².
/// Returns (beta, r2).
fn weighted_least_squares_with_intercept<T: Float>(
    masks: &[bool],
    y: &[T],
    w: &[T],
    n: usize,
    p: usize,
) -> (Vec<T>, T) {
    let cols = p + 1; // intercept + p features

    // Build X^T W X and X^T W y.
    let mut xt_wx = vec![T::zero(); cols * cols];
    let mut xt_wy = vec![T::zero(); cols];

    for i in 0..n {
        let wi = w[i];
        let yi = y[i];

        // x_row[0] = 1 (intercept), x_row[j+1] = mask value
        // Intercept column interactions.
        xt_wx[0] += wi; // 1 * wi * 1
        xt_wy[0] += wi * yi;

        for j in 0..p {
            let xj = if masks[i * p + j] {
                T::one()
            } else {
                T::zero()
            };

            // intercept-feature cross terms
            xt_wx[j + 1] += wi * xj; // row 0, col j+1
            xt_wx[(j + 1) * cols] += wi * xj; // row j+1, col 0

            xt_wy[j + 1] += xj * wi * yi;

            for k in 0..p {
                let xk = if masks[i * p + k] {
                    T::one()
                } else {
                    T::zero()
                };
                xt_wx[(j + 1) * cols + (k + 1)] += xj * wi * xk;
            }
        }
    }

    // Ridge regularization for numerical stability.
    for j in 0..cols {
        xt_wx[j * cols + j] += T::from_f64(1e-10);
    }

    let beta = gauss_solve(&mut xt_wx, &mut xt_wy, cols);

    // Compute weighted R² score.
    let r2 = compute_weighted_r2(masks, y, w, &beta, n, p);

    (beta, r2)
}

/// Gaussian elimination with partial pivoting.
fn gauss_solve<T: Float>(a: &mut [T], b: &mut [T], n: usize) -> Vec<T> {
    for col in 0..n {
        // Partial pivot.
        let mut max_row = col;
        let mut max_val = a[col * n + col].abs();
        for row in (col + 1)..n {
            let v = a[row * n + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_row != col {
            for k in 0..n {
                a.swap(col * n + k, max_row * n + k);
            }
            b.swap(col, max_row);
        }

        let pivot = a[col * n + col];
        if pivot.abs() < T::epsilon() {
            continue;
        }

        for row in (col + 1)..n {
            let factor = a[row * n + col] / pivot;
            for k in col..n {
                let val = a[col * n + k];
                a[row * n + k] -= factor * val;
            }
            let bval = b[col];
            b[row] -= factor * bval;
        }
    }

    let mut x = vec![T::zero(); n];
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in (i + 1)..n {
            sum -= a[i * n + j] * x[j];
        }
        let diag = a[i * n + i];
        x[i] = if diag.abs() > T::epsilon() {
            sum / diag
        } else {
            T::zero()
        };
    }
    x
}

/// Compute weighted R² = 1 - SS_res / SS_tot.
fn compute_weighted_r2<T: Float>(
    masks: &[bool],
    y: &[T],
    w: &[T],
    beta: &[T],
    n: usize,
    p: usize,
) -> T {
    // Weighted mean of y.
    let mut w_sum = T::zero();
    let mut wy_sum = T::zero();
    for i in 0..n {
        w_sum += w[i];
        wy_sum += w[i] * y[i];
    }
    let y_mean = if w_sum > T::epsilon() {
        wy_sum / w_sum
    } else {
        T::zero()
    };

    let mut ss_res = T::zero();
    let mut ss_tot = T::zero();

    for i in 0..n {
        // Compute prediction: beta[0] + sum(beta[j+1] * mask[j])
        let mut y_hat = beta[0];
        for j in 0..p {
            if masks[i * p + j] {
                y_hat += beta[j + 1];
            }
        }

        let residual = y[i] - y_hat;
        ss_res += w[i] * residual * residual;

        let deviation = y[i] - y_mean;
        ss_tot += w[i] * deviation * deviation;
    }

    if ss_tot > T::epsilon() {
        T::one() - ss_res / ss_tot
    } else {
        T::zero()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::Predictor;

    /// A simple model that only looks at feature 0: predict = 3 * x[0].
    struct Feature0Model;

    impl Predictor<f64> for Feature0Model {
        fn fit(&mut self, _x: &Tensor<f64>, _y: &Tensor<f64>) -> Result<()> {
            Ok(())
        }

        fn predict(&self, x: &Tensor<f64>) -> Result<Tensor<f64>> {
            let shape = x.shape();
            let n = shape[0];
            let p = shape[1];
            let data = x.as_slice();
            let mut preds = Vec::with_capacity(n);
            for i in 0..n {
                preds.push(data[i * p] * 3.0);
            }
            Tensor::from_vec(preds, vec![n]).map_err(MlError::from)
        }
    }

    #[test]
    fn test_lime_important_feature() {
        let x_train = Tensor::from_vec(
            vec![
                1.0, 0.5, 0.2, 2.0, 1.5, 0.8, 3.0, 2.5, 0.1, 4.0, 3.5, 0.9, 5.0, 4.5, 0.3,
            ],
            vec![5, 3],
        )
        .unwrap();

        let x_explain = Tensor::from_vec(vec![3.0, 2.0, 0.5], vec![1, 3]).unwrap();

        let explanations = lime(&Feature0Model, &x_train, &x_explain, 1000, None, 42).unwrap();

        assert_eq!(explanations.len(), 1);
        let exp = &explanations[0];
        assert_eq!(exp.weights.len(), 3);

        // Feature 0 should have the largest absolute weight.
        let abs0 = exp.weights[0].abs();
        let abs1 = exp.weights[1].abs();
        let abs2 = exp.weights[2].abs();
        assert!(
            abs0 > abs1 && abs0 > abs2,
            "feature 0 weight ({abs0}) should dominate, but got [{abs0}, {abs1}, {abs2}]"
        );
    }

    #[test]
    fn test_lime_explanation_count() {
        let x_train = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]).unwrap();

        let x_explain = Tensor::from_vec(vec![2.0, 3.0, 4.0, 5.0], vec![2, 2]).unwrap();

        let explanations = lime(&Feature0Model, &x_train, &x_explain, 200, None, 123).unwrap();

        assert_eq!(explanations.len(), 2);
        for exp in &explanations {
            assert_eq!(exp.weights.len(), 2);
            assert!(exp.prediction.is_finite());
            assert!(exp.intercept.is_finite());
        }
    }

    #[test]
    fn test_lime_local_fidelity() {
        let x_train = Tensor::from_vec(
            vec![1.0, 0.1, 2.0, 0.2, 3.0, 0.3, 4.0, 0.4, 5.0, 0.5],
            vec![5, 2],
        )
        .unwrap();

        let x_explain = Tensor::from_vec(vec![3.0, 0.3], vec![1, 2]).unwrap();

        let explanations = lime(&Feature0Model, &x_train, &x_explain, 500, None, 99).unwrap();

        let exp = &explanations[0];
        // R² should be positive (local model has some fit).
        assert!(exp.score > 0.0, "R² score should be > 0, got {}", exp.score);
    }

    #[test]
    fn test_lime_zero_features() {
        // x_train with 0 features (invalid).
        let x_train = Tensor::from_vec(Vec::<f64>::new(), vec![0, 0]);
        // Tensor creation with empty shape may fail or succeed — either way lime should error.
        if let Ok(xt) = x_train {
            let x_explain = Tensor::from_vec(Vec::<f64>::new(), vec![0, 0]);
            if let Ok(xe) = x_explain {
                let result = lime(&Feature0Model, &xt, &xe, 100, None, 0);
                assert!(result.is_err());
            }
        }
        // Also test with valid shape but 0 columns.
        let xt2 = Tensor::from_vec(Vec::<f64>::new(), vec![1, 0]);
        if let Ok(xt) = xt2 {
            let xe2 = Tensor::from_vec(Vec::<f64>::new(), vec![1, 0]);
            if let Ok(xe) = xe2 {
                let result = lime(&Feature0Model, &xt, &xe, 100, None, 0);
                assert!(result.is_err());
            }
        }
    }
}
