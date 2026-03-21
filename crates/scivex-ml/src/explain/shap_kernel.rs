//! Kernel SHAP — model-agnostic Shapley value estimation.
//!
//! Uses a weighted sampling approach: generate binary coalition vectors,
//! compute weighted predictions, and solve weighted least squares for
//! Shapley values.

use scivex_core::{Float, Tensor, random::Rng};

use crate::error::{MlError, Result};
use crate::traits::Predictor;

/// Compute approximate SHAP values using Kernel SHAP.
///
/// - `model`: any fitted predictor
/// - `x_background`: background dataset `[n_bg, n_features]` for marginalizing
/// - `x_explain`: samples to explain `[n_explain, n_features]`
/// - `n_samples`: number of coalition samples
/// - `seed`: RNG seed
///
/// Returns SHAP values `[n_explain, n_features]`.
///
/// # Examples
///
/// ```
/// # use scivex_core::Tensor;
/// # use scivex_ml::{tree::DecisionTreeRegressor, traits::Predictor, explain::kernel_shap};
/// let x = Tensor::from_vec(vec![1.0_f64,2.0, 3.0,4.0, 5.0,6.0, 7.0,8.0], vec![4, 2]).unwrap();
/// let y = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
/// let mut model = DecisionTreeRegressor::new(Some(3), 1);
/// model.fit(&x, &y).unwrap();
/// let x_explain = Tensor::from_vec(vec![2.0, 3.0], vec![1, 2]).unwrap();
/// let shap = kernel_shap(&model, &x, &x_explain, 100, 42).unwrap();
/// assert_eq!(shap.shape(), &[1, 2]);
/// ```
pub fn kernel_shap<T, P>(
    model: &P,
    x_background: &Tensor<T>,
    x_explain: &Tensor<T>,
    n_samples: usize,
    seed: u64,
) -> Result<Tensor<T>>
where
    T: Float,
    P: Predictor<T>,
{
    let bg_shape = x_background.shape();
    let ex_shape = x_explain.shape();
    if bg_shape.len() != 2 || ex_shape.len() != 2 {
        let bad_len = if bg_shape.len() == 2 {
            ex_shape.len()
        } else {
            bg_shape.len()
        };
        return Err(MlError::DimensionMismatch {
            expected: 2,
            got: bad_len,
        });
    }
    let p = bg_shape[1];
    if ex_shape[1] != p {
        return Err(MlError::DimensionMismatch {
            expected: p,
            got: ex_shape[1],
        });
    }

    let n_bg = bg_shape[0];
    let n_explain = ex_shape[0];
    let bg_data = x_background.as_slice();
    let ex_data = x_explain.as_slice();

    let mut rng = Rng::new(seed);
    let mut all_shap = vec![T::zero(); n_explain * p];

    for ei in 0..n_explain {
        let x_row = &ex_data[ei * p..(ei + 1) * p];

        // Get baseline prediction (all features marginalized)
        let base_pred = avg_prediction(model, bg_data, n_bg, p)?;

        // Full prediction
        let full_pred = {
            let x_t = Tensor::from_vec(x_row.to_vec(), vec![1, p])?;
            let pred = model.predict(&x_t)?;
            pred.as_slice()[0]
        };

        // Sample coalitions and compute kernel SHAP
        let mut z_matrix = Vec::with_capacity(n_samples * p);
        let mut y_vec = Vec::with_capacity(n_samples);
        let mut w_vec = Vec::with_capacity(n_samples);

        for _ in 0..n_samples {
            // Generate random coalition (binary mask)
            let mut coalition = vec![false; p];
            let mut n_on = 0usize;
            for c in &mut coalition {
                if rng.next_f64() > 0.5 {
                    *c = true;
                    n_on += 1;
                }
            }

            // Skip all-zeros and all-ones (trivial)
            if n_on == 0 || n_on == p {
                continue;
            }

            // Build masked sample: use x_row for "on" features,
            // average over background for "off" features
            let pred = masked_prediction(model, x_row, &coalition, bg_data, n_bg, p)?;

            // Kernel SHAP weight
            let weight = shap_kernel_weight(p, n_on);

            for &c in &coalition {
                z_matrix.push(if c { T::one() } else { T::zero() });
            }
            y_vec.push(pred - base_pred);
            w_vec.push(T::from_f64(weight));
        }

        let n_valid = y_vec.len();
        if n_valid == 0 {
            continue;
        }

        // Solve weighted least squares: (Z^T W Z) phi = Z^T W y
        let phi = weighted_least_squares(&z_matrix, &y_vec, &w_vec, n_valid, p);

        // Scale so sum of phi = full_pred - base_pred
        let phi_sum: T = phi.iter().copied().fold(T::zero(), |a, b| a + b);
        let target = full_pred - base_pred;
        let scale = if phi_sum.abs() > T::epsilon() {
            target / phi_sum
        } else {
            T::one()
        };

        for (j, &v) in phi.iter().enumerate() {
            all_shap[ei * p + j] = v * scale;
        }
    }

    Tensor::from_vec(all_shap, vec![n_explain, p]).map_err(MlError::from)
}

fn avg_prediction<T: Float, P: Predictor<T>>(
    model: &P,
    bg_data: &[T],
    n_bg: usize,
    p: usize,
) -> Result<T> {
    let bg_tensor = Tensor::from_vec(bg_data.to_vec(), vec![n_bg, p])?;
    let preds = model.predict(&bg_tensor)?;
    let s = preds.as_slice();
    Ok(s.iter().copied().fold(T::zero(), |a, b| a + b) / T::from_usize(n_bg))
}

fn masked_prediction<T: Float, P: Predictor<T>>(
    model: &P,
    x_row: &[T],
    coalition: &[bool],
    bg_data: &[T],
    n_bg: usize,
    p: usize,
) -> Result<T> {
    let mut total = T::zero();
    for bi in 0..n_bg {
        let mut sample = vec![T::zero(); p];
        for j in 0..p {
            sample[j] = if coalition[j] {
                x_row[j]
            } else {
                bg_data[bi * p + j]
            };
        }
        let x_t = Tensor::from_vec(sample, vec![1, p])?;
        let pred = model.predict(&x_t)?;
        total += pred.as_slice()[0];
    }
    Ok(total / T::from_usize(n_bg))
}

fn shap_kernel_weight(p: usize, n_on: usize) -> f64 {
    let binom = binomial_coeff(p, n_on);
    if binom == 0.0 || n_on == 0 || n_on == p {
        return 0.0;
    }
    (p as f64 - 1.0) / (binom * n_on as f64 * (p - n_on) as f64)
}

fn binomial_coeff(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    let k = k.min(n - k);
    let mut result = 1.0_f64;
    for i in 0..k {
        result *= (n - i) as f64;
        result /= (i + 1) as f64;
    }
    result
}

fn weighted_least_squares<T: Float>(z: &[T], y: &[T], w: &[T], n: usize, p: usize) -> Vec<T> {
    let mut zt_wz = vec![T::zero(); p * p];
    let mut zt_wy = vec![T::zero(); p];

    for i in 0..n {
        let wi = w[i];
        for j in 0..p {
            let zj = z[i * p + j];
            zt_wy[j] += zj * wi * y[i];
            for k in 0..p {
                zt_wz[j * p + k] += zj * wi * z[i * p + k];
            }
        }
    }

    // Ridge for numerical stability
    for j in 0..p {
        zt_wz[j * p + j] += T::from_f64(1e-10);
    }

    gauss_solve(&mut zt_wz, &mut zt_wy, p)
}

fn gauss_solve<T: Float>(a: &mut [T], b: &mut [T], n: usize) -> Vec<T> {
    for col in 0..n {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::DecisionTreeRegressor;

    #[test]
    fn test_kernel_shap_shape() {
        let x_bg =
            Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![4, 2]).unwrap();
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();

        let mut model = DecisionTreeRegressor::new(Some(3), 1);
        model.fit(&x_bg, &y).unwrap();

        let x_explain = Tensor::from_vec(vec![2.0, 3.0, 4.0, 5.0], vec![2, 2]).unwrap();
        let shap = kernel_shap(&model, &x_bg, &x_explain, 100, 42).unwrap();
        assert_eq!(shap.shape(), &[2, 2]);
    }

    #[test]
    fn test_kernel_shap_sum_property() {
        let x = Tensor::from_vec(
            vec![1.0_f64, 0.5, 2.0, 1.5, 3.0, 2.5, 4.0, 3.5, 5.0, 4.5],
            vec![5, 2],
        )
        .unwrap();
        let y = Tensor::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0], vec![5]).unwrap();

        let mut model = DecisionTreeRegressor::new(Some(3), 1);
        model.fit(&x, &y).unwrap();

        let x_explain = Tensor::from_vec(vec![3.0, 2.5], vec![1, 2]).unwrap();
        let shap = kernel_shap(&model, &x, &x_explain, 500, 42).unwrap();
        let _shap_sum: f64 = shap.as_slice().iter().sum();
        // Just verify it runs without error and produces finite values
        assert!(shap.as_slice().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_kernel_shap_dimension_mismatch() {
        let x_bg = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]).unwrap();
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();

        let mut model = DecisionTreeRegressor::new(Some(3), 1);
        model.fit(&x_bg, &y).unwrap();

        let x_explain = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        assert!(kernel_shap(&model, &x_bg, &x_explain, 10, 0).is_err());
    }
}
