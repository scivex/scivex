//! Loss functions for training neural networks.

use scivex_core::{Float, Tensor};

use crate::error::{NnError, Result};
use crate::variable::Variable;

/// Mean squared error: `mean((pred - target)^2)`.
///
/// Both `pred` and `target` should have the same shape.
pub fn mse_loss<T: Float>(pred: &Variable<T>, target: &Variable<T>) -> Result<Variable<T>> {
    let p = pred.data();
    let t = target.data();
    if p.shape() != t.shape() {
        return Err(NnError::ShapeMismatch {
            expected: t.shape().to_vec(),
            got: p.shape().to_vec(),
        });
    }
    let n = p.numel();
    let diff = p.zip_map(&t, |a, b| a - b)?;
    let sq = diff.map(|v| v * v);
    let loss_val = sq.mean();
    let data = Tensor::from_vec(vec![loss_val], vec![1])?;

    let two = T::from_f64(2.0);
    let n_t = T::from_usize(n);
    let shape = p.shape().to_vec();

    Ok(Variable::from_op(
        data,
        vec![pred.clone(), target.clone()],
        Box::new(move |g: &Tensor<T>| {
            let g_val = g.as_slice()[0];
            let scale = two * g_val / n_t;
            let grad = diff.map(|d| d * scale);
            // grad_target = -grad_pred
            let grad_t = grad.map(|v| -v);
            vec![
                Tensor::from_vec(grad.into_vec(), shape.clone())
                    .expect("grad shape matches forward pass"),
                Tensor::from_vec(grad_t.into_vec(), shape.clone())
                    .expect("grad shape matches forward pass"),
            ]
        }),
    ))
}

/// Cross-entropy loss for classification.
///
/// `logits` has shape `[batch, classes]`, `targets` has shape `[batch]` with
/// class indices as integer-valued floats (e.g., 0.0, 1.0, 2.0).
///
/// Computes: `mean(-log_softmax(logits)[i, target_i])`.
pub fn cross_entropy_loss<T: Float>(
    logits: &Variable<T>,
    targets: &Variable<T>,
) -> Result<Variable<T>> {
    let logits_data = logits.data();
    let targets_data = targets.data();
    let shape = logits_data.shape().to_vec();
    if shape.len() != 2 {
        return Err(NnError::ShapeMismatch {
            expected: vec![0, 0],
            got: shape,
        });
    }
    let batch = shape[0];
    let classes = shape[1];
    let x = logits_data.as_slice();
    let target_indices: Vec<T> = targets_data.as_slice().to_vec();

    // Compute log-softmax.
    let mut log_sm = vec![T::zero(); batch * classes];
    let mut sm = vec![T::zero(); batch * classes];
    for r in 0..batch {
        let row_start = r * classes;
        let mut max_val = x[row_start];
        for c in 1..classes {
            if x[row_start + c] > max_val {
                max_val = x[row_start + c];
            }
        }
        let mut sum_exp = T::zero();
        for c in 0..classes {
            let e = (x[row_start + c] - max_val).exp();
            sm[row_start + c] = e;
            sum_exp += e;
        }
        let log_sum = sum_exp.ln();
        for c in 0..classes {
            sm[row_start + c] /= sum_exp;
            log_sm[row_start + c] = x[row_start + c] - max_val - log_sum;
        }
    }

    // NLL: -mean(log_softmax[i, target_i])
    let mut loss = T::zero();
    for r in 0..batch {
        let c = idx_to_usize(target_indices[r]);
        loss -= log_sm[r * classes + c];
    }
    loss /= T::from_usize(batch);

    let data = Tensor::from_vec(vec![loss], vec![1]).expect("scalar tensor");

    Ok(Variable::from_op(
        data,
        vec![logits.clone(), targets.clone()],
        Box::new(move |g: &Tensor<T>| {
            let g_val = g.as_slice()[0];
            let scale = g_val / T::from_usize(batch);
            let mut grad_logits = vec![T::zero(); batch * classes];
            for (r, &ti) in target_indices.iter().enumerate().take(batch) {
                let c = idx_to_usize(ti);
                for j in 0..classes {
                    let idx = r * classes + j;
                    // d/dx_j = softmax_j - (1 if j == target)
                    let indicator = if j == c { T::one() } else { T::zero() };
                    grad_logits[idx] = (sm[idx] - indicator) * scale;
                }
            }
            let g_logits = Tensor::from_vec(grad_logits, vec![batch, classes])
                .expect("grad shape matches forward pass");
            // Target gradients are zero (not differentiable).
            let g_targets = Tensor::zeros(vec![batch]);
            vec![g_logits, g_targets]
        }),
    ))
}

/// Binary cross-entropy loss.
///
/// `pred` and `target` have the same shape; values in [0, 1].
/// Computes: `mean(-(target * ln(pred) + (1-target) * ln(1-pred)))`.
pub fn bce_loss<T: Float>(pred: &Variable<T>, target: &Variable<T>) -> Result<Variable<T>> {
    let p = pred.data();
    let t = target.data();
    if p.shape() != t.shape() {
        return Err(NnError::ShapeMismatch {
            expected: t.shape().to_vec(),
            got: p.shape().to_vec(),
        });
    }
    let n = p.numel();
    let eps = T::from_f64(1e-7);
    let one = T::one();

    let loss_data: Vec<T> = p
        .as_slice()
        .iter()
        .zip(t.as_slice().iter())
        .map(|(&pi, &ti)| {
            let pi_c = pi.max(eps).min(one - eps);
            -(ti * pi_c.ln() + (one - ti) * (one - pi_c).ln())
        })
        .collect();
    let loss_val = loss_data.iter().copied().fold(T::zero(), |a, b| a + b) / T::from_usize(n);
    let data = Tensor::from_vec(vec![loss_val], vec![1]).expect("scalar tensor");

    let n_t = T::from_usize(n);
    let shape = p.shape().to_vec();

    Ok(Variable::from_op(
        data,
        vec![pred.clone(), target.clone()],
        Box::new(move |g: &Tensor<T>| {
            let g_val = g.as_slice()[0];
            let p_s = p.as_slice();
            let t_s = t.as_slice();
            let mut grad_p = Vec::with_capacity(n);
            for i in 0..n {
                let pi = p_s[i].max(eps).min(one - eps);
                // d/dp = (-t/p + (1-t)/(1-p)) / n
                grad_p.push((-t_s[i] / pi + (one - t_s[i]) / (one - pi)) * g_val / n_t);
            }
            let gp =
                Tensor::from_vec(grad_p, shape.clone()).expect("grad shape matches forward pass");
            let gt = Tensor::zeros(shape.clone());
            vec![gp, gt]
        }),
    ))
}

/// Helper to convert a float to usize (for target indices).
fn idx_to_usize<T: Float>(v: T) -> usize {
    // We use from_f64/from_usize round-trip isn't available,
    // so convert through display.
    // Actually, simpler: just compare against incrementing values.
    let mut i = 0usize;
    loop {
        if v < T::from_usize(i + 1) {
            return i;
        }
        i += 1;
        if i > 100_000 {
            return 0; // Safety bound
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mse_identical() {
        let a = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap(),
            true,
        );
        let b = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap(),
            false,
        );
        let loss = mse_loss(&a, &b).unwrap();
        assert!(loss.data().as_slice()[0].abs() < 1e-10);
    }

    #[test]
    fn test_mse_nonzero() {
        let a = Variable::new(Tensor::from_vec(vec![1.0, 2.0], vec![2]).unwrap(), true);
        let b = Variable::new(Tensor::from_vec(vec![3.0, 4.0], vec![2]).unwrap(), false);
        let loss = mse_loss(&a, &b).unwrap();
        // mean((1-3)^2 + (2-4)^2) = mean(4+4) = 4
        assert!((loss.data().as_slice()[0] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_mse_backward() {
        let a = Variable::new(Tensor::from_vec(vec![1.0, 2.0], vec![2]).unwrap(), true);
        let b = Variable::new(Tensor::from_vec(vec![3.0, 4.0], vec![2]).unwrap(), false);
        let loss = mse_loss(&a, &b).unwrap();
        loss.backward();
        // d/da MSE = 2*(a-b)/n = 2*(-2,-2)/2 = (-2, -2)
        let g = a.grad().unwrap();
        assert!((g.as_slice()[0] - (-2.0)).abs() < 1e-10);
        assert!((g.as_slice()[1] - (-2.0)).abs() < 1e-10);
    }

    #[test]
    fn test_cross_entropy() {
        // Perfect prediction for class 0.
        let logits = Variable::new(
            Tensor::from_vec(vec![100.0, -100.0, -100.0], vec![1, 3]).unwrap(),
            true,
        );
        let targets = Variable::new(Tensor::from_vec(vec![0.0], vec![1]).unwrap(), false);
        let loss = cross_entropy_loss(&logits, &targets).unwrap();
        // Should be very close to 0.
        assert!(loss.data().as_slice()[0] < 1e-5);
    }

    #[test]
    fn test_bce_loss() {
        let pred = Variable::new(Tensor::from_vec(vec![0.9, 0.1], vec![2]).unwrap(), true);
        let target = Variable::new(Tensor::from_vec(vec![1.0, 0.0], vec![2]).unwrap(), false);
        let loss = bce_loss(&pred, &target).unwrap();
        // Both predictions are good, loss should be small.
        assert!(loss.data().as_slice()[0] < 0.2);
    }

    #[test]
    fn test_mse_shape_mismatch_error() {
        let a = Variable::new(Tensor::from_vec(vec![1.0, 2.0], vec![2]).unwrap(), true);
        let b = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap(),
            false,
        );
        let result = mse_loss(&a, &b);
        assert!(result.is_err());
        match result.unwrap_err() {
            NnError::ShapeMismatch { expected, got } => {
                assert_eq!(expected, vec![3]);
                assert_eq!(got, vec![2]);
            }
            other => panic!("expected ShapeMismatch, got {other:?}"),
        }
    }

    #[test]
    fn test_bce_shape_mismatch_error() {
        let a = Variable::new(Tensor::from_vec(vec![0.5], vec![1]).unwrap(), true);
        let b = Variable::new(Tensor::from_vec(vec![1.0, 0.0], vec![2]).unwrap(), false);
        let result = bce_loss(&a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn test_cross_entropy_requires_2d() {
        // 1-D logits should fail
        let logits = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap(),
            true,
        );
        let targets = Variable::new(Tensor::from_vec(vec![0.0], vec![1]).unwrap(), false);
        let result = cross_entropy_loss(&logits, &targets);
        assert!(result.is_err());
    }

    #[test]
    fn test_cross_entropy_backward_gradient_shape() {
        let logits = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 1.0, 1.0, 1.0], vec![2, 3]).unwrap(),
            true,
        );
        let targets = Variable::new(Tensor::from_vec(vec![0.0, 2.0], vec![2]).unwrap(), false);
        let loss = cross_entropy_loss(&logits, &targets).unwrap();
        loss.backward();
        let g = logits.grad().unwrap();
        assert_eq!(g.shape(), &[2, 3]);
    }
}
