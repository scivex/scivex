//! Loss functions for training neural networks.

use scivex_core::{Float, Tensor};

use crate::error::{NnError, Result};
use crate::variable::Variable;

/// Mean squared error: `mean((pred - target)^2)`.
///
/// Both `pred` and `target` should have the same shape.
///
/// # Examples
///
/// ```
/// # use scivex_core::Tensor;
/// # use scivex_nn::variable::Variable;
/// # use scivex_nn::loss::mse_loss;
/// let pred = Variable::new(Tensor::from_vec(vec![1.0_f64, 2.0, 3.0], vec![3]).unwrap(), false);
/// let target = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap(), false);
/// let loss = mse_loss(&pred, &target).unwrap();
/// assert!(loss.data().as_slice()[0].abs() < 1e-12); // perfect match → 0 loss
/// ```
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
    let diff = &p - &t;
    let sq = &diff * &diff;
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
            let grad = &diff * scale;
            // grad_target = -grad_pred
            let grad_t = -&grad;
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
///
/// # Examples
///
/// ```
/// # use scivex_core::Tensor;
/// # use scivex_nn::variable::Variable;
/// # use scivex_nn::loss::cross_entropy_loss;
/// let logits = Variable::new(
///     Tensor::from_vec(vec![100.0_f64, -100.0, -100.0], vec![1, 3]).unwrap(), false,
/// );
/// let targets = Variable::new(Tensor::from_vec(vec![0.0_f64], vec![1]).unwrap(), false);
/// let loss = cross_entropy_loss(&logits, &targets).unwrap();
/// assert!(loss.data().as_slice()[0] < 1e-5); // confident correct prediction
/// ```
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
///
/// # Examples
///
/// ```
/// # use scivex_core::Tensor;
/// # use scivex_nn::variable::Variable;
/// # use scivex_nn::loss::bce_loss;
/// let pred = Variable::new(Tensor::from_vec(vec![0.9_f64, 0.1], vec![2]).unwrap(), false);
/// let target = Variable::new(Tensor::from_vec(vec![1.0_f64, 0.0], vec![2]).unwrap(), false);
/// let loss = bce_loss(&pred, &target).unwrap();
/// assert!(loss.data().as_slice()[0] < 0.2); // good predictions → low loss
/// ```
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

/// Huber (smooth L1) loss.
///
/// For each element, computes:
/// - `0.5 * d^2` if `|d| <= delta`
/// - `delta * (|d| - 0.5 * delta)` otherwise
///
/// where `d = pred - target`. Returns the mean over all elements.
///
/// # Examples
///
/// ```
/// # use scivex_core::Tensor;
/// # use scivex_nn::variable::Variable;
/// # use scivex_nn::loss::huber_loss;
/// let pred = Variable::new(Tensor::from_vec(vec![1.0_f64, 2.0], vec![2]).unwrap(), false);
/// let target = Variable::new(Tensor::from_vec(vec![1.1_f64, 2.1], vec![2]).unwrap(), false);
/// let loss = huber_loss(&pred, &target, 1.0).unwrap();
/// assert!(loss.data().as_slice()[0] < 0.01); // small error → small loss
/// ```
pub fn huber_loss<T: Float>(
    pred: &Variable<T>,
    target: &Variable<T>,
    delta: T,
) -> Result<Variable<T>> {
    let p = pred.data();
    let t = target.data();
    if p.shape() != t.shape() {
        return Err(NnError::ShapeMismatch {
            expected: t.shape().to_vec(),
            got: p.shape().to_vec(),
        });
    }
    let n = p.numel();
    let n_t = T::from_usize(n);
    let half = T::from_f64(0.5);

    let diff = &p - &t;

    let loss_val = diff
        .as_slice()
        .iter()
        .map(|&d| {
            if d.abs() <= delta {
                half * d * d
            } else {
                delta * (d.abs() - half * delta)
            }
        })
        .fold(T::zero(), |a, b| a + b)
        / n_t;

    let data = Tensor::from_vec(vec![loss_val], vec![1])?;
    let shape = p.shape().to_vec();

    Ok(Variable::from_op(
        data,
        vec![pred.clone(), target.clone()],
        Box::new(move |g: &Tensor<T>| {
            let g_val = g.as_slice()[0];
            let grad_p: Vec<T> = diff
                .as_slice()
                .iter()
                .map(|&d| {
                    let raw = if d.abs() <= delta {
                        d
                    } else if d > T::zero() {
                        delta
                    } else {
                        -delta
                    };
                    raw * g_val / n_t
                })
                .collect();
            let gp =
                Tensor::from_vec(grad_p, shape.clone()).expect("grad shape matches forward pass");
            let gt = Tensor::zeros(shape.clone());
            vec![gp, gt]
        }),
    ))
}

/// Focal loss for binary classification.
///
/// `logits` and `targets` have the same shape; targets contain 0 or 1 values.
///
/// Computes: `-mean(alpha_t * (1 - p_t)^gamma * ln(p_t + eps))`
/// where `p = sigmoid(logits)`, `p_t = p * t + (1 - p) * (1 - t)`,
/// and `alpha_t = alpha * t + (1 - alpha) * (1 - t)`.
///
/// # Examples
///
/// ```
/// # use scivex_core::Tensor;
/// # use scivex_nn::variable::Variable;
/// # use scivex_nn::loss::focal_loss;
/// let logits = Variable::new(Tensor::from_vec(vec![10.0_f64], vec![1]).unwrap(), false);
/// let targets = Variable::new(Tensor::from_vec(vec![1.0_f64], vec![1]).unwrap(), false);
/// let loss = focal_loss(&logits, &targets, 2.0, 0.25).unwrap();
/// assert!(loss.data().as_slice()[0] < 1e-5); // confident correct → near zero
/// ```
pub fn focal_loss<T: Float>(
    logits: &Variable<T>,
    targets: &Variable<T>,
    gamma: T,
    alpha: T,
) -> Result<Variable<T>> {
    let x = logits.data();
    let t = targets.data();
    if x.shape() != t.shape() {
        return Err(NnError::ShapeMismatch {
            expected: t.shape().to_vec(),
            got: x.shape().to_vec(),
        });
    }
    let n = x.numel();
    let n_t = T::from_usize(n);
    let one = T::one();
    let eps = T::from_f64(1e-7);

    // Numerically stable sigmoid.
    let sigmoid: Vec<T> = x
        .as_slice()
        .iter()
        .map(|&v| {
            if v >= T::zero() {
                one / (one + (-v).exp())
            } else {
                let ev = v.exp();
                ev / (one + ev)
            }
        })
        .collect();

    let t_v: Vec<T> = t.as_slice().to_vec();
    let mut loss = T::zero();
    for i in 0..n {
        let p = sigmoid[i];
        let ti = t_v[i];
        let pt = p * ti + (one - p) * (one - ti);
        let at = alpha * ti + (one - alpha) * (one - ti);
        loss += at * (one - pt).max(T::zero()).powf(gamma) * (pt + eps).ln();
    }
    loss = -loss / n_t;

    let data = Tensor::from_vec(vec![loss], vec![1])?;
    let shape = x.shape().to_vec();

    Ok(Variable::from_op(
        data,
        vec![logits.clone(), targets.clone()],
        Box::new(move |g: &Tensor<T>| {
            let g_val = g.as_slice()[0];
            let mut grad_x = Vec::with_capacity(n);
            for i in 0..n {
                let p = sigmoid[i];
                let ti = t_v[i];
                let pt = p * ti + (one - p) * (one - ti);
                let at = alpha * ti + (one - alpha) * (one - ti);
                // sign: dp_t / dp = 2*t - 1
                let dpt_dp = ti + ti - one;
                // dp/dx = p * (1 - p)  (sigmoid derivative)
                let dp_dx = p * (one - p);
                // d/dx [ -at * (1-pt)^gamma * ln(pt+eps) ]
                // = -at * [ -gamma * (1-pt)^(gamma-1) * dpt_dp * dp_dx * ln(pt+eps)
                //           + (1-pt)^gamma * dpt_dp * dp_dx / (pt+eps) ]
                let one_minus_pt = (one - pt).max(T::zero());
                let pow_g = one_minus_pt.powf(gamma);
                let pow_gm1 = if gamma > T::zero() {
                    one_minus_pt.powf(gamma - one)
                } else {
                    T::zero()
                };
                let term1 = -gamma * pow_gm1 * dpt_dp * dp_dx * (pt + eps).ln();
                let term2 = pow_g * dpt_dp * dp_dx / (pt + eps);
                grad_x.push(-at * (term1 + term2) * g_val / n_t);
            }
            let gx =
                Tensor::from_vec(grad_x, shape.clone()).expect("grad shape matches forward pass");
            let gt = Tensor::zeros(shape.clone());
            vec![gx, gt]
        }),
    ))
}

/// KL divergence: `KL(P || Q) = mean(exp(log_p) * (log_p - log_q))`.
///
/// Both inputs are log-probabilities with the same shape.
///
/// # Examples
///
/// ```
/// # use scivex_core::Tensor;
/// # use scivex_nn::variable::Variable;
/// # use scivex_nn::loss::kl_divergence;
/// let log_p = Variable::new(Tensor::from_vec(vec![-1.0_f64, -2.0], vec![2]).unwrap(), false);
/// let log_q = Variable::new(Tensor::from_vec(vec![-1.0_f64, -2.0], vec![2]).unwrap(), false);
/// let loss = kl_divergence(&log_p, &log_q).unwrap();
/// assert!(loss.data().as_slice()[0].abs() < 1e-10); // KL(P || P) = 0
/// ```
pub fn kl_divergence<T: Float>(log_p: &Variable<T>, log_q: &Variable<T>) -> Result<Variable<T>> {
    let lp = log_p.data();
    let lq = log_q.data();
    if lp.shape() != lq.shape() {
        return Err(NnError::ShapeMismatch {
            expected: lq.shape().to_vec(),
            got: lp.shape().to_vec(),
        });
    }
    let n = lp.numel();
    let n_t = T::from_usize(n);
    let one = T::one();

    let lp_v: Vec<T> = lp.as_slice().to_vec();
    let lq_v: Vec<T> = lq.as_slice().to_vec();
    let mut loss = T::zero();
    for i in 0..n {
        loss += lp_v[i].exp() * (lp_v[i] - lq_v[i]);
    }
    loss /= n_t;

    let data = Tensor::from_vec(vec![loss], vec![1])?;
    let shape = lp.shape().to_vec();

    Ok(Variable::from_op(
        data,
        vec![log_p.clone(), log_q.clone()],
        Box::new(move |g: &Tensor<T>| {
            let g_val = g.as_slice()[0];
            let mut grad_lp = Vec::with_capacity(n);
            let mut grad_lq = Vec::with_capacity(n);
            for i in 0..n {
                let exp_lp = lp_v[i].exp();
                // d/d(log_p) = exp(log_p) * (log_p - log_q + 1) / n
                grad_lp.push(exp_lp * (lp_v[i] - lq_v[i] + one) * g_val / n_t);
                // d/d(log_q) = -exp(log_p) / n
                grad_lq.push(-exp_lp * g_val / n_t);
            }
            let glp =
                Tensor::from_vec(grad_lp, shape.clone()).expect("grad shape matches forward pass");
            let glq =
                Tensor::from_vec(grad_lq, shape.clone()).expect("grad shape matches forward pass");
            vec![glp, glq]
        }),
    ))
}

/// Hinge loss for binary classification.
///
/// `target` values should be -1 or +1.
/// Computes: `mean(max(0, 1 - target * pred))`.
///
/// # Examples
///
/// ```
/// # use scivex_core::Tensor;
/// # use scivex_nn::variable::Variable;
/// # use scivex_nn::loss::hinge_loss;
/// let pred = Variable::new(Tensor::from_vec(vec![2.0_f64, -2.0], vec![2]).unwrap(), false);
/// let target = Variable::new(Tensor::from_vec(vec![1.0_f64, -1.0], vec![2]).unwrap(), false);
/// let loss = hinge_loss(&pred, &target).unwrap();
/// assert!(loss.data().as_slice()[0].abs() < 1e-10); // correct margin → 0 loss
/// ```
pub fn hinge_loss<T: Float>(pred: &Variable<T>, target: &Variable<T>) -> Result<Variable<T>> {
    let p = pred.data();
    let t = target.data();
    if p.shape() != t.shape() {
        return Err(NnError::ShapeMismatch {
            expected: t.shape().to_vec(),
            got: p.shape().to_vec(),
        });
    }
    let n = p.numel();
    let n_t = T::from_usize(n);
    let one = T::one();
    let zero = T::zero();

    let p_v: Vec<T> = p.as_slice().to_vec();
    let t_v: Vec<T> = t.as_slice().to_vec();
    let mut loss = T::zero();
    for i in 0..n {
        let margin = one - t_v[i] * p_v[i];
        if margin > zero {
            loss += margin;
        }
    }
    loss /= n_t;

    let data = Tensor::from_vec(vec![loss], vec![1])?;
    let shape = p.shape().to_vec();

    Ok(Variable::from_op(
        data,
        vec![pred.clone(), target.clone()],
        Box::new(move |g: &Tensor<T>| {
            let g_val = g.as_slice()[0];
            let mut grad_p = Vec::with_capacity(n);
            for i in 0..n {
                let margin = one - t_v[i] * p_v[i];
                if margin > zero {
                    grad_p.push(-t_v[i] * g_val / n_t);
                } else {
                    grad_p.push(T::zero());
                }
            }
            let gp =
                Tensor::from_vec(grad_p, shape.clone()).expect("grad shape matches forward pass");
            let gt = Tensor::zeros(shape.clone());
            vec![gp, gt]
        }),
    ))
}

/// Smooth L1 loss parameterized by `beta`.
///
/// For each element `d = pred - target`:
/// - `0.5 * d^2 / beta` if `|d| < beta`
/// - `|d| - 0.5 * beta` otherwise
///
/// Returns the mean over all elements.
///
/// # Examples
///
/// ```
/// # use scivex_core::Tensor;
/// # use scivex_nn::variable::Variable;
/// # use scivex_nn::loss::smooth_l1_loss;
/// let pred = Variable::new(Tensor::from_vec(vec![1.0_f64, 2.0], vec![2]).unwrap(), false);
/// let target = Variable::new(Tensor::from_vec(vec![1.0_f64, 2.0], vec![2]).unwrap(), false);
/// let loss = smooth_l1_loss(&pred, &target, 1.0).unwrap();
/// assert!(loss.data().as_slice()[0].abs() < 1e-12); // identical → 0
/// ```
pub fn smooth_l1_loss<T: Float>(
    pred: &Variable<T>,
    target: &Variable<T>,
    beta: T,
) -> Result<Variable<T>> {
    let p = pred.data();
    let t = target.data();
    if p.shape() != t.shape() {
        return Err(NnError::ShapeMismatch {
            expected: t.shape().to_vec(),
            got: p.shape().to_vec(),
        });
    }
    let n = p.numel();
    let n_t = T::from_usize(n);
    let half = T::from_f64(0.5);

    let diff = &p - &t;

    let loss_val = diff
        .as_slice()
        .iter()
        .map(|&d| {
            if d.abs() < beta {
                half * d * d / beta
            } else {
                d.abs() - half * beta
            }
        })
        .fold(T::zero(), |a, b| a + b)
        / n_t;

    let data = Tensor::from_vec(vec![loss_val], vec![1])?;
    let shape = p.shape().to_vec();

    Ok(Variable::from_op(
        data,
        vec![pred.clone(), target.clone()],
        Box::new(move |g: &Tensor<T>| {
            let g_val = g.as_slice()[0];
            let grad_p: Vec<T> = diff
                .as_slice()
                .iter()
                .map(|&d| {
                    let raw = if d.abs() < beta {
                        d / beta
                    } else if d > T::zero() {
                        T::one()
                    } else {
                        -T::one()
                    };
                    raw * g_val / n_t
                })
                .collect();
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

    #[test]
    fn test_huber_loss_quadratic() {
        // Small differences (|d| <= delta) should give quadratic behavior: 0.5 * d^2.
        let pred = Variable::new(Tensor::from_vec(vec![1.0, 2.0], vec![2]).unwrap(), true);
        let target = Variable::new(Tensor::from_vec(vec![1.1, 2.1], vec![2]).unwrap(), false);
        let loss = huber_loss(&pred, &target, 1.0).unwrap();
        // d = -0.1 for both, 0.5 * 0.01 = 0.005 each, mean = 0.005
        assert!((loss.data().as_slice()[0] - 0.005).abs() < 1e-10);
    }

    #[test]
    fn test_huber_loss_linear() {
        // Large differences (|d| > delta) should give linear behavior.
        let pred = Variable::new(Tensor::from_vec(vec![0.0], vec![1]).unwrap(), true);
        let target = Variable::new(Tensor::from_vec(vec![5.0], vec![1]).unwrap(), false);
        let delta = 1.0;
        let loss = huber_loss(&pred, &target, delta).unwrap();
        // |d| = 5, delta*(|d| - 0.5*delta) = 1*(5 - 0.5) = 4.5
        assert!((loss.data().as_slice()[0] - 4.5).abs() < 1e-10);
    }

    #[test]
    fn test_focal_loss_easy_example() {
        // High logit for target=1 → sigmoid near 1 → pt near 1 → loss near 0.
        let logits = Variable::new(Tensor::from_vec(vec![10.0], vec![1]).unwrap(), true);
        let targets = Variable::new(Tensor::from_vec(vec![1.0], vec![1]).unwrap(), false);
        let loss = focal_loss(&logits, &targets, 2.0, 0.25).unwrap();
        assert!(loss.data().as_slice()[0] < 1e-5);
    }

    #[test]
    fn test_kl_divergence_same_distribution() {
        // KL(P || P) = mean(exp(log_p) * 0) = 0.
        let log_p = Variable::new(
            Tensor::from_vec(vec![-1.0_f64, -2.0, -0.5], vec![3]).unwrap(),
            true,
        );
        let log_q = Variable::new(
            Tensor::from_vec(vec![-1.0_f64, -2.0, -0.5], vec![3]).unwrap(),
            true,
        );
        let loss = kl_divergence(&log_p, &log_q).unwrap();
        assert!(loss.data().as_slice()[0].abs() < 1e-10);
    }

    #[test]
    fn test_hinge_loss_correct_margin() {
        // target * pred > 1 → zero loss.
        let pred = Variable::new(Tensor::from_vec(vec![2.0, -2.0], vec![2]).unwrap(), true);
        let target = Variable::new(Tensor::from_vec(vec![1.0, -1.0], vec![2]).unwrap(), false);
        let loss = hinge_loss(&pred, &target).unwrap();
        // 1 - 1*2 = -1 (clamped to 0), 1 - (-1)*(-2) = -1 (clamped to 0)
        assert!(loss.data().as_slice()[0].abs() < 1e-10);
    }

    #[test]
    fn test_smooth_l1_loss_backward() {
        let pred = Variable::new(
            Tensor::from_vec(vec![1.0, 5.0, 3.0], vec![3]).unwrap(),
            true,
        );
        let target = Variable::new(
            Tensor::from_vec(vec![1.5, 0.0, 3.0], vec![3]).unwrap(),
            false,
        );
        let loss = smooth_l1_loss(&pred, &target, 1.0).unwrap();
        loss.backward();
        let g = pred.grad().unwrap();
        assert_eq!(g.shape(), &[3]);
        // First element: |d|=0.5 < beta=1, grad = d/beta/n = -0.5/1/3
        assert!((g.as_slice()[0] - (-0.5 / 3.0)).abs() < 1e-10);
        // Second element: |d|=5 >= beta=1, grad = sign(d)/n = 1/3
        assert!((g.as_slice()[1] - (1.0 / 3.0)).abs() < 1e-10);
        // Third element: d=0 < beta=1, grad = 0/beta/n = 0
        assert!(g.as_slice()[2].abs() < 1e-10);
    }
}
