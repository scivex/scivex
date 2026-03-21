//! Gradient clipping utilities.

use scivex_core::{Float, Tensor};

use crate::variable::Variable;

/// Clip gradients by maximum L2 norm.
///
/// If the total L2 norm of all parameter gradients exceeds `max_norm`, every
/// gradient is scaled down proportionally so the combined norm equals `max_norm`.
///
/// Returns the original (unclipped) total norm.
///
/// # Examples
///
/// ```
/// # use scivex_nn::variable::Variable;
/// # use scivex_nn::training::clip_grad_norm;
/// # use scivex_core::Tensor;
/// let p = Variable::new(Tensor::from_vec(vec![1.0_f64, 0.0], vec![2]).unwrap(), true);
/// p.set_grad(Tensor::from_vec(vec![3.0_f64, 4.0], vec![2]).unwrap());
/// let norm = clip_grad_norm(&[p], 1.0_f64);
/// assert!((norm - 5.0).abs() < 1e-10);
/// ```
pub fn clip_grad_norm<T: Float>(parameters: &[Variable<T>], max_norm: T) -> T {
    // Compute total L2 norm across all parameter gradients.
    let mut total_norm_sq = T::zero();
    for p in parameters {
        if let Some(g) = p.grad() {
            for &val in g.as_slice() {
                total_norm_sq += val * val;
            }
        }
    }
    let total_norm = total_norm_sq.sqrt();

    if total_norm > max_norm {
        let scale = max_norm / total_norm;
        for p in parameters {
            if let Some(g) = p.grad() {
                let clipped = &g * scale;
                p.zero_grad();
                p.acc_grad(&clipped);
            }
        }
    }

    total_norm
}

/// Clip gradients by value (element-wise clamping).
///
/// Every gradient element is clamped to the range `[-clip_value, clip_value]`.
///
/// # Examples
///
/// ```
/// # use scivex_nn::variable::Variable;
/// # use scivex_nn::training::clip_grad_value;
/// # use scivex_core::Tensor;
/// let p = Variable::new(Tensor::from_vec(vec![0.0_f64], vec![1]).unwrap(), true);
/// p.set_grad(Tensor::from_vec(vec![10.0_f64], vec![1]).unwrap());
/// clip_grad_value(&[p.clone()], 1.0_f64);
/// assert!((p.grad().unwrap().as_slice()[0] - 1.0).abs() < 1e-10);
/// ```
pub fn clip_grad_value<T: Float>(parameters: &[Variable<T>], clip_value: T) {
    let neg_clip = T::zero() - clip_value;
    for p in parameters {
        if let Some(g) = p.grad() {
            let data: Vec<T> = g
                .as_slice()
                .iter()
                .map(|&v| {
                    if v > clip_value {
                        clip_value
                    } else if v < neg_clip {
                        neg_clip
                    } else {
                        v
                    }
                })
                .collect();
            let clipped =
                Tensor::from_vec(data, g.shape().to_vec()).expect("shape unchanged; cannot fail");
            p.zero_grad();
            p.acc_grad(&clipped);
        }
    }
}
