//! GPU loss functions for training neural networks.
//!
//! All loss functions operate on [`GpuVariable`]s and produce scalar outputs
//! suitable for calling [`backward()`](super::variable::GpuVariable::backward).

use super::ops;
use super::variable::GpuVariable;
use crate::error::Result;

/// GPU Mean Squared Error loss: `mean((pred - target)^2)`.
///
/// Both `pred` and `target` must have the same shape.
/// Returns a scalar (shape `[1]`) variable.
pub fn gpu_mse_loss(pred: &GpuVariable, target: &GpuVariable) -> Result<GpuVariable> {
    let diff = ops::gpu_sub(pred, target);
    let sq = ops::gpu_mul(&diff, &diff);
    Ok(ops::gpu_mean(&sq))
}

/// GPU Cross-entropy loss (simplified).
///
/// `pred` should be log-probabilities (e.g. output of log-softmax) and `target`
/// should be one-hot encoded labels. Computes: `-mean(target * pred)`.
///
/// Returns a scalar (shape `[1]`) variable.
pub fn gpu_cross_entropy_loss(pred: &GpuVariable, target: &GpuVariable) -> Result<GpuVariable> {
    let prod = ops::gpu_mul(target, pred);
    let neg_prod = ops::gpu_neg(&prod);
    Ok(ops::gpu_mean(&neg_prod))
}
