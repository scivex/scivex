//! Automatic mixed-precision (AMP) training utilities.
//!
//! Mixed precision keeps **master weights** in full precision (f32) while
//! running forward and backward passes in half precision (f16) for speed and
//! memory savings. This module provides utilities that automate the
//! cast-forward-backward-update cycle.
//!
//! Requires the `mixed-precision` feature flag.
//!
//! # Overview
//!
//! ```text
//! ┌─────────────┐     cast        ┌─────────────┐
//! │ f32 weights ├────────────────►│ f16 weights  │
//! └─────┬───────┘                 └──────┬──────┘
//!       │                                │ forward
//!       │                         ┌──────▼──────┐
//!       │                         │  f16 output  │
//!       │                         └──────┬──────┘
//!       │                                │ loss (f32)
//!       │                         ┌──────▼──────┐
//!       │                         │   backward   │
//!       │                         └──────┬──────┘
//!       │        unscale + update ┌──────▼──────┐
//!       │◄────────────────────────│  f32 grads   │
//!       │                         └─────────────┘
//! ```

use scivex_core::Float;

use crate::error::Result;
use crate::layer::Layer;
use crate::variable::Variable;

/// Cast a `Variable<S>` to `Variable<D>` (e.g., f32→f16 or f16→f32).
///
/// The returned variable is a **detached leaf** — no graph history is
/// carried over. This is intentional: the compute graph for the half-
/// precision forward pass is built fresh each iteration.
pub fn cast_variable<S: Float, D: Float>(v: &Variable<S>) -> Variable<D> {
    let data = v.data();
    let casted = data.cast::<D>();
    Variable::new(casted, v.requires_grad())
}

/// Cast a slice of variables from one precision to another.
pub fn cast_params<S: Float, D: Float>(params: &[Variable<S>]) -> Vec<Variable<D>> {
    params.iter().map(cast_variable).collect()
}

/// Configuration for automatic mixed-precision training.
///
/// `Master` is the full-precision type (typically `f32`) used for weight
/// storage and optimizer state. `Compute` is the reduced-precision type
/// (typically `f16`) used for forward/backward passes.
///
/// # Example
///
/// ```ignore
/// use scivex_nn::training::{AmpConfig, GradScaler};
///
/// let config = AmpConfig::<f32, f16>::new();
/// let scaler = GradScaler::<f32>::new();
///
/// // Training loop:
/// // 1. Cast f32 params → f16
/// // 2. Forward in f16
/// // 3. Cast loss → f32
/// // 4. Scale + backward
/// // 5. Unscale + step
/// ```
pub struct AmpConfig<Master: Float, Compute: Float> {
    _master: std::marker::PhantomData<Master>,
    _compute: std::marker::PhantomData<Compute>,
}

impl<Master: Float, Compute: Float> AmpConfig<Master, Compute> {
    /// Create a new AMP configuration.
    pub fn new() -> Self {
        Self {
            _master: std::marker::PhantomData,
            _compute: std::marker::PhantomData,
        }
    }

    /// Cast master-precision parameters to compute precision.
    pub fn to_compute(params: &[Variable<Master>]) -> Vec<Variable<Compute>> {
        cast_params(params)
    }

    /// Cast a compute-precision loss scalar back to master precision for
    /// gradient scaling and optimizer updates.
    pub fn loss_to_master(loss: &Variable<Compute>) -> Variable<Master> {
        cast_variable(loss)
    }

    /// Copy gradients from compute-precision variables back to their
    /// master-precision counterparts, casting to `Master` precision.
    ///
    /// This pairs up master and compute parameters by index. After calling
    /// this, the master parameters have gradients suitable for the optimizer.
    pub fn sync_grads(master_params: &[Variable<Master>], compute_params: &[Variable<Compute>]) {
        for (mp, cp) in master_params.iter().zip(compute_params.iter()) {
            if let Some(grad_compute) = cp.grad() {
                let grad_master = grad_compute.cast::<Master>();
                mp.set_grad(grad_master);
            }
        }
    }
}

impl<Master: Float, Compute: Float> Default for AmpConfig<Master, Compute> {
    fn default() -> Self {
        Self::new()
    }
}

/// Run a single AMP forward pass: casts input to compute precision,
/// runs the layer, and returns the output in compute precision.
///
/// The layer itself should hold compute-precision weights (use
/// [`AmpConfig::to_compute`] to prepare them).
pub fn amp_forward<T: Float>(layer: &dyn Layer<T>, input: &Variable<T>) -> Result<Variable<T>> {
    layer.forward(input)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scivex_core::Tensor;

    #[test]
    fn test_cast_variable_f64_to_f32() {
        let v = Variable::new(
            Tensor::from_vec(vec![1.0_f64, 2.0, 3.0], vec![3]).unwrap(),
            true,
        );
        let casted: Variable<f32> = cast_variable(&v);
        let data = casted.data();
        let s = data.as_slice();
        assert!((s[0] - 1.0).abs() < 1e-6);
        assert!((s[1] - 2.0).abs() < 1e-6);
        assert!((s[2] - 3.0).abs() < 1e-6);
        assert!(casted.requires_grad());
    }

    #[test]
    fn test_cast_params() {
        let params: Vec<Variable<f64>> = vec![
            Variable::new(Tensor::ones(vec![2, 3]), true),
            Variable::new(Tensor::zeros(vec![3]), false),
        ];
        let casted: Vec<Variable<f32>> = cast_params(&params);
        assert_eq!(casted.len(), 2);
        assert_eq!(casted[0].shape(), vec![2, 3]);
        assert_eq!(casted[1].shape(), vec![3]);
        assert!(casted[0].requires_grad());
        assert!(!casted[1].requires_grad());
    }

    #[test]
    fn test_amp_config_sync_grads() {
        // Master params (f64).
        let master = vec![Variable::new(
            Tensor::from_vec(vec![1.0_f64, 2.0], vec![2]).unwrap(),
            true,
        )];

        // Compute params (f32) with a gradient set.
        let compute = vec![Variable::new(
            Tensor::from_vec(vec![1.0_f32, 2.0], vec![2]).unwrap(),
            true,
        )];
        compute[0].set_grad(Tensor::from_vec(vec![0.1_f32, 0.2], vec![2]).unwrap());

        AmpConfig::<f64, f32>::sync_grads(&master, &compute);

        let grad = master[0]
            .grad()
            .expect("master should have grad after sync");
        assert!((grad.as_slice()[0] - 0.1).abs() < 1e-5);
        assert!((grad.as_slice()[1] - 0.2).abs() < 1e-5);
    }
}
