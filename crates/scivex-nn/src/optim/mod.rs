//! Optimizers and learning rate schedulers for training neural networks.

mod adam;
/// Learning rate schedulers.
pub mod scheduler;
mod sgd;

pub use adam::Adam;
pub use scheduler::{
    CosineAnnealingLR, ExponentialLR, LinearLR, LrScheduler, ReduceLROnPlateau, StepLR,
    WarmupCosineDecay,
};
pub use sgd::SGD;

use scivex_core::Float;

/// An optimizer updates model parameters using their gradients.
pub trait Optimizer<T: Float> {
    /// Perform a single optimization step (parameter update).
    fn step(&mut self);

    /// Reset all parameter gradients to zero.
    fn zero_grad(&mut self);
}
