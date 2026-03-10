//! Optimizers for training neural networks.

mod adam;
mod sgd;

pub use adam::Adam;
pub use sgd::SGD;

use scivex_core::Float;

/// An optimizer updates model parameters using their gradients.
pub trait Optimizer<T: Float> {
    /// Perform a single optimization step (parameter update).
    fn step(&mut self);

    /// Reset all parameter gradients to zero.
    fn zero_grad(&mut self);
}
