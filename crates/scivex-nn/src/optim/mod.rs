//! Optimizers and learning rate schedulers for training neural networks.

mod adagrad;
mod adam;
mod adamw;
mod rmsprop;
/// Learning rate schedulers.
pub mod scheduler;
mod sgd;

pub use adagrad::Adagrad;
pub use adam::Adam;
pub use adamw::AdamW;
pub use rmsprop::RMSprop;
pub use scheduler::{
    CosineAnnealingLR, ExponentialLR, LinearLR, LrScheduler, ReduceLROnPlateau, StepLR,
    WarmupCosineDecay,
};
pub use sgd::SGD;

use scivex_core::Float;

/// An optimizer updates model parameters using their gradients.
///
/// # Examples
///
/// ```
/// # use scivex_nn::optim::{Optimizer, SGD};
/// # use scivex_nn::layer::{Linear, Layer};
/// # use scivex_nn::variable::Variable;
/// # use scivex_core::{Tensor, random::Rng};
/// let mut rng = Rng::new(42);
/// let layer = Linear::<f64>::new(2, 1, false, &mut rng);
/// let params = layer.parameters();
/// let mut opt = SGD::new(params, 0.01_f64);
/// opt.zero_grad();
/// opt.step();
/// ```
pub trait Optimizer<T: Float> {
    /// Perform a single optimization step (parameter update).
    fn step(&mut self);

    /// Reset all parameter gradients to zero.
    fn zero_grad(&mut self);
}
