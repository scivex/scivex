//! Training utilities: trainer loop, callbacks, gradient clipping, mixed precision, and LR finder.

/// Automatic mixed-precision (AMP) casting utilities.
pub mod amp;
mod callbacks;
mod grad_clip;
/// Dynamic gradient scaling for mixed-precision training.
pub mod grad_scaler;
/// Learning rate range test (LR Finder).
pub mod lr_finder;
mod trainer;

pub use amp::{AmpConfig, cast_params, cast_variable};
pub use callbacks::{Callback, CallbackAction, EarlyStopping, LossLogger, ModelCheckpoint};
pub use grad_clip::{clip_grad_norm, clip_grad_value};
pub use grad_scaler::GradScaler;
pub use lr_finder::{LrFinder, LrFinderResult};
pub use trainer::{Trainer, TrainingHistory};
