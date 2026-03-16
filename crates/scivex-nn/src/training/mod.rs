//! Training utilities: trainer loop, callbacks, and gradient clipping.

mod callbacks;
mod grad_clip;
mod trainer;

pub use callbacks::{Callback, CallbackAction, EarlyStopping, LossLogger, ModelCheckpoint};
pub use grad_clip::{clip_grad_norm, clip_grad_value};
pub use trainer::{Trainer, TrainingHistory};
