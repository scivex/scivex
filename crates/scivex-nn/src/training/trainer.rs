//! Trainer — runs the training loop with callback support.

use scivex_core::Float;

use crate::error::Result;
use crate::training::callbacks::{Callback, CallbackAction};

/// Record of a completed training run.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct TrainingHistory<T: Float> {
    /// Loss value at each epoch.
    pub losses: Vec<T>,
    /// Whether training was stopped early by a callback.
    pub stopped_early: bool,
    /// Epoch index that achieved the lowest loss.
    pub best_epoch: usize,
}

/// Orchestrates a training loop with configurable callbacks.
pub struct Trainer<T: Float> {
    epochs: usize,
    callbacks: Vec<Box<dyn Callback<T>>>,
}

impl<T: Float> Trainer<T> {
    /// Create a new `Trainer` that will run for `epochs` epochs.
    pub fn new(epochs: usize) -> Self {
        Self {
            epochs,
            callbacks: Vec::new(),
        }
    }

    /// Register a callback to be invoked during training.
    pub fn add_callback(&mut self, cb: Box<dyn Callback<T>>) -> &mut Self {
        self.callbacks.push(cb);
        self
    }

    /// Run the training loop.
    ///
    /// `train_fn` is called once per epoch with the epoch index (0-based) and
    /// must return the training loss for that epoch.
    pub fn fit<F>(&mut self, mut train_fn: F) -> Result<TrainingHistory<T>>
    where
        F: FnMut(usize) -> Result<T>,
    {
        // Notify callbacks.
        for cb in &mut self.callbacks {
            cb.on_train_begin();
        }

        let mut losses: Vec<T> = Vec::with_capacity(self.epochs);
        let mut stopped_early = false;
        let mut best_epoch: usize = 0;
        let mut best_loss: Option<T> = None;

        for epoch in 0..self.epochs {
            let loss = train_fn(epoch)?;
            losses.push(loss);

            // Track best epoch.
            let is_best = match best_loss {
                None => true,
                Some(prev) => loss < prev,
            };
            if is_best {
                best_loss = Some(loss);
                best_epoch = epoch;
            }

            // Run callbacks; stop if any returns Stop.
            let mut should_stop = false;
            for cb in &mut self.callbacks {
                if cb.on_epoch_end(epoch, loss) == CallbackAction::Stop {
                    should_stop = true;
                }
            }
            if should_stop {
                stopped_early = true;
                break;
            }
        }

        // Notify callbacks.
        for cb in &mut self.callbacks {
            cb.on_train_end();
        }

        Ok(TrainingHistory {
            losses,
            stopped_early,
            best_epoch,
        })
    }
}
