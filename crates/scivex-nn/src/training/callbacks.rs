//! Training callbacks: early stopping, model checkpointing, loss logging.

use scivex_core::Float;

/// Action returned by a callback to control the training loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallbackAction {
    /// Continue training.
    Continue,
    /// Stop training immediately.
    Stop,
}

/// Trait for training callbacks invoked by [`super::Trainer`].
pub trait Callback<T: Float> {
    /// Called at the end of each epoch with the epoch index and training loss.
    fn on_epoch_end(&mut self, epoch: usize, loss: T) -> CallbackAction;

    /// Called once before training begins.
    fn on_train_begin(&mut self) {
        // Default: no-op.
    }

    /// Called once after training ends (whether early-stopped or not).
    fn on_train_end(&mut self) {
        // Default: no-op.
    }
}

// ── EarlyStopping ────────────────────────────────────────────────────

/// Stops training when the loss has not improved for a given number of epochs.
pub struct EarlyStopping<T: Float> {
    /// Number of epochs with no improvement before stopping.
    patience: usize,
    /// Minimum absolute decrease in loss to qualify as an improvement.
    min_delta: T,
    /// Best loss seen so far.
    best_loss: Option<T>,
    /// Number of epochs since the last improvement.
    wait: usize,
}

impl<T: Float> EarlyStopping<T> {
    /// Create a new `EarlyStopping` callback.
    ///
    /// * `patience` — number of epochs to wait for improvement before stopping.
    /// * `min_delta` — minimum decrease in loss to count as an improvement.
    pub fn new(patience: usize, min_delta: T) -> Self {
        Self {
            patience,
            min_delta,
            best_loss: None,
            wait: 0,
        }
    }
}

impl<T: Float> Callback<T> for EarlyStopping<T> {
    fn on_epoch_end(&mut self, _epoch: usize, loss: T) -> CallbackAction {
        match self.best_loss {
            None => {
                self.best_loss = Some(loss);
                self.wait = 0;
                CallbackAction::Continue
            }
            Some(best) => {
                if best - loss > self.min_delta {
                    // Improvement.
                    self.best_loss = Some(loss);
                    self.wait = 0;
                    CallbackAction::Continue
                } else {
                    self.wait += 1;
                    if self.wait >= self.patience {
                        CallbackAction::Stop
                    } else {
                        CallbackAction::Continue
                    }
                }
            }
        }
    }

    fn on_train_begin(&mut self) {
        self.best_loss = None;
        self.wait = 0;
    }
}

// ── ModelCheckpoint ──────────────────────────────────────────────────

/// Tracks the epoch with the best (lowest) loss.
pub struct ModelCheckpoint<T: Float> {
    /// Best loss seen so far.
    best_loss: Option<T>,
    /// Epoch index that achieved the best loss.
    best_epoch: usize,
}

impl<T: Float> ModelCheckpoint<T> {
    /// Create a new `ModelCheckpoint` callback.
    pub fn new() -> Self {
        Self {
            best_loss: None,
            best_epoch: 0,
        }
    }

    /// Return the epoch that achieved the best (lowest) loss.
    pub fn best_epoch(&self) -> usize {
        self.best_epoch
    }

    /// Return the best loss value, if any epoch has completed.
    pub fn best_loss(&self) -> Option<T> {
        self.best_loss
    }
}

impl<T: Float> Default for ModelCheckpoint<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float> Callback<T> for ModelCheckpoint<T> {
    fn on_epoch_end(&mut self, epoch: usize, loss: T) -> CallbackAction {
        let is_best = match self.best_loss {
            None => true,
            Some(prev) => loss < prev,
        };
        if is_best {
            self.best_loss = Some(loss);
            self.best_epoch = epoch;
        }
        CallbackAction::Continue
    }

    fn on_train_begin(&mut self) {
        self.best_loss = None;
        self.best_epoch = 0;
    }
}

// ── LossLogger ───────────────────────────────────────────────────────

/// Records the loss at every epoch.
pub struct LossLogger<T: Float> {
    /// Recorded losses, one per epoch.
    losses: Vec<T>,
}

impl<T: Float> LossLogger<T> {
    /// Create a new `LossLogger`.
    pub fn new() -> Self {
        Self { losses: Vec::new() }
    }

    /// Return the recorded losses.
    pub fn losses(&self) -> &[T] {
        &self.losses
    }
}

impl<T: Float> Default for LossLogger<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float> Callback<T> for LossLogger<T> {
    fn on_epoch_end(&mut self, _epoch: usize, loss: T) -> CallbackAction {
        self.losses.push(loss);
        CallbackAction::Continue
    }

    fn on_train_begin(&mut self) {
        self.losses.clear();
    }
}
