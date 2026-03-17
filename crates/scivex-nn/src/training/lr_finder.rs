//! Learning rate range test (LR Finder).
//!
//! Implements Leslie Smith's learning rate range test: train for a short run
//! while exponentially increasing the learning rate from a very small value
//! to a large one. Record the loss at each step. The optimal learning rate is
//! typically where the loss decreases most steeply (steepest negative gradient
//! on the loss-vs-log(lr) curve).
//!
//! # References
//!
//! - Smith, L.N. "Cyclical Learning Rates for Training Neural Networks" (2017)
//! - fastai's `lr_find()` implementation

use scivex_core::Float;

use crate::error::Result;

/// Result of a learning rate range test.
#[derive(Debug, Clone)]
pub struct LrFinderResult<T: Float> {
    /// Learning rates tested (one per step).
    pub lrs: Vec<T>,
    /// Smoothed loss at each step.
    pub losses: Vec<T>,
    /// Raw (unsmoothed) loss at each step.
    pub raw_losses: Vec<T>,
    /// Suggested learning rate (steepest loss descent).
    pub suggested_lr: T,
}

/// Configuration for the LR range test.
pub struct LrFinder<T: Float> {
    start_lr: T,
    end_lr: T,
    num_steps: usize,
    smoothing: T,
    diverge_threshold: T,
}

impl<T: Float> LrFinder<T> {
    /// Create a new LR finder.
    ///
    /// Defaults:
    /// - `start_lr`: 1e-7
    /// - `end_lr`: 10.0
    /// - `num_steps`: 100
    /// - `smoothing`: 0.05 (exponential moving average weight for new loss)
    /// - `diverge_threshold`: 4.0 (stop if loss exceeds 4x the best loss)
    pub fn new() -> Self {
        Self {
            start_lr: T::from_f64(1e-7),
            end_lr: T::from_f64(10.0),
            num_steps: 100,
            smoothing: T::from_f64(0.05),
            diverge_threshold: T::from_f64(4.0),
        }
    }

    /// Set the starting learning rate.
    #[must_use]
    pub fn with_start_lr(mut self, lr: T) -> Self {
        self.start_lr = lr;
        self
    }

    /// Set the ending learning rate.
    #[must_use]
    pub fn with_end_lr(mut self, lr: T) -> Self {
        self.end_lr = lr;
        self
    }

    /// Set the number of steps.
    #[must_use]
    pub fn with_num_steps(mut self, n: usize) -> Self {
        self.num_steps = n;
        self
    }

    /// Set the smoothing factor (0 = no smoothing, 1 = full smoothing).
    #[must_use]
    pub fn with_smoothing(mut self, s: T) -> Self {
        self.smoothing = s;
        self
    }

    /// Set the divergence threshold multiplier (default 4.0).
    #[must_use]
    pub fn with_diverge_threshold(mut self, t: T) -> Self {
        self.diverge_threshold = t;
        self
    }

    /// Run the learning rate range test.
    ///
    /// `train_step` is called once per step with the current learning rate.
    /// It should:
    /// 1. Set the optimizer's learning rate to the given value
    /// 2. Run one forward + backward + step
    /// 3. Return the loss as a scalar `T`
    ///
    /// The test stops early if the loss diverges (exceeds `diverge_threshold`
    /// times the best observed loss).
    pub fn run<F>(&self, mut train_step: F) -> Result<LrFinderResult<T>>
    where
        F: FnMut(T) -> Result<T>,
    {
        let mut lrs = Vec::with_capacity(self.num_steps);
        let mut losses = Vec::with_capacity(self.num_steps);
        let mut raw_losses = Vec::with_capacity(self.num_steps);

        // Exponential schedule: lr = start_lr * (end_lr / start_lr) ^ (step / (num_steps - 1))
        let ratio = self.end_lr / self.start_lr;
        let one = T::one();

        let mut best_loss = T::infinity();
        let mut smoothed_loss = T::zero();

        let denom = if self.num_steps > 1 {
            T::from_usize(self.num_steps - 1)
        } else {
            one
        };

        for step in 0..self.num_steps {
            let t = T::from_usize(step) / denom;
            let lr = self.start_lr * ratio.powf(t);

            let raw_loss = train_step(lr)?;

            // Exponential moving average smoothing.
            smoothed_loss = if step == 0 {
                raw_loss
            } else {
                self.smoothing * raw_loss + (one - self.smoothing) * smoothed_loss
            };

            lrs.push(lr);
            losses.push(smoothed_loss);
            raw_losses.push(raw_loss);

            if smoothed_loss < best_loss {
                best_loss = smoothed_loss;
            }

            // Stop if loss has diverged.
            if step > 0 && smoothed_loss > best_loss * self.diverge_threshold {
                break;
            }
        }

        let suggested_lr = find_steepest_descent(&lrs, &losses);

        Ok(LrFinderResult {
            lrs,
            losses,
            raw_losses,
            suggested_lr,
        })
    }
}

impl<T: Float> Default for LrFinder<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Find the learning rate at the point of steepest loss descent.
///
/// Computes the finite-difference gradient of loss w.r.t. log(lr) and returns
/// the LR at the most negative gradient point.
fn find_steepest_descent<T: Float>(lrs: &[T], losses: &[T]) -> T {
    if lrs.len() < 3 {
        // Not enough data; return the midpoint LR.
        return lrs[lrs.len() / 2];
    }

    let mut best_idx = 1;
    let mut best_grad = T::infinity();

    // Use central differences for interior points.
    for i in 1..lrs.len() - 1 {
        let log_lr_prev = lrs[i - 1].ln();
        let log_lr_next = lrs[i + 1].ln();
        let d_log_lr = log_lr_next - log_lr_prev;

        if d_log_lr.abs() < T::from_f64(1e-12) {
            continue;
        }

        let grad = (losses[i + 1] - losses[i - 1]) / d_log_lr;

        if grad < best_grad {
            best_grad = grad;
            best_idx = i;
        }
    }

    lrs[best_idx]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lr_finder_basic() {
        // Simulate a convex loss: loss = (lr - 0.01)^2 + 0.1
        // Minimum at lr = 0.01, steepest descent should be before that.
        let finder = LrFinder::<f64>::new()
            .with_start_lr(1e-5)
            .with_end_lr(1.0)
            .with_num_steps(50)
            .with_diverge_threshold(10.0);

        let result = finder
            .run(|lr| {
                let loss = (lr - 0.01) * (lr - 0.01) + 0.1;
                Ok(loss)
            })
            .unwrap();

        assert!(!result.lrs.is_empty());
        assert_eq!(result.lrs.len(), result.losses.len());
        assert!(result.suggested_lr > 0.0);
        // Suggested LR should be in a reasonable range (before divergence).
        assert!(result.suggested_lr < 1.0);
    }

    #[test]
    fn test_lr_finder_early_stop_on_divergence() {
        // Loss that explodes after lr > 0.1.
        let finder = LrFinder::<f64>::new()
            .with_start_lr(1e-5)
            .with_end_lr(10.0)
            .with_num_steps(100)
            .with_diverge_threshold(4.0);

        let result = finder
            .run(|lr| {
                let loss = if lr < 0.1 { 1.0 - lr * 5.0 } else { lr * 100.0 };
                Ok(loss)
            })
            .unwrap();

        // Should have stopped before 100 steps due to divergence.
        assert!(result.lrs.len() < 100);
    }

    #[test]
    fn test_lr_finder_steepest_descent() {
        // Provide pre-computed data to test the gradient calculation.
        let lrs: Vec<f64> = (0..10)
            .map(|i| 10.0_f64.powf(-4.0 + 0.5 * f64::from(i)))
            .collect();
        // Loss decreases most between steps 3-5, then increases.
        let losses: Vec<f64> = vec![1.0, 0.9, 0.7, 0.4, 0.2, 0.15, 0.2, 0.5, 1.0, 2.0];

        let suggested = find_steepest_descent(&lrs, &losses);
        // Steepest descent is around index 3-4 (lr ≈ 0.01 to 0.03).
        assert!(suggested > 1e-4);
        assert!(suggested < 1.0);
    }

    #[test]
    fn test_lr_finder_default() {
        let finder = LrFinder::<f32>::default();
        assert_eq!(finder.num_steps, 100);
    }
}
