//! Learning rate schedulers for dynamic learning rate adjustment during training.
//!
//! Schedulers adjust the learning rate of an optimizer at each step or epoch,
//! implementing common strategies from the deep learning literature.

use scivex_core::Float;

/// A learning rate scheduler computes the current learning rate given the step/epoch.
///
/// # Examples
///
/// ```
/// # use scivex_nn::optim::scheduler::{StepLR, LrScheduler};
/// let sched = StepLR::new(0.1_f64, 10, 0.5);
/// assert!((sched.get_lr(0) - 0.1).abs() < 1e-10);
/// assert!((sched.get_lr(10) - 0.05).abs() < 1e-10);
/// assert_eq!(sched.base_lr(), 0.1);
/// ```
pub trait LrScheduler<T: Float> {
    /// Return the learning rate for the given step (0-indexed).
    fn get_lr(&self, step: usize) -> T;

    /// Return the base (initial) learning rate.
    fn base_lr(&self) -> T;
}

// ─────────────────────────────────────────────────────────────────────
// StepLR — reduce LR by a factor every `step_size` epochs
// ─────────────────────────────────────────────────────────────────────

/// Decays the learning rate by `gamma` every `step_size` epochs.
///
/// `lr = base_lr * gamma^(step / step_size)`
pub struct StepLR<T: Float> {
    base: T,
    step_size: usize,
    gamma: T,
}

impl<T: Float> StepLR<T> {
    /// Create a new `StepLR` scheduler.
    ///
    /// - `base_lr`: initial learning rate
    /// - `step_size`: period of learning rate decay (in epochs/steps)
    /// - `gamma`: multiplicative factor of learning rate decay (default: 0.1)
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_nn::optim::scheduler::{StepLR, LrScheduler};
    /// let sched = StepLR::new(0.1_f64, 10, 0.1);
    /// assert!((sched.get_lr(0) - 0.1).abs() < 1e-10);
    /// assert!((sched.get_lr(10) - 0.01).abs() < 1e-10);
    /// ```
    pub fn new(base_lr: T, step_size: usize, gamma: T) -> Self {
        Self {
            base: base_lr,
            step_size,
            gamma,
        }
    }
}

impl<T: Float> LrScheduler<T> for StepLR<T> {
    fn get_lr(&self, step: usize) -> T {
        let n = step / self.step_size;
        self.base * self.gamma.powf(T::from_usize(n))
    }

    fn base_lr(&self) -> T {
        self.base
    }
}

// ─────────────────────────────────────────────────────────────────────
// ExponentialLR — decay LR exponentially every step
// ─────────────────────────────────────────────────────────────────────

/// Decays the learning rate exponentially: `lr = base_lr * gamma^step`.
pub struct ExponentialLR<T: Float> {
    base: T,
    gamma: T,
}

impl<T: Float> ExponentialLR<T> {
    /// Create a new `ExponentialLR` scheduler.
    ///
    /// - `base_lr`: initial learning rate
    /// - `gamma`: multiplicative factor applied every step
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_nn::optim::scheduler::{ExponentialLR, LrScheduler};
    /// let sched = ExponentialLR::new(1.0_f64, 0.9);
    /// assert!((sched.get_lr(0) - 1.0).abs() < 1e-10);
    /// assert!((sched.get_lr(1) - 0.9).abs() < 1e-10);
    /// assert!((sched.get_lr(2) - 0.81).abs() < 1e-10);
    /// ```
    pub fn new(base_lr: T, gamma: T) -> Self {
        Self {
            base: base_lr,
            gamma,
        }
    }
}

impl<T: Float> LrScheduler<T> for ExponentialLR<T> {
    fn get_lr(&self, step: usize) -> T {
        self.base * self.gamma.powf(T::from_usize(step))
    }

    fn base_lr(&self) -> T {
        self.base
    }
}

// ─────────────────────────────────────────────────────────────────────
// CosineAnnealingLR — cosine decay to eta_min over T_max steps
// ─────────────────────────────────────────────────────────────────────

/// Cosine annealing schedule: decays LR from `base_lr` to `eta_min` over
/// `t_max` steps following a half-cosine curve.
///
/// `lr = eta_min + 0.5 * (base_lr - eta_min) * (1 + cos(pi * step / t_max))`
pub struct CosineAnnealingLR<T: Float> {
    base: T,
    t_max: usize,
    eta_min: T,
}

impl<T: Float> CosineAnnealingLR<T> {
    /// Create a new `CosineAnnealingLR` scheduler.
    ///
    /// - `base_lr`: initial learning rate
    /// - `t_max`: maximum number of steps (half-period of cosine)
    /// - `eta_min`: minimum learning rate (default: 0.0)
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_nn::optim::scheduler::{CosineAnnealingLR, LrScheduler};
    /// let sched = CosineAnnealingLR::new(0.1_f64, 100, 0.0);
    /// assert!((sched.get_lr(0) - 0.1).abs() < 1e-10);
    /// assert!(sched.get_lr(100).abs() < 1e-10); // decays to eta_min
    /// ```
    pub fn new(base_lr: T, t_max: usize, eta_min: T) -> Self {
        Self {
            base: base_lr,
            t_max,
            eta_min,
        }
    }
}

impl<T: Float> LrScheduler<T> for CosineAnnealingLR<T> {
    fn get_lr(&self, step: usize) -> T {
        let clamped = if step >= self.t_max { self.t_max } else { step };
        let pi = T::from_f64(std::f64::consts::PI);
        let ratio = T::from_usize(clamped) / T::from_usize(self.t_max);
        let cos_val = (pi * ratio).cos();
        self.eta_min + T::from_f64(0.5) * (self.base - self.eta_min) * (T::one() + cos_val)
    }

    fn base_lr(&self) -> T {
        self.base
    }
}

// ─────────────────────────────────────────────────────────────────────
// LinearLR — linearly interpolate from start_factor to end_factor
// ─────────────────────────────────────────────────────────────────────

/// Linearly interpolates the learning rate multiplier from `start_factor`
/// to `end_factor` over `total_steps`.
///
/// `lr = base_lr * (start_factor + (end_factor - start_factor) * min(step, total) / total)`
pub struct LinearLR<T: Float> {
    base: T,
    start_factor: T,
    end_factor: T,
    total_steps: usize,
}

impl<T: Float> LinearLR<T> {
    /// Create a new `LinearLR` scheduler.
    ///
    /// - `base_lr`: initial learning rate
    /// - `start_factor`: multiplier at step 0
    /// - `end_factor`: multiplier at `total_steps`
    /// - `total_steps`: number of steps for the linear ramp
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_nn::optim::scheduler::{LinearLR, LrScheduler};
    /// let sched = LinearLR::new(0.1_f64, 0.1, 1.0, 10);
    /// // At step 0: lr = 0.1 * 0.1 = 0.01
    /// assert!((sched.get_lr(0) - 0.01).abs() < 1e-10);
    /// // At step 10: lr = 0.1 * 1.0 = 0.1
    /// assert!((sched.get_lr(10) - 0.1).abs() < 1e-10);
    /// ```
    pub fn new(base_lr: T, start_factor: T, end_factor: T, total_steps: usize) -> Self {
        Self {
            base: base_lr,
            start_factor,
            end_factor,
            total_steps,
        }
    }
}

impl<T: Float> LrScheduler<T> for LinearLR<T> {
    fn get_lr(&self, step: usize) -> T {
        let clamped = if step >= self.total_steps {
            self.total_steps
        } else {
            step
        };
        let ratio = T::from_usize(clamped) / T::from_usize(self.total_steps);
        let factor = self.start_factor + (self.end_factor - self.start_factor) * ratio;
        self.base * factor
    }

    fn base_lr(&self) -> T {
        self.base
    }
}

// ─────────────────────────────────────────────────────────────────────
// WarmupCosineDecay — linear warmup then cosine decay
// ─────────────────────────────────────────────────────────────────────

/// Linear warmup for `warmup_steps`, then cosine decay to `eta_min`
/// over the remaining steps until `total_steps`.
pub struct WarmupCosineDecay<T: Float> {
    base: T,
    warmup_steps: usize,
    total_steps: usize,
    eta_min: T,
}

impl<T: Float> WarmupCosineDecay<T> {
    /// Create a new `WarmupCosineDecay` scheduler.
    ///
    /// - `base_lr`: peak learning rate (reached at end of warmup)
    /// - `warmup_steps`: number of linear warmup steps
    /// - `total_steps`: total training steps
    /// - `eta_min`: minimum learning rate
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_nn::optim::scheduler::{WarmupCosineDecay, LrScheduler};
    /// let sched = WarmupCosineDecay::new(0.1_f64, 10, 110, 0.0);
    /// // During warmup: linear ramp
    /// assert!((sched.get_lr(0) - 0.01).abs() < 1e-10); // 0.1 * 1/10
    /// // At warmup end: peak LR
    /// assert!((sched.get_lr(9) - 0.1).abs() < 1e-10);
    /// // At total_steps: decays to eta_min
    /// assert!(sched.get_lr(110).abs() < 1e-6);
    /// ```
    pub fn new(base_lr: T, warmup_steps: usize, total_steps: usize, eta_min: T) -> Self {
        Self {
            base: base_lr,
            warmup_steps,
            total_steps,
            eta_min,
        }
    }
}

impl<T: Float> LrScheduler<T> for WarmupCosineDecay<T> {
    fn get_lr(&self, step: usize) -> T {
        if step < self.warmup_steps {
            // Linear warmup: lr = base_lr * (step + 1) / warmup_steps
            self.base * T::from_usize(step + 1) / T::from_usize(self.warmup_steps)
        } else {
            // Cosine decay phase
            let decay_steps = self.total_steps - self.warmup_steps;
            if decay_steps == 0 {
                return self.base;
            }
            let progress = step - self.warmup_steps;
            let clamped = if progress >= decay_steps {
                decay_steps
            } else {
                progress
            };
            let pi = T::from_f64(std::f64::consts::PI);
            let ratio = T::from_usize(clamped) / T::from_usize(decay_steps);
            let cos_val = (pi * ratio).cos();
            self.eta_min + T::from_f64(0.5) * (self.base - self.eta_min) * (T::one() + cos_val)
        }
    }

    fn base_lr(&self) -> T {
        self.base
    }
}

// ─────────────────────────────────────────────────────────────────────
// ReduceLROnPlateau — reduce LR when a metric stops improving
// ─────────────────────────────────────────────────────────────────────

/// Reduces the learning rate when a monitored metric has stopped improving.
///
/// Unlike the other schedulers which are purely step-based, this one
/// requires explicit calls to [`report`](ReduceLROnPlateau::report) with
/// the current metric value.
pub struct ReduceLROnPlateau<T: Float> {
    current_lr: T,
    factor: T,
    patience: usize,
    min_lr: T,
    best: Option<T>,
    num_bad_epochs: usize,
    mode_min: bool,
}

impl<T: Float> ReduceLROnPlateau<T> {
    /// Create a new `ReduceLROnPlateau` scheduler.
    ///
    /// - `initial_lr`: starting learning rate
    /// - `factor`: factor by which LR is reduced (`new_lr = lr * factor`)
    /// - `patience`: number of epochs with no improvement before reducing LR
    /// - `min_lr`: lower bound on the learning rate
    /// - `mode_min`: if `true`, reduction triggered when metric stops decreasing;
    ///   if `false`, when metric stops increasing.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_nn::optim::scheduler::ReduceLROnPlateau;
    /// let mut sched = ReduceLROnPlateau::new(0.1_f64, 0.5, 2, 0.001, true);
    /// sched.report(1.0);  // initial best
    /// sched.report(1.1);  // bad epoch 1
    /// let lr = sched.report(1.1);  // bad epoch 2 → triggers reduction
    /// assert!((lr - 0.05).abs() < 1e-10);
    /// ```
    pub fn new(initial_lr: T, factor: T, patience: usize, min_lr: T, mode_min: bool) -> Self {
        Self {
            current_lr: initial_lr,
            factor,
            patience,
            min_lr,
            best: None,
            num_bad_epochs: 0,
            mode_min,
        }
    }

    /// Report the current metric value. Returns the new learning rate.
    pub fn report(&mut self, metric: T) -> T {
        let improved = match self.best {
            None => true,
            Some(best) => {
                if self.mode_min {
                    metric < best
                } else {
                    metric > best
                }
            }
        };

        if improved {
            self.best = Some(metric);
            self.num_bad_epochs = 0;
        } else {
            self.num_bad_epochs += 1;
            if self.num_bad_epochs >= self.patience {
                let new_lr = self.current_lr * self.factor;
                if new_lr >= self.min_lr {
                    self.current_lr = new_lr;
                } else {
                    self.current_lr = self.min_lr;
                }
                self.num_bad_epochs = 0;
            }
        }
        self.current_lr
    }

    /// Return the current learning rate.
    pub fn current_lr(&self) -> T {
        self.current_lr
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step_lr_basic() {
        let s = StepLR::new(0.1_f64, 10, 0.1);
        assert!((s.get_lr(0) - 0.1).abs() < 1e-10);
        assert!((s.get_lr(9) - 0.1).abs() < 1e-10);
        assert!((s.get_lr(10) - 0.01).abs() < 1e-10);
        assert!((s.get_lr(20) - 0.001).abs() < 1e-10);
    }

    #[test]
    fn exponential_lr_basic() {
        let s = ExponentialLR::new(1.0_f64, 0.9);
        assert!((s.get_lr(0) - 1.0).abs() < 1e-10);
        assert!((s.get_lr(1) - 0.9).abs() < 1e-10);
        assert!((s.get_lr(2) - 0.81).abs() < 1e-10);
    }

    #[test]
    fn cosine_annealing_endpoints() {
        let s = CosineAnnealingLR::new(0.1_f64, 100, 0.0);
        // At step 0: lr = 0.1
        assert!((s.get_lr(0) - 0.1).abs() < 1e-10);
        // At step 100: lr = 0.0 (eta_min)
        assert!(s.get_lr(100).abs() < 1e-10);
        // At step 50: lr = 0.05 (halfway)
        assert!((s.get_lr(50) - 0.05).abs() < 1e-6);
    }

    #[test]
    fn linear_lr_ramp() {
        let s = LinearLR::new(0.1_f64, 0.1, 1.0, 10);
        // At step 0: lr = 0.1 * 0.1 = 0.01
        assert!((s.get_lr(0) - 0.01).abs() < 1e-10);
        // At step 10: lr = 0.1 * 1.0 = 0.1
        assert!((s.get_lr(10) - 0.1).abs() < 1e-10);
        // At step 5: lr = 0.1 * 0.55 = 0.055
        assert!((s.get_lr(5) - 0.055).abs() < 1e-10);
    }

    #[test]
    fn warmup_cosine_decay_phases() {
        let s = WarmupCosineDecay::new(0.1_f64, 10, 110, 0.0);
        // During warmup: linear ramp
        assert!((s.get_lr(0) - 0.01).abs() < 1e-10); // 0.1 * 1/10
        assert!((s.get_lr(4) - 0.05).abs() < 1e-10); // 0.1 * 5/10
        assert!((s.get_lr(9) - 0.1).abs() < 1e-10); // 0.1 * 10/10
        // At warmup end: peak
        assert!((s.get_lr(10) - 0.1).abs() < 1e-6);
        // At total_steps: eta_min
        assert!(s.get_lr(110).abs() < 1e-6);
    }

    #[test]
    fn reduce_on_plateau_reduces() {
        let mut s = ReduceLROnPlateau::new(0.1_f64, 0.5, 3, 0.001, true);
        // Report improving metrics
        assert!((s.report(1.0) - 0.1).abs() < 1e-10);
        assert!((s.report(0.9) - 0.1).abs() < 1e-10);
        // Now stall for patience=3
        assert!((s.report(0.95) - 0.1).abs() < 1e-10); // bad 1
        assert!((s.report(0.95) - 0.1).abs() < 1e-10); // bad 2
        // bad 3 triggers reduction
        let lr = s.report(0.95);
        assert!((lr - 0.05).abs() < 1e-10);
    }

    #[test]
    fn reduce_on_plateau_respects_min_lr() {
        let mut s = ReduceLROnPlateau::new(0.01_f64, 0.1, 1, 0.005, true);
        s.report(1.0);
        // One bad epoch triggers reduction: 0.01 * 0.1 = 0.001 < min_lr
        let lr = s.report(2.0);
        assert!((lr - 0.005).abs() < 1e-10);
    }
}
