//! Dynamic gradient scaling for mixed-precision training.
//!
//! When training with reduced precision (f16), small gradient magnitudes can
//! underflow to zero. [`GradScaler`] multiplies the loss by a large scale
//! factor before `backward()`, then divides gradients by the same factor before
//! the optimizer step. If any gradient is non-finite (overflow), the step is
//! skipped and the scale is reduced.
//!
//! This mirrors PyTorch's `torch.amp.GradScaler`.

use scivex_core::Float;

use crate::optim::Optimizer;
use crate::variable::Variable;

/// Dynamic loss scaler for mixed-precision training.
///
/// # Usage
///
/// ```ignore
/// let mut scaler = GradScaler::new();
/// for (x, y) in data {
///     optimizer.zero_grad();
///     let pred = model.forward(&x);
///     let loss = mse_loss(&pred, &y);
///     let scaled_loss = scaler.scale(&loss);
///     scaled_loss.backward();
///     scaler.step(&mut optimizer, &params);
///     scaler.update();
/// }
/// ```
pub struct GradScaler<T: Float> {
    scale: T,
    growth_factor: T,
    backoff_factor: T,
    growth_interval: usize,
    steps_since_growth: usize,
    found_inf: bool,
}

impl<T: Float> GradScaler<T> {
    /// Create a new scaler with default settings.
    ///
    /// - Initial scale: 2^16 = 65536
    /// - Growth factor: 2.0
    /// - Backoff factor: 0.5
    /// - Growth interval: 2000 steps without overflow
    pub fn new() -> Self {
        Self {
            scale: T::from_f64(65536.0),
            growth_factor: T::from_f64(2.0),
            backoff_factor: T::from_f64(0.5),
            growth_interval: 2000,
            steps_since_growth: 0,
            found_inf: false,
        }
    }

    /// Set the initial scale factor.
    #[must_use]
    pub fn with_init_scale(mut self, scale: T) -> Self {
        self.scale = scale;
        self
    }

    /// Set the growth factor (default 2.0).
    #[must_use]
    pub fn with_growth_factor(mut self, factor: T) -> Self {
        self.growth_factor = factor;
        self
    }

    /// Set the backoff factor (default 0.5).
    #[must_use]
    pub fn with_backoff_factor(mut self, factor: T) -> Self {
        self.backoff_factor = factor;
        self
    }

    /// Set the growth interval in steps (default 2000).
    #[must_use]
    pub fn with_growth_interval(mut self, interval: usize) -> Self {
        self.growth_interval = interval;
        self
    }

    /// Return the current scale factor.
    pub fn get_scale(&self) -> T {
        self.scale
    }

    /// Multiply the loss by the current scale factor.
    ///
    /// Returns a new `Variable` whose value is `loss * scale`. Call
    /// `.backward()` on the returned variable so that all gradients are
    /// scaled by the same factor.
    pub fn scale(&self, loss: &Variable<T>) -> Variable<T> {
        crate::ops::scalar_mul(loss, self.scale)
    }

    /// Unscale gradients, check for infs/NaNs, and conditionally step.
    ///
    /// 1. Divides every parameter's gradient by the current scale.
    /// 2. If any gradient is non-finite, marks `found_inf = true` and
    ///    **skips** the optimizer step.
    /// 3. Otherwise, calls `optimizer.step()`.
    pub fn step<O: Optimizer<T>>(&mut self, optimizer: &mut O, params: &[Variable<T>]) {
        self.found_inf = false;
        let inv_scale = self.scale.recip();

        // Unscale gradients and check for non-finite values.
        for p in params {
            if let Some(grad) = p.grad() {
                let unscaled = grad.map(|g| g * inv_scale);
                let has_inf = unscaled.as_slice().iter().any(|&v| !v.is_finite());
                if has_inf {
                    self.found_inf = true;
                    return;
                }
                // Replace the gradient with the unscaled version.
                p.set_grad(unscaled);
            }
        }

        optimizer.step();
    }

    /// Update the scale factor after each training step.
    ///
    /// - If overflow was detected, scale is reduced by `backoff_factor`.
    /// - If `growth_interval` consecutive steps pass without overflow,
    ///   scale is increased by `growth_factor`.
    pub fn update(&mut self) {
        if self.found_inf {
            self.scale *= self.backoff_factor;
            self.steps_since_growth = 0;
        } else {
            self.steps_since_growth += 1;
            if self.steps_since_growth >= self.growth_interval {
                self.scale *= self.growth_factor;
                self.steps_since_growth = 0;
            }
        }
    }
}

impl<T: Float> Default for GradScaler<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scivex_core::Tensor;

    #[test]
    fn test_grad_scaler_scale() {
        let scaler = GradScaler::<f64>::new();
        let v = Variable::new(Tensor::from_vec(vec![2.0], vec![1]).unwrap(), true);
        let scaled = scaler.scale(&v);
        assert!((scaled.data().as_slice()[0] - 2.0 * 65536.0).abs() < 1e-6);
    }

    #[test]
    fn test_grad_scaler_backoff_on_inf() {
        let mut scaler = GradScaler::<f64>::new();
        let initial_scale = scaler.get_scale();

        // Simulate inf gradient.
        scaler.found_inf = true;
        scaler.update();

        assert!((scaler.get_scale() - initial_scale * 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_grad_scaler_growth() {
        let mut scaler = GradScaler::<f64>::new().with_growth_interval(2);
        let initial_scale = scaler.get_scale();

        // Two successful steps → growth.
        scaler.found_inf = false;
        scaler.update();
        assert!((scaler.get_scale() - initial_scale).abs() < 1e-6);

        scaler.update();
        assert!((scaler.get_scale() - initial_scale * 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_grad_scaler_default() {
        let scaler = GradScaler::<f32>::default();
        assert!((scaler.get_scale() - 65536.0).abs() < 1e-2);
    }
}
