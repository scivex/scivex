//! Gradient accumulation over multiple micro-batches.
//!
//! [`GradAccumulator`] wraps an [`Optimizer`] and delays parameter updates until
//! gradients have been accumulated across a configurable number of micro-batches.
//! This effectively simulates a larger batch size without requiring more memory.

use std::fmt;
use std::marker::PhantomData;

use scivex_core::Float;

use crate::error::{NnError, Result};
use crate::optim::Optimizer;
use crate::variable::Variable;

/// Accumulates gradients over multiple micro-batches before performing an
/// optimizer step.
///
/// After each micro-batch backward pass, call [`step`](GradAccumulator::step).
/// Gradients naturally accumulate in each [`Variable`] via repeated `backward()`
/// calls. Every `accumulation_steps` calls, the accumulated gradients are
/// divided by `accumulation_steps`, the inner optimizer performs a step, and
/// gradients are zeroed.
///
/// # Examples
///
/// ```
/// # use scivex_core::Tensor;
/// # use scivex_nn::variable::Variable;
/// # use scivex_nn::optim::{SGD, Optimizer};
/// # use scivex_nn::training::GradAccumulator;
/// let w = Variable::new(Tensor::from_vec(vec![1.0_f64, 2.0], vec![2]).unwrap(), true);
/// let params = vec![w.clone()];
/// let sgd = SGD::new(params.clone(), 0.01);
/// let mut accum = GradAccumulator::new(sgd, params, 4).unwrap();
/// ```
pub struct GradAccumulator<T: Float, O: Optimizer<T>> {
    optimizer: O,
    parameters: Vec<Variable<T>>,
    accumulation_steps: usize,
    current_step: usize,
    _marker: PhantomData<T>,
}

impl<T: Float, O: Optimizer<T>> fmt::Debug for GradAccumulator<T, O> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GradAccumulator")
            .field("accumulation_steps", &self.accumulation_steps)
            .field("current_step", &self.current_step)
            .field("num_parameters", &self.parameters.len())
            .finish_non_exhaustive()
    }
}

impl<T: Float, O: Optimizer<T>> GradAccumulator<T, O> {
    /// Create a new `GradAccumulator`.
    ///
    /// # Arguments
    ///
    /// * `optimizer` — the inner optimizer that will perform parameter updates.
    /// * `parameters` — the parameters whose gradients are accumulated and scaled.
    /// * `accumulation_steps` — number of micro-batches to accumulate before updating.
    ///
    /// # Errors
    ///
    /// Returns [`NnError::InvalidParameter`] if `accumulation_steps` is zero.
    pub fn new(
        optimizer: O,
        parameters: Vec<Variable<T>>,
        accumulation_steps: usize,
    ) -> Result<Self> {
        if accumulation_steps == 0 {
            return Err(NnError::InvalidParameter {
                name: "accumulation_steps",
                reason: "must be at least 1",
            });
        }
        Ok(Self {
            optimizer,
            parameters,
            accumulation_steps,
            current_step: 0,
            _marker: PhantomData,
        })
    }

    /// Called after each micro-batch backward pass.
    ///
    /// Increments the internal counter. Every `accumulation_steps` calls the
    /// accumulated gradients are divided by `accumulation_steps`, the inner
    /// optimizer performs a step, and gradients are zeroed.
    ///
    /// Returns `true` if an optimizer step was performed on this call.
    pub fn step(&mut self) -> bool {
        self.current_step += 1;
        if self.current_step >= self.accumulation_steps {
            self.scale_and_step();
            true
        } else {
            false
        }
    }

    /// Force an optimizer update even if fewer than `accumulation_steps` micro-batches
    /// have been accumulated.
    ///
    /// Gradients are scaled by `1 / current_step` (the actual number of accumulated
    /// micro-batches). If no steps have been accumulated this is a no-op.
    pub fn flush(&mut self) {
        if self.current_step > 0 {
            self.scale_and_step();
        }
    }

    /// Reset the internal step counter without performing an update.
    ///
    /// This also zeros the gradients on all parameters so that previously
    /// accumulated gradients do not leak into the next accumulation window.
    pub fn reset(&mut self) {
        self.current_step = 0;
        self.optimizer.zero_grad();
    }

    /// Return a reference to the inner optimizer.
    pub fn inner(&self) -> &O {
        &self.optimizer
    }

    /// Return a mutable reference to the inner optimizer.
    pub fn inner_mut(&mut self) -> &mut O {
        &mut self.optimizer
    }

    /// Return the number of accumulation steps.
    pub fn accumulation_steps(&self) -> usize {
        self.accumulation_steps
    }

    /// Return the current micro-batch counter (how many steps since the last update).
    pub fn current_step(&self) -> usize {
        self.current_step
    }

    /// Scale gradients by `1/current_step`, run optimizer step, and zero grads.
    fn scale_and_step(&mut self) {
        let n = self.current_step;
        let scale = T::one() / T::from_usize(n);

        for param in &self.parameters {
            if let Some(g) = param.grad() {
                let scaled = &g * scale;
                param.set_grad(scaled);
            }
        }

        self.optimizer.step();
        self.optimizer.zero_grad();
        self.current_step = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optim::SGD;
    use scivex_core::Tensor;

    #[test]
    fn test_construction() {
        let w = Variable::new(Tensor::from_vec(vec![1.0_f64, 2.0], vec![2]).unwrap(), true);
        let params = vec![w.clone()];
        let sgd = SGD::new(params.clone(), 0.01);
        let accum = GradAccumulator::new(sgd, params, 4).unwrap();
        assert_eq!(accum.accumulation_steps(), 4);
        assert_eq!(accum.current_step(), 0);
    }

    #[test]
    fn test_zero_accumulation_steps_errors() {
        let w = Variable::new(Tensor::from_vec(vec![1.0_f64], vec![1]).unwrap(), true);
        let params = vec![w.clone()];
        let sgd = SGD::new(params.clone(), 0.01);
        let result = GradAccumulator::new(sgd, params, 0);
        assert!(result.is_err());
        match result.unwrap_err() {
            NnError::InvalidParameter { name, .. } => {
                assert_eq!(name, "accumulation_steps");
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn test_step_counting() {
        let w = Variable::new(Tensor::from_vec(vec![1.0_f64, 2.0], vec![2]).unwrap(), true);
        let params = vec![w.clone()];
        let sgd = SGD::new(params.clone(), 0.1);
        let mut accum = GradAccumulator::new(sgd, params, 3).unwrap();

        // Simulate 3 micro-batches.
        w.set_grad(Tensor::from_vec(vec![1.0, 1.0], vec![2]).unwrap());
        assert!(!accum.step()); // step 1 of 3

        w.set_grad(Tensor::from_vec(vec![1.0, 1.0], vec![2]).unwrap());
        assert!(!accum.step()); // step 2 of 3

        w.set_grad(Tensor::from_vec(vec![1.0, 1.0], vec![2]).unwrap());
        assert!(accum.step()); // step 3 of 3 — triggers update

        // Counter should be reset.
        assert_eq!(accum.current_step(), 0);
    }

    #[test]
    fn test_gradient_scaling() {
        // Verify that gradients are scaled by 1/N before the optimizer step.
        // Use lr=1.0 so the parameter update equals the (scaled) gradient.
        let w = Variable::new(Tensor::from_vec(vec![0.0_f64], vec![1]).unwrap(), true);
        let params = vec![w.clone()];
        let sgd = SGD::new(params.clone(), 1.0);
        let mut accum = GradAccumulator::new(sgd, params, 2).unwrap();

        // Micro-batch 1: set grad = [6.0]
        w.set_grad(Tensor::from_vec(vec![6.0_f64], vec![1]).unwrap());
        assert!(!accum.step());

        // Micro-batch 2: set grad = [6.0] (accumulated on top → [12.0])
        // But in our design the user is responsible for calling backward which
        // accumulates into the variable. Since we are manually setting, let's
        // simulate accumulated grads by setting the total.
        w.set_grad(Tensor::from_vec(vec![12.0_f64], vec![1]).unwrap());
        assert!(accum.step());

        // Scaled grad = 12.0 / 2 = 6.0
        // new w = 0.0 - 1.0 * 6.0 = -6.0
        let val = w.data().as_slice()[0];
        assert!((val - (-6.0)).abs() < 1e-10, "expected w = -6.0, got {val}");
    }

    #[test]
    fn test_flush_before_full_accumulation() {
        let w = Variable::new(Tensor::from_vec(vec![0.0_f64], vec![1]).unwrap(), true);
        let params = vec![w.clone()];
        let sgd = SGD::new(params.clone(), 1.0);
        let mut accum = GradAccumulator::new(sgd, params, 4).unwrap();

        // Only 1 micro-batch accumulated, then flush.
        w.set_grad(Tensor::from_vec(vec![8.0_f64], vec![1]).unwrap());
        accum.step(); // current_step = 1

        // Actually after step() with current_step < accumulation_steps, current_step is 1.
        // Now flush: scale by 1/1 = 1.0 — but grads were already set and not consumed.
        // Wait: step() didn't trigger update, so grad [8.0] is still there.
        // But step() incremented current_step to 1.  Actually let's re-examine.
        //
        // step() increments current_step to 1 which is < 4, returns false.
        // Grad [8.0] is still on w.
        // Now call flush(): current_step is 1, scale = 1/1 = 1.0, step, zero_grad.
        accum.flush();

        // new w = 0.0 - 1.0 * (8.0 / 1) = -8.0
        let val = w.data().as_slice()[0];
        assert!((val - (-8.0)).abs() < 1e-10, "expected w = -8.0, got {val}");

        // Counter should be reset.
        assert_eq!(accum.current_step(), 0);
    }

    #[test]
    fn test_flush_no_op_when_no_steps() {
        let w = Variable::new(Tensor::from_vec(vec![5.0_f64], vec![1]).unwrap(), true);
        let params = vec![w.clone()];
        let sgd = SGD::new(params.clone(), 1.0);
        let mut accum = GradAccumulator::new(sgd, params, 4).unwrap();

        // Flush with no accumulated steps should be a no-op.
        accum.flush();
        let val = w.data().as_slice()[0];
        assert!(
            (val - 5.0).abs() < 1e-10,
            "expected w unchanged at 5.0, got {val}"
        );
    }
}
