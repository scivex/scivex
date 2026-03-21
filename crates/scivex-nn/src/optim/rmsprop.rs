//! RMSprop optimizer (Hinton, unpublished).

use scivex_core::{Float, Tensor};

use crate::variable::Variable;

use super::Optimizer;

/// RMSprop optimizer — adaptive learning rate using a running average of
/// squared gradients.
pub struct RMSprop<T: Float> {
    params: Vec<Variable<T>>,
    lr: T,
    alpha: T,
    eps: T,
    weight_decay: T,
    momentum: T,
    v: Vec<Option<Tensor<T>>>,
    buf: Vec<Option<Tensor<T>>>,
}

impl<T: Float> RMSprop<T> {
    /// Create a new RMSprop optimizer.
    ///
    /// - `params`: parameters to optimize
    /// - `lr`: learning rate
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_nn::optim::{RMSprop, Optimizer};
    /// # use scivex_nn::variable::Variable;
    /// # use scivex_core::Tensor;
    /// let x = Variable::new(Tensor::from_vec(vec![1.0_f64], vec![1]).unwrap(), true);
    /// let mut optim = RMSprop::new(vec![x.clone()], 0.01);
    /// optim.zero_grad();
    /// ```
    pub fn new(params: Vec<Variable<T>>, lr: T) -> Self {
        let n = params.len();
        Self {
            params,
            lr,
            alpha: T::from_f64(0.99),
            eps: T::from_f64(1e-8),
            weight_decay: T::zero(),
            momentum: T::zero(),
            v: vec![None; n],
            buf: vec![None; n],
        }
    }

    /// Set alpha (smoothing constant, default 0.99).
    pub fn with_alpha(mut self, alpha: T) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set epsilon for numerical stability.
    pub fn with_eps(mut self, eps: T) -> Self {
        self.eps = eps;
        self
    }

    /// Set weight decay (default 0.0).
    pub fn with_weight_decay(mut self, wd: T) -> Self {
        self.weight_decay = wd;
        self
    }

    /// Set momentum (default 0.0).
    pub fn with_momentum(mut self, momentum: T) -> Self {
        self.momentum = momentum;
        self
    }
}

impl<T: Float> Optimizer<T> for RMSprop<T> {
    fn step(&mut self) {
        let lr = self.lr;
        let alpha = self.alpha;
        let eps = self.eps;
        let wd = self.weight_decay;
        let mom = self.momentum;

        for (i, param) in self.params.iter().enumerate() {
            let Some(raw_grad) = param.grad() else {
                continue;
            };

            // Apply L2 weight decay to gradient if requested.
            let grad = if wd > T::zero() {
                let data = param.data();
                raw_grad
                    .zip_map(&data, |g, p| g + wd * p)
                    .expect("grad and param shapes match")
            } else {
                raw_grad
            };

            // Update running average of squared gradients: v = alpha*v + (1-alpha)*g^2
            let new_v = match self.v[i].take() {
                Some(prev_v) => prev_v
                    .zip_map(&grad, |vi, gi| alpha * vi + (T::one() - alpha) * gi * gi)
                    .expect("v and grad shapes match"),
                None => grad.map(|gi| (T::one() - alpha) * gi * gi),
            };

            let data = param.data();

            if mom > T::zero() {
                // Momentum variant: buf = mom*buf + grad / (sqrt(v) + eps)
                let update = grad
                    .zip_map(&new_v, |gi, vi| gi / (vi.sqrt() + eps))
                    .expect("grad and v shapes match");

                let new_buf = match self.buf[i].take() {
                    Some(prev_buf) => prev_buf
                        .zip_map(&update, |bi, ui| mom * bi + ui)
                        .expect("buf and update shapes match"),
                    None => update,
                };

                let new_data = data
                    .zip_map(&new_buf, |p, b| p - lr * b)
                    .expect("param and buf shapes match");
                param.set_data(new_data);

                self.buf[i] = Some(new_buf);
            } else {
                // Standard update: param -= lr * grad / (sqrt(v) + eps)
                let new_data = data
                    .as_slice()
                    .iter()
                    .zip(grad.as_slice().iter())
                    .zip(new_v.as_slice().iter())
                    .map(|((&p, &gi), &vi)| p - lr * gi / (vi.sqrt() + eps))
                    .collect();
                let new_tensor = Tensor::from_vec(new_data, data.shape().to_vec())
                    .expect("update data length matches param shape");
                param.set_data(new_tensor);
            }

            self.v[i] = Some(new_v);
        }
    }

    fn zero_grad(&mut self) {
        for param in &self.params {
            param.zero_grad();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loss;
    use scivex_core::Tensor;

    #[test]
    fn test_rmsprop_converges() {
        // Minimize (x - 3)^2 with RMSprop.
        let x = Variable::new(Tensor::from_vec(vec![0.0_f64], vec![1]).unwrap(), true);
        let target = Variable::new(Tensor::from_vec(vec![3.0], vec![1]).unwrap(), false);

        let mut optim = RMSprop::new(vec![x.clone()], 0.1);

        let initial_loss = loss::mse_loss(&x, &target).unwrap().data().as_slice()[0];

        for _ in 0..100 {
            optim.zero_grad();
            let l = loss::mse_loss(&x, &target).unwrap();
            l.backward();
            optim.step();
        }

        let final_loss = loss::mse_loss(&x, &target).unwrap().data().as_slice()[0];
        assert!(
            final_loss < initial_loss * 0.01,
            "RMSprop did not converge: initial={initial_loss}, final={final_loss}"
        );
    }
}
