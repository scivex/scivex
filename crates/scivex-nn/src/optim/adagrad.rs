//! Adagrad optimizer (Duchi et al., 2011).

use scivex_core::{Float, Tensor};

use crate::variable::Variable;

use super::Optimizer;

/// Adagrad optimizer — adapts the learning rate per-parameter using the
/// accumulated history of squared gradients.
pub struct Adagrad<T: Float> {
    params: Vec<Variable<T>>,
    lr: T,
    eps: T,
    weight_decay: T,
    sum_sq: Vec<Option<Tensor<T>>>,
}

impl<T: Float> Adagrad<T> {
    /// Create a new Adagrad optimizer.
    ///
    /// - `params`: parameters to optimize
    /// - `lr`: learning rate
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_nn::optim::{Adagrad, Optimizer};
    /// # use scivex_nn::variable::Variable;
    /// # use scivex_core::Tensor;
    /// let x = Variable::new(Tensor::from_vec(vec![1.0_f64], vec![1]).unwrap(), true);
    /// let mut optim = Adagrad::new(vec![x.clone()], 0.1);
    /// optim.zero_grad();
    /// ```
    pub fn new(params: Vec<Variable<T>>, lr: T) -> Self {
        let n = params.len();
        Self {
            params,
            lr,
            eps: T::from_f64(1e-10),
            weight_decay: T::zero(),
            sum_sq: vec![None; n],
        }
    }

    /// Set epsilon for numerical stability (default 1e-10).
    pub fn with_eps(mut self, eps: T) -> Self {
        self.eps = eps;
        self
    }

    /// Set weight decay (default 0.0).
    pub fn with_weight_decay(mut self, wd: T) -> Self {
        self.weight_decay = wd;
        self
    }
}

impl<T: Float> Optimizer<T> for Adagrad<T> {
    fn step(&mut self) {
        let lr = self.lr;
        let eps = self.eps;
        let wd = self.weight_decay;

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

            // Accumulate squared gradients: sum_sq += grad^2
            let new_sum_sq = match self.sum_sq[i].take() {
                Some(prev) => prev
                    .zip_map(&grad, |si, gi| si + gi * gi)
                    .expect("sum_sq and grad shapes match"),
                None => grad.map(|gi| gi * gi),
            };

            // Update: param -= lr * grad / (sqrt(sum_sq) + eps)
            let data = param.data();
            let new_data = data
                .as_slice()
                .iter()
                .zip(grad.as_slice().iter())
                .zip(new_sum_sq.as_slice().iter())
                .map(|((&p, &gi), &si)| p - lr * gi / (si.sqrt() + eps))
                .collect();
            let new_tensor = Tensor::from_vec(new_data, data.shape().to_vec())
                .expect("update data length matches param shape");
            param.set_data(new_tensor);

            self.sum_sq[i] = Some(new_sum_sq);
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
    fn test_adagrad_converges() {
        // Minimize (x - 3)^2 with Adagrad.
        let x = Variable::new(Tensor::from_vec(vec![0.0_f64], vec![1]).unwrap(), true);
        let target = Variable::new(Tensor::from_vec(vec![3.0], vec![1]).unwrap(), false);

        let mut optim = Adagrad::new(vec![x.clone()], 0.5);

        let initial_loss = loss::mse_loss(&x, &target).unwrap().data().as_slice()[0];

        for _ in 0..200 {
            optim.zero_grad();
            let l = loss::mse_loss(&x, &target).unwrap();
            l.backward();
            optim.step();
        }

        let final_loss = loss::mse_loss(&x, &target).unwrap().data().as_slice()[0];
        assert!(
            final_loss < initial_loss * 0.01,
            "Adagrad did not converge: initial={initial_loss}, final={final_loss}"
        );
    }
}
