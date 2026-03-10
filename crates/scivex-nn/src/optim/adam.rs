//! Adam optimizer (Kingma & Ba, 2014).

use scivex_core::{Float, Tensor};

use crate::variable::Variable;

use super::Optimizer;

/// Adam optimizer with bias-corrected first and second moment estimates.
pub struct Adam<T: Float> {
    params: Vec<Variable<T>>,
    lr: T,
    beta1: T,
    beta2: T,
    eps: T,
    t: usize,
    m: Vec<Option<Tensor<T>>>,
    v: Vec<Option<Tensor<T>>>,
}

impl<T: Float> Adam<T> {
    /// Create a new Adam optimizer with default hyperparameters.
    ///
    /// - `params`: parameters to optimize
    /// - `lr`: learning rate (default suggestion: 0.001)
    pub fn new(params: Vec<Variable<T>>, lr: T) -> Self {
        let n = params.len();
        Self {
            params,
            lr,
            beta1: T::from_f64(0.9),
            beta2: T::from_f64(0.999),
            eps: T::from_f64(1e-8),
            t: 0,
            m: vec![None; n],
            v: vec![None; n],
        }
    }

    /// Set beta1 (first moment decay).
    pub fn with_beta1(mut self, beta1: T) -> Self {
        self.beta1 = beta1;
        self
    }

    /// Set beta2 (second moment decay).
    pub fn with_beta2(mut self, beta2: T) -> Self {
        self.beta2 = beta2;
        self
    }

    /// Set epsilon for numerical stability.
    pub fn with_eps(mut self, eps: T) -> Self {
        self.eps = eps;
        self
    }
}

impl<T: Float> Optimizer<T> for Adam<T> {
    fn step(&mut self) {
        self.t += 1;
        let lr = self.lr;
        let b1 = self.beta1;
        let b2 = self.beta2;
        let eps = self.eps;
        let t = self.t;

        // Bias correction factors.
        let bc1 = T::one() - b1.powf(T::from_usize(t));
        let bc2 = T::one() - b2.powf(T::from_usize(t));

        for (i, param) in self.params.iter().enumerate() {
            let Some(grad) = param.grad() else {
                continue;
            };

            // Update first moment: m = b1*m + (1-b1)*g
            let new_m = match self.m[i].take() {
                Some(prev_m) => prev_m
                    .zip_map(&grad, |mi, gi| b1 * mi + (T::one() - b1) * gi)
                    .expect("moment and grad shapes match"),
                None => grad.map(|gi| (T::one() - b1) * gi),
            };

            // Update second moment: v = b2*v + (1-b2)*g^2
            let new_v = match self.v[i].take() {
                Some(prev_v) => prev_v
                    .zip_map(&grad, |vi, gi| b2 * vi + (T::one() - b2) * gi * gi)
                    .expect("moment and grad shapes match"),
                None => grad.map(|gi| (T::one() - b2) * gi * gi),
            };

            // Bias-corrected estimates.
            let m_hat = new_m.map(|mi| mi / bc1);
            let v_hat = new_v.map(|vi| vi / bc2);

            // Update: param = param - lr * m_hat / (sqrt(v_hat) + eps)
            let data = param.data();
            let new_data = data
                .as_slice()
                .iter()
                .zip(m_hat.as_slice().iter())
                .zip(v_hat.as_slice().iter())
                .map(|((&p, &mi), &vi)| p - lr * mi / (vi.sqrt() + eps))
                .collect();
            let new_tensor = Tensor::from_vec(new_data, data.shape().to_vec())
                .expect("update data length matches param shape");
            param.set_data(new_tensor);

            self.m[i] = Some(new_m);
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
    fn test_adam_converges() {
        // Simple: minimize (x - 3)^2 with Adam
        let x = Variable::new(Tensor::from_vec(vec![0.0_f64], vec![1]).unwrap(), true);
        let target = Variable::new(Tensor::from_vec(vec![3.0], vec![1]).unwrap(), false);

        let mut optim = Adam::new(vec![x.clone()], 0.1);

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
            "Adam did not converge: initial={initial_loss}, final={final_loss}"
        );
    }
}
