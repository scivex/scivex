//! Stochastic Gradient Descent optimizer with optional momentum.

use scivex_core::{Float, Tensor};

use crate::variable::Variable;

use super::Optimizer;

/// SGD optimizer with optional momentum and weight decay.
pub struct SGD<T: Float> {
    params: Vec<Variable<T>>,
    lr: T,
    momentum: T,
    weight_decay: T,
    velocities: Vec<Option<Tensor<T>>>,
}

impl<T: Float> SGD<T> {
    /// Create a new SGD optimizer.
    ///
    /// - `params`: parameters to optimize
    /// - `lr`: learning rate
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_core::Tensor;
    /// # use scivex_nn::variable::Variable;
    /// # use scivex_nn::optim::{SGD, Optimizer};
    /// let w = Variable::new(Tensor::from_vec(vec![1.0_f64, 2.0], vec![2]).unwrap(), true);
    /// let mut sgd = SGD::new(vec![w.clone()], 0.01);
    /// ```
    pub fn new(params: Vec<Variable<T>>, lr: T) -> Self {
        let n = params.len();
        Self {
            params,
            lr,
            momentum: T::zero(),
            weight_decay: T::zero(),
            velocities: vec![None; n],
        }
    }

    /// Set momentum (default 0.0).
    pub fn with_momentum(mut self, momentum: T) -> Self {
        self.momentum = momentum;
        self
    }

    /// Set weight decay (default 0.0).
    pub fn with_weight_decay(mut self, wd: T) -> Self {
        self.weight_decay = wd;
        self
    }
}

impl<T: Float> Optimizer<T> for SGD<T> {
    fn step(&mut self) {
        let lr = self.lr;
        let momentum = self.momentum;
        let wd = self.weight_decay;

        for (i, param) in self.params.iter().enumerate() {
            let Some(grad) = param.grad() else {
                continue;
            };

            let data = param.data();
            let mut grad_with_wd = if wd > T::zero() {
                // L2 regularization: grad += wd * param
                grad.zip_map(&data, |g, p| g + wd * p)
                    .expect("grad and param shapes match")
            } else {
                grad
            };

            if momentum > T::zero() {
                let v = match self.velocities[i].take() {
                    Some(prev_v) => {
                        // v = momentum * v_prev + grad
                        prev_v
                            .zip_map(&grad_with_wd, |vi, gi| momentum * vi + gi)
                            .expect("velocity and grad shapes match")
                    }
                    None => grad_with_wd.clone(),
                };
                grad_with_wd = v.clone();
                self.velocities[i] = Some(v);
            }

            // param = param - lr * update
            let new_data = data
                .zip_map(&grad_with_wd, |p, g| p - lr * g)
                .expect("param and grad shapes match");

            // Update the variable's data in-place via the inner node.
            // We create a new variable with updated data and swap.
            update_param_data(param, new_data);
        }
    }

    fn zero_grad(&mut self) {
        for param in &self.params {
            param.zero_grad();
        }
    }
}

/// Helper to update a parameter's data in-place through the `Variable` API.
///
/// This reaches into the `Rc<RefCell<Node>>` to replace the data tensor.
pub(crate) fn update_param_data<T: Float>(param: &Variable<T>, new_data: Tensor<T>) {
    // Variable exposes no set_data, so we use a workaround: we directly
    // access the RefCell. Since Variable's inner is pub(crate), we add
    // a helper method.
    param.set_data(new_data);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loss;
    use scivex_core::Tensor;

    #[test]
    fn test_sgd_reduces_loss() {
        // Simple: minimize (x - 3)^2
        let x = Variable::new(Tensor::from_vec(vec![0.0_f64], vec![1]).unwrap(), true);
        let target = Variable::new(Tensor::from_vec(vec![3.0], vec![1]).unwrap(), false);

        let mut optim = SGD::new(vec![x.clone()], 0.1);

        let initial_loss;
        {
            let l = loss::mse_loss(&x, &target).unwrap();
            initial_loss = l.data().as_slice()[0];
        }

        for _ in 0..10 {
            optim.zero_grad();
            let l = loss::mse_loss(&x, &target).unwrap();
            l.backward();
            optim.step();
        }

        let final_loss = loss::mse_loss(&x, &target).unwrap().data().as_slice()[0];
        assert!(final_loss < initial_loss, "loss did not decrease");
    }
}
