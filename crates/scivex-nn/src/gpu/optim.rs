//! GPU optimizers for training neural networks.
//!
//! Optimizers download parameters and gradients to CPU, compute updates, and
//! re-upload the new values. This is correct and simple; a future version can
//! perform the update entirely on GPU for better performance.

use scivex_core::Tensor;
use scivex_gpu::GpuTensor;

use super::variable::GpuVariable;

/// A GPU optimizer updates model parameters using their gradients.
pub trait GpuOptimizer {
    /// Perform a single optimization step (parameter update).
    fn step(&mut self);

    /// Reset all parameter gradients to zero.
    fn zero_grad(&mut self);
}

// ---------------------------------------------------------------------------
// GpuSGD
// ---------------------------------------------------------------------------

/// GPU Stochastic Gradient Descent with optional momentum and weight decay.
pub struct GpuSGD {
    params: Vec<GpuVariable>,
    lr: f32,
    momentum: f32,
    weight_decay: f32,
    velocities: Vec<Option<Tensor<f32>>>,
}

impl GpuSGD {
    /// Create a new GPU SGD optimizer.
    pub fn new(params: Vec<GpuVariable>, lr: f32) -> Self {
        let n = params.len();
        Self {
            params,
            lr,
            momentum: 0.0,
            weight_decay: 0.0,
            velocities: vec![None; n],
        }
    }

    /// Set momentum (default 0.0).
    #[must_use]
    pub fn with_momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
        self
    }

    /// Set weight decay (default 0.0).
    #[must_use]
    pub fn with_weight_decay(mut self, wd: f32) -> Self {
        self.weight_decay = wd;
        self
    }
}

impl GpuOptimizer for GpuSGD {
    fn step(&mut self) {
        let lr = self.lr;
        let momentum = self.momentum;
        let wd = self.weight_decay;

        for (i, param) in self.params.iter().enumerate() {
            let grad_result = param.grad_cpu();
            let Some(Ok(grad)) = grad_result else {
                continue;
            };

            let data = param.data_cpu().expect("download param data");
            let device = param.device();

            // Apply weight decay: grad += wd * param
            let mut update = if wd > 0.0 {
                grad.zip_map(&data, |g, p| g + wd * p)
                    .expect("grad and param shapes match")
            } else {
                grad
            };

            // Apply momentum
            if momentum > 0.0 {
                let v = match self.velocities[i].take() {
                    Some(prev_v) => prev_v
                        .zip_map(&update, |vi, gi| momentum * vi + gi)
                        .expect("velocity and update shapes match"),
                    None => update.clone(),
                };
                update = v.clone();
                self.velocities[i] = Some(v);
            }

            // param = param - lr * update
            let new_data = data
                .zip_map(&update, |p, g| p - lr * g)
                .expect("param and update shapes match");

            let new_gpu = GpuTensor::from_tensor(&device, &new_data);
            param.set_data(new_gpu);
        }
    }

    fn zero_grad(&mut self) {
        for param in &self.params {
            param.zero_grad();
        }
    }
}

// ---------------------------------------------------------------------------
// GpuAdam
// ---------------------------------------------------------------------------

/// GPU Adam optimizer (Kingma & Ba, 2014) with bias-corrected moment estimates.
pub struct GpuAdam {
    params: Vec<GpuVariable>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    t: usize,
    m: Vec<Option<Tensor<f32>>>,
    v: Vec<Option<Tensor<f32>>>,
}

impl GpuAdam {
    /// Create a new GPU Adam optimizer.
    pub fn new(params: Vec<GpuVariable>, lr: f32) -> Self {
        let n = params.len();
        Self {
            params,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            t: 0,
            m: vec![None; n],
            v: vec![None; n],
        }
    }

    /// Set beta1 (first moment decay).
    #[must_use]
    pub fn with_beta1(mut self, beta1: f32) -> Self {
        self.beta1 = beta1;
        self
    }

    /// Set beta2 (second moment decay).
    #[must_use]
    pub fn with_beta2(mut self, beta2: f32) -> Self {
        self.beta2 = beta2;
        self
    }

    /// Set epsilon for numerical stability.
    #[must_use]
    pub fn with_eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }
}

impl GpuOptimizer for GpuAdam {
    fn step(&mut self) {
        self.t += 1;
        let lr = self.lr;
        let b1 = self.beta1;
        let b2 = self.beta2;
        let eps = self.eps;
        let t = self.t;

        #[allow(clippy::cast_possible_wrap)]
        let t_i32 = t as i32;
        let bc1 = 1.0 - b1.powi(t_i32);
        let bc2 = 1.0 - b2.powi(t_i32);

        for (i, param) in self.params.iter().enumerate() {
            let grad_result = param.grad_cpu();
            let Some(Ok(grad)) = grad_result else {
                continue;
            };

            // Update first moment: m = b1*m + (1-b1)*g
            let new_m = match self.m[i].take() {
                Some(prev_m) => prev_m
                    .zip_map(&grad, |mi, gi| b1 * mi + (1.0 - b1) * gi)
                    .expect("moment shapes match"),
                None => grad.map(|gi| (1.0 - b1) * gi),
            };

            // Update second moment: v = b2*v + (1-b2)*g^2
            let new_v = match self.v[i].take() {
                Some(prev_v) => prev_v
                    .zip_map(&grad, |vi, gi| b2 * vi + (1.0 - b2) * gi * gi)
                    .expect("moment shapes match"),
                None => grad.map(|gi| (1.0 - b2) * gi * gi),
            };

            // Bias-corrected estimates
            let m_hat = new_m.map(|mi| mi / bc1);
            let v_hat = new_v.map(|vi| vi / bc2);

            // param = param - lr * m_hat / (sqrt(v_hat) + eps)
            let data = param.data_cpu().expect("download param data");
            let new_data: Vec<f32> = data
                .as_slice()
                .iter()
                .zip(m_hat.as_slice().iter())
                .zip(v_hat.as_slice().iter())
                .map(|((&p, &mi), &vi)| p - lr * mi / (vi.sqrt() + eps))
                .collect();

            let new_tensor = Tensor::from_vec(new_data, data.shape().to_vec())
                .expect("update shape matches param");
            let device = param.device();
            let new_gpu = GpuTensor::from_tensor(&device, &new_tensor);
            param.set_data(new_gpu);

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
