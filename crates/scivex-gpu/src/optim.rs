//! GPU on-device optimizers — parameter updates stay on the GPU.
//!
//! Instead of downloading params/grads to CPU, computing updates, and
//! re-uploading, these optimizers run WGSL compute shaders that update
//! parameter buffers in-place on the device.

use std::sync::Arc;

use crate::device::GpuDevice;
use crate::error::Result;
use crate::tensor::GpuTensor;

// ---------------------------------------------------------------------------
// WGSL shaders for on-device parameter updates
// ---------------------------------------------------------------------------

const SGD_SHADER: &str = r"
struct Hyperparams {
    lr: f32,
    momentum: f32,
    weight_decay: f32,
    _pad: f32,
}

@group(0) @binding(0) var<storage, read_write> params: array<f32>;
@group(0) @binding(1) var<storage, read> grads: array<f32>;
@group(0) @binding(2) var<storage, read_write> velocity: array<f32>;
@group(0) @binding(3) var<uniform> hp: Hyperparams;

@compute @workgroup_size(256)
fn sgd_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= arrayLength(&params) { return; }

    var g = grads[idx];

    // L2 weight decay: g += wd * p
    if hp.weight_decay > 0.0 {
        g = g + hp.weight_decay * params[idx];
    }

    // Momentum: v = momentum * v + g; update = v
    if hp.momentum > 0.0 {
        let v = hp.momentum * velocity[idx] + g;
        velocity[idx] = v;
        g = v;
    }

    // SGD update: p = p - lr * g
    params[idx] = params[idx] - hp.lr * g;
}
";

const ADAM_SHADER: &str = r"
struct Hyperparams {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    bc1: f32,      // 1 - beta1^t (bias correction for first moment)
    bc2: f32,      // 1 - beta2^t (bias correction for second moment)
    _pad0: f32,
    _pad1: f32,
}

@group(0) @binding(0) var<storage, read_write> params: array<f32>;
@group(0) @binding(1) var<storage, read> grads: array<f32>;
@group(0) @binding(2) var<storage, read_write> m: array<f32>;
@group(0) @binding(3) var<storage, read_write> v: array<f32>;
@group(0) @binding(4) var<uniform> hp: Hyperparams;

@compute @workgroup_size(256)
fn adam_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= arrayLength(&params) { return; }

    let g = grads[idx];

    // Update biased first moment: m = b1*m + (1-b1)*g
    let new_m = hp.beta1 * m[idx] + (1.0 - hp.beta1) * g;
    m[idx] = new_m;

    // Update biased second moment: v = b2*v + (1-b2)*g^2
    let new_v = hp.beta2 * v[idx] + (1.0 - hp.beta2) * g * g;
    v[idx] = new_v;

    // Bias-corrected estimates
    let m_hat = new_m / hp.bc1;
    let v_hat = new_v / hp.bc2;

    // Adam update: p = p - lr * m_hat / (sqrt(v_hat) + eps)
    params[idx] = params[idx] - hp.lr * m_hat / (sqrt(v_hat) + hp.eps);
}
";

// ---------------------------------------------------------------------------
// GPU SGD optimizer
// ---------------------------------------------------------------------------

/// GPU on-device SGD optimizer with optional momentum and weight decay.
///
/// All parameter updates run entirely on the GPU — no CPU roundtrip.
pub struct GpuSGD {
    device: GpuDevice,
    lr: f32,
    momentum: f32,
    weight_decay: f32,
    velocities: Vec<Option<GpuTensor>>,
}

impl GpuSGD {
    /// Create a new GPU SGD optimizer.
    pub fn new(device: &GpuDevice, n_params: usize, lr: f32) -> Self {
        Self {
            device: device.clone(),
            lr,
            momentum: 0.0,
            weight_decay: 0.0,
            velocities: (0..n_params).map(|_| None).collect(),
        }
    }

    /// Set momentum (default 0.0).
    pub fn with_momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
        self
    }

    /// Set weight decay (default 0.0).
    pub fn with_weight_decay(mut self, wd: f32) -> Self {
        self.weight_decay = wd;
        self
    }

    /// Perform one SGD step — updates `params` in-place on the GPU.
    ///
    /// `params` and `grads` must be paired slices of the same length.
    pub fn step(&mut self, params: &mut [GpuTensor], grads: &[GpuTensor]) -> Result<()> {
        let dev = &self.device;

        for (i, (param, grad)) in params.iter_mut().zip(grads.iter()).enumerate() {
            let n = param.numel();

            // Lazily create zero-initialized velocity buffer.
            if self.velocities[i].is_none() && self.momentum > 0.0 {
                self.velocities[i] = Some(crate::ops::fill(dev, param.shape().to_vec(), 0.0)?);
            }

            // Use a dummy velocity if momentum is 0.
            let vel = if let Some(v) = &self.velocities[i] {
                Arc::clone(&v.buffer)
            } else {
                let dummy = crate::ops::fill(dev, param.shape().to_vec(), 0.0)?;
                Arc::clone(&dummy.buffer)
            };

            let hp_data: [f32; 4] = [self.lr, self.momentum, self.weight_decay, 0.0];
            let hp_buf = {
                use wgpu::util::DeviceExt;
                dev.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: None,
                        contents: bytemuck::cast_slice(&hp_data),
                        usage: wgpu::BufferUsages::UNIFORM,
                    })
            };

            let module = dev
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("sgd_optim"),
                    source: wgpu::ShaderSource::Wgsl(SGD_SHADER.into()),
                });

            let pipeline = dev
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: None,
                    layout: None,
                    module: &module,
                    entry_point: Some("sgd_step"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                });

            let bind_group = dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: param.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: grad.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: vel.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: hp_buf.as_entire_binding(),
                    },
                ],
            });

            let mut encoder = dev
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(n.div_ceil(256) as u32, 1, 1);
            }
            dev.queue.submit(Some(encoder.finish()));
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// GPU Adam optimizer
// ---------------------------------------------------------------------------

/// GPU on-device Adam optimizer — all updates run on the GPU.
pub struct GpuAdam {
    device: GpuDevice,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    t: usize,
    first_moments: Vec<Option<GpuTensor>>,
    second_moments: Vec<Option<GpuTensor>>,
}

impl GpuAdam {
    /// Create a new GPU Adam optimizer.
    pub fn new(device: &GpuDevice, n_params: usize, lr: f32) -> Self {
        Self {
            device: device.clone(),
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            t: 0,
            first_moments: (0..n_params).map(|_| None).collect(),
            second_moments: (0..n_params).map(|_| None).collect(),
        }
    }

    /// Set beta1 (first moment decay).
    pub fn with_beta1(mut self, beta1: f32) -> Self {
        self.beta1 = beta1;
        self
    }

    /// Set beta2 (second moment decay).
    pub fn with_beta2(mut self, beta2: f32) -> Self {
        self.beta2 = beta2;
        self
    }

    /// Set epsilon for numerical stability.
    pub fn with_eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    /// Perform one Adam step — updates `params` in-place on the GPU.
    pub fn step(&mut self, params: &mut [GpuTensor], grads: &[GpuTensor]) -> Result<()> {
        self.t += 1;
        let dev = &self.device;

        #[allow(clippy::cast_possible_wrap)]
        let t_i32 = self.t as i32;
        let bc1 = 1.0 - self.beta1.powi(t_i32);
        let bc2 = 1.0 - self.beta2.powi(t_i32);

        for (i, (param, grad)) in params.iter_mut().zip(grads.iter()).enumerate() {
            let n = param.numel();

            // Lazily create zero-initialized moment buffers.
            if self.first_moments[i].is_none() {
                self.first_moments[i] = Some(crate::ops::fill(dev, param.shape().to_vec(), 0.0)?);
            }
            if self.second_moments[i].is_none() {
                self.second_moments[i] = Some(crate::ops::fill(dev, param.shape().to_vec(), 0.0)?);
            }

            let m_buf = Arc::clone(
                &self.first_moments[i]
                    .as_ref()
                    .expect("just initialized")
                    .buffer,
            );
            let v_buf = Arc::clone(
                &self.second_moments[i]
                    .as_ref()
                    .expect("just initialized")
                    .buffer,
            );

            // 8 floats: lr, beta1, beta2, eps, bc1, bc2, pad, pad
            let hp_data: [f32; 8] = [
                self.lr, self.beta1, self.beta2, self.eps, bc1, bc2, 0.0, 0.0,
            ];
            let hp_buf = {
                use wgpu::util::DeviceExt;
                dev.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: None,
                        contents: bytemuck::cast_slice(&hp_data),
                        usage: wgpu::BufferUsages::UNIFORM,
                    })
            };

            let module = dev
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("adam_optim"),
                    source: wgpu::ShaderSource::Wgsl(ADAM_SHADER.into()),
                });

            let pipeline = dev
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: None,
                    layout: None,
                    module: &module,
                    entry_point: Some("adam_step"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                });

            let bind_group = dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: param.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: grad.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: m_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: v_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: hp_buf.as_entire_binding(),
                    },
                ],
            });

            let mut encoder = dev
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(n.div_ceil(256) as u32, 1, 1);
            }
            dev.queue.submit(Some(encoder.finish()));
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::GpuDevice;

    fn get_device() -> Option<GpuDevice> {
        GpuDevice::new().ok()
    }

    #[test]
    fn test_gpu_sgd_reduces_params() {
        let Some(dev) = get_device() else { return };

        // param = [10.0], grad = [2.0], lr = 0.1
        // After step: 10.0 - 0.1 * 2.0 = 9.8
        let mut params = vec![GpuTensor::from_slice(&dev, &[10.0_f32], vec![1]).unwrap()];
        let grads = vec![GpuTensor::from_slice(&dev, &[2.0_f32], vec![1]).unwrap()];

        let mut sgd = GpuSGD::new(&dev, 1, 0.1);
        sgd.step(&mut params, &grads).unwrap();

        let result = params[0].to_tensor().unwrap();
        assert!((result.as_slice()[0] - 9.8).abs() < 1e-5);
    }

    #[test]
    fn test_gpu_adam_reduces_params() {
        let Some(dev) = get_device() else { return };

        let mut params = vec![GpuTensor::from_slice(&dev, &[5.0_f32], vec![1]).unwrap()];
        let grads = vec![GpuTensor::from_slice(&dev, &[1.0_f32], vec![1]).unwrap()];

        let mut adam = GpuAdam::new(&dev, 1, 0.1);

        // Run a few steps — param should decrease.
        let initial = params[0].to_tensor().unwrap().as_slice()[0];
        for _ in 0..5 {
            adam.step(&mut params, &grads).unwrap();
        }
        let final_val = params[0].to_tensor().unwrap().as_slice()[0];
        assert!(
            final_val < initial,
            "Adam did not reduce params: {initial} -> {final_val}"
        );
    }

    #[test]
    fn test_gpu_sgd_with_momentum() {
        let Some(dev) = get_device() else { return };

        let mut params = vec![GpuTensor::from_slice(&dev, &[10.0_f32], vec![1]).unwrap()];
        let grads = vec![GpuTensor::from_slice(&dev, &[1.0_f32], vec![1]).unwrap()];

        let mut sgd = GpuSGD::new(&dev, 1, 0.1).with_momentum(0.9);

        let initial = params[0].to_tensor().unwrap().as_slice()[0];
        for _ in 0..10 {
            sgd.step(&mut params, &grads).unwrap();
        }
        let final_val = params[0].to_tensor().unwrap().as_slice()[0];
        assert!(
            final_val < initial,
            "SGD+momentum did not reduce params: {initial} -> {final_val}"
        );
    }
}
