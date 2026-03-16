//! GPU neural network layers.

use scivex_core::Tensor;
use scivex_core::random::Rng;
use scivex_gpu::{GpuDevice, GpuTensor};

use super::variable::GpuVariable;
use crate::error::Result;

/// A GPU neural network layer.
pub trait GpuLayer {
    /// Forward pass on GPU.
    fn forward(&self, x: &GpuVariable) -> Result<GpuVariable>;

    /// Return all learnable parameters.
    fn parameters(&self) -> Vec<GpuVariable>;

    /// Set training mode.
    fn train(&mut self);

    /// Set evaluation mode.
    fn eval(&mut self);
}

// ---------------------------------------------------------------------------
// Helper: transpose a 2-D GpuVariable within the computation graph
// ---------------------------------------------------------------------------

/// Transpose a 2-D [`GpuVariable`], creating a new variable connected in the
/// autograd graph so that gradients flow back through the transpose.
fn gpu_transpose_var(x: &GpuVariable) -> Result<GpuVariable> {
    let transposed = x.with_data(scivex_gpu::ops::transpose)?;
    Ok(GpuVariable::from_op(
        transposed,
        vec![x.clone()],
        Box::new(|grad: &GpuTensor| {
            // Gradient of transpose is the transpose of the gradient.
            vec![scivex_gpu::ops::transpose(grad).expect("transpose in backward")]
        }),
    ))
}

/// Perform matrix multiplication of two [`GpuVariable`]s on the GPU,
/// recording the operation in the autograd graph.
///
/// `a` must have shape `[m, k]` and `b` must have shape `[k, n]`.
/// Returns a variable of shape `[m, n]`.
fn gpu_matmul_var(a: &GpuVariable, b: &GpuVariable) -> Result<GpuVariable> {
    let result = {
        let a_ref = a.clone();
        let b_ref = b.clone();
        a_ref.with_data(|a_data| b_ref.with_data(|b_data| scivex_gpu::ops::matmul(a_data, b_data)))
    }?;

    let a_clone = a.clone();
    let b_clone = b.clone();

    Ok(GpuVariable::from_op(
        result,
        vec![a.clone(), b.clone()],
        Box::new(move |grad: &GpuTensor| {
            // grad_a = grad @ b^T
            let bt = b_clone
                .with_data(scivex_gpu::ops::transpose)
                .expect("transpose b in backward");
            let grad_a = scivex_gpu::ops::matmul(grad, &bt).expect("matmul grad_a in backward");

            // grad_b = a^T @ grad
            let at = a_clone
                .with_data(scivex_gpu::ops::transpose)
                .expect("transpose a in backward");
            let grad_b = scivex_gpu::ops::matmul(&at, grad).expect("matmul grad_b in backward");

            vec![grad_a, grad_b]
        }),
    ))
}

/// Add a bias vector to every row of a 2-D [`GpuVariable`] in the autograd graph.
///
/// `input` has shape `[batch, features]` and `bias` has shape `[features]`.
/// Broadcasting is performed by replicating the bias across the batch dimension.
fn gpu_add_bias_var(input: &GpuVariable, bias: &GpuVariable) -> Result<GpuVariable> {
    // Perform bias addition on CPU since we don't have a dedicated broadcast-add
    // GPU kernel. Download, compute, re-upload.
    let input_cpu = input.data_cpu()?;
    let bias_cpu = bias.data_cpu()?;

    let batch = input_cpu.shape()[0];
    let features = input_cpu.shape()[1];
    let in_slice = input_cpu.as_slice();
    let b_slice = bias_cpu.as_slice();

    let mut out = Vec::with_capacity(batch * features);
    for row in 0..batch {
        for col in 0..features {
            out.push(in_slice[row * features + col] + b_slice[col]);
        }
    }

    let out_tensor = Tensor::from_vec(out, vec![batch, features]).expect("bias add output shape");
    let device = input.device();
    let out_gpu = GpuTensor::from_tensor(&device, &out_tensor);

    let batch_size = batch;
    Ok(GpuVariable::from_op(
        out_gpu,
        vec![input.clone(), bias.clone()],
        Box::new(move |grad: &GpuTensor| {
            // grad_input = grad (same shape)
            // grad_bias = sum over batch dimension of grad
            let grad_cpu = grad.to_tensor().expect("download grad in bias backward");
            let g_slice = grad_cpu.as_slice();
            let feats = grad_cpu.shape()[1];

            // grad_input is just grad itself — re-upload
            let dev = grad.device().clone();
            let grad_input = GpuTensor::from_tensor(&dev, &grad_cpu);

            // grad_bias: sum rows
            let mut bias_grad = vec![0.0f32; feats];
            for row in 0..batch_size {
                for col in 0..feats {
                    bias_grad[col] += g_slice[row * feats + col];
                }
            }
            let bias_grad_tensor =
                Tensor::from_vec(bias_grad, vec![feats]).expect("bias grad shape");
            let grad_bias = GpuTensor::from_tensor(&dev, &bias_grad_tensor);

            vec![grad_input, grad_bias]
        }),
    ))
}

// ---------------------------------------------------------------------------
// GpuLinear
// ---------------------------------------------------------------------------

/// GPU fully-connected linear layer: `y = x @ W^T + b`.
///
/// Uses Kaiming uniform initialization (same as the CPU [`Linear`](crate::layer::Linear) layer).
/// All data is `f32` because most GPUs lack native `f64` support.
pub struct GpuLinear {
    weight: GpuVariable,
    bias: Option<GpuVariable>,
}

impl GpuLinear {
    /// Create a new GPU linear layer.
    ///
    /// - `device`: the GPU device to allocate on
    /// - `in_features`: size of each input sample
    /// - `out_features`: size of each output sample
    /// - `use_bias`: whether to include a bias term
    /// - `rng`: random number generator for initialization
    pub fn new(
        device: &GpuDevice,
        in_features: usize,
        out_features: usize,
        use_bias: bool,
        rng: &mut Rng,
    ) -> Self {
        let w_data = crate::init::kaiming_uniform::<f32>(&[out_features, in_features], rng);
        let w_gpu = GpuTensor::from_tensor(device, &w_data);
        let weight = GpuVariable::new(w_gpu, true);

        let bias = if use_bias {
            let b_data = Tensor::<f32>::zeros(vec![out_features]);
            let b_gpu = GpuTensor::from_tensor(device, &b_data);
            Some(GpuVariable::new(b_gpu, true))
        } else {
            None
        };

        Self { weight, bias }
    }

    /// Return the weight variable.
    pub fn weight(&self) -> &GpuVariable {
        &self.weight
    }

    /// Return the bias variable, if present.
    pub fn bias(&self) -> Option<&GpuVariable> {
        self.bias.as_ref()
    }
}

impl GpuLayer for GpuLinear {
    fn forward(&self, x: &GpuVariable) -> Result<GpuVariable> {
        // y = x @ W^T + b
        // 1. Transpose weight (connected in the graph so gradients flow back).
        let wt_var = gpu_transpose_var(&self.weight)?;

        // 2. Matrix multiply: x [batch, in] @ W^T [in, out] = [batch, out].
        let y = gpu_matmul_var(x, &wt_var)?;

        // 3. Add bias if present.
        match &self.bias {
            Some(b) => Ok(gpu_add_bias_var(&y, b)?),
            None => Ok(y),
        }
    }

    fn parameters(&self) -> Vec<GpuVariable> {
        let mut params = vec![self.weight.clone()];
        if let Some(b) = &self.bias {
            params.push(b.clone());
        }
        params
    }

    fn train(&mut self) {}
    fn eval(&mut self) {}
}
