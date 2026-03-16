//! GPU activation functions with automatic differentiation.
//!
//! Each function computes a forward pass on the GPU and records a backward
//! closure for reverse-mode autodiff via [`GpuVariable::backward`].

use scivex_gpu::GpuTensor;

use super::variable::GpuVariable;

/// GPU ReLU activation: `max(0, x)`.
///
/// Backward: `grad * (input > 0 ? 1 : 0)`.
pub fn gpu_relu(x: &GpuVariable) -> GpuVariable {
    let input_cpu = x.data_cpu().expect("download input for relu backward");
    let device = x.device();

    let result = x.with_data(|data| scivex_gpu::ops::relu(data).expect("gpu relu forward"));

    GpuVariable::from_op(
        result,
        vec![x.clone()],
        Box::new(move |grad: &GpuTensor| {
            let input_gpu = GpuTensor::from_tensor(&device, &input_cpu);
            vec![scivex_gpu::ops::relu_backward(grad, &input_gpu).expect("relu backward")]
        }),
    )
}

/// GPU Sigmoid activation: `1 / (1 + exp(-x))`.
///
/// Backward: `grad * out * (1 - out)`.
pub fn gpu_sigmoid(x: &GpuVariable) -> GpuVariable {
    let result = x.with_data(|data| scivex_gpu::ops::sigmoid(data).expect("gpu sigmoid forward"));

    let out_cpu = result
        .to_tensor()
        .expect("download sigmoid output for backward");
    let device = x.device();

    GpuVariable::from_op(
        result,
        vec![x.clone()],
        Box::new(move |grad: &GpuTensor| {
            let out = GpuTensor::from_tensor(&device, &out_cpu);
            let ones =
                scivex_gpu::ops::fill(&device, out_cpu.shape().to_vec(), 1.0).expect("fill ones");
            let one_minus_out = scivex_gpu::ops::sub(&ones, &out).expect("1 - sigmoid");
            let deriv = scivex_gpu::ops::mul(&out, &one_minus_out).expect("sigmoid * (1-sigmoid)");
            vec![scivex_gpu::ops::mul(grad, &deriv).expect("grad * deriv")]
        }),
    )
}

/// GPU Tanh activation.
///
/// Backward: `grad * (1 - out^2)`.
pub fn gpu_tanh(x: &GpuVariable) -> GpuVariable {
    let result = x.with_data(|data| scivex_gpu::ops::tanh_op(data).expect("gpu tanh forward"));

    let out_cpu = result
        .to_tensor()
        .expect("download tanh output for backward");
    let device = x.device();

    GpuVariable::from_op(
        result,
        vec![x.clone()],
        Box::new(move |grad: &GpuTensor| {
            let out = GpuTensor::from_tensor(&device, &out_cpu);
            let out_sq = scivex_gpu::ops::mul(&out, &out).expect("tanh^2");
            let ones =
                scivex_gpu::ops::fill(&device, out_cpu.shape().to_vec(), 1.0).expect("fill ones");
            let one_minus_sq = scivex_gpu::ops::sub(&ones, &out_sq).expect("1 - tanh^2");
            vec![scivex_gpu::ops::mul(grad, &one_minus_sq).expect("grad * (1 - tanh^2)")]
        }),
    )
}

/// GPU exponential: `e^x`.
///
/// Backward: `grad * exp(x) = grad * out`.
pub fn gpu_exp(x: &GpuVariable) -> GpuVariable {
    let result = x.with_data(|data| scivex_gpu::ops::exp(data).expect("gpu exp forward"));

    let out_cpu = result
        .to_tensor()
        .expect("download exp output for backward");
    let device = x.device();

    GpuVariable::from_op(
        result,
        vec![x.clone()],
        Box::new(move |grad: &GpuTensor| {
            let out = GpuTensor::from_tensor(&device, &out_cpu);
            vec![scivex_gpu::ops::mul(grad, &out).expect("grad * exp(x)")]
        }),
    )
}

/// GPU natural logarithm: `ln(x)`.
///
/// Backward: `grad / x`.
pub fn gpu_ln(x: &GpuVariable) -> GpuVariable {
    let input_cpu = x.data_cpu().expect("download input for ln backward");
    let device = x.device();

    let result = x.with_data(|data| scivex_gpu::ops::log(data).expect("gpu log forward"));

    GpuVariable::from_op(
        result,
        vec![x.clone()],
        Box::new(move |grad: &GpuTensor| {
            let input = GpuTensor::from_tensor(&device, &input_cpu);
            vec![scivex_gpu::ops::div(grad, &input).expect("grad / x")]
        }),
    )
}
