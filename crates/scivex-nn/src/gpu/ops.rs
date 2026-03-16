//! GPU differentiable operations.
//!
//! Each function records a forward computation on the GPU and captures a
//! backward closure so that [`GpuVariable::backward`](super::variable::GpuVariable::backward)
//! can compute gradients via reverse-mode autodiff.
//!
//! Data needed for the backward pass is downloaded to CPU at forward time and
//! re-uploaded inside the gradient closure. This is correct and simple; a
//! future version can cache GPU buffers to avoid the round-trip.

use scivex_gpu::GpuTensor;

use super::variable::GpuVariable;

// ── Element-wise binary ops ─────────────────────────────────────────

/// Element-wise addition of two GPU variables.
///
/// Backward: `grad_a = grad`, `grad_b = grad`.
pub fn gpu_add(a: &GpuVariable, b: &GpuVariable) -> GpuVariable {
    let result = a.with_data(|a_gpu| {
        b.with_data(|b_gpu| scivex_gpu::ops::add(a_gpu, b_gpu).expect("gpu add forward"))
    });

    let device = a.device();
    GpuVariable::from_op(
        result,
        vec![a.clone(), b.clone()],
        Box::new(move |grad: &GpuTensor| {
            // Both parents receive the upstream gradient unchanged.
            let g_cpu = grad.to_tensor().expect("grad download");
            let ga = GpuTensor::from_tensor(&device, &g_cpu);
            let gb = GpuTensor::from_tensor(&device, &g_cpu);
            vec![ga, gb]
        }),
    )
}

/// Element-wise subtraction of two GPU variables.
///
/// Backward: `grad_a = grad`, `grad_b = -grad`.
pub fn gpu_sub(a: &GpuVariable, b: &GpuVariable) -> GpuVariable {
    let result = a.with_data(|a_gpu| {
        b.with_data(|b_gpu| scivex_gpu::ops::sub(a_gpu, b_gpu).expect("gpu sub forward"))
    });

    let device = a.device();
    GpuVariable::from_op(
        result,
        vec![a.clone(), b.clone()],
        Box::new(move |grad: &GpuTensor| {
            let g_cpu = grad.to_tensor().expect("grad download");
            let ga = GpuTensor::from_tensor(&device, &g_cpu);
            let neg_g = scivex_gpu::ops::negate(&ga).expect("negate grad");
            vec![ga, neg_g]
        }),
    )
}

/// Element-wise multiplication (Hadamard product) of two GPU variables.
///
/// Backward: `grad_a = grad * b_data`, `grad_b = grad * a_data`.
pub fn gpu_mul(a: &GpuVariable, b: &GpuVariable) -> GpuVariable {
    // Save forward data for backward (download to CPU).
    let a_data_cpu = a.data_cpu().expect("download a for backward");
    let b_data_cpu = b.data_cpu().expect("download b for backward");
    let device = a.device();

    let result = a.with_data(|a_gpu| {
        b.with_data(|b_gpu| scivex_gpu::ops::mul(a_gpu, b_gpu).expect("gpu mul forward"))
    });

    GpuVariable::from_op(
        result,
        vec![a.clone(), b.clone()],
        Box::new(move |grad: &GpuTensor| {
            let a_re = GpuTensor::from_tensor(&device, &a_data_cpu);
            let b_re = GpuTensor::from_tensor(&device, &b_data_cpu);
            let ga = scivex_gpu::ops::mul(grad, &b_re).expect("grad_a = grad * b");
            let gb = scivex_gpu::ops::mul(grad, &a_re).expect("grad_b = grad * a");
            vec![ga, gb]
        }),
    )
}

/// Negation of a GPU variable.
///
/// Backward: `grad_input = -grad`.
pub fn gpu_neg(a: &GpuVariable) -> GpuVariable {
    let result = a.with_data(|a_gpu| scivex_gpu::ops::negate(a_gpu).expect("gpu negate forward"));

    GpuVariable::from_op(
        result,
        vec![a.clone()],
        Box::new(|grad: &GpuTensor| vec![scivex_gpu::ops::negate(grad).expect("negate grad")]),
    )
}

// ── Matrix operations ───────────────────────────────────────────────

/// Matrix multiplication: `a @ b`.
///
/// `a` has shape `[m, k]`, `b` has shape `[k, n]`, result is `[m, n]`.
///
/// Backward: `grad_a = grad @ b^T`, `grad_b = a^T @ grad`.
pub fn gpu_matmul(a: &GpuVariable, b: &GpuVariable) -> GpuVariable {
    // Save forward data for backward.
    let a_data_cpu = a.data_cpu().expect("download a for backward");
    let b_data_cpu = b.data_cpu().expect("download b for backward");
    let device = a.device();

    let result = a.with_data(|a_gpu| {
        b.with_data(|b_gpu| scivex_gpu::ops::matmul(a_gpu, b_gpu).expect("gpu matmul forward"))
    });

    GpuVariable::from_op(
        result,
        vec![a.clone(), b.clone()],
        Box::new(move |grad: &GpuTensor| {
            let a_re = GpuTensor::from_tensor(&device, &a_data_cpu);
            let b_re = GpuTensor::from_tensor(&device, &b_data_cpu);

            // grad_a = grad @ b^T
            let bt = scivex_gpu::ops::transpose(&b_re).expect("transpose b");
            let ga = scivex_gpu::ops::matmul(grad, &bt).expect("grad @ b^T");

            // grad_b = a^T @ grad
            let at = scivex_gpu::ops::transpose(&a_re).expect("transpose a");
            let gb = scivex_gpu::ops::matmul(&at, grad).expect("a^T @ grad");

            vec![ga, gb]
        }),
    )
}

// ── Reductions ──────────────────────────────────────────────────────

/// Sum all elements to a scalar GPU variable.
///
/// Backward: broadcast scalar grad to input shape.
pub fn gpu_sum(a: &GpuVariable) -> GpuVariable {
    let shape = a.shape();
    let device = a.device();

    let sum_val = a.with_data(|a_gpu| scivex_gpu::ops::sum(a_gpu).expect("gpu sum forward"));

    let result =
        GpuTensor::from_slice(&device, &[sum_val], vec![1]).expect("scalar tensor from sum");

    let device2 = device.clone();
    GpuVariable::from_op(
        result,
        vec![a.clone()],
        Box::new(move |grad: &GpuTensor| {
            // grad is a scalar [1]. Broadcast to input shape.
            let g_cpu = grad.to_tensor().expect("grad download");
            let g_val = g_cpu.as_slice()[0];
            let full = scivex_gpu::ops::fill(&device2, shape.clone(), g_val)
                .expect("fill for sum backward");
            vec![full]
        }),
    )
}

/// Mean of all elements to a scalar GPU variable.
///
/// Backward: `grad_input = grad / n`.
pub fn gpu_mean(a: &GpuVariable) -> GpuVariable {
    let n = a.numel();
    let shape = a.shape();
    let device = a.device();

    let mean_val = a.with_data(|a_gpu| scivex_gpu::ops::mean(a_gpu).expect("gpu mean forward"));

    let result =
        GpuTensor::from_slice(&device, &[mean_val], vec![1]).expect("scalar tensor from mean");

    let device2 = device.clone();
    GpuVariable::from_op(
        result,
        vec![a.clone()],
        Box::new(move |grad: &GpuTensor| {
            let g_cpu = grad.to_tensor().expect("grad download");
            let g_val = g_cpu.as_slice()[0];
            let scale = g_val / n as f32;
            let full = scivex_gpu::ops::fill(&device2, shape.clone(), scale)
                .expect("fill for mean backward");
            vec![full]
        }),
    )
}

// ── Scalar operations ───────────────────────────────────────────────

/// Multiply every element by a scalar.
///
/// Backward: `grad_input = grad * scalar`.
pub fn gpu_scalar_mul(a: &GpuVariable, scalar: f32) -> GpuVariable {
    let result = a.with_data(|a_gpu| {
        scivex_gpu::ops::mul_scalar(a_gpu, scalar).expect("gpu mul_scalar forward")
    });

    GpuVariable::from_op(
        result,
        vec![a.clone()],
        Box::new(move |grad: &GpuTensor| {
            vec![scivex_gpu::ops::mul_scalar(grad, scalar).expect("grad * scalar")]
        }),
    )
}
