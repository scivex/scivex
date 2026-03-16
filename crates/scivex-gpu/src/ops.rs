//! GPU compute operations via WGSL shaders.

use std::sync::Arc;

use crate::error::{GpuError, Result};
use crate::tensor::GpuTensor;

// ---------------------------------------------------------------------------
// WGSL shader sources
// ---------------------------------------------------------------------------

const ELEMENTWISE_SHADER: &str = r"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;

@compute @workgroup_size(256)
fn add_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx < arrayLength(&a) {
        result[idx] = a[idx] + b[idx];
    }
}

@compute @workgroup_size(256)
fn sub_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx < arrayLength(&a) {
        result[idx] = a[idx] - b[idx];
    }
}

@compute @workgroup_size(256)
fn mul_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx < arrayLength(&a) {
        result[idx] = a[idx] * b[idx];
    }
}

@compute @workgroup_size(256)
fn div_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx < arrayLength(&a) {
        result[idx] = a[idx] / b[idx];
    }
}
";

const SCALAR_SHADER: &str = r"
struct Params {
    scalar: f32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn add_scalar_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx < arrayLength(&input) {
        result[idx] = input[idx] + params.scalar;
    }
}

@compute @workgroup_size(256)
fn mul_scalar_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx < arrayLength(&input) {
        result[idx] = input[idx] * params.scalar;
    }
}
";

const MATMUL_SHADER: &str = r"
struct Dims {
    m: u32,
    n: u32,
    k: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

@compute @workgroup_size(16, 16)
fn matmul_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    let col = gid.y;

    if row >= dims.m || col >= dims.n {
        return;
    }

    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < dims.k; i = i + 1u) {
        sum = sum + a[row * dims.k + i] * b[i * dims.n + col];
    }
    result[row * dims.n + col] = sum;
}
";

const REDUCTION_SHADER: &str = r"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;

var<workgroup> wg_data: array<f32, 256>;

@compute @workgroup_size(256)
fn sum_kernel(@builtin(global_invocation_id) gid: vec3<u32>,
              @builtin(local_invocation_id) lid: vec3<u32>,
              @builtin(workgroup_id) wid: vec3<u32>) {
    let global_idx = gid.x;
    let local_idx = lid.x;

    if global_idx < arrayLength(&input) {
        wg_data[local_idx] = input[global_idx];
    } else {
        wg_data[local_idx] = 0.0;
    }
    workgroupBarrier();

    var stride: u32 = 128u;
    loop {
        if stride == 0u { break; }
        if local_idx < stride {
            wg_data[local_idx] = wg_data[local_idx] + wg_data[local_idx + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if local_idx == 0u {
        result[wid.x] = wg_data[0];
    }
}
";

const UNARY_SHADER: &str = r"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;

@compute @workgroup_size(256)
fn relu_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx < arrayLength(&input) {
        result[idx] = max(input[idx], 0.0);
    }
}

@compute @workgroup_size(256)
fn sigmoid_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx < arrayLength(&input) {
        result[idx] = 1.0 / (1.0 + exp(-input[idx]));
    }
}

@compute @workgroup_size(256)
fn tanh_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx < arrayLength(&input) {
        let e = exp(input[idx]);
        let em = exp(-input[idx]);
        result[idx] = (e - em) / (e + em);
    }
}

@compute @workgroup_size(256)
fn negate_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx < arrayLength(&input) {
        result[idx] = -input[idx];
    }
}

@compute @workgroup_size(256)
fn exp_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx < arrayLength(&input) {
        result[idx] = exp(input[idx]);
    }
}

@compute @workgroup_size(256)
fn log_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx < arrayLength(&input) {
        result[idx] = log(input[idx]);
    }
}
";

const BINARY_BACKWARD_SHADER: &str = r"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;

@compute @workgroup_size(256)
fn relu_backward_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx < arrayLength(&a) {
        result[idx] = select(0.0, a[idx], b[idx] > 0.0);
    }
}
";

const TRANSPOSE_SHADER: &str = r"
struct Dims {
    rows: u32,
    cols: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;
@group(0) @binding(2) var<uniform> dims: Dims;

@compute @workgroup_size(16, 16)
fn transpose_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    let col = gid.y;
    if row >= dims.rows || col >= dims.cols {
        return;
    }
    result[col * dims.rows + row] = input[row * dims.cols + col];
}
";

const FILL_SHADER: &str = r"
struct Params {
    value: f32,
}

@group(0) @binding(0) var<storage, read_write> result: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(256)
fn fill_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx < arrayLength(&result) {
        result[idx] = params.value;
    }
}
";

// ---------------------------------------------------------------------------
// Helper: run a two-input elementwise kernel
// ---------------------------------------------------------------------------

fn run_elementwise(a: &GpuTensor, b: &GpuTensor, entry: &str) -> Result<GpuTensor> {
    if a.shape() != b.shape() {
        return Err(GpuError::DimensionMismatch {
            expected: a.shape().to_vec(),
            got: b.shape().to_vec(),
        });
    }

    let dev = &a.device;
    let n = a.numel();
    let out_buf = dev.create_buffer(
        a.byte_size(),
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );

    let module = dev
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("elementwise"),
            source: wgpu::ShaderSource::Wgsl(ELEMENTWISE_SHADER.into()),
        });

    let pipeline = dev
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: Some(entry),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

    let bind_group = dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: a.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: b.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: out_buf.as_entire_binding(),
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
        pass.dispatch_workgroups(dispatch_size(n, 256), 1, 1);
    }
    dev.queue.submit(Some(encoder.finish()));

    Ok(GpuTensor {
        buffer: Arc::new(out_buf),
        shape: a.shape().to_vec(),
        device: dev.clone(),
    })
}

// ---------------------------------------------------------------------------
// Public elementwise operations
// ---------------------------------------------------------------------------

/// Element-wise addition on the GPU.
pub fn add(a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
    run_elementwise(a, b, "add_kernel")
}

/// Element-wise subtraction on the GPU.
pub fn sub(a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
    run_elementwise(a, b, "sub_kernel")
}

/// Element-wise multiplication on the GPU.
pub fn mul(a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
    run_elementwise(a, b, "mul_kernel")
}

/// Element-wise division on the GPU.
pub fn div(a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
    run_elementwise(a, b, "div_kernel")
}

// ---------------------------------------------------------------------------
// Scalar operations
// ---------------------------------------------------------------------------

#[allow(clippy::unnecessary_wraps)]
fn run_scalar_op(input: &GpuTensor, scalar: f32, entry: &str) -> Result<GpuTensor> {
    let dev = &input.device;
    let n = input.numel();
    let out_buf = dev.create_buffer(
        input.byte_size(),
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );

    let params_buf = dev.create_buffer_init(&[scalar], wgpu::BufferUsages::UNIFORM);

    let module = dev
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scalar_op"),
            source: wgpu::ShaderSource::Wgsl(SCALAR_SHADER.into()),
        });

    let pipeline = dev
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: Some(entry),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

    let bind_group = dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: out_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buf.as_entire_binding(),
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
        pass.dispatch_workgroups(dispatch_size(n, 256), 1, 1);
    }
    dev.queue.submit(Some(encoder.finish()));

    Ok(GpuTensor {
        buffer: Arc::new(out_buf),
        shape: input.shape().to_vec(),
        device: dev.clone(),
    })
}

/// Add a scalar to every element on the GPU.
pub fn add_scalar(input: &GpuTensor, scalar: f32) -> Result<GpuTensor> {
    run_scalar_op(input, scalar, "add_scalar_kernel")
}

/// Multiply every element by a scalar on the GPU.
pub fn mul_scalar(input: &GpuTensor, scalar: f32) -> Result<GpuTensor> {
    run_scalar_op(input, scalar, "mul_scalar_kernel")
}

// ---------------------------------------------------------------------------
// Matrix multiplication
// ---------------------------------------------------------------------------

/// Matrix multiplication on the GPU: `a @ b`.
///
/// `a` must have shape `[m, k]` and `b` must have shape `[k, n]`.
/// Returns a tensor of shape `[m, n]`.
pub fn matmul(a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
    if a.ndim() != 2 || b.ndim() != 2 {
        return Err(GpuError::InvalidShape {
            reason: "matmul requires 2-D tensors",
        });
    }
    let m = a.shape()[0];
    let k = a.shape()[1];
    if b.shape()[0] != k {
        return Err(GpuError::DimensionMismatch {
            expected: vec![k, b.shape()[1]],
            got: b.shape().to_vec(),
        });
    }
    let n = b.shape()[1];

    let dev = &a.device;
    let out_size = (m * n * std::mem::size_of::<f32>()) as u64;
    let out_buf = dev.create_buffer(
        out_size,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );

    let dims_data: [u32; 4] = [m as u32, n as u32, k as u32, 0];
    let dims_buf = {
        use wgpu::util::DeviceExt;
        dev.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&dims_data),
                usage: wgpu::BufferUsages::UNIFORM,
            })
    };

    let module = dev
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("matmul"),
            source: wgpu::ShaderSource::Wgsl(MATMUL_SHADER.into()),
        });

    let pipeline = dev
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: Some("matmul_kernel"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

    let bind_group = dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: a.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: b.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: out_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: dims_buf.as_entire_binding(),
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
        pass.dispatch_workgroups(dispatch_size(m, 16), dispatch_size(n, 16), 1);
    }
    dev.queue.submit(Some(encoder.finish()));

    Ok(GpuTensor {
        buffer: Arc::new(out_buf),
        shape: vec![m, n],
        device: dev.clone(),
    })
}

// ---------------------------------------------------------------------------
// Reduction: sum
// ---------------------------------------------------------------------------

/// Sum all elements of a GPU tensor, returning a scalar.
pub fn sum(input: &GpuTensor) -> Result<f32> {
    let dev = &input.device;

    let mut current_buf = Arc::clone(&input.buffer);
    let mut current_n = input.numel();

    while current_n > 1 {
        let n_groups = dispatch_size(current_n, 256) as usize;
        let out_size = (n_groups * std::mem::size_of::<f32>()) as u64;
        let out_buf = dev.create_buffer(
            out_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        let module = dev
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("reduction"),
                source: wgpu::ShaderSource::Wgsl(REDUCTION_SHADER.into()),
            });

        let pipeline = dev
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &module,
                entry_point: Some("sum_kernel"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let bind_group = dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: current_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out_buf.as_entire_binding(),
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
            pass.dispatch_workgroups(n_groups as u32, 1, 1);
        }
        dev.queue.submit(Some(encoder.finish()));

        current_buf = Arc::new(out_buf);
        current_n = n_groups;
    }

    let result = dev.read_buffer(&current_buf, 1)?;
    Ok(result[0])
}

/// Mean of all elements.
pub fn mean(input: &GpuTensor) -> Result<f32> {
    let s = sum(input)?;
    Ok(s / input.numel() as f32)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn dispatch_size(total: usize, workgroup_size: usize) -> u32 {
    total.div_ceil(workgroup_size) as u32
}

// ---------------------------------------------------------------------------
// Helper: run a single-input unary kernel
// ---------------------------------------------------------------------------

#[allow(clippy::unnecessary_wraps)]
fn run_unary(input: &GpuTensor, shader_src: &str, entry: &str) -> Result<GpuTensor> {
    let dev = &input.device;
    let n = input.numel();
    let out_buf = dev.create_buffer(
        input.byte_size(),
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );

    let module = dev
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("unary"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

    let pipeline = dev
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: Some(entry),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

    let bind_group = dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: out_buf.as_entire_binding(),
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
        pass.dispatch_workgroups(dispatch_size(n, 256), 1, 1);
    }
    dev.queue.submit(Some(encoder.finish()));

    Ok(GpuTensor {
        buffer: Arc::new(out_buf),
        shape: input.shape().to_vec(),
        device: dev.clone(),
    })
}

// ---------------------------------------------------------------------------
// Helper: run a two-input binary kernel (e.g. backward ops)
// ---------------------------------------------------------------------------

fn run_binary(a: &GpuTensor, b: &GpuTensor, shader_src: &str, entry: &str) -> Result<GpuTensor> {
    if a.shape() != b.shape() {
        return Err(GpuError::DimensionMismatch {
            expected: a.shape().to_vec(),
            got: b.shape().to_vec(),
        });
    }

    let dev = &a.device;
    let n = a.numel();
    let out_buf = dev.create_buffer(
        a.byte_size(),
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );

    let module = dev
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("binary"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

    let pipeline = dev
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: Some(entry),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

    let bind_group = dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: a.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: b.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: out_buf.as_entire_binding(),
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
        pass.dispatch_workgroups(dispatch_size(n, 256), 1, 1);
    }
    dev.queue.submit(Some(encoder.finish()));

    Ok(GpuTensor {
        buffer: Arc::new(out_buf),
        shape: a.shape().to_vec(),
        device: dev.clone(),
    })
}

// ---------------------------------------------------------------------------
// Activation / unary operations
// ---------------------------------------------------------------------------

/// Element-wise ReLU activation on the GPU: `max(x, 0)`.
pub fn relu(input: &GpuTensor) -> Result<GpuTensor> {
    run_unary(input, UNARY_SHADER, "relu_kernel")
}

/// Element-wise sigmoid activation on the GPU: `1 / (1 + exp(-x))`.
pub fn sigmoid(input: &GpuTensor) -> Result<GpuTensor> {
    run_unary(input, UNARY_SHADER, "sigmoid_kernel")
}

/// Element-wise tanh activation on the GPU.
///
/// Named `tanh_op` to avoid conflict with `f32::tanh`.
pub fn tanh_op(input: &GpuTensor) -> Result<GpuTensor> {
    run_unary(input, UNARY_SHADER, "tanh_kernel")
}

/// Element-wise negation on the GPU: `-x`.
pub fn negate(input: &GpuTensor) -> Result<GpuTensor> {
    run_unary(input, UNARY_SHADER, "negate_kernel")
}

/// Element-wise exponential on the GPU: `e^x`.
pub fn exp(input: &GpuTensor) -> Result<GpuTensor> {
    run_unary(input, UNARY_SHADER, "exp_kernel")
}

/// Element-wise natural logarithm on the GPU: `ln(x)`.
pub fn log(input: &GpuTensor) -> Result<GpuTensor> {
    run_unary(input, UNARY_SHADER, "log_kernel")
}

// ---------------------------------------------------------------------------
// Backward operations
// ---------------------------------------------------------------------------

/// ReLU backward pass on the GPU.
///
/// Computes `grad_output * (input > 0 ? 1 : 0)`.
pub fn relu_backward(grad_output: &GpuTensor, input: &GpuTensor) -> Result<GpuTensor> {
    run_binary(
        grad_output,
        input,
        BINARY_BACKWARD_SHADER,
        "relu_backward_kernel",
    )
}

// ---------------------------------------------------------------------------
// Transpose
// ---------------------------------------------------------------------------

/// Transpose a 2-D GPU tensor (swap rows and columns).
///
/// Input must have exactly 2 dimensions `[rows, cols]`.
/// Returns a tensor of shape `[cols, rows]`.
pub fn transpose(input: &GpuTensor) -> Result<GpuTensor> {
    if input.ndim() != 2 {
        return Err(GpuError::InvalidShape {
            reason: "transpose requires a 2-D tensor",
        });
    }

    let rows = input.shape()[0];
    let cols = input.shape()[1];
    let dev = &input.device;

    let out_size = (rows * cols * std::mem::size_of::<f32>()) as u64;
    let out_buf = dev.create_buffer(
        out_size,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );

    let dims_data: [u32; 4] = [rows as u32, cols as u32, 0, 0];
    let dims_buf = {
        use wgpu::util::DeviceExt;
        dev.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&dims_data),
                usage: wgpu::BufferUsages::UNIFORM,
            })
    };

    let module = dev
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("transpose"),
            source: wgpu::ShaderSource::Wgsl(TRANSPOSE_SHADER.into()),
        });

    let pipeline = dev
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: Some("transpose_kernel"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

    let bind_group = dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: out_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: dims_buf.as_entire_binding(),
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
        pass.dispatch_workgroups(dispatch_size(rows, 16), dispatch_size(cols, 16), 1);
    }
    dev.queue.submit(Some(encoder.finish()));

    Ok(GpuTensor {
        buffer: Arc::new(out_buf),
        shape: vec![cols, rows],
        device: dev.clone(),
    })
}

// ---------------------------------------------------------------------------
// Fill
// ---------------------------------------------------------------------------

/// Create a GPU tensor filled with a constant scalar value.
pub fn fill(device: &crate::device::GpuDevice, shape: Vec<usize>, value: f32) -> Result<GpuTensor> {
    let n: usize = shape.iter().product();
    let out_size = (n * std::mem::size_of::<f32>()) as u64;
    let out_buf = device.create_buffer(
        out_size,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );

    let params_buf = device.create_buffer_init(&[value], wgpu::BufferUsages::UNIFORM);

    let module = device
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fill"),
            source: wgpu::ShaderSource::Wgsl(FILL_SHADER.into()),
        });

    let pipeline = device
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: Some("fill_kernel"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

    let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: out_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(dispatch_size(n, 256), 1, 1);
    }
    device.queue.submit(Some(encoder.finish()));

    Ok(GpuTensor {
        buffer: Arc::new(out_buf),
        shape,
        device: device.clone(),
    })
}

// ---------------------------------------------------------------------------
// Sub scalar
// ---------------------------------------------------------------------------

/// Subtract a scalar from every element on the GPU.
pub fn sub_scalar(input: &GpuTensor, scalar: f32) -> Result<GpuTensor> {
    add_scalar(input, -scalar)
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
    fn test_gpu_add() {
        let Some(dev) = get_device() else { return };
        let a = GpuTensor::from_slice(&dev, &[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let b = GpuTensor::from_slice(&dev, &[10.0, 20.0, 30.0, 40.0], vec![4]).unwrap();
        let c = add(&a, &b).unwrap();
        let result = c.to_tensor().unwrap();
        assert_eq!(result.as_slice(), &[11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn test_gpu_sub() {
        let Some(dev) = get_device() else { return };
        let a = GpuTensor::from_slice(&dev, &[10.0, 20.0, 30.0], vec![3]).unwrap();
        let b = GpuTensor::from_slice(&dev, &[1.0, 2.0, 3.0], vec![3]).unwrap();
        let c = sub(&a, &b).unwrap();
        let result = c.to_tensor().unwrap();
        assert_eq!(result.as_slice(), &[9.0, 18.0, 27.0]);
    }

    #[test]
    fn test_gpu_mul() {
        let Some(dev) = get_device() else { return };
        let a = GpuTensor::from_slice(&dev, &[2.0, 3.0, 4.0], vec![3]).unwrap();
        let b = GpuTensor::from_slice(&dev, &[5.0, 6.0, 7.0], vec![3]).unwrap();
        let c = mul(&a, &b).unwrap();
        let result = c.to_tensor().unwrap();
        assert_eq!(result.as_slice(), &[10.0, 18.0, 28.0]);
    }

    #[test]
    fn test_gpu_add_scalar() {
        let Some(dev) = get_device() else { return };
        let a = GpuTensor::from_slice(&dev, &[1.0, 2.0, 3.0], vec![3]).unwrap();
        let c = add_scalar(&a, 10.0).unwrap();
        let result = c.to_tensor().unwrap();
        assert_eq!(result.as_slice(), &[11.0, 12.0, 13.0]);
    }

    #[test]
    fn test_gpu_mul_scalar() {
        let Some(dev) = get_device() else { return };
        let a = GpuTensor::from_slice(&dev, &[1.0, 2.0, 3.0], vec![3]).unwrap();
        let c = mul_scalar(&a, 3.0).unwrap();
        let result = c.to_tensor().unwrap();
        assert_eq!(result.as_slice(), &[3.0, 6.0, 9.0]);
    }

    #[test]
    fn test_gpu_matmul() {
        let Some(dev) = get_device() else { return };
        // [2,3] @ [3,2] = [2,2]
        let a = GpuTensor::from_slice(&dev, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let b =
            GpuTensor::from_slice(&dev, &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]).unwrap();
        let c = matmul(&a, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        let result = c.to_tensor().unwrap();
        // Row 0: 1*7+2*9+3*11=58,  1*8+2*10+3*12=64
        // Row 1: 4*7+5*9+6*11=139, 4*8+5*10+6*12=154
        assert_eq!(result.as_slice(), &[58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_gpu_matmul_shape_mismatch() {
        let Some(dev) = get_device() else { return };
        let a = GpuTensor::from_slice(&dev, &[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = GpuTensor::from_slice(&dev, &[1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        assert!(matmul(&a, &b).is_err());
    }

    #[test]
    fn test_gpu_sum() {
        let Some(dev) = get_device() else { return };
        let a = GpuTensor::from_slice(&dev, &[1.0, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
        let s = sum(&a).unwrap();
        assert!((s - 15.0).abs() < 1e-5);
    }

    #[test]
    fn test_gpu_mean() {
        let Some(dev) = get_device() else { return };
        let a = GpuTensor::from_slice(&dev, &[2.0, 4.0, 6.0, 8.0], vec![4]).unwrap();
        let m = mean(&a).unwrap();
        assert!((m - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_gpu_dimension_mismatch() {
        let Some(dev) = get_device() else { return };
        let a = GpuTensor::from_slice(&dev, &[1.0, 2.0, 3.0], vec![3]).unwrap();
        let b = GpuTensor::from_slice(&dev, &[1.0, 2.0], vec![2]).unwrap();
        assert!(add(&a, &b).is_err());
    }

    #[test]
    fn test_gpu_relu() {
        let Some(dev) = get_device() else { return };
        let a = GpuTensor::from_slice(&dev, &[-2.0, -1.0, 0.0, 1.0, 2.0], vec![5]).unwrap();
        let c = relu(&a).unwrap();
        let result = c.to_tensor().unwrap();
        assert_eq!(result.as_slice(), &[0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_gpu_sigmoid() {
        let Some(dev) = get_device() else { return };
        let a = GpuTensor::from_slice(&dev, &[-10.0, 0.0, 10.0], vec![3]).unwrap();
        let c = sigmoid(&a).unwrap();
        let result = c.to_tensor().unwrap();
        let vals = result.as_slice();
        // sigmoid(-10) ~ 0, sigmoid(0) = 0.5, sigmoid(10) ~ 1
        assert!(vals[0] < 0.01);
        assert!((vals[1] - 0.5).abs() < 1e-5);
        assert!(vals[2] > 0.99);
        // All values must be in [0, 1]
        for &v in vals {
            assert!((0.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn test_gpu_tanh() {
        let Some(dev) = get_device() else { return };
        let a = GpuTensor::from_slice(&dev, &[-10.0, 0.0, 10.0], vec![3]).unwrap();
        let c = tanh_op(&a).unwrap();
        let result = c.to_tensor().unwrap();
        let vals = result.as_slice();
        // tanh(-10) ~ -1, tanh(0) = 0, tanh(10) ~ 1
        assert!((vals[0] + 1.0).abs() < 0.01);
        assert!(vals[1].abs() < 1e-5);
        assert!((vals[2] - 1.0).abs() < 0.01);
        // All values must be in [-1, 1]
        for &v in vals {
            assert!((-1.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn test_gpu_negate() {
        let Some(dev) = get_device() else { return };
        let a = GpuTensor::from_slice(&dev, &[1.0, -2.0, 3.0, 0.0], vec![4]).unwrap();
        let c = negate(&a).unwrap();
        let result = c.to_tensor().unwrap();
        assert_eq!(result.as_slice(), &[-1.0, 2.0, -3.0, 0.0]);
    }

    #[test]
    fn test_gpu_exp() {
        let Some(dev) = get_device() else { return };
        let a = GpuTensor::from_slice(&dev, &[0.0, 1.0, 2.0], vec![3]).unwrap();
        let c = exp(&a).unwrap();
        let result = c.to_tensor().unwrap();
        let vals = result.as_slice();
        assert!((vals[0] - 1.0).abs() < 1e-5);
        assert!((vals[1] - std::f32::consts::E).abs() < 1e-4);
        assert!((vals[2] - std::f32::consts::E * std::f32::consts::E).abs() < 1e-3);
    }

    #[test]
    fn test_gpu_log() {
        let Some(dev) = get_device() else { return };
        let a = GpuTensor::from_slice(&dev, &[1.0, std::f32::consts::E, 10.0], vec![3]).unwrap();
        let c = log(&a).unwrap();
        let result = c.to_tensor().unwrap();
        let vals = result.as_slice();
        assert!(vals[0].abs() < 1e-5); // ln(1) = 0
        assert!((vals[1] - 1.0).abs() < 1e-4); // ln(e) = 1
        assert!((vals[2] - 10.0_f32.ln()).abs() < 1e-4);
    }

    #[test]
    fn test_gpu_relu_backward() {
        let Some(dev) = get_device() else { return };
        let grad = GpuTensor::from_slice(&dev, &[1.0, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
        let input = GpuTensor::from_slice(&dev, &[-1.0, 0.5, -0.1, 2.0, 0.0], vec![5]).unwrap();
        let c = relu_backward(&grad, &input).unwrap();
        let result = c.to_tensor().unwrap();
        // grad * (input > 0 ? 1 : 0)
        // -1 <= 0 -> 0, 0.5 > 0 -> 2, -0.1 <= 0 -> 0, 2 > 0 -> 4, 0 <= 0 -> 0
        assert_eq!(result.as_slice(), &[0.0, 2.0, 0.0, 4.0, 0.0]);
    }

    #[test]
    fn test_gpu_transpose() {
        let Some(dev) = get_device() else { return };
        // 2x3 matrix: [[1,2,3],[4,5,6]]
        let a = GpuTensor::from_slice(&dev, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let c = transpose(&a).unwrap();
        assert_eq!(c.shape(), &[3, 2]);
        let result = c.to_tensor().unwrap();
        // Transposed: [[1,4],[2,5],[3,6]]
        assert_eq!(result.as_slice(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_gpu_fill() {
        let Some(dev) = get_device() else { return };
        let c = fill(&dev, vec![2, 3], 7.0).unwrap();
        assert_eq!(c.shape(), &[2, 3]);
        let result = c.to_tensor().unwrap();
        assert_eq!(result.as_slice(), &[7.0, 7.0, 7.0, 7.0, 7.0, 7.0]);
    }

    #[test]
    fn test_gpu_sub_scalar() {
        let Some(dev) = get_device() else { return };
        let a = GpuTensor::from_slice(&dev, &[10.0, 20.0, 30.0], vec![3]).unwrap();
        let c = sub_scalar(&a, 5.0).unwrap();
        let result = c.to_tensor().unwrap();
        assert_eq!(result.as_slice(), &[5.0, 15.0, 25.0]);
    }
}
