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
}
