use std::sync::Arc;

use scivex_core::Tensor;

use crate::device::GpuDevice;
use crate::error::{GpuError, Result};

/// A tensor stored on the GPU as an f32 buffer.
///
/// GPU tensors are always `f32` because most GPUs lack native f64 support.
/// Use [`GpuTensor::from_tensor`] to upload and [`GpuTensor::to_tensor`] to download.
pub struct GpuTensor {
    pub(crate) buffer: Arc<wgpu::Buffer>,
    pub(crate) shape: Vec<usize>,
    pub(crate) device: GpuDevice,
}

impl GpuTensor {
    /// Upload a CPU `Tensor<f32>` to the GPU.
    pub fn from_tensor(device: &GpuDevice, tensor: &Tensor<f32>) -> Self {
        let data = tensor.as_slice();
        let buffer = device.create_buffer_init(
            data,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        Self {
            buffer: Arc::new(buffer),
            shape: tensor.shape().to_vec(),
            device: device.clone(),
        }
    }

    /// Upload a CPU `Tensor<f64>` to the GPU (values are converted to f32).
    pub fn from_tensor_f64(device: &GpuDevice, tensor: &Tensor<f64>) -> Self {
        let data: Vec<f32> = tensor.as_slice().iter().map(|&v| v as f32).collect();
        let buffer = device.create_buffer_init(
            &data,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        Self {
            buffer: Arc::new(buffer),
            shape: tensor.shape().to_vec(),
            device: device.clone(),
        }
    }

    /// Create a GPU tensor from raw f32 data and shape.
    pub fn from_slice(device: &GpuDevice, data: &[f32], shape: Vec<usize>) -> Result<Self> {
        let expected: usize = shape.iter().product();
        if expected != data.len() {
            return Err(GpuError::InvalidShape {
                reason: "data length does not match shape",
            });
        }
        let buffer = device.create_buffer_init(
            data,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        Ok(Self {
            buffer: Arc::new(buffer),
            shape,
            device: device.clone(),
        })
    }

    /// Create a GPU tensor of zeros.
    pub fn zeros(device: &GpuDevice, shape: Vec<usize>) -> Self {
        let n: usize = shape.iter().product();
        let data = vec![0.0f32; n];
        let buffer = device.create_buffer_init(
            &data,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        Self {
            buffer: Arc::new(buffer),
            shape,
            device: device.clone(),
        }
    }

    /// Download the GPU tensor back to a CPU `Tensor<f32>`.
    pub fn to_tensor(&self) -> Result<Tensor<f32>> {
        let n: usize = self.shape.iter().product();
        let data = self.device.read_buffer(&self.buffer, n)?;
        Tensor::from_vec(data, self.shape.clone()).map_err(GpuError::from)
    }

    /// Download the GPU tensor back as `Tensor<f64>` (values converted from f32).
    pub fn to_tensor_f64(&self) -> Result<Tensor<f64>> {
        let n: usize = self.shape.iter().product();
        let data = self.device.read_buffer(&self.buffer, n)?;
        let f64_data: Vec<f64> = data.iter().map(|&v| f64::from(v)).collect();
        Tensor::from_vec(f64_data, self.shape.clone()).map_err(GpuError::from)
    }

    /// Return the shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Byte size of the buffer.
    pub(crate) fn byte_size(&self) -> u64 {
        (self.numel() * std::mem::size_of::<f32>()) as u64
    }

    /// Get a reference to the underlying GPU device.
    pub fn device(&self) -> &GpuDevice {
        &self.device
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_tensor_roundtrip() {
        let Ok(dev) = GpuDevice::new() else {
            println!("No GPU — skipping");
            return;
        };

        let cpu = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let gpu = GpuTensor::from_tensor(&dev, &cpu);
        assert_eq!(gpu.shape(), &[2, 2]);
        assert_eq!(gpu.numel(), 4);

        let back = gpu.to_tensor().unwrap();
        assert_eq!(back.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(back.shape(), &[2, 2]);
    }

    #[test]
    fn test_gpu_tensor_f64_roundtrip() {
        let Ok(dev) = GpuDevice::new() else { return };

        let cpu = Tensor::from_vec(vec![1.5f64, 2.5, 3.5], vec![3]).unwrap();
        let gpu = GpuTensor::from_tensor_f64(&dev, &cpu);
        let back = gpu.to_tensor_f64().unwrap();
        for (&a, &b) in cpu.as_slice().iter().zip(back.as_slice()) {
            assert!((a - b).abs() < 1e-5);
        }
    }
}
