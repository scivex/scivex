//! CUDA-backed tensor type.

use scivex_core::Tensor;

use crate::error::{GpuError, Result};

use super::memory::CudaBuffer;

/// A tensor whose data resides in CUDA device memory.
pub struct CudaTensor {
    buffer: CudaBuffer,
    shape: Vec<usize>,
    device_id: i32,
}

impl CudaTensor {
    /// Upload a CPU `Tensor<f32>` to a CUDA device.
    pub fn from_tensor(tensor: &Tensor<f32>, device_id: i32) -> Result<Self> {
        let data = tensor.as_slice();
        let buffer = CudaBuffer::from_host(data, device_id)?;
        Ok(Self {
            buffer,
            shape: tensor.shape().to_vec(),
            device_id,
        })
    }

    /// Download this CUDA tensor back to a CPU `Tensor<f32>`.
    pub fn to_tensor(&self) -> Result<Tensor<f32>> {
        let n = self.numel();
        let mut data = vec![0.0f32; n];
        self.buffer.to_host(&mut data)?;
        Tensor::from_vec(data, self.shape.clone()).map_err(|e| GpuError::CoreError(e))
    }

    /// Create a zero-filled CUDA tensor.
    pub fn zeros(shape: Vec<usize>, device_id: i32) -> Result<Self> {
        let n: usize = shape.iter().product();
        let zeros = vec![0.0f32; n];
        let buffer = CudaBuffer::from_host(&zeros, device_id)?;
        Ok(Self {
            buffer,
            shape,
            device_id,
        })
    }

    /// Shape of this tensor.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Reference to the underlying device buffer.
    pub fn buffer(&self) -> &CudaBuffer {
        &self.buffer
    }

    /// Mutable reference to the underlying device buffer.
    pub fn buffer_mut(&mut self) -> &mut CudaBuffer {
        &mut self.buffer
    }

    /// Device ordinal this tensor lives on.
    pub fn device_id(&self) -> i32 {
        self.device_id
    }
}
