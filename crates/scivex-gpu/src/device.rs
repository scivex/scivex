use std::fmt;
use std::sync::Arc;

use crate::error::{GpuError, Result};

/// A GPU compute context wrapping a `wgpu` device and queue.
///
/// Clone is cheap — it is an `Arc` wrapper.
#[derive(Clone)]
pub struct GpuDevice {
    pub(crate) device: Arc<wgpu::Device>,
    pub(crate) queue: Arc<wgpu::Queue>,
    info: GpuInfo,
}

/// Information about the GPU adapter.
#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub name: String,
    pub backend: String,
    pub device_type: String,
}

impl GpuDevice {
    /// Create a GPU context, requesting the best available adapter.
    pub fn new() -> Result<Self> {
        pollster::block_on(Self::new_async())
    }

    /// Async version of [`GpuDevice::new`].
    pub async fn new_async() -> Result<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or(GpuError::NoAdapter)?;

        let adapter_info = adapter.get_info();
        let info = GpuInfo {
            name: adapter_info.name.clone(),
            backend: format!("{:?}", adapter_info.backend),
            device_type: format!("{:?}", adapter_info.device_type),
        };

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("scivex-gpu"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    ..Default::default()
                },
                None,
            )
            .await
            .map_err(|e| GpuError::DeviceCreationFailed {
                reason: e.to_string(),
            })?;

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            info,
        })
    }

    /// Return information about the GPU.
    pub fn info(&self) -> &GpuInfo {
        &self.info
    }

    /// Create a storage buffer from f32 data.
    pub(crate) fn create_buffer_init(
        &self,
        data: &[f32],
        usage: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(data),
                usage,
            })
    }

    /// Create an empty storage buffer of the given byte size.
    pub(crate) fn create_buffer(&self, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage,
            mapped_at_creation: false,
        })
    }

    /// Read f32 data back from a GPU buffer.
    pub(crate) fn read_buffer(&self, buffer: &wgpu::Buffer, count: usize) -> Result<Vec<f32>> {
        let size = (count * std::mem::size_of::<f32>()) as u64;
        let staging = self.create_buffer(
            size,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        self.device.poll(wgpu::Maintain::Wait);

        rx.recv()
            .map_err(|e| GpuError::TransferError {
                reason: e.to_string(),
            })?
            .map_err(|e| GpuError::TransferError {
                reason: e.to_string(),
            })?;

        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();

        Ok(result)
    }
}

impl fmt::Display for GpuInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({}, {})", self.name, self.backend, self.device_type)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_device_creation() {
        if let Ok(dev) = GpuDevice::new() {
            let info = dev.info();
            assert!(!info.name.is_empty());
            println!("GPU: {info}");
        } else {
            println!("No GPU available — skipping");
        }
    }
}
