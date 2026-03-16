//! Automatic GPU backend selection.

/// Available GPU compute backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuBackend {
    /// wgpu-based backend (Vulkan / Metal / DX12 / WebGPU).
    Wgpu,
    /// NVIDIA CUDA backend (requires `cuda` feature and CUDA toolkit).
    #[cfg(feature = "cuda")]
    Cuda,
}

impl GpuBackend {
    /// Detect the best available backend at runtime.
    ///
    /// When the `cuda` feature is enabled, CUDA is preferred if at least one
    /// NVIDIA device is present. Otherwise falls back to wgpu.
    pub fn detect() -> Self {
        #[cfg(feature = "cuda")]
        {
            if cuda_available() {
                return Self::Cuda;
            }
        }
        Self::Wgpu
    }

    /// Human-readable backend name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Wgpu => "wgpu",
            #[cfg(feature = "cuda")]
            Self::Cuda => "cuda",
        }
    }
}

impl std::fmt::Display for GpuBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// Check whether CUDA is available at runtime.
#[cfg(feature = "cuda")]
fn cuda_available() -> bool {
    crate::cuda::CudaDevice::count().map_or(false, |n| n > 0)
}
