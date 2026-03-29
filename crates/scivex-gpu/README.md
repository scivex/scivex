# scivex-gpu

GPU-accelerated tensor operations for Scivex via wgpu compute shaders.
Works on any GPU that supports Vulkan, Metal, or DX12 — no CUDA required.

## Highlights

- **GpuDevice** — Device discovery, queue management, and buffer allocation
- **GpuTensor** — GPU-resident tensors with host upload/download
- **Matrix multiply** — Tiled WGSL compute shader for gemm
- **Element-wise ops** — add, sub, mul, div, exp, log, sqrt, relu on GPU
- **Reduction** — sum, mean, max, min across axes
- **Cross-platform** — Vulkan (Linux/Windows), Metal (macOS), DX12 (Windows)

## Usage

```rust
use scivex_gpu::prelude::*;

let device = GpuDevice::new().unwrap();
let a = GpuTensor::from_slice(&device, &[1.0f32, 2.0, 3.0, 4.0], &[2, 2]);
let b = GpuTensor::from_slice(&device, &[5.0f32, 6.0, 7.0, 8.0], &[2, 2]);

let c = device.matmul(&a, &b).unwrap();
let result = c.to_vec();
```

## License

MIT
