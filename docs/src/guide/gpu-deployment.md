# GPU Computing & Deployment

This guide covers GPU-accelerated computing, model export, and deployment options.

## GPU Backend Setup

Scivex uses [wgpu](https://wgpu.rs/) for GPU acceleration, supporting Vulkan (Linux/Windows),
Metal (macOS), DirectX 12 (Windows), and WebGPU (browsers).

### Feature Flags

```toml
[dependencies]
scivex = { version = "0.1", features = ["gpu", "nn-gpu"] }
```

- `gpu` — GPU tensor operations, matrix multiply, element-wise ops
- `nn-gpu` — GPU-accelerated neural network training (autograd on GPU)

### Device Selection

```rust
use scivex_gpu::prelude::*;

// Auto-select best available GPU
let device = GpuDevice::auto()?;
println!("Using: {}", device.name());

// List all available devices
let devices = GpuDevice::enumerate()?;
for d in &devices {
    println!("  {} ({:?})", d.name(), d.backend());
}
```

## GPU Tensor Operations

```rust
use scivex_gpu::prelude::*;

let device = GpuDevice::auto()?;

// Create GPU tensors
let a = GpuTensor::from_vec(vec![1.0f32; 1024 * 1024], vec![1024, 1024], &device)?;
let b = GpuTensor::from_vec(vec![2.0f32; 1024 * 1024], vec![1024, 1024], &device)?;

// GPU matrix multiplication (uses optimized WGSL shaders with tiling)
let c = a.matmul(&b)?;

// Element-wise operations stay on GPU
let d = a.add(&b)?;
let e = a.mul(&b)?;

// Transfer back to CPU when needed
let cpu_tensor = c.to_cpu()?;
println!("Result shape: {:?}", cpu_tensor.shape());
```

## GPU Neural Network Training

```rust
use scivex_nn::prelude::*;
use scivex_gpu::prelude::*;

let device = GpuDevice::auto()?;

// Create GPU-backed model
let model = Sequential::new()
    .add(Linear::new(784, 256))
    .add(ReLU::new())
    .add(Linear::new(256, 10));

// GPU optimizer
let mut optimizer = GpuAdam::new(model.parameters(), 0.001, &device)?;

// Training loop
for epoch in 0..10 {
    // Data is automatically transferred to GPU
    let x_gpu = GpuVariable::from_tensor(&x_batch, &device)?;
    let y_gpu = GpuVariable::from_tensor(&y_batch, &device)?;

    let output = model.forward(&x_gpu);
    let loss = cross_entropy_loss(&output, &y_gpu);

    loss.backward();
    optimizer.step();
    optimizer.zero_grad();

    println!("Epoch {}: loss = {:.4}", epoch, loss.data().to_cpu()?.sum());
}
```

## Memory Management

```rust
// GPU memory is managed through Rust's ownership system.
// Tensors are freed when dropped.

{
    let big_tensor = GpuTensor::zeros(vec![4096, 4096], &device)?;
    // ... use tensor
} // big_tensor is freed here, GPU memory released

// For fine-grained control, use the memory pool
let pool = GpuMemoryPool::new(&device, 1024 * 1024 * 512)?; // 512 MB pool
let t = pool.alloc(vec![1024, 1024])?;
```

## Model Export & Persistence

### ONNX Export

```rust
use scivex_nn::onnx;

// Export trained model to ONNX format
onnx::export(&model, "model.onnx", &[1, 784])?; // input shape

// Load ONNX model for inference
let loaded = onnx::load("model.onnx")?;
let output = loaded.run(&input_tensor)?;
```

### SafeTensors Format

```rust
use scivex_nn::persistence;

// Save weights in SafeTensors format (compatible with HuggingFace)
persistence::save_safetensors(model.parameters(), "model.safetensors")?;

// Load weights
let params = persistence::load_safetensors("model.safetensors")?;
model.load_parameters(&params)?;
```

### GGUF Format

```rust
use scivex_nn::persistence;

// Save in GGUF format (compatible with llama.cpp)
persistence::save_gguf(model.parameters(), "model.gguf")?;
```

### Binary Format

```rust
use scivex_nn::persistence;

// Save/load in Scivex native binary format (fastest)
persistence::save_weights(model.parameters(), "model.bin")?;
persistence::load_weights("model.bin")?;
```

## Inference Server

```rust
use scivex_nn::serve::{InferenceServer, InferenceModel};

// Load a trained model
let model = onnx::load("model.onnx")?;

// Create inference server
let server = InferenceServer::new()
    .model(model)
    .port(8080)
    .workers(4)
    .batch_size(32);

// Start serving (blocks)
server.serve()?;
// Accepts POST /predict with JSON tensor input
```

## Deployment Options

### Native Binary

```bash
# Build optimized release binary
cargo build --release -p my_inference_app

# The binary includes all Scivex code — no Python runtime needed
./target/release/my_inference_app
```

### WebAssembly

```bash
# Build for web deployment
wasm-pack build crates/scivex-wasm --target web

# The generated .wasm runs in any modern browser
# Includes: tensors, ML inference, basic stats
```

### FFI / C Bindings

```rust
use scivex_nn::ffi;

// Export model as C-compatible function
#[no_mangle]
pub extern "C" fn predict(input: *const f32, len: usize, output: *mut f32) {
    // Load model once (lazy static or similar)
    // Run inference
    // Write results to output pointer
}
```

### Docker

```dockerfile
FROM rust:1.85-slim AS builder
COPY . .
RUN cargo build --release -p inference-server

FROM debian:bookworm-slim
COPY --from=builder /target/release/inference-server /usr/local/bin/
EXPOSE 8080
CMD ["inference-server"]
```
