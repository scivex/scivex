# Installation & Feature Flags

## Basic Installation

Add Scivex to your `Cargo.toml`:

```toml
[dependencies]
scivex = "0.1"
```

This gives you the `core` feature (tensors, linear algebra, FFT) by default.

## Feature Flags

Scivex uses feature flags so you only compile what you need:

```toml
[dependencies]
scivex = { version = "0.1", features = ["full"] }
```

### Available Features

| Feature | What it enables | Default |
|---------|----------------|---------|
| `core` | Tensors, BLAS, LAPACK, FFT, type promotion | Yes |
| `frame` | DataFrames, Series, GroupBy, joins, lazy eval | No |
| `stats` | Distributions, hypothesis tests, regression, GLM | No |
| `io` | CSV, JSON, Parquet, HDF5, Arrow IPC, Excel | No |
| `optim` | Optimization, integration, ODE solvers, curve fitting | No |
| `viz` | Plotting, charts, SVG/PNG/terminal/HTML rendering | No |
| `ml` | Decision trees, SVM, KNN, clustering, ensembles, pipelines | No |
| `nn` | Neural networks, autograd, layers, optimizers, ONNX | No |
| `image` | Image loading, convolution, morphology, transforms | No |
| `signal` | FFT, STFT, wavelets, filters, spectrograms | No |
| `graph` | Graph algorithms, shortest path, PageRank | No |
| `nlp` | Tokenization, TF-IDF, embeddings, text processing | No |
| `sym` | Symbolic math, CAS, differentiation | No |
| `gpu` | GPU tensors via wgpu (Vulkan/Metal/DX12/WebGPU) | No |
| `rl` | Reinforcement learning environments and agents | No |
| `full` | All features except `gpu` | No |

### Modifier Features

| Feature | Description |
|---------|-------------|
| `simd` | Enable hand-tuned SIMD kernels (AVX2 on x86_64, NEON on aarch64) |
| `parallel` | Enable Rayon-based parallelism for tensor and DataFrame ops |
| `serde-support` | Derive `Serialize`/`Deserialize` on all public types |
| `nn-gpu` | Neural network training on GPU |
| `frame-regex` | Regex support for DataFrame string operations |
| `image-png` | PNG image format support |
| `image-jpeg` | JPEG image format support |

### Common Configurations

**Data science workstation:**
```toml
scivex = { version = "0.1", features = ["full", "simd", "parallel"] }
```

**ML inference only:**
```toml
scivex = { version = "0.1", features = ["core", "ml", "nn", "io"] }
```

**GPU training:**
```toml
scivex = { version = "0.1", features = ["nn", "gpu", "nn-gpu"] }
```

**Minimal data analysis:**
```toml
scivex = { version = "0.1", features = ["frame", "io", "stats", "viz"] }
```

**WebAssembly (via scivex-wasm):**
```toml
scivex-wasm = { path = "crates/scivex-wasm" }
```

## Minimum Supported Rust Version

Scivex requires **Rust edition 2024** (stable Rust 1.85+).

## Building from Source

```bash
git clone https://github.com/scivex/scivex.git
cd scivex

# Build everything
cargo build --workspace

# Run all tests
cargo test --workspace

# Build docs
cargo doc --workspace --no-deps --open
```

## Using Individual Crates

You can depend on individual crates directly for finer control:

```toml
[dependencies]
scivex-core = "0.1"
scivex-frame = "0.1"
```

Each crate has its own prelude:

```rust
use scivex_core::prelude::*;
use scivex_frame::prelude::*;
```
