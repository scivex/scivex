# Scivex Roadmap

> Generated 2026-04-10 from benchmark analysis (1178 results, scivex v0.1.1)

## Current State

**Published:**
- crates.io: all 16 crates at v0.1.1, umbrella at v0.1.2
- PyPI: pyscivex v0.1.1

**Performance overview (Rust vs Python):**

| Module | Median Speedup | Status |
|--------|---------------|--------|
| Sym | 68,458x | Excellent |
| RL | 295x | Excellent |
| Viz | 125x | Excellent |
| Graph | 44x | Excellent |
| Optim | 15x | Excellent |
| NLP | 11x | Great |
| IO | 5.5x | Great |
| ML | 5.4x | Great |
| Stats | 4.3x | Good |
| Frame | 1.8x | Needs work |
| Core | 1.2x | Needs work |
| NN | 1.2x | Needs work |
| GPU | 1.0x | Needs work |
| Image | 0.5x | Slower than Python |
| Signal | N/A | Slower than Python |

---

## Phase 1: Fix Regressions (v0.2.0)

Priority: operations where Rust is **slower** than Python.

### 1.1 Core Element-wise Ops (add, mul, scalar_mul) — 0.1x

**Root cause:** For large arrays (100K+), SIMD dispatch overhead + memory bandwidth saturation means we match but don't beat numpy's optimized C + MKL backend. At small sizes (1K), we're already 2.3x faster.

**Actions:**
- [ ] Add in-place `AddAssign`/`MulAssign` operators to eliminate allocation
- [ ] Implement tiled/blocked element-wise ops for cache friendliness at large N
- [ ] Benchmark against numpy without MKL to isolate the gap
- [ ] Consider BLAS-backed `scal`/`axpy` for scalar-vector ops

### 1.2 Core GEMM — 0.1x

**Root cause:** Naive 3-loop matmul vs numpy's BLAS (Accelerate/MKL). Our 4-way unrolled micro-kernel helps but doesn't match vendor-tuned libraries.

**Actions:**
- [ ] Implement Goto-style GEMM: MC×KC×NC blocking, pack A/B into contiguous panels
- [ ] Add register-level micro-kernel with full NEON/AVX utilization (6×16 for f64 NEON)
- [ ] Feature-gate optional linking to system BLAS via `blas-backend` feature
- [ ] Target: ≥0.5x vs numpy for n≥128, ≥1x for n≤64

### 1.3 Signal Filter — 0.01x

**Root cause:** SciPy uses optimized C + potentially FFT-based convolution for large filters. Our Direct Form II implementation is correct but slow.

**Actions:**
- [ ] Implement FFT-based convolution path for large filter orders (n > 64)
- [ ] SIMD-accelerate the inner FIR loop (vectorized multiply-accumulate)
- [ ] Add `sosfilt` (second-order sections) which is numerically better for high-order filters

### 1.4 Image Filter — 0.5x

**Root cause:** Naive 5-nested-loop convolution vs OpenCV/Pillow optimized backends.

**Actions:**
- [ ] Complete separable convolution for Gaussian/Sobel (currently partial)
- [ ] Add integral image optimization for box filters
- [ ] SIMD-accelerate convolution inner loop

### 1.5 NN Autograd Overhead — 0.0x-0.1x

**Root cause:** PyTorch uses C++/CUDA backend. Our autograd allocates tensors per-op and the gradient accumulation path allocates new tensors via `&*existing + g`.

**Actions:**
- [ ] In-place gradient accumulation with `AddAssign`
- [ ] Arena allocator for forward-pass temporaries
- [ ] Fused backward kernels (e.g., fused relu_backward + add_backward)
- [ ] Consider computation graph optimization (op fusion, dead node elimination)

### 1.6 Frame GroupBy — 0.1x

**Root cause:** String-based group keys. Every numeric value gets formatted to string for hashing.

**Actions:**
- [ ] Hash raw numeric bits (`u64::from_ne_bytes`) for numeric columns
- [ ] Pre-sort + run-length encoding path for sorted groups
- [ ] Parallel groupby with per-thread hash maps + merge

---

## Phase 2: Strengthen Advantages (v0.3.0)

Modules already faster than Python but with room to grow.

### 2.1 Core Decompositions

- [ ] Blocked Householder QR (currently column-by-column, 0.7x)
- [ ] Divide-and-conquer SVD (currently one-sided Jacobi, already 11.4x)
- [ ] Parallel eigendecomposition for large matrices

### 2.2 Stats Module

- [ ] SIMD-accelerated descriptive stats (variance, skewness, kurtosis)
- [ ] Parallel hypothesis testing for multiple-comparison scenarios
- [ ] Online/streaming statistics algorithms

### 2.3 ML Module

- [ ] Parallel tree construction (currently single-threaded)
- [ ] SIMD-accelerated distance computations for KNN/clustering
- [ ] Ensemble parallel fitting (random forest trees in parallel)

### 2.4 IO Module

- [ ] Parallel CSV parsing with per-chunk parsing
- [ ] Memory-mapped Parquet reading
- [ ] Arrow IPC zero-copy path

---

## Phase 3: Feature Completeness (v0.4.0)

### 3.1 Missing Python Equivalents

- [ ] `scivex-core`: `einsum`, `broadcast_to`, advanced indexing
- [ ] `scivex-frame`: `pivot_table`, `melt`, `rolling` windows, `resample`
- [ ] `scivex-stats`: Bayesian inference, MCMC sampling
- [ ] `scivex-ml`: Gradient boosting (XGBoost equivalent), neural architecture search
- [ ] `scivex-nn`: Convolutional layers, RNN/LSTM, attention mechanisms
- [ ] `scivex-signal`: Wavelet transforms, spectrogram
- [ ] `scivex-image`: JPEG/PNG decode (currently depends on external crates)

### 3.2 GPU Acceleration

- [ ] WebGPU compute shaders for GEMM, convolution, reduction
- [ ] Automatic CPU↔GPU transfer optimization
- [ ] GPU-backed autograd for neural network training

### 3.3 pyscivex Completeness

- [ ] Expose all Rust APIs through PyO3 bindings
- [ ] NumPy interop (zero-copy array protocol)
- [ ] DataFrame ↔ pandas conversion
- [ ] Multi-platform wheels (Linux x86_64, macOS ARM64, Windows)

---

## Phase 4: Production Readiness (v1.0.0)

### 4.1 Stability

- [ ] Full test coverage (currently varying across crates)
- [ ] Fuzzing for numeric edge cases (NaN, Inf, denormals)
- [ ] Property-based testing for mathematical invariants
- [ ] Miri runs for all unsafe blocks

### 4.2 Documentation

- [ ] Complete API docs with examples for every public function
- [ ] Cookbook with Python-to-Scivex migration guide
- [ ] Performance tuning guide (feature flags, allocation strategies)

### 4.3 Ecosystem

- [ ] `serde` support for all types (behind feature flag)
- [ ] Integration with Arrow ecosystem
- [ ] Jupyter kernel via pyscivex
- [ ] WASM build for browser-based computation

### 4.4 CI/CD

- [ ] Automated benchmark regression testing in CI
- [ ] Multi-platform release builds (GitHub Actions matrix)
- [ ] Automated crates.io + PyPI publishing on tag

---

## Release Plan

| Version | Target | Focus |
|---------|--------|-------|
| v0.1.1 | Done | SIMD operators, hot-path optimizations |
| v0.2.0 | Next | Fix all regressions (≥1x vs Python everywhere) |
| v0.3.0 | After | Strengthen advantages, parallel everywhere |
| v0.4.0 | After | Feature completeness, GPU, pyscivex parity |
| v1.0.0 | After | Production stability, full docs, ecosystem |

## Versioning Policy

- **Workspace version** (`Cargo.toml`): all crates share the same version via `version.workspace = true`
- **Umbrella crate** (`scivex`): may have a higher patch version when only re-exports change
- **pyscivex**: tracks the workspace version
- **Publishing order**: bottom-up through dependency graph (core → frame/stats → ... → umbrella → PyPI)
- **Branch protection**: all changes via PR to `master`, CI must pass
