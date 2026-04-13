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

### 1.1 Core Element-wise Ops (add, mul, scalar_mul) — ~~0.1x~~ FIXED

**Status:** SIMD wired into `+`/`-`/`*`/`/` operators via TypeId dispatch (PR #24). In-place `AddAssign`/`SubAssign`/`MulAssign`/`DivAssign` with SIMD dispatch added. At small N (1K): 2.3x. At large N (100K+): 0.8-0.9x (memory bandwidth parity with numpy's MKL backend).

**Done:**
- [x] Wire SIMD kernels into standard operators via TypeId dispatch
- [x] Add in-place `AddAssign`/`MulAssign` operators to eliminate allocation
- [x] Add SIMD f32 kernels (add, sub, mul, div)
- [x] Add missing f64 SIMD kernels (sub, div)

**Remaining (Phase 2):**
- [ ] Tiled/blocked element-wise ops for cache friendliness at large N

### 1.2 Core GEMM — ~~0.1x~~ FIXED

**Status:** Optional `blas-backend` feature links to system BLAS (Accelerate on macOS, OpenBLAS on Linux) via CBLAS FFI. TypeId dispatch routes f64/f32 GEMM/GEMV to vendor-tuned libraries. Without blas-backend, our hand-tuned blocked GEMM is competitive at small N.

**Done:**
- [x] Feature-gate optional linking to system BLAS via `blas-backend` feature
- [x] CBLAS FFI for dgemm/sgemm/dgemv/sgemv
- [x] TypeId dispatch in `gemm`/`gemv` to system BLAS

**Remaining (Phase 2):**
- [ ] Implement Goto-style GEMM for non-BLAS path
- [ ] Register-level micro-kernel with full NEON/AVX utilization

### 1.3 Signal Filter — ~~0.01x~~ ~0.76x (benchmark mismatch)

**Status:** Original "0.01x" was a benchmark parameter mismatch, not a real regression. Our Direct Form II Transposed implementation with unsafe indexing runs at ~0.76x vs SciPy. Direct convolution was tested but is slower for large filter orders due to cache locality of DF2T state vector.

**Remaining (Phase 2):**
- [x] FFT-based convolution with auto-dispatch (O(N log N) for min(na,nb) > 64)
- [x] f64 fast path with fused multiply-add for FIR/IIR inner loops
- [x] f64 fast path with 4-way FMA unrolling for 1D convolution

### 1.4 Image Filter — ~~0.5x~~ IMPROVED

**Status:** Gaussian blur already used separable convolution. Sobel X/Y now use separable 1D passes instead of general 2D convolution. Box blur uses O(1)-per-pixel running sum approach.

**Done:**
- [x] Separable convolution for Sobel (two 1D passes)
- [x] Running-sum box blur (O(1) per pixel, radius-independent)
- [x] Branchless interior + bounds-checked border split in convolve2d

**Done (Phase 2):**
- [x] FMA-accelerated convolution inner loops (4-way ILP unrolling in convolve2d, gaussian_blur, separable_convolve)

### 1.5 NN Autograd Overhead — ~~0.0x-0.1x~~ IMPROVED

**Status:** Gradient accumulation now uses in-place SIMD-accelerated `AddAssign`. Forward/backward pass GEMM benefits from blas-backend. Remaining gap is autograd overhead (tensor allocation per op, graph traversal).

**Done:**
- [x] In-place gradient accumulation with SIMD-backed `AddAssign`
- [x] GEMM acceleration via blas-backend

**Remaining (Phase 2):**
- [ ] Arena allocator for forward-pass temporaries
- [ ] Fused backward kernels
- [ ] Computation graph optimization

### 1.6 Frame GroupBy — ~~0.1x~~ IMPROVED

**Status:** Replaced string-based hashing with raw bytes identity hash for numeric columns. Cache-friendly sequential scan with `group_ids: Vec<u32>` for aggregation (Sum/Min/Max/Mean). Group IDs built directly during grouping phase.

**Done:**
- [x] Hash raw numeric bits (identity hash) for numeric columns
- [x] Cache-friendly aggregation via sequential `group_ids` scan
- [x] Direct group_id building during grouping (no post-hoc reconstruction)
- [x] Single-column fast path (u64 hash key, no Vec alloc per row)

**Remaining (Phase 2):**
- [x] Pre-sort detection + run-length encoding path for sorted groups (O(n) no hashing)
- [ ] Parallel groupby with per-thread hash maps + merge

### 1.7 RNG (uniform/normal/randint) — IMPROVED

**Status:** TypeId dispatch for f64/f32 fast paths eliminates per-element `from_f64` conversion. Pre-allocated buffers with unsafe `set_len` instead of iterator collect.

**Done:**
- [x] Direct f64/f32 generation without type conversion overhead
- [x] Pre-allocated buffer with unsafe set_len

---

## Phase 2: Strengthen Advantages (v0.3.0)

Modules already faster than Python but with room to grow.

### 2.1 Core Decompositions

- [x] Blocked Householder QR (compact WY representation, BLOCK_SIZE=32)
- [ ] Divide-and-conquer SVD (currently one-sided Jacobi, already 11.4x)
- [ ] Parallel eigendecomposition for large matrices

### 2.2 Stats Module

- [x] SIMD-accelerated descriptive stats (4-way f64 accumulators for mean, variance)
- [x] Parallel hypothesis testing (`batch_t_test_one_sample`, `batch_t_test_two_sample` with Rayon `parallel` feature)
- [x] Online/streaming statistics (Welford's algorithm with Terriberry extensions for skewness/kurtosis, Chan's parallel merge)

### 2.3 ML Module

- [x] Sorted-scan split finding for decision trees (O(n log n) vs O(n²) per feature)
- [x] SIMD-accelerated distance computations for KNN/clustering (4-way f64 accumulators)
- [x] Ensemble parallel fitting (random forest `fit()` auto-dispatches to `par_fit()` with `parallel` feature)

### 2.4 IO Module

- [x] Zero-copy CSV column building (eliminate row→column transpose, build series directly from records)
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
