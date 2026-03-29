# Scivex Benchmarks

Baseline performance measurements for core operations.

**Hardware:** Apple Silicon (M-series), macOS
**Rust:** 1.85+ (edition 2024)
**Profile:** `--release` with default opt-level
**Date:** 2026-03-24

## Linear Algebra (scivex-core)

| Operation | Size | Mean Time | Throughput | Notes |
|-----------|------|-----------|------------|-------|
| GEMM (f64) | 64x64 | 26.7 µs | 19.7 GFLOP/s | Blocked + NEON 4x4 micro-kernel |
| GEMM (f64) | 128x128 | 222 µs | 18.9 GFLOP/s | Blocked + NEON 4x4 micro-kernel |
| GEMM (f64) | 256x256 | 2.22 ms | 15.1 GFLOP/s | Blocked + NEON 4x4 micro-kernel |
| Dot product (f64) | 1,000 | 852 ns | 2.35 GFLOP/s | NEON vectorized |
| Dot product (f64) | 10,000 | 9.23 µs | 2.17 GFLOP/s | NEON vectorized |
| Dot product (f64) | 100,000 | 81.9 µs | 2.44 GFLOP/s | NEON vectorized |
| Transpose (f64) | 64x64 | 1.93 µs | — | |
| Transpose (f64) | 256x256 | 145 µs | — | |
| Element-wise add (f64) | 10,000 | 2.49 µs | — | NEON vectorized |
| Element-wise add (f64) | 100,000 | 21.0 µs | — | NEON vectorized |

## FFT (scivex-core)

| Operation | Size | Mean Time | Throughput | Notes |
|-----------|------|-----------|------------|-------|
| FFT (f64) | 256 | 3.54 µs | — | Mixed-radix, precomputed twiddles |
| FFT (f64) | 1024 | 12.6 µs | — | Mixed-radix, precomputed twiddles |
| FFT (f64) | 4096 | 50.1 µs | — | Mixed-radix, precomputed twiddles |

## DataFrame Operations (scivex-frame)

| Operation | Size | Mean Time | Notes |
|-----------|------|-----------|-------|
| Create DataFrame | 100K rows x 5 cols | TBD | |
| GroupBy + Sum | 100K rows, 100 groups | TBD | |
| Sort | 100K rows | TBD | |
| Filter | 100K rows | TBD | |
| Join (inner) | 10K x 10K rows | TBD | |
| CSV read | 100K rows | TBD | |

## Machine Learning (scivex-ml)

| Operation | Size | Mean Time | Notes |
|-----------|------|-----------|-------|
| RandomForest fit | 10K x 10, 100 trees | TBD | |
| RandomForest predict | 1K x 10 | TBD | |
| KMeans fit | 10K x 10, k=10 | TBD | |
| SVM fit (RBF) | 5K x 10 | TBD | |
| HistGBM fit | 10K x 10, 100 trees | TBD | |

## Neural Networks (scivex-nn)

| Operation | Size | Mean Time | Notes |
|-----------|------|-----------|-------|
| Linear forward | batch=32, 784→256 | TBD | |
| Conv2d forward | batch=1, 32x32x3, 3x3 | TBD | Uses SIMD-accelerated GEMM via im2col |
| Backward pass | 3-layer MLP | TBD | |
| Adam step | 10K parameters | TBD | |

## Statistics (scivex-stats)

| Operation | Size | Mean Time | Notes |
|-----------|------|-----------|-------|
| Descriptive stats | 100K elements | TBD | |
| Normal distribution CDF | 100K evaluations | TBD | |
| Linear regression | 10K x 5 | TBD | |
| t-test | 10K elements | TBD | |

## Signal Processing (scivex-signal)

| Operation | Size | Mean Time | Notes |
|-----------|------|-----------|-------|
| STFT | 44100 samples, 1024 FFT | TBD | |
| Mel spectrogram | 44100 samples | TBD | |
| MFCC | 44100 samples, 13 coeffs | TBD | |

## Comparison vs Python Libraries

| Operation | Scivex | NumPy | Ratio | Notes |
|-----------|--------|-------|-------|-------|
| Matmul 256x256 | 2.22 ms | ~0.15 ms | ~0.07x | NumPy uses Accelerate/OpenBLAS |
| FFT 4096 | 50.1 µs | ~15 µs | ~0.3x | NumPy wraps FFTW/vDSP |
| Dot 100K | 81.9 µs | ~20 µs | ~0.24x | NumPy uses Accelerate BLAS |

> Note: Python comparisons include interpreter overhead, which Scivex avoids.
> Large matrix operations (GEMM) will favor NumPy when it uses optimized C/Fortran
> backends like MKL or Apple Accelerate. Scivex is competitive for small-to-medium
> operations where Python overhead dominates and in pipeline workloads where
> avoiding serialization/deserialization between Rust and Python pays off.
>
> The primary value of Scivex is the unified, pure-Rust stack — no FFI overhead,
> no GIL, full control over memory layout, and a single dependency graph.

## Running Benchmarks

```bash
# All benchmarks
cargo bench --workspace

# Specific crate
cargo bench -p scivex-core

# Quick mode (fewer iterations)
cargo bench -p scivex-core -- --quick

# Quick timing (no Criterion overhead)
cargo run --release --example quick_bench -p scivex-core

# Save baseline for comparison
cargo bench -p scivex-core -- --save-baseline main

# Compare against baseline
cargo bench -p scivex-core -- --baseline main
```

## Profile-Guided Optimization (PGO)

For maximum performance, use PGO:

```bash
# Step 1: Build with profiling instrumentation
RUSTFLAGS="-Cprofile-generate=/tmp/pgo-data" cargo build --release -p scivex-core

# Step 2: Run benchmarks to generate profile data
cargo bench -p scivex-core

# Step 3: Merge profile data
llvm-profdata merge -o /tmp/pgo-data/merged.profdata /tmp/pgo-data

# Step 4: Rebuild with profile data
RUSTFLAGS="-Cprofile-use=/tmp/pgo-data/merged.profdata" cargo build --release -p scivex-core

# Expected improvement: 5-15% on hot paths (GEMM, FFT)
```
