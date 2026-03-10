# scivex-core

The foundation crate of the Scivex ecosystem. Provides N-dimensional tensors,
linear algebra, FFT, and math primitives — all implemented from scratch with
zero external dependencies for math.

## Highlights

- **Tensor<T>** — Dynamic N-dimensional array with row-major storage and broadcasting
- **BLAS Level 1-3** — `dot`, `axpy`, `nrm2`, `gemv`, `gemm` from scratch
- **Matrix decompositions** — LU, QR, Cholesky, SVD, Eigendecomposition
- **Linear solvers** — `solve(A, b)`, `inv(A)`, `det(A)`, `lstsq(A, b)`
- **Sparse matrices** — COO, CSR, CSC formats
- **FFT** — Real and complex transforms (Cooley-Tukey radix-2)
- **PRNG** — xoshiro256** with normal/uniform sampling
- **35+ math functions** — sin, cos, exp, ln, sqrt, abs, etc. (element-wise on tensors)
- **Generic** — All operations generic over `Scalar > Float > Real` trait hierarchy

## Usage

```rust
use scivex_core::prelude::*;

// Create tensors
let a = Tensor::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
let b = Tensor::<f64>::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]);

// Arithmetic with broadcasting
let c = &a + &b;
let d = &a * &b;

// Linear algebra
let x = scivex_core::linalg::solve(&a, &vec![1.0, 2.0]).unwrap();
let svd = scivex_core::linalg::SvdDecomposition::decompose(&a).unwrap();

// FFT
let spectrum = scivex_core::fft::rfft(&signal);
```

## License

MIT
