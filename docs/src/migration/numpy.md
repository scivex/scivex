# NumPy to Scivex Migration Guide

This guide maps NumPy operations to their Scivex equivalents. All examples use
`scivex_core` directly. Every API listed here exists in the source code of the
`scivex-core` crate.

## Setup

**Python (NumPy)**
```python
import numpy as np
```

**Rust (Scivex)**
```rust
use scivex_core::prelude::*;            // Tensor, Float, Scalar, etc.
use scivex_core::{fft, math, random};
use scivex_core::linalg;
use scivex_core::tensor::SliceRange;
```

---

## Quick Reference Table

| NumPy | Scivex | Notes |
|---|---|---|
| `np.array([1,2,3])` | `Tensor::from_vec(vec![1,2,3], vec![3])?` | Returns `Result` |
| `np.zeros((2,3))` | `Tensor::<f64>::zeros(vec![2,3])` | Requires explicit type |
| `np.ones((2,3))` | `Tensor::<f64>::ones(vec![2,3])` | |
| `np.full((2,3), 7.0)` | `Tensor::full(vec![2,3], 7.0)` | |
| `np.arange(5)` | `Tensor::<i32>::arange(5)` | Integer indices |
| `np.linspace(0,1,5)` | `Tensor::<f64>::linspace(0.0, 1.0, 5)?` | Requires `n >= 2` |
| `np.eye(3)` | `Tensor::<f64>::eye(3)` | |
| `a.shape` | `t.shape()` | Returns `&[usize]` |
| `a.ndim` | `t.ndim()` | |
| `a.size` | `t.numel()` | |
| `a[i,j]` | `t.get(&[i,j])?` | Returns `Result<&T>` |
| `a[i,j] = v` | `t.set(&[i,j], v)?` | Returns `Result<()>` |
| `a[0:2, 1:3]` | `t.slice(&[SliceRange::range(0,2), SliceRange::range(1,3)])?` | |
| `a[::2]` | `t.slice(&[SliceRange::new(0, n, 2)])?` | |
| `a[i]` (row select) | `t.select(0, i)?` | Reduces dimensionality |
| `a + b` | `&a + &b` or `a + b` | Element-wise, same shape |
| `a + 5.0` | `&a + 5.0` or `a + 5.0` | Scalar broadcast |
| `a * b` | `&a * &b` | Element-wise multiply |
| `-a` | `-&a` or `-a` | Requires `Float` |
| `np.sin(a)` | `t.sin()` or `math::sin(&t)` | |
| `np.exp(a)` | `t.exp()` or `math::exp(&t)` | |
| `np.sqrt(a)` | `t.sqrt()` or `math::sqrt(&t)` | |
| `a.sum()` | `t.sum()` | Returns scalar `T` |
| `a.sum(axis=0)` | `t.sum_axis(0)?` | Returns `Result<Tensor>` |
| `a.mean()` | `t.mean()` | Requires `Float` |
| `a.min()` | `t.min_element()` | Returns `Option<T>` |
| `a.max()` | `t.max_element()` | Returns `Option<T>` |
| `a.prod()` | `t.product()` | |
| `a.reshape(2,3)` | `t.reshape(vec![2,3])?` | Consumes `self` |
| `a.flatten()` | `t.flatten()` | Consumes `self` |
| `a.T` | `t.transpose()?` | 2-D only; copies data |
| `np.transpose(a, axes)` | `t.permute(&axes)?` | Arbitrary rank |
| `a[np.newaxis,:]` | `t.unsqueeze(0)?` | |
| `np.squeeze(a)` | `t.squeeze()` | |
| `np.concatenate([a,b], axis=0)` | `Tensor::concat(&[&a,&b], 0)?` | |
| `np.stack([a,b], axis=0)` | `Tensor::stack(&[&a,&b], 0)?` | |
| `np.sort(a)` | `t.sort()` | Returns flat 1-D |
| `np.argsort(a)` | `t.argsort()` | Returns `Tensor<usize>` |
| `np.sort(a, axis=0)` | `t.sort_axis(0)?` | |
| `np.dot(x,y)` | `linalg::dot(&x, &y)?` | 1-D vectors only |
| `a @ b` | `a.matmul(&b)?` | 2-D matrices |
| `np.linalg.solve(A,b)` | `linalg::solve(&a, &b)?` | |
| `np.linalg.inv(A)` | `linalg::inv(&a)?` | |
| `np.linalg.det(A)` | `linalg::det(&a)?` | |
| `np.linalg.norm(x)` | `linalg::nrm2(&x)?` | L2 norm, 1-D |
| `np.fft.fft(x)` | `fft::fft(&x)?` | Input shape `[N,2]` |
| `np.fft.rfft(x)` | `fft::rfft(&x)?` | Input shape `[N]` |
| `np.random.rand(2,3)` | `random::uniform(&mut rng, vec![2,3])` | Explicit `Rng` |
| `np.random.randn(2,3)` | `random::standard_normal(&mut rng, vec![2,3])` | |

---

## Array Creation

### From data

**NumPy**
```python
a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
b = np.array([[1, 2, 3], [4, 5, 6]])
```

**Scivex**
```rust
let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![6])?;
let b = Tensor::from_vec(vec![1, 2, 3, 4, 5, 6], vec![2, 3])?;

// From a borrowed slice (copies data):
let data = [1.0, 2.0, 3.0];
let c = Tensor::from_slice(&data, vec![3])?;

// Scalar (0-dimensional):
let s = Tensor::scalar(42.0_f64);
assert_eq!(s.ndim(), 0);
```

### Factory functions

**NumPy**
```python
z = np.zeros((3, 4))
o = np.ones((2, 2))
f = np.full((2, 3), 7.0)
r = np.arange(10)
l = np.linspace(0, 1, 50)
e = np.eye(4)
```

**Scivex**
```rust
let z = Tensor::<f64>::zeros(vec![3, 4]);
let o = Tensor::<f64>::ones(vec![2, 2]);
let f = Tensor::full(vec![2, 3], 7.0_f64);
let r = Tensor::<i32>::arange(10);
let l = Tensor::<f64>::linspace(0.0, 1.0, 50)?;
let e = Tensor::<f64>::eye(4);
```

Key difference: Scivex requires an explicit element type annotation when the
compiler cannot infer it. NumPy defaults to `float64`; Scivex has no default.

---

## Indexing and Slicing

### Single-element access

**NumPy**
```python
val = a[1, 2]       # read
a[1, 2] = 99.0      # write
```

**Scivex**
```rust
let val = *t.get(&[1, 2])?;    // returns Result<&T>
t.set(&[1, 2], 99.0)?;         // returns Result<()>
```

### Slicing

**NumPy**
```python
sub = a[0:2, 1:3]      # rows 0..2, cols 1..3
every_other = a[::2]    # step of 2
row = a[1]              # select row 1
```

**Scivex**
```rust
let sub = t.slice(&[
    SliceRange::range(0, 2),
    SliceRange::range(1, 3),
])?;

let every_other = t.slice(&[
    SliceRange::new(0, t.shape()[0], 2),
])?;

let row = t.select(0, 1)?;   // reduces rank by 1
let col = t.select(1, 0)?;   // select column 0
```

Slicing always returns a new tensor with copied data. There are no views or
shared-memory slices.

### Fancy indexing

**NumPy**
```python
selected = a[[4, 0, 2]]             # integer array indexing
masked = a[mask]                      # boolean mask
a[mask] = values                      # masked assignment
```

**Scivex**
```rust
let selected = t.index_select(0, &[4, 0, 2])?;
let masked = t.masked_select(&mask)?;             // flat 1-D result
let masked_rows = t.masked_select_along(&mask)?;  // preserves dims

t.masked_put(&mask, &values)?;
t.index_put(0, &[0, 2], &replacement_tensor)?;
```

### Gather

**NumPy / PyTorch**
```python
# torch.gather(input, dim, index)
```

**Scivex**
```rust
let result = t.gather(axis, &index_tensor)?;
```

---

## Element-wise Operations

### Arithmetic

**NumPy**
```python
c = a + b          # element-wise add
c = a - b          # element-wise subtract
c = a * b          # element-wise multiply (Hadamard)
c = a / b          # element-wise divide
c = a + 5.0        # scalar broadcast
c = -a             # negation
```

**Scivex**
```rust
let c = &a + &b;          // borrows; both a and b remain usable
let c = &a - &b;
let c = &a * &b;
let c = &a / &b;
let c = &a + 5.0;         // scalar broadcast (right-hand side only)
let c = -&a;              // requires T: Float

// Consuming variants (a and b are moved):
let c = a + b;

// Non-panicking variants that return Result:
let c = a.add_checked(&b)?;
let c = a.sub_checked(&b)?;
let c = a.mul_checked(&b)?;
let c = a.div_checked(&b)?;
```

The operator overloads (`+`, `-`, `*`, `/`) panic on shape mismatch. Use the
`_checked` methods when you need `Result`-based error handling.

### Custom element-wise operations

**NumPy**
```python
b = np.vectorize(lambda x: x**2 + 1)(a)
```

**Scivex**
```rust
let b = a.map(|x| x * x + 1.0);               // unary
let c = a.zip_map(&b, |x, y| x + y)?;          // binary, same shape
a.apply(|x| x * 2.0);                          // in-place mutation
```

### Math functions

**NumPy**
```python
np.sin(a)
np.cos(a)
np.tan(a)
np.exp(a)
np.log(a)
np.log2(a)
np.log10(a)
np.sqrt(a)
np.abs(a)
np.floor(a)
np.ceil(a)
np.round(a)
np.power(a, 3.0)
np.clip(a, 0.0, 1.0)
```

**Scivex**
```rust
t.sin();           // or math::sin(&t)
t.cos();           // or math::cos(&t)
t.tan();           // or math::tan(&t)
t.exp();           // or math::exp(&t)
t.ln();            // or math::ln(&t)       -- note: ln, not log
t.log2();          // or math::log2(&t)
t.log10();         // or math::log10(&t)
t.sqrt();          // or math::sqrt(&t)
t.abs();           // or math::abs(&t)
t.floor();         // or math::floor(&t)
t.ceil();          // or math::ceil(&t)
t.round();         // or math::round(&t)
t.powf(3.0);       // float exponent
t.powi(3);         // integer exponent
t.clamp(0.0, 1.0);
t.recip();         // 1/x, equivalent to np.reciprocal
```

All math methods require `T: Float`. Each returns a new `Tensor<T>` (no
in-place variants except `.apply()`).

---

## Reductions

**NumPy**
```python
a.sum()
a.prod()
a.min()
a.max()
a.mean()
a.sum(axis=0)
```

**Scivex**
```rust
t.sum();                 // -> T
t.product();             // -> T
t.min_element();         // -> Option<T>   (None for empty tensors)
t.max_element();         // -> Option<T>
t.mean();                // -> T           (requires Float)
t.sum_axis(0)?;          // -> Result<Tensor<T>>
```

Note: `min_element` and `max_element` return `Option` rather than panicking on
empty input. `mean` is only available for `Float` types.

---

## Shape Manipulation

**NumPy**
```python
b = a.reshape(2, 3)
b = a.flatten()
b = a.T                     # transpose
b = np.transpose(a, (2,0,1))
b = a[np.newaxis, :]         # add dimension
b = np.squeeze(a)
c = np.concatenate([a, b], axis=0)
c = np.stack([a, b], axis=0)
```

**Scivex**
```rust
let b = t.reshape(vec![2, 3])?;      // consumes t, no copy
let b = t.reshaped(vec![2, 3])?;     // clones, t remains valid
let b = t.flatten();                  // consumes t
let b = t.flattened();                // clones
let b = t.transpose()?;              // 2-D only, copies data
let b = t.permute(&[2, 0, 1])?;     // arbitrary rank
let b = t.unsqueeze(0)?;             // insert axis, consumes t
let b = t.squeeze();                 // remove all size-1 dims, consumes t
let c = Tensor::concat(&[&a, &b], 0)?;
let c = Tensor::stack(&[&a, &b], 0)?;
```

`reshape` and `flatten` consume the tensor (take ownership) and avoid copying
data. Use `reshaped` and `flattened` when you need to keep the original.

---

## Sorting

**NumPy**
```python
sorted_a = np.sort(a)
indices = np.argsort(a)
sorted_by_col = np.sort(a, axis=0)
```

**Scivex**
```rust
let sorted = t.sort();                  // returns flat 1-D, ascending
let indices = t.argsort();              // returns Tensor<usize>
let sorted_by_col = t.sort_axis(0)?;   // preserves shape
let idx_by_col = t.argsort_axis(0)?;   // returns Tensor<usize>, same shape
```

---

## Linear Algebra

### Dot product and matrix multiplication

**NumPy**
```python
d = np.dot(x, y)       # vector dot product
c = a @ b               # matrix multiply
c = np.matmul(a, b)
```

**Scivex**
```rust
let d = linalg::dot(&x, &y)?;       // 1-D vectors only
let c = a.matmul(&b)?;              // 2-D matrices: [m,k] x [k,n] -> [m,n]
```

### BLAS operations

**NumPy**
```python
# These map to underlying BLAS calls in NumPy/SciPy
np.linalg.norm(x)           # L2 norm
np.sum(np.abs(x))           # L1 norm (asum)
```

**Scivex**
```rust
linalg::nrm2(&x)?;          // L2 norm, 1-D vector
linalg::asum(&x)?;          // L1 norm (sum of absolute values)
linalg::scal(2.0, &mut x)?; // x *= 2.0 in place
linalg::axpy(2.0, &x, &mut y)?;  // y += 2.0 * x
linalg::iamax(&x)?;         // index of max absolute value
linalg::gemv(alpha, &a, &x, beta, &mut y)?;  // y = alpha*A*x + beta*y
linalg::gemm(alpha, &a, &b, beta, &mut c)?;  // C = alpha*A*B + beta*C
```

### Decompositions and solvers

**NumPy**
```python
x = np.linalg.solve(A, b)
inv_a = np.linalg.inv(A)
d = np.linalg.det(A)

# Decompositions
P, L, U = scipy.linalg.lu(A)
Q, R = np.linalg.qr(A)
L = np.linalg.cholesky(A)
U, s, Vt = np.linalg.svd(A)
eigenvalues, eigenvectors = np.linalg.eig(A)

# Least squares
x, residuals, rank, sv = np.linalg.lstsq(A, b)
```

**Scivex**
```rust
let x = linalg::solve(&a, &b)?;
let inv_a = linalg::inv(&a)?;
let d = linalg::det(&a)?;

// LU decomposition: PA = LU
let lu = linalg::LuDecomposition::decompose(&a)?;
let x = lu.solve(&b)?;
let inv = lu.inverse()?;
let det = lu.det();

// QR decomposition: A = QR
let qr = linalg::QrDecomposition::decompose(&a)?;

// Cholesky decomposition: A = L * L^T (symmetric positive definite)
let chol = linalg::CholeskyDecomposition::decompose(&a)?;

// SVD: A = U * diag(s) * V^T
let svd = linalg::SvdDecomposition::decompose(&a)?;

// Eigendecomposition: A = V * diag(d) * V^T (symmetric)
let eig = linalg::EigDecomposition::decompose(&a)?;

// Least squares
let x = linalg::lstsq(&a, &b)?;
```

### Sparse matrices

Scivex provides three sparse matrix formats:

```rust
use scivex_core::linalg::{CsrMatrix, CscMatrix, CooMatrix};
```

These correspond to SciPy's `scipy.sparse.csr_matrix`, `csc_matrix`, and
`coo_matrix`.

---

## FFT

Scivex represents complex numbers as tensors with a trailing dimension of
size 2: `[..., 2]` where index 0 is real and index 1 is imaginary.

**NumPy**
```python
spectrum = np.fft.fft(signal)
signal = np.fft.ifft(spectrum)
spectrum = np.fft.rfft(real_signal)
signal = np.fft.irfft(spectrum, n=original_length)
spectrum_2d = np.fft.fft2(data)
spectrum_2d = np.fft.ifft2(data)
spectrum_2d = np.fft.rfft2(real_data)
```

**Scivex**
```rust
// Complex FFT: input and output shape [N, 2]
let spectrum = fft::fft(&complex_signal)?;
let recovered = fft::ifft(&spectrum)?;

// Real FFT: input shape [N], output shape [N/2+1, 2]
let spectrum = fft::rfft(&real_signal)?;
let recovered = fft::irfft(&spectrum, original_length)?;

// 2-D complex FFT: input and output shape [M, N, 2]
let spectrum = fft::fft2(&complex_2d)?;
let recovered = fft::ifft2(&spectrum)?;

// 2-D real FFT: input [M, N], output [M, N/2+1, 2]
let spectrum = fft::rfft2(&real_2d)?;
```

Key difference: NumPy uses a native complex type. Scivex uses interleaved
`[real, imag]` pairs in the last dimension. To build complex input:

```rust
// Build a complex signal: [1+0i, 0+1i, -1+0i, 0-1i]
let complex = Tensor::from_vec(
    vec![1.0, 0.0,  0.0, 1.0,  -1.0, 0.0,  0.0, -1.0],
    vec![4, 2],
)?;
let spectrum = fft::fft(&complex)?;
```

Supports arbitrary lengths (not just powers of two) via mixed-radix
Cooley-Tukey for composite sizes and Bluestein's algorithm for primes.

---

## Random Number Generation

NumPy uses implicit global state. Scivex requires an explicit `Rng` instance,
making it thread-safe and reproducible by construction.

**NumPy**
```python
np.random.seed(42)
a = np.random.rand(3, 4)          # uniform [0, 1)
b = np.random.randn(3, 4)         # standard normal
c = np.random.uniform(2.0, 5.0, (3, 4))
d = np.random.normal(10.0, 2.0, (3, 4))
e = np.random.randint(0, 10, (3, 4))
np.random.shuffle(arr)
s = np.random.choice(arr, size=5, replace=False)
```

**Scivex**
```rust
use scivex_core::random::{self, Rng};

let mut rng = Rng::new(42);                              // explicit seed

let a = random::uniform::<f64>(&mut rng, vec![3, 4]);               // [0, 1)
let b = random::standard_normal::<f64>(&mut rng, vec![3, 4]);       // N(0, 1)
let c = random::uniform_range::<f64>(&mut rng, vec![3, 4], 2.0, 5.0)?;
let d = random::normal::<f64>(&mut rng, vec![3, 4], 10.0, 2.0);
let e = random::randint::<i32>(&mut rng, vec![3, 4], 0, 10)?;

random::shuffle(&mut rng, &mut tensor);                  // in-place

let s = random::choice(&mut rng, &tensor, 5, false)?;   // without replacement
let s = random::choice(&mut rng, &tensor, 5, true)?;    // with replacement

// Bernoulli distribution (0/1 values with probability p)
let mask = random::bernoulli::<f64>(&mut rng, vec![3, 4], 0.5)?;

// Fork RNG for parallel workloads
let children: Vec<Rng> = rng.fork(4);
```

---

## Type Casting

**NumPy**
```python
b = a.astype(np.float32)
```

**Scivex**
```rust
let b: Tensor<f32> = a.cast::<f32>();   // requires T: Float, U: Float
```

The `cast` method uses `to_f64()` then `from_f64()` internally. Conversions
like `f64` to `f32` are intentionally lossy.

---

## Key Differences

### 1. Ownership and borrowing

NumPy arrays are reference-counted objects with shared mutable state. Scivex
tensors follow Rust's ownership rules:

```rust
let a = Tensor::<f64>::zeros(vec![3, 3]);
let b = a + Tensor::<f64>::ones(vec![3, 3]);  // a is MOVED into the addition
// a is no longer usable here

// Use references to avoid moves:
let a = Tensor::<f64>::zeros(vec![3, 3]);
let b = &a + &Tensor::<f64>::ones(vec![3, 3]);  // a is borrowed, still usable
println!("{:?}", a.shape());  // OK
```

### 2. Explicit error handling

NumPy raises Python exceptions. Scivex returns `Result<T, CoreError>`:

```rust
// NumPy: raises ValueError on shape mismatch
// Scivex: returns Err(CoreError::DimensionMismatch { .. })
let result = a.add_checked(&b);
match result {
    Ok(tensor) => { /* use tensor */ }
    Err(CoreError::DimensionMismatch { expected, got }) => {
        eprintln!("Shape mismatch: expected {:?}, got {:?}", expected, got);
    }
    Err(e) => { eprintln!("Other error: {e}"); }
}
```

Common error variants:
- `CoreError::DimensionMismatch` -- shape mismatch in binary operations
- `CoreError::InvalidShape` -- invalid reshape target
- `CoreError::IndexOutOfBounds` -- index exceeds dimension
- `CoreError::AxisOutOfBounds` -- axis index exceeds rank
- `CoreError::InvalidArgument` -- general invalid parameter
- `CoreError::SingularMatrix` -- non-invertible matrix in linalg

### 3. Generic numeric types

NumPy defaults to `float64`. Scivex is generic over a trait hierarchy:

```
Scalar          -- all numeric types: Copy + arithmetic + zero/one
  Integer       -- i8, i16, i32, i64, u8, u16, u32, u64, usize
  Float         -- f32, f64: adds sin/cos/exp/ln/sqrt/abs/...
    Real        -- non-complex floats (currently same as Float)
```

Many operations are available on `Scalar` (sums, products, slicing,
reshaping). Math functions like `sin`, `exp`, `mean` require `Float`.

### 4. No implicit broadcasting

NumPy broadcasts shapes automatically (e.g., adding a `(3,)` array to a
`(2, 3)` array). Scivex element-wise operators require matching shapes.
Scalar-tensor operations (`tensor + scalar`) are supported.

To replicate NumPy broadcasting, explicitly reshape or tile before operating.

### 5. No views; copies everywhere

NumPy slicing returns a view that shares memory with the original array.
Scivex slicing (`slice`, `select`, `transpose`) always returns a new tensor
with its own data. This avoids aliasing bugs but uses more memory.

Use `reshape` (which consumes self) for zero-copy shape changes.

### 6. Row-major storage

Both NumPy (C order, the default) and Scivex store data in row-major order.
Element `[i, j]` of a `[M, N]` tensor is at flat index `i * N + j`.

### 7. Explicit RNG state

NumPy uses global or thread-local random state. Scivex requires passing an
`&mut Rng` to every random function, making reproducibility explicit:

```rust
let mut rng = Rng::new(42);
let a = random::uniform::<f64>(&mut rng, vec![100]);
// Exact same sequence every time with seed 42
```
