# Tensors

The `Tensor<T>` type is the fundamental data structure in Scivex, analogous to
NumPy's `ndarray`. It stores elements in row-major (C) order, owns its data,
and is generic over any type implementing the `Scalar` trait.

This guide covers everything you need to work with tensors effectively.

## Creating Tensors

### From data

The most direct way to create a tensor is from a flat `Vec` or slice plus a
shape. The product of the shape dimensions must equal the data length.

```rust
use scivex_core::prelude::*;
use scivex_core::tensor::Tensor;

// 2x3 matrix from a flat Vec
let a = Tensor::from_vec(
    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    vec![2, 3],
).unwrap();
assert_eq!(a.shape(), &[2, 3]);
assert_eq!(a.numel(), 6);

// From a slice (copies the data)
let data = [10, 20, 30, 40];
let b = Tensor::from_slice(&data, vec![2, 2]).unwrap();

// Scalar (0-dimensional) tensor
let s = Tensor::scalar(42.0_f64);
assert_eq!(s.ndim(), 0);
assert_eq!(s.numel(), 1);
```

### Factory functions

```rust
use scivex_core::tensor::Tensor;

// All zeros
let z = Tensor::<f64>::zeros(vec![3, 4]);

// All ones
let o = Tensor::<f32>::ones(vec![2, 2]);

// Filled with a constant
let f = Tensor::full(vec![2, 3], 7_i32);

// Integer range [0, n)
let r = Tensor::<i32>::arange(5);
assert_eq!(r.as_slice(), &[0, 1, 2, 3, 4]);

// Identity matrix
let eye = Tensor::<f64>::eye(3);
assert_eq!(eye.shape(), &[3, 3]);

// Evenly spaced values (inclusive endpoints, requires Float)
let lin = Tensor::<f64>::linspace(0.0, 1.0, 5).unwrap();
assert_eq!(lin.shape(), &[5]);
// Values: 0.0, 0.25, 0.5, 0.75, 1.0
```

### Random tensors

Random tensor creation requires an explicit `Rng` -- there is no hidden global
state.

```rust
use scivex_core::tensor::Tensor;
use scivex_core::random::{Rng, uniform, uniform_range, normal, standard_normal, randint, bernoulli};

let mut rng = Rng::new(42); // seed for reproducibility

// Uniform [0, 1)
let u = uniform::<f64>(&mut rng, vec![2, 3]);

// Uniform in a custom range [low, high)
let ur = uniform_range::<f64>(&mut rng, vec![100], -1.0, 1.0).unwrap();

// Gaussian N(mean, std_dev)
let g = normal::<f64>(&mut rng, vec![3, 3], 0.0, 1.0);

// Standard normal N(0, 1) -- shorthand
let sn = standard_normal::<f64>(&mut rng, vec![1000]);

// Random integers in [low, high)
let ri = randint::<i32>(&mut rng, vec![10], 0, 100).unwrap();

// Bernoulli (0 or 1) with probability p
let b = bernoulli::<f64>(&mut rng, vec![100], 0.5).unwrap();
```

The `Rng` type is a xoshiro256\*\* generator. You can fork it into independent
child RNGs for parallel workloads:

```rust
use scivex_core::random::Rng;

let mut rng = Rng::new(0);
let children = rng.fork(4); // 4 independent RNGs
```

## Shape and Reshaping

### Inspecting shape

```rust
use scivex_core::tensor::Tensor;

let t = Tensor::from_vec(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();

assert_eq!(t.shape(), &[2, 3]);     // dimensions
assert_eq!(t.strides(), &[3, 1]);   // row-major strides
assert_eq!(t.ndim(), 2);            // rank
assert_eq!(t.numel(), 6);           // total elements
assert!(!t.is_empty());
```

### Reshape

Reshape changes the logical shape without copying data (consumes `self`).
The total number of elements must stay the same.

```rust
use scivex_core::tensor::Tensor;

let t = Tensor::from_vec(vec![1, 2, 3, 4, 5, 6], vec![6]).unwrap();
let t = t.reshape(vec![2, 3]).unwrap();
assert_eq!(t.shape(), &[2, 3]);

// Non-consuming version (clones the data)
let t2 = t.reshaped(vec![3, 2]).unwrap();
```

### Flatten

```rust
use scivex_core::tensor::Tensor;

let t = Tensor::from_vec(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
let flat = t.flatten(); // consumes, no copy
assert_eq!(flat.shape(), &[6]);

// Non-consuming version
let t = Tensor::from_vec(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
let flat = t.flattened(); // clones
```

### Transpose and permute

`transpose()` works on 2-D tensors and returns a new tensor with copied data.

```rust
use scivex_core::tensor::Tensor;

let t = Tensor::from_vec(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
let tt = t.transpose().unwrap();
assert_eq!(tt.shape(), &[3, 2]);
```

For higher-rank tensors, use `permute()`:

```rust
use scivex_core::tensor::Tensor;

let t = Tensor::<i32>::arange(24).reshape(vec![2, 3, 4]).unwrap();
let p = t.permute(&[2, 0, 1]).unwrap();
assert_eq!(p.shape(), &[4, 2, 3]);
```

### Squeeze and unsqueeze

```rust
use scivex_core::tensor::Tensor;

let t = Tensor::from_vec(vec![1, 2, 3], vec![3]).unwrap();

// Add a dimension of size 1
let t = t.unsqueeze(0).unwrap();
assert_eq!(t.shape(), &[1, 3]);

// Remove all size-1 dimensions
let t = t.squeeze();
assert_eq!(t.shape(), &[3]);
```

### Concatenate and stack

```rust
use scivex_core::tensor::Tensor;

let a = Tensor::from_vec(vec![1, 2, 3], vec![1, 3]).unwrap();
let b = Tensor::from_vec(vec![4, 5, 6], vec![1, 3]).unwrap();

// Concatenate along an existing axis
let c = Tensor::concat(&[&a, &b], 0).unwrap();
assert_eq!(c.shape(), &[2, 3]);

// Stack along a new axis
let a = Tensor::from_vec(vec![1, 2, 3], vec![3]).unwrap();
let b = Tensor::from_vec(vec![4, 5, 6], vec![3]).unwrap();
let s = Tensor::stack(&[&a, &b], 0).unwrap();
assert_eq!(s.shape(), &[2, 3]);
```

## Indexing and Slicing

### Element access

```rust
use scivex_core::tensor::Tensor;

let mut t = Tensor::from_vec(vec![10, 20, 30, 40], vec![2, 2]).unwrap();

// Read
assert_eq!(*t.get(&[0, 1]).unwrap(), 20);
assert_eq!(*t.get(&[1, 0]).unwrap(), 30);

// Write
t.set(&[0, 1], 99).unwrap();
assert_eq!(*t.get(&[0, 1]).unwrap(), 99);

// Mutable reference
*t.get_mut(&[1, 1]).unwrap() = 77;
```

### Slicing with `SliceRange`

`SliceRange` mirrors Python's `start:stop:step` notation. You supply one range
per axis.

```rust
use scivex_core::tensor::Tensor;
use scivex_core::tensor::SliceRange;

let t = Tensor::from_vec(
    vec![1, 2, 3, 4, 5, 6, 7, 8, 9],
    vec![3, 3],
).unwrap();

// Rows 0..2, columns 1..3
let s = t.slice(&[
    SliceRange::range(0, 2),
    SliceRange::range(1, 3),
]).unwrap();
assert_eq!(s.shape(), &[2, 2]);
assert_eq!(s.as_slice(), &[2, 3, 5, 6]);

// With step: every 3rd element
let v = Tensor::<i32>::arange(10);
let stepped = v.slice(&[SliceRange::new(0, 10, 3)]).unwrap();
assert_eq!(stepped.as_slice(), &[0, 3, 6, 9]);

// Full axis (select everything)
let full = t.slice(&[
    SliceRange::full(3),
    SliceRange::full(3),
]).unwrap();
assert_eq!(full, t);
```

### Select (dimension reduction)

`select()` picks one index along an axis, reducing the rank by 1.

```rust
use scivex_core::tensor::Tensor;

let t = Tensor::from_vec(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();

// Select row 1 -> 1-D tensor
let row = t.select(0, 1).unwrap();
assert_eq!(row.shape(), &[3]);
assert_eq!(row.as_slice(), &[4, 5, 6]);

// Select column 0 -> 1-D tensor
let col = t.select(1, 0).unwrap();
assert_eq!(col.shape(), &[2]);
assert_eq!(col.as_slice(), &[1, 4]);
```

### Fancy indexing

```rust
use scivex_core::tensor::Tensor;

let t = Tensor::from_vec(vec![10, 20, 30, 40, 50], vec![5]).unwrap();

// Select by index array (like np.take)
let s = t.index_select(0, &[4, 0, 2]).unwrap();
assert_eq!(s.as_slice(), &[50, 10, 30]);

// Boolean mask (flat)
let mask = vec![true, false, true, false, true];
let m = t.masked_select(&mask).unwrap();
assert_eq!(m.as_slice(), &[10, 30, 50]);
```

For 2-D tensors, `masked_select_along` selects rows where the mask is true:

```rust
use scivex_core::tensor::Tensor;

let t = Tensor::from_vec(vec![1, 2, 3, 4, 5, 6], vec![3, 2]).unwrap();
let mask = vec![false, true, true];
let s = t.masked_select_along(&mask).unwrap();
assert_eq!(s.shape(), &[2, 2]);
assert_eq!(s.as_slice(), &[3, 4, 5, 6]);
```

Scatter operations write values back into a tensor at specified indices:

```rust
use scivex_core::tensor::Tensor;

let mut t = Tensor::from_vec(vec![1, 2, 3, 4, 5], vec![5]).unwrap();

// Masked put
let mask = vec![false, true, false, true, false];
t.masked_put(&mask, &[99, 88]).unwrap();
assert_eq!(t.as_slice(), &[1, 99, 3, 88, 5]);
```

## Element-wise Arithmetic

Scivex implements the standard Rust operators (`+`, `-`, `*`, `/`) for
element-wise tensor arithmetic. Both owned and borrowed operands are supported.

```rust
use scivex_core::tensor::Tensor;

let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
let b = Tensor::from_vec(vec![10.0, 20.0, 30.0], vec![3]).unwrap();

// Tensor + Tensor (element-wise, same shape required)
let c = &a + &b; // borrows -- a and b remain usable
assert_eq!(c.as_slice(), &[11.0, 22.0, 33.0]);

let d = &a - &b;
let e = &a * &b;
let f = &a / &b;

// Tensor + scalar (broadcasts scalar to every element)
let g = &a * 10.0;
assert_eq!(g.as_slice(), &[10.0, 20.0, 30.0]);

let h = &a + 100.0;
let i = &a / 2.0;

// Negation (requires Float)
let neg = -&a;
assert_eq!(neg.as_slice(), &[-1.0, -2.0, -3.0]);
```

### Non-panicking variants

The operator overloads panic on shape mismatch. For fallible code, use the
`_checked` methods which return `Result`:

```rust
use scivex_core::tensor::Tensor;

let a = Tensor::from_vec(vec![1.0, 2.0], vec![2]).unwrap();
let b = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();

// Returns Err instead of panicking
assert!(a.add_checked(&b).is_err());
assert!(a.sub_checked(&b).is_err());
assert!(a.mul_checked(&b).is_err());
assert!(a.div_checked(&b).is_err());
```

### Map and apply

For custom element-wise operations:

```rust
use scivex_core::tensor::Tensor;

let t = Tensor::from_vec(vec![1, 2, 3, 4], vec![2, 2]).unwrap();

// map: returns a new tensor
let doubled = t.map(|x| x * 2);
assert_eq!(doubled.as_slice(), &[2, 4, 6, 8]);

// apply: modifies in place
let mut t = Tensor::from_vec(vec![1.0, 4.0, 9.0], vec![3]).unwrap();
t.apply(|x: f64| x.sqrt());
assert_eq!(t.as_slice(), &[1.0, 2.0, 3.0]);

// zip_map: combine two tensors element-wise
let a = Tensor::from_vec(vec![1, 2, 3], vec![3]).unwrap();
let b = Tensor::from_vec(vec![10, 20, 30], vec![3]).unwrap();
let c = a.zip_map(&b, |x, y| x + y).unwrap();
assert_eq!(c.as_slice(), &[11, 22, 33]);
```

## Broadcasting Rules

Scivex's element-wise operators (`+`, `-`, `*`, `/`) require tensors to have
identical shapes. Unlike NumPy, shapes are not implicitly broadcast -- this is
a deliberate design choice for safety and predictability.

To combine tensors of different shapes, use explicit reshaping or the
scalar-broadcast operators:

```rust
use scivex_core::tensor::Tensor;

// Scalar broadcast: every element is multiplied by the same value
let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
let scaled = &t * 5.0;

// For row/column broadcast, manually expand:
// Add a row vector [10, 20, 30] to every row of a 3x3 matrix
let mat = Tensor::from_vec(
    vec![1, 2, 3, 4, 5, 6, 7, 8, 9],
    vec![3, 3],
).unwrap();
// Use zip_map with matched shapes, or iterate with map
```

The error type `CoreError::BroadcastError` exists in the error enum for cases
where broadcasting is attempted on incompatible shapes in higher-level crates.

## Reductions

### Global reductions

```rust
use scivex_core::tensor::Tensor;

let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();

let s = t.sum();              // 10.0
let p = t.product();          // 24.0
let mn = t.min_element();     // Some(1.0)
let mx = t.max_element();     // Some(9.0)  -- returns Option
let avg = t.mean();           // 2.5 (requires Float)
```

`min_element()` and `max_element()` return `None` for empty tensors.

### Axis reductions

`sum_axis` reduces along a single axis, removing it from the shape:

```rust
use scivex_core::tensor::Tensor;

// [[1, 2, 3],
//  [4, 5, 6]]
let t = Tensor::from_vec(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();

// Sum along axis 0 (collapse rows) -> [5, 7, 9]
let s0 = t.sum_axis(0).unwrap();
assert_eq!(s0.shape(), &[3]);
assert_eq!(s0.as_slice(), &[5, 7, 9]);

// Sum along axis 1 (collapse columns) -> [6, 15]
let s1 = t.sum_axis(1).unwrap();
assert_eq!(s1.shape(), &[2]);
assert_eq!(s1.as_slice(), &[6, 15]);
```

### Sorting and argsort

```rust
use scivex_core::tensor::Tensor;

let t = Tensor::from_vec(vec![3, 1, 4, 1, 5, 9], vec![6]).unwrap();

// Flat sort (always returns 1-D)
let sorted = t.sort();
assert_eq!(sorted.as_slice(), &[1, 1, 3, 4, 5, 9]);

// Argsort: indices that would sort the data
let indices = t.argsort();
// indices.as_slice() gives the permutation

// Sort along a specific axis
let mat = Tensor::from_vec(vec![3, 1, 4, 2], vec![2, 2]).unwrap();
let sorted_rows = mat.sort_axis(1).unwrap();
assert_eq!(sorted_rows.as_slice(), &[1, 3, 2, 4]);
```

## Matrix Operations

### Matrix multiplication

```rust
use scivex_core::tensor::Tensor;

// matmul: matrix-matrix product
let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
let c = a.matmul(&b).unwrap();
assert_eq!(c.as_slice(), &[19.0, 22.0, 43.0, 50.0]);

// matvec: matrix-vector product
let x = Tensor::from_vec(vec![5.0, 6.0], vec![2]).unwrap();
let y = a.matvec(&x).unwrap();
assert_eq!(y.as_slice(), &[17.0, 39.0]);
```

### Dot product

```rust
use scivex_core::tensor::Tensor;

let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
let y = Tensor::from_vec(vec![4.0, 5.0, 6.0], vec![3]).unwrap();

// Method syntax
let d = x.dot(&y).unwrap();
assert!((d - 32.0).abs() < 1e-10);

// Or free function
use scivex_core::linalg::dot;
let d2 = dot(&x, &y).unwrap();
```

### Norm

```rust
use scivex_core::tensor::Tensor;

let x = Tensor::from_vec(vec![3.0_f64, 4.0], vec![2]).unwrap();
let n = x.norm().unwrap(); // L2 norm = 5.0
```

### BLAS-level functions

For fine-grained control, `scivex_core::linalg` exposes all three BLAS levels:

```rust
use scivex_core::tensor::Tensor;
use scivex_core::linalg::{axpy, scal, asum, nrm2, iamax, gemv, gemm};

// Level 1: axpy -- y = alpha * x + y (in-place)
let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
let mut y = Tensor::from_vec(vec![10.0, 20.0, 30.0], vec![3]).unwrap();
axpy(2.0, &x, &mut y).unwrap();
assert_eq!(y.as_slice(), &[12.0, 24.0, 36.0]);

// Level 1: scal -- x = alpha * x (in-place)
let mut v = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
scal(10.0, &mut v).unwrap();

// Level 1: asum -- sum of absolute values (L1 norm)
let v = Tensor::from_vec(vec![-1.0, 2.0, -3.0], vec![3]).unwrap();
let s = asum(&v).unwrap(); // 6.0

// Level 1: iamax -- index of element with largest absolute value
let v = Tensor::from_vec(vec![1.0_f64, -5.0, 3.0], vec![3]).unwrap();
assert_eq!(iamax(&v).unwrap(), Some(1));

// Level 2: gemv -- y = alpha * A * x + beta * y
let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
let x = Tensor::from_vec(vec![5.0, 6.0], vec![2]).unwrap();
let mut y = Tensor::<f64>::zeros(vec![2]);
gemv(1.0, &a, &x, 0.0, &mut y).unwrap();

// Level 3: gemm -- C = alpha * A * B + beta * C
let mut c = Tensor::<f64>::zeros(vec![2, 2]);
gemm(1.0, &a, &a, 0.0, &mut c).unwrap();
```

## Linear Algebra

All decompositions and solvers are implemented from scratch -- no external
BLAS/LAPACK bindings.

### Solving linear systems

```rust
use scivex_core::tensor::Tensor;
use scivex_core::linalg;

// Solve Ax = b
let a = Tensor::from_vec(vec![2.0_f64, 1.0, 1.0, 4.0], vec![2, 2]).unwrap();
let b = Tensor::from_vec(vec![5.0_f64, 6.0], vec![2]).unwrap();

// Free function
let x = linalg::solve(&a, &b).unwrap();

// Or method syntax
let x = a.solve(&b).unwrap();
assert!((x.as_slice()[0] - 2.0).abs() < 1e-10);
assert!((x.as_slice()[1] - 1.0).abs() < 1e-10);
```

### Matrix inverse

```rust
use scivex_core::tensor::Tensor;
use scivex_core::linalg;

let a = Tensor::from_vec(vec![2.0_f64, 1.0, 1.0, 4.0], vec![2, 2]).unwrap();

let inv = linalg::inv(&a).unwrap();
// Or: let inv = a.inv().unwrap();

// Verify: A * A^-1 ~ I
let eye = a.matmul(&inv).unwrap();
assert!((eye.as_slice()[0] - 1.0).abs() < 1e-10);
```

### Determinant

```rust
use scivex_core::tensor::Tensor;
use scivex_core::linalg;

let a = Tensor::from_vec(vec![2.0_f64, 1.0, 1.0, 4.0], vec![2, 2]).unwrap();
let det = linalg::det(&a).unwrap();
// Or: let det = a.det().unwrap();
assert!((det - 7.0).abs() < 1e-10);
```

### Least-squares

```rust
use scivex_core::tensor::Tensor;

// Overdetermined system: find x minimizing ||Ax - b||_2
let a = Tensor::from_vec(
    vec![1.0_f64, 1.0, 1.0, 2.0, 1.0, 3.0],
    vec![3, 2],
).unwrap();
let b = Tensor::from_vec(vec![6.0, 5.0, 7.0], vec![3]).unwrap();
let x = a.lstsq(&b).unwrap();
```

### LU decomposition

```rust
use scivex_core::tensor::Tensor;
use scivex_core::linalg::LuDecomposition;

let a = Tensor::from_vec(vec![2.0_f64, 1.0, 1.0, 4.0], vec![2, 2]).unwrap();
let lu = LuDecomposition::decompose(&a).unwrap();

let l = lu.l();          // lower triangular (unit diagonal)
let u = lu.u();          // upper triangular
let det = lu.det();      // determinant
let x = lu.solve(&b).unwrap();   // solve Ax = b
let inv = lu.inverse().unwrap();  // A^-1
```

### QR decomposition

```rust
use scivex_core::tensor::Tensor;
use scivex_core::linalg::QrDecomposition;

let a = Tensor::from_vec(
    vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
    vec![3, 2],
).unwrap();
let qr = QrDecomposition::decompose(&a).unwrap();

let q = qr.q();  // orthogonal matrix (m x m)
let r = qr.r();  // upper triangular (m x n)

// Verify Q is orthogonal: Q^T Q ~ I
let qtq = q.transpose().unwrap().matmul(&q).unwrap();
```

### Cholesky decomposition

For symmetric positive-definite matrices:

```rust
use scivex_core::tensor::Tensor;
use scivex_core::linalg::CholeskyDecomposition;

let a = Tensor::from_vec(vec![4.0_f64, 2.0, 2.0, 3.0], vec![2, 2]).unwrap();
let chol = CholeskyDecomposition::decompose(&a).unwrap();

let l = chol.l();              // lower triangular factor
let x = chol.solve(&b).unwrap();  // solve Ax = b
let inv = chol.inverse().unwrap();
```

### SVD (Singular Value Decomposition)

```rust
use scivex_core::tensor::Tensor;
use scivex_core::linalg::SvdDecomposition;

let a = Tensor::from_vec(vec![3.0_f64, 0.0, 0.0, 4.0], vec![2, 2]).unwrap();
let svd = SvdDecomposition::decompose(&a).unwrap();

let s = svd.singular_values();  // &[T], descending order
let u = svd.u();                // left singular vectors (m x m)
let vt = svd.vt();              // right singular vectors transposed (n x n)

// Singular values of diag(3,4) are [4.0, 3.0]
assert!((s[0] - 4.0).abs() < 1e-10);
assert!((s[1] - 3.0).abs() < 1e-10);
```

### Eigendecomposition

For symmetric matrices:

```rust
use scivex_core::tensor::Tensor;
use scivex_core::linalg::EigDecomposition;

let a = Tensor::from_vec(vec![2.0_f64, 1.0, 1.0, 3.0], vec![2, 2]).unwrap();
let eig = EigDecomposition::decompose_symmetric(&a).unwrap();

let vals: &[f64] = eig.eigenvalues();         // descending by |value|
let vecs: Tensor<f64> = eig.eigenvectors();   // columns are eigenvectors
```

## FFT Operations

The `scivex_core::fft` module provides forward and inverse FFTs for complex and
real-valued data. Complex data uses an interleaved layout: a tensor with
trailing dimension 2, where index 0 is real and index 1 is imaginary.

Arbitrary lengths are supported -- not just powers of two. The implementation
uses radix-2 Cooley-Tukey for power-of-2 sizes, mixed-radix (2/3/5/7)
for composite sizes, and Bluestein's chirp-z transform for prime lengths.

### 1-D transforms

```rust
use scivex_core::tensor::Tensor;
use scivex_core::fft;

// Real-to-complex FFT
let signal = Tensor::from_vec(vec![1.0, 0.0, -1.0, 0.0], vec![4]).unwrap();
let spectrum = fft::rfft(&signal).unwrap();
assert_eq!(spectrum.shape(), &[3, 2]); // N/2+1 complex bins

// Inverse: recover the signal
let recovered = fft::irfft(&spectrum, 4).unwrap();
assert_eq!(recovered.shape(), &[4]);

// Complex-to-complex FFT (input shape [N, 2])
let complex_signal = Tensor::from_vec(
    vec![1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0],
    vec![4, 2],
).unwrap();
let spectrum = fft::fft(&complex_signal).unwrap();
let recovered = fft::ifft(&spectrum).unwrap();
```

### 2-D transforms

```rust
use scivex_core::tensor::Tensor;
use scivex_core::fft;

// 2-D real FFT
let image = Tensor::from_vec(
    (0..16).map(|i| i as f64).collect(),
    vec![4, 4],
).unwrap();
let spectrum = fft::rfft2(&image).unwrap();
assert_eq!(spectrum.shape(), &[4, 3, 2]); // [M, N/2+1, 2]

// 2-D complex FFT / inverse
let complex_image = Tensor::from_vec(vec![0.0; 4 * 4 * 2], vec![4, 4, 2]).unwrap();
let spectrum = fft::fft2(&complex_image).unwrap();
let recovered = fft::ifft2(&spectrum).unwrap();
```

## Type Promotion and the Float Trait

### The trait hierarchy

Scivex tensors are generic over the `Scalar` trait hierarchy:

| Trait     | Purpose                                | Types                     |
|-----------|----------------------------------------|---------------------------|
| `Scalar`  | Base: arithmetic, Copy, zero/one       | all numeric types         |
| `Integer` | Adds remainder                         | i8, i16, i32, i64, u8 .. |
| `Float`   | Adds sqrt, sin, cos, exp, ln, etc.     | f32, f64                  |
| `Real`    | Alias for non-complex floats           | f32, f64                  |

Integer tensors are first-class: you can create, reshape, slice, add, and
reduce `Tensor<i32>` just like `Tensor<f64>`. Operations like `mean()`,
`linspace()`, and all linear algebra require `Float`.

### Casting between types

```rust
use scivex_core::tensor::Tensor;

let t_f32 = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0], vec![3]).unwrap();

// Cast f32 -> f64 (both sides must implement Float)
let t_f64: Tensor<f64> = t_f32.cast();
```

### Runtime dtype tags

The `promote` module provides numpy-style type promotion rules and runtime
type tags:

```rust
use scivex_core::promote::{DType, promote};

// Determine the result type when combining two dtypes
let result = promote(DType::I32, DType::F64);
assert_eq!(result, DType::F64); // integer + float -> float

let result = promote(DType::F32, DType::F64);
assert_eq!(result, DType::F64); // narrower float promotes to wider
```

## Performance Tips

### SIMD acceleration

Enable the `simd` feature flag to activate hand-tuned SIMD kernels for
f32 and f64 operations. On x86-64 this uses SSE/AVX; on aarch64 it uses NEON.

```toml
[dependencies]
scivex-core = { path = "../scivex-core", features = ["simd"] }
```

SIMD-accelerated operations include:

- Element-wise add and multiply (`add_simd`, `mul_simd`)
- Sum, min, max reductions
- Dot product
- axpy, scal, nrm2, asum (BLAS Level 1)

When `simd` is enabled, the generic implementations automatically dispatch to
SIMD where the element type is `f32` or `f64`. No code changes required for
`sum()`, `dot()`, `nrm2()`, etc. -- they use SIMD transparently.

For explicit SIMD calls on tensors:

```rust,ignore
// Only available with feature = "simd"
let a = Tensor::from_vec(vec![1.0_f64; 1000], vec![1000]).unwrap();
let b = Tensor::from_vec(vec![2.0_f64; 1000], vec![1000]).unwrap();
let c = a.add_simd(&b);
let d = a.mul_simd(&b);
```

### Parallel execution

Enable the `parallel` feature for Rayon-based parallel execution:

```toml
[dependencies]
scivex-core = { path = "../scivex-core", features = ["parallel"] }
```

### Cache-aware GEMM

The matrix multiplication kernel (`gemm`) uses blocked tiling with tile sizes
tuned for L1/L2 cache (64x256x256 blocks). The IKJ loop order within each
tile ensures the innermost loop is a contiguous AXPY, enabling auto-
vectorization. This means `matmul` performs well on large matrices without
any special configuration.

### General tips

- Prefer `reshape()` (consumes self, no copy) over `reshaped()` (clones)
  when you no longer need the original tensor.
- Use `flatten()` instead of `reshape(vec![n])` -- it skips the validation.
- Use `apply()` for in-place element-wise transforms instead of `map()` when
  you do not need to keep the original.
- Use `as_slice()` and `as_mut_slice()` for zero-copy access to the
  underlying data when interfacing with other code.
- Use `into_vec()` to consume a tensor and get the raw `Vec<T>` without
  copying.
- All errors are returned as `Result<T, CoreError>` -- check for
  `DimensionMismatch`, `InvalidShape`, `SingularMatrix`, etc.
