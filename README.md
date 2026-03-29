# Scivex

[![CI](https://github.com/scivex/scivex/actions/workflows/ci.yml/badge.svg)](https://github.com/scivex/scivex/actions)
[![codecov](https://codecov.io/gh/scivex/scivex/branch/master/graph/badge.svg)](https://codecov.io/gh/scivex/scivex)
[![Crates.io](https://img.shields.io/crates/v/scivex.svg)](https://crates.io/crates/scivex)
[![PyPI](https://img.shields.io/pypi/v/pyscivex.svg)](https://pypi.org/project/pyscivex/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![MSRV: 1.85.0](https://img.shields.io/badge/MSRV-1.85.0-orange.svg)](https://blog.rust-lang.org/)

**A comprehensive Rust library replacing the entire Python data science
ecosystem with a single, from-scratch Cargo workspace.**

One `use scivex::prelude::*;` gives you tensors, DataFrames, statistics,
machine learning, neural networks, visualization, signal processing, NLP,
graph analysis, image processing, symbolic math, reinforcement learning,
and GPU acceleration — all implemented from first principles in pure Rust.

---

## Why Scivex?

- **One dependency, full stack.** No juggling `numpy` + `pandas` + `scipy` +
  `sklearn` + `matplotlib` + `pytorch`. Scivex is a single unified ecosystem.
- **Zero external math dependencies.** The core implements tensors, BLAS, LAPACK
  decompositions, and FFT from scratch. No C/Fortran build toolchain required.
- **Type safe.** Generic over `Scalar > Float > Real` trait hierarchy. `Result<T>`
  everywhere — no panics in library code.
- **Cross-platform.** Compiles and tests on Linux, macOS, and Windows.
- **Feature-gated.** Only compile the sub-crates you need.
- **Python bindings.** Full Python API via [pyscivex](crates/pyscivex/) — `pip install pyscivex`.
- **WebAssembly ready.** Run in the browser via [scivex-wasm](crates/scivex-wasm/).

---

## Quick Start

### Rust

Add Scivex to your `Cargo.toml`:

```toml
[dependencies]
scivex = { version = "0.1", features = ["full"] }
```

Or pick only what you need:

```toml
[dependencies]
scivex = { version = "0.1", features = ["core", "frame", "stats", "ml"] }
```

### Python

```bash
pip install pyscivex
```

```python
import pyscivex as sv

# Tensors
a = sv.Tensor([[1.0, 2.0], [3.0, 4.0]])
b = sv.Tensor.ones([2, 2])
c = a + b

# DataFrames
df = sv.DataFrame()
df.add_column("x", [1.0, 2.0, 3.0])

# ML
model = sv.ml.RandomForestClassifier(n_trees=100, max_depth=5)
```

---

## Rust Examples

### Tensor Operations

```rust
use scivex::prelude::*;

// Create tensors
let a = Tensor::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
let b = Tensor::<f64>::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]);

// Matrix multiplication
let c = scivex::core::linalg::matmul(&a, &b).unwrap();

// Decompositions
let lu = scivex::core::linalg::LuDecomposition::decompose(&a).unwrap();
let qr = scivex::core::linalg::QrDecomposition::decompose(&a).unwrap();
let svd = scivex::core::linalg::SvdDecomposition::decompose(&a).unwrap();
```

### DataFrames

```rust
use scivex::prelude::*;

let df = DataFrameBuilder::new()
    .add_column(Series::new("name", vec!["Alice", "Bob", "Carol"]))
    .add_column(Series::new("age", vec![30i32, 25, 35]))
    .add_column(Series::new("score", vec![95.0f64, 87.5, 92.0]))
    .build()
    .unwrap();

let adults = df.filter_expr("age", FilterOp::Gt, &30).unwrap();
let grouped = df.group_by(&["name"]).unwrap();
```

### Machine Learning

```rust
use scivex::prelude::*;

let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, &mut rng);

let mut rf = RandomForestClassifier::new(100, 5);
rf.fit(&x_train, &y_train).unwrap();
let predictions = rf.predict(&x_test).unwrap();

let acc = accuracy(&y_test, &predictions);
let f1 = f1_score(&y_test, &predictions);
```

### Statistics

```rust
use scivex::prelude::*;

let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];

let m = mean(&data);
let s = std_dev(&data);
let q = quantile(&data, 0.75);

// Hypothesis testing
let result = t_test_one_sample(&data, 4.0).unwrap();
println!("t = {}, p = {}", result.statistic, result.p_value);

// Distributions
let normal = Normal::new(0.0, 1.0);
let p = normal.cdf(1.96);  // ~ 0.975
```

### Neural Networks

```rust
use scivex::prelude::*;

let mut model = Sequential::new(vec![
    Box::new(Linear::new(784, 128, &mut rng)),
    Box::new(ReLU),
    Box::new(Dropout::new(0.2)),
    Box::new(Linear::new(128, 10, &mut rng)),
]);

let mut optimizer = Adam::new(model.parameters(), 0.001);
let loss = cross_entropy_loss(&logits, &targets);
loss.backward(None);
optimizer.step();
```

### Visualization

```rust
use scivex::prelude::*;

let fig = Figure::new()
    .size(800.0, 600.0)
    .add_axes(
        Axes::new()
            .title("Training Loss")
            .x_label("Epoch")
            .y_label("Loss")
            .add_plot(LinePlot::new(&epochs, &losses).color(Color::BLUE))
    );

let svg = fig.render(&SvgBackend).unwrap();
std::fs::write("loss.svg", svg).unwrap();
```

---

## Sub-Crates

| Crate | Replaces | Highlights |
|-------|----------|------------|
| [`scivex-core`](crates/scivex-core/) | NumPy, SciPy.linalg | Tensors, BLAS L1-L3, LU/QR/SVD/Cholesky/Eig, FFT, sparse matrices, PRNG |
| [`scivex-frame`](crates/scivex-frame/) | Pandas | DataFrames, Series, joins, groupby, pivot, rolling windows, string ops |
| [`scivex-io`](crates/scivex-io/) | Pandas I/O | CSV, JSON, Parquet, Arrow, Excel, SQLite, PostgreSQL, NPY, HDF5, ORC, Avro |
| [`scivex-stats`](crates/scivex-stats/) | SciPy.stats, statsmodels | 15+ distributions, hypothesis tests, correlation, regression, time series |
| [`scivex-optim`](crates/scivex-optim/) | SciPy.optimize | Root finding, BFGS, Nelder-Mead, L-BFGS-B, linear programming, curve fitting, ODE solvers |
| [`scivex-viz`](crates/scivex-viz/) | Matplotlib, Seaborn | Line/scatter/bar/histogram/heatmap/boxplot, SVG + terminal backends |
| [`scivex-ml`](crates/scivex-ml/) | scikit-learn | Linear models, trees, random forests, gradient boosting, SVM, k-NN, K-Means, pipelines, SHAP |
| [`scivex-nn`](crates/scivex-nn/) | PyTorch, TensorFlow | Autograd, Linear/Conv/RNN/Transformer/Attention, SGD/Adam/AdamW, mixed precision |
| [`scivex-image`](crates/scivex-image/) | Pillow, OpenCV | Image types, resize, crop, rotate, blur, edge detection, morphology, PNG/JPEG/BMP |
| [`scivex-signal`](crates/scivex-signal/) | SciPy.signal | FIR/IIR filters, STFT, spectrograms, peak detection, wavelets, resampling |
| [`scivex-graph`](crates/scivex-graph/) | NetworkX | Directed/undirected graphs, Dijkstra, BFS/DFS, PageRank, MST, SCC, max flow |
| [`scivex-nlp`](crates/scivex-nlp/) | NLTK, spaCy | Tokenizers (BPE, WordPiece, Unigram), stemming, TF-IDF, embeddings, sentiment |
| [`scivex-sym`](crates/scivex-sym/) | SymPy | Expression AST, differentiation, simplification, equation solving, polynomials |
| [`scivex-gpu`](crates/scivex-gpu/) | CuPy, CUDA | GPU tensor ops via wgpu compute shaders, matrix multiply, element-wise kernels |
| [`scivex-rl`](crates/scivex-rl/) | Gymnasium, Stable-Baselines3 | Environments, DQN, PPO, A2C, replay buffers, multi-agent support |

### Language Bindings

| Crate | Target | Description |
|-------|--------|-------------|
| [`pyscivex`](crates/pyscivex/) | Python (PyPI) | Full Python API via PyO3 — `pip install pyscivex` |
| [`scivex-wasm`](crates/scivex-wasm/) | JavaScript (npm) | WebAssembly bindings for browsers and Node.js |
| [`scivex-ffi`](crates/scivex-ffi/) | C/Julia/R | C FFI shared library for any language with C interop |

---

## Feature Flags

| Feature | Enables | Default |
|---------|---------|---------|
| `core` | Tensors, linear algebra, FFT, math primitives | Yes |
| `frame` | DataFrames, Series, GroupBy, joins | |
| `stats` | Distributions, hypothesis tests, regression | |
| `io` | CSV, JSON, Parquet, Arrow, Excel, SQL I/O | |
| `optim` | Optimization, root finding, integration, ODE | |
| `viz` | Visualization, plotting, chart rendering | |
| `ml` | Classical ML: trees, ensembles, clustering, pipelines | |
| `nn` | Neural networks, autograd, layers, optimizers | |
| `image` | Image loading, transforms, filters | |
| `signal` | Signal processing, FFT, wavelets, audio | |
| `graph` | Graph data structures, algorithms, network | |
| `nlp` | Tokenization, embeddings, text processing | |
| `sym` | Symbolic math, CAS, expression simplification | |
| `gpu` | GPU-accelerated tensor operations via wgpu | |
| `rl` | Reinforcement learning environments and agents | |
| `full` | All of the above | |

Additional flags: `simd`, `parallel`, `serde-support`, `image-png`, `image-jpeg`, `nn-gpu`.

---

## Building from Source

```bash
# Build the entire workspace
cargo build --workspace

# Run all tests (1800+ tests)
cargo test --workspace

# Lint
cargo clippy --workspace --all-targets -- -D warnings

# Format
cargo fmt --all

# Build documentation
cargo doc --workspace --no-deps --open
```

## Minimum Supported Rust Version

The MSRV is **1.85.0** (Rust edition 2024).

## Architecture

All crates are implemented from first principles with zero external
dependencies for core math. The dependency graph flows downward:

```
                          scivex (umbrella)
                                |
      +---------+---------+-----+-----+---------+---------+
      |         |         |           |         |         |
  scivex-nn  scivex-ml  scivex-nlp  viz     signal     image
      |         |         |           |         |         |
      |    scivex-rl      |           |         |         |
      |         |         |           |         |         |
      +---------+---------+     scivex-graph    |         |
      |                               |         |         |
  scivex-optim                        |   scivex-gpu      |
      |                               |         |         |
      +------+--------+------+--------+---------+---------+
             |
         scivex-stats
             |
         scivex-frame -- scivex-io
             |
         scivex-core  (foundation -- zero external deps for math)
```

Language bindings (`pyscivex`, `scivex-wasm`, `scivex-ffi`) depend on the
full stack and are not published to crates.io.

## Contributing

Contributions are welcome! Please ensure your changes pass all CI checks:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

## License

Licensed under the [MIT License](LICENSE).
