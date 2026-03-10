# Scivex

[![CI](https://github.com/scivex/scivex/actions/workflows/ci.yml/badge.svg)](https://github.com/scivex/scivex/actions)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![MSRV: 1.85.0](https://img.shields.io/badge/MSRV-1.85.0-orange.svg)](https://blog.rust-lang.org/)

**A comprehensive Rust library replacing the entire Python data science
ecosystem with a single, from-scratch Cargo workspace.**

One `use scivex::prelude::*;` gives you tensors, DataFrames, statistics,
machine learning, neural networks, visualization, signal processing, NLP,
graph analysis, image processing, and symbolic math — all implemented from
first principles in pure Rust.

---

## Why Scivex?

- **One dependency, full stack.** No juggling `numpy` + `pandas` + `scipy` +
  `sklearn` + `matplotlib`. Scivex is a single unified ecosystem.
- **Zero external math dependencies.** The core implements tensors, BLAS, LAPACK
  decompositions, and FFT from scratch. No C/Fortran build toolchain required.
- **Type safe.** Generic over `Scalar > Float > Real` trait hierarchy. `Result<T>`
  everywhere — no panics in library code.
- **Cross-platform.** Compiles and tests on Linux, macOS, and Windows.
- **Feature-gated.** Only compile the sub-crates you need.

---

## Quick Start

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
```

### DataFrames

```rust
use scivex::prelude::*;

// Build a DataFrame
let df = DataFrameBuilder::new()
    .add_column(Series::new("name", vec!["Alice", "Bob", "Carol"]))
    .add_column(Series::new("age", vec![30i32, 25, 35]))
    .add_column(Series::new("score", vec![95.0f64, 87.5, 92.0]))
    .build()
    .unwrap();

// Filter, group, aggregate
let adults = df.filter_expr("age", FilterOp::Gt, &30).unwrap();
let grouped = df.group_by(&["name"]).unwrap();
```

### Machine Learning

```rust
use scivex::prelude::*;

// Prepare data
let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, &mut rng);

// Train a random forest
let mut rf = RandomForestClassifier::new(100, 5);
rf.fit(&x_train, &y_train).unwrap();
let predictions = rf.predict(&x_test).unwrap();

// Evaluate
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
let p = normal.cdf(1.96);  // ≈ 0.975
```

### Neural Networks

```rust
use scivex::prelude::*;

// Build a network
let mut model = Sequential::new(vec![
    Box::new(Linear::new(784, 128, &mut rng)),
    Box::new(ReLU),
    Box::new(Dropout::new(0.2)),
    Box::new(Linear::new(128, 10, &mut rng)),
]);

// Train with Adam optimizer
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

// Render to SVG
let svg = fig.render(&SvgBackend).unwrap();
std::fs::write("loss.svg", svg).unwrap();
```

---

## Sub-Crates

| Crate | Replaces | Highlights |
|-------|----------|------------|
| [`scivex-core`](crates/scivex-core/) | NumPy, SciPy.linalg | Tensors, BLAS L1-L3, LU/QR/SVD/Cholesky/Eig, FFT, sparse matrices, PRNG |
| [`scivex-frame`](crates/scivex-frame/) | Pandas | DataFrames, Series, joins, groupby, pivot, rolling windows, string ops |
| [`scivex-io`](crates/scivex-io/) | Pandas I/O | CSV reader/writer, JSON reader/writer, auto type inference |
| [`scivex-stats`](crates/scivex-stats/) | SciPy.stats, statsmodels | 10 distributions, hypothesis tests, correlation, OLS regression |
| [`scivex-optim`](crates/scivex-optim/) | SciPy.optimize | Root finding, minimization (gradient descent, BFGS), numerical integration |
| [`scivex-viz`](crates/scivex-viz/) | Matplotlib, Seaborn | Line/scatter/bar/histogram/heatmap/boxplot, SVG + terminal backends |
| [`scivex-ml`](crates/scivex-ml/) | scikit-learn | Linear models, trees, random forests, k-NN, K-Means, Naive Bayes, metrics |
| [`scivex-nn`](crates/scivex-nn/) | PyTorch, TensorFlow | Autograd, Linear/Dropout/BatchNorm, SGD/Adam, MSE/CrossEntropy losses |
| [`scivex-image`](crates/scivex-image/) | Pillow, OpenCV | Image types, resize, crop, rotate, blur, edge detection, BMP/PPM I/O |
| [`scivex-signal`](crates/scivex-signal/) | SciPy.signal | FIR/IIR filters, STFT, spectrograms, peak detection, wavelets, resampling |
| [`scivex-graph`](crates/scivex-graph/) | NetworkX | Directed/undirected graphs, Dijkstra, BFS/DFS, PageRank, MST, SCC |
| [`scivex-nlp`](crates/scivex-nlp/) | NLTK, spaCy | Tokenizers, Porter stemmer, TF-IDF, word embeddings, sentiment analysis |
| [`scivex-sym`](crates/scivex-sym/) | SymPy | Expression AST, differentiation, simplification, equation solving, polynomials |

---

## Feature Flags

| Feature | Enables | Default |
|---------|---------|---------|
| `core` | Tensors, linear algebra, FFT, math primitives | Yes |
| `frame` | DataFrames, Series, GroupBy, joins | |
| `stats` | Distributions, hypothesis tests, regression | |
| `io` | CSV and JSON reading/writing | |
| `optim` | Optimization, root finding, integration | |
| `viz` | Visualization, plotting, chart rendering | |
| `ml` | Classical ML: trees, ensembles, clustering | |
| `nn` | Neural networks, autograd, layers, optimizers | |
| `image` | Image loading, transforms, filters | |
| `signal` | Signal processing, FFT, wavelets, audio | |
| `graph` | Graph data structures, algorithms, network | |
| `nlp` | Tokenization, embeddings, text processing | |
| `sym` | Symbolic math, CAS, expression simplification | |
| `full` | All of the above | |

---

## Building from Source

```bash
# Build the entire workspace
cargo build --workspace

# Run all tests (855 tests)
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
        +----------+--------+-+--+--------+---------+
        |          |        |    |        |         |
    scivex-nn  scivex-ml  nlp  viz    signal    image
        |          |        |    |        |         |
        +----------+--------+   |        |         |
        |                       |        |         |
    scivex-optim          scivex-graph   |         |
        |                       |        |         |
        +----------+------------+--------+---------+
        |
    scivex-stats
        |
    scivex-frame -- scivex-io
        |
    scivex-core  (foundation -- zero external deps for math)
```

## Contributing

Contributions are welcome. Please ensure your changes pass all CI checks:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

## License

Licensed under the [MIT License](LICENSE).
