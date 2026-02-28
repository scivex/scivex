# Scivex

A comprehensive Rust library replacing the entire Python data science ecosystem with a single, from-scratch Cargo workspace.

> **Status:** Early development (Phase 0 — Bootstrap)

## Vision

One `use scivex::prelude::*;` gives you tensors, dataframes, statistics, machine learning, neural networks, visualization, and more — all implemented from first principles in pure Rust. No wrapping C/Fortran libraries.

## Sub-Crates

| Crate | Replaces | Status |
|---|---|---|
| `scivex-core` | NumPy, SciPy.linalg | In progress |
| `scivex-frame` | Pandas | Planned |
| `scivex-io` | Pandas I/O, PyArrow, h5py | Planned |
| `scivex-stats` | SciPy.stats, statsmodels | Planned |
| `scivex-optim` | SciPy.optimize, SciPy.integrate | Planned |
| `scivex-viz` | Matplotlib, Seaborn, Plotly | Planned |
| `scivex-ml` | scikit-learn, XGBoost, LightGBM | Planned |
| `scivex-nn` | TensorFlow, PyTorch, Keras, JAX | Planned |
| `scivex-image` | Pillow, OpenCV, scikit-image | Planned |
| `scivex-signal` | SciPy.signal, librosa | Planned |
| `scivex-graph` | NetworkX, igraph | Planned |
| `scivex-nlp` | NLTK, spaCy, Gensim | Planned |
| `scivex-sym` | SymPy | Planned |

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
scivex = "0.1"
```

```rust
use scivex::prelude::*;
```

## Building from Source

```bash
# Build the entire workspace
cargo build --workspace

# Run all tests
cargo test --workspace

# Lint
cargo clippy --workspace --all-targets -- -D warnings

# Format
cargo fmt --all
```

## Minimum Supported Rust Version

The MSRV is **1.85.0** (Rust edition 2024).

## License

Licensed under the [MIT License](LICENSE).
