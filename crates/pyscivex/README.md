# pyscivex

Python bindings for the Scivex data science library. One `pip install pyscivex`
replaces numpy, pandas, scipy, scikit-learn, matplotlib, and more — powered
by Rust for maximum performance.

## Highlights

- **Tensor** — N-dimensional arrays with broadcasting, slicing, and linear algebra
- **DataFrame & Series** — Tabular data with filtering, groupby, joins, and rolling windows
- **Statistics** — 15+ distributions, hypothesis tests, correlation, time series (ARIMA, GARCH)
- **Machine Learning** — Random forests, gradient boosting, SVM, k-NN, pipelines, cross-validation
- **Neural Networks** — Linear, Conv, RNN, Transformer layers with autograd and optimizers
- **Optimization** — BFGS, Nelder-Mead, linear programming, curve fitting, ODE solvers
- **Signal Processing** — FFT, FIR/IIR filters, STFT, wavelets, spectrograms
- **Image Processing** — Resize, rotate, blur, edge detection, morphology
- **NLP** — BPE/WordPiece/Unigram tokenizers, TF-IDF, embeddings, sentiment analysis
- **Graph Analysis** — Dijkstra, PageRank, MST, community detection, max flow
- **Symbolic Math** — Differentiation, simplification, equation solving
- **Visualization** — Line/scatter/bar/histogram charts, SVG rendering
- **I/O** — CSV, JSON, Parquet, Arrow, Excel, SQLite, NPY, HDF5, ORC, Avro

## Installation

```bash
pip install pyscivex
```

## Usage

```python
import pyscivex as sv

# Tensors
a = sv.Tensor([[1.0, 2.0], [3.0, 4.0]])
inv = sv.linalg.inv(a)

# DataFrames
df = sv.DataFrame()
df.add_column("x", [1.0, 2.0, 3.0])
df.add_column("y", [4.0, 5.0, 6.0])

# ML
model = sv.ml.RandomForestClassifier(n_trees=100, max_depth=5)

# Statistics
dist = sv.stats.Normal(0.0, 1.0)
p = dist.cdf(1.96)
```

## Building from Source

```bash
pip install maturin
cd crates/pyscivex
maturin develop --release
```

## License

MIT
