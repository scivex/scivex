# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **scivex-core**: Tensor type with dynamic shapes, SIMD-accelerated BLAS (NEON/AVX), blocked GEMM with 4x4 micro-kernel
- **scivex-core**: Mixed-radix FFT (radix-2/3/5/7) with precomputed twiddle factors, Bluestein for arbitrary lengths
- **scivex-core**: LU, QR, SVD, Cholesky, Eigenvalue decompositions
- **scivex-core**: Einsum, named tensors, sparse tensors, tensor decompositions (CP, Tucker, NMF)
- **scivex-core**: KD-tree spatial indexing, NumExpr-style JIT expression evaluator
- **scivex-frame**: DataFrame with typed columns, GroupBy, Sort, Filter, Join, Pivot, Melt, Lazy evaluation
- **scivex-stats**: 15+ probability distributions, hypothesis tests (t-test, chi-squared, ANOVA, Mann-Whitney), correlation, OLS/WLS regression
- **scivex-stats**: Bayesian inference with NUTS sampler, linear mixed models
- **scivex-optim**: BFGS, Nelder-Mead, L-BFGS-B optimizers; Newton-Cotes and Gauss-Legendre integration; RK45 ODE solver
- **scivex-optim**: Simplex linear programming, Levenberg-Marquardt curve fitting
- **scivex-io**: CSV, JSON, Parquet, Arrow IPC, HDF5, SQL readers/writers
- **scivex-viz**: Scatter, line, bar, histogram, heatmap, box, violin plots with SVG/PNG backends
- **scivex-ml**: Decision trees, Random Forest, Gradient Boosting (Hist-GBM), CatBoost, XGBoost-style ensembles
- **scivex-ml**: KMeans, DBSCAN, Spectral clustering; SVM (linear, RBF, polynomial); KNN
- **scivex-ml**: Stacking ensemble, feature selection (RFE, SelectKBest), target/ordinal encoding
- **scivex-ml**: Pipeline API with fit/predict/transform, cross-validation, metrics
- **scivex-nn**: Autograd engine with reverse-mode differentiation
- **scivex-nn**: Linear, Conv1d/2d/3d, BatchNorm, LayerNorm, Dropout, RNN/LSTM/GRU layers
- **scivex-nn**: SGD, Adam, AdamW, RMSProp, Adagrad optimizers; learning rate schedulers
- **scivex-nn**: ONNX, SafeTensors, GGUF model import/export
- **scivex-nn**: GNN layers (GCN, GAT, GraphSAGE) for graph neural networks
- **scivex-nn**: SHAP feature importance explanations
- **scivex-image**: Image loading (PNG, JPEG), resize, crop, rotate, convolution filters, HOG features
- **scivex-signal**: STFT, mel spectrogram, MFCC, FIR/IIR filters, wavelets (Haar, Daubechies), beat tracking
- **scivex-signal**: Prophet-style time series forecasting, feature extraction, anomaly detection
- **scivex-graph**: Adjacency list graph, BFS, DFS, Dijkstra, Bellman-Ford, A*, Floyd-Warshall, MST (Kruskal, Prim)
- **scivex-graph**: PageRank, connected components, topological sort, network flow
- **scivex-nlp**: Whitespace/regex/BPE/WordPiece tokenizers, TF-IDF vectorizer, word embeddings
- **scivex-sym**: Symbolic expressions, differentiation, simplification, polynomial operations
- **scivex-gpu**: wgpu-based GPU tensor operations, GPU neural network training
- **scivex-wasm**: WebAssembly bindings for browser and Node.js
- **scivex-ffi**: C FFI bindings for cross-language interop
- **scivex-rl**: DQN, PPO, A2C agents; CartPole, MountainCar, Pendulum environments
- **pyscivex**: Python bindings via PyO3
- Property-based tests (proptest) for core and frame crates
- Fuzz targets for tensor indexing, einsum parsing, CSV parsing
- Criterion benchmark suite across 8 crates
- CI/CD: formatting, clippy, tests (Linux/macOS/Windows), MSRV, coverage, security audit

## [0.1.0] — Unreleased

Initial release of the Scivex scientific computing library for Rust.

[Unreleased]: https://github.com/scivex/scivex/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/scivex/scivex/releases/tag/v0.1.0
