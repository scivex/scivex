# Changelog — pyscivex

## v0.1.0 (2026-03-28)

Initial release of pyscivex — the complete data science toolkit powered by Rust.

### Highlights

- **16 submodules** covering tensors, DataFrames, statistics, ML, neural networks, optimization, visualization, signal processing, image processing, NLP, graphs, symbolic math, reinforcement learning, and GPU acceleration
- **120+ Python classes** including Tensor, DataFrame, 14 distributions, 30+ ML models, Sequential neural networks, Graph/DiGraph, and more
- **270+ functions** spanning statistics, metrics, signal processing, image filters, NLP utilities, and symbolic math
- **Zero external Python dependencies** — pure Rust underneath
- **GPU acceleration** via wgpu (Metal on macOS, Vulkan on Linux/Windows)
- **Type stubs** (`.pyi`) for full IDE autocomplete and mypy support
- **Multi-platform wheels** for Linux (x86_64, aarch64), macOS (x86_64, arm64), and Windows (x64)

### Submodules

| Module | Description |
|--------|-------------|
| `sv.Tensor` | N-d tensors, broadcasting, element-wise math, linear algebra |
| `sv.DataFrame` | Column-oriented DataFrames, GroupBy, joins, pivots, rolling windows, string/datetime ops |
| `sv.stats` | Distributions, hypothesis tests, regression, time series, ARIMA, Prophet, Bayesian MCMC, survival analysis |
| `sv.ml` | Trees, forests, SVM, KNN, NB, clustering, PCA, t-SNE, pipelines, feature selection, explainability, online learning |
| `sv.nn` | Autograd, Linear/Conv/LSTM/GRU/Attention layers, optimizers, schedulers, ONNX import, SafeTensors, GNN |
| `sv.optim` | Root finding, minimization, ODE solvers, curve fitting, LP, sparse solvers |
| `sv.viz` | Line, scatter, bar, histogram, heatmap, box, violin, grammar-of-graphics chart, animation, SVG output |
| `sv.signal` | FFT, STFT, wavelets, filters, MFCC, beat tracking, WAV I/O |
| `sv.image` | Filters, transforms, features, contours, Hough, augmentation |
| `sv.nlp` | Tokenizers, TF-IDF, Word2Vec, sentiment, POS tagging, NER, LDA |
| `sv.graph` | Graph/DiGraph, BFS/DFS, Dijkstra, PageRank, max flow, MST |
| `sv.sym` | Symbolic expressions, differentiation, integration, solving |
| `sv.rl` | CartPole/MountainCar/GridWorld, DQN, PPO, A2C, SAC, TD3, HER |
| `sv.gpu` | GPU tensors, matmul, activations, Adam/SGD optimizers |
| `sv.linalg` | LU, QR, SVD, Eigendecomposition, Cholesky |
| `sv.fft` | 1D/2D FFT, real FFT, inverse FFT |
| `sv.io` | CSV, JSON, Parquet, Excel, Arrow, SQL read/write |
