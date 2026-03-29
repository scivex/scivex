# pyscivex

**The complete data science toolkit — powered by Rust.**

One `pip install pyscivex` replaces numpy, pandas, scipy, scikit-learn, matplotlib, and more — with Rust performance and **zero external Python dependencies**.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

---

## Quick Start

```python
import pyscivex as sv

# Tensors (numpy replacement)
a = sv.Tensor([[1.0, 2.0], [3.0, 4.0]])
b = sv.Tensor.ones([2, 2])
c = a @ b
print(c.shape())  # [2, 2]

# DataFrames (pandas replacement)
df = sv.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
print(df.shape())  # (3, 2)

# Statistics (scipy.stats replacement)
data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
print(sv.mean(data))      # 5.0
print(sv.std_dev(data))   # ~2.0

# Machine Learning (scikit-learn replacement)
model = sv.ml.LinearRegression()
model.fit(sv.Tensor([[1.0], [2.0], [3.0]]), sv.Tensor([2.0, 4.0, 6.0]))
pred = model.predict(sv.Tensor([[4.0]]))

# Visualization (matplotlib replacement)
fig = sv.Figure()
fig.line_plot([1, 2, 3, 4], [1, 4, 9, 16], label="y=x^2")
fig.title("My Plot")
fig.save_svg("plot.svg")

# GPU Acceleration
dev = sv.gpu.Device()
gt = sv.gpu.GpuTensor(dev, [1.0, 2.0, 3.0, 4.0], [2, 2])
result = sv.gpu.matmul(gt, gt)
print(result.to_list())
```

---

## What's Included

| Submodule | Replaces | Highlights |
|-----------|----------|------------|
| **Core** (`sv.Tensor`) | numpy | N-d tensors, broadcasting, element-wise math, linear algebra |
| **`sv.DataFrame`** | pandas | Column-oriented DataFrames, GroupBy, joins, pivots, SQL |
| **`sv.stats`** | scipy.stats | 14 distributions, hypothesis tests, regression, effect sizes |
| **`sv.ml`** | scikit-learn | Trees, forests, SVM, KNN, NB, clustering, PCA, t-SNE, pipelines |
| **`sv.nn`** | PyTorch | Autograd, Linear/Conv/LSTM/GRU/Attention layers, optimizers |
| **`sv.optim`** | scipy.optimize | Root finding, minimization, ODE solvers, curve fitting, LP |
| **`sv.viz`** | matplotlib | Line, scatter, bar, histogram, heatmap, box, violin, SVG output |
| **`sv.signal`** | scipy.signal + librosa | FFT, STFT, wavelets, filters, MFCC, beat tracking, WAV I/O |
| **`sv.image`** | OpenCV + Pillow | Filters, transforms, features, contours, Hough, augmentation |
| **`sv.nlp`** | NLTK + spaCy + gensim | Tokenizers, TF-IDF, Word2Vec, sentiment, POS tagging, NER, LDA |
| **`sv.graph`** | NetworkX | Graph/DiGraph, BFS/DFS, Dijkstra, PageRank, max flow, MST |
| **`sv.sym`** | SymPy | Symbolic expressions, differentiation, integration, solving |
| **`sv.rl`** | Stable-Baselines3 | CartPole/MountainCar/GridWorld, DQN, PPO, A2C, SAC, TD3 |
| **`sv.gpu`** | CuPy | GPU tensors via wgpu/Metal/Vulkan, matmul, activations |
| **`sv.linalg`** | numpy.linalg | LU, QR, SVD, Eigendecomposition, Cholesky |
| **`sv.fft`** | numpy.fft | 1D/2D FFT, real FFT, inverse FFT |
| **`sv.io`** | pandas I/O | CSV, JSON read/write |

---

## Key Features

- **Zero dependencies** — no numpy, no pandas, no scipy. Pure Rust underneath.
- **697 tests** — comprehensive test coverage across all modules.
- **Full type stubs** — `.pyi` stubs for IDE autocomplete and mypy support.
- **GPU acceleration** — wgpu backend (Metal on macOS, Vulkan on Linux/Windows).
- **107 Python classes** — from Tensor to DQN to Word2Vec.
- **207 functions** — statistics, ML metrics, signal processing, image filters, and more.
- **16 submodules** — one import covers the entire data science stack.

---

## Installation

```bash
pip install pyscivex
```

Requires Python 3.9+. No compiler needed — prebuilt wheels available for:
- macOS (arm64, x86_64)
- Linux (x86_64, aarch64)
- Windows (x64)

---

## Examples

### Neural Network Training

```python
import pyscivex as sv

# Build model
model = sv.nn.Sequential([
    sv.nn.Linear(784, 128, seed=42),
    sv.nn.ReLU(),
    sv.nn.Linear(128, 10, seed=43),
])

# Training loop
optimizer = sv.nn.Adam(model.parameters(), lr=0.001)
for epoch in range(10):
    x = sv.nn.tensor([[1.0] * 784], requires_grad=True)
    y = sv.nn.tensor([[1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    pred = model.forward(x)
    loss = sv.nn.mse_loss(pred, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### Statistical Analysis

```python
import pyscivex as sv

group_a = [23.0, 25.0, 28.0, 30.0, 32.0]
group_b = [18.0, 20.0, 22.0, 24.0, 26.0]

result = sv.stats.ttest_ind(group_a, group_b)
print(f"t={result['statistic']:.3f}, p={result['p_value']:.4f}")

effect = sv.stats.cohens_d(group_a, group_b)
print(f"Cohen's d = {effect:.3f}")
```

### Reinforcement Learning

```python
import pyscivex as sv

agent = sv.rl.DQN(4, 2, learning_rate=0.001, seed=42)
result = agent.train_cartpole(episodes=100)
print(f"Mean reward (last 10): {result['episode_rewards'][-10:]}")
```

### Graph Analysis

```python
import pyscivex as sv

g = sv.graph.Graph.from_edges([(0, 1, 1.0), (1, 2, 2.0), (0, 2, 4.0)])
dists, _ = g.dijkstra(0)
print(f"Shortest distances from 0: {dists}")

mst, cost = g.kruskal()
print(f"MST cost: {cost}")
```

### Symbolic Math

```python
import pyscivex as sv

x = sv.sym.var("x")
expr = x ** sv.sym.constant(2.0) + sv.sym.constant(3.0) * x
deriv = sv.sym.sym_diff(expr, "x")
print(deriv.eval({"x": 2.0}))  # 7.0 (2x + 3 at x=2)
```

### GPU Computing

```python
import pyscivex as sv

dev = sv.gpu.Device()
print(dev.info())  # {'name': 'Apple M...', 'backend': 'Metal', ...}

a = sv.gpu.GpuTensor(dev, [1.0, 2.0, 3.0, 4.0], [2, 2])
b = sv.gpu.GpuTensor(dev, [5.0, 6.0, 7.0, 8.0], [2, 2])
c = sv.gpu.matmul(a, b)
print(c.to_list())  # [19.0, 22.0, 43.0, 50.0]
```

---

## License

MIT
