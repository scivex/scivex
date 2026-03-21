# Scivex

**The Rust data science library that replaces the entire Python ecosystem.**

Scivex is a comprehensive, from-scratch Cargo workspace providing tensors, dataframes, statistics, machine learning, neural networks, visualization, and more — all in pure Rust with zero external math dependencies.

One import gives you everything:

```rust
use scivex::prelude::*;
```

## Why Scivex?

| Pain point in Python | Scivex solution |
|----------------------|-----------------|
| NumPy, Pandas, sklearn, PyTorch, Matplotlib are separate packages with incompatible types | Single workspace, shared `Tensor<T>` and `DataFrame` types across all modules |
| GIL limits parallelism | Native threads + Rayon, SIMD (AVX2/NEON), GPU via wgpu |
| Runtime type errors | Compile-time generics: `Tensor<f32>`, `Tensor<f64>`, `Tensor<i32>` |
| Deployment requires Python runtime | Compiles to a single static binary, runs anywhere |
| C/Fortran dependency hell | Everything implemented from first principles in Rust |

## Workspace Crates

| Crate | Replaces | Highlights |
|-------|----------|------------|
| `scivex-core` | NumPy | N-d tensors, BLAS, LAPACK, FFT, broadcasting, SIMD |
| `scivex-frame` | Pandas | DataFrame, Series, GroupBy, joins, lazy evaluation |
| `scivex-stats` | statsmodels | Distributions, hypothesis tests, GLM, GARCH, Kalman, Bayesian |
| `scivex-optim` | SciPy.optimize | BFGS, L-BFGS-B, Nelder-Mead, simplex LP, Levenberg-Marquardt, ODE solvers |
| `scivex-io` | Pandas I/O | CSV, JSON, Parquet, HDF5, Arrow IPC, Excel, npy/npz |
| `scivex-viz` | Matplotlib/Seaborn | SVG/PNG/terminal/HTML backends, 3D surface, pair/joint plots, animation |
| `scivex-ml` | scikit-learn | Trees, SVM, KNN, clustering, ensembles, pipelines, SHAP |
| `scivex-nn` | PyTorch | Autograd, layers, CNN, RNN, Transformer, optimizers, ONNX |
| `scivex-image` | Pillow/OpenCV | Load/save images, convolution, morphology, transforms |
| `scivex-signal` | SciPy.signal | FFT, STFT, wavelets, filters, resampling, spectrograms |
| `scivex-graph` | NetworkX | Graph algorithms, shortest path, PageRank, community detection |
| `scivex-nlp` | NLTK/spaCy | Tokenization, TF-IDF, word embeddings, text classification |
| `scivex-sym` | SymPy | Symbolic algebra, differentiation, simplification |
| `scivex-gpu` | CuPy/CUDA | GPU tensors via wgpu (Vulkan/Metal/DX12), tiled matmul, on-device optimizers |

## Quick Example

```rust
use scivex::prelude::*;

// Create a tensor
let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
let c = a.matmul(&b).unwrap();
println!("{:?}", c.as_slice()); // [19.0, 22.0, 43.0, 50.0]
```

## License

MIT
