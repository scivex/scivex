//! # Scivex
//!
//! A comprehensive Rust library replacing the Python data science ecosystem.
//!
//! One `use scivex::prelude::*;` gives you tensors, dataframes, statistics,
//! machine learning, neural networks, visualization, and more — all
//! implemented from scratch in pure Rust.
//!
//! ## Quick Start
//!
//! ```toml
//! [dependencies]
//! scivex = { version = "0.1", features = ["full"] }
//! ```
//!
//! ```rust,ignore
//! use scivex::prelude::*;
//!
//! let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
//! println!("{:?}", t.shape()); // [2, 2]
//! ```
//!
//! ## Modules
//!
//! Each module corresponds to a sub-crate in the Scivex ecosystem:
//!
//! | Module | Crate | Description |
//! |--------|-------|-------------|
//! | `core` | [`scivex-core`](https://docs.rs/scivex-core) | Tensors, linear algebra, FFT, SIMD, math primitives |
//! | `frame` | [`scivex-frame`](https://docs.rs/scivex-frame) | DataFrames, Series, GroupBy, joins, reshaping |
//! | `stats` | [`scivex-stats`](https://docs.rs/scivex-stats) | Distributions, hypothesis tests, regression, Bayesian |
//! | `io` | [`scivex-io`](https://docs.rs/scivex-io) | CSV, JSON, Parquet, Arrow IPC, HDF5, Excel, SQLite |
//! | `optim` | [`scivex-optim`](https://docs.rs/scivex-optim) | Optimization, root finding, integration, ODE solvers |
//! | `viz` | [`scivex-viz`](https://docs.rs/scivex-viz) | Plotting, charts, SVG/PNG rendering |
//! | `ml` | [`scivex-ml`](https://docs.rs/scivex-ml) | Trees, ensembles, SVM, clustering, pipelines, AutoML |
//! | `nn` | [`scivex-nn`](https://docs.rs/scivex-nn) | Neural networks, autograd, layers, optimizers |
//! | `image` | [`scivex-image`](https://docs.rs/scivex-image) | Image loading, transforms, filters, morphology |
//! | `signal` | [`scivex-signal`](https://docs.rs/scivex-signal) | Signal processing, FFT, wavelets, audio |
//! | `graph` | [`scivex-graph`](https://docs.rs/scivex-graph) | Graph data structures, algorithms, network analysis |
//! | `nlp` | [`scivex-nlp`](https://docs.rs/scivex-nlp) | Tokenization, embeddings, text processing |
//! | `sym` | [`scivex-sym`](https://docs.rs/scivex-sym) | Symbolic math, CAS, expression simplification |
//! | `gpu` | [`scivex-gpu`](https://docs.rs/scivex-gpu) | GPU-accelerated tensor ops via wgpu |
//! | `rl` | [`scivex-rl`](https://docs.rs/scivex-rl) | Reinforcement learning: DQN, PPO, A2C |
//!
//! ## Feature Flags
//!
//! | Feature | Enables |
//! |---------|---------|
//! | `core` *(default)* | Tensors, linear algebra, FFT, math primitives |
//! | `frame`            | DataFrames, Series, GroupBy, joins             |
//! | `stats`            | Distributions, hypothesis tests, regression    |
//! | `io`               | CSV and JSON reading/writing                   |
//! | `optim`            | Optimization, root finding, integration        |
//! | `viz`              | Visualization, plotting, chart rendering        |
//! | `ml`               | Classical ML: trees, ensembles, clustering      |
//! | `nn`               | Neural networks, autograd, layers, optimizers   |
//! | `image`            | Image loading, transforms, filters              |
//! | `signal`           | Signal processing, FFT, wavelets, audio         |
//! | `graph`            | Graph data structures, algorithms, network      |
//! | `nlp`              | Tokenization, embeddings, text processing       |
//! | `sym`              | Symbolic math, CAS, expression simplification   |
//! | `gpu`              | GPU-accelerated tensor ops via wgpu              |
//! | `rl`               | Reinforcement learning: DQN, PPO, A2C           |
//! | `full`             | All of the above (except `gpu`)                 |
//! | `simd`             | SIMD-accelerated kernels in core                |
//! | `parallel`         | Rayon-based parallelism                         |
//! | `serde-support`    | Serde Serialize/Deserialize for all types       |

/// Tensors, linear algebra, FFT, math primitives, and numeric traits.
///
/// See [`scivex-core` on docs.rs](https://docs.rs/scivex-core) for full API.
pub use scivex_core as core;

/// DataFrames, Series, GroupBy, joins, and tabular data manipulation.
///
/// See [`scivex-frame` on docs.rs](https://docs.rs/scivex-frame) for full API.
#[cfg(feature = "frame")]
pub use scivex_frame as frame;

/// Statistical distributions, hypothesis tests, regression, and Bayesian methods.
///
/// See [`scivex-stats` on docs.rs](https://docs.rs/scivex-stats) for full API.
#[cfg(feature = "stats")]
pub use scivex_stats as stats;

/// File I/O: CSV, JSON, Parquet, Arrow IPC, HDF5, Excel, SQLite.
///
/// See [`scivex-io` on docs.rs](https://docs.rs/scivex-io) for full API.
#[cfg(feature = "io")]
pub use scivex_io as io;

/// Optimization, root finding, numerical integration, and ODE solvers.
///
/// See [`scivex-optim` on docs.rs](https://docs.rs/scivex-optim) for full API.
#[cfg(feature = "optim")]
pub use scivex_optim as optim;

/// Visualization, plotting, and chart rendering (SVG/PNG).
///
/// See [`scivex-viz` on docs.rs](https://docs.rs/scivex-viz) for full API.
#[cfg(feature = "viz")]
pub use scivex_viz as viz;

/// Classical machine learning: trees, ensembles, SVM, clustering, pipelines, AutoML.
///
/// See [`scivex-ml` on docs.rs](https://docs.rs/scivex-ml) for full API.
#[cfg(feature = "ml")]
pub use scivex_ml as ml;

/// Neural networks, automatic differentiation, layers, and optimizers.
///
/// See [`scivex-nn` on docs.rs](https://docs.rs/scivex-nn) for full API.
#[cfg(feature = "nn")]
pub use scivex_nn as nn;

/// Image loading, transforms, filters, and morphological operations.
///
/// See [`scivex-image` on docs.rs](https://docs.rs/scivex-image) for full API.
#[cfg(feature = "image")]
pub use scivex_image as image;

/// Signal processing, FFT, wavelets, and audio analysis.
///
/// See [`scivex-signal` on docs.rs](https://docs.rs/scivex-signal) for full API.
#[cfg(feature = "signal")]
pub use scivex_signal as signal;

/// Graph data structures, algorithms, and network analysis.
///
/// See [`scivex-graph` on docs.rs](https://docs.rs/scivex-graph) for full API.
#[cfg(feature = "graph")]
pub use scivex_graph as graph;

/// Natural language processing: tokenization, embeddings, text processing.
///
/// See [`scivex-nlp` on docs.rs](https://docs.rs/scivex-nlp) for full API.
#[cfg(feature = "nlp")]
pub use scivex_nlp as nlp;

/// Symbolic math, computer algebra system, and expression simplification.
///
/// See [`scivex-sym` on docs.rs](https://docs.rs/scivex-sym) for full API.
#[cfg(feature = "sym")]
pub use scivex_sym as sym;

/// GPU-accelerated tensor operations via wgpu compute shaders.
///
/// See [`scivex-gpu` on docs.rs](https://docs.rs/scivex-gpu) for full API.
#[cfg(feature = "gpu")]
pub use scivex_gpu as gpu;

/// Reinforcement learning: environments, DQN, PPO, A2C.
///
/// See [`scivex-rl` on docs.rs](https://docs.rs/scivex-rl) for full API.
#[cfg(feature = "rl")]
pub use scivex_rl as rl;

/// Glob-import convenience: `use scivex::prelude::*;`
///
/// Note: When multiple features are enabled, some names may conflict
/// (e.g. `Result`, `Kernel`, `mean`). In such cases, use qualified paths
/// like `scivex::stats::prelude::mean` or `scivex::nn::prelude::mean`.
#[allow(ambiguous_glob_reexports)]
pub mod prelude {
    pub use scivex_core::prelude::*;

    #[cfg(feature = "frame")]
    pub use scivex_frame::prelude::*;

    #[cfg(feature = "stats")]
    pub use scivex_stats::prelude::*;

    #[cfg(feature = "io")]
    pub use scivex_io::prelude::*;

    #[cfg(feature = "optim")]
    pub use scivex_optim::prelude::*;

    #[cfg(feature = "viz")]
    pub use scivex_viz::prelude::*;

    #[cfg(feature = "ml")]
    pub use scivex_ml::prelude::*;

    #[cfg(feature = "nn")]
    pub use scivex_nn::prelude::*;

    #[cfg(feature = "image")]
    pub use scivex_image::prelude::*;

    #[cfg(feature = "signal")]
    pub use scivex_signal::prelude::*;

    #[cfg(feature = "graph")]
    pub use scivex_graph::prelude::*;

    #[cfg(feature = "nlp")]
    pub use scivex_nlp::prelude::*;

    #[cfg(feature = "sym")]
    pub use scivex_sym::prelude::*;

    #[cfg(feature = "rl")]
    pub use scivex_rl::prelude::*;
}

// Re-export the Tensor type at the top level for ergonomics.
pub use scivex_core::Tensor;

#[cfg(feature = "frame")]
pub use scivex_frame::DataFrame;
