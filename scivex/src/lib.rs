//! # Scivex
//!
//! A comprehensive Rust library replacing the Python data science ecosystem.
//!
//! One `use scivex::prelude::*;` gives you tensors, dataframes, statistics,
//! machine learning, neural networks, visualization, and more — all
//! implemented from scratch in pure Rust.
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
//! | `full`             | All of the above (except `gpu`)                 |

pub use scivex_core as core;

#[cfg(feature = "frame")]
pub use scivex_frame as frame;

#[cfg(feature = "stats")]
pub use scivex_stats as stats;

#[cfg(feature = "io")]
pub use scivex_io as io;

#[cfg(feature = "optim")]
pub use scivex_optim as optim;

#[cfg(feature = "viz")]
pub use scivex_viz as viz;

#[cfg(feature = "ml")]
pub use scivex_ml as ml;

#[cfg(feature = "nn")]
pub use scivex_nn as nn;

#[cfg(feature = "image")]
pub use scivex_image as image;

#[cfg(feature = "signal")]
pub use scivex_signal as signal;

#[cfg(feature = "graph")]
pub use scivex_graph as graph;

#[cfg(feature = "nlp")]
pub use scivex_nlp as nlp;

#[cfg(feature = "sym")]
pub use scivex_sym as sym;

#[cfg(feature = "gpu")]
pub use scivex_gpu as gpu;

#[cfg(feature = "rl")]
pub use scivex_rl as rl;

/// Glob-import convenience: `use scivex::prelude::*;`
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
