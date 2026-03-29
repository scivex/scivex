#![allow(
    clippy::unnecessary_wraps,
    clippy::needless_pass_by_value,
    clippy::unused_self,
    clippy::redundant_closure_for_method_calls,
    clippy::uninlined_format_args,
    clippy::format_push_string,
    clippy::elidable_lifetime_names,
    clippy::unreadable_literal,
    clippy::too_many_lines,
    clippy::cast_lossless,
    clippy::match_same_arms,
    clippy::map_unwrap_or,
    clippy::doc_link_with_quotes,
    clippy::bool_to_int_with_if
)]
//! `pyscivex` — Python bindings for the Scivex ecosystem via PyO3.
//!
//! Exposes tensors, data frames, statistics, ML models, visualization,
//! linear algebra, FFT, and random number generation to Python through
//! a native extension module.
//!
//! ```python
//! import pyscivex as sv
//! t = sv.Tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
//! print(t.shape())   # [2, 2]
//! ```

use pyo3::prelude::*;

mod fft;
mod frame;
mod gpu;
mod graph;
mod image;
mod io;
mod linalg;
mod ml;
mod nlp;
mod nn;
mod optim;
mod random;
mod rl;
mod signal;
mod stats;
mod sym;
pub mod tensor;
mod viz;

/// Python module: `import pyscivex._native`
///
/// The public API is re-exported through `python/pyscivex/__init__.py`
/// so users write `import pyscivex as sv`.
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Tensor
    m.add_class::<tensor::PyTensor>()?;
    // DataFrame
    m.add_class::<frame::PyDataFrame>()?;
    // LazyFrame
    m.add_class::<frame::PyLazyFrame>()?;
    // Series
    m.add_class::<frame::PySeries>()?;
    // Stats
    m.add_function(wrap_pyfunction!(stats::mean, m)?)?;
    m.add_function(wrap_pyfunction!(stats::variance, m)?)?;
    m.add_function(wrap_pyfunction!(stats::std_dev, m)?)?;
    m.add_function(wrap_pyfunction!(stats::median, m)?)?;
    m.add_function(wrap_pyfunction!(stats::pearson, m)?)?;
    // ML (top-level convenience — full API in ml submodule)
    m.add_class::<ml::PyLinearRegression>()?;
    m.add_class::<ml::PyKMeans>()?;
    // Viz
    m.add_class::<viz::PyFigure>()?;
    // Submodules
    linalg::register(m)?;
    fft::register(m)?;
    random::register(m)?;
    io::register(m)?;
    stats::register(m)?;
    ml::register(m)?;
    optim::register(m)?;
    nn::register(m)?;
    signal::register(m)?;
    image::register(m)?;
    nlp::register(m)?;
    graph::register(m)?;
    sym::register(m)?;
    rl::register(m)?;
    gpu::register(m)?;
    viz::register(m)?;
    Ok(())
}
