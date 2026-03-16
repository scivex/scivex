//! `pyscivex` — Python bindings for the Scivex ecosystem via PyO3.
//!
//! Exposes tensors, data frames, statistics, ML models, and visualization
//! to Python through a native extension module.
//!
//! ```python
//! import pyscivex
//! t = pyscivex.Tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
//! print(t.shape())   # [2, 2]
//! ```

use pyo3::prelude::*;

mod frame;
mod ml;
mod stats;
mod tensor;
mod viz;

/// Python module: `import pyscivex`
#[pymodule]
fn pyscivex(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Tensor
    m.add_class::<tensor::PyTensor>()?;
    // DataFrame
    m.add_class::<frame::PyDataFrame>()?;
    // Stats
    m.add_function(wrap_pyfunction!(stats::mean, m)?)?;
    m.add_function(wrap_pyfunction!(stats::variance, m)?)?;
    m.add_function(wrap_pyfunction!(stats::std_dev, m)?)?;
    m.add_function(wrap_pyfunction!(stats::median, m)?)?;
    m.add_function(wrap_pyfunction!(stats::pearson, m)?)?;
    // ML
    m.add_class::<ml::PyLinearRegression>()?;
    m.add_class::<ml::PyKMeans>()?;
    // Viz
    m.add_class::<viz::PyFigure>()?;
    Ok(())
}
