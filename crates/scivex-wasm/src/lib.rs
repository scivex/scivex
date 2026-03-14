//! # scivex-wasm
//!
//! WebAssembly bindings for the Scivex scientific computing library.
//!
//! Provides JavaScript/TypeScript-friendly wrappers around core Scivex types
//! and algorithms, enabling high-performance data science in the browser
//! and Node.js.
//!
//! ## Quick start (JavaScript)
//!
//! ```js
//! import init, { WasmTensor, WasmLinearRegression, stats_mean } from 'scivex-wasm';
//!
//! await init();
//!
//! // Tensor operations
//! const t = WasmTensor.from_array(new Float64Array([1, 2, 3, 4]), [2, 2]);
//! console.log(t.shape());    // [2, 2]
//! console.log(t.to_array()); // Float64Array([1, 2, 3, 4])
//!
//! // Statistics
//! const mean = stats_mean(new Float64Array([1, 2, 3, 4, 5]));
//!
//! // Linear regression
//! const lr = new WasmLinearRegression();
//! lr.fit(x_tensor, y_tensor);
//! const predictions = lr.predict(new_x);
//! ```

mod frame;
mod graph;
mod ml;
mod signal;
mod stats;
mod sym;
mod tensor;

pub use frame::*;
pub use graph::*;
pub use ml::*;
pub use signal::*;
pub use stats::*;
pub use sym::*;
pub use tensor::*;
