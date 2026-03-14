//! DataFrame bindings for JavaScript.

use scivex_frame::{DataFrame, Series};
use wasm_bindgen::prelude::*;

/// A tabular data structure with named columns.
#[wasm_bindgen]
pub struct WasmDataFrame {
    inner: DataFrame,
}

#[wasm_bindgen]
impl WasmDataFrame {
    /// Create a DataFrame using the builder pattern.
    /// Call `addColumnF64` to add columns, then use `build` to finalize.
    #[allow(clippy::new_ret_no_self)]
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmDataFrameBuilder {
        WasmDataFrameBuilder {
            builder: DataFrame::builder(),
        }
    }

    /// Get the number of rows.
    #[wasm_bindgen(js_name = "nRows")]
    pub fn n_rows(&self) -> usize {
        self.inner.nrows()
    }

    /// Get the number of columns.
    #[wasm_bindgen(js_name = "nCols")]
    pub fn n_cols(&self) -> usize {
        self.inner.ncols()
    }

    /// Get column names.
    #[wasm_bindgen(js_name = "columnNames")]
    pub fn column_names(&self) -> Vec<String> {
        self.inner
            .column_names()
            .iter()
            .map(std::string::ToString::to_string)
            .collect()
    }

    /// Get a Float64 column as an array.
    #[wasm_bindgen(js_name = "getColumnF64")]
    pub fn get_column_f64(&self, name: &str) -> Result<Vec<f64>, JsError> {
        let col = self
            .inner
            .column(name)
            .map_err(|e| JsError::new(&e.to_string()))?;
        let series = col
            .as_any()
            .downcast_ref::<Series<f64>>()
            .ok_or_else(|| JsError::new("column is not Float64"))?;
        Ok(series.as_slice().to_vec())
    }

    /// Filter rows by a boolean mask (pass as Uint8Array: 0=exclude, 1=include).
    pub fn filter(&self, mask: &[u8]) -> Result<WasmDataFrame, JsError> {
        let bool_mask: Vec<bool> = mask.iter().map(|&v| v != 0).collect();
        let result = self
            .inner
            .filter(&bool_mask)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(WasmDataFrame { inner: result })
    }

    /// Select specific columns by name.
    #[allow(clippy::needless_pass_by_value)]
    pub fn select(&self, names: Vec<String>) -> Result<WasmDataFrame, JsError> {
        let name_refs: Vec<&str> = names.iter().map(String::as_str).collect();
        let result = self
            .inner
            .select(&name_refs)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(WasmDataFrame { inner: result })
    }

    /// Get first n rows.
    pub fn head(&self, n: usize) -> WasmDataFrame {
        WasmDataFrame {
            inner: self.inner.head(n),
        }
    }

    /// Get last n rows.
    pub fn tail(&self, n: usize) -> WasmDataFrame {
        WasmDataFrame {
            inner: self.inner.tail(n),
        }
    }

    /// Shape as `[n_rows, n_cols]`.
    pub fn shape(&self) -> Vec<usize> {
        vec![self.inner.nrows(), self.inner.ncols()]
    }

    /// String representation.
    #[wasm_bindgen(js_name = "toString")]
    pub fn to_string_js(&self) -> String {
        format!("{}", self.inner)
    }
}

/// Builder for creating a `WasmDataFrame`.
#[wasm_bindgen]
pub struct WasmDataFrameBuilder {
    builder: scivex_frame::DataFrameBuilder,
}

#[wasm_bindgen]
impl WasmDataFrameBuilder {
    /// Add a Float64 column.
    #[wasm_bindgen(js_name = "addColumnF64")]
    pub fn add_column_f64(self, name: &str, data: &[f64]) -> WasmDataFrameBuilder {
        WasmDataFrameBuilder {
            builder: self.builder.add_column(name, data.to_vec()),
        }
    }

    /// Add an Int32 column.
    #[wasm_bindgen(js_name = "addColumnI32")]
    pub fn add_column_i32(self, name: &str, data: &[i32]) -> WasmDataFrameBuilder {
        WasmDataFrameBuilder {
            builder: self.builder.add_column(name, data.to_vec()),
        }
    }

    /// Build the DataFrame.
    pub fn build(self) -> Result<WasmDataFrame, JsError> {
        let df = self
            .builder
            .build()
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(WasmDataFrame { inner: df })
    }
}
