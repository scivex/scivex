//! Symbolic math bindings for JavaScript.

use scivex_sym::{Expr, constant, cos, diff, exp, expand, ln, simplify, sin, var};
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

/// A symbolic mathematical expression.
#[wasm_bindgen]
pub struct WasmExpr {
    inner: Expr,
}

#[wasm_bindgen]
impl WasmExpr {
    /// Create a constant expression.
    #[wasm_bindgen(js_name = "constant")]
    pub fn new_constant(val: f64) -> WasmExpr {
        WasmExpr {
            inner: constant(val),
        }
    }

    /// Create a variable expression.
    #[wasm_bindgen(js_name = "variable")]
    pub fn new_var(name: &str) -> WasmExpr {
        WasmExpr { inner: var(name) }
    }

    /// Add two expressions: self + other.
    pub fn add(&self, other: &WasmExpr) -> WasmExpr {
        WasmExpr {
            inner: self.inner.clone() + other.inner.clone(),
        }
    }

    /// Multiply two expressions: self * other.
    pub fn mul(&self, other: &WasmExpr) -> WasmExpr {
        WasmExpr {
            inner: self.inner.clone() * other.inner.clone(),
        }
    }

    /// Subtract: self - other.
    pub fn sub(&self, other: &WasmExpr) -> WasmExpr {
        WasmExpr {
            inner: self.inner.clone() - other.inner.clone(),
        }
    }

    /// Divide: self / other.
    pub fn div(&self, other: &WasmExpr) -> WasmExpr {
        WasmExpr {
            inner: self.inner.clone() / other.inner.clone(),
        }
    }

    /// Raise to a power: self ^ exponent.
    pub fn pow(&self, exponent: f64) -> WasmExpr {
        WasmExpr {
            inner: Expr::Pow(
                Box::new(self.inner.clone()),
                Box::new(Expr::Const(exponent)),
            ),
        }
    }

    /// Apply sin.
    #[wasm_bindgen(js_name = "sin")]
    pub fn apply_sin(&self) -> WasmExpr {
        WasmExpr {
            inner: sin(self.inner.clone()),
        }
    }

    /// Apply cos.
    #[wasm_bindgen(js_name = "cos")]
    pub fn apply_cos(&self) -> WasmExpr {
        WasmExpr {
            inner: cos(self.inner.clone()),
        }
    }

    /// Apply exp.
    #[wasm_bindgen(js_name = "exp")]
    pub fn apply_exp(&self) -> WasmExpr {
        WasmExpr {
            inner: exp(self.inner.clone()),
        }
    }

    /// Apply ln.
    #[wasm_bindgen(js_name = "ln")]
    pub fn apply_ln(&self) -> WasmExpr {
        WasmExpr {
            inner: ln(self.inner.clone()),
        }
    }

    /// Differentiate with respect to a variable.
    pub fn diff(&self, var_name: &str) -> WasmExpr {
        WasmExpr {
            inner: diff::diff(&self.inner, var_name),
        }
    }

    /// Simplify the expression.
    pub fn simplify(&self) -> WasmExpr {
        WasmExpr {
            inner: simplify::simplify(&self.inner),
        }
    }

    /// Expand (distribute multiplication over addition).
    pub fn expand(&self) -> WasmExpr {
        WasmExpr {
            inner: expand(&self.inner),
        }
    }

    /// Evaluate the expression with variable values.
    /// Pass variable names and values as parallel arrays.
    #[allow(clippy::needless_pass_by_value)]
    pub fn eval(&self, var_names: Vec<String>, var_values: &[f64]) -> Result<f64, JsError> {
        let mut vars = HashMap::new();
        for (name, &val) in var_names.iter().zip(var_values.iter()) {
            vars.insert(name.clone(), val);
        }
        self.inner
            .eval(&vars)
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// String representation.
    #[wasm_bindgen(js_name = "toString")]
    pub fn to_string_js(&self) -> String {
        format!("{}", self.inner)
    }
}
