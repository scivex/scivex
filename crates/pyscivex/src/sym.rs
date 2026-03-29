//! Python bindings for scivex-sym — symbolic mathematics.

use pyo3::prelude::*;
use scivex_sym::prelude::{
    Expr, Polynomial, abs, constant as sym_const, cos, definite_integral, diff, diff_n, e, exp,
    expand, factor_out, integrate, ln, maclaurin, one, pi, simplify, sin, solve_linear,
    solve_quadratic, sqrt, tan, taylor, var as sym_var, zero,
};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn py_err(e: impl std::fmt::Display) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}

// ---------------------------------------------------------------------------
// PyExpr — symbolic expression
// ---------------------------------------------------------------------------

#[pyclass(name = "Expr")]
#[derive(Clone)]
pub struct PyExpr {
    inner: Expr,
}

impl PyExpr {
    fn wrap(expr: Expr) -> Self {
        Self { inner: expr }
    }
}

#[pymethods]
impl PyExpr {
    /// Evaluate the expression with given variable bindings.
    fn eval(&self, vars: HashMap<String, f64>) -> PyResult<f64> {
        self.inner.eval(&vars).map_err(py_err)
    }

    /// Substitute a variable with another expression.
    fn substitute(&self, var: &str, replacement: &PyExpr) -> Self {
        Self::wrap(self.inner.substitute(var, &replacement.inner))
    }

    /// Get free variables in this expression.
    fn free_variables(&self) -> Vec<String> {
        self.inner.free_variables().into_iter().collect()
    }

    fn is_zero(&self) -> bool {
        self.inner.is_zero()
    }

    fn is_one(&self) -> bool {
        self.inner.is_one()
    }

    fn is_const(&self) -> bool {
        self.inner.is_const()
    }

    fn as_const(&self) -> Option<f64> {
        self.inner.as_const()
    }

    // -- Operators --

    fn __add__(&self, other: &PyExpr) -> Self {
        Self::wrap(self.inner.clone() + other.inner.clone())
    }

    fn __radd__(&self, other: &PyExpr) -> Self {
        Self::wrap(other.inner.clone() + self.inner.clone())
    }

    fn __sub__(&self, other: &PyExpr) -> Self {
        Self::wrap(self.inner.clone() - other.inner.clone())
    }

    fn __rsub__(&self, other: &PyExpr) -> Self {
        Self::wrap(other.inner.clone() - self.inner.clone())
    }

    fn __mul__(&self, other: &PyExpr) -> Self {
        Self::wrap(self.inner.clone() * other.inner.clone())
    }

    fn __rmul__(&self, other: &PyExpr) -> Self {
        Self::wrap(other.inner.clone() * self.inner.clone())
    }

    fn __truediv__(&self, other: &PyExpr) -> Self {
        Self::wrap(self.inner.clone() / other.inner.clone())
    }

    fn __neg__(&self) -> Self {
        Self::wrap(-self.inner.clone())
    }

    fn __pow__(&self, exp: &PyExpr, _modulo: Option<&Bound<'_, PyAny>>) -> Self {
        Self::wrap(Expr::Pow(
            Box::new(self.inner.clone()),
            Box::new(exp.inner.clone()),
        ))
    }

    fn __repr__(&self) -> String {
        format!("{}", self.inner)
    }

    fn __str__(&self) -> String {
        format!("{}", self.inner)
    }
}

// ---------------------------------------------------------------------------
// Constructor functions
// ---------------------------------------------------------------------------

/// Create a symbolic variable.
#[pyfunction]
fn var(name: &str) -> PyExpr {
    PyExpr::wrap(sym_var(name))
}

/// Create a symbolic constant.
#[pyfunction]
fn constant(v: f64) -> PyExpr {
    PyExpr::wrap(sym_const(v))
}

/// Pi constant.
#[pyfunction]
fn sym_pi() -> PyExpr {
    PyExpr::wrap(pi())
}

/// Euler's number.
#[pyfunction]
fn sym_e() -> PyExpr {
    PyExpr::wrap(e())
}

/// Zero expression.
#[pyfunction]
fn sym_zero() -> PyExpr {
    PyExpr::wrap(zero())
}

/// One expression.
#[pyfunction]
fn sym_one() -> PyExpr {
    PyExpr::wrap(one())
}

// -- Math functions --

#[pyfunction]
fn sym_sin(expr: &PyExpr) -> PyExpr {
    PyExpr::wrap(sin(expr.inner.clone()))
}

#[pyfunction]
fn sym_cos(expr: &PyExpr) -> PyExpr {
    PyExpr::wrap(cos(expr.inner.clone()))
}

#[pyfunction]
fn sym_tan(expr: &PyExpr) -> PyExpr {
    PyExpr::wrap(tan(expr.inner.clone()))
}

#[pyfunction]
fn sym_exp(expr: &PyExpr) -> PyExpr {
    PyExpr::wrap(exp(expr.inner.clone()))
}

#[pyfunction]
fn sym_ln(expr: &PyExpr) -> PyExpr {
    PyExpr::wrap(ln(expr.inner.clone()))
}

#[pyfunction]
fn sym_sqrt(expr: &PyExpr) -> PyExpr {
    PyExpr::wrap(sqrt(expr.inner.clone()))
}

#[pyfunction]
fn sym_abs(expr: &PyExpr) -> PyExpr {
    PyExpr::wrap(abs(expr.inner.clone()))
}

// ---------------------------------------------------------------------------
// Calculus
// ---------------------------------------------------------------------------

/// Symbolic differentiation: d(expr)/d(var).
#[pyfunction]
#[pyo3(signature = (expr, wrt, n = 1))]
fn sym_diff(expr: &PyExpr, wrt: &str, n: usize) -> PyExpr {
    if n <= 1 {
        PyExpr::wrap(diff(&expr.inner, wrt))
    } else {
        PyExpr::wrap(diff_n(&expr.inner, wrt, n))
    }
}

/// Symbolic indefinite integration.
#[pyfunction]
fn sym_integrate(expr: &PyExpr, wrt: &str) -> PyResult<PyExpr> {
    let result = integrate(&expr.inner, wrt).map_err(py_err)?;
    Ok(PyExpr::wrap(result))
}

/// Definite integral from a to b.
#[pyfunction]
fn sym_definite_integral(expr: &PyExpr, wrt: &str, a: f64, b: f64) -> PyResult<f64> {
    definite_integral(&expr.inner, wrt, a, b).map_err(py_err)
}

// ---------------------------------------------------------------------------
// Algebra
// ---------------------------------------------------------------------------

/// Simplify expression.
#[pyfunction]
fn sym_simplify(expr: &PyExpr) -> PyExpr {
    PyExpr::wrap(simplify(&expr.inner))
}

/// Expand products over sums.
#[pyfunction]
fn sym_expand(expr: &PyExpr) -> PyExpr {
    PyExpr::wrap(expand(&expr.inner))
}

/// Factor out a common term.
#[pyfunction]
fn sym_factor(expr: &PyExpr, term: &PyExpr) -> PyExpr {
    PyExpr::wrap(factor_out(&expr.inner, &term.inner))
}

// ---------------------------------------------------------------------------
// Solving
// ---------------------------------------------------------------------------

/// Solve linear equation expr = 0 for var.
#[pyfunction]
fn sym_solve_linear(expr: &PyExpr, wrt: &str) -> PyResult<PyExpr> {
    let result = solve_linear(&expr.inner, wrt).map_err(py_err)?;
    Ok(PyExpr::wrap(result))
}

/// Solve quadratic equation expr = 0 for var.
#[pyfunction]
fn sym_solve_quadratic(expr: &PyExpr, wrt: &str) -> PyResult<Vec<PyExpr>> {
    let roots = solve_quadratic(&expr.inner, wrt).map_err(py_err)?;
    Ok(roots.into_iter().map(PyExpr::wrap).collect())
}

// ---------------------------------------------------------------------------
// Taylor series
// ---------------------------------------------------------------------------

/// Taylor series expansion around center to order n.
#[pyfunction]
fn sym_taylor(expr: &PyExpr, wrt: &str, center: f64, n: usize) -> PyResult<PyExpr> {
    let result = taylor(&expr.inner, wrt, center, n).map_err(py_err)?;
    Ok(PyExpr::wrap(result))
}

/// Maclaurin series (Taylor around 0).
#[pyfunction]
fn sym_maclaurin(expr: &PyExpr, wrt: &str, n: usize) -> PyResult<PyExpr> {
    let result = maclaurin(&expr.inner, wrt, n).map_err(py_err)?;
    Ok(PyExpr::wrap(result))
}

// ---------------------------------------------------------------------------
// Polynomial
// ---------------------------------------------------------------------------

#[pyclass(name = "Polynomial")]
#[derive(Clone)]
pub struct PyPolynomial {
    inner: Polynomial<f64>,
}

#[pymethods]
impl PyPolynomial {
    /// Create from coefficients in ascending power order: [a0, a1, a2, ...] = a0 + a1*x + a2*x^2 + ...
    #[new]
    fn new(coeffs: Vec<f64>) -> Self {
        Self {
            inner: Polynomial::new(coeffs),
        }
    }

    fn degree(&self) -> usize {
        self.inner.degree()
    }

    fn coeffs(&self) -> Vec<f64> {
        self.inner.coeffs().to_vec()
    }

    fn eval(&self, x: f64) -> f64 {
        self.inner.eval(x)
    }

    fn roots(&self) -> PyResult<Vec<f64>> {
        self.inner.roots().map_err(py_err)
    }

    fn add(&self, other: &PyPolynomial) -> Self {
        Self {
            inner: self.inner.add(&other.inner),
        }
    }

    fn mul(&self, other: &PyPolynomial) -> Self {
        Self {
            inner: self.inner.mul(&other.inner),
        }
    }

    /// Convert to symbolic expression.
    #[pyo3(signature = (var_name = "x"))]
    fn to_expr(&self, var_name: &str) -> PyExpr {
        PyExpr::wrap(self.inner.to_expr(var_name))
    }

    fn __repr__(&self) -> String {
        let coeffs = self.inner.coeffs();
        let mut parts = Vec::new();
        for (i, &c) in coeffs.iter().enumerate() {
            if c == 0.0 {
                continue;
            }
            if i == 0 {
                parts.push(format!("{c}"));
            } else if i == 1 {
                parts.push(format!("{c}*x"));
            } else {
                parts.push(format!("{c}*x^{i}"));
            }
        }
        if parts.is_empty() {
            "Polynomial(0)".to_string()
        } else {
            format!("Polynomial({})", parts.join(" + "))
        }
    }
}

// ---------------------------------------------------------------------------
// Register submodule
// ---------------------------------------------------------------------------

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "sym")?;

    // Core class
    m.add_class::<PyExpr>()?;
    m.add_class::<PyPolynomial>()?;

    // Constructors
    m.add_function(wrap_pyfunction!(var, &m)?)?;
    m.add_function(wrap_pyfunction!(constant, &m)?)?;
    m.add_function(wrap_pyfunction!(sym_pi, &m)?)?;
    m.add_function(wrap_pyfunction!(sym_e, &m)?)?;
    m.add_function(wrap_pyfunction!(sym_zero, &m)?)?;
    m.add_function(wrap_pyfunction!(sym_one, &m)?)?;

    // Math functions
    m.add_function(wrap_pyfunction!(sym_sin, &m)?)?;
    m.add_function(wrap_pyfunction!(sym_cos, &m)?)?;
    m.add_function(wrap_pyfunction!(sym_tan, &m)?)?;
    m.add_function(wrap_pyfunction!(sym_exp, &m)?)?;
    m.add_function(wrap_pyfunction!(sym_ln, &m)?)?;
    m.add_function(wrap_pyfunction!(sym_sqrt, &m)?)?;
    m.add_function(wrap_pyfunction!(sym_abs, &m)?)?;

    // Calculus
    m.add_function(wrap_pyfunction!(sym_diff, &m)?)?;
    m.add_function(wrap_pyfunction!(sym_integrate, &m)?)?;
    m.add_function(wrap_pyfunction!(sym_definite_integral, &m)?)?;

    // Algebra
    m.add_function(wrap_pyfunction!(sym_simplify, &m)?)?;
    m.add_function(wrap_pyfunction!(sym_expand, &m)?)?;
    m.add_function(wrap_pyfunction!(sym_factor, &m)?)?;

    // Solving
    m.add_function(wrap_pyfunction!(sym_solve_linear, &m)?)?;
    m.add_function(wrap_pyfunction!(sym_solve_quadratic, &m)?)?;

    // Taylor
    m.add_function(wrap_pyfunction!(sym_taylor, &m)?)?;
    m.add_function(wrap_pyfunction!(sym_maclaurin, &m)?)?;

    parent.add_submodule(&m)?;
    Ok(())
}
