//! Python bindings for `scivex-optim` — optimization, root finding,
//! integration, interpolation, ODE solvers, curve fitting, and linear programming.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use scivex_core::Tensor;

use crate::tensor::PyTensor;

// ---------------------------------------------------------------------------
// Error helper
// ---------------------------------------------------------------------------

#[allow(clippy::needless_pass_by_value)]
fn opt_err(e: scivex_optim::OptimError) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}

// ===========================================================================
// ROOT FINDING
// ===========================================================================

fn root_opts(xtol: f64, ftol: f64, max_iter: usize) -> scivex_optim::RootOptions<f64> {
    scivex_optim::RootOptions {
        xtol,
        ftol,
        max_iter,
    }
}

fn root_to_dict<'py>(
    py: Python<'py>,
    r: &scivex_optim::RootResult<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("root", r.root)?;
    d.set_item("f_root", r.f_root)?;
    d.set_item("iterations", r.iterations)?;
    d.set_item("converged", r.converged)?;
    Ok(d)
}

/// Find a root of `f` on the interval `[a, b]` using the bisection method.
///
/// Returns a dict with keys: `root`, `f_root`, `iterations`, `converged`.
#[pyfunction]
#[pyo3(signature = (f, a, b, xtol=1e-12, ftol=1e-12, max_iter=100))]
fn bisection<'py>(
    py: Python<'py>,
    f: PyObject,
    a: f64,
    b: f64,
    xtol: f64,
    ftol: f64,
    max_iter: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let opts = root_opts(xtol, ftol, max_iter);
    let r = scivex_optim::bisection(
        |x| {
            Python::with_gil(|py2| {
                f.call1(py2, (x,))
                    .and_then(|v| v.extract::<f64>(py2))
                    .unwrap_or(f64::NAN)
            })
        },
        a,
        b,
        &opts,
    )
    .map_err(opt_err)?;
    root_to_dict(py, &r)
}

/// Find a root of `f` on `[a, b]` using Brent's method.
///
/// Returns a dict with keys: `root`, `f_root`, `iterations`, `converged`.
#[pyfunction]
#[pyo3(signature = (f, a, b, xtol=1e-12, ftol=1e-12, max_iter=100))]
fn brentq<'py>(
    py: Python<'py>,
    f: PyObject,
    a: f64,
    b: f64,
    xtol: f64,
    ftol: f64,
    max_iter: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let opts = root_opts(xtol, ftol, max_iter);
    let r = scivex_optim::brent_root(
        |x| {
            Python::with_gil(|py2| {
                f.call1(py2, (x,))
                    .and_then(|v| v.extract::<f64>(py2))
                    .unwrap_or(f64::NAN)
            })
        },
        a,
        b,
        &opts,
    )
    .map_err(opt_err)?;
    root_to_dict(py, &r)
}

/// Find a root of `f` using Newton's method, given the derivative `fprime`.
///
/// Parameters: `f` — objective, `fprime` — derivative, `x0` — initial guess.
/// Returns a dict with keys: `root`, `f_root`, `iterations`, `converged`.
#[pyfunction]
#[pyo3(signature = (f, fprime, x0, xtol=1e-12, ftol=1e-12, max_iter=100))]
fn newton<'py>(
    py: Python<'py>,
    f: PyObject,
    fprime: PyObject,
    x0: f64,
    xtol: f64,
    ftol: f64,
    max_iter: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let opts = root_opts(xtol, ftol, max_iter);
    let r = scivex_optim::newton(
        |x| {
            Python::with_gil(|py2| {
                f.call1(py2, (x,))
                    .and_then(|v| v.extract::<f64>(py2))
                    .unwrap_or(f64::NAN)
            })
        },
        |x| {
            Python::with_gil(|py2| {
                fprime
                    .call1(py2, (x,))
                    .and_then(|v| v.extract::<f64>(py2))
                    .unwrap_or(f64::NAN)
            })
        },
        x0,
        &opts,
    )
    .map_err(opt_err)?;
    root_to_dict(py, &r)
}

// ===========================================================================
// NUMERICAL INTEGRATION
// ===========================================================================

fn quad_to_dict<'py>(
    py: Python<'py>,
    r: &scivex_optim::QuadResult<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("value", r.value)?;
    d.set_item("error_estimate", r.error_estimate)?;
    d.set_item("n_evals", r.n_evals)?;
    Ok(d)
}

/// Numerically integrate `f` over `[a, b]` using the composite trapezoidal rule with `n` panels.
///
/// Returns a dict with keys: `value`, `error_estimate`, `n_evals`.
#[pyfunction]
#[pyo3(signature = (f, a, b, n=1000))]
fn trapezoid<'py>(
    py: Python<'py>,
    f: PyObject,
    a: f64,
    b: f64,
    n: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let r = scivex_optim::trapezoid(
        |x| {
            Python::with_gil(|py2| {
                f.call1(py2, (x,))
                    .and_then(|v| v.extract::<f64>(py2))
                    .unwrap_or(f64::NAN)
            })
        },
        a,
        b,
        n,
    );
    quad_to_dict(py, &r)
}

/// Numerically integrate `f` over `[a, b]` using Simpson's rule with `n` panels.
///
/// Returns a dict with keys: `value`, `error_estimate`, `n_evals`.
#[pyfunction]
#[pyo3(signature = (f, a, b, n=1000))]
fn simpson<'py>(
    py: Python<'py>,
    f: PyObject,
    a: f64,
    b: f64,
    n: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let r = scivex_optim::simpson(
        |x| {
            Python::with_gil(|py2| {
                f.call1(py2, (x,))
                    .and_then(|v| v.extract::<f64>(py2))
                    .unwrap_or(f64::NAN)
            })
        },
        a,
        b,
        n,
    )
    .map_err(opt_err)?;
    quad_to_dict(py, &r)
}

/// Adaptive numerical integration of `f` over `[a, b]` using Gauss-Kronrod quadrature.
///
/// Returns a dict with keys: `value`, `error_estimate`, `n_evals`.
#[pyfunction]
#[pyo3(signature = (f, a, b, abs_tol=1e-10, rel_tol=1e-10, max_subdivisions=50))]
fn quad<'py>(
    py: Python<'py>,
    f: PyObject,
    a: f64,
    b: f64,
    abs_tol: f64,
    rel_tol: f64,
    max_subdivisions: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let opts = scivex_optim::QuadOptions {
        abs_tol,
        rel_tol,
        max_subdivisions,
    };
    let r = scivex_optim::quad(
        |x| {
            Python::with_gil(|py2| {
                f.call1(py2, (x,))
                    .and_then(|v| v.extract::<f64>(py2))
                    .unwrap_or(f64::NAN)
            })
        },
        a,
        b,
        &opts,
    )
    .map_err(opt_err)?;
    quad_to_dict(py, &r)
}

// ===========================================================================
// INTERPOLATION
// ===========================================================================

/// Interpolate 1-D data at the given `query` points.
///
/// `method` can be `"linear"`, `"cubic"` (cubic spline), or `"bspline"`.
/// Returns a list of interpolated values corresponding to `query`.
#[pyfunction]
#[pyo3(signature = (xs, ys, query, method="linear"))]
fn interp1d(xs: Vec<f64>, ys: Vec<f64>, query: Vec<f64>, method: &str) -> PyResult<Vec<f64>> {
    let m = match method.to_lowercase().as_str() {
        "linear" => scivex_optim::Interp1dMethod::Linear,
        "cubic" | "cubic_spline" => scivex_optim::Interp1dMethod::CubicSpline,
        "bspline" | "b-spline" => scivex_optim::Interp1dMethod::BSpline,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "method must be 'linear', 'cubic', or 'bspline'",
            ));
        }
    };
    scivex_optim::interp1d(&xs, &ys, &query, m).map_err(opt_err)
}

/// A cubic spline interpolator fitted to the given data points.
#[pyclass(name = "CubicSpline")]
pub struct PyCubicSpline {
    inner: scivex_optim::CubicSpline<f64>,
}

#[pymethods]
impl PyCubicSpline {
    /// Construct a cubic spline from knot points `xs` and values `ys`.
    ///
    /// `boundary` selects the boundary condition (currently only `"natural"`).
    #[new]
    #[pyo3(signature = (xs, ys, boundary="natural"))]
    fn new(xs: Vec<f64>, ys: Vec<f64>, boundary: &str) -> PyResult<Self> {
        let bc = match boundary.to_lowercase().as_str() {
            "natural" => scivex_optim::SplineBoundary::Natural,
            _ => scivex_optim::SplineBoundary::Natural,
        };
        let inner = scivex_optim::CubicSpline::new(&xs, &ys, bc, scivex_optim::Extrapolate::Clamp)
            .map_err(opt_err)?;
        Ok(Self { inner })
    }

    /// Evaluate the spline at a single point `x`.
    fn eval(&self, x: f64) -> PyResult<f64> {
        self.inner.eval(x).map_err(opt_err)
    }

    /// Evaluate the spline at multiple points, returning a list of values.
    fn eval_many(&self, xs: Vec<f64>) -> PyResult<Vec<f64>> {
        self.inner.eval_many(&xs).map_err(opt_err)
    }
}

// ===========================================================================
// MINIMIZATION
// ===========================================================================

fn minimize_opts(
    gtol: f64,
    xtol: f64,
    ftol: f64,
    max_iter: usize,
    learning_rate: f64,
) -> scivex_optim::MinimizeOptions<f64> {
    scivex_optim::MinimizeOptions {
        gtol,
        xtol,
        ftol,
        max_iter,
        learning_rate,
    }
}

fn minimize_to_dict<'py>(
    py: Python<'py>,
    r: &scivex_optim::MinimizeResult<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("x", PyTensor::from_f64(r.x.clone()).into_pyobject(py)?)?;
    d.set_item("fun", r.f_val)?;
    d.set_item("iterations", r.iterations)?;
    d.set_item("f_evals", r.f_evals)?;
    d.set_item("g_evals", r.g_evals)?;
    d.set_item("converged", r.converged)?;
    if let Some(ref g) = r.grad {
        d.set_item("grad", PyTensor::from_f64(g.clone()).into_pyobject(py)?)?;
    }
    Ok(d)
}

/// Call a Python callable that returns a scalar, passing a Tensor.
fn call_scalar_fn(f: &PyObject, x: &Tensor<f64>) -> f64 {
    Python::with_gil(|py| {
        let py_x = PyTensor::from_f64(x.clone()).into_pyobject(py).ok();
        match py_x {
            Some(px) => f
                .call1(py, (px,))
                .and_then(|v| v.extract::<f64>(py))
                .unwrap_or(f64::NAN),
            None => f64::NAN,
        }
    })
}

/// Call a Python callable that returns a Tensor (gradient), passing a Tensor.
fn call_grad_fn(g: &PyObject, x: &Tensor<f64>) -> Tensor<f64> {
    Python::with_gil(|py| {
        let py_x = PyTensor::from_f64(x.clone()).into_pyobject(py).ok();
        match py_x {
            Some(px) => g
                .call1(py, (px,))
                .and_then(|v| v.extract::<PyTensor>(py))
                .map(|t| t.to_f64_tensor())
                .unwrap_or_else(|_| Tensor::zeros(x.shape().to_vec())),
            None => Tensor::zeros(x.shape().to_vec()),
        }
    })
}

/// Minimize a scalar function of one or more variables.
///
/// `method` can be `"bfgs"`, `"nelder-mead"`, `"gradient-descent"`, or `"l-bfgs-b"`.
/// `jac` is an optional gradient callable (required for gradient-descent and l-bfgs-b).
/// Returns a dict with keys: `x`, `fun`, `iterations`, `f_evals`, `g_evals`, `converged`, `grad`.
#[pyfunction]
#[pyo3(signature = (f, x0, method="bfgs", jac=None, gtol=1e-5, xtol=1e-9, ftol=1e-9, max_iter=1000, learning_rate=0.01))]
#[allow(clippy::too_many_arguments)]
fn minimize<'py>(
    py: Python<'py>,
    f: PyObject,
    x0: &PyTensor,
    method: &str,
    jac: Option<PyObject>,
    gtol: f64,
    xtol: f64,
    ftol: f64,
    max_iter: usize,
    learning_rate: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let opts = minimize_opts(gtol, xtol, ftol, max_iter, learning_rate);
    let f_ref = f.clone_ref(py);

    let result = match method.to_lowercase().as_str() {
        "nelder-mead" | "nelder_mead" => {
            scivex_optim::nelder_mead(|x| call_scalar_fn(&f_ref, x), x0.as_f64()?, &opts)
                .map_err(opt_err)?
        }

        "gradient-descent" | "gradient_descent" | "gd" => {
            let g = jac.ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(
                    "gradient-descent requires 'jac' (gradient function)",
                )
            })?;
            scivex_optim::gradient_descent(
                |x| call_scalar_fn(&f_ref, x),
                |x| call_grad_fn(&g, x),
                x0.as_f64()?,
                &opts,
            )
            .map_err(opt_err)?
        }

        "bfgs" => {
            // Use numerical gradient if jac not provided
            let result = if let Some(g) = jac {
                scivex_optim::bfgs(
                    |x| call_scalar_fn(&f_ref, x),
                    |x| call_grad_fn(&g, x),
                    x0.as_f64()?,
                    &opts,
                )
            } else {
                let f_ref2 = f.clone_ref(py);
                scivex_optim::bfgs(
                    |x| call_scalar_fn(&f_ref, x),
                    |x| scivex_optim::numerical_gradient(&|xx| call_scalar_fn(&f_ref2, xx), x),
                    x0.as_f64()?,
                    &opts,
                )
            };
            result.map_err(opt_err)?
        }

        "l-bfgs-b" | "lbfgsb" => {
            let g = jac.ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(
                    "l-bfgs-b requires 'jac' (gradient function)",
                )
            })?;
            // No bounds from this interface — use empty bounds
            let bounds: Vec<scivex_optim::Bounds<f64>> =
                vec![scivex_optim::Bounds::none(); x0.as_f64()?.as_slice().len()];
            scivex_optim::lbfgsb(
                |x| call_scalar_fn(&f_ref, x),
                |x| call_grad_fn(&g, x),
                x0.as_f64()?,
                &bounds,
                &opts,
            )
            .map_err(opt_err)?
        }

        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "method must be 'bfgs', 'nelder-mead', 'gradient-descent', or 'l-bfgs-b'",
            ));
        }
    };

    minimize_to_dict(py, &result)
}

// ===========================================================================
// ODE SOLVERS
// ===========================================================================

/// Solve an initial-value ODE problem: y'(t) = f(t, y), y(t0) = y0.
///
/// `method` can be `"euler"`, `"rk45"`, or `"bdf2"`.
/// Returns a dict with keys: `t`, `y`, `n_evals`, `n_steps`, `success`.
#[pyfunction]
#[pyo3(signature = (f, t_span, y0, method="rk45", atol=1e-6, rtol=1e-3, max_steps=10000))]
#[allow(clippy::too_many_arguments)]
fn solve_ivp<'py>(
    py: Python<'py>,
    f: PyObject,
    t_span: [f64; 2],
    y0: Vec<f64>,
    method: &str,
    atol: f64,
    rtol: f64,
    max_steps: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let m = match method.to_lowercase().as_str() {
        "euler" => scivex_optim::OdeMethod::Euler,
        "rk45" => scivex_optim::OdeMethod::RK45,
        "bdf2" | "bdf" => scivex_optim::OdeMethod::BDF2,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "method must be 'euler', 'rk45', or 'bdf2'",
            ));
        }
    };

    let opts = scivex_optim::OdeOptions {
        atol,
        rtol,
        first_step: None,
        max_steps,
        event_fn: None,
    };

    let r = scivex_optim::solve_ivp(
        |t, y| {
            Python::with_gil(|py2| {
                let y_list: Vec<f64> = y.to_vec();
                f.call1(py2, (t, y_list))
                    .and_then(|v| v.extract::<Vec<f64>>(py2))
                    .unwrap_or_else(|_| vec![f64::NAN; y.len()])
            })
        },
        t_span,
        &y0,
        m,
        &opts,
    )
    .map_err(opt_err)?;

    let d = PyDict::new(py);
    d.set_item("t", r.t)?;
    d.set_item("y", r.y)?;
    d.set_item("n_evals", r.n_evals)?;
    d.set_item("n_steps", r.n_steps)?;
    d.set_item("success", r.success)?;
    Ok(d)
}

// ===========================================================================
// CURVE FITTING
// ===========================================================================

/// Fit a parametric `model(x, params)` to observed `(x_data, y_data)` using Levenberg-Marquardt.
///
/// `p0` is the initial parameter guess.
/// Returns a dict with keys: `params`, `residuals`, `cost`, `iterations`, `converged`.
#[pyfunction]
#[pyo3(signature = (model, x_data, y_data, p0, max_iter=1000, tol=1e-8))]
fn curve_fit<'py>(
    py: Python<'py>,
    model: PyObject,
    x_data: Vec<f64>,
    y_data: Vec<f64>,
    p0: Vec<f64>,
    max_iter: usize,
    tol: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let r = scivex_optim::levenberg_marquardt(
        |x, params| {
            Python::with_gil(|py2| {
                let params_list: Vec<f64> = params.to_vec();
                model
                    .call1(py2, (x, params_list))
                    .and_then(|v| v.extract::<f64>(py2))
                    .unwrap_or(f64::NAN)
            })
        },
        &x_data,
        &y_data,
        &p0,
        max_iter,
        tol,
    )
    .map_err(opt_err)?;

    let d = PyDict::new(py);
    d.set_item("params", r.params)?;
    d.set_item("residuals", r.residuals)?;
    d.set_item("cost", r.cost)?;
    d.set_item("iterations", r.iterations)?;
    d.set_item("converged", r.converged)?;
    Ok(d)
}

// ===========================================================================
// LINEAR PROGRAMMING
// ===========================================================================

/// Solve a linear program: minimize `c^T x` subject to `A_ub @ x <= b_ub`, `x >= 0`.
///
/// Returns a dict with keys: `x`, `fun`, `slack`, `iterations`, `converged`.
#[pyfunction]
fn linprog<'py>(
    py: Python<'py>,
    c: Vec<f64>,
    a_ub: Vec<Vec<f64>>,
    b_ub: Vec<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let r = scivex_optim::linprog(&c, &a_ub, &b_ub).map_err(opt_err)?;
    let d = PyDict::new(py);
    d.set_item("x", r.x)?;
    d.set_item("fun", r.fun)?;
    d.set_item("slack", r.slack)?;
    d.set_item("iterations", r.iterations)?;
    d.set_item("converged", r.converged)?;
    Ok(d)
}

// ===========================================================================
// 1-D MINIMIZATION
// ===========================================================================

fn min1d_to_dict<'py>(
    py: Python<'py>,
    r: &scivex_optim::Minimize1dResult<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("x_min", r.x_min)?;
    d.set_item("f_min", r.f_min)?;
    d.set_item("iterations", r.iterations)?;
    d.set_item("converged", r.converged)?;
    Ok(d)
}

/// Find the minimum of a unimodal function `f` on `[a, b]` using golden-section search.
///
/// Returns a dict with keys: `x_min`, `f_min`, `iterations`, `converged`.
#[pyfunction]
#[pyo3(signature = (f, a, b, xtol=1e-12, max_iter=500))]
fn golden_section<'py>(
    py: Python<'py>,
    f: PyObject,
    a: f64,
    b: f64,
    xtol: f64,
    max_iter: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let r = scivex_optim::golden_section(
        |x| {
            Python::with_gil(|py2| {
                f.call1(py2, (x,))
                    .and_then(|v| v.extract::<f64>(py2))
                    .unwrap_or(f64::NAN)
            })
        },
        a,
        b,
        xtol,
        max_iter,
    )
    .map_err(opt_err)?;
    min1d_to_dict(py, &r)
}

/// Find the minimum of a unimodal function `f` on `[a, b]` using Brent's method.
///
/// Returns a dict with keys: `x_min`, `f_min`, `iterations`, `converged`.
#[pyfunction]
#[pyo3(signature = (f, a, b, xtol=1e-12, max_iter=500))]
fn brent_min<'py>(
    py: Python<'py>,
    f: PyObject,
    a: f64,
    b: f64,
    xtol: f64,
    max_iter: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let r = scivex_optim::brent_min(
        |x| {
            Python::with_gil(|py2| {
                f.call1(py2, (x,))
                    .and_then(|v| v.extract::<f64>(py2))
                    .unwrap_or(f64::NAN)
            })
        },
        a,
        b,
        xtol,
        max_iter,
    )
    .map_err(opt_err)?;
    min1d_to_dict(py, &r)
}

// ===========================================================================
// PDE SOLVERS
// ===========================================================================

fn pde_to_dict<'py>(
    py: Python<'py>,
    r: &scivex_optim::PdeResult<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("u", &r.u)?;
    d.set_item("x", &r.x)?;
    d.set_item("t", &r.t_or_y)?;
    d.set_item("steps", r.steps)?;
    d.set_item("converged", r.converged)?;
    Ok(d)
}

/// Solve the 1-D heat equation u_t = alpha * u_xx with Dirichlet boundary conditions.
///
/// `initial` is a callable providing the initial temperature distribution.
/// Returns a dict with keys: `u`, `x`, `t`, `steps`, `converged`.
#[pyfunction]
#[pyo3(signature = (initial, x_range, n_x, t_final, n_t, alpha=1.0, left_bc=0.0, right_bc=0.0))]
#[allow(clippy::too_many_arguments)]
fn heat_equation_1d<'py>(
    py: Python<'py>,
    initial: PyObject,
    x_range: (f64, f64),
    n_x: usize,
    t_final: f64,
    n_t: usize,
    alpha: f64,
    left_bc: f64,
    right_bc: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let r = scivex_optim::heat_equation_1d(
        x_range,
        n_x,
        t_final,
        n_t,
        alpha,
        &|x| {
            Python::with_gil(|py2| {
                initial
                    .call1(py2, (x,))
                    .and_then(|v| v.extract::<f64>(py2))
                    .unwrap_or(0.0)
            })
        },
        scivex_optim::BoundaryCondition::Dirichlet(left_bc),
        scivex_optim::BoundaryCondition::Dirichlet(right_bc),
    )
    .map_err(opt_err)?;

    pde_to_dict(py, &r)
}

/// Solve the 1-D wave equation u_tt = c^2 * u_xx with Dirichlet boundary conditions.
///
/// `initial_u` provides the initial displacement, `initial_ut` the initial velocity.
/// Returns a dict with keys: `u`, `x`, `t`, `steps`, `converged`.
#[pyfunction]
#[pyo3(signature = (initial_u, initial_ut, x_range, n_x, t_final, n_t, c=1.0, left_bc=0.0, right_bc=0.0))]
#[allow(clippy::too_many_arguments)]
fn wave_equation_1d<'py>(
    py: Python<'py>,
    initial_u: PyObject,
    initial_ut: PyObject,
    x_range: (f64, f64),
    n_x: usize,
    t_final: f64,
    n_t: usize,
    c: f64,
    left_bc: f64,
    right_bc: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let r = scivex_optim::wave_equation_1d(
        x_range,
        n_x,
        t_final,
        n_t,
        c,
        &|x| {
            Python::with_gil(|py2| {
                initial_u
                    .call1(py2, (x,))
                    .and_then(|v| v.extract::<f64>(py2))
                    .unwrap_or(0.0)
            })
        },
        &|x| {
            Python::with_gil(|py2| {
                initial_ut
                    .call1(py2, (x,))
                    .and_then(|v| v.extract::<f64>(py2))
                    .unwrap_or(0.0)
            })
        },
        scivex_optim::BoundaryCondition::Dirichlet(left_bc),
        scivex_optim::BoundaryCondition::Dirichlet(right_bc),
    )
    .map_err(opt_err)?;

    pde_to_dict(py, &r)
}

/// Solve the 2-D Laplace equation (steady-state) on a rectangular domain.
///
/// `boundary` is a callable `(x, y) -> float | None`.  Return a float for boundary
/// points and `None` for interior points.
/// Returns a dict with keys: `u`, `x`, `t` (y-axis grid), `steps`, `converged`.
#[pyfunction]
#[pyo3(signature = (boundary, x_range, y_range, n_x, n_y, max_iter=10000, tol=1e-8))]
#[allow(clippy::too_many_arguments)]
fn laplace_2d<'py>(
    py: Python<'py>,
    boundary: PyObject,
    x_range: (f64, f64),
    y_range: (f64, f64),
    n_x: usize,
    n_y: usize,
    max_iter: usize,
    tol: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let r = scivex_optim::laplace_2d(
        x_range,
        y_range,
        n_x,
        n_y,
        &|x, y| {
            Python::with_gil(|py2| {
                boundary
                    .call1(py2, (x, y))
                    .and_then(|v| v.extract::<Option<f64>>(py2))
                    .unwrap_or(None)
            })
        },
        max_iter,
        tol,
    )
    .map_err(opt_err)?;

    pde_to_dict(py, &r)
}

// ===========================================================================
// SPARSE SOLVERS
// ===========================================================================

/// Solve a sparse symmetric positive-definite system `Ax = b` using
/// the Conjugate Gradient method.
///
/// `a_rows`, `a_cols`, `a_vals` describe the sparse matrix in COO (triplet)
/// format.  Returns a dict with keys: `x`, `iterations`, `residual_norm`,
/// `converged`.
#[pyfunction]
#[pyo3(signature = (n, a_rows, a_cols, a_vals, b, x0=None, max_iter=1000, tol=1e-10))]
#[allow(clippy::too_many_arguments)]
fn conjugate_gradient<'py>(
    py: Python<'py>,
    n: usize,
    a_rows: Vec<usize>,
    a_cols: Vec<usize>,
    a_vals: Vec<f64>,
    b: Vec<f64>,
    x0: Option<Vec<f64>>,
    max_iter: usize,
    tol: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let csr = scivex_core::linalg::CsrMatrix::from_triplets(n, n, a_rows, a_cols, a_vals)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let result = scivex_optim::conjugate_gradient(&csr, &b, x0.as_deref(), max_iter, tol)
        .map_err(opt_err)?;
    let d = PyDict::new(py);
    d.set_item("x", result.x)?;
    d.set_item("iterations", result.iterations)?;
    d.set_item("residual_norm", result.residual_norm)?;
    d.set_item("converged", result.converged)?;
    Ok(d)
}

/// Solve a general (non-symmetric) sparse system `Ax = b` using BiCGSTAB.
///
/// `a_rows`, `a_cols`, `a_vals` describe the sparse matrix in COO (triplet)
/// format.  Returns a dict with keys: `x`, `iterations`, `residual_norm`,
/// `converged`.
#[pyfunction]
#[pyo3(signature = (n, a_rows, a_cols, a_vals, b, x0=None, max_iter=1000, tol=1e-10))]
#[allow(clippy::too_many_arguments)]
fn bicgstab<'py>(
    py: Python<'py>,
    n: usize,
    a_rows: Vec<usize>,
    a_cols: Vec<usize>,
    a_vals: Vec<f64>,
    b: Vec<f64>,
    x0: Option<Vec<f64>>,
    max_iter: usize,
    tol: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let csr = scivex_core::linalg::CsrMatrix::from_triplets(n, n, a_rows, a_cols, a_vals)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let result = scivex_optim::bicgstab(&csr, &b, x0.as_deref(), max_iter, tol).map_err(opt_err)?;
    let d = PyDict::new(py);
    d.set_item("x", result.x)?;
    d.set_item("iterations", result.iterations)?;
    d.set_item("residual_norm", result.residual_norm)?;
    d.set_item("converged", result.converged)?;
    Ok(d)
}

// ===========================================================================
// QUADRATIC PROGRAMMING
// ===========================================================================

/// Solve a convex quadratic program: minimize `0.5 x^T H x + c^T x`
/// subject to `A_ub @ x <= b_ub`.
///
/// Returns a dict with keys: `x`, `fun`, `iterations`, `converged`.
#[pyfunction]
#[pyo3(signature = (h, c, a_ub, b_ub, max_iter=200))]
fn quadprog<'py>(
    py: Python<'py>,
    h: Vec<Vec<f64>>,
    c: Vec<f64>,
    a_ub: Vec<Vec<f64>>,
    b_ub: Vec<f64>,
    max_iter: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let r = scivex_optim::quadprog(&h, &c, &a_ub, &b_ub, max_iter).map_err(opt_err)?;
    let d = PyDict::new(py);
    d.set_item("x", r.x)?;
    d.set_item("fun", r.fun)?;
    d.set_item("iterations", r.iterations)?;
    d.set_item("converged", r.converged)?;
    Ok(d)
}

// ===========================================================================
// 2-D INTERPOLATION
// ===========================================================================

/// Interpolate on a 2-D rectilinear grid.
///
/// `xs` and `ys` are the grid coordinates; `zs` is a 2-D list (`zs[i][j]`
/// corresponds to `(xs[i], ys[j])`).  `query` is a list of `[x, y]` pairs.
/// `method` can be `"bilinear"` or `"bicubic"`.
///
/// Returns a list of interpolated values.
#[pyfunction]
#[pyo3(signature = (xs, ys, zs, query, method="bilinear"))]
fn interp2d(
    xs: Vec<f64>,
    ys: Vec<f64>,
    zs: Vec<Vec<f64>>,
    query: Vec<[f64; 2]>,
    method: &str,
) -> PyResult<Vec<f64>> {
    let m = match method.to_lowercase().as_str() {
        "bilinear" => scivex_optim::Interp2dMethod::Bilinear,
        "bicubic" => scivex_optim::Interp2dMethod::Bicubic,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "method must be 'bilinear' or 'bicubic'",
            ));
        }
    };
    let pairs: Vec<(f64, f64)> = query.iter().map(|p| (p[0], p[1])).collect();
    scivex_optim::interp2d(xs, ys, zs, &pairs, m).map_err(opt_err)
}

// ===========================================================================
// B-SPLINE
// ===========================================================================

/// A B-spline interpolator.
#[pyclass(name = "BSpline")]
pub struct PyBSpline {
    inner: scivex_optim::BSpline<f64>,
}

#[pymethods]
impl PyBSpline {
    /// Fit a B-spline of given `degree` that interpolates `(xs, ys)`.
    #[new]
    #[pyo3(signature = (xs, ys, degree=3))]
    fn new(xs: Vec<f64>, ys: Vec<f64>, degree: usize) -> PyResult<Self> {
        let inner = scivex_optim::BSpline::fit(&xs, &ys, degree, scivex_optim::Extrapolate::Clamp)
            .map_err(opt_err)?;
        Ok(Self { inner })
    }

    /// Evaluate the B-spline at a single point.
    fn eval(&self, x: f64) -> PyResult<f64> {
        self.inner.eval(x).map_err(opt_err)
    }

    /// Evaluate the B-spline at multiple points.
    fn eval_many(&self, xs: Vec<f64>) -> PyResult<Vec<f64>> {
        self.inner.eval_many(&xs).map_err(opt_err)
    }
}

// ===========================================================================
// PRECONDITIONED CONJUGATE GRADIENT
// ===========================================================================

/// Solve `Ax = b` using the Preconditioned Conjugate Gradient method
/// with Jacobi (diagonal) preconditioning.
///
/// `a_rows`, `a_cols`, `a_vals` describe the sparse matrix in COO (triplet)
/// format.  Returns a dict with keys: `x`, `iterations`, `residual_norm`,
/// `converged`.
#[pyfunction]
#[pyo3(signature = (n, a_rows, a_cols, a_vals, b, x0=None, max_iter=1000, tol=1e-10))]
#[allow(clippy::too_many_arguments)]
fn preconditioned_cg<'py>(
    py: Python<'py>,
    n: usize,
    a_rows: Vec<usize>,
    a_cols: Vec<usize>,
    a_vals: Vec<f64>,
    b: Vec<f64>,
    x0: Option<Vec<f64>>,
    max_iter: usize,
    tol: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let csr = scivex_core::linalg::CsrMatrix::from_triplets(n, n, a_rows, a_cols, a_vals)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let precond = scivex_optim::JacobiPreconditioner::new(&csr);
    let result = scivex_optim::preconditioned_cg(&csr, &b, &precond, x0.as_deref(), max_iter, tol)
        .map_err(opt_err)?;
    let d = PyDict::new(py);
    d.set_item("x", result.x)?;
    d.set_item("iterations", result.iterations)?;
    d.set_item("residual_norm", result.residual_norm)?;
    d.set_item("converged", result.converged)?;
    Ok(d)
}

// ===========================================================================
// NUMERICAL GRADIENT
// ===========================================================================

/// Compute the numerical gradient of a scalar function `f` at point `x`
/// using central finite differences.
///
/// Returns a Tensor of the same shape as `x`.
#[pyfunction]
fn numerical_gradient(f: PyObject, x: &PyTensor) -> PyResult<PyTensor> {
    let x_tensor = x.as_f64()?.clone();
    let grad =
        scivex_optim::numerical_gradient(&|xv: &Tensor<f64>| call_scalar_fn(&f, xv), &x_tensor);
    Ok(PyTensor::from_f64(grad))
}

// ===========================================================================
// INDIVIDUAL ODE SOLVERS
// ===========================================================================

/// Helper: build ODE options and result dict for individual solvers.
fn ode_opts(atol: f64, rtol: f64, max_steps: usize) -> scivex_optim::OdeOptions<f64> {
    scivex_optim::OdeOptions {
        atol,
        rtol,
        first_step: None,
        max_steps,
        event_fn: None,
    }
}

fn ode_to_dict<'py>(
    py: Python<'py>,
    r: &scivex_optim::OdeResult<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("t", &r.t)?;
    d.set_item("y", &r.y)?;
    d.set_item("n_evals", r.n_evals)?;
    d.set_item("n_steps", r.n_steps)?;
    d.set_item("success", r.success)?;
    Ok(d)
}

/// Helper: create the ODE closure from a Python callable.
fn make_ode_fn(f: &PyObject) -> impl Fn(f64, &[f64]) -> Vec<f64> + '_ {
    move |t, y| {
        Python::with_gil(|py2| {
            let y_list: Vec<f64> = y.to_vec();
            f.call1(py2, (t, y_list))
                .and_then(|v| v.extract::<Vec<f64>>(py2))
                .unwrap_or_else(|_| vec![f64::NAN; y.len()])
        })
    }
}

/// Solve an ODE using the forward Euler method.
///
/// Returns a dict with keys: `t`, `y`, `n_evals`, `n_steps`, `success`.
#[pyfunction]
#[pyo3(signature = (f, t_span, y0, atol=1e-6, rtol=1e-3, max_steps=10000))]
fn euler<'py>(
    py: Python<'py>,
    f: PyObject,
    t_span: [f64; 2],
    y0: Vec<f64>,
    atol: f64,
    rtol: f64,
    max_steps: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let opts = ode_opts(atol, rtol, max_steps);
    let r = scivex_optim::euler(make_ode_fn(&f), t_span, &y0, &opts).map_err(opt_err)?;
    ode_to_dict(py, &r)
}

/// Solve an ODE using the Dormand-Prince RK4(5) adaptive method.
///
/// Returns a dict with keys: `t`, `y`, `n_evals`, `n_steps`, `success`.
#[pyfunction]
#[pyo3(signature = (f, t_span, y0, atol=1e-6, rtol=1e-3, max_steps=10000))]
fn rk45<'py>(
    py: Python<'py>,
    f: PyObject,
    t_span: [f64; 2],
    y0: Vec<f64>,
    atol: f64,
    rtol: f64,
    max_steps: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let opts = ode_opts(atol, rtol, max_steps);
    let r = scivex_optim::rk45(make_ode_fn(&f), t_span, &y0, &opts).map_err(opt_err)?;
    ode_to_dict(py, &r)
}

/// Solve an ODE using the BDF-2 implicit method (suited for stiff systems).
///
/// Returns a dict with keys: `t`, `y`, `n_evals`, `n_steps`, `success`.
#[pyfunction]
#[pyo3(signature = (f, t_span, y0, atol=1e-6, rtol=1e-3, max_steps=10000))]
fn bdf2<'py>(
    py: Python<'py>,
    f: PyObject,
    t_span: [f64; 2],
    y0: Vec<f64>,
    atol: f64,
    rtol: f64,
    max_steps: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let opts = ode_opts(atol, rtol, max_steps);
    let r = scivex_optim::bdf2(make_ode_fn(&f), t_span, &y0, &opts).map_err(opt_err)?;
    ode_to_dict(py, &r)
}

// ===========================================================================
// Submodule registration
// ===========================================================================

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let m = PyModule::new(py, "optim")?;

    // Root finding
    m.add_function(wrap_pyfunction!(bisection, &m)?)?;
    m.add_function(wrap_pyfunction!(brentq, &m)?)?;
    m.add_function(wrap_pyfunction!(newton, &m)?)?;

    // Integration
    m.add_function(wrap_pyfunction!(trapezoid, &m)?)?;
    m.add_function(wrap_pyfunction!(simpson, &m)?)?;
    m.add_function(wrap_pyfunction!(quad, &m)?)?;

    // Interpolation
    m.add_function(wrap_pyfunction!(interp1d, &m)?)?;
    m.add_function(wrap_pyfunction!(interp2d, &m)?)?;
    m.add_class::<PyCubicSpline>()?;
    m.add_class::<PyBSpline>()?;

    // Minimization
    m.add_function(wrap_pyfunction!(minimize, &m)?)?;
    m.add_function(wrap_pyfunction!(numerical_gradient, &m)?)?;

    // 1-D minimization
    m.add_function(wrap_pyfunction!(golden_section, &m)?)?;
    m.add_function(wrap_pyfunction!(brent_min, &m)?)?;

    // ODE solvers
    m.add_function(wrap_pyfunction!(solve_ivp, &m)?)?;
    m.add_function(wrap_pyfunction!(euler, &m)?)?;
    m.add_function(wrap_pyfunction!(rk45, &m)?)?;
    m.add_function(wrap_pyfunction!(bdf2, &m)?)?;

    // Curve fitting
    m.add_function(wrap_pyfunction!(curve_fit, &m)?)?;

    // Linear programming
    m.add_function(wrap_pyfunction!(linprog, &m)?)?;

    // Quadratic programming
    m.add_function(wrap_pyfunction!(quadprog, &m)?)?;

    // PDE solvers
    m.add_function(wrap_pyfunction!(heat_equation_1d, &m)?)?;
    m.add_function(wrap_pyfunction!(wave_equation_1d, &m)?)?;
    m.add_function(wrap_pyfunction!(laplace_2d, &m)?)?;

    // Sparse solvers
    m.add_function(wrap_pyfunction!(conjugate_gradient, &m)?)?;
    m.add_function(wrap_pyfunction!(bicgstab, &m)?)?;
    m.add_function(wrap_pyfunction!(preconditioned_cg, &m)?)?;

    parent.add_submodule(&m)?;
    Ok(())
}
