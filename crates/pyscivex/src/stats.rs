//! Python bindings for `scivex-stats`.
//!
//! Top-level convenience functions (`mean`, `variance`, `std_dev`, `median`,
//! `pearson`) are registered directly on the root module by `lib.rs`.
//!
//! Everything else lives under a `stats` submodule created by [`register`].

use pyo3::prelude::*;
use pyo3::types::PyDict;
use scivex_core::random::Rng;
use scivex_stats::distributions::Distribution;

use crate::tensor::PyTensor;

// ---------------------------------------------------------------------------
// Error conversion
// ---------------------------------------------------------------------------

/// Convert a `StatsError` into a Python `ValueError`.
#[allow(clippy::needless_pass_by_value)]
fn stats_err(e: scivex_stats::StatsError) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}

// =========================================================================
// Top-level functions (registered on the root module by lib.rs)
// =========================================================================

/// Arithmetic mean of a list of values.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub fn mean(data: Vec<f64>) -> PyResult<f64> {
    scivex_stats::mean(&data).map_err(stats_err)
}

/// Population variance of a list of values.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub fn variance(data: Vec<f64>) -> PyResult<f64> {
    scivex_stats::variance(&data).map_err(stats_err)
}

/// Population standard deviation of a list of values.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub fn std_dev(data: Vec<f64>) -> PyResult<f64> {
    scivex_stats::std_dev(&data).map_err(stats_err)
}

/// Median of a list of values.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub fn median(data: Vec<f64>) -> PyResult<f64> {
    scivex_stats::median(&data).map_err(stats_err)
}

/// Pearson correlation coefficient between two lists.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub fn pearson(x: Vec<f64>, y: Vec<f64>) -> PyResult<f64> {
    scivex_stats::pearson(&x, &y).map_err(stats_err)
}

// =========================================================================
// Submodule-only functions (registered under `stats.*`)
// =========================================================================

// ---------------------------------------------------------------------------
// Additional descriptive statistics
// ---------------------------------------------------------------------------

/// Compute the `q`-th quantile (0..1) of a list of values.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn quantile(data: Vec<f64>, q: f64) -> PyResult<f64> {
    scivex_stats::quantile(&data, q).map_err(stats_err)
}

/// Sample skewness.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn skewness(data: Vec<f64>) -> PyResult<f64> {
    scivex_stats::skewness(&data).map_err(stats_err)
}

/// Sample excess kurtosis.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn kurtosis(data: Vec<f64>) -> PyResult<f64> {
    scivex_stats::kurtosis(&data).map_err(stats_err)
}

/// Variance with a custom degrees-of-freedom adjustment.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn variance_with_ddof(data: Vec<f64>, ddof: usize) -> PyResult<f64> {
    scivex_stats::variance_with_ddof(&data, ddof).map_err(stats_err)
}

/// Standard deviation with a custom degrees-of-freedom adjustment.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn std_dev_with_ddof(data: Vec<f64>, ddof: usize) -> PyResult<f64> {
    scivex_stats::std_dev_with_ddof(&data, ddof).map_err(stats_err)
}

/// Compute summary statistics (count, mean, std, min, q25, median, q75, max,
/// skewness, kurtosis) and return them as a Python dict.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn describe(py: Python<'_>, data: Vec<f64>) -> PyResult<PyObject> {
    let r = scivex_stats::describe(&data).map_err(stats_err)?;
    let d = PyDict::new(py);
    d.set_item("count", r.count)?;
    d.set_item("mean", r.mean)?;
    d.set_item("std", r.std_dev)?;
    d.set_item("min", r.min)?;
    d.set_item("q25", r.q25)?;
    d.set_item("median", r.median)?;
    d.set_item("q75", r.q75)?;
    d.set_item("max", r.max)?;
    d.set_item("skewness", r.skewness)?;
    d.set_item("kurtosis", r.kurtosis)?;
    Ok(d.into_any().unbind())
}

// ---------------------------------------------------------------------------
// Distribution classes
// ---------------------------------------------------------------------------

/// Macro to generate a PyO3 wrapper class for a `scivex-stats` distribution.
///
/// Each generated class exposes: `pdf`, `cdf`, `ppf`, `mean`, `variance`,
/// `sample`, and `sample_n`.
macro_rules! dist_pyclass {
    (two $py_name:ident, $py_str:literal, $rust_ty:ty, $p1:ident, $p2:ident) => {
        #[pyclass(name = $py_str)]
        struct $py_name {
            inner: $rust_ty,
        }

        #[pymethods]
        impl $py_name {
            #[new]
            fn new($p1: f64, $p2: f64) -> PyResult<Self> {
                let inner = <$rust_ty>::new($p1, $p2).map_err(stats_err)?;
                Ok(Self { inner })
            }
            fn pdf(&self, x: f64) -> f64 {
                Distribution::pdf(&self.inner, x)
            }
            fn cdf(&self, x: f64) -> f64 {
                Distribution::cdf(&self.inner, x)
            }
            fn ppf(&self, p: f64) -> PyResult<f64> {
                Distribution::ppf(&self.inner, p).map_err(stats_err)
            }
            fn mean(&self) -> f64 {
                Distribution::mean(&self.inner)
            }
            fn variance(&self) -> f64 {
                Distribution::variance(&self.inner)
            }
            fn sample(&self, seed: u64) -> f64 {
                let mut rng = Rng::new(seed);
                Distribution::sample(&self.inner, &mut rng)
            }
            fn sample_n(&self, n: usize, seed: u64) -> Vec<f64> {
                let mut rng = Rng::new(seed);
                Distribution::sample_n(&self.inner, &mut rng, n)
            }
        }
    };

    (one $py_name:ident, $py_str:literal, $rust_ty:ty, $p1:ident) => {
        #[pyclass(name = $py_str)]
        struct $py_name {
            inner: $rust_ty,
        }

        #[pymethods]
        impl $py_name {
            #[new]
            fn new($p1: f64) -> PyResult<Self> {
                let inner = <$rust_ty>::new($p1).map_err(stats_err)?;
                Ok(Self { inner })
            }
            fn pdf(&self, x: f64) -> f64 {
                Distribution::pdf(&self.inner, x)
            }
            fn cdf(&self, x: f64) -> f64 {
                Distribution::cdf(&self.inner, x)
            }
            fn ppf(&self, p: f64) -> PyResult<f64> {
                Distribution::ppf(&self.inner, p).map_err(stats_err)
            }
            fn mean(&self) -> f64 {
                Distribution::mean(&self.inner)
            }
            fn variance(&self) -> f64 {
                Distribution::variance(&self.inner)
            }
            fn sample(&self, seed: u64) -> f64 {
                let mut rng = Rng::new(seed);
                Distribution::sample(&self.inner, &mut rng)
            }
            fn sample_n(&self, n: usize, seed: u64) -> Vec<f64> {
                let mut rng = Rng::new(seed);
                Distribution::sample_n(&self.inner, &mut rng, n)
            }
        }
    };
}

// Two-parameter distributions
dist_pyclass!(two PyNormal, "Normal", scivex_stats::distributions::Normal<f64>, mu, sigma);
dist_pyclass!(two PyUniform, "Uniform", scivex_stats::distributions::Uniform<f64>, a, b);
dist_pyclass!(two PyBeta, "Beta", scivex_stats::distributions::Beta<f64>, alpha, beta);
dist_pyclass!(two PyGamma, "Gamma", scivex_stats::distributions::Gamma<f64>, alpha, beta);
dist_pyclass!(two PyCauchy, "Cauchy", scivex_stats::distributions::Cauchy<f64>, x0, gamma);
dist_pyclass!(two PyLogNormal, "LogNormal", scivex_stats::distributions::LogNormal<f64>, mu, sigma);
dist_pyclass!(two PyWeibull, "Weibull", scivex_stats::distributions::Weibull<f64>, k, lambda);
dist_pyclass!(two PyLaplace, "Laplace", scivex_stats::distributions::Laplace<f64>, mu, b);

// Single-parameter distributions
dist_pyclass!(one PyExponential, "Exponential", scivex_stats::distributions::Exponential<f64>, lambda);
dist_pyclass!(one PyStudentT, "StudentT", scivex_stats::distributions::StudentT<f64>, df);
dist_pyclass!(one PyPoisson, "Poisson", scivex_stats::distributions::Poisson<f64>, lambda);
dist_pyclass!(one PyChiSquared, "ChiSquared", scivex_stats::distributions::ChiSquared<f64>, df);
dist_pyclass!(one PyBernoulli, "Bernoulli", scivex_stats::distributions::Bernoulli<f64>, p);

// NegativeBinomial: two f64 params, can use the macro
dist_pyclass!(two PyNegativeBinomial, "NegativeBinomial", scivex_stats::distributions::NegativeBinomial<f64>, r, p);

// Pareto: two f64 params, can use the macro
dist_pyclass!(two PyPareto, "Pareto", scivex_stats::distributions::Pareto<f64>, alpha, x_m);

// Binomial needs special handling: Python `n` is an integer, Rust takes `usize`.
#[pyclass(name = "Binomial")]
struct PyBinomial {
    inner: scivex_stats::distributions::Binomial<f64>,
}

#[pymethods]
impl PyBinomial {
    #[new]
    fn new(n: usize, p: f64) -> PyResult<Self> {
        let inner = scivex_stats::distributions::Binomial::new(n, p).map_err(stats_err)?;
        Ok(Self { inner })
    }

    fn pdf(&self, x: f64) -> f64 {
        Distribution::pdf(&self.inner, x)
    }

    fn cdf(&self, x: f64) -> f64 {
        Distribution::cdf(&self.inner, x)
    }

    fn ppf(&self, p: f64) -> PyResult<f64> {
        Distribution::ppf(&self.inner, p).map_err(stats_err)
    }

    fn mean(&self) -> f64 {
        Distribution::mean(&self.inner)
    }

    fn variance(&self) -> f64 {
        Distribution::variance(&self.inner)
    }

    fn sample(&self, seed: u64) -> f64 {
        let mut rng = Rng::new(seed);
        Distribution::sample(&self.inner, &mut rng)
    }

    fn sample_n(&self, n: usize, seed: u64) -> Vec<f64> {
        let mut rng = Rng::new(seed);
        Distribution::sample_n(&self.inner, &mut rng, n)
    }
}

// Hypergeometric needs special handling: all three params are usize.
#[pyclass(name = "Hypergeometric")]
struct PyHypergeometric {
    inner: scivex_stats::distributions::Hypergeometric<f64>,
}

#[pymethods]
impl PyHypergeometric {
    #[new]
    fn new(big_n: usize, big_k: usize, n: usize) -> PyResult<Self> {
        let inner =
            scivex_stats::distributions::Hypergeometric::new(big_n, big_k, n).map_err(stats_err)?;
        Ok(Self { inner })
    }

    fn pdf(&self, x: f64) -> f64 {
        Distribution::pdf(&self.inner, x)
    }

    fn cdf(&self, x: f64) -> f64 {
        Distribution::cdf(&self.inner, x)
    }

    fn ppf(&self, p: f64) -> PyResult<f64> {
        Distribution::ppf(&self.inner, p).map_err(stats_err)
    }

    fn mean(&self) -> f64 {
        Distribution::mean(&self.inner)
    }

    fn variance(&self) -> f64 {
        Distribution::variance(&self.inner)
    }

    fn sample(&self, seed: u64) -> f64 {
        let mut rng = Rng::new(seed);
        Distribution::sample(&self.inner, &mut rng)
    }

    fn sample_n(&self, n: usize, seed: u64) -> Vec<f64> {
        let mut rng = Rng::new(seed);
        Distribution::sample_n(&self.inner, &mut rng, n)
    }
}

// ---------------------------------------------------------------------------
// Hypothesis tests
// ---------------------------------------------------------------------------

/// Convert a `TestResult<f64>` to a Python dict.
fn test_result_to_dict(py: Python<'_>, r: &scivex_stats::TestResult<f64>) -> PyResult<PyObject> {
    let d = PyDict::new(py);
    d.set_item("statistic", r.statistic)?;
    d.set_item("p_value", r.p_value)?;
    d.set_item("df", r.df)?;
    Ok(d.into_any().unbind())
}

/// One-sample t-test: test whether the population mean equals `mu`.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn ttest_1samp(py: Python<'_>, data: Vec<f64>, mu: f64) -> PyResult<PyObject> {
    let r = scivex_stats::t_test_one_sample(&data, mu).map_err(stats_err)?;
    test_result_to_dict(py, &r)
}

/// Independent two-sample t-test.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn ttest_ind(py: Python<'_>, x: Vec<f64>, y: Vec<f64>) -> PyResult<PyObject> {
    let r = scivex_stats::t_test_two_sample(&x, &y).map_err(stats_err)?;
    test_result_to_dict(py, &r)
}

/// Chi-square goodness-of-fit test.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn chi2_test(py: Python<'_>, observed: Vec<f64>, expected: Vec<f64>) -> PyResult<PyObject> {
    let r = scivex_stats::chi_square_test(&observed, &expected).map_err(stats_err)?;
    test_result_to_dict(py, &r)
}

/// Two-sample Kolmogorov-Smirnov test.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn ks_2samp(py: Python<'_>, x: Vec<f64>, y: Vec<f64>) -> PyResult<PyObject> {
    let r = scivex_stats::ks_test_two_sample(&x, &y).map_err(stats_err)?;
    test_result_to_dict(py, &r)
}

/// One-way ANOVA.
///
/// `groups` is a list of lists, one per group.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn anova_oneway(py: Python<'_>, groups: Vec<Vec<f64>>) -> PyResult<PyObject> {
    let refs: Vec<&[f64]> = groups.iter().map(|g| g.as_slice()).collect();
    let r = scivex_stats::anova_oneway(&refs).map_err(stats_err)?;
    test_result_to_dict(py, &r)
}

/// Mann-Whitney U test.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn mann_whitney_u(py: Python<'_>, x: Vec<f64>, y: Vec<f64>) -> PyResult<PyObject> {
    let r = scivex_stats::mann_whitney_u(&x, &y).map_err(stats_err)?;
    test_result_to_dict(py, &r)
}

// ---------------------------------------------------------------------------
// Correlation
// ---------------------------------------------------------------------------

/// Spearman rank correlation coefficient.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn spearman(x: Vec<f64>, y: Vec<f64>) -> PyResult<f64> {
    scivex_stats::spearman(&x, &y).map_err(stats_err)
}

/// Kendall tau-b correlation coefficient.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn kendall(x: Vec<f64>, y: Vec<f64>) -> PyResult<f64> {
    scivex_stats::kendall(&x, &y).map_err(stats_err)
}

/// Compute a correlation matrix.
///
/// `method` is one of `"pearson"`, `"spearman"`, or `"kendall"`.
#[pyfunction]
fn corr_matrix(data: &PyTensor, method: &str) -> PyResult<PyTensor> {
    let m = match method {
        "pearson" => scivex_stats::CorrelationMethod::Pearson,
        "spearman" => scivex_stats::CorrelationMethod::Spearman,
        "kendall" => scivex_stats::CorrelationMethod::Kendall,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown correlation method: {other:?}; expected \"pearson\", \"spearman\", or \"kendall\""
            )));
        }
    };
    let result = scivex_stats::corr_matrix(data.as_f64()?, m).map_err(stats_err)?;
    Ok(PyTensor::from_f64(result))
}

// ---------------------------------------------------------------------------
// Regression
// ---------------------------------------------------------------------------

/// Ordinary Least Squares regression.
///
/// `x` is a 2-D Tensor `[n_obs, n_features]` (no intercept column needed).
/// Returns a dict with full regression summary.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn ols(py: Python<'_>, x: &PyTensor, y: Vec<f64>) -> PyResult<PyObject> {
    let r = scivex_stats::ols(x.as_f64()?, &y).map_err(stats_err)?;
    let d = PyDict::new(py);
    d.set_item("coefficients", r.coefficients.clone())?;
    d.set_item("std_errors", r.std_errors.clone())?;
    d.set_item("t_statistics", r.t_statistics.clone())?;
    d.set_item("p_values", r.p_values.clone())?;
    d.set_item("r_squared", r.r_squared)?;
    d.set_item("r_squared_adj", r.r_squared_adj)?;
    d.set_item("f_statistic", r.f_statistic)?;
    d.set_item("f_p_value", r.f_p_value)?;
    d.set_item("residuals", r.residuals.clone())?;
    d.set_item("n_obs", r.n_obs)?;
    d.set_item("df_resid", r.df_resid)?;
    Ok(d.into_any().unbind())
}

/// Generalized Linear Model (IRLS).
///
/// `family` is one of `"gaussian"`, `"binomial"`, `"poisson"`, or `"gamma"`.
/// `link` is optional; one of `"identity"`, `"logit"`, `"log"`, `"inverse"`.
/// If omitted, the canonical link for the chosen family is used.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
#[pyo3(signature = (x, y, family, link = None))]
fn glm(
    py: Python<'_>,
    x: &PyTensor,
    y: Vec<f64>,
    family: &str,
    link: Option<&str>,
) -> PyResult<PyObject> {
    let fam = match family {
        "gaussian" => scivex_stats::Family::Gaussian,
        "binomial" => scivex_stats::Family::Binomial,
        "poisson" => scivex_stats::Family::Poisson,
        "gamma" => scivex_stats::Family::Gamma,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown family: {other:?}; expected \"gaussian\", \"binomial\", \"poisson\", or \"gamma\""
            )));
        }
    };

    let lnk = match link {
        Some("identity") => scivex_stats::LinkFunction::Identity,
        Some("logit") => scivex_stats::LinkFunction::Logit,
        Some("log") => scivex_stats::LinkFunction::Log,
        Some("inverse") => scivex_stats::LinkFunction::Inverse,
        None => match fam {
            scivex_stats::Family::Gaussian => scivex_stats::LinkFunction::Identity,
            scivex_stats::Family::Binomial => scivex_stats::LinkFunction::Logit,
            scivex_stats::Family::Poisson => scivex_stats::LinkFunction::Log,
            scivex_stats::Family::Gamma => scivex_stats::LinkFunction::Inverse,
        },
        Some(other) => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown link: {other:?}; expected \"identity\", \"logit\", \"log\", or \"inverse\""
            )));
        }
    };

    let r = scivex_stats::glm(x.as_f64()?, &y, fam, lnk).map_err(stats_err)?;
    let d = PyDict::new(py);
    d.set_item("coefficients", r.coefficients.clone())?;
    d.set_item("std_errors", r.std_errors.clone())?;
    d.set_item("z_scores", r.z_scores.clone())?;
    d.set_item("p_values", r.p_values.clone())?;
    d.set_item("deviance", r.deviance)?;
    d.set_item("aic", r.aic)?;
    d.set_item("log_likelihood", r.log_likelihood)?;
    d.set_item("n_iter", r.n_iter)?;
    Ok(d.into_any().unbind())
}

// ---------------------------------------------------------------------------
// Confidence intervals
// ---------------------------------------------------------------------------

/// Confidence interval for the mean using the t-distribution.
///
/// Returns a dict with keys: `lower`, `upper`, `estimate`, `confidence`.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn ci_mean(py: Python<'_>, data: Vec<f64>, confidence: f64) -> PyResult<PyObject> {
    let r = scivex_stats::ci_mean(&data, confidence).map_err(stats_err)?;
    let d = PyDict::new(py);
    d.set_item("lower", r.lower)?;
    d.set_item("upper", r.upper)?;
    d.set_item("estimate", r.estimate)?;
    d.set_item("confidence", r.confidence)?;
    Ok(d.into_any().unbind())
}

/// Confidence interval for a proportion (Wald method).
///
/// Returns a dict with keys: `lower`, `upper`, `estimate`, `confidence`.
#[pyfunction]
fn ci_proportion(
    py: Python<'_>,
    successes: usize,
    total: usize,
    confidence: f64,
) -> PyResult<PyObject> {
    let r = scivex_stats::ci_proportion(successes, total, confidence).map_err(stats_err)?;
    let d = PyDict::new(py);
    d.set_item("lower", r.lower)?;
    d.set_item("upper", r.upper)?;
    d.set_item("estimate", r.estimate)?;
    d.set_item("confidence", r.confidence)?;
    Ok(d.into_any().unbind())
}

// ---------------------------------------------------------------------------
// Effect sizes
// ---------------------------------------------------------------------------

/// Cohen's d — standardised mean difference with pooled std.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn cohens_d(group1: Vec<f64>, group2: Vec<f64>) -> PyResult<f64> {
    scivex_stats::cohens_d(&group1, &group2).map_err(stats_err)
}

/// Hedges' g — bias-corrected Cohen's d.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn hedges_g(group1: Vec<f64>, group2: Vec<f64>) -> PyResult<f64> {
    scivex_stats::hedges_g(&group1, &group2).map_err(stats_err)
}

/// Glass's delta — mean difference standardised by the control group std.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn glass_delta(treatment: Vec<f64>, control: Vec<f64>) -> PyResult<f64> {
    scivex_stats::glass_delta(&treatment, &control).map_err(stats_err)
}

/// Eta-squared effect size for one-way ANOVA.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn eta_squared(groups: Vec<Vec<f64>>) -> PyResult<f64> {
    let refs: Vec<&[f64]> = groups.iter().map(|g| g.as_slice()).collect();
    scivex_stats::eta_squared(&refs).map_err(stats_err)
}

/// Omega-squared effect size for one-way ANOVA.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn omega_squared(groups: Vec<Vec<f64>>) -> PyResult<f64> {
    let refs: Vec<&[f64]> = groups.iter().map(|g| g.as_slice()).collect();
    scivex_stats::omega_squared(&refs).map_err(stats_err)
}

// ---------------------------------------------------------------------------
// Multiple comparison corrections
// ---------------------------------------------------------------------------

/// Bonferroni correction for multiple p-values.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn bonferroni(p_values: Vec<f64>) -> PyResult<Vec<f64>> {
    scivex_stats::bonferroni(&p_values).map_err(stats_err)
}

/// Benjamini-Hochberg FDR correction for multiple p-values.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn benjamini_hochberg(p_values: Vec<f64>) -> PyResult<Vec<f64>> {
    scivex_stats::benjamini_hochberg(&p_values).map_err(stats_err)
}

// ---------------------------------------------------------------------------
// Time series (P4.10)
// ---------------------------------------------------------------------------

/// Autocorrelation function (ACF) for lags 0..max_lag.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn acf(data: Vec<f64>, max_lag: usize) -> PyResult<Vec<f64>> {
    scivex_stats::acf(&data, max_lag).map_err(stats_err)
}

/// Partial autocorrelation function (PACF) for lags 0..max_lag.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn pacf(data: Vec<f64>, max_lag: usize) -> PyResult<Vec<f64>> {
    scivex_stats::pacf(&data, max_lag).map_err(stats_err)
}

/// Augmented Dickey-Fuller test for stationarity.
///
/// Returns a dict with keys: `statistic`, `n_lags`, `n_obs`, `reject_null`.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
#[pyo3(signature = (data, max_lags = None))]
fn adf_test(py: Python<'_>, data: Vec<f64>, max_lags: Option<usize>) -> PyResult<PyObject> {
    let r = scivex_stats::adf_test(&data, max_lags).map_err(stats_err)?;
    let d = PyDict::new(py);
    d.set_item("statistic", r.statistic)?;
    d.set_item("n_lags", r.n_lags)?;
    d.set_item("n_obs", r.n_obs)?;
    d.set_item("reject_null", r.reject_null)?;
    Ok(d.into_any().unbind())
}

/// Additive seasonal decomposition.
///
/// Returns a dict with keys: `trend`, `seasonal`, `residual`.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn seasonal_decompose(py: Python<'_>, data: Vec<f64>, period: usize) -> PyResult<PyObject> {
    let r = scivex_stats::seasonal_decompose(&data, period).map_err(stats_err)?;
    let d = PyDict::new(py);
    d.set_item("trend", r.trend)?;
    d.set_item("seasonal", r.seasonal)?;
    d.set_item("residual", r.residual)?;
    Ok(d.into_any().unbind())
}

// ---------------------------------------------------------------------------
// ARIMA (P4.11)
// ---------------------------------------------------------------------------

/// ARIMA(p, d, q) time series model.
#[pyclass(name = "ARIMA")]
struct PyARIMA {
    inner: scivex_stats::Arima<f64>,
}

#[pymethods]
impl PyARIMA {
    /// Create a new ARIMA(p, d, q) model.
    #[new]
    fn new(p: usize, d: usize, q: usize) -> PyResult<Self> {
        let inner = scivex_stats::Arima::new(p, d, q).map_err(stats_err)?;
        Ok(Self { inner })
    }

    /// Fit the model to data.
    #[allow(clippy::needless_pass_by_value)]
    fn fit(&mut self, data: Vec<f64>) -> PyResult<()> {
        self.inner.fit(&data).map_err(stats_err)
    }

    /// Forecast `steps` steps ahead.
    fn predict(&self, steps: usize) -> PyResult<Vec<f64>> {
        self.inner.forecast(steps).map_err(stats_err)
    }
}

// ---------------------------------------------------------------------------
// Prophet (P4.12)
// ---------------------------------------------------------------------------

/// Prophet-style additive decomposable time series forecasting model.
#[pyclass(name = "Prophet")]
struct PyProphet {
    inner: scivex_stats::Prophet<f64>,
}

#[pymethods]
impl PyProphet {
    /// Create a new Prophet model with optional configuration.
    #[new]
    #[pyo3(signature = (
        yearly_seasonality = true,
        weekly_seasonality = true,
        n_changepoints = 25,
        changepoint_prior_scale = 0.05,
        seasonality_prior_scale = 10.0,
    ))]
    fn new(
        yearly_seasonality: bool,
        weekly_seasonality: bool,
        n_changepoints: usize,
        changepoint_prior_scale: f64,
        seasonality_prior_scale: f64,
    ) -> Self {
        let mut config = scivex_stats::ProphetConfig::<f64>::new();
        config.yearly_seasonality = yearly_seasonality;
        config.weekly_seasonality = weekly_seasonality;
        config.n_changepoints = n_changepoints;
        config.changepoint_prior_scale = changepoint_prior_scale;
        config.seasonality_prior_scale = seasonality_prior_scale;
        Self {
            inner: scivex_stats::Prophet::new(config),
        }
    }

    /// Fit the model to time series data.
    #[allow(clippy::needless_pass_by_value)]
    fn fit(&mut self, t: Vec<f64>, y: Vec<f64>) -> PyResult<()> {
        self.inner.fit(&t, &y).map_err(stats_err)
    }

    /// Predict at given time points.
    ///
    /// Returns a dict with keys: `yhat`, `trend`, `seasonality`.
    #[allow(clippy::needless_pass_by_value)]
    fn predict(&self, py: Python<'_>, t_future: Vec<f64>) -> PyResult<PyObject> {
        let r = self.inner.predict(&t_future).map_err(stats_err)?;
        let d = PyDict::new(py);
        d.set_item("yhat", r.yhat)?;
        d.set_item("trend", r.trend)?;
        d.set_item("seasonality", r.seasonality)?;
        Ok(d.into_any().unbind())
    }
}

// ---------------------------------------------------------------------------
// Survival analysis (P4.13)
// ---------------------------------------------------------------------------

/// Kaplan-Meier survival estimate.
///
/// Returns a dict with keys: `times`, `survival_prob`, `at_risk`, `events`,
/// `ci_lower`, `ci_upper`.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn kaplan_meier(py: Python<'_>, times: Vec<f64>, events: Vec<bool>) -> PyResult<PyObject> {
    if times.len() != events.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "times and events must have the same length",
        ));
    }
    let records: Vec<scivex_stats::SurvivalRecord<f64>> = times
        .into_iter()
        .zip(events)
        .map(|(time, event)| scivex_stats::SurvivalRecord { time, event })
        .collect();
    let r = scivex_stats::kaplan_meier(&records).map_err(stats_err)?;
    let d = PyDict::new(py);
    d.set_item("times", r.times)?;
    d.set_item("survival_prob", r.survival_prob)?;
    d.set_item("at_risk", r.at_risk)?;
    d.set_item("events", r.events)?;
    d.set_item("ci_lower", r.ci_lower)?;
    d.set_item("ci_upper", r.ci_upper)?;
    Ok(d.into_any().unbind())
}

/// Two-sample log-rank test comparing survival distributions.
///
/// Returns a dict with keys: `statistic`, `p_value`.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn log_rank_test(
    py: Python<'_>,
    times1: Vec<f64>,
    events1: Vec<bool>,
    times2: Vec<f64>,
    events2: Vec<bool>,
) -> PyResult<PyObject> {
    if times1.len() != events1.len() || times2.len() != events2.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "times and events must have the same length within each group",
        ));
    }
    let g1: Vec<scivex_stats::SurvivalRecord<f64>> = times1
        .into_iter()
        .zip(events1)
        .map(|(time, event)| scivex_stats::SurvivalRecord { time, event })
        .collect();
    let g2: Vec<scivex_stats::SurvivalRecord<f64>> = times2
        .into_iter()
        .zip(events2)
        .map(|(time, event)| scivex_stats::SurvivalRecord { time, event })
        .collect();
    let r = scivex_stats::log_rank_test(&g1, &g2).map_err(stats_err)?;
    let d = PyDict::new(py);
    d.set_item("statistic", r.statistic)?;
    d.set_item("p_value", r.p_value)?;
    Ok(d.into_any().unbind())
}

/// Cox proportional-hazards regression.
///
/// Returns a dict with keys: `coefficients`, `std_errors`, `hazard_ratios`,
/// `p_values`, `log_likelihood`, `concordance`.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn cox_ph(
    py: Python<'_>,
    times: Vec<f64>,
    events: Vec<bool>,
    covariates: &PyTensor,
) -> PyResult<PyObject> {
    let r = scivex_stats::cox_ph(&times, &events, covariates.as_f64()?).map_err(stats_err)?;
    let d = PyDict::new(py);
    d.set_item("coefficients", r.coefficients)?;
    d.set_item("std_errors", r.std_errors)?;
    d.set_item("hazard_ratios", r.hazard_ratios)?;
    d.set_item("p_values", r.p_values)?;
    d.set_item("log_likelihood", r.log_likelihood)?;
    d.set_item("concordance", r.concordance)?;
    Ok(d.into_any().unbind())
}

// ---------------------------------------------------------------------------
// Bayesian MCMC (P4.14)
// ---------------------------------------------------------------------------

/// Metropolis-Hastings MCMC sampler.
///
/// Since Python closures cannot be passed directly as Rust `Fn` trait objects,
/// this class wraps the sampler around a simple multivariate Gaussian target
/// with given mean and precision (inverse covariance diagonal).
#[pyclass(name = "MetropolisHastings")]
struct PyMetropolisHastings {
    proposal_scale: Vec<f64>,
}

#[pymethods]
impl PyMetropolisHastings {
    /// Create a new MH sampler with proposal standard deviations per parameter.
    #[new]
    #[allow(clippy::needless_pass_by_value)]
    fn new(proposal_scale: Vec<f64>) -> Self {
        Self { proposal_scale }
    }

    /// Sample from a multivariate Gaussian target with diagonal precision.
    ///
    /// `target_mean` and `target_precision` define N(mean, diag(1/precision)).
    /// Returns a dict with keys: `samples`, `acceptance_rate`, `log_probs`.
    #[allow(clippy::needless_pass_by_value, clippy::too_many_arguments)]
    fn sample_gaussian(
        &self,
        py: Python<'_>,
        target_mean: Vec<f64>,
        target_precision: Vec<f64>,
        initial: Vec<f64>,
        n_samples: usize,
        n_warmup: usize,
        seed: u64,
        thin: usize,
    ) -> PyResult<PyObject> {
        let mh = scivex_stats::bayesian::MetropolisHastings::new(self.proposal_scale.clone());
        let config =
            scivex_stats::bayesian::McmcConfig::<f64>::new(n_samples, n_warmup, seed, thin);
        let tm = target_mean;
        let tp = target_precision;
        let log_prob = move |x: &[f64]| -> f64 {
            let mut lp = 0.0;
            for i in 0..x.len() {
                let d = x[i] - tm[i];
                lp -= 0.5 * tp[i] * d * d;
            }
            lp
        };
        let result = mh.sample(log_prob, &initial, &config).map_err(stats_err)?;
        mcmc_result_to_dict(py, &result)
    }
}

/// Hamiltonian Monte Carlo (HMC) sampler.
///
/// Uses leapfrog integration to generate proposals informed by gradient
/// information, yielding much better acceptance rates than random-walk MH.
/// Since Python closures cannot be passed directly as Rust `Fn` trait objects,
/// this class provides a `sample_gaussian` convenience method that samples
/// from a multivariate Gaussian target with diagonal precision.
#[pyclass(name = "HMC")]
struct PyHMC {
    step_size: f64,
    n_leapfrog: usize,
}

#[pymethods]
impl PyHMC {
    /// Create a new HMC sampler.
    ///
    /// * `step_size` — leapfrog integration step size.
    /// * `n_leapfrog` — number of leapfrog steps per proposal.
    #[new]
    fn new(step_size: f64, n_leapfrog: usize) -> Self {
        Self {
            step_size,
            n_leapfrog,
        }
    }

    /// Sample from a multivariate Gaussian target with diagonal precision.
    ///
    /// `target_mean` and `target_precision` define N(mean, diag(1/precision)).
    /// Returns a dict with keys: `samples`, `acceptance_rate`, `log_probs`.
    #[allow(clippy::needless_pass_by_value, clippy::too_many_arguments)]
    fn sample_gaussian(
        &self,
        py: Python<'_>,
        target_mean: Vec<f64>,
        target_precision: Vec<f64>,
        initial: Vec<f64>,
        n_samples: usize,
        n_warmup: usize,
        seed: u64,
        thin: usize,
    ) -> PyResult<PyObject> {
        let hmc = scivex_stats::bayesian::HamiltonianMC::new(self.step_size, self.n_leapfrog);
        let config =
            scivex_stats::bayesian::McmcConfig::<f64>::new(n_samples, n_warmup, seed, thin);
        let tm = target_mean.clone();
        let tp = target_precision.clone();
        let log_prob = move |x: &[f64]| -> f64 {
            let mut lp = 0.0;
            for i in 0..x.len() {
                let d = x[i] - tm[i];
                lp -= 0.5 * tp[i] * d * d;
            }
            lp
        };
        let tm2 = target_mean;
        let tp2 = target_precision;
        let grad = move |x: &[f64]| -> Vec<f64> {
            x.iter()
                .enumerate()
                .map(|(i, &xi)| -tp2[i] * (xi - tm2[i]))
                .collect()
        };
        let result = hmc
            .sample(log_prob, grad, &initial, &config)
            .map_err(stats_err)?;
        mcmc_result_to_dict(py, &result)
    }
}

/// No-U-Turn Sampler (NUTS).
///
/// An extension of HMC that automatically tunes the trajectory length using
/// recursive tree doubling and adapts the step size via dual averaging during
/// warmup. Since Python closures cannot be passed directly as Rust `Fn` trait
/// objects, this class provides a `sample_gaussian` convenience method that
/// samples from a multivariate Gaussian target with diagonal precision.
#[pyclass(name = "NUTS")]
struct PyNUTS {
    initial_step_size: f64,
    max_tree_depth: usize,
    target_accept: f64,
}

#[pymethods]
impl PyNUTS {
    /// Create a new NUTS sampler.
    ///
    /// * `max_tree_depth` — maximum number of tree doublings (default 10).
    /// * `initial_step_size` — starting leapfrog step size (adapted during warmup, default 0.1).
    /// * `target_accept` — target acceptance probability for dual averaging (default 0.8).
    #[new]
    #[pyo3(signature = (max_tree_depth = 10, initial_step_size = 0.1, target_accept = 0.8))]
    fn new(max_tree_depth: usize, initial_step_size: f64, target_accept: f64) -> Self {
        Self {
            initial_step_size,
            max_tree_depth,
            target_accept,
        }
    }

    /// Sample from a multivariate Gaussian target with diagonal precision.
    ///
    /// `target_mean` and `target_precision` define N(mean, diag(1/precision)).
    /// Returns a dict with keys: `samples`, `acceptance_rate`, `log_probs`.
    #[allow(clippy::needless_pass_by_value, clippy::too_many_arguments)]
    fn sample_gaussian(
        &self,
        py: Python<'_>,
        target_mean: Vec<f64>,
        target_precision: Vec<f64>,
        initial: Vec<f64>,
        n_samples: usize,
        n_warmup: usize,
        seed: u64,
        thin: usize,
    ) -> PyResult<PyObject> {
        let nuts = scivex_stats::bayesian::Nuts::new(
            self.initial_step_size,
            self.max_tree_depth,
            self.target_accept,
        );
        let config =
            scivex_stats::bayesian::McmcConfig::<f64>::new(n_samples, n_warmup, seed, thin);
        let tm = target_mean.clone();
        let tp = target_precision.clone();
        let log_prob = move |x: &[f64]| -> f64 {
            let mut lp = 0.0;
            for i in 0..x.len() {
                let d = x[i] - tm[i];
                lp -= 0.5 * tp[i] * d * d;
            }
            lp
        };
        let tm2 = target_mean;
        let tp2 = target_precision;
        let grad = move |x: &[f64]| -> Vec<f64> {
            x.iter()
                .enumerate()
                .map(|(i, &xi)| -tp2[i] * (xi - tm2[i]))
                .collect()
        };
        let result = nuts
            .sample(log_prob, grad, &initial, &config)
            .map_err(stats_err)?;
        mcmc_result_to_dict(py, &result)
    }
}

/// Convert an McmcResult to a Python dict.
fn mcmc_result_to_dict(
    py: Python<'_>,
    r: &scivex_stats::bayesian::McmcResult<f64>,
) -> PyResult<PyObject> {
    let d = PyDict::new(py);
    d.set_item("samples", r.samples.clone())?;
    d.set_item("acceptance_rate", r.acceptance_rate.clone())?;
    d.set_item("log_probs", r.log_probs.clone())?;
    Ok(d.into_any().unbind())
}

/// Compute MCMC trace summary from samples.
///
/// `samples` has shape `[n_chains][n_samples][n_params]`.
/// Returns a dict with keys: `mean`, `std`, `q025`, `q25`, `q50`, `q75`,
/// `q975`, `rhat`, `n_eff`.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn trace_summary(
    py: Python<'_>,
    samples: Vec<Vec<Vec<f64>>>,
    acceptance_rate: Vec<f64>,
    log_probs: Vec<Vec<f64>>,
) -> PyResult<PyObject> {
    let result = scivex_stats::bayesian::McmcResult {
        samples,
        acceptance_rate,
        log_probs,
    };
    let ts = scivex_stats::bayesian::trace_summary(&result);
    let d = PyDict::new(py);
    d.set_item("mean", ts.mean)?;
    d.set_item("std", ts.std)?;
    d.set_item("q025", ts.q025)?;
    d.set_item("q25", ts.q25)?;
    d.set_item("q50", ts.q50)?;
    d.set_item("q75", ts.q75)?;
    d.set_item("q975", ts.q975)?;
    d.set_item("rhat", ts.rhat)?;
    d.set_item("n_eff", ts.n_eff)?;
    Ok(d.into_any().unbind())
}

/// Compute R-hat convergence diagnostic from MCMC chains.
///
/// `chains` has shape `[n_chains][n_samples][n_params]`.
/// Returns one R-hat value per parameter.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn rhat(chains: Vec<Vec<Vec<f64>>>) -> Vec<f64> {
    scivex_stats::bayesian::rhat(&chains)
}

/// Compute effective sample size from MCMC chains.
///
/// `chains` has shape `[n_chains][n_samples][n_params]`.
/// Returns one ESS value per parameter.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn effective_sample_size(chains: Vec<Vec<Vec<f64>>>) -> Vec<f64> {
    scivex_stats::bayesian::effective_sample_size(&chains)
}

// ---------------------------------------------------------------------------
// Bayesian optimization (P4.15)
// ---------------------------------------------------------------------------

/// Bayesian optimizer using a Gaussian process surrogate.
///
/// Since Python closures cannot be passed as Rust `Fn`, this class provides
/// `minimize_quadratic` and `maximize_quadratic` convenience methods that
/// optimize a weighted sum-of-squares objective: f(x) = sum_i w_i (x_i - c_i)^2.
#[pyclass(name = "BayesianOptimizer")]
struct PyBayesianOptimizer {
    bounds: Vec<(f64, f64)>,
    n_initial: usize,
    n_iterations: usize,
    seed: u64,
    noise_variance: f64,
}

#[pymethods]
impl PyBayesianOptimizer {
    /// Create a new Bayesian optimizer.
    ///
    /// `bounds` is a list of `(lower, upper)` tuples, one per dimension.
    #[new]
    #[pyo3(signature = (bounds, n_initial = 5, n_iterations = 50, seed = 42, noise_variance = 1e-6))]
    #[allow(clippy::needless_pass_by_value)]
    fn new(
        bounds: Vec<(f64, f64)>,
        n_initial: usize,
        n_iterations: usize,
        seed: u64,
        noise_variance: f64,
    ) -> Self {
        Self {
            bounds,
            n_initial,
            n_iterations,
            seed,
            noise_variance,
        }
    }

    /// Minimize a weighted quadratic objective: f(x) = sum_i w_i * (x_i - c_i)^2.
    ///
    /// Returns a dict with keys: `best_x`, `best_y`, `x_history`, `y_history`, `iterations`.
    #[allow(clippy::needless_pass_by_value)]
    fn minimize_quadratic(
        &self,
        py: Python<'_>,
        centers: Vec<f64>,
        weights: Vec<f64>,
    ) -> PyResult<PyObject> {
        let config = scivex_stats::BayesOptConfig {
            n_initial: self.n_initial,
            n_iterations: self.n_iterations,
            seed: self.seed,
            noise_variance: self.noise_variance,
            ..Default::default()
        };
        let mut opt = scivex_stats::BayesianOptimizer::new(self.bounds.clone(), config);
        let c = centers;
        let w = weights;
        let result = opt
            .minimize(move |x: &[f64]| {
                let mut val = 0.0;
                for i in 0..x.len() {
                    val += w[i] * (x[i] - c[i]) * (x[i] - c[i]);
                }
                val
            })
            .map_err(stats_err)?;
        bayesopt_result_to_dict(py, &result)
    }

    /// Maximize a negative weighted quadratic: f(x) = -sum_i w_i * (x_i - c_i)^2.
    ///
    /// Returns a dict with keys: `best_x`, `best_y`, `x_history`, `y_history`, `iterations`.
    #[allow(clippy::needless_pass_by_value)]
    fn maximize_quadratic(
        &self,
        py: Python<'_>,
        centers: Vec<f64>,
        weights: Vec<f64>,
    ) -> PyResult<PyObject> {
        let config = scivex_stats::BayesOptConfig {
            n_initial: self.n_initial,
            n_iterations: self.n_iterations,
            seed: self.seed,
            noise_variance: self.noise_variance,
            ..Default::default()
        };
        let mut opt = scivex_stats::BayesianOptimizer::new(self.bounds.clone(), config);
        let c = centers;
        let w = weights;
        let result = opt
            .maximize(move |x: &[f64]| {
                let mut val = 0.0;
                for i in 0..x.len() {
                    val -= w[i] * (x[i] - c[i]) * (x[i] - c[i]);
                }
                val
            })
            .map_err(stats_err)?;
        bayesopt_result_to_dict(py, &result)
    }
}

/// Convert a `BayesOptResult` to a Python dict.
fn bayesopt_result_to_dict(
    py: Python<'_>,
    r: &scivex_stats::BayesOptResult<f64>,
) -> PyResult<PyObject> {
    let d = PyDict::new(py);
    d.set_item("best_x", r.best_x.clone())?;
    d.set_item("best_y", r.best_y)?;
    d.set_item("x_history", r.x_history.clone())?;
    d.set_item("y_history", r.y_history.clone())?;
    d.set_item("iterations", r.iterations)?;
    Ok(d.into_any().unbind())
}

// ---------------------------------------------------------------------------
// Anomaly detection (P4.16)
// ---------------------------------------------------------------------------

/// Detect anomalies using sliding-window z-score.
///
/// Returns a dict with keys: `anomaly_indices`, `scores`, `threshold`.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn zscore_anomaly(
    py: Python<'_>,
    data: Vec<f64>,
    window_size: usize,
    threshold: f64,
) -> PyResult<PyObject> {
    let r = scivex_stats::zscore_anomaly(&data, window_size, threshold).map_err(stats_err)?;
    anomaly_result_to_dict(py, &r)
}

/// Detect anomalies using seasonal decomposition residuals.
///
/// Returns a dict with keys: `anomaly_indices`, `scores`, `threshold`.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn seasonal_anomaly(
    py: Python<'_>,
    data: Vec<f64>,
    period: usize,
    n_sigma: f64,
) -> PyResult<PyObject> {
    let r = scivex_stats::seasonal_anomaly(&data, period, n_sigma).map_err(stats_err)?;
    anomaly_result_to_dict(py, &r)
}

/// Detect anomalies using Isolation Forest adapted for time series.
///
/// Returns a dict with keys: `anomaly_indices`, `scores`, `threshold`.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn isolation_forest_anomaly(
    py: Python<'_>,
    data: Vec<f64>,
    window_size: usize,
    n_trees: usize,
    contamination: f64,
    seed: u64,
) -> PyResult<PyObject> {
    let r =
        scivex_stats::isolation_forest_anomaly(&data, window_size, n_trees, contamination, seed)
            .map_err(stats_err)?;
    anomaly_result_to_dict(py, &r)
}

/// Detect anomalies using exponentially weighted moving average (EWMA).
///
/// Returns a dict with keys: `anomaly_indices`, `scores`, `threshold`.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn ewma_anomaly(py: Python<'_>, data: Vec<f64>, alpha: f64, n_sigma: f64) -> PyResult<PyObject> {
    let r = scivex_stats::ewma_anomaly(&data, alpha, n_sigma).map_err(stats_err)?;
    anomaly_result_to_dict(py, &r)
}

/// Convert an `AnomalyResult` to a Python dict.
fn anomaly_result_to_dict(
    py: Python<'_>,
    r: &scivex_stats::AnomalyResult<f64>,
) -> PyResult<PyObject> {
    let d = PyDict::new(py);
    d.set_item("anomaly_indices", r.anomaly_indices.clone())?;
    d.set_item("scores", r.scores.clone())?;
    d.set_item("threshold", r.threshold)?;
    Ok(d.into_any().unbind())
}

// ---------------------------------------------------------------------------
// Kalman filter (P4.17)
// ---------------------------------------------------------------------------

/// Kalman filter for linear state estimation.
#[pyclass(name = "KalmanFilter")]
struct PyKalmanFilter {
    inner: scivex_stats::KalmanFilter<f64>,
}

#[pymethods]
impl PyKalmanFilter {
    /// Create a new Kalman filter with given state dimension, initial state,
    /// and initial covariance (flat row-major array of size state_dim^2).
    #[new]
    #[allow(clippy::needless_pass_by_value)]
    fn new(state_dim: usize, initial_state: Vec<f64>, initial_cov: Vec<f64>) -> PyResult<Self> {
        let cov = scivex_core::Tensor::from_vec(initial_cov, vec![state_dim, state_dim])
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let inner =
            scivex_stats::KalmanFilter::new(state_dim, &initial_state, &cov).map_err(stats_err)?;
        Ok(Self { inner })
    }

    /// Predict step: propagate state forward using transition matrix F and
    /// process noise Q (both flat row-major arrays of size state_dim^2).
    #[allow(clippy::needless_pass_by_value)]
    fn predict(&mut self, transition: Vec<f64>, process_noise: Vec<f64>) -> PyResult<Vec<f64>> {
        let dim = self.inner.state().len();
        let f = scivex_core::Tensor::from_vec(transition, vec![dim, dim])
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let q = scivex_core::Tensor::from_vec(process_noise, vec![dim, dim])
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.inner.predict(&f, &q).map_err(stats_err)?;
        Ok(self.inner.state().to_vec())
    }

    /// Update step: incorporate an observation using observation matrix H
    /// (flat row-major [obs_dim x state_dim]) and noise covariance R
    /// (flat row-major [obs_dim x obs_dim]).
    #[allow(clippy::needless_pass_by_value)]
    fn update(
        &mut self,
        observation: Vec<f64>,
        obs_matrix: Vec<f64>,
        obs_noise: Vec<f64>,
    ) -> PyResult<Vec<f64>> {
        let obs_dim = observation.len();
        let state_dim = self.inner.state().len();
        let h = scivex_core::Tensor::from_vec(obs_matrix, vec![obs_dim, state_dim])
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let r = scivex_core::Tensor::from_vec(obs_noise, vec![obs_dim, obs_dim])
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.inner.update(&observation, &h, &r).map_err(stats_err)?;
        Ok(self.inner.state().to_vec())
    }

    /// Return the current state estimate.
    fn state(&self) -> Vec<f64> {
        self.inner.state().to_vec()
    }
}

// ---------------------------------------------------------------------------
// GARCH (P4.18)
// ---------------------------------------------------------------------------

/// GARCH(p,q) conditional volatility model.
#[pyclass(name = "GARCH")]
struct PyGarch {
    inner: scivex_stats::Garch<f64>,
}

#[pymethods]
impl PyGarch {
    /// Create a new GARCH(p, q) model.
    #[new]
    fn new(p: usize, q: usize) -> PyResult<Self> {
        let inner = scivex_stats::Garch::new(p, q).map_err(stats_err)?;
        Ok(Self { inner })
    }

    /// Fit the model to a series of returns.
    #[allow(clippy::needless_pass_by_value)]
    fn fit(&mut self, data: Vec<f64>) -> PyResult<()> {
        self.inner.fit(&data).map_err(stats_err)
    }

    /// Forecast conditional variance for `steps` periods ahead.
    fn forecast(&self, steps: usize) -> PyResult<Vec<f64>> {
        self.inner.forecast_variance(steps).map_err(stats_err)
    }
}

// ---------------------------------------------------------------------------
// VAR model (P4.19)
// ---------------------------------------------------------------------------

/// Vector Autoregression (VAR) model for multivariate time series.
#[pyclass(name = "VAR")]
struct PyVarModel {
    inner: scivex_stats::VarModel<f64>,
    /// Cached data for Granger causality tests after fitting.
    cached_data: Option<Vec<Vec<f64>>>,
}

#[pymethods]
impl PyVarModel {
    /// Create a new VAR model of order `p`.
    #[new]
    fn new(order: usize) -> PyResult<Self> {
        let inner = scivex_stats::VarModel::new(order).map_err(stats_err)?;
        Ok(Self {
            inner,
            cached_data: None,
        })
    }

    /// Fit the model to multivariate data (list of series, each of equal length).
    #[allow(clippy::needless_pass_by_value)]
    fn fit(&mut self, data: Vec<Vec<f64>>) -> PyResult<()> {
        self.inner.fit(&data).map_err(stats_err)?;
        self.cached_data = Some(data);
        Ok(())
    }

    /// Forecast `steps` steps ahead. Returns a list of series.
    fn forecast(&self, steps: usize) -> PyResult<Vec<Vec<f64>>> {
        self.inner.forecast(steps).map_err(stats_err)
    }

    /// Granger causality test: does variable `cause` Granger-cause `effect`?
    ///
    /// Returns a dict with keys: `f_statistic`, `p_value`, `significant`.
    fn granger_test(&self, py: Python<'_>, cause: usize, effect: usize) -> PyResult<PyObject> {
        let data = self.cached_data.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("model must be fitted before Granger test")
        })?;
        let r = self
            .inner
            .granger_causality(data, cause, effect)
            .map_err(stats_err)?;
        let d = PyDict::new(py);
        d.set_item("f_statistic", r.f_statistic)?;
        d.set_item("p_value", r.p_value)?;
        d.set_item("significant", r.significant)?;
        Ok(d.into_any().unbind())
    }
}

// ---------------------------------------------------------------------------
// Exponential Smoothing (P4.20)
// ---------------------------------------------------------------------------

/// Exponential smoothing models: simple, Holt (double), or Holt-Winters (triple).
#[pyclass(name = "ExponentialSmoothing")]
struct PyExponentialSmoothing {
    inner: scivex_stats::ExponentialSmoothing<f64>,
}

#[pymethods]
impl PyExponentialSmoothing {
    /// Create a new ExponentialSmoothing model.
    ///
    /// `method` is one of `"simple"` / `"ses"`, `"double"` / `"holt"`,
    /// `"triple"` / `"holt-winters"`.
    ///
    /// Optional smoothing parameters: `alpha`, `beta`, `gamma`, `season_length`.
    #[new]
    #[pyo3(signature = (method, alpha = 0.3, beta = 0.1, gamma = 0.1, season_length = 12))]
    fn new(
        method: &str,
        alpha: f64,
        beta: f64,
        gamma: f64,
        season_length: usize,
    ) -> PyResult<Self> {
        let inner = match method {
            "simple" | "ses" => {
                scivex_stats::ExponentialSmoothing::simple(alpha).map_err(stats_err)?
            }
            "double" | "holt" => {
                scivex_stats::ExponentialSmoothing::holt(alpha, beta).map_err(stats_err)?
            }
            "triple" | "holt-winters" => {
                scivex_stats::ExponentialSmoothing::holt_winters(alpha, beta, gamma, season_length)
                    .map_err(stats_err)?
            }
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown method: {method}; expected \"simple\", \"ses\", \"double\", \
                     \"holt\", \"triple\", or \"holt-winters\""
                )));
            }
        };
        Ok(Self { inner })
    }

    /// Fit the model to data.
    #[allow(clippy::needless_pass_by_value)]
    fn fit(&mut self, data: Vec<f64>) -> PyResult<()> {
        self.inner.fit(&data).map_err(stats_err)
    }

    /// Forecast `steps` steps ahead.
    fn forecast(&self, steps: usize) -> PyResult<Vec<f64>> {
        self.inner.forecast(steps).map_err(stats_err)
    }
}

// ---------------------------------------------------------------------------
// SARIMAX (P4.21)
// ---------------------------------------------------------------------------

/// SARIMAX model: Seasonal ARIMA with optional exogenous variables.
#[pyclass(name = "SARIMAX")]
struct PySarimax {
    inner: scivex_stats::Sarimax<f64>,
}

#[pymethods]
impl PySarimax {
    /// Create a new SARIMAX model.
    ///
    /// `order_p`, `order_d`, `order_q` are the regular ARIMA orders.
    /// `seasonal_p`, `seasonal_d`, `seasonal_q`, `seasonal_period` are the
    /// seasonal orders and period.
    #[new]
    fn new(
        order_p: usize,
        order_d: usize,
        order_q: usize,
        seasonal_p: usize,
        seasonal_d: usize,
        seasonal_q: usize,
        seasonal_period: usize,
    ) -> PyResult<Self> {
        let inner = scivex_stats::Sarimax::new(
            order_p,
            order_d,
            order_q,
            (seasonal_p, seasonal_d, seasonal_q, seasonal_period),
        )
        .map_err(stats_err)?;
        Ok(Self { inner })
    }

    /// Fit the model to data (no exogenous variables).
    #[allow(clippy::needless_pass_by_value)]
    fn fit(&mut self, data: Vec<f64>) -> PyResult<()> {
        self.inner.fit(&data, None).map_err(stats_err)
    }

    /// Forecast `steps` steps ahead (no exogenous variables).
    fn forecast(&self, steps: usize) -> PyResult<Vec<f64>> {
        self.inner.forecast(steps, None).map_err(stats_err)
    }
}

// ---------------------------------------------------------------------------
// Mixed-effects model function (P4.22)
// ---------------------------------------------------------------------------

/// Linear mixed-effects model (LMM).
///
/// Returns a dict with keys: `fixed_effects`, `random_effects`,
/// `residual_variance`, `random_effect_variance`, `log_likelihood`,
/// `iterations`, `converged`.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn lmm(py: Python<'_>, y: Vec<f64>, x: Vec<Vec<f64>>, groups: Vec<usize>) -> PyResult<PyObject> {
    let n = y.len();
    let ncols = if x.is_empty() { 0 } else { x.len() };
    // Build a Tensor from column-major input (each inner Vec is a column)
    let mut flat = Vec::with_capacity(n * ncols);
    for row in 0..n {
        for col in &x {
            flat.push(col[row]);
        }
    }
    let x_tensor = scivex_core::Tensor::from_vec(flat, vec![n, ncols])
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let r = scivex_stats::lmm(&y, &x_tensor, &groups, None, 100, 1e-6).map_err(stats_err)?;
    let d = PyDict::new(py);
    d.set_item("fixed_effects", r.fixed_effects)?;
    d.set_item("random_effects", r.random_effects)?;
    d.set_item("residual_variance", r.residual_variance)?;
    d.set_item("random_effect_variance", r.random_effect_variance)?;
    d.set_item("log_likelihood", r.log_likelihood)?;
    d.set_item("iterations", r.iterations)?;
    d.set_item("converged", r.converged)?;
    Ok(d.into_any().unbind())
}

// ---------------------------------------------------------------------------
// Time series feature extraction (P4.23)
// ---------------------------------------------------------------------------

/// Extract default time series features using rolling windows.
///
/// Returns a dict with keys: `feature_names`, `features`.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn extract_features(py: Python<'_>, data: Vec<f64>, window_size: usize) -> PyResult<PyObject> {
    let r = scivex_stats::extract_default_features(&data, window_size).map_err(stats_err)?;
    let d = PyDict::new(py);
    d.set_item("feature_names", r.feature_names)?;
    d.set_item("features", r.features)?;
    Ok(d.into_any().unbind())
}

// ---------------------------------------------------------------------------
// Additional effect sizes (P4.24)
// ---------------------------------------------------------------------------

/// Cramer's V — association measure for contingency tables.
///
/// `observed` is a list of rows (each row is a list of counts).
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn cramers_v(observed: Vec<Vec<f64>>) -> PyResult<f64> {
    let rows = observed.len();
    let cols = if rows > 0 { observed[0].len() } else { 0 };
    scivex_stats::cramers_v(&observed, rows, cols).map_err(stats_err)
}

/// Cohen's w — effect size for chi-square goodness-of-fit.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn cohens_w(observed: Vec<f64>, expected: Vec<f64>) -> PyResult<f64> {
    scivex_stats::cohens_w(&observed, &expected).map_err(stats_err)
}

/// Point-biserial correlation between a binary variable and a continuous one.
///
/// `binary` is a list of booleans, `continuous` is a list of floats.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn point_biserial(binary: Vec<bool>, continuous: Vec<f64>) -> PyResult<f64> {
    scivex_stats::point_biserial(&binary, &continuous).map_err(stats_err)
}

// =========================================================================
// Submodule registration
// =========================================================================

/// Register the `stats` submodule on `parent`.
pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let m = PyModule::new(py, "stats")?;

    // -- Descriptive statistics --
    m.add_function(wrap_pyfunction!(quantile, &m)?)?;
    m.add_function(wrap_pyfunction!(skewness, &m)?)?;
    m.add_function(wrap_pyfunction!(kurtosis, &m)?)?;
    m.add_function(wrap_pyfunction!(variance_with_ddof, &m)?)?;
    m.add_function(wrap_pyfunction!(std_dev_with_ddof, &m)?)?;
    m.add_function(wrap_pyfunction!(describe, &m)?)?;

    // -- Distribution classes --
    m.add_class::<PyNormal>()?;
    m.add_class::<PyUniform>()?;
    m.add_class::<PyExponential>()?;
    m.add_class::<PyStudentT>()?;
    m.add_class::<PyBeta>()?;
    m.add_class::<PyGamma>()?;
    m.add_class::<PyPoisson>()?;
    m.add_class::<PyBinomial>()?;
    m.add_class::<PyChiSquared>()?;
    m.add_class::<PyBernoulli>()?;
    m.add_class::<PyCauchy>()?;
    m.add_class::<PyLogNormal>()?;
    m.add_class::<PyWeibull>()?;
    m.add_class::<PyLaplace>()?;
    m.add_class::<PyHypergeometric>()?;
    m.add_class::<PyNegativeBinomial>()?;
    m.add_class::<PyPareto>()?;

    // -- Hypothesis tests --
    m.add_function(wrap_pyfunction!(ttest_1samp, &m)?)?;
    m.add_function(wrap_pyfunction!(ttest_ind, &m)?)?;
    m.add_function(wrap_pyfunction!(chi2_test, &m)?)?;
    m.add_function(wrap_pyfunction!(ks_2samp, &m)?)?;
    m.add_function(wrap_pyfunction!(anova_oneway, &m)?)?;
    m.add_function(wrap_pyfunction!(mann_whitney_u, &m)?)?;

    // -- Correlation --
    m.add_function(wrap_pyfunction!(spearman, &m)?)?;
    m.add_function(wrap_pyfunction!(kendall, &m)?)?;
    m.add_function(wrap_pyfunction!(corr_matrix, &m)?)?;

    // -- Regression --
    m.add_function(wrap_pyfunction!(ols, &m)?)?;
    m.add_function(wrap_pyfunction!(glm, &m)?)?;

    // -- Confidence intervals --
    m.add_function(wrap_pyfunction!(ci_mean, &m)?)?;
    m.add_function(wrap_pyfunction!(ci_proportion, &m)?)?;

    // -- Effect sizes --
    m.add_function(wrap_pyfunction!(cohens_d, &m)?)?;
    m.add_function(wrap_pyfunction!(hedges_g, &m)?)?;
    m.add_function(wrap_pyfunction!(glass_delta, &m)?)?;
    m.add_function(wrap_pyfunction!(eta_squared, &m)?)?;
    m.add_function(wrap_pyfunction!(omega_squared, &m)?)?;
    m.add_function(wrap_pyfunction!(cramers_v, &m)?)?;
    m.add_function(wrap_pyfunction!(cohens_w, &m)?)?;
    m.add_function(wrap_pyfunction!(point_biserial, &m)?)?;

    // -- Multiple comparison corrections --
    m.add_function(wrap_pyfunction!(bonferroni, &m)?)?;
    m.add_function(wrap_pyfunction!(benjamini_hochberg, &m)?)?;

    // -- Time series --
    m.add_function(wrap_pyfunction!(acf, &m)?)?;
    m.add_function(wrap_pyfunction!(pacf, &m)?)?;
    m.add_function(wrap_pyfunction!(adf_test, &m)?)?;
    m.add_function(wrap_pyfunction!(seasonal_decompose, &m)?)?;

    // -- ARIMA --
    m.add_class::<PyARIMA>()?;

    // -- SARIMAX --
    m.add_class::<PySarimax>()?;

    // -- Exponential smoothing --
    m.add_class::<PyExponentialSmoothing>()?;

    // -- Prophet --
    m.add_class::<PyProphet>()?;

    // -- Survival analysis --
    m.add_function(wrap_pyfunction!(kaplan_meier, &m)?)?;
    m.add_function(wrap_pyfunction!(log_rank_test, &m)?)?;
    m.add_function(wrap_pyfunction!(cox_ph, &m)?)?;

    // -- Bayesian MCMC --
    m.add_class::<PyMetropolisHastings>()?;
    m.add_function(wrap_pyfunction!(trace_summary, &m)?)?;
    m.add_function(wrap_pyfunction!(rhat, &m)?)?;
    m.add_function(wrap_pyfunction!(effective_sample_size, &m)?)?;

    // -- Bayesian optimization --
    m.add_class::<PyBayesianOptimizer>()?;

    // -- Anomaly detection --
    m.add_function(wrap_pyfunction!(zscore_anomaly, &m)?)?;
    m.add_function(wrap_pyfunction!(seasonal_anomaly, &m)?)?;
    m.add_function(wrap_pyfunction!(isolation_forest_anomaly, &m)?)?;
    m.add_function(wrap_pyfunction!(ewma_anomaly, &m)?)?;

    // -- Kalman filter --
    m.add_class::<PyKalmanFilter>()?;

    // -- GARCH --
    m.add_class::<PyGarch>()?;

    // -- VAR --
    m.add_class::<PyVarModel>()?;

    // -- Mixed-effects --
    m.add_function(wrap_pyfunction!(lmm, &m)?)?;

    // -- Time series features --
    m.add_function(wrap_pyfunction!(extract_features, &m)?)?;

    parent.add_submodule(&m)?;
    Ok(())
}
