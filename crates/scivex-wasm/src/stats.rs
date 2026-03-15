//! Statistics bindings for JavaScript.

use wasm_bindgen::prelude::*;

/// Compute the mean of a Float64Array.
#[wasm_bindgen(js_name = "statsMean")]
pub fn stats_mean(data: &[f64]) -> Result<f64, JsError> {
    scivex_stats::mean(data).map_err(|e| JsError::new(&e.to_string()))
}

/// Compute the sample variance (ddof=1) of a Float64Array.
#[wasm_bindgen(js_name = "statsVariance")]
pub fn stats_variance(data: &[f64]) -> Result<f64, JsError> {
    scivex_stats::variance(data).map_err(|e| JsError::new(&e.to_string()))
}

/// Compute the sample standard deviation (ddof=1).
#[wasm_bindgen(js_name = "statsStdDev")]
pub fn stats_std_dev(data: &[f64]) -> Result<f64, JsError> {
    scivex_stats::std_dev(data).map_err(|e| JsError::new(&e.to_string()))
}

/// Compute the median.
#[wasm_bindgen(js_name = "statsMedian")]
pub fn stats_median(data: &[f64]) -> Result<f64, JsError> {
    scivex_stats::median(data).map_err(|e| JsError::new(&e.to_string()))
}

/// Compute the Pearson correlation coefficient between two arrays.
#[wasm_bindgen(js_name = "statsPearson")]
pub fn stats_pearson(x: &[f64], y: &[f64]) -> Result<f64, JsError> {
    scivex_stats::pearson(x, y).map_err(|e| JsError::new(&e.to_string()))
}

/// Compute the skewness.
#[wasm_bindgen(js_name = "statsSkewness")]
pub fn stats_skewness(data: &[f64]) -> Result<f64, JsError> {
    scivex_stats::skewness(data).map_err(|e| JsError::new(&e.to_string()))
}

/// Compute the excess kurtosis.
#[wasm_bindgen(js_name = "statsKurtosis")]
pub fn stats_kurtosis(data: &[f64]) -> Result<f64, JsError> {
    scivex_stats::kurtosis(data).map_err(|e| JsError::new(&e.to_string()))
}

/// Normal distribution.
#[wasm_bindgen]
pub struct WasmNormal {
    inner: scivex_stats::distributions::Normal<f64>,
}

#[wasm_bindgen]
impl WasmNormal {
    /// Create a normal distribution N(mu, sigma).
    #[wasm_bindgen(constructor)]
    pub fn new(mu: f64, sigma: f64) -> Result<WasmNormal, JsError> {
        let n = scivex_stats::distributions::Normal::new(mu, sigma)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(WasmNormal { inner: n })
    }

    /// Standard normal N(0, 1).
    pub fn standard() -> WasmNormal {
        WasmNormal {
            inner: scivex_stats::distributions::Normal::standard(),
        }
    }

    /// Probability density function at x.
    pub fn pdf(&self, x: f64) -> f64 {
        use scivex_stats::distributions::Distribution;
        self.inner.pdf(x)
    }

    /// Cumulative distribution function at x.
    pub fn cdf(&self, x: f64) -> f64 {
        use scivex_stats::distributions::Distribution;
        self.inner.cdf(x)
    }

    /// Percent point function (inverse CDF) at p.
    pub fn ppf(&self, p: f64) -> Result<f64, JsError> {
        use scivex_stats::distributions::Distribution;
        self.inner.ppf(p).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Distribution mean.
    #[wasm_bindgen(js_name = "mean")]
    pub fn dist_mean(&self) -> f64 {
        use scivex_stats::distributions::Distribution;
        self.inner.mean()
    }

    /// Distribution variance.
    #[wasm_bindgen(js_name = "variance")]
    pub fn dist_variance(&self) -> f64 {
        use scivex_stats::distributions::Distribution;
        self.inner.variance()
    }
}
