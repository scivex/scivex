//! Machine learning bindings for JavaScript.

use wasm_bindgen::prelude::*;

use crate::tensor::WasmTensor;

/// Linear regression model.
#[wasm_bindgen]
pub struct WasmLinearRegression {
    inner: scivex_ml::linear::LinearRegression<f64>,
}

impl Default for WasmLinearRegression {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl WasmLinearRegression {
    /// Create a new linear regression model.
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmLinearRegression {
        WasmLinearRegression {
            inner: scivex_ml::linear::LinearRegression::new(),
        }
    }

    /// Fit the model on training data.
    /// `x` shape: `[n_samples, n_features]`, `y` shape: `[n_samples]`.
    pub fn fit(&mut self, x: &WasmTensor, y: &WasmTensor) -> Result<(), JsError> {
        use scivex_ml::traits::Predictor;
        self.inner
            .fit(x.inner(), y.inner())
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Predict target values for new data.
    pub fn predict(&self, x: &WasmTensor) -> Result<WasmTensor, JsError> {
        use scivex_ml::traits::Predictor;
        let pred = self
            .inner
            .predict(x.inner())
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(WasmTensor::from_inner(pred))
    }

    /// Get fitted weights.
    pub fn weights(&self) -> Result<Vec<f64>, JsError> {
        self.inner
            .weights()
            .map(<[f64]>::to_vec)
            .ok_or_else(|| JsError::new("model not fitted"))
    }

    /// Get fitted bias (intercept).
    pub fn bias(&self) -> Result<f64, JsError> {
        self.inner
            .bias()
            .ok_or_else(|| JsError::new("model not fitted"))
    }
}

/// K-Means clustering.
#[wasm_bindgen]
pub struct WasmKMeans {
    inner: scivex_ml::cluster::KMeans<f64>,
}

#[wasm_bindgen]
impl WasmKMeans {
    /// Create a new KMeans instance.
    ///
    /// - `n_clusters`: number of clusters
    /// - `max_iter`: maximum iterations (e.g. 300)
    /// - `n_init`: number of random restarts (e.g. 10)
    /// - `seed`: random seed
    #[wasm_bindgen(constructor)]
    pub fn new(
        n_clusters: usize,
        max_iter: usize,
        n_init: usize,
        seed: u64,
    ) -> Result<WasmKMeans, JsError> {
        let inner = scivex_ml::cluster::KMeans::new(n_clusters, max_iter, 1e-8, n_init, seed)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(WasmKMeans { inner })
    }

    /// Fit the model on data `x` (shape `[n_samples, n_features]`).
    pub fn fit(&mut self, x: &WasmTensor) -> Result<(), JsError> {
        self.inner
            .fit(x.inner())
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Predict cluster labels.
    pub fn predict(&self, x: &WasmTensor) -> Result<WasmTensor, JsError> {
        let labels = self
            .inner
            .predict(x.inner())
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(WasmTensor::from_inner(labels))
    }

    /// Get the inertia (sum of squared distances to centroids).
    pub fn inertia(&self) -> Option<f64> {
        self.inner.inertia()
    }
}

/// DBSCAN clustering.
#[wasm_bindgen]
pub struct WasmDBSCAN {
    inner: scivex_ml::cluster::DBSCAN<f64>,
}

#[wasm_bindgen]
impl WasmDBSCAN {
    /// Create a new DBSCAN instance.
    ///
    /// - `eps`: neighbourhood radius
    /// - `min_samples`: minimum points to form a core
    #[wasm_bindgen(constructor)]
    pub fn new(eps: f64, min_samples: usize) -> Result<WasmDBSCAN, JsError> {
        let inner = scivex_ml::cluster::DBSCAN::new(eps, min_samples)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(WasmDBSCAN { inner })
    }

    /// Fit and return labels as a tensor (-1 = noise).
    #[wasm_bindgen(js_name = "fitPredict")]
    pub fn fit_predict(&mut self, x: &WasmTensor) -> Result<WasmTensor, JsError> {
        let labels = self
            .inner
            .fit_predict(x.inner())
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(WasmTensor::from_inner(labels))
    }

    /// Number of clusters found (excluding noise).
    #[wasm_bindgen(js_name = "nClusters")]
    pub fn n_clusters(&self) -> Option<usize> {
        self.inner.n_clusters()
    }
}

/// StandardScaler — zero mean, unit variance.
#[wasm_bindgen]
pub struct WasmStandardScaler {
    inner: scivex_ml::preprocessing::StandardScaler<f64>,
}

impl Default for WasmStandardScaler {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl WasmStandardScaler {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmStandardScaler {
        WasmStandardScaler {
            inner: scivex_ml::preprocessing::StandardScaler::new(),
        }
    }

    /// Fit and transform the data.
    #[wasm_bindgen(js_name = "fitTransform")]
    pub fn fit_transform(&mut self, x: &WasmTensor) -> Result<WasmTensor, JsError> {
        use scivex_ml::traits::Transformer;
        let result = self
            .inner
            .fit_transform(x.inner())
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(WasmTensor::from_inner(result))
    }

    /// Transform new data using fitted parameters.
    pub fn transform(&self, x: &WasmTensor) -> Result<WasmTensor, JsError> {
        use scivex_ml::traits::Transformer;
        let result = self
            .inner
            .transform(x.inner())
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(WasmTensor::from_inner(result))
    }
}

/// PCA — Principal Component Analysis.
#[wasm_bindgen]
pub struct WasmPCA {
    inner: scivex_ml::decomposition::PCA<f64>,
}

#[wasm_bindgen]
impl WasmPCA {
    /// Create a PCA with `n_components` components.
    #[wasm_bindgen(constructor)]
    pub fn new(n_components: usize) -> Result<WasmPCA, JsError> {
        let inner = scivex_ml::decomposition::PCA::new(n_components)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(WasmPCA { inner })
    }

    /// Fit and transform the data.
    #[wasm_bindgen(js_name = "fitTransform")]
    pub fn fit_transform(&mut self, x: &WasmTensor) -> Result<WasmTensor, JsError> {
        use scivex_ml::traits::Transformer;
        let result = self
            .inner
            .fit_transform(x.inner())
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(WasmTensor::from_inner(result))
    }

    /// Get explained variance ratios.
    #[wasm_bindgen(js_name = "explainedVarianceRatio")]
    pub fn explained_variance_ratio(&self) -> Result<Vec<f64>, JsError> {
        self.inner
            .explained_variance_ratio()
            .ok_or_else(|| JsError::new("PCA not fitted"))
    }
}

/// Accuracy score for classification.
#[wasm_bindgen(js_name = "metricsAccuracy")]
pub fn metrics_accuracy(y_true: &[f64], y_pred: &[f64]) -> Result<f64, JsError> {
    scivex_ml::metrics::accuracy(y_true, y_pred).map_err(|e| JsError::new(&e.to_string()))
}

/// Mean squared error for regression.
#[wasm_bindgen(js_name = "metricsMse")]
pub fn metrics_mse(y_true: &[f64], y_pred: &[f64]) -> Result<f64, JsError> {
    scivex_ml::metrics::mse(y_true, y_pred).map_err(|e| JsError::new(&e.to_string()))
}
