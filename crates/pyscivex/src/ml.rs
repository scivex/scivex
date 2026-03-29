//! Python bindings for ML models — full classical ML toolkit.
//!
//! Exposes supervised models (regression, classification, ensemble, SVM, KNN,
//! naive Bayes), unsupervised models (clustering, decomposition), preprocessing
//! (scalers, encoders), metrics, and model selection utilities.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use scivex_core::Tensor;
use scivex_ml::traits::{Classifier, Predictor, Transformer};

use crate::tensor::PyTensor;

// ---------------------------------------------------------------------------
// Error helpers
// ---------------------------------------------------------------------------

#[allow(clippy::needless_pass_by_value)]
fn ml_err(e: scivex_ml::MlError) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}

#[allow(clippy::needless_pass_by_value)]
fn core_err(e: scivex_core::CoreError) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}

// ===========================================================================
// Macros for repetitive patterns
// ===========================================================================

/// Supervised model with fit(x,y) / predict(x) via the Predictor trait.
macro_rules! predictor_pyclass {
    (
        $py_name:ident, $py_str:literal, $rust_ty:ty,
        new(  $( $p:ident : $pt:ty ),* ) $body:block
        $( extra { $($extra:tt)* } )?
    ) => {
        #[pyclass(name = $py_str)]
        pub struct $py_name { inner: $rust_ty }

        #[pymethods]
        impl $py_name {
            #[new]
            #[allow(clippy::new_without_default)]
            fn new( $( $p : $pt ),* ) -> PyResult<Self> {
                let inner = $body;
                Ok(Self { inner })
            }

            fn fit(&mut self, x: &PyTensor, y: &PyTensor) -> PyResult<()> {
                self.inner.fit(x.as_f64()?, y.as_f64()?).map_err(ml_err)
            }

            fn predict(&self, x: &PyTensor) -> PyResult<PyTensor> {
                let r = self.inner.predict(x.as_f64()?).map_err(ml_err)?;
                Ok(PyTensor::from_f64(r))
            }

            $( $($extra)* )?
        }
    };
}

/// Transformer model with fit(x) / transform(x) / fit_transform(x).
macro_rules! transformer_pyclass {
    (
        $py_name:ident, $py_str:literal, $rust_ty:ty,
        new( $( $p:ident : $pt:ty ),* ) $body:block
        $( extra { $($extra:tt)* } )?
    ) => {
        #[pyclass(name = $py_str)]
        pub struct $py_name { inner: $rust_ty }

        #[pymethods]
        impl $py_name {
            #[new]
            #[allow(clippy::new_without_default)]
            fn new( $( $p : $pt ),* ) -> PyResult<Self> {
                let inner = $body;
                Ok(Self { inner })
            }

            fn fit(&mut self, x: &PyTensor) -> PyResult<()> {
                self.inner.fit(x.as_f64()?).map_err(ml_err)
            }

            fn transform(&self, x: &PyTensor) -> PyResult<PyTensor> {
                let r = self.inner.transform(x.as_f64()?).map_err(ml_err)?;
                Ok(PyTensor::from_f64(r))
            }

            fn fit_transform(&mut self, x: &PyTensor) -> PyResult<PyTensor> {
                let r = self.inner.fit_transform(x.as_f64()?).map_err(ml_err)?;
                Ok(PyTensor::from_f64(r))
            }

            $( $($extra)* )?
        }
    };
}

// ===========================================================================
// LINEAR MODELS
// ===========================================================================

/// Ordinary least-squares linear regression model.
#[pyclass(name = "LinearRegression")]
pub struct PyLinearRegression {
    inner: scivex_ml::linear::LinearRegression<f64>,
}

#[pymethods]
impl PyLinearRegression {
    #[new]
    fn new() -> Self {
        Self {
            inner: scivex_ml::linear::LinearRegression::new(),
        }
    }

    fn fit(&mut self, x: &PyTensor, y: &PyTensor) -> PyResult<()> {
        self.inner.fit(x.as_f64()?, y.as_f64()?).map_err(ml_err)
    }

    fn predict(&self, x: &PyTensor) -> PyResult<PyTensor> {
        let r = self.inner.predict(x.as_f64()?).map_err(ml_err)?;
        Ok(PyTensor::from_f64(r))
    }

    fn weights(&self) -> Option<Vec<f64>> {
        self.inner.weights().map(<[f64]>::to_vec)
    }

    fn bias(&self) -> Option<f64> {
        self.inner.bias()
    }
}

// Ridge regression with L2 regularization.
predictor_pyclass!(
    PyRidge, "Ridge", scivex_ml::linear::Ridge<f64>,
    new(alpha: f64) {
        scivex_ml::linear::Ridge::new(alpha).map_err(ml_err)?
    }
);

// Logistic regression classifier with gradient descent optimization.
predictor_pyclass!(
    PyLogisticRegression, "LogisticRegression",
    scivex_ml::linear::LogisticRegression<f64>,
    new(learning_rate: f64, max_iter: usize, tol: f64) {
        scivex_ml::linear::LogisticRegression::new(learning_rate, max_iter, tol).map_err(ml_err)?
    }
    extra {
        fn predict_proba(&self, x: &PyTensor) -> PyResult<PyTensor> {
            let r = self.inner.predict_proba(x.as_f64()?).map_err(ml_err)?;
            Ok(PyTensor::from_f64(r))
        }
    }
);

// ===========================================================================
// TREE MODELS
// ===========================================================================

// Decision tree classifier with configurable depth and minimum split size.
predictor_pyclass!(
    PyDecisionTreeClassifier, "DecisionTreeClassifier",
    scivex_ml::tree::DecisionTreeClassifier<f64>,
    new(max_depth: Option<usize>, min_samples_split: usize) {
        scivex_ml::tree::DecisionTreeClassifier::new(max_depth, min_samples_split)
    }
);

// Decision tree regressor with configurable depth and minimum split size.
predictor_pyclass!(
    PyDecisionTreeRegressor, "DecisionTreeRegressor",
    scivex_ml::tree::DecisionTreeRegressor<f64>,
    new(max_depth: Option<usize>, min_samples_split: usize) {
        scivex_ml::tree::DecisionTreeRegressor::new(max_depth, min_samples_split)
    }
);

// ===========================================================================
// ENSEMBLE MODELS
// ===========================================================================

// Random forest classifier using bagged decision trees.
predictor_pyclass!(
    PyRandomForestClassifier, "RandomForestClassifier",
    scivex_ml::ensemble::RandomForestClassifier<f64>,
    new(n_trees: usize, max_depth: Option<usize>, max_features: Option<usize>, seed: u64) {
        scivex_ml::ensemble::RandomForestClassifier::new(n_trees, max_depth, max_features, seed)
            .map_err(ml_err)?
    }
);

// Random forest regressor using bagged decision trees.
predictor_pyclass!(
    PyRandomForestRegressor, "RandomForestRegressor",
    scivex_ml::ensemble::RandomForestRegressor<f64>,
    new(n_trees: usize, max_depth: Option<usize>, max_features: Option<usize>, seed: u64) {
        scivex_ml::ensemble::RandomForestRegressor::new(n_trees, max_depth, max_features, seed)
            .map_err(ml_err)?
    }
);

/// Gradient boosting regressor with configurable loss ('mse', 'mae', 'huber').
#[pyclass(name = "GradientBoostingRegressor")]
pub struct PyGradientBoostingRegressor {
    inner: scivex_ml::ensemble::GradientBoostingRegressor<f64>,
}

#[pymethods]
impl PyGradientBoostingRegressor {
    #[new]
    #[pyo3(signature = (n_estimators, learning_rate, max_depth=None, loss="mse"))]
    fn new(
        n_estimators: usize,
        learning_rate: f64,
        max_depth: Option<usize>,
        loss: &str,
    ) -> PyResult<Self> {
        use scivex_ml::ensemble::GBLoss;
        let loss = match loss.to_lowercase().as_str() {
            "mse" => GBLoss::Mse,
            "mae" => GBLoss::Mae,
            "huber" => GBLoss::Huber,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "loss must be 'mse', 'mae', or 'huber'",
                ));
            }
        };
        let inner = scivex_ml::ensemble::GradientBoostingRegressor::new(
            n_estimators,
            learning_rate,
            max_depth,
            loss,
        )
        .map_err(ml_err)?;
        Ok(Self { inner })
    }

    fn fit(&mut self, x: &PyTensor, y: &PyTensor) -> PyResult<()> {
        self.inner.fit(x.as_f64()?, y.as_f64()?).map_err(ml_err)
    }

    fn predict(&self, x: &PyTensor) -> PyResult<PyTensor> {
        let r = self.inner.predict(x.as_f64()?).map_err(ml_err)?;
        Ok(PyTensor::from_f64(r))
    }
}

// Gradient boosting classifier using staged additive trees.
predictor_pyclass!(
    PyGradientBoostingClassifier, "GradientBoostingClassifier",
    scivex_ml::ensemble::GradientBoostingClassifier<f64>,
    new(n_estimators: usize, learning_rate: f64, max_depth: Option<usize>) {
        scivex_ml::ensemble::GradientBoostingClassifier::new(n_estimators, learning_rate, max_depth)
            .map_err(ml_err)?
    }
);

// ===========================================================================
// SVM
// ===========================================================================

/// Parse a kernel string name into the corresponding `Kernel` enum variant.
fn parse_kernel(
    kernel: &str,
    gamma: f64,
    degree: u32,
    coef0: f64,
) -> PyResult<scivex_ml::svm::Kernel<f64>> {
    match kernel.to_lowercase().as_str() {
        "linear" => Ok(scivex_ml::svm::Kernel::Linear),
        "rbf" => Ok(scivex_ml::svm::Kernel::Rbf { gamma }),
        "poly" | "polynomial" => Ok(scivex_ml::svm::Kernel::Poly {
            degree,
            gamma,
            coef0,
        }),
        "sigmoid" => Ok(scivex_ml::svm::Kernel::Sigmoid { gamma, coef0 }),
        _ => Err(pyo3::exceptions::PyValueError::new_err(
            "kernel must be 'linear', 'rbf', 'poly', or 'sigmoid'",
        )),
    }
}

/// Support Vector Classifier with configurable kernel ('linear', 'rbf', 'poly', 'sigmoid').
#[pyclass(name = "SVC")]
pub struct PySVC {
    inner: scivex_ml::svm::SVC<f64>,
}

#[pymethods]
impl PySVC {
    #[new]
    #[pyo3(signature = (kernel="rbf", c=1.0, gamma=1.0, degree=3, coef0=0.0))]
    fn new(kernel: &str, c: f64, gamma: f64, degree: u32, coef0: f64) -> PyResult<Self> {
        let k = parse_kernel(kernel, gamma, degree, coef0)?;
        let inner = scivex_ml::svm::SVC::new(k, c).map_err(ml_err)?;
        Ok(Self { inner })
    }

    fn fit(&mut self, x: &PyTensor, y: &PyTensor) -> PyResult<()> {
        self.inner.fit(x.as_f64()?, y.as_f64()?).map_err(ml_err)
    }

    fn predict(&self, x: &PyTensor) -> PyResult<PyTensor> {
        let r = self.inner.predict(x.as_f64()?).map_err(ml_err)?;
        Ok(PyTensor::from_f64(r))
    }
}

/// Support Vector Regressor with configurable kernel and epsilon-insensitive loss.
#[pyclass(name = "SVR")]
pub struct PySVR {
    inner: scivex_ml::svm::SVR<f64>,
}

#[pymethods]
impl PySVR {
    #[new]
    #[pyo3(signature = (kernel="rbf", c=1.0, epsilon=0.1, gamma=1.0, degree=3, coef0=0.0))]
    fn new(
        kernel: &str,
        c: f64,
        epsilon: f64,
        gamma: f64,
        degree: u32,
        coef0: f64,
    ) -> PyResult<Self> {
        let k = parse_kernel(kernel, gamma, degree, coef0)?;
        let inner = scivex_ml::svm::SVR::new(k, c, epsilon).map_err(ml_err)?;
        Ok(Self { inner })
    }

    fn fit(&mut self, x: &PyTensor, y: &PyTensor) -> PyResult<()> {
        self.inner.fit(x.as_f64()?, y.as_f64()?).map_err(ml_err)
    }

    fn predict(&self, x: &PyTensor) -> PyResult<PyTensor> {
        let r = self.inner.predict(x.as_f64()?).map_err(ml_err)?;
        Ok(PyTensor::from_f64(r))
    }
}

// ===========================================================================
// NEIGHBORS
// ===========================================================================

// K-nearest neighbors classifier.
predictor_pyclass!(
    PyKNNClassifier, "KNNClassifier",
    scivex_ml::neighbors::KNNClassifier<f64>,
    new(k: usize) {
        scivex_ml::neighbors::KNNClassifier::new(k).map_err(ml_err)?
    }
);

// K-nearest neighbors regressor.
predictor_pyclass!(
    PyKNNRegressor, "KNNRegressor",
    scivex_ml::neighbors::KNNRegressor<f64>,
    new(k: usize) {
        scivex_ml::neighbors::KNNRegressor::new(k).map_err(ml_err)?
    }
);

// ===========================================================================
// NAIVE BAYES
// ===========================================================================

/// Gaussian Naive Bayes classifier assuming normally distributed features.
#[pyclass(name = "GaussianNB")]
pub struct PyGaussianNB {
    inner: scivex_ml::naive_bayes::GaussianNB<f64>,
}

#[pymethods]
impl PyGaussianNB {
    #[new]
    fn new() -> Self {
        Self {
            inner: scivex_ml::naive_bayes::GaussianNB::new(),
        }
    }

    fn fit(&mut self, x: &PyTensor, y: &PyTensor) -> PyResult<()> {
        self.inner.fit(x.as_f64()?, y.as_f64()?).map_err(ml_err)
    }

    fn predict(&self, x: &PyTensor) -> PyResult<PyTensor> {
        let r = self.inner.predict(x.as_f64()?).map_err(ml_err)?;
        Ok(PyTensor::from_f64(r))
    }

    fn predict_proba(&self, x: &PyTensor) -> PyResult<PyTensor> {
        let r = self.inner.predict_proba(x.as_f64()?).map_err(ml_err)?;
        Ok(PyTensor::from_f64(r))
    }
}

// ===========================================================================
// CLUSTERING
// ===========================================================================

/// K-Means clustering with multiple random initializations.
#[pyclass(name = "KMeans")]
pub struct PyKMeans {
    inner: scivex_ml::cluster::KMeans<f64>,
}

#[pymethods]
impl PyKMeans {
    #[new]
    #[pyo3(signature = (n_clusters, max_iter=100, n_init=3, seed=42))]
    fn new(n_clusters: usize, max_iter: usize, n_init: usize, seed: u64) -> PyResult<Self> {
        let inner = scivex_ml::cluster::KMeans::new(n_clusters, max_iter, 1e-6, n_init, seed)
            .map_err(ml_err)?;
        Ok(Self { inner })
    }

    fn fit(&mut self, x: &PyTensor) -> PyResult<()> {
        self.inner.fit(x.as_f64()?).map_err(ml_err)
    }

    fn predict(&self, x: &PyTensor) -> PyResult<PyTensor> {
        let r = self.inner.predict(x.as_f64()?).map_err(ml_err)?;
        Ok(PyTensor::from_f64(r))
    }

    fn inertia(&self) -> Option<f64> {
        self.inner.inertia()
    }

    fn centroids(&self) -> PyResult<Option<PyTensor>> {
        match self.inner.centroids() {
            Some(data) => {
                let t = Tensor::from_vec(data.to_vec(), vec![data.len()]).map_err(core_err)?;
                Ok(Some(PyTensor::from_f64(t)))
            }
            None => Ok(None),
        }
    }
}

/// DBSCAN density-based clustering. Labels noise points as -1.
#[pyclass(name = "DBSCAN")]
pub struct PyDBSCAN {
    inner: scivex_ml::cluster::DBSCAN<f64>,
}

#[pymethods]
impl PyDBSCAN {
    #[new]
    #[pyo3(signature = (eps=0.5, min_samples=5))]
    fn new(eps: f64, min_samples: usize) -> PyResult<Self> {
        let inner = scivex_ml::cluster::DBSCAN::new(eps, min_samples).map_err(ml_err)?;
        Ok(Self { inner })
    }

    fn fit(&mut self, x: &PyTensor) -> PyResult<()> {
        self.inner.fit(x.as_f64()?).map_err(ml_err)
    }

    fn labels(&self) -> PyResult<Vec<i64>> {
        self.inner
            .labels()
            .map(|l| l.to_vec())
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("model not fitted"))
    }

    fn fit_predict(&mut self, x: &PyTensor) -> PyResult<PyTensor> {
        // fit_predict returns Result<Tensor<T>> for DBSCAN
        let r = self.inner.fit_predict(x.as_f64()?).map_err(ml_err)?;
        Ok(PyTensor::from_f64(r))
    }
}

/// Agglomerative (hierarchical) clustering with configurable linkage strategy.
#[pyclass(name = "AgglomerativeClustering")]
pub struct PyAgglomerativeClustering {
    inner: scivex_ml::cluster::AgglomerativeClustering<f64>,
}

#[pymethods]
impl PyAgglomerativeClustering {
    #[new]
    #[pyo3(signature = (n_clusters, linkage="ward"))]
    fn new(n_clusters: usize, linkage: &str) -> PyResult<Self> {
        let l = match linkage.to_lowercase().as_str() {
            "single" => scivex_ml::cluster::Linkage::Single,
            "complete" => scivex_ml::cluster::Linkage::Complete,
            "average" => scivex_ml::cluster::Linkage::Average,
            "ward" => scivex_ml::cluster::Linkage::Ward,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "linkage must be 'single', 'complete', 'average', or 'ward'",
                ));
            }
        };
        let inner =
            scivex_ml::cluster::AgglomerativeClustering::new(n_clusters, l).map_err(ml_err)?;
        Ok(Self { inner })
    }

    fn fit(&mut self, x: &PyTensor) -> PyResult<()> {
        self.inner.fit(x.as_f64()?).map_err(ml_err)
    }

    fn labels(&self) -> PyResult<Vec<usize>> {
        self.inner.labels().map(|l| l.to_vec()).map_err(ml_err)
    }

    fn fit_predict(&mut self, x: &PyTensor) -> PyResult<Vec<usize>> {
        self.inner.fit_predict(x.as_f64()?).map_err(ml_err)
    }
}

// ===========================================================================
// DECOMPOSITION
// ===========================================================================

// Principal Component Analysis for dimensionality reduction.
transformer_pyclass!(
    PyPCA, "PCA", scivex_ml::decomposition::PCA<f64>,
    new(n_components: usize) {
        scivex_ml::decomposition::PCA::new(n_components).map_err(ml_err)?
    }
    extra {
        fn explained_variance(&self) -> Option<Vec<f64>> {
            self.inner.explained_variance().map(<[f64]>::to_vec)
        }

        fn explained_variance_ratio(&self) -> Option<Vec<f64>> {
            self.inner.explained_variance_ratio()
        }

        fn components(&self) -> Option<Vec<f64>> {
            self.inner.components().map(<[f64]>::to_vec)
        }
    }
);

// Truncated SVD (aka LSA) for dimensionality reduction on sparse-friendly data.
transformer_pyclass!(
    PyTruncatedSVD, "TruncatedSVD", scivex_ml::decomposition::TruncatedSVD<f64>,
    new(n_components: usize) {
        scivex_ml::decomposition::TruncatedSVD::new(n_components).map_err(ml_err)?
    }
    extra {
        fn explained_variance(&self) -> Option<Vec<f64>> {
            self.inner.explained_variance().map(<[f64]>::to_vec)
        }

        fn explained_variance_ratio(&self) -> Option<Vec<f64>> {
            self.inner.explained_variance_ratio()
        }

        fn components(&self) -> Option<Vec<f64>> {
            self.inner.components().map(<[f64]>::to_vec)
        }
    }
);

/// t-SNE non-linear dimensionality reduction for visualization.
#[pyclass(name = "TSNE")]
pub struct PyTSNE {
    inner: scivex_ml::decomposition::TSNE<f64>,
}

#[pymethods]
impl PyTSNE {
    #[new]
    #[pyo3(signature = (n_components=2, perplexity=30.0))]
    fn new(n_components: usize, perplexity: f64) -> PyResult<Self> {
        let inner =
            scivex_ml::decomposition::TSNE::new(n_components, perplexity).map_err(ml_err)?;
        Ok(Self { inner })
    }

    fn fit_transform(&self, x: &PyTensor) -> PyResult<PyTensor> {
        let r = self.inner.fit_transform(x.as_f64()?).map_err(ml_err)?;
        Ok(PyTensor::from_f64(r))
    }
}

// ===========================================================================
// PREPROCESSING
// ===========================================================================

// Standardize features by removing the mean and scaling to unit variance.
transformer_pyclass!(
    PyStandardScaler, "StandardScaler", scivex_ml::preprocessing::StandardScaler<f64>,
    new() {
        scivex_ml::preprocessing::StandardScaler::new()
    }
);

// Scale features to the [0, 1] range based on per-feature min and max.
transformer_pyclass!(
    PyMinMaxScaler, "MinMaxScaler", scivex_ml::preprocessing::MinMaxScaler<f64>,
    new() {
        scivex_ml::preprocessing::MinMaxScaler::new()
    }
);

// Encode categorical integer features as a one-hot binary tensor.
transformer_pyclass!(
    PyOneHotEncoder, "OneHotEncoder", scivex_ml::preprocessing::OneHotEncoder<f64>,
    new() {
        scivex_ml::preprocessing::OneHotEncoder::new()
    }
);

/// Encode target labels as integer indices and decode them back.
#[pyclass(name = "LabelEncoder")]
pub struct PyLabelEncoder {
    inner: scivex_ml::preprocessing::LabelEncoder<f64>,
}

#[pymethods]
impl PyLabelEncoder {
    #[new]
    fn new() -> Self {
        Self {
            inner: scivex_ml::preprocessing::LabelEncoder::new(),
        }
    }

    fn fit(&mut self, y: Vec<f64>) -> PyResult<()> {
        self.inner.fit(&y).map_err(ml_err)
    }

    fn transform(&self, y: Vec<f64>) -> PyResult<Vec<usize>> {
        self.inner.transform(&y).map_err(ml_err)
    }

    fn inverse_transform(&self, indices: Vec<usize>) -> PyResult<Vec<f64>> {
        self.inner.inverse_transform(&indices).map_err(ml_err)
    }

    fn n_classes(&self) -> Option<usize> {
        self.inner.n_classes()
    }
}

// ===========================================================================
// METRICS
// ===========================================================================

/// Compute classification accuracy (fraction of correct predictions).
#[pyfunction]
pub fn accuracy(y_true: Vec<f64>, y_pred: Vec<f64>) -> PyResult<f64> {
    scivex_ml::metrics::accuracy(&y_true, &y_pred).map_err(ml_err)
}

/// Compute precision (positive predictive value) for binary classification.
#[pyfunction]
pub fn precision(y_true: Vec<f64>, y_pred: Vec<f64>) -> PyResult<f64> {
    scivex_ml::metrics::precision(&y_true, &y_pred).map_err(ml_err)
}

/// Compute recall (sensitivity / true positive rate) for binary classification.
#[pyfunction]
pub fn recall(y_true: Vec<f64>, y_pred: Vec<f64>) -> PyResult<f64> {
    scivex_ml::metrics::recall(&y_true, &y_pred).map_err(ml_err)
}

/// Compute the F1 score (harmonic mean of precision and recall).
#[pyfunction]
pub fn f1_score(y_true: Vec<f64>, y_pred: Vec<f64>) -> PyResult<f64> {
    scivex_ml::metrics::f1_score(&y_true, &y_pred).map_err(ml_err)
}

/// Build an n_classes x n_classes confusion matrix from true and predicted labels.
#[pyfunction]
pub fn confusion_matrix(
    y_true: Vec<f64>,
    y_pred: Vec<f64>,
    n_classes: usize,
) -> PyResult<Vec<Vec<usize>>> {
    scivex_ml::metrics::confusion_matrix(&y_true, &y_pred, n_classes).map_err(ml_err)
}

/// Compute mean squared error between true and predicted values.
#[pyfunction]
pub fn mse(y_true: Vec<f64>, y_pred: Vec<f64>) -> PyResult<f64> {
    scivex_ml::metrics::mse(&y_true, &y_pred).map_err(ml_err)
}

/// Compute root mean squared error between true and predicted values.
#[pyfunction]
pub fn rmse(y_true: Vec<f64>, y_pred: Vec<f64>) -> PyResult<f64> {
    scivex_ml::metrics::rmse(&y_true, &y_pred).map_err(ml_err)
}

/// Compute mean absolute error between true and predicted values.
#[pyfunction]
pub fn mae(y_true: Vec<f64>, y_pred: Vec<f64>) -> PyResult<f64> {
    scivex_ml::metrics::mae(&y_true, &y_pred).map_err(ml_err)
}

/// Compute the R-squared (coefficient of determination) regression score.
#[pyfunction]
pub fn r2_score(y_true: Vec<f64>, y_pred: Vec<f64>) -> PyResult<f64> {
    scivex_ml::metrics::r2_score(&y_true, &y_pred).map_err(ml_err)
}

// ===========================================================================
// MODEL SELECTION
// ===========================================================================

/// Split tensors x and y into random train/test subsets. Returns a dict with
/// keys 'x_train', 'x_test', 'y_train', 'y_test'.
#[pyfunction]
#[pyo3(signature = (x, y, test_ratio=0.2, seed=42))]
pub fn train_test_split<'py>(
    py: Python<'py>,
    x: &PyTensor,
    y: &PyTensor,
    test_ratio: f64,
    seed: u64,
) -> PyResult<Bound<'py, PyDict>> {
    let mut rng = scivex_core::random::Rng::new(seed);
    let (x_train, x_test, y_train, y_test) = scivex_ml::model_selection::train_test_split(
        x.as_f64()?,
        y.as_f64()?,
        test_ratio,
        &mut rng,
    )
    .map_err(ml_err)?;
    let d = PyDict::new(py);
    d.set_item("x_train", PyTensor::from_f64(x_train).into_pyobject(py)?)?;
    d.set_item("x_test", PyTensor::from_f64(x_test).into_pyobject(py)?)?;
    d.set_item("y_train", PyTensor::from_f64(y_train).into_pyobject(py)?)?;
    d.set_item("y_test", PyTensor::from_f64(y_test).into_pyobject(py)?)?;
    Ok(d)
}

// ===========================================================================
// FEATURE SELECTION
// ===========================================================================

/// Univariate feature selector that keeps the k best features scored by
/// chi2 or f_classif.
#[pyclass(name = "SelectKBest")]
pub struct PySelectKBest {
    inner: scivex_ml::feature_selection::SelectKBest<f64>,
}

#[pymethods]
impl PySelectKBest {
    /// Create a new SelectKBest selector.
    ///
    /// `scoring` must be `"chi2"` or `"f_classif"`.
    #[new]
    fn new(k: usize, scoring: &str) -> PyResult<Self> {
        let s = match scoring.to_lowercase().as_str() {
            "chi2" => scivex_ml::feature_selection::ScoringFunction::Chi2,
            "f_classif" => scivex_ml::feature_selection::ScoringFunction::FClassif,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "scoring must be 'chi2' or 'f_classif'",
                ));
            }
        };
        Ok(Self {
            inner: scivex_ml::feature_selection::SelectKBest::new(k, s),
        })
    }

    /// Compute feature scores and determine the top-k indices.
    fn fit(&mut self, x: &PyTensor, y: &PyTensor) -> PyResult<()> {
        self.inner.fit(x.as_f64()?, y.as_f64()?).map_err(ml_err)
    }

    /// Project `x` onto the selected features.
    fn transform(&self, x: &PyTensor) -> PyResult<PyTensor> {
        let r = self.inner.transform(x.as_f64()?).map_err(ml_err)?;
        Ok(PyTensor::from_f64(r))
    }

    /// Fit and transform in one step.
    fn fit_transform(&mut self, x: &PyTensor, y: &PyTensor) -> PyResult<PyTensor> {
        let r = self
            .inner
            .fit_transform(x.as_f64()?, y.as_f64()?)
            .map_err(ml_err)?;
        Ok(PyTensor::from_f64(r))
    }

    /// Per-feature scores computed during fit.
    fn scores(&self) -> Option<Vec<f64>> {
        self.inner.scores().map(<[f64]>::to_vec)
    }

    /// Indices of the selected features (sorted ascending).
    fn selected_features(&self) -> Option<Vec<usize>> {
        self.inner.selected_features().map(<[usize]>::to_vec)
    }
}

/// Recursive Feature Elimination using an estimator's importances.
///
/// Currently uses `LinearRegression` as the internal estimator.
#[pyclass(name = "RFE")]
pub struct PyRFE {
    inner: scivex_ml::feature_selection::RFE<f64>,
}

#[pymethods]
impl PyRFE {
    /// Create a new RFE that selects `n_features_to_select` features.
    #[new]
    #[pyo3(signature = (n_features_to_select, step=1))]
    fn new(n_features_to_select: usize, step: usize) -> Self {
        let mut rfe = scivex_ml::feature_selection::RFE::new(n_features_to_select);
        if step > 1 {
            rfe.set_step(step);
        }
        Self { inner: rfe }
    }

    /// Fit RFE using a LinearRegression estimator.
    fn fit(&mut self, x: &PyTensor, y: &PyTensor) -> PyResult<()> {
        let estimator = scivex_ml::linear::LinearRegression::<f64>::new();
        self.inner
            .fit(&estimator, x.as_f64()?, y.as_f64()?)
            .map_err(ml_err)
    }

    /// Project `x` onto the selected features.
    fn transform(&self, x: &PyTensor) -> PyResult<PyTensor> {
        let r = self.inner.transform(x.as_f64()?).map_err(ml_err)?;
        Ok(PyTensor::from_f64(r))
    }

    /// Boolean mask of selected features.
    fn support(&self) -> Option<Vec<bool>> {
        self.inner.support().map(<[bool]>::to_vec)
    }

    /// Feature ranking: 1 = selected, higher = eliminated earlier.
    fn ranking(&self) -> Option<Vec<usize>> {
        self.inner.ranking().map(<[usize]>::to_vec)
    }
}

// ===========================================================================
// EXPLAINABILITY
// ===========================================================================

/// Compute LIME explanations for instances using a DecisionTreeRegressor.
///
/// Returns a list of dicts, each with keys: 'weights', 'intercept', 'score',
/// 'prediction'.
#[pyfunction]
#[pyo3(signature = (model, x_train, x_explain, n_perturbations=500, kernel_width=None, seed=42))]
pub fn lime_explain<'py>(
    py: Python<'py>,
    model: &PyDecisionTreeRegressor,
    x_train: &PyTensor,
    x_explain: &PyTensor,
    n_perturbations: usize,
    kernel_width: Option<f64>,
    seed: u64,
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    let explanations = scivex_ml::explain::lime(
        &model.inner,
        x_train.as_f64()?,
        x_explain.as_f64()?,
        n_perturbations,
        kernel_width,
        seed,
    )
    .map_err(ml_err)?;

    let mut results = Vec::with_capacity(explanations.len());
    for exp in &explanations {
        let d = PyDict::new(py);
        d.set_item("weights", exp.weights.clone())?;
        d.set_item("intercept", exp.intercept)?;
        d.set_item("score", exp.score)?;
        d.set_item("prediction", exp.prediction)?;
        results.push(d);
    }
    Ok(results)
}

/// Compute partial dependence for a single feature using a
/// DecisionTreeRegressor.
///
/// Returns a dict with keys: 'feature_values', 'average_predictions'.
#[pyfunction]
pub fn partial_dependence<'py>(
    py: Python<'py>,
    model: &PyDecisionTreeRegressor,
    x: &PyTensor,
    feature_index: usize,
    grid_resolution: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let pdp = scivex_ml::explain::partial_dependence(
        &model.inner,
        x.as_f64()?,
        feature_index,
        grid_resolution,
    )
    .map_err(ml_err)?;

    let d = PyDict::new(py);
    d.set_item(
        "feature_values",
        PyTensor::from_f64(pdp.feature_values).into_pyobject(py)?,
    )?;
    d.set_item(
        "average_predictions",
        PyTensor::from_f64(pdp.average_predictions).into_pyobject(py)?,
    )?;
    Ok(d)
}

/// Compute permutation feature importance using a DecisionTreeRegressor
/// with R2 as the scoring metric.
///
/// Returns a dict with keys: 'importances_mean', 'importances_std'.
#[pyfunction]
#[pyo3(signature = (model, x, y, n_repeats=5, seed=42))]
pub fn permutation_importance<'py>(
    py: Python<'py>,
    model: &PyDecisionTreeRegressor,
    x: &PyTensor,
    y: &PyTensor,
    n_repeats: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyDict>> {
    fn r2_scorer(y_true: &Tensor<f64>, y_pred: &Tensor<f64>) -> f64 {
        let yt = y_true.as_slice();
        let yp = y_pred.as_slice();
        let n = yt.len() as f64;
        let mean = yt.iter().sum::<f64>() / n;
        let ss_res: f64 = yt.iter().zip(yp).map(|(t, p)| (t - p).powi(2)).sum();
        let ss_tot: f64 = yt.iter().map(|t| (t - mean).powi(2)).sum();
        if ss_tot.abs() < 1e-15 {
            0.0
        } else {
            1.0 - ss_res / ss_tot
        }
    }

    let result = scivex_ml::explain::permutation_importance(
        &model.inner,
        x.as_f64()?,
        y.as_f64()?,
        n_repeats,
        r2_scorer,
        seed,
    )
    .map_err(ml_err)?;

    let d = PyDict::new(py);
    d.set_item(
        "importances_mean",
        PyTensor::from_f64(result.importances_mean).into_pyobject(py)?,
    )?;
    d.set_item(
        "importances_std",
        PyTensor::from_f64(result.importances_std).into_pyobject(py)?,
    )?;
    Ok(d)
}

/// Compute Kernel SHAP values for instances using a DecisionTreeRegressor.
///
/// Returns a Tensor of shape `[n_explain, n_features]` with SHAP values.
#[pyfunction]
#[pyo3(signature = (model, x_background, x_explain, n_samples=100, seed=42))]
pub fn kernel_shap(
    model: &PyDecisionTreeRegressor,
    x_background: &PyTensor,
    x_explain: &PyTensor,
    n_samples: usize,
    seed: u64,
) -> PyResult<PyTensor> {
    let r = scivex_ml::explain::kernel_shap(
        &model.inner,
        x_background.as_f64()?,
        x_explain.as_f64()?,
        n_samples,
        seed,
    )
    .map_err(ml_err)?;
    Ok(PyTensor::from_f64(r))
}

// ===========================================================================
// ONLINE / STREAMING LEARNING
// ===========================================================================

/// Online linear regression trained with mini-batch SGD.
#[pyclass(name = "SGDRegressor")]
pub struct PySGDRegressor {
    inner: scivex_ml::online::SGDRegressor<f64>,
}

#[pymethods]
impl PySGDRegressor {
    /// Create a new SGDRegressor with the given learning rate.
    #[new]
    #[pyo3(signature = (learning_rate=0.01, l2_penalty=0.0))]
    fn new(learning_rate: f64, l2_penalty: f64) -> PyResult<Self> {
        let mut inner =
            scivex_ml::online::SGDRegressor::<f64>::new(learning_rate).map_err(ml_err)?;
        if l2_penalty > 0.0 {
            inner.set_l2_penalty(l2_penalty);
        }
        Ok(Self { inner })
    }

    /// Update the model with a mini-batch of data.
    fn partial_fit(&mut self, x: &PyTensor, y: &PyTensor) -> PyResult<()> {
        use scivex_ml::online::IncrementalPredictor;
        self.inner
            .partial_fit(x.as_f64()?, y.as_f64()?)
            .map_err(ml_err)
    }

    /// Predict targets for new features.
    fn predict(&self, x: &PyTensor) -> PyResult<PyTensor> {
        use scivex_ml::online::IncrementalPredictor;
        let r = self.inner.predict(x.as_f64()?).map_err(ml_err)?;
        Ok(PyTensor::from_f64(r))
    }

    /// Total number of training samples seen so far.
    fn n_samples_seen(&self) -> usize {
        use scivex_ml::online::IncrementalPredictor;
        self.inner.n_samples_seen()
    }
}

/// Online binary classifier trained with logistic-loss SGD.
#[pyclass(name = "SGDClassifier")]
pub struct PySGDClassifier {
    inner: scivex_ml::online::SGDClassifier<f64>,
}

#[pymethods]
impl PySGDClassifier {
    /// Create a new SGDClassifier with the given learning rate.
    #[new]
    #[pyo3(signature = (learning_rate=0.01, l2_penalty=0.0))]
    fn new(learning_rate: f64, l2_penalty: f64) -> PyResult<Self> {
        let mut inner =
            scivex_ml::online::SGDClassifier::<f64>::new(learning_rate).map_err(ml_err)?;
        if l2_penalty > 0.0 {
            inner.set_l2_penalty(l2_penalty);
        }
        Ok(Self { inner })
    }

    /// Update the model with a mini-batch of data.
    fn partial_fit(&mut self, x: &PyTensor, y: &PyTensor) -> PyResult<()> {
        use scivex_ml::online::IncrementalPredictor;
        self.inner
            .partial_fit(x.as_f64()?, y.as_f64()?)
            .map_err(ml_err)
    }

    /// Predict class labels (0.0 or 1.0) for new features.
    fn predict(&self, x: &PyTensor) -> PyResult<PyTensor> {
        use scivex_ml::online::IncrementalPredictor;
        let r = self.inner.predict(x.as_f64()?).map_err(ml_err)?;
        Ok(PyTensor::from_f64(r))
    }

    /// Total number of training samples seen so far.
    fn n_samples_seen(&self) -> usize {
        use scivex_ml::online::IncrementalPredictor;
        self.inner.n_samples_seen()
    }
}

/// Online (mini-batch) K-Means clustering.
#[pyclass(name = "OnlineKMeans")]
pub struct PyOnlineKMeans {
    inner: scivex_ml::online::OnlineKMeans<f64>,
}

#[pymethods]
impl PyOnlineKMeans {
    /// Create a new OnlineKMeans with the given number of clusters.
    #[new]
    #[pyo3(signature = (n_clusters, seed=42))]
    fn new(n_clusters: usize, seed: u64) -> PyResult<Self> {
        let inner =
            scivex_ml::online::OnlineKMeans::<f64>::new(n_clusters, seed).map_err(ml_err)?;
        Ok(Self { inner })
    }

    /// Update centroids with a new batch of data.
    fn partial_fit(&mut self, x: &PyTensor) -> PyResult<()> {
        self.inner.partial_fit(x.as_f64()?).map_err(ml_err)
    }

    /// Assign each sample to the nearest centroid.
    fn predict(&self, x: &PyTensor) -> PyResult<PyTensor> {
        let r = self.inner.predict(x.as_f64()?).map_err(ml_err)?;
        Ok(PyTensor::from_f64(r))
    }

    /// Current centroid coordinates (flattened).
    fn centroids(&self) -> Option<Vec<f64>> {
        self.inner.centroids().map(<[f64]>::to_vec)
    }

    /// Total number of training samples seen across all batches.
    fn n_samples_seen(&self) -> usize {
        self.inner.n_samples_seen()
    }
}

// ===========================================================================
// MODEL PERSISTENCE
// ===========================================================================

/// Save a LinearRegression model to a binary file.
#[pyfunction]
pub fn save_model(model_type: &str, path: &str, model: &Bound<'_, PyAny>) -> PyResult<()> {
    use scivex_ml::persist::Persistable;
    match model_type {
        "LinearRegression" => {
            let m: PyRef<'_, PyLinearRegression> = model.extract()?;
            m.inner.save(path).map_err(ml_err)
        }
        "Ridge" => {
            let m: PyRef<'_, PyRidge> = model.extract()?;
            m.inner.save(path).map_err(ml_err)
        }
        "DecisionTreeClassifier" => {
            let m: PyRef<'_, PyDecisionTreeClassifier> = model.extract()?;
            m.inner.save(path).map_err(ml_err)
        }
        "DecisionTreeRegressor" => {
            let m: PyRef<'_, PyDecisionTreeRegressor> = model.extract()?;
            m.inner.save(path).map_err(ml_err)
        }
        "KMeans" => {
            let m: PyRef<'_, PyKMeans> = model.extract()?;
            m.inner.save(path).map_err(ml_err)
        }
        "StandardScaler" => {
            let m: PyRef<'_, PyStandardScaler> = model.extract()?;
            m.inner.save(path).map_err(ml_err)
        }
        "MinMaxScaler" => {
            let m: PyRef<'_, PyMinMaxScaler> = model.extract()?;
            m.inner.save(path).map_err(ml_err)
        }
        "PCA" => {
            let m: PyRef<'_, PyPCA> = model.extract()?;
            m.inner.save(path).map_err(ml_err)
        }
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unsupported model type for persistence: {model_type}"
        ))),
    }
}

/// Load a model from a binary file. Returns the appropriate model object.
///
/// `model_type` must be one of: 'LinearRegression', 'Ridge',
/// 'DecisionTreeClassifier', 'DecisionTreeRegressor', 'KMeans',
/// 'StandardScaler', 'MinMaxScaler', 'PCA'.
#[pyfunction]
pub fn load_model(py: Python<'_>, model_type: &str, path: &str) -> PyResult<PyObject> {
    use scivex_ml::persist::Persistable;
    match model_type {
        "LinearRegression" => {
            let m = scivex_ml::linear::LinearRegression::<f64>::load(path).map_err(ml_err)?;
            Ok(PyLinearRegression { inner: m }
                .into_pyobject(py)?
                .into_any()
                .unbind())
        }
        "Ridge" => {
            let m = scivex_ml::linear::Ridge::<f64>::load(path).map_err(ml_err)?;
            Ok(PyRidge { inner: m }.into_pyobject(py)?.into_any().unbind())
        }
        "DecisionTreeClassifier" => {
            let m = scivex_ml::tree::DecisionTreeClassifier::<f64>::load(path).map_err(ml_err)?;
            Ok(PyDecisionTreeClassifier { inner: m }
                .into_pyobject(py)?
                .into_any()
                .unbind())
        }
        "DecisionTreeRegressor" => {
            let m = scivex_ml::tree::DecisionTreeRegressor::<f64>::load(path).map_err(ml_err)?;
            Ok(PyDecisionTreeRegressor { inner: m }
                .into_pyobject(py)?
                .into_any()
                .unbind())
        }
        "KMeans" => {
            let m = scivex_ml::cluster::KMeans::<f64>::load(path).map_err(ml_err)?;
            Ok(PyKMeans { inner: m }.into_pyobject(py)?.into_any().unbind())
        }
        "StandardScaler" => {
            let m = scivex_ml::preprocessing::StandardScaler::<f64>::load(path).map_err(ml_err)?;
            Ok(PyStandardScaler { inner: m }
                .into_pyobject(py)?
                .into_any()
                .unbind())
        }
        "MinMaxScaler" => {
            let m = scivex_ml::preprocessing::MinMaxScaler::<f64>::load(path).map_err(ml_err)?;
            Ok(PyMinMaxScaler { inner: m }
                .into_pyobject(py)?
                .into_any()
                .unbind())
        }
        "PCA" => {
            let m = scivex_ml::decomposition::PCA::<f64>::load(path).map_err(ml_err)?;
            Ok(PyPCA { inner: m }.into_pyobject(py)?.into_any().unbind())
        }
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unsupported model type for loading: {model_type}"
        ))),
    }
}

// ===========================================================================
// APPROXIMATE NEAREST NEIGHBORS
// ===========================================================================

/// Approximate nearest neighbor index using random projection trees (Annoy-style).
#[pyclass(name = "AnnoyIndex")]
pub struct PyAnnoyIndex {
    inner: scivex_ml::ann::AnnoyIndex<f64>,
}

#[pymethods]
impl PyAnnoyIndex {
    /// Build an AnnoyIndex from a list of data points.
    ///
    /// `data` is a flat list of floats; `dim` is the dimensionality per point.
    /// `n_trees` controls index quality (more trees = more accurate, slower build).
    /// `max_leaf_size` controls how many points sit in each leaf node.
    #[new]
    #[pyo3(signature = (data, dim, n_trees=10, max_leaf_size=10, seed=42))]
    fn new(
        data: Vec<f64>,
        dim: usize,
        n_trees: usize,
        max_leaf_size: usize,
        seed: u64,
    ) -> PyResult<Self> {
        if dim == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err("dim must be >= 1"));
        }
        if data.len() % dim != 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "data length must be divisible by dim",
            ));
        }
        let points: Vec<Vec<f64>> = data.chunks(dim).map(|c| c.to_vec()).collect();
        let mut rng = scivex_core::random::Rng::new(seed);
        let inner = scivex_ml::ann::AnnoyIndex::build(points, n_trees, max_leaf_size, &mut rng)
            .map_err(ml_err)?;
        Ok(Self { inner })
    }

    /// Query for the `k` approximate nearest neighbors of `point`.
    ///
    /// Returns a list of `(index, distance)` tuples sorted by ascending distance.
    fn query(&self, point: Vec<f64>, k: usize) -> Vec<(usize, f64)> {
        self.inner.query(&point, k)
    }

    /// Number of data points in the index.
    fn len(&self) -> usize {
        self.inner.len()
    }

    /// Number of trees in the index.
    fn n_trees(&self) -> usize {
        self.inner.n_trees()
    }

    /// Dimensionality of the indexed vectors.
    fn dim(&self) -> usize {
        self.inner.dim()
    }
}

// ===========================================================================
// ADDITIONAL ENSEMBLE MODELS
// ===========================================================================

// Histogram-based gradient boosting classifier (fast, bin-based splits).
predictor_pyclass!(
    PyHistGradientBoostingClassifier, "HistGradientBoostingClassifier",
    scivex_ml::ensemble::HistGradientBoostingClassifier<f64>,
    new(n_estimators: usize, learning_rate: f64, max_leaf_nodes: usize) {
        scivex_ml::ensemble::HistGradientBoostingClassifier::new(n_estimators, learning_rate, max_leaf_nodes)
            .map_err(ml_err)?
    }
);

// Histogram-based gradient boosting regressor (fast, bin-based splits).
predictor_pyclass!(
    PyHistGradientBoostingRegressor, "HistGradientBoostingRegressor",
    scivex_ml::ensemble::HistGradientBoostingRegressor<f64>,
    new(n_estimators: usize, learning_rate: f64, max_leaf_nodes: usize) {
        scivex_ml::ensemble::HistGradientBoostingRegressor::new(n_estimators, learning_rate, max_leaf_nodes)
            .map_err(ml_err)?
    }
);

// CatBoost-style ordered boosting classifier with oblivious trees.
predictor_pyclass!(
    PyCatBoostClassifier, "CatBoostClassifier",
    scivex_ml::ensemble::CatBoostClassifier<f64>,
    new(n_estimators: usize, learning_rate: f64, max_depth: usize, l2_reg: f64) {
        scivex_ml::ensemble::CatBoostClassifier::new(n_estimators, learning_rate, max_depth, l2_reg)
            .map_err(ml_err)?
    }
);

// CatBoost-style ordered boosting regressor with oblivious trees.
predictor_pyclass!(
    PyCatBoostRegressor, "CatBoostRegressor",
    scivex_ml::ensemble::CatBoostRegressor<f64>,
    new(n_estimators: usize, learning_rate: f64, max_depth: usize, l2_reg: f64) {
        scivex_ml::ensemble::CatBoostRegressor::new(n_estimators, learning_rate, max_depth, l2_reg)
            .map_err(ml_err)?
    }
);

// Explainable Boosting Machine classifier (EBM / GA²M).
predictor_pyclass!(
    PyEbmClassifier, "EbmClassifier",
    scivex_ml::ensemble::EbmClassifier<f64>,
    new(max_rounds: usize, max_bins: usize, learning_rate: f64) {
        scivex_ml::ensemble::EbmClassifier::new(max_rounds, max_bins, learning_rate)
            .map_err(ml_err)?
    }
);

// Explainable Boosting Machine regressor (EBM / GA²M).
predictor_pyclass!(
    PyEbmRegressor, "EbmRegressor",
    scivex_ml::ensemble::EbmRegressor<f64>,
    new(max_rounds: usize, max_bins: usize, learning_rate: f64) {
        scivex_ml::ensemble::EbmRegressor::new(max_rounds, max_bins, learning_rate)
            .map_err(ml_err)?
    }
);

/// Stacking regressor that combines base estimators via a meta-learner.
///
/// Uses LinearRegression as both base estimators and meta-learner by default.
/// `n_base` controls how many base LinearRegression estimators to stack.
#[pyclass(name = "StackingRegressor", unsendable)]
pub struct PyStackingRegressor {
    inner: scivex_ml::ensemble::StackingRegressor<f64>,
}

#[pymethods]
impl PyStackingRegressor {
    #[new]
    #[pyo3(signature = (n_base=3, cv_folds=5, seed=42))]
    fn new(n_base: usize, cv_folds: usize, seed: u64) -> PyResult<Self> {
        use scivex_ml::ensemble::EstimatorFactory;
        let base_factories: Vec<EstimatorFactory<f64>> = (0..n_base)
            .map(|_| {
                Box::new(|| {
                    Box::new(scivex_ml::linear::LinearRegression::<f64>::new())
                        as Box<dyn Predictor<f64>>
                }) as EstimatorFactory<f64>
            })
            .collect();
        let meta =
            Box::new(scivex_ml::linear::LinearRegression::<f64>::new()) as Box<dyn Predictor<f64>>;
        let inner =
            scivex_ml::ensemble::StackingRegressor::new(base_factories, meta, cv_folds, seed)
                .map_err(ml_err)?;
        Ok(Self { inner })
    }

    fn fit(&mut self, x: &PyTensor, y: &PyTensor) -> PyResult<()> {
        self.inner.fit(x.as_f64()?, y.as_f64()?).map_err(ml_err)
    }

    fn predict(&self, x: &PyTensor) -> PyResult<PyTensor> {
        let r = self.inner.predict(x.as_f64()?).map_err(ml_err)?;
        Ok(PyTensor::from_f64(r))
    }
}

/// Stacking classifier that combines base estimators via a meta-learner.
///
/// Uses LogisticRegression as base estimators and meta-learner by default.
/// `n_base` controls how many base estimators to stack.
#[pyclass(name = "StackingClassifier", unsendable)]
pub struct PyStackingClassifier {
    inner: scivex_ml::ensemble::StackingClassifier<f64>,
}

#[pymethods]
impl PyStackingClassifier {
    #[new]
    #[pyo3(signature = (n_base=3, cv_folds=5, seed=42))]
    fn new(n_base: usize, cv_folds: usize, seed: u64) -> PyResult<Self> {
        use scivex_ml::ensemble::EstimatorFactory;
        let base_factories: Vec<EstimatorFactory<f64>> = (0..n_base)
            .map(|_| {
                Box::new(|| {
                    Box::new(
                        scivex_ml::linear::LogisticRegression::<f64>::new(0.01, 100, 1e-6).unwrap(),
                    ) as Box<dyn Predictor<f64>>
                }) as EstimatorFactory<f64>
            })
            .collect();
        let meta =
            Box::new(scivex_ml::linear::LogisticRegression::<f64>::new(0.01, 100, 1e-6).unwrap())
                as Box<dyn Predictor<f64>>;
        let inner =
            scivex_ml::ensemble::StackingClassifier::new(base_factories, meta, cv_folds, seed)
                .map_err(ml_err)?;
        Ok(Self { inner })
    }

    fn fit(&mut self, x: &PyTensor, y: &PyTensor) -> PyResult<()> {
        self.inner.fit(x.as_f64()?, y.as_f64()?).map_err(ml_err)
    }

    fn predict(&self, x: &PyTensor) -> PyResult<PyTensor> {
        let r = self.inner.predict(x.as_f64()?).map_err(ml_err)?;
        Ok(PyTensor::from_f64(r))
    }
}

// ===========================================================================
// ADDITIONAL PREPROCESSORS
// ===========================================================================

// Ordinal encoder for categorical features (maps categories to integers).
transformer_pyclass!(
    PyOrdinalEncoder, "OrdinalEncoder", scivex_ml::preprocessing::OrdinalEncoder<f64>,
    new() {
        scivex_ml::preprocessing::OrdinalEncoder::new()
    }
);

// Binary encoder for categorical features (binary-code representation).
transformer_pyclass!(
    PyBinaryEncoder, "BinaryEncoder", scivex_ml::preprocessing::BinaryEncoder<f64>,
    new() {
        scivex_ml::preprocessing::BinaryEncoder::new()
    }
);

/// Target (supervised) encoder that uses target statistics to encode categories.
///
/// Unlike standard transformers, TargetEncoder needs both X and y for fitting.
#[pyclass(name = "TargetEncoder")]
pub struct PyTargetEncoder {
    inner: scivex_ml::preprocessing::TargetEncoder<f64>,
}

#[pymethods]
impl PyTargetEncoder {
    #[new]
    #[pyo3(signature = (smoothing=1.0))]
    fn new(smoothing: f64) -> Self {
        Self {
            inner: scivex_ml::preprocessing::TargetEncoder::new(smoothing),
        }
    }

    /// Fit the encoder using both features X and target y.
    fn fit(&mut self, x: &PyTensor, y: &PyTensor) -> PyResult<()> {
        self.inner
            .fit_supervised(x.as_f64()?, y.as_f64()?)
            .map_err(ml_err)
    }

    /// Transform features using the fitted encoding.
    fn transform(&self, x: &PyTensor) -> PyResult<PyTensor> {
        let r = self.inner.transform(x.as_f64()?).map_err(ml_err)?;
        Ok(PyTensor::from_f64(r))
    }

    /// Fit and transform in one step.
    fn fit_transform(&mut self, x: &PyTensor, y: &PyTensor) -> PyResult<PyTensor> {
        self.inner
            .fit_supervised(x.as_f64()?, y.as_f64()?)
            .map_err(ml_err)?;
        let r = self.inner.transform(x.as_f64()?).map_err(ml_err)?;
        Ok(PyTensor::from_f64(r))
    }
}

// ===========================================================================
// FEATURE ENGINEERING
// ===========================================================================

// Generate polynomial and interaction features.
transformer_pyclass!(
    PyPolynomialFeatures, "PolynomialFeatures",
    scivex_ml::feature_engineering::PolynomialFeatures<f64>,
    new(degree: usize, include_bias: bool, interaction_only: bool) {
        scivex_ml::feature_engineering::PolynomialFeatures::new(degree, include_bias, interaction_only)
            .map_err(ml_err)?
    }
);

// ===========================================================================
// PIPELINE
// ===========================================================================

/// A machine learning pipeline that chains transformer steps followed by a
/// predictor.
///
/// Build the pipeline by adding named transformer steps and optionally setting
/// a final predictor. During `fit`, each transformer is fit-transformed
/// sequentially; the final transformed data is passed to the predictor.
#[pyclass(name = "Pipeline", unsendable)]
pub struct PyPipeline {
    inner: scivex_ml::pipeline::Pipeline<f64>,
}

#[pymethods]
impl PyPipeline {
    /// Create an empty pipeline.
    #[new]
    fn new() -> Self {
        Self {
            inner: scivex_ml::pipeline::Pipeline::new(),
        }
    }

    /// Add a StandardScaler transformer step.
    fn add_standard_scaler(&mut self, name: &str) {
        let pipe = std::mem::take(&mut self.inner);
        self.inner = pipe.add_step(
            name,
            Box::new(scivex_ml::preprocessing::StandardScaler::<f64>::new()),
        );
    }

    /// Add a MinMaxScaler transformer step.
    fn add_min_max_scaler(&mut self, name: &str) {
        let pipe = std::mem::take(&mut self.inner);
        self.inner = pipe.add_step(
            name,
            Box::new(scivex_ml::preprocessing::MinMaxScaler::<f64>::new()),
        );
    }

    /// Add a PCA transformer step.
    fn add_pca(&mut self, name: &str, n_components: usize) -> PyResult<()> {
        let pca = scivex_ml::decomposition::PCA::new(n_components).map_err(ml_err)?;
        let pipe = std::mem::take(&mut self.inner);
        self.inner = pipe.add_step(name, Box::new(pca));
        Ok(())
    }

    /// Add a PolynomialFeatures transformer step.
    fn add_polynomial_features(
        &mut self,
        name: &str,
        degree: usize,
        include_bias: bool,
        interaction_only: bool,
    ) -> PyResult<()> {
        let pf = scivex_ml::feature_engineering::PolynomialFeatures::new(
            degree,
            include_bias,
            interaction_only,
        )
        .map_err(ml_err)?;
        let pipe = std::mem::take(&mut self.inner);
        self.inner = pipe.add_step(name, Box::new(pf));
        Ok(())
    }

    /// Set a LinearRegression as the final predictor.
    fn set_linear_regression(&mut self, name: &str) {
        let pipe = std::mem::take(&mut self.inner);
        self.inner = pipe.set_predictor(
            name,
            Box::new(scivex_ml::linear::LinearRegression::<f64>::new()),
        );
    }

    /// Set a Ridge regression as the final predictor.
    fn set_ridge(&mut self, name: &str, alpha: f64) -> PyResult<()> {
        let ridge = scivex_ml::linear::Ridge::new(alpha).map_err(ml_err)?;
        let pipe = std::mem::take(&mut self.inner);
        self.inner = pipe.set_predictor(name, Box::new(ridge));
        Ok(())
    }

    /// Set a DecisionTreeRegressor as the final predictor.
    fn set_decision_tree_regressor(
        &mut self,
        name: &str,
        max_depth: Option<usize>,
        min_samples_split: usize,
    ) {
        let dt = scivex_ml::tree::DecisionTreeRegressor::new(max_depth, min_samples_split);
        let pipe = std::mem::take(&mut self.inner);
        self.inner = pipe.set_predictor(name, Box::new(dt));
    }

    /// Number of transformer steps (excluding the final predictor).
    fn n_steps(&self) -> usize {
        self.inner.n_steps()
    }

    /// Fit the full pipeline (transformers + predictor) on X and y.
    fn fit(&mut self, x: &PyTensor, y: &PyTensor) -> PyResult<()> {
        self.inner.fit(x.as_f64()?, y.as_f64()?).map_err(ml_err)
    }

    /// Predict using the fitted pipeline.
    fn predict(&self, x: &PyTensor) -> PyResult<PyTensor> {
        let r = self.inner.predict(x.as_f64()?).map_err(ml_err)?;
        Ok(PyTensor::from_f64(r))
    }
}

// ===========================================================================
// MODEL SELECTION FUNCTIONS
// ===========================================================================

/// Perform k-fold cross-validation and return per-fold R² scores.
///
/// Uses a LinearRegression model internally.
#[pyfunction]
#[pyo3(signature = (x, y, n_folds=5, seed=42))]
pub fn cross_val_score(
    x: &PyTensor,
    y: &PyTensor,
    n_folds: usize,
    seed: u64,
) -> PyResult<Vec<f64>> {
    let model = scivex_ml::linear::LinearRegression::<f64>::new();
    let mut rng = scivex_core::random::Rng::new(seed);
    scivex_ml::model_selection::cross_val_score(
        &model,
        x.as_f64()?,
        y.as_f64()?,
        n_folds,
        scivex_ml::metrics::regression::r2_score,
        &mut rng,
    )
    .map_err(ml_err)
}

/// Perform KFold split and return a list of (train_indices, test_indices) tuples.
#[pyfunction]
#[pyo3(signature = (n_samples, n_folds=5, seed=42))]
pub fn kfold(
    n_samples: usize,
    n_folds: usize,
    seed: u64,
) -> PyResult<Vec<(Vec<usize>, Vec<usize>)>> {
    let mut rng = scivex_core::random::Rng::new(seed);
    let kf =
        scivex_ml::model_selection::KFold::new(n_folds, n_samples, &mut rng).map_err(ml_err)?;
    Ok(kf.iter().collect())
}

/// Grid search cross-validation over a list of Ridge alpha values.
///
/// Returns a dict with keys: 'best_index', 'best_score', 'mean_scores',
/// 'all_scores'.
#[pyfunction]
#[pyo3(signature = (x, y, alphas, n_folds=5, seed=42))]
pub fn grid_search_cv<'py>(
    py: Python<'py>,
    x: &PyTensor,
    y: &PyTensor,
    alphas: Vec<f64>,
    n_folds: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyDict>> {
    let candidates: Vec<scivex_ml::linear::Ridge<f64>> = alphas
        .iter()
        .map(|&a| scivex_ml::linear::Ridge::new(a))
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(ml_err)?;
    let mut rng = scivex_core::random::Rng::new(seed);
    let result = scivex_ml::search::grid_search_cv(
        &candidates,
        x.as_f64()?,
        y.as_f64()?,
        n_folds,
        scivex_ml::metrics::regression::r2_score,
        &mut rng,
    )
    .map_err(ml_err)?;

    let d = PyDict::new(py);
    d.set_item("best_index", result.best_index)?;
    d.set_item("best_score", result.best_score)?;
    d.set_item("mean_scores", result.mean_scores)?;
    d.set_item("all_scores", result.all_scores)?;
    Ok(d)
}

/// Random search cross-validation: randomly samples Ridge alpha values.
///
/// Returns a dict with keys: 'best_index', 'best_score', 'mean_scores',
/// 'all_scores'.
#[pyfunction]
#[pyo3(signature = (x, y, n_iter=10, alpha_min=0.001, alpha_max=10.0, n_folds=5, seed=42))]
#[allow(clippy::too_many_arguments)]
pub fn random_search_cv<'py>(
    py: Python<'py>,
    x: &PyTensor,
    y: &PyTensor,
    n_iter: usize,
    alpha_min: f64,
    alpha_max: f64,
    n_folds: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyDict>> {
    let mut rng = scivex_core::random::Rng::new(seed);
    let builder = move |rng: &mut scivex_core::random::Rng| -> scivex_ml::error::Result<scivex_ml::linear::Ridge<f64>> {
        let alpha = alpha_min + rng.next_f64() * (alpha_max - alpha_min);
        scivex_ml::linear::Ridge::new(alpha)
    };
    let result = scivex_ml::search::random_search_cv(
        builder,
        n_iter,
        x.as_f64()?,
        y.as_f64()?,
        n_folds,
        scivex_ml::metrics::regression::r2_score,
        &mut rng,
    )
    .map_err(ml_err)?;

    let d = PyDict::new(py);
    d.set_item("best_index", result.best_index)?;
    d.set_item("best_score", result.best_score)?;
    d.set_item("mean_scores", result.mean_scores)?;
    d.set_item("all_scores", result.all_scores)?;
    Ok(d)
}

// ===========================================================================
// HNSW INDEX
// ===========================================================================

/// Hierarchical Navigable Small World graph index for approximate nearest
/// neighbor search.
#[pyclass(name = "HnswIndex")]
pub struct PyHnswIndex {
    inner: scivex_ml::neighbors::HnswIndex<f64>,
}

#[pymethods]
impl PyHnswIndex {
    /// Create a new empty HNSW index.
    ///
    /// `dim` is the dimensionality of vectors.
    /// `metric` is one of 'l2', 'cosine', 'dot', 'hamming'.
    #[new]
    #[pyo3(signature = (dim, metric="l2"))]
    fn new(dim: usize, metric: &str) -> PyResult<Self> {
        let m = match metric.to_lowercase().as_str() {
            "l2" | "euclidean" => scivex_ml::neighbors::DistanceMetric::L2,
            "cosine" => scivex_ml::neighbors::DistanceMetric::Cosine,
            "dot" | "dotproduct" | "dot_product" => {
                scivex_ml::neighbors::DistanceMetric::DotProduct
            }
            "hamming" => scivex_ml::neighbors::DistanceMetric::Hamming,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "metric must be 'l2', 'cosine', 'dot', or 'hamming'",
                ));
            }
        };
        let inner = scivex_ml::neighbors::HnswIndex::new(dim, m).map_err(ml_err)?;
        Ok(Self { inner })
    }

    /// Set the ef_search parameter (controls search accuracy vs speed).
    fn set_ef_search(&mut self, ef: usize) {
        self.inner.set_ef_search(ef);
    }

    /// Add vectors (as a 2-D tensor of shape [n, dim]) to the index.
    fn add(&mut self, vectors: &PyTensor) -> PyResult<()> {
        self.inner.add(vectors.as_f64()?).map_err(ml_err)
    }

    /// Search for the k nearest neighbors of `query` (a flat list of dim floats).
    ///
    /// Returns a tuple `(indices, distances)`.
    fn search(&self, query: Vec<f64>, k: usize) -> PyResult<(Vec<usize>, Vec<f64>)> {
        let result = self.inner.search(&query, k).map_err(ml_err)?;
        Ok((result.indices, result.distances))
    }

    /// Batch search for the k nearest neighbors of each query row.
    ///
    /// `queries` is a 2-D tensor of shape [n_queries, dim].
    /// Returns a list of `(indices, distances)` tuples, one per query.
    fn batch_search(&self, queries: &PyTensor, k: usize) -> PyResult<Vec<(Vec<usize>, Vec<f64>)>> {
        let results = self
            .inner
            .batch_search(queries.as_f64()?, k)
            .map_err(ml_err)?;
        Ok(results
            .into_iter()
            .map(|r| (r.indices, r.distances))
            .collect())
    }
}

// ===========================================================================
// TREE SHAP
// ===========================================================================

/// Compute Tree SHAP values for a fitted DecisionTreeRegressor.
///
/// Returns a Tensor of shape `[n_samples, n_features]` with SHAP values.
#[pyfunction]
pub fn tree_shap(model: &PyDecisionTreeRegressor, x: &PyTensor) -> PyResult<PyTensor> {
    let r = scivex_ml::explain::tree_shap(&model.inner, x.as_f64()?).map_err(ml_err)?;
    Ok(PyTensor::from_f64(r))
}

// ===========================================================================
// AUTOML
// ===========================================================================

/// Run AutoML pipeline optimization over (scaler, model) combinations.
///
/// Searches over StandardScaler + MinMaxScaler paired with
/// LinearRegression + Ridge, evaluating via k-fold cross-validation with
/// R2 score.
///
/// Returns a dict with keys: 'best_score', 'best_scaler_idx',
/// 'best_model_idx', 'scores'.
#[pyfunction]
#[pyo3(signature = (x, y, k_folds=3, ridge_alpha=1.0, seed=42))]
pub fn pipeline_optimize<'py>(
    py: Python<'py>,
    x: &PyTensor,
    y: &PyTensor,
    k_folds: usize,
    ridge_alpha: f64,
    seed: u64,
) -> PyResult<Bound<'py, PyDict>> {
    let space = scivex_ml::automl::SearchSpace {
        scalers: vec![
            Box::new(|| {
                Box::new(scivex_ml::preprocessing::StandardScaler::<f64>::new())
                    as Box<dyn Transformer<f64>>
            }),
            Box::new(|| {
                Box::new(scivex_ml::preprocessing::MinMaxScaler::<f64>::new())
                    as Box<dyn Transformer<f64>>
            }),
        ],
        models: vec![
            Box::new(|| {
                Box::new(scivex_ml::linear::LinearRegression::<f64>::new())
                    as Box<dyn Predictor<f64>>
            }),
            Box::new(move || {
                Box::new(scivex_ml::linear::Ridge::<f64>::new(ridge_alpha).unwrap())
                    as Box<dyn Predictor<f64>>
            }),
        ],
    };

    let mut rng = scivex_core::random::Rng::new(seed);
    let result = scivex_ml::automl::pipeline_optimize(
        &space,
        x.as_f64()?,
        y.as_f64()?,
        k_folds,
        scivex_ml::metrics::regression::r2_score,
        &mut rng,
    )
    .map_err(ml_err)?;

    let d = PyDict::new(py);
    d.set_item("best_score", result.best_score)?;
    d.set_item("best_scaler_idx", result.best_scaler_idx)?;
    d.set_item("best_model_idx", result.best_model_idx)?;
    let scores_py: Vec<Vec<f64>> = result.scores;
    d.set_item("scores", scores_py)?;
    Ok(d)
}

// ===========================================================================
// Submodule registration
// ===========================================================================

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let ml = PyModule::new(py, "ml")?;

    // Linear models
    ml.add_class::<PyLinearRegression>()?;
    ml.add_class::<PyRidge>()?;
    ml.add_class::<PyLogisticRegression>()?;

    // Trees
    ml.add_class::<PyDecisionTreeClassifier>()?;
    ml.add_class::<PyDecisionTreeRegressor>()?;

    // Ensembles
    ml.add_class::<PyRandomForestClassifier>()?;
    ml.add_class::<PyRandomForestRegressor>()?;
    ml.add_class::<PyGradientBoostingClassifier>()?;
    ml.add_class::<PyGradientBoostingRegressor>()?;
    ml.add_class::<PyHistGradientBoostingClassifier>()?;
    ml.add_class::<PyHistGradientBoostingRegressor>()?;
    ml.add_class::<PyCatBoostClassifier>()?;
    ml.add_class::<PyCatBoostRegressor>()?;
    ml.add_class::<PyEbmClassifier>()?;
    ml.add_class::<PyEbmRegressor>()?;
    ml.add_class::<PyStackingClassifier>()?;
    ml.add_class::<PyStackingRegressor>()?;

    // SVM
    ml.add_class::<PySVC>()?;
    ml.add_class::<PySVR>()?;

    // Neighbors
    ml.add_class::<PyKNNClassifier>()?;
    ml.add_class::<PyKNNRegressor>()?;
    ml.add_class::<PyHnswIndex>()?;

    // Naive Bayes
    ml.add_class::<PyGaussianNB>()?;

    // Clustering
    ml.add_class::<PyKMeans>()?;
    ml.add_class::<PyDBSCAN>()?;
    ml.add_class::<PyAgglomerativeClustering>()?;

    // Decomposition
    ml.add_class::<PyPCA>()?;
    ml.add_class::<PyTruncatedSVD>()?;
    ml.add_class::<PyTSNE>()?;

    // Preprocessing
    ml.add_class::<PyStandardScaler>()?;
    ml.add_class::<PyMinMaxScaler>()?;
    ml.add_class::<PyOneHotEncoder>()?;
    ml.add_class::<PyLabelEncoder>()?;
    ml.add_class::<PyOrdinalEncoder>()?;
    ml.add_class::<PyBinaryEncoder>()?;
    ml.add_class::<PyTargetEncoder>()?;

    // Metrics
    ml.add_function(wrap_pyfunction!(accuracy, &ml)?)?;
    ml.add_function(wrap_pyfunction!(precision, &ml)?)?;
    ml.add_function(wrap_pyfunction!(recall, &ml)?)?;
    ml.add_function(wrap_pyfunction!(f1_score, &ml)?)?;
    ml.add_function(wrap_pyfunction!(confusion_matrix, &ml)?)?;
    ml.add_function(wrap_pyfunction!(mse, &ml)?)?;
    ml.add_function(wrap_pyfunction!(rmse, &ml)?)?;
    ml.add_function(wrap_pyfunction!(mae, &ml)?)?;
    ml.add_function(wrap_pyfunction!(r2_score, &ml)?)?;

    // Model selection
    ml.add_function(wrap_pyfunction!(train_test_split, &ml)?)?;
    ml.add_function(wrap_pyfunction!(cross_val_score, &ml)?)?;
    ml.add_function(wrap_pyfunction!(kfold, &ml)?)?;
    ml.add_function(wrap_pyfunction!(grid_search_cv, &ml)?)?;
    ml.add_function(wrap_pyfunction!(random_search_cv, &ml)?)?;

    // Feature engineering
    ml.add_class::<PyPolynomialFeatures>()?;

    // Pipeline
    ml.add_class::<PyPipeline>()?;

    // Feature selection
    ml.add_class::<PySelectKBest>()?;
    ml.add_class::<PyRFE>()?;

    // Explainability
    ml.add_function(wrap_pyfunction!(lime_explain, &ml)?)?;
    ml.add_function(wrap_pyfunction!(partial_dependence, &ml)?)?;
    ml.add_function(wrap_pyfunction!(permutation_importance, &ml)?)?;
    ml.add_function(wrap_pyfunction!(kernel_shap, &ml)?)?;
    ml.add_function(wrap_pyfunction!(tree_shap, &ml)?)?;

    // Online / streaming learning
    ml.add_class::<PySGDRegressor>()?;
    ml.add_class::<PySGDClassifier>()?;
    ml.add_class::<PyOnlineKMeans>()?;

    // Model persistence
    ml.add_function(wrap_pyfunction!(save_model, &ml)?)?;
    ml.add_function(wrap_pyfunction!(load_model, &ml)?)?;

    // Approximate nearest neighbors
    ml.add_class::<PyAnnoyIndex>()?;

    // AutoML
    ml.add_function(wrap_pyfunction!(pipeline_optimize, &ml)?)?;

    parent.add_submodule(&ml)?;
    Ok(())
}
