//! # scivex-ml
//!
//! Classical machine learning algorithms for the Scivex ecosystem.
//!
//! Provides estimators following a consistent trait-based API:
//!
//! - [`Transformer`] — unsupervised fit / transform (scalers, encoders)
//! - [`Predictor`] — supervised fit / predict (regression, classification)
//! - [`Classifier`] — extends `Predictor` with `predict_proba`
//!
//! ## Modules
//!
//! | Module | Contents |
//! |--------|----------|
//! | [`metrics`] | Accuracy, precision, recall, F1, MSE, R2, etc. |
//! | [`preprocessing`] | `StandardScaler`, `MinMaxScaler`, `LabelEncoder` |
//! | [`linear`] | `LinearRegression`, `Ridge`, `LogisticRegression` |
//! | [`tree`] | `DecisionTreeClassifier`, `DecisionTreeRegressor` |
//! | [`ensemble`] | `RandomForestClassifier`, `RandomForestRegressor` |
//! | [`neighbors`] | `KNNClassifier`, `KNNRegressor` |
//! | [`cluster`] | `KMeans` |
//! | [`naive_bayes`] | `GaussianNB` |
//! | [`model_selection`] | `train_test_split`, `KFold`, `cross_val_score` |

/// K-Means clustering.
pub mod cluster;
/// Ensemble methods (Random Forest).
pub mod ensemble;
/// ML error types.
pub mod error;
/// Linear models (regression, ridge, logistic).
pub mod linear;
/// Evaluation metrics for classification and regression.
pub mod metrics;
/// Train/test split, k-fold cross-validation.
pub mod model_selection;
/// Naive Bayes classifiers.
pub mod naive_bayes;
/// K-nearest neighbors classifiers and regressors.
pub mod neighbors;
/// Data preprocessing: scaling and encoding.
pub mod preprocessing;
/// Estimator trait hierarchy (`Transformer`, `Predictor`, `Classifier`).
pub mod traits;
/// Decision tree classifiers and regressors.
pub mod tree;

pub use error::{MlError, Result};
pub use traits::{Classifier, Predictor, Transformer};

/// Convenience re-exports.
pub mod prelude {
    pub use crate::error::{MlError, Result};
    pub use crate::traits::{Classifier, Predictor, Transformer};

    // Preprocessing
    pub use crate::preprocessing::{LabelEncoder, MinMaxScaler, StandardScaler};

    // Linear models
    pub use crate::linear::{LinearRegression, LogisticRegression, Ridge};

    // Trees & ensembles
    pub use crate::ensemble::{RandomForestClassifier, RandomForestRegressor};
    pub use crate::tree::{DecisionTreeClassifier, DecisionTreeRegressor};

    // Neighbors
    pub use crate::neighbors::{KNNClassifier, KNNRegressor};

    // Clustering
    pub use crate::cluster::KMeans;

    // Naive Bayes
    pub use crate::naive_bayes::GaussianNB;

    // Model selection
    pub use crate::model_selection::{KFold, cross_val_score, train_test_split};
}
