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

pub mod cluster;
pub mod ensemble;
pub mod error;
pub mod linear;
pub mod metrics;
pub mod model_selection;
pub mod naive_bayes;
pub mod neighbors;
pub mod preprocessing;
pub mod traits;
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
