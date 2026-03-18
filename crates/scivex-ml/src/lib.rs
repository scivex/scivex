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
//! | [`preprocessing`] | `StandardScaler`, `MinMaxScaler`, `LabelEncoder`, `OneHotEncoder` |
//! | [`linear`] | `LinearRegression`, `Ridge`, `LogisticRegression` |
//! | [`tree`] | `DecisionTreeClassifier`, `DecisionTreeRegressor` |
//! | [`ensemble`] | `RandomForest*`, `GradientBoosting*` |
//! | [`svm`] | `SVC`, `SVR`, `Kernel` |
//! | [`neighbors`] | `KNNClassifier`, `KNNRegressor` |
//! | [`cluster`] | `KMeans`, `DBSCAN` |
//! | [`naive_bayes`] | `GaussianNB` |
//! | [`pipeline`] | `Pipeline`, `FeatureUnion`, `ColumnTransformer` |
//! | [`search`] | `grid_search_cv`, `random_search_cv` |
//! | [`decomposition`] | `PCA`, `TruncatedSVD`, `TSNE` |
//! | [`persist`] | Persistable trait, binary save/load for all models |
//! | [`model_selection`] | `train_test_split`, `KFold`, `cross_val_score` |

/// K-Means clustering.
pub mod cluster;
/// Dimensionality reduction: PCA, TruncatedSVD, t-SNE.
pub mod decomposition;
/// Ensemble methods (Random Forest, Gradient Boosting).
pub mod ensemble;
/// ML error types.
pub mod error;
/// Model explainability: feature importance, SHAP, partial dependence.
pub mod explain;
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
/// Online / streaming ML algorithms.
pub mod online;
/// Model persistence: save and load trained models.
pub mod persist;
/// Pipeline composition: chaining transformers and predictors.
pub mod pipeline;
/// Data preprocessing: scaling and encoding.
pub mod preprocessing;
/// Hyperparameter search: grid search, random search.
pub mod search;
/// Support Vector Machines (classifier and regressor).
pub mod svm;
/// Estimator trait hierarchy (`Transformer`, `Predictor`, `Classifier`).
pub mod traits;
/// Decision tree classifiers and regressors.
pub mod tree;

pub use error::{MlError, Result};
pub use online::IncrementalPredictor;
pub use traits::{Classifier, Predictor, Transformer};

/// Convenience re-exports.
pub mod prelude {
    pub use crate::error::{MlError, Result};
    pub use crate::traits::{Classifier, Predictor, Transformer};

    // Preprocessing
    pub use crate::preprocessing::{LabelEncoder, MinMaxScaler, OneHotEncoder, StandardScaler};

    // Linear models
    pub use crate::linear::{LinearRegression, LogisticRegression, Ridge};

    // Trees & ensembles
    pub use crate::ensemble::{
        GBLoss, GradientBoostingClassifier, GradientBoostingRegressor,
        HistGradientBoostingClassifier, HistGradientBoostingRegressor, ImportanceType,
        RandomForestClassifier, RandomForestRegressor,
    };
    pub use crate::tree::{DecisionTreeClassifier, DecisionTreeRegressor};

    // SVM
    pub use crate::svm::{Kernel, SVC, SVR};

    // Neighbors
    pub use crate::neighbors::{
        BruteForceIndex, DistanceMetric, HnswIndex, KNNClassifier, KNNRegressor,
        NearestNeighborResult, ProductQuantizer,
    };

    // Clustering
    pub use crate::cluster::{AgglomerativeClustering, DBSCAN, DendrogramNode, KMeans, Linkage};

    // Naive Bayes
    pub use crate::naive_bayes::GaussianNB;

    // Dimensionality reduction
    pub use crate::decomposition::{PCA, TSNE, TruncatedSVD};

    // Pipeline & composition
    pub use crate::pipeline::{ColumnTransformer, FeatureUnion, Pipeline};

    // Hyperparameter search
    pub use crate::search::{SearchResult, grid_search_cv, random_search_cv};

    // Model selection
    pub use crate::model_selection::{KFold, cross_val_score, train_test_split};

    // Persistence
    pub use crate::persist::Persistable;

    // Explainability
    pub use crate::explain::{
        LimeExplanation, PartialDependence, PermutationImportanceResult, kernel_shap, lime,
        partial_dependence, permutation_importance, tree_shap,
    };

    // Online / streaming
    pub use crate::online::{
        IncrementalPredictor, OnlineKMeans, SGDClassifier, SGDRegressor, StreamingMean,
        StreamingVariance,
    };
}
