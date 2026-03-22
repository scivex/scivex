//! Ensemble learning methods.

pub mod catboost;
pub mod ebm;
pub mod gradient_boosting;
pub mod hist_gradient_boosting;
pub mod random_forest;
pub mod stacking;

pub use catboost::{CatBoostClassifier, CatBoostRegressor};
pub use ebm::{EbmClassifier, EbmRegressor};
pub use gradient_boosting::{
    GradientBoostingClassifier, GradientBoostingRegressor, Loss as GBLoss,
};
pub use hist_gradient_boosting::{
    HistGradientBoostingClassifier, HistGradientBoostingRegressor, ImportanceType,
};
pub use random_forest::{RandomForestClassifier, RandomForestRegressor};
pub use stacking::{EstimatorFactory, StackingClassifier, StackingRegressor};
