//! Ensemble learning methods.

pub mod gradient_boosting;
pub mod hist_gradient_boosting;
pub mod random_forest;

pub use gradient_boosting::{
    GradientBoostingClassifier, GradientBoostingRegressor, Loss as GBLoss,
};
pub use hist_gradient_boosting::{
    HistGradientBoostingClassifier, HistGradientBoostingRegressor, ImportanceType,
};
pub use random_forest::{RandomForestClassifier, RandomForestRegressor};
