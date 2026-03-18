//! Model explainability: feature importance, partial dependence, and SHAP values.

mod lime;
mod pdp;
mod permutation_importance;
mod shap_kernel;
mod shap_tree;

pub use lime::{LimeExplanation, lime};
pub use pdp::{PartialDependence, partial_dependence};
pub use permutation_importance::{PermutationImportanceResult, permutation_importance};
pub use shap_kernel::kernel_shap;
pub use shap_tree::tree_shap;
