//! Dimensionality reduction: PCA, TruncatedSVD, t-SNE.

mod pca;
mod truncated_svd;
mod tsne;

pub use pca::PCA;
pub use truncated_svd::TruncatedSVD;
pub use tsne::TSNE;
