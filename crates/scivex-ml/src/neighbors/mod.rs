//! Nearest-neighbour algorithms.

pub mod brute_force;
pub mod distance;
pub mod hnsw;
pub mod knn;
pub mod pq;

pub use brute_force::{BruteForceIndex, NearestNeighborResult};
pub use distance::DistanceMetric;
pub use hnsw::HnswIndex;
pub use knn::{KNNClassifier, KNNRegressor};
pub use pq::ProductQuantizer;
