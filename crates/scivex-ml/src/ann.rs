//! Approximate nearest neighbor search via random projection trees.
//!
//! Implements an Annoy-style index that builds multiple random projection trees
//! for fast approximate nearest neighbor queries. Each tree recursively splits
//! the data along random hyperplanes until leaf nodes are small enough, then
//! queries collect candidates from all trees and compute exact distances to
//! return the top-k results.
//!
//! # Examples
//!
//! ```
//! # use scivex_ml::ann::AnnoyIndex;
//! # use scivex_core::random::Rng;
//! let data = vec![
//!     vec![0.0_f64, 0.0],
//!     vec![1.0, 0.0],
//!     vec![0.0, 1.0],
//!     vec![1.0, 1.0],
//!     vec![10.0, 10.0],
//! ];
//! let mut rng = Rng::new(42);
//! let index = AnnoyIndex::build(data, 3, 2, &mut rng).unwrap();
//! let results = index.query(&[0.1, 0.1], 2);
//! // The two nearest points to (0.1, 0.1) should be (0,0) and (1,0) or (0,1)
//! assert_eq!(results.len(), 2);
//! assert_eq!(results[0].0, 0); // index 0 is (0,0), the closest
//! ```

use scivex_core::{Float, random::Rng};

use crate::error::{MlError, Result};

// ---------------------------------------------------------------------------
// Tree node types
// ---------------------------------------------------------------------------

/// A node in a random projection tree.
enum RpNode<T: Float> {
    /// An internal split node.
    Split {
        /// Random hyperplane normal vector.
        normal: Vec<T>,
        /// Split threshold (dot product offset).
        offset: T,
        /// Index of the left child node.
        left: usize,
        /// Index of the right child node.
        right: usize,
    },
    /// A leaf node containing data point indices.
    Leaf {
        /// Indices into the original dataset.
        indices: Vec<usize>,
    },
}

/// A single random projection tree.
struct RpTree<T: Float> {
    nodes: Vec<RpNode<T>>,
}

// ---------------------------------------------------------------------------
// AnnoyIndex
// ---------------------------------------------------------------------------

/// An approximate nearest neighbor index using random projection trees.
///
/// Multiple trees are built, each splitting the data along random hyperplanes.
/// At query time, all trees are traversed to collect candidate points, then
/// exact distances are computed to return the top-k nearest neighbors.
pub struct AnnoyIndex<T: Float> {
    trees: Vec<RpTree<T>>,
    data: Vec<Vec<T>>,
    dim: usize,
}

impl<T: Float> AnnoyIndex<T> {
    /// Build an `AnnoyIndex` from the given data points.
    ///
    /// # Arguments
    ///
    /// * `data` - Vector of data points, each a `Vec<T>` of the same dimension.
    /// * `n_trees` - Number of random projection trees to build.
    /// * `max_leaf_size` - Maximum number of points in a leaf node before
    ///   splitting stops.
    /// * `rng` - Random number generator.
    ///
    /// # Errors
    ///
    /// Returns [`MlError::EmptyInput`] if `data` is empty, and
    /// [`MlError::InvalidParameter`] if `n_trees` is 0 or dimensions are
    /// inconsistent.
    pub fn build(
        data: Vec<Vec<T>>,
        n_trees: usize,
        max_leaf_size: usize,
        rng: &mut Rng,
    ) -> Result<Self> {
        if data.is_empty() {
            return Err(MlError::EmptyInput);
        }
        if n_trees == 0 {
            return Err(MlError::InvalidParameter {
                name: "n_trees",
                reason: "must be at least 1",
            });
        }
        let dim = data[0].len();
        if dim == 0 {
            return Err(MlError::InvalidParameter {
                name: "data",
                reason: "data points must have at least one dimension",
            });
        }
        for (i, pt) in data.iter().enumerate() {
            if pt.len() != dim {
                return Err(MlError::DimensionMismatch {
                    expected: dim,
                    got: pt.len(),
                });
            }
            let _ = i; // suppress unused warning
        }

        let max_leaf = if max_leaf_size == 0 { 1 } else { max_leaf_size };
        let indices: Vec<usize> = (0..data.len()).collect();

        let mut trees = Vec::with_capacity(n_trees);
        for _ in 0..n_trees {
            let tree = Self::build_tree(&data, &indices, dim, max_leaf, rng);
            trees.push(tree);
        }

        Ok(Self { trees, data, dim })
    }

    /// Query the index for the `k` approximate nearest neighbors of `point`.
    ///
    /// Returns a vector of `(index, distance)` pairs sorted by ascending
    /// Euclidean distance. If fewer than `k` unique candidates are found
    /// across all trees, fewer results may be returned.
    ///
    /// # Panics
    ///
    /// This method does not panic. If `point` has the wrong dimension, an
    /// empty result is returned.
    pub fn query(&self, point: &[T], k: usize) -> Vec<(usize, T)> {
        if point.len() != self.dim || k == 0 {
            return Vec::new();
        }

        // Collect candidate indices from all trees.
        let mut candidates = Vec::new();
        for tree in &self.trees {
            Self::search_tree(tree, point, &mut candidates);
        }

        // De-duplicate.
        candidates.sort_unstable();
        candidates.dedup();

        // Compute exact distances and return top-k.
        let mut scored: Vec<(usize, T)> = candidates
            .into_iter()
            .map(|idx| (idx, euclidean_dist(point, &self.data[idx])))
            .collect();

        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }

    /// Number of data points in the index.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Number of trees in the index.
    pub fn n_trees(&self) -> usize {
        self.trees.len()
    }

    /// Dimensionality of the indexed vectors.
    pub fn dim(&self) -> usize {
        self.dim
    }

    // ── private helpers ──

    fn build_tree(
        data: &[Vec<T>],
        indices: &[usize],
        dim: usize,
        max_leaf_size: usize,
        rng: &mut Rng,
    ) -> RpTree<T> {
        let mut nodes: Vec<RpNode<T>> = Vec::new();
        Self::build_node(data, indices, dim, max_leaf_size, rng, &mut nodes);
        RpTree { nodes }
    }

    fn build_node(
        data: &[Vec<T>],
        indices: &[usize],
        dim: usize,
        max_leaf_size: usize,
        rng: &mut Rng,
        nodes: &mut Vec<RpNode<T>>,
    ) -> usize {
        if indices.len() <= max_leaf_size {
            let idx = nodes.len();
            nodes.push(RpNode::Leaf {
                indices: indices.to_vec(),
            });
            return idx;
        }

        // Pick two distinct random points to define the splitting hyperplane.
        let n = indices.len();
        let a_idx = (rng.next_u64() as usize) % n;
        let mut b_idx = (rng.next_u64() as usize) % n;
        if b_idx == a_idx {
            b_idx = (a_idx + 1) % n;
        }

        let a = &data[indices[a_idx]];
        let b = &data[indices[b_idx]];

        // Normal = b - a (direction from a to b).
        let normal: Vec<T> = (0..dim).map(|d| b[d] - a[d]).collect();

        // Check for degenerate case (identical points).
        let norm_sq: T = normal
            .iter()
            .copied()
            .map(|v| v * v)
            .fold(T::zero(), |s, v| s + v);
        if norm_sq <= T::epsilon() {
            let idx = nodes.len();
            nodes.push(RpNode::Leaf {
                indices: indices.to_vec(),
            });
            return idx;
        }

        // Midpoint offset: dot(normal, midpoint) where midpoint = (a + b) / 2.
        let two = T::from_usize(2);
        let offset: T = (0..dim)
            .map(|d| normal[d] * ((a[d] + b[d]) / two))
            .fold(T::zero(), |s, v| s + v);

        // Partition indices.
        let mut left_indices = Vec::new();
        let mut right_indices = Vec::new();
        for &idx in indices {
            let proj: T = (0..dim)
                .map(|d| normal[d] * data[idx][d])
                .fold(T::zero(), |s, v| s + v);
            if proj <= offset {
                left_indices.push(idx);
            } else {
                right_indices.push(idx);
            }
        }

        // If partition is degenerate (all on one side), make a leaf.
        if left_indices.is_empty() || right_indices.is_empty() {
            let idx = nodes.len();
            nodes.push(RpNode::Leaf {
                indices: indices.to_vec(),
            });
            return idx;
        }

        // Reserve a slot for this split node, then build children.
        let node_idx = nodes.len();
        nodes.push(RpNode::Leaf {
            indices: Vec::new(),
        }); // placeholder

        let left_child = Self::build_node(data, &left_indices, dim, max_leaf_size, rng, nodes);
        let right_child = Self::build_node(data, &right_indices, dim, max_leaf_size, rng, nodes);

        nodes[node_idx] = RpNode::Split {
            normal,
            offset,
            left: left_child,
            right: right_child,
        };

        node_idx
    }

    fn search_tree(tree: &RpTree<T>, point: &[T], candidates: &mut Vec<usize>) {
        if tree.nodes.is_empty() {
            return;
        }
        Self::search_node(&tree.nodes, 0, point, candidates);
    }

    fn search_node(nodes: &[RpNode<T>], node_idx: usize, point: &[T], candidates: &mut Vec<usize>) {
        match &nodes[node_idx] {
            RpNode::Leaf { indices } => {
                candidates.extend_from_slice(indices);
            }
            RpNode::Split {
                normal,
                offset,
                left,
                right,
            } => {
                let proj: T = normal
                    .iter()
                    .zip(point.iter())
                    .map(|(&n, &p)| n * p)
                    .fold(T::zero(), |s, v| s + v);
                if proj <= *offset {
                    Self::search_node(nodes, *left, point, candidates);
                } else {
                    Self::search_node(nodes, *right, point, candidates);
                }
            }
        }
    }
}

/// Euclidean distance between two points.
fn euclidean_dist<T: Float>(a: &[T], b: &[T]) -> T {
    let sum: T = a
        .iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| {
            let d = ai - bi;
            d * d
        })
        .fold(T::zero(), |s, v| s + v);
    sum.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use scivex_core::random::Rng;

    fn sample_data() -> Vec<Vec<f64>> {
        vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![10.0, 10.0],
            vec![10.0, 11.0],
            vec![11.0, 10.0],
            vec![11.0, 11.0],
        ]
    }

    #[test]
    fn test_build_and_query() {
        let data = sample_data();
        let mut rng = Rng::new(42);
        let index = AnnoyIndex::build(data, 5, 2, &mut rng).unwrap();

        assert_eq!(index.len(), 8);
        assert_eq!(index.dim(), 2);
        assert_eq!(index.n_trees(), 5);

        // Query near origin — should find points near (0,0)
        let results = index.query(&[0.1, 0.1], 3);
        assert!(!results.is_empty());
        assert!(results.len() <= 3);
        // Closest should be index 0 = (0,0)
        assert_eq!(results[0].0, 0);
    }

    #[test]
    fn test_query_near_cluster() {
        let data = sample_data();
        let mut rng = Rng::new(123);
        let index = AnnoyIndex::build(data, 10, 2, &mut rng).unwrap();

        // Query near the (10,10) cluster
        let results = index.query(&[10.5, 10.5], 4);
        assert!(!results.is_empty());
        // All results should be from the far cluster (indices 4-7)
        for &(idx, _) in &results {
            assert!(idx >= 4, "expected far-cluster index >= 4, got {idx}");
        }
    }

    #[test]
    fn test_query_k_larger_than_data() {
        let data = vec![vec![1.0_f64, 2.0], vec![3.0, 4.0]];
        let mut rng = Rng::new(42);
        let index = AnnoyIndex::build(data, 2, 1, &mut rng).unwrap();
        let results = index.query(&[1.0, 2.0], 100);
        // Should return at most 2 results (the entire dataset)
        assert!(results.len() <= 2);
    }

    #[test]
    fn test_empty_data() {
        let data: Vec<Vec<f64>> = vec![];
        let mut rng = Rng::new(42);
        assert!(AnnoyIndex::build(data, 3, 2, &mut rng).is_err());
    }

    #[test]
    fn test_zero_trees() {
        let data = vec![vec![1.0_f64]];
        let mut rng = Rng::new(42);
        assert!(AnnoyIndex::build(data, 0, 2, &mut rng).is_err());
    }

    #[test]
    fn test_wrong_query_dim() {
        let data = vec![vec![1.0_f64, 2.0], vec![3.0, 4.0]];
        let mut rng = Rng::new(42);
        let index = AnnoyIndex::build(data, 2, 1, &mut rng).unwrap();
        // Wrong dimension query returns empty
        let results = index.query(&[1.0, 2.0, 3.0], 1);
        assert!(results.is_empty());
    }

    #[test]
    fn test_distances_sorted() {
        let data = sample_data();
        let mut rng = Rng::new(42);
        let index = AnnoyIndex::build(data, 10, 2, &mut rng).unwrap();
        let results = index.query(&[5.0, 5.0], 8);
        for window in results.windows(2) {
            assert!(
                window[0].1 <= window[1].1,
                "results should be sorted by distance"
            );
        }
    }

    #[test]
    fn test_single_point() {
        let data = vec![vec![42.0_f64, 7.0]];
        let mut rng = Rng::new(42);
        let index = AnnoyIndex::build(data, 3, 1, &mut rng).unwrap();
        let results = index.query(&[42.0, 7.0], 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 0);
        assert!(results[0].1 < 1e-10);
    }

    #[test]
    fn test_euclidean_dist() {
        let d = euclidean_dist(&[0.0_f64, 0.0], &[3.0, 4.0]);
        assert!((d - 5.0).abs() < 1e-10);
    }
}
