//! Spatial data structures: KD-tree, ball tree.
//!
//! Provides efficient nearest-neighbor and range queries in multi-dimensional
//! space.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::dtype::Float;
use crate::error::{CoreError, Result};
use crate::tensor::Tensor;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Max-heap entry for KNN search. Stores squared distance and original index.
struct HeapEntry<T> {
    sq_dist: T,
    index: usize,
}

impl<T: Float> PartialEq for HeapEntry<T> {
    fn eq(&self, other: &Self) -> bool {
        self.sq_dist.to_f64() == other.sq_dist.to_f64()
    }
}

impl<T: Float> Eq for HeapEntry<T> {}

impl<T: Float> PartialOrd for HeapEntry<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Float> Ord for HeapEntry<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.sq_dist
            .to_f64()
            .partial_cmp(&other.sq_dist.to_f64())
            .unwrap_or(Ordering::Equal)
    }
}

// ---------------------------------------------------------------------------
// KD-tree node
// ---------------------------------------------------------------------------

/// Leaf size threshold — leaves hold up to this many points.
const LEAF_SIZE: usize = 10;

#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
enum KdNode {
    Leaf {
        indices: Vec<usize>,
    },
    Internal {
        split_dim: usize,
        split_value: f64,
        left: Box<KdNode>,
        right: Box<KdNode>,
    },
}

// ---------------------------------------------------------------------------
// KdTree
// ---------------------------------------------------------------------------

/// A KD-tree for efficient spatial queries.
///
/// Points are stored as a flat `Vec<T>` with known dimensionality. The tree
/// structure is a recursive enum of internal split nodes and leaf nodes.
///
/// # Examples
///
/// ```
/// use scivex_core::spatial::KdTree;
///
/// let points: Vec<Vec<f64>> = vec![
///     vec![0.0, 0.0],
///     vec![1.0, 0.0],
///     vec![0.0, 1.0],
///     vec![1.0, 1.0],
/// ];
/// let refs: Vec<&[f64]> = points.iter().map(|p| p.as_slice()).collect();
/// let tree = KdTree::build(&refs).unwrap();
///
/// let (indices, dists) = tree.query(&[0.1, 0.1], 1).unwrap();
/// assert_eq!(indices[0], 0);
/// ```
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct KdTree<T: Float> {
    /// Flat storage: point `i` occupies `data[i*dim .. (i+1)*dim]`.
    data: Vec<T>,
    /// Dimensionality of each point.
    dim: usize,
    /// Number of points.
    n_points: usize,
    /// Root of the tree.
    root: KdNode,
}

impl<T: Float> KdTree<T> {
    /// Build a KD-tree from a set of points.
    ///
    /// `points` is a slice of point slices, each of length `dim`.
    ///
    /// # Errors
    ///
    /// Returns `CoreError::InvalidArgument` if `points` is empty or if the
    /// point slices have inconsistent lengths.
    pub fn build(points: &[&[T]]) -> Result<Self> {
        if points.is_empty() {
            return Err(CoreError::InvalidArgument {
                reason: "cannot build KD-tree from empty point set",
            });
        }
        let dim = points[0].len();
        if dim == 0 {
            return Err(CoreError::InvalidArgument {
                reason: "point dimensionality must be at least 1",
            });
        }
        for (i, p) in points.iter().enumerate() {
            if p.len() != dim {
                return Err(CoreError::InvalidArgument {
                    reason: "all points must have the same dimensionality",
                });
            }
            let _ = i; // suppress unused warning
        }

        let n_points = points.len();
        let mut data = Vec::with_capacity(n_points * dim);
        for p in points {
            data.extend_from_slice(p);
        }

        let indices: Vec<usize> = (0..n_points).collect();
        let root = Self::build_recursive(&data, dim, indices);

        Ok(Self {
            data,
            dim,
            n_points,
            root,
        })
    }

    /// Build from a 2D tensor where each row is a point.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor is not 2-dimensional or is empty.
    pub fn from_tensor(tensor: &Tensor<T>) -> Result<Self> {
        let shape = tensor.shape();
        if shape.len() != 2 {
            return Err(CoreError::InvalidArgument {
                reason: "tensor must be 2-dimensional (rows = points, cols = dims)",
            });
        }
        let n = shape[0];
        let dim = shape[1];
        if n == 0 {
            return Err(CoreError::InvalidArgument {
                reason: "cannot build KD-tree from empty point set",
            });
        }

        let slice = tensor.as_slice();
        let refs: Vec<&[T]> = (0..n).map(|i| &slice[i * dim..(i + 1) * dim]).collect();
        Self::build(&refs)
    }

    /// Find the k nearest neighbors to `query`.
    ///
    /// Returns `(indices, distances)` sorted by distance (ascending).
    ///
    /// # Errors
    ///
    /// Returns an error if `k == 0` or the query dimension does not match.
    pub fn query(&self, query: &[T], k: usize) -> Result<(Vec<usize>, Vec<T>)> {
        if k == 0 {
            return Err(CoreError::InvalidArgument {
                reason: "k must be at least 1",
            });
        }
        if query.len() != self.dim {
            return Err(CoreError::InvalidArgument {
                reason: "query dimensionality does not match tree",
            });
        }
        let k = k.min(self.n_points);

        let mut heap: BinaryHeap<HeapEntry<T>> = BinaryHeap::new();
        self.knn_recursive(&self.root, query, k, &mut heap);

        // Drain the max-heap into a vec and reverse so smallest distance first.
        let mut results: Vec<(usize, T)> = heap
            .into_sorted_vec()
            .into_iter()
            .map(|e| (e.index, e.sq_dist.sqrt()))
            .collect();
        results.sort_by(|a, b| {
            a.1.to_f64()
                .partial_cmp(&b.1.to_f64())
                .unwrap_or(Ordering::Equal)
        });
        let indices = results.iter().map(|(i, _)| *i).collect();
        let dists = results.iter().map(|(_, d)| *d).collect();
        Ok((indices, dists))
    }

    /// Find all points within distance `radius` of `query`.
    ///
    /// Returns `(indices, distances)`.
    ///
    /// # Errors
    ///
    /// Returns an error if the query dimension does not match.
    pub fn query_radius(&self, query: &[T], radius: T) -> Result<(Vec<usize>, Vec<T>)> {
        if query.len() != self.dim {
            return Err(CoreError::InvalidArgument {
                reason: "query dimensionality does not match tree",
            });
        }
        let sq_radius = radius * radius;
        let mut results: Vec<(usize, T)> = Vec::new();
        self.range_recursive(&self.root, query, sq_radius, &mut results);

        // Sort by distance.
        results.sort_by(|a, b| {
            a.1.to_f64()
                .partial_cmp(&b.1.to_f64())
                .unwrap_or(Ordering::Equal)
        });
        let indices = results.iter().map(|(i, _)| *i).collect();
        let dists = results.into_iter().map(|(_, d)| d.sqrt()).collect();
        Ok((indices, dists))
    }

    /// Find all pairs of points within distance `r` of each other.
    ///
    /// Returns pairs as `(i, j)` with `i < j`.
    pub fn query_pairs(&self, r: T) -> Vec<(usize, usize)> {
        let sq_r = r * r;
        let mut pairs = Vec::new();
        for i in 0..self.n_points {
            let point = &self.data[i * self.dim..(i + 1) * self.dim];
            let mut neighbors: Vec<(usize, T)> = Vec::new();
            self.range_recursive(&self.root, point, sq_r, &mut neighbors);
            for (j, _) in neighbors {
                if i < j {
                    pairs.push((i, j));
                }
            }
        }
        pairs.sort_unstable();
        pairs.dedup();
        pairs
    }

    /// Return the number of points.
    #[inline]
    pub fn len(&self) -> usize {
        self.n_points
    }

    /// Whether the tree contains no points.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.n_points == 0
    }

    /// Return the dimensionality.
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    // -----------------------------------------------------------------------
    // Internal build
    // -----------------------------------------------------------------------

    fn build_recursive(data: &[T], dim: usize, mut indices: Vec<usize>) -> KdNode {
        if indices.len() <= LEAF_SIZE {
            return KdNode::Leaf { indices };
        }

        // Find dimension with widest spread.
        let split_dim = Self::widest_spread_dim(data, dim, &indices);

        // Sort indices by the split dimension.
        indices.sort_by(|&a, &b| {
            let va = data[a * dim + split_dim].to_f64();
            let vb = data[b * dim + split_dim].to_f64();
            va.partial_cmp(&vb).unwrap_or(Ordering::Equal)
        });

        let median_idx = indices.len() / 2;
        let split_value = data[indices[median_idx] * dim + split_dim].to_f64();

        let right_indices = indices.split_off(median_idx);
        let left_indices = indices;

        let left = Box::new(Self::build_recursive(data, dim, left_indices));
        let right = Box::new(Self::build_recursive(data, dim, right_indices));

        KdNode::Internal {
            split_dim,
            split_value,
            left,
            right,
        }
    }

    fn widest_spread_dim(data: &[T], dim: usize, indices: &[usize]) -> usize {
        let mut best_dim = 0;
        let mut best_spread = f64::NEG_INFINITY;
        for d in 0..dim {
            let mut lo = f64::INFINITY;
            let mut hi = f64::NEG_INFINITY;
            for &idx in indices {
                let v = data[idx * dim + d].to_f64();
                if v < lo {
                    lo = v;
                }
                if v > hi {
                    hi = v;
                }
            }
            let spread = hi - lo;
            if spread > best_spread {
                best_spread = spread;
                best_dim = d;
            }
        }
        best_dim
    }

    // -----------------------------------------------------------------------
    // KNN search
    // -----------------------------------------------------------------------

    fn squared_distance(&self, a: &[T], b_idx: usize) -> T {
        let mut sum = T::zero();
        let offset = b_idx * self.dim;
        for (d, a_val) in a.iter().enumerate().take(self.dim) {
            let diff = *a_val - self.data[offset + d];
            sum += diff * diff;
        }
        sum
    }

    fn knn_recursive(
        &self,
        node: &KdNode,
        query: &[T],
        k: usize,
        heap: &mut BinaryHeap<HeapEntry<T>>,
    ) {
        match node {
            KdNode::Leaf { indices } => {
                for &idx in indices {
                    let sq_dist = self.squared_distance(query, idx);
                    if heap.len() < k {
                        heap.push(HeapEntry {
                            sq_dist,
                            index: idx,
                        });
                    } else if heap
                        .peek()
                        .is_some_and(|worst| sq_dist.to_f64() < worst.sq_dist.to_f64())
                    {
                        heap.pop();
                        heap.push(HeapEntry {
                            sq_dist,
                            index: idx,
                        });
                    }
                }
            }
            KdNode::Internal {
                split_dim,
                split_value,
                left,
                right,
            } => {
                let query_val = query[*split_dim].to_f64();
                let diff = query_val - split_value;

                let (first, second) = if diff <= 0.0 {
                    (left, right)
                } else {
                    (right, left)
                };

                self.knn_recursive(first, query, k, heap);

                // Prune: only visit second child if the split plane is closer
                // than the worst candidate.
                let should_visit =
                    heap.len() < k || diff * diff < heap.peek().unwrap().sq_dist.to_f64();
                if should_visit {
                    self.knn_recursive(second, query, k, heap);
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Range search
    // -----------------------------------------------------------------------

    fn range_recursive(
        &self,
        node: &KdNode,
        query: &[T],
        sq_radius: T,
        results: &mut Vec<(usize, T)>,
    ) {
        match node {
            KdNode::Leaf { indices } => {
                for &idx in indices {
                    let sq_dist = self.squared_distance(query, idx);
                    if sq_dist.to_f64() <= sq_radius.to_f64() {
                        results.push((idx, sq_dist));
                    }
                }
            }
            KdNode::Internal {
                split_dim,
                split_value,
                left,
                right,
            } => {
                let query_val = query[*split_dim].to_f64();
                let diff = query_val - split_value;
                let sq_diff = diff * diff;

                let (first, second) = if diff <= 0.0 {
                    (left, right)
                } else {
                    (right, left)
                };

                self.range_recursive(first, query, sq_radius, results);

                if sq_diff <= sq_radius.to_f64() {
                    self.range_recursive(second, query, sq_radius, results);
                }
            }
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kd_tree_knn_exact_match() {
        let pts: [[f64; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let refs: Vec<&[f64]> = pts.iter().map(<[f64; 2]>::as_slice).collect();
        let tree = KdTree::build(&refs).unwrap();

        let (indices, dists) = tree.query(&[0.0, 0.0], 1).unwrap();
        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0], 0);
        assert!(dists[0].abs() < 1e-12);
    }

    #[test]
    fn test_kd_tree_knn_k3_sorted() {
        let pts: [[f64; 2]; 5] = [[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [5.0, 0.0], [10.0, 0.0]];
        let refs: Vec<&[f64]> = pts.iter().map(<[f64; 2]>::as_slice).collect();
        let tree = KdTree::build(&refs).unwrap();

        let (indices, dists) = tree.query(&[0.5, 0.0], 3).unwrap();
        assert_eq!(indices.len(), 3);
        // Closest: pt0 (0.5), pt1 (0.5), pt2 (2.5)
        assert!(dists[0] <= dists[1]);
        assert!(dists[1] <= dists[2]);
        // The two closest are pts 0 and 1 (both at distance 0.5)
        assert!((dists[0] - 0.5).abs() < 1e-12);
        assert!((dists[1] - 0.5).abs() < 1e-12);
        assert!((dists[2] - 2.5).abs() < 1e-12);
    }

    #[test]
    fn test_kd_tree_range_query() {
        let pts: [[f64; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [10.0, 0.0]];
        let refs: Vec<&[f64]> = pts.iter().map(<[f64; 2]>::as_slice).collect();
        let tree = KdTree::build(&refs).unwrap();

        let (indices, _dists) = tree.query_radius(&[0.0, 0.0], 1.5).unwrap();
        // Should find pts 0 and 1 (distances 0 and 1)
        assert_eq!(indices.len(), 2);
        assert!(indices.contains(&0));
        assert!(indices.contains(&1));
    }

    #[test]
    fn test_kd_tree_query_pairs() {
        let pts: [[f64; 2]; 3] = [[0.0, 0.0], [0.5, 0.0], [10.0, 0.0]];
        let refs: Vec<&[f64]> = pts.iter().map(<[f64; 2]>::as_slice).collect();
        let tree = KdTree::build(&refs).unwrap();

        let pairs = tree.query_pairs(1.0);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0], (0, 1));
    }

    #[test]
    fn test_kd_tree_from_tensor() {
        let data = vec![0.0_f64, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let tensor = Tensor::from_vec(data, vec![4, 2]).unwrap();
        let tree = KdTree::from_tensor(&tensor).unwrap();
        assert_eq!(tree.len(), 4);
        assert_eq!(tree.dim(), 2);

        let (indices, _) = tree.query(&[0.0, 0.0], 1).unwrap();
        assert_eq!(indices[0], 0);
    }

    #[test]
    fn test_kd_tree_high_dimensional() {
        // 5-dimensional points
        let pts: [[f64; 5]; 3] = [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0, 2.0],
        ];
        let refs: Vec<&[f64]> = pts.iter().map(<[f64; 5]>::as_slice).collect();
        let tree = KdTree::build(&refs).unwrap();

        let (indices, dists) = tree.query(&[0.0, 0.0, 0.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(indices[0], 0);
        assert!(dists[0].abs() < 1e-12);

        // Distance from origin to (1,1,1,1,1) = sqrt(5)
        let (indices, dists) = tree.query(&[0.0, 0.0, 0.0, 0.0, 0.0], 2).unwrap();
        assert_eq!(indices.len(), 2);
        assert!((dists[1] - 5.0_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn test_kd_tree_single_point() {
        let pts: [[f64; 2]; 1] = [[42.0_f64, 7.0]];
        let refs: Vec<&[f64]> = pts.iter().map(<[f64; 2]>::as_slice).collect();
        let tree = KdTree::build(&refs).unwrap();
        assert_eq!(tree.len(), 1);

        let (indices, dists) = tree.query(&[42.0, 7.0], 1).unwrap();
        assert_eq!(indices[0], 0);
        assert!(dists[0].abs() < 1e-12);
    }

    #[test]
    fn test_kd_tree_error_empty() {
        let refs: Vec<&[f64]> = vec![];
        let result = KdTree::build(&refs);
        assert!(result.is_err());
    }

    #[test]
    fn test_kd_tree_error_k_zero() {
        let pts: [[f64; 2]; 1] = [[0.0_f64, 0.0]];
        let refs: Vec<&[f64]> = pts.iter().map(<[f64; 2]>::as_slice).collect();
        let tree = KdTree::build(&refs).unwrap();

        let result = tree.query(&[0.0, 0.0], 0);
        assert!(result.is_err());
    }
}
