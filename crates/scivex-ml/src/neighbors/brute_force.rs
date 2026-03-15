use std::collections::BinaryHeap;

use scivex_core::{Float, Tensor};

use crate::error::{MlError, Result};

use super::distance::{DistanceMetric, compute_distance};

/// Result of a nearest-neighbour search.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct NearestNeighborResult<T: Float> {
    /// Indices of the nearest neighbours.
    pub indices: Vec<usize>,
    /// Distances to the nearest neighbours (same order as `indices`).
    pub distances: Vec<T>,
}

/// Exact K-nearest neighbour search using brute-force distance computation.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct BruteForceIndex<T: Float> {
    vectors: Vec<T>,
    n_vectors: usize,
    dim: usize,
    metric: DistanceMetric,
}

impl<T: Float> BruteForceIndex<T> {
    /// Create a new empty brute-force index with the given distance metric.
    #[must_use]
    pub fn new(metric: DistanceMetric) -> Self {
        Self {
            vectors: Vec::new(),
            n_vectors: 0,
            dim: 0,
            metric,
        }
    }

    /// Add vectors to the index.
    ///
    /// `vectors` must be a 2-D tensor of shape `[n, dim]`. The dimensionality
    /// must match previously added vectors (or this is the first batch).
    pub fn add(&mut self, vectors: &Tensor<T>) -> Result<()> {
        let s = vectors.shape();
        if s.len() != 2 {
            return Err(MlError::DimensionMismatch {
                expected: 2,
                got: s.len(),
            });
        }
        let n = s[0];
        let d = s[1];
        if n == 0 {
            return Err(MlError::EmptyInput);
        }
        if self.n_vectors > 0 {
            if d != self.dim {
                return Err(MlError::DimensionMismatch {
                    expected: self.dim,
                    got: d,
                });
            }
        } else {
            self.dim = d;
        }
        self.vectors.extend_from_slice(vectors.as_slice());
        self.n_vectors += n;
        Ok(())
    }

    /// Search for the `k` nearest neighbours of `query`.
    pub fn search(&self, query: &[T], k: usize) -> Result<NearestNeighborResult<T>> {
        if self.n_vectors == 0 {
            return Err(MlError::EmptyInput);
        }
        if query.len() != self.dim {
            return Err(MlError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        if k == 0 {
            return Err(MlError::InvalidParameter {
                name: "k",
                reason: "must be at least 1",
            });
        }
        let k = k.min(self.n_vectors);
        let mut heap: BinaryHeap<DistIdx<T>> = BinaryHeap::with_capacity(k + 1);

        for i in 0..self.n_vectors {
            let vec_slice = &self.vectors[i * self.dim..(i + 1) * self.dim];
            let dist = compute_distance(query, vec_slice, self.metric);
            if heap.len() < k {
                heap.push(DistIdx { dist, idx: i });
            } else {
                let should_insert = heap.peek().is_some_and(|top| dist < top.dist);
                if should_insert {
                    heap.pop();
                    heap.push(DistIdx { dist, idx: i });
                }
            }
        }

        let mut results: Vec<DistIdx<T>> = heap.into_sorted_vec();
        results.sort_by(|a, b| {
            a.dist
                .partial_cmp(&b.dist)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let indices = results.iter().map(|r| r.idx).collect();
        let distances = results.iter().map(|r| r.dist).collect();

        Ok(NearestNeighborResult { indices, distances })
    }

    /// Search for the `k` nearest neighbours of each row in `queries`.
    pub fn batch_search(
        &self,
        queries: &Tensor<T>,
        k: usize,
    ) -> Result<Vec<NearestNeighborResult<T>>> {
        let s = queries.shape();
        if s.len() != 2 {
            return Err(MlError::DimensionMismatch {
                expected: 2,
                got: s.len(),
            });
        }
        let n = s[0];
        let d = s[1];
        if d != self.dim {
            return Err(MlError::DimensionMismatch {
                expected: self.dim,
                got: d,
            });
        }
        let data = queries.as_slice();
        let mut results = Vec::with_capacity(n);
        for i in 0..n {
            let query = &data[i * d..(i + 1) * d];
            results.push(self.search(query, k)?);
        }
        Ok(results)
    }

    /// Number of vectors in the index.
    #[must_use]
    pub fn len(&self) -> usize {
        self.n_vectors
    }

    /// Whether the index is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.n_vectors == 0
    }

    /// Dimensionality of the vectors.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }
}

/// Helper for the max-heap: ordered by distance descending so the largest
/// distance sits at the top.
#[derive(Clone)]
struct DistIdx<T: Float> {
    dist: T,
    idx: usize,
}

impl<T: Float> PartialEq for DistIdx<T> {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist
    }
}

impl<T: Float> Eq for DistIdx<T> {}

impl<T: Float> PartialOrd for DistIdx<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Float> Ord for DistIdx<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.dist
            .partial_cmp(&other.dist)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_points() -> Tensor<f64> {
        // 4 points in 2D
        Tensor::from_vec(vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0], vec![4, 2]).unwrap()
    }

    #[test]
    fn test_brute_force_basic() {
        let mut idx = BruteForceIndex::new(DistanceMetric::L2);
        idx.add(&make_points()).unwrap();
        let result = idx.search(&[0.1, 0.1], 1).unwrap();
        assert_eq!(result.indices, vec![0]);
    }

    #[test]
    fn test_brute_force_k_neighbors() {
        let mut idx = BruteForceIndex::new(DistanceMetric::L2);
        idx.add(&make_points()).unwrap();
        let result = idx.search(&[0.1, 0.1], 3).unwrap();
        assert_eq!(result.indices.len(), 3);
        // Nearest should be (0,0)
        assert_eq!(result.indices[0], 0);
    }

    #[test]
    fn test_brute_force_batch() {
        let mut idx = BruteForceIndex::new(DistanceMetric::L2);
        idx.add(&make_points()).unwrap();
        let queries = Tensor::from_vec(vec![0.1_f64, 0.1, 0.9, 0.9], vec![2, 2]).unwrap();
        let results = idx.batch_search(&queries, 1).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].indices[0], 0);
        assert_eq!(results[1].indices[0], 3);
    }

    #[test]
    fn test_brute_force_add_incremental() {
        let mut idx = BruteForceIndex::new(DistanceMetric::L2);
        let first = Tensor::from_vec(vec![0.0_f64, 0.0, 1.0, 0.0], vec![2, 2]).unwrap();
        let second = Tensor::from_vec(vec![10.0_f64, 10.0], vec![1, 2]).unwrap();
        idx.add(&first).unwrap();
        idx.add(&second).unwrap();
        assert_eq!(idx.len(), 3);
        let result = idx.search(&[9.9, 9.9], 1).unwrap();
        assert_eq!(result.indices[0], 2);
    }

    #[test]
    fn test_brute_force_empty() {
        let idx: BruteForceIndex<f64> = BruteForceIndex::new(DistanceMetric::L2);
        let result = idx.search(&[1.0, 2.0], 1);
        assert!(result.is_err());
    }
}
