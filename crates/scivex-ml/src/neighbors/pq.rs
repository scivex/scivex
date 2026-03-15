use std::collections::BinaryHeap;

use scivex_core::{Float, Tensor};

use crate::cluster::KMeans;
use crate::error::{MlError, Result};

use super::brute_force::NearestNeighborResult;
use super::distance::{DistanceMetric, compute_distance};

/// Product Quantization for compressed approximate nearest-neighbour search.
///
/// Splits each vector into `n_subvectors` sub-spaces, learns a codebook per
/// sub-space using K-Means, and represents each vector as a sequence of
/// centroid indices (codes). Search uses Asymmetric Distance Computation (ADC).
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct ProductQuantizer<T: Float> {
    dim: usize,
    n_subvectors: usize,
    n_centroids: usize,
    sub_dim: usize,
    codebooks: Option<Vec<Vec<T>>>,
    metric: DistanceMetric,
    seed: u64,
}

impl<T: Float> ProductQuantizer<T> {
    /// Create a new product quantizer.
    ///
    /// - `dim`: full vector dimensionality
    /// - `n_subvectors`: number of sub-spaces (must evenly divide `dim`)
    /// - `n_centroids`: centroids per sub-space (typically 256, max 256 for u8 codes)
    pub fn new(dim: usize, n_subvectors: usize, n_centroids: usize) -> Result<Self> {
        if dim == 0 {
            return Err(MlError::InvalidParameter {
                name: "dim",
                reason: "must be at least 1",
            });
        }
        if n_subvectors == 0 {
            return Err(MlError::InvalidParameter {
                name: "n_subvectors",
                reason: "must be at least 1",
            });
        }
        #[allow(clippy::manual_is_multiple_of)]
        if dim % n_subvectors != 0 {
            return Err(MlError::InvalidParameter {
                name: "n_subvectors",
                reason: "dim must be divisible by n_subvectors",
            });
        }
        if n_centroids == 0 || n_centroids > 256 {
            return Err(MlError::InvalidParameter {
                name: "n_centroids",
                reason: "must be between 1 and 256",
            });
        }
        Ok(Self {
            dim,
            n_subvectors,
            n_centroids,
            sub_dim: dim / n_subvectors,
            codebooks: None,
            metric: DistanceMetric::L2,
            seed: 42,
        })
    }

    /// Set the random seed for K-Means training.
    pub fn set_seed(&mut self, seed: u64) -> &mut Self {
        self.seed = seed;
        self
    }

    /// Set the distance metric (default: L2).
    pub fn set_metric(&mut self, metric: DistanceMetric) -> &mut Self {
        self.metric = metric;
        self
    }

    /// Train codebooks on a set of vectors using K-Means.
    ///
    /// `vectors` must be a 2-D tensor of shape `[n_samples, dim]`.
    pub fn train(&mut self, vectors: &Tensor<T>) -> Result<()> {
        let s = vectors.shape();
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
        if n < self.n_centroids {
            return Err(MlError::InvalidParameter {
                name: "n_samples",
                reason: "must have at least n_centroids samples",
            });
        }

        let data = vectors.as_slice();
        let mut codebooks = Vec::with_capacity(self.n_subvectors);

        for sv in 0..self.n_subvectors {
            // Extract sub-vectors for this sub-space
            let offset = sv * self.sub_dim;
            let mut sub_data = Vec::with_capacity(n * self.sub_dim);
            for i in 0..n {
                let row_start = i * d + offset;
                sub_data.extend_from_slice(&data[row_start..row_start + self.sub_dim]);
            }

            let sub_tensor = Tensor::from_vec(sub_data, vec![n, self.sub_dim])?;

            let mut km = KMeans::new(
                self.n_centroids,
                100, // max_iter
                T::from_f64(1e-6),
                1, // n_init
                self.seed.wrapping_add(sv as u64),
            )?;
            km.fit(&sub_tensor)?;

            let centroids = km.centroids().ok_or(MlError::NotFitted)?.to_vec();
            codebooks.push(centroids);
        }

        self.codebooks = Some(codebooks);
        Ok(())
    }

    /// Encode vectors into PQ codes.
    ///
    /// Returns a vector of codes, one per input vector. Each code is a
    /// `Vec<u8>` of length `n_subvectors`.
    pub fn encode(&self, vectors: &Tensor<T>) -> Result<Vec<Vec<u8>>> {
        let codebooks = self.codebooks.as_ref().ok_or(MlError::NotFitted)?;
        let s = vectors.shape();
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

        let data = vectors.as_slice();
        let mut codes = Vec::with_capacity(n);

        for i in 0..n {
            let mut code = Vec::with_capacity(self.n_subvectors);
            for (sv, cb) in codebooks.iter().enumerate() {
                let offset = sv * self.sub_dim;
                let sub_vec = &data[i * d + offset..i * d + offset + self.sub_dim];
                let nearest = self.nearest_centroid(sub_vec, cb);
                #[allow(clippy::cast_possible_truncation)]
                code.push(nearest as u8);
            }
            codes.push(code);
        }

        Ok(codes)
    }

    /// Search for the `k` nearest neighbours using Asymmetric Distance Computation.
    ///
    /// `query` is a full-dimensional vector. `codes` are the PQ-encoded database vectors.
    pub fn search(
        &self,
        query: &[T],
        codes: &[Vec<u8>],
        k: usize,
    ) -> Result<NearestNeighborResult<T>> {
        let codebooks = self.codebooks.as_ref().ok_or(MlError::NotFitted)?;
        if query.len() != self.dim {
            return Err(MlError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        if codes.is_empty() {
            return Err(MlError::EmptyInput);
        }
        if k == 0 {
            return Err(MlError::InvalidParameter {
                name: "k",
                reason: "must be at least 1",
            });
        }

        // Precompute distance table: [n_subvectors][n_centroids]
        let mut dist_table = vec![vec![T::zero(); self.n_centroids]; self.n_subvectors];
        for sv in 0..self.n_subvectors {
            let offset = sv * self.sub_dim;
            let query_sub = &query[offset..offset + self.sub_dim];
            let cb = &codebooks[sv];
            for c in 0..self.n_centroids {
                let centroid = &cb[c * self.sub_dim..(c + 1) * self.sub_dim];
                dist_table[sv][c] = compute_distance(query_sub, centroid, self.metric);
            }
        }

        let k = k.min(codes.len());
        let mut heap: BinaryHeap<PqDistIdx<T>> = BinaryHeap::with_capacity(k + 1);

        for (i, code) in codes.iter().enumerate() {
            let mut dist = T::zero();
            for (sv, &c) in code.iter().enumerate() {
                dist += dist_table[sv][c as usize];
            }

            if heap.len() < k {
                heap.push(PqDistIdx { dist, idx: i });
            } else {
                let should_insert = heap.peek().is_some_and(|top| dist < top.dist);
                if should_insert {
                    heap.pop();
                    heap.push(PqDistIdx { dist, idx: i });
                }
            }
        }

        let mut results: Vec<PqDistIdx<T>> = heap.into_vec();
        results.sort_by(|a, b| {
            a.dist
                .partial_cmp(&b.dist)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let indices = results.iter().map(|r| r.idx).collect();
        let distances = results.iter().map(|r| r.dist).collect();

        Ok(NearestNeighborResult { indices, distances })
    }

    fn nearest_centroid(&self, sub_vec: &[T], codebook: &[T]) -> usize {
        let mut best = 0;
        let mut best_dist = T::infinity();
        for c in 0..self.n_centroids {
            let centroid = &codebook[c * self.sub_dim..(c + 1) * self.sub_dim];
            let d = compute_distance(sub_vec, centroid, self.metric);
            if d < best_dist {
                best_dist = d;
                best = c;
            }
        }
        best
    }
}

#[derive(Clone)]
struct PqDistIdx<T: Float> {
    dist: T,
    idx: usize,
}

impl<T: Float> PartialEq for PqDistIdx<T> {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist
    }
}

impl<T: Float> Eq for PqDistIdx<T> {}

impl<T: Float> PartialOrd for PqDistIdx<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Float> Ord for PqDistIdx<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.dist
            .partial_cmp(&other.dist)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scivex_core::random::Rng;

    fn make_random_points(n: usize, dim: usize, seed: u64) -> Tensor<f64> {
        let mut rng = Rng::new(seed);
        let data: Vec<f64> = (0..n * dim).map(|_| rng.next_f64()).collect();
        Tensor::from_vec(data, vec![n, dim]).unwrap()
    }

    #[test]
    fn test_pq_train_encode() {
        let data = make_random_points(100, 8, 42);
        let mut pq = ProductQuantizer::<f64>::new(8, 4, 16).unwrap();
        pq.train(&data).unwrap();
        let codes = pq.encode(&data).unwrap();
        assert_eq!(codes.len(), 100);
        assert_eq!(codes[0].len(), 4);
        // Each code value should be < n_centroids
        for code in &codes {
            for &c in code {
                assert!((c as usize) < 16);
            }
        }
    }

    #[test]
    fn test_pq_search() {
        let data = make_random_points(200, 8, 42);
        let mut pq = ProductQuantizer::<f64>::new(8, 4, 16).unwrap();
        pq.train(&data).unwrap();
        let codes = pq.encode(&data).unwrap();

        let query_tensor = make_random_points(1, 8, 99);
        let query = query_tensor.as_slice();
        let result = pq.search(query, &codes, 5).unwrap();
        assert_eq!(result.indices.len(), 5);
        // Distances should be non-decreasing
        for w in result.distances.windows(2) {
            assert!(w[0] <= w[1] + 1e-10);
        }
    }

    #[test]
    fn test_pq_invalid_dim() {
        let result = ProductQuantizer::<f64>::new(7, 4, 16);
        assert!(result.is_err());
    }

    #[test]
    fn test_pq_not_trained() {
        let pq = ProductQuantizer::<f64>::new(8, 4, 16).unwrap();
        let data = make_random_points(10, 8, 1);
        let result = pq.encode(&data);
        assert!(result.is_err());
    }
}
