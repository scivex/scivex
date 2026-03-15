use std::collections::BinaryHeap;

use scivex_core::{Float, Tensor, random::Rng};

use crate::error::{MlError, Result};

use super::brute_force::NearestNeighborResult;
use super::distance::{DistanceMetric, compute_distance};

/// HNSW (Hierarchical Navigable Small World) approximate nearest neighbour index.
#[derive(Debug, Clone)]
pub struct HnswIndex<T: Float> {
    nodes: Vec<HnswNode<T>>,
    dim: usize,
    metric: DistanceMetric,
    m: usize,
    m_max0: usize,
    ef_construction: usize,
    ef_search: usize,
    max_level: usize,
    entry_point: Option<usize>,
    level_mult: f64,
    seed: u64,
    rng_counter: u64,
}

#[derive(Debug, Clone)]
struct HnswNode<T: Float> {
    vector: Vec<T>,
    connections: Vec<Vec<usize>>,
    level: usize,
}

impl<T: Float> HnswIndex<T> {
    /// Create a new HNSW index for vectors of the given dimensionality.
    pub fn new(dim: usize, metric: DistanceMetric) -> Result<Self> {
        if dim == 0 {
            return Err(MlError::InvalidParameter {
                name: "dim",
                reason: "must be at least 1",
            });
        }
        let m = 16;
        let level_mult = 1.0 / (m as f64).ln();
        Ok(Self {
            nodes: Vec::new(),
            dim,
            metric,
            m,
            m_max0: 2 * m,
            ef_construction: 200,
            ef_search: 50,
            max_level: 0,
            entry_point: None,
            level_mult,
            seed: 42,
            rng_counter: 0,
        })
    }

    /// Set the maximum number of connections per layer.
    pub fn set_m(&mut self, m: usize) -> &mut Self {
        self.m = m;
        self.m_max0 = 2 * m;
        self.level_mult = 1.0 / (m as f64).ln();
        self
    }

    /// Set the search width used during index construction.
    pub fn set_ef_construction(&mut self, ef: usize) -> &mut Self {
        self.ef_construction = ef;
        self
    }

    /// Set the search width used during queries.
    pub fn set_ef_search(&mut self, ef: usize) -> &mut Self {
        self.ef_search = ef;
        self
    }

    /// Set the random seed (resets the internal RNG).
    pub fn set_seed(&mut self, seed: u64) -> &mut Self {
        self.seed = seed;
        self.rng_counter = 0;
        self
    }

    /// Add a batch of vectors to the index.
    ///
    /// `vectors` must be a 2-D tensor of shape `[n, dim]`.
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
        if d != self.dim {
            return Err(MlError::DimensionMismatch {
                expected: self.dim,
                got: d,
            });
        }
        if n == 0 {
            return Err(MlError::EmptyInput);
        }
        let data = vectors.as_slice();
        for i in 0..n {
            let vec_data = data[i * d..(i + 1) * d].to_vec();
            self.add_single(&vec_data)?;
        }
        Ok(())
    }

    /// Add a single vector and return its node id.
    pub fn add_single(&mut self, vector: &[T]) -> Result<usize> {
        if vector.len() != self.dim {
            return Err(MlError::DimensionMismatch {
                expected: self.dim,
                got: vector.len(),
            });
        }
        let node_level = self.random_level();
        let node_id = self.nodes.len();

        let mut connections = Vec::with_capacity(node_level + 1);
        for _ in 0..=node_level {
            connections.push(Vec::new());
        }

        let node = HnswNode {
            vector: vector.to_vec(),
            connections,
            level: node_level,
        };
        self.nodes.push(node);

        if let Some(ep) = self.entry_point {
            let mut current_ep = ep;

            // Greedy descent from top level to node_level + 1
            let top = self.max_level;
            if top > node_level {
                for level in (node_level + 1..=top).rev() {
                    current_ep = self.greedy_closest(vector, current_ep, level);
                }
            }

            // Insert at each layer from min(node_level, max_level) down to 0
            let insert_top = node_level.min(self.max_level);
            let mut entry_points = vec![current_ep];

            for level in (0..=insert_top).rev() {
                let m_max = if level == 0 { self.m_max0 } else { self.m };
                let found =
                    self.search_layer(vector, &entry_points, self.ef_construction, level);

                // Select top-M neighbours
                let neighbours: Vec<usize> = found
                    .iter()
                    .take(self.m)
                    .map(|di| di.idx)
                    .collect();

                // Connect the new node to neighbours
                self.nodes[node_id].connections[level].clone_from(&neighbours);

                // Connect neighbours back to the new node (with pruning)
                for &nb in &neighbours {
                    let nb_node_level = self.nodes[nb].level;
                    if level <= nb_node_level {
                        self.nodes[nb].connections[level].push(node_id);
                        if self.nodes[nb].connections[level].len() > m_max {
                            self.prune_connections(nb, level, m_max);
                        }
                    }
                }

                entry_points = neighbours;
            }

            if node_level > self.max_level {
                self.max_level = node_level;
                self.entry_point = Some(node_id);
            }
        } else {
            self.entry_point = Some(node_id);
            self.max_level = node_level;
        }

        Ok(node_id)
    }

    /// Search for the `k` nearest neighbours of `query`.
    pub fn search(&self, query: &[T], k: usize) -> Result<NearestNeighborResult<T>> {
        if self.nodes.is_empty() {
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

        let mut current_ep = self.entry_point.unwrap_or(0);

        // Greedy descent from top to layer 1
        if self.max_level > 0 {
            for level in (1..=self.max_level).rev() {
                current_ep = self.greedy_closest(query, current_ep, level);
            }
        }

        // Search layer 0
        let ef = self.ef_search.max(k);
        let results = self.search_layer(query, &[current_ep], ef, 0);

        let take = k.min(results.len());
        let indices = results.iter().take(take).map(|r| r.idx).collect();
        let distances = results.iter().take(take).map(|r| r.dist).collect();

        Ok(NearestNeighborResult { indices, distances })
    }

    /// Batch search: search for each row in `queries`.
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

    /// Number of nodes in the index.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the index is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    // ── internal helpers ──

    fn random_level(&mut self) -> usize {
        let mut rng = Rng::new(self.seed.wrapping_add(self.rng_counter));
        self.rng_counter += 1;
        let r = rng.next_f64();
        // Clamp to avoid ln(0)
        let r = if r < 1e-15 { 1e-15 } else { r };
        let lvl = (-r.ln() * self.level_mult).floor() as usize;
        // Cap at a reasonable maximum to avoid degenerate graphs
        lvl.min(32)
    }

    fn dist_to_node(&self, query: &[T], node_id: usize) -> T {
        compute_distance(query, &self.nodes[node_id].vector, self.metric)
    }

    fn greedy_closest(&self, query: &[T], mut ep: usize, level: usize) -> usize {
        let mut best_dist = self.dist_to_node(query, ep);
        loop {
            let mut changed = false;
            if level < self.nodes[ep].connections.len() {
                // Clone connections to avoid borrow issues
                let neighbors = self.nodes[ep].connections[level].clone();
                for nb in neighbors {
                    let d = self.dist_to_node(query, nb);
                    if d < best_dist {
                        best_dist = d;
                        ep = nb;
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
        ep
    }

    /// Beam search at a given level. Returns results sorted by distance (closest first).
    fn search_layer(
        &self,
        query: &[T],
        entry_points: &[usize],
        ef: usize,
        level: usize,
    ) -> Vec<DistIdx<T>> {
        let mut visited = Vec::new();

        // Candidates: min-heap (pop closest first)
        let mut candidates: BinaryHeap<std::cmp::Reverse<DistIdx<T>>> = BinaryHeap::new();
        // Results: max-heap (pop farthest first to keep bounded)
        let mut results: BinaryHeap<DistIdx<T>> = BinaryHeap::new();

        for &ep in entry_points {
            let d = self.dist_to_node(query, ep);
            candidates.push(std::cmp::Reverse(DistIdx { dist: d, idx: ep }));
            results.push(DistIdx { dist: d, idx: ep });
            visited.push(ep);
        }

        while let Some(std::cmp::Reverse(closest)) = candidates.pop() {
            let farthest_dist = results
                .peek()
                .map_or(T::infinity(), |r| r.dist);
            if closest.dist > farthest_dist {
                break;
            }

            // Explore neighbours at this level
            if level < self.nodes[closest.idx].connections.len() {
                let neighbors = self.nodes[closest.idx].connections[level].clone();
                for nb in neighbors {
                    if visited.contains(&nb) {
                        continue;
                    }
                    visited.push(nb);

                    let d = self.dist_to_node(query, nb);
                    let farthest_dist = results
                        .peek()
                        .map_or(T::infinity(), |r| r.dist);

                    if results.len() < ef {
                        candidates.push(std::cmp::Reverse(DistIdx { dist: d, idx: nb }));
                        results.push(DistIdx { dist: d, idx: nb });
                    } else if d < farthest_dist {
                        candidates.push(std::cmp::Reverse(DistIdx { dist: d, idx: nb }));
                        results.push(DistIdx { dist: d, idx: nb });
                        results.pop();
                    }
                }
            }
        }

        // Sort results by distance
        let mut sorted: Vec<DistIdx<T>> = results.into_vec();
        sorted.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(std::cmp::Ordering::Equal));
        sorted
    }

    fn prune_connections(&mut self, node_id: usize, level: usize, m_max: usize) {
        let query = self.nodes[node_id].vector.clone();
        let conns = self.nodes[node_id].connections[level].clone();

        let mut dists: Vec<DistIdx<T>> = conns
            .into_iter()
            .map(|nb| {
                let d = compute_distance(&query, &self.nodes[nb].vector, self.metric);
                DistIdx { dist: d, idx: nb }
            })
            .collect();

        dists.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(std::cmp::Ordering::Equal));
        dists.truncate(m_max);

        self.nodes[node_id].connections[level] = dists.into_iter().map(|d| d.idx).collect();
    }
}

/// Helper for heap ordering by distance.
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
    use crate::neighbors::brute_force::BruteForceIndex;

    fn make_random_points(n: usize, dim: usize, seed: u64) -> Tensor<f64> {
        let mut rng = Rng::new(seed);
        let data: Vec<f64> = (0..n * dim).map(|_| rng.next_f64()).collect();
        Tensor::from_vec(data, vec![n, dim]).unwrap()
    }

    #[test]
    fn test_hnsw_basic_search() {
        let points = make_random_points(100, 8, 123);
        let mut index = HnswIndex::new(8, DistanceMetric::L2).unwrap();
        index.set_seed(42);
        index.add(&points).unwrap();

        let query_tensor = make_random_points(1, 8, 999);
        let query = query_tensor.as_slice();
        let result = index.search(query, 1).unwrap();
        assert_eq!(result.indices.len(), 1);
    }

    #[test]
    fn test_hnsw_recall() {
        let points = make_random_points(200, 8, 55);
        let mut hnsw = HnswIndex::new(8, DistanceMetric::L2).unwrap();
        hnsw.set_seed(42);
        hnsw.set_ef_construction(100);
        hnsw.set_ef_search(50);
        hnsw.add(&points).unwrap();

        let mut bf = BruteForceIndex::new(DistanceMetric::L2);
        bf.add(&points).unwrap();

        let queries = make_random_points(10, 8, 77);
        let k = 10;
        let mut total_recall = 0.0_f64;

        let qdata = queries.as_slice();
        for i in 0..10 {
            let q = &qdata[i * 8..(i + 1) * 8];
            let hnsw_result = hnsw.search(q, k).unwrap();
            let bf_result = bf.search(q, k).unwrap();

            let hits = hnsw_result
                .indices
                .iter()
                .filter(|idx| bf_result.indices.contains(idx))
                .count();
            total_recall += hits as f64 / k as f64;
        }

        let avg_recall = total_recall / 10.0;
        assert!(
            avg_recall >= 0.8,
            "recall {avg_recall} is below 0.8"
        );
    }

    #[test]
    fn test_hnsw_add_single() {
        let mut index = HnswIndex::new(3, DistanceMetric::L2).unwrap();
        index.set_seed(1);
        index.add_single(&[1.0_f64, 0.0, 0.0]).unwrap();
        index.add_single(&[0.0, 1.0, 0.0]).unwrap();
        index.add_single(&[0.0, 0.0, 1.0]).unwrap();
        assert_eq!(index.len(), 3);
        let result = index.search(&[0.9, 0.1, 0.0], 1).unwrap();
        assert_eq!(result.indices[0], 0);
    }

    #[test]
    fn test_hnsw_batch_search() {
        let points = make_random_points(50, 4, 10);
        let mut index = HnswIndex::new(4, DistanceMetric::L2).unwrap();
        index.set_seed(42);
        index.add(&points).unwrap();

        let queries = make_random_points(5, 4, 20);
        let batch_results = index.batch_search(&queries, 3).unwrap();
        assert_eq!(batch_results.len(), 5);

        let qdata = queries.as_slice();
        for i in 0..5 {
            let q = &qdata[i * 4..(i + 1) * 4];
            let single = index.search(q, 3).unwrap();
            assert_eq!(batch_results[i].indices, single.indices);
        }
    }

    #[test]
    fn test_hnsw_cosine() {
        let mut index = HnswIndex::new(3, DistanceMetric::Cosine).unwrap();
        index.set_seed(7);
        index.add_single(&[1.0_f64, 0.0, 0.0]).unwrap();
        index.add_single(&[0.0, 1.0, 0.0]).unwrap();
        index.add_single(&[0.7, 0.7, 0.0]).unwrap();
        // Query close to [1,0,0] direction
        let result = index.search(&[0.9, 0.1, 0.0], 1).unwrap();
        assert_eq!(result.indices[0], 0);
    }

    #[test]
    fn test_hnsw_empty() {
        let index: HnswIndex<f64> =
            HnswIndex::new(4, DistanceMetric::L2).unwrap();
        let result = index.search(&[1.0, 2.0, 3.0, 4.0], 1);
        assert!(result.is_err());
    }
}
