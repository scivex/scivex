use std::collections::VecDeque;

use scivex_core::Float;

use crate::digraph::DiGraph;
use crate::error::{GraphError, Result};
use crate::graph::Graph;

/// Degree centrality for each node: `degree(v) / (n - 1)`.
///
/// Returns a vector indexed by node slot. Inactive nodes have centrality 0.
pub fn degree_centrality<T: Float>(graph: &Graph<T>) -> Vec<T> {
    let n = graph.capacity();
    let nc = graph.node_count();
    let mut centrality = vec![T::zero(); n];

    if nc <= 1 {
        return centrality;
    }

    let denom = T::from_f64((nc - 1) as f64);
    for u in graph.node_ids() {
        let deg = graph.adj_raw(u).len();
        centrality[u] = T::from_f64(deg as f64) / denom;
    }

    centrality
}

/// In-degree centrality for each node in a directed graph.
pub fn in_degree_centrality<T: Float>(graph: &DiGraph<T>) -> Vec<T> {
    let n = graph.capacity();
    let nc = graph.node_count();
    let mut centrality = vec![T::zero(); n];

    if nc <= 1 {
        return centrality;
    }

    let denom = T::from_f64((nc - 1) as f64);
    for u in graph.node_ids() {
        let deg = graph.in_adj_raw(u).len();
        centrality[u] = T::from_f64(deg as f64) / denom;
    }

    centrality
}

/// Out-degree centrality for each node in a directed graph.
pub fn out_degree_centrality<T: Float>(graph: &DiGraph<T>) -> Vec<T> {
    let n = graph.capacity();
    let nc = graph.node_count();
    let mut centrality = vec![T::zero(); n];

    if nc <= 1 {
        return centrality;
    }

    let denom = T::from_f64((nc - 1) as f64);
    for u in graph.node_ids() {
        let deg = graph.adj_raw(u).len();
        centrality[u] = T::from_f64(deg as f64) / denom;
    }

    centrality
}

/// Betweenness centrality using Brandes' algorithm (O(V * E) for unweighted).
///
/// Computes the fraction of shortest paths through each node.
/// The result is normalized by `2 / ((n-1)(n-2))` for undirected graphs.
pub fn betweenness_centrality<T: Float>(graph: &Graph<T>) -> Vec<T> {
    let n = graph.capacity();
    let nc = graph.node_count();
    let mut cb = vec![T::zero(); n];

    if nc <= 2 {
        return cb;
    }

    let node_ids: Vec<usize> = graph.node_ids().collect();

    for &s in &node_ids {
        // BFS-based Brandes
        let mut stack = Vec::new();
        let mut predecessors: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut sigma = vec![T::zero(); n]; // number of shortest paths
        sigma[s] = T::one();
        let mut dist: Vec<i64> = vec![-1; n];
        dist[s] = 0;

        let mut queue = VecDeque::new();
        queue.push_back(s);

        while let Some(v) = queue.pop_front() {
            stack.push(v);
            for &(w, _) in graph.adj_raw(v) {
                // First visit
                if dist[w] < 0 {
                    dist[w] = dist[v] + 1;
                    queue.push_back(w);
                }
                // Shortest path via v?
                if dist[w] == dist[v] + 1 {
                    let sv = sigma[v];
                    sigma[w] += sv;
                    predecessors[w].push(v);
                }
            }
        }

        // Back-propagation
        let mut delta = vec![T::zero(); n];
        while let Some(w) = stack.pop() {
            for &v in &predecessors[w] {
                let contribution = (sigma[v] / sigma[w]) * (T::one() + delta[w]);
                delta[v] += contribution;
            }
            if w != s {
                cb[w] += delta[w];
            }
        }
    }

    // Normalize: undirected graph counts each pair twice
    let norm = T::from_f64(2.0) / T::from_f64(((nc - 1) * (nc - 2)) as f64);
    for u in graph.node_ids() {
        cb[u] *= norm;
    }

    cb
}

/// PageRank for a directed graph using power iteration.
///
/// - `damping`: damping factor (typically 0.85)
/// - `max_iter`: maximum iterations
/// - `tol`: convergence tolerance
pub fn pagerank<T: Float>(
    graph: &DiGraph<T>,
    damping: T,
    max_iter: usize,
    tol: T,
) -> Result<Vec<T>> {
    if damping < T::zero() || damping > T::one() {
        return Err(GraphError::InvalidParameter {
            name: "damping",
            reason: "must be between 0 and 1",
        });
    }

    let n = graph.capacity();
    let nc = graph.node_count();
    if nc == 0 {
        return Ok(Vec::new());
    }

    let n_f = T::from_f64(nc as f64);
    let mut rank = vec![T::zero(); n];
    let initial = T::one() / n_f;
    for u in graph.node_ids() {
        rank[u] = initial;
    }

    let node_ids: Vec<usize> = graph.node_ids().collect();

    for _ in 0..max_iter {
        let mut new_rank = vec![T::zero(); n];

        // Collect dangling node mass (nodes with no outgoing edges)
        let mut dangling_sum = T::zero();
        for &u in &node_ids {
            if graph.adj_raw(u).is_empty() {
                dangling_sum += rank[u];
            }
        }

        let teleport = (T::one() - damping) / n_f;
        let dangling_contrib = damping * dangling_sum / n_f;

        for &v in &node_ids {
            let mut incoming_sum = T::zero();
            for &(u, _) in graph.in_adj_raw(v) {
                let out_deg = graph.adj_raw(u).len();
                incoming_sum += rank[u] / T::from_f64(out_deg as f64);
            }
            new_rank[v] = teleport + dangling_contrib + damping * incoming_sum;
        }

        // Check convergence
        let mut diff = T::zero();
        for &u in &node_ids {
            diff += (new_rank[u] - rank[u]).abs();
        }

        rank = new_rank;

        if diff < tol {
            break;
        }
    }

    Ok(rank)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_degree_centrality_star() {
        // Star: hub 0 connected to 1,2,3,4
        let g = Graph::from_edges(&[(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0), (0, 4, 1.0)]).unwrap();

        let dc = degree_centrality(&g);
        assert!((dc[0] - 1.0).abs() < 1e-10); // hub: 4/(5-1) = 1.0
        assert!((dc[1] - 0.25).abs() < 1e-10); // leaf: 1/4 = 0.25
    }

    #[test]
    fn test_betweenness_path() {
        // Path: 0-1-2-3-4
        let g = Graph::from_edges(&[(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0)]).unwrap();

        let bc = betweenness_centrality(&g);
        // Interior nodes should have higher betweenness than endpoints
        assert!(bc[2] > bc[0]);
        assert!(bc[2] > bc[4]);
        assert!(bc[1] > bc[0]);
    }

    #[test]
    fn test_pagerank_sums_to_one() {
        let g = DiGraph::from_edges(&[(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0), (2, 3, 1.0)]).unwrap();

        let pr = pagerank(&g, 0.85, 100, 1e-8).unwrap();
        let sum: f64 = pr.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_pagerank_invalid_damping() {
        let g = DiGraph::from_edges(&[(0, 1, 1.0)]).unwrap();
        let result = pagerank(&g, 1.5, 100, 1e-8);
        assert!(result.is_err());
    }

    #[test]
    fn test_in_out_degree_centrality() {
        let g = DiGraph::from_edges(&[(0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0)]).unwrap();

        let idc = in_degree_centrality(&g);
        let odc = out_degree_centrality(&g);

        assert!((idc[0]).abs() < 1e-10); // no incoming edges
        assert!((idc[2] - 1.0).abs() < 1e-10); // 2 incoming, 2/(3-1)=1.0
        assert!((odc[0] - 1.0).abs() < 1e-10); // 2 outgoing, 2/(3-1)=1.0
    }
}
