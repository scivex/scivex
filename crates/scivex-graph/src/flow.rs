//! Network flow algorithms: max-flow (Edmonds-Karp) and bipartite matching.

use scivex_core::Float;
use std::collections::VecDeque;

use crate::digraph::DiGraph;
use crate::error::{GraphError, Result};

/// Result of a max-flow computation.
#[derive(Debug, Clone)]
pub struct MaxFlowResult<T: Float> {
    /// The maximum flow value from source to sink.
    pub max_flow: T,
    /// Flow on each edge, indexed by `(from, to)`.
    /// Stored as a flat adjacency matrix of size `n * n`.
    pub flow: Vec<T>,
    /// Number of nodes in the graph.
    pub n: usize,
}

impl<T: Float> MaxFlowResult<T> {
    /// Get the flow on a specific edge.
    pub fn edge_flow(&self, from: usize, to: usize) -> T {
        if from < self.n && to < self.n {
            self.flow[from * self.n + to]
        } else {
            T::zero()
        }
    }
}

/// Compute the maximum flow from `source` to `sink` using the Edmonds-Karp
/// algorithm (BFS-based Ford-Fulkerson).
///
/// Time complexity: O(V * E^2)
///
/// Edge weights are treated as capacities.
pub fn max_flow<T: Float>(
    graph: &DiGraph<T>,
    source: usize,
    sink: usize,
) -> Result<MaxFlowResult<T>> {
    let n = graph.capacity();
    if n == 0 {
        return Err(GraphError::EmptyGraph);
    }
    if !graph.is_active(source) {
        return Err(GraphError::NodeNotFound { id: source });
    }
    if !graph.is_active(sink) {
        return Err(GraphError::NodeNotFound { id: sink });
    }

    // Build capacity matrix and residual graph.
    let mut capacity = vec![T::zero(); n * n];
    for (u, v, w) in graph.edges() {
        capacity[u * n + v] = w;
    }

    let mut flow = vec![T::zero(); n * n];
    let mut total_flow = T::zero();

    loop {
        // BFS to find augmenting path in residual graph.
        let mut parent = vec![None::<usize>; n];
        let mut visited = vec![false; n];
        let mut queue = VecDeque::new();
        visited[source] = true;
        queue.push_back(source);

        while let Some(u) = queue.pop_front() {
            if u == sink {
                break;
            }
            for v in 0..n {
                if !visited[v] && !graph.is_active(v) {
                    continue;
                }
                let residual = capacity[u * n + v] - flow[u * n + v];
                if !visited[v] && residual > T::zero() {
                    visited[v] = true;
                    parent[v] = Some(u);
                    queue.push_back(v);
                }
            }
        }

        if !visited[sink] {
            break; // No augmenting path found
        }

        // Find bottleneck capacity along the path.
        let mut path_flow = T::infinity();
        let mut v = sink;
        while let Some(u) = parent[v] {
            let residual = capacity[u * n + v] - flow[u * n + v];
            if residual < path_flow {
                path_flow = residual;
            }
            v = u;
        }

        // Update flow along the path.
        v = sink;
        while let Some(u) = parent[v] {
            flow[u * n + v] += path_flow;
            flow[v * n + u] -= path_flow;
            v = u;
        }

        total_flow += path_flow;
    }

    Ok(MaxFlowResult {
        max_flow: total_flow,
        flow,
        n,
    })
}

/// Find a maximum matching in a bipartite graph.
///
/// - `left_nodes`: node IDs of the left partition
/// - `right_nodes`: node IDs of the right partition
///
/// Returns a list of matched pairs `(left, right)` and the matching size.
#[derive(Debug, Clone)]
pub struct MatchingResult {
    /// Matched pairs `(left, right)`.
    pub pairs: Vec<(usize, usize)>,
    /// Size of the matching.
    pub size: usize,
}

/// Find a maximum cardinality matching in a bipartite graph using the
/// Hopcroft-Karp algorithm (augmenting paths via BFS+DFS).
///
/// The graph should be undirected (use `Graph`) but this function accepts
/// node partitions and adjacency information from a `DiGraph`. For undirected
/// graphs, convert to `DiGraph` or provide edges in both directions.
///
/// Alternatively, this function works directly with adjacency lists for
/// simplicity.
pub fn bipartite_matching(
    left_size: usize,
    right_size: usize,
    edges: &[(usize, usize)],
) -> MatchingResult {
    // DFS: augment along shortest paths (Hopcroft-Karp helper).
    fn dfs(
        u: usize,
        adj: &[Vec<usize>],
        match_left: &mut [usize],
        match_right: &mut [usize],
        dist: &mut [usize],
    ) -> bool {
        const NIL: usize = usize::MAX;
        const INF: usize = usize::MAX;
        for &v in &adj[u] {
            let w = match_right[v];
            if w == NIL || (dist[w] == dist[u] + 1 && dfs(w, adj, match_left, match_right, dist)) {
                match_left[u] = v;
                match_right[v] = u;
                return true;
            }
        }
        dist[u] = INF;
        false
    }

    // Hopcroft-Karp algorithm.
    // left nodes: 0..left_size, right nodes: 0..right_size
    const NIL: usize = usize::MAX;
    const INF: usize = usize::MAX;

    // Build adjacency list for left nodes.
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); left_size];
    for &(l, r) in edges {
        if l < left_size && r < right_size {
            adj[l].push(r);
        }
    }

    let mut match_left = vec![NIL; left_size];
    let mut match_right = vec![NIL; right_size];
    let mut dist = vec![0usize; left_size];

    // BFS: find shortest augmenting path lengths.
    let bfs = |match_left: &[usize],
               match_right: &[usize],
               dist: &mut [usize],
               adj: &[Vec<usize>]|
     -> bool {
        let mut queue = VecDeque::new();
        for u in 0..left_size {
            if match_left[u] == NIL {
                dist[u] = 0;
                queue.push_back(u);
            } else {
                dist[u] = INF;
            }
        }
        let mut found = false;
        while let Some(u) = queue.pop_front() {
            for &v in &adj[u] {
                let w = match_right[v];
                if w == NIL {
                    found = true;
                } else if dist[w] == INF {
                    dist[w] = dist[u] + 1;
                    queue.push_back(w);
                }
            }
        }
        found
    };

    let mut matching = 0;
    while bfs(&match_left, &match_right, &mut dist, &adj) {
        for u in 0..left_size {
            if match_left[u] == NIL && dfs(u, &adj, &mut match_left, &mut match_right, &mut dist) {
                matching += 1;
            }
        }
    }

    let mut pairs = Vec::with_capacity(matching);
    for (l, &r) in match_left.iter().enumerate() {
        if r != NIL {
            pairs.push((l, r));
        }
    }

    MatchingResult {
        pairs,
        size: matching,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_flow_simple() {
        // s=0, t=3
        // 0 -> 1 (cap 10), 0 -> 2 (cap 10)
        // 1 -> 3 (cap 10), 2 -> 3 (cap 10)
        // 1 -> 2 (cap 1)
        let g = DiGraph::from_edges(&[
            (0, 1, 10.0_f64),
            (0, 2, 10.0),
            (1, 3, 10.0),
            (2, 3, 10.0),
            (1, 2, 1.0),
        ])
        .unwrap();

        let result = max_flow(&g, 0, 3).unwrap();
        assert!((result.max_flow - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_max_flow_bottleneck() {
        // 0 -> 1 (cap 5), 1 -> 2 (cap 3)
        // Max flow = 3 (bottleneck at edge 1->2)
        let g = DiGraph::from_edges(&[(0, 1, 5.0_f64), (1, 2, 3.0)]).unwrap();
        let result = max_flow(&g, 0, 2).unwrap();
        assert!((result.max_flow - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_max_flow_no_path() {
        // 0 -> 1, but sink is 2 (no path from 0 to 2)
        let mut g = DiGraph::<f64>::new();
        g.add_node(); // 0
        g.add_node(); // 1
        g.add_node(); // 2
        g.add_edge(0, 1, 10.0).unwrap();
        let result = max_flow(&g, 0, 2).unwrap();
        assert!((result.max_flow).abs() < 1e-10);
    }

    #[test]
    fn test_max_flow_diamond() {
        //     1
        //   / | \
        // 0   |  3
        //   \ | /
        //     2
        // 0->1: 3, 0->2: 2, 1->3: 2, 2->3: 3, 1->2: 1
        let g = DiGraph::from_edges(&[
            (0, 1, 3.0_f64),
            (0, 2, 2.0),
            (1, 3, 2.0),
            (2, 3, 3.0),
            (1, 2, 1.0),
        ])
        .unwrap();
        let result = max_flow(&g, 0, 3).unwrap();
        assert!((result.max_flow - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_max_flow_edge_flow() {
        let g = DiGraph::from_edges(&[(0, 1, 5.0_f64), (1, 2, 3.0)]).unwrap();
        let result = max_flow(&g, 0, 2).unwrap();
        // Flow through edge 0->1 should be 3
        assert!((result.edge_flow(0, 1) - 3.0).abs() < 1e-10);
        // Flow through edge 1->2 should be 3
        assert!((result.edge_flow(1, 2) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_bipartite_matching_perfect() {
        // Left: {0, 1, 2}, Right: {0, 1, 2}
        // Edges: 0-0, 0-1, 1-1, 2-2
        let edges = vec![(0, 0), (0, 1), (1, 1), (2, 2)];
        let result = bipartite_matching(3, 3, &edges);
        assert_eq!(result.size, 3);
    }

    #[test]
    fn test_bipartite_matching_partial() {
        // Left: {0, 1}, Right: {0}
        // Only one can match
        let edges = vec![(0, 0), (1, 0)];
        let result = bipartite_matching(2, 1, &edges);
        assert_eq!(result.size, 1);
    }

    #[test]
    fn test_bipartite_matching_no_edges() {
        let result = bipartite_matching(3, 3, &[]);
        assert_eq!(result.size, 0);
        assert!(result.pairs.is_empty());
    }

    #[test]
    fn test_bipartite_matching_full_bipartite() {
        // Complete bipartite K3,3
        let mut edges = Vec::new();
        for l in 0..3 {
            for r in 0..3 {
                edges.push((l, r));
            }
        }
        let result = bipartite_matching(3, 3, &edges);
        assert_eq!(result.size, 3);
    }

    #[test]
    fn test_max_flow_empty_graph() {
        let g = DiGraph::<f64>::new();
        assert!(max_flow(&g, 0, 1).is_err());
    }
}
