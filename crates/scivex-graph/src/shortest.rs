use std::collections::BinaryHeap;

use scivex_core::Float;

use crate::digraph::DiGraph;
use crate::error::{GraphError, Result};
use crate::graph::Graph;

/// Result of a single-source shortest path algorithm.
///
/// # Examples
///
/// ```
/// # use scivex_graph::{Graph, shortest};
/// let g = Graph::from_edges(&[(0, 1, 1.0_f64), (1, 2, 3.0)]).unwrap();
/// let result = shortest::dijkstra(&g, 0).unwrap();
/// assert!((result.distances[2] - 4.0).abs() < 1e-10);
/// ```
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct ShortestPathResult<T> {
    /// Distance from the source to each node. Unreachable nodes have `T::infinity()`.
    pub distances: Vec<T>,
    /// Predecessor of each node on the shortest path tree, or `None` if unreachable.
    pub predecessors: Vec<Option<usize>>,
}

impl<T: Float> ShortestPathResult<T> {
    /// Reconstruct the path from `source` to `target`.
    ///
    /// Returns `None` if `target` is unreachable.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_graph::{Graph, shortest};
    /// let g = Graph::from_edges(&[(0, 1, 1.0_f64), (1, 2, 1.0)]).unwrap();
    /// let result = shortest::dijkstra(&g, 0).unwrap();
    /// assert_eq!(result.path_to(0, 2), Some(vec![0, 1, 2]));
    /// ```
    pub fn path_to(&self, source: usize, target: usize) -> Option<Vec<usize>> {
        if self.distances[target] == T::infinity() {
            return None;
        }
        let mut path = Vec::new();
        let mut current = target;
        while current != source {
            path.push(current);
            current = self.predecessors[current]?;
        }
        path.push(source);
        path.reverse();
        Some(path)
    }
}

/// Min-heap state wrapper that implements Ord for Float types.
#[derive(Debug, Clone, Copy)]
struct State<T> {
    dist: T,
    node: usize,
}

impl<T: Float> PartialEq for State<T> {
    fn eq(&self, other: &Self) -> bool {
        self.node == other.node && self.dist == other.dist
    }
}

impl<T: Float> Eq for State<T> {}

impl<T: Float> PartialOrd for State<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Float> Ord for State<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse comparison for min-heap. NaN is treated as infinity.
        other
            .dist
            .partial_cmp(&self.dist)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Dijkstra's algorithm on an undirected graph.
///
/// Returns `Err(NegativeWeight)` if any edge has a negative weight.
///
/// # Examples
///
/// ```
/// # use scivex_graph::{Graph, shortest};
/// let mut g = Graph::<f64>::new();
/// let a = g.add_node();
/// let b = g.add_node();
/// let c = g.add_node();
/// g.add_edge(a, b, 1.0).unwrap();
/// g.add_edge(b, c, 2.0).unwrap();
/// let result = shortest::dijkstra(&g, a).unwrap();
/// assert!((result.distances[c] - 3.0).abs() < 1e-10);
/// ```
pub fn dijkstra<T: Float>(graph: &Graph<T>, source: usize) -> Result<ShortestPathResult<T>> {
    if !graph.is_active(source) {
        return Err(GraphError::NodeNotFound { id: source });
    }

    let n = graph.capacity();
    let mut dist = vec![T::infinity(); n];
    let mut pred: Vec<Option<usize>> = vec![None; n];
    dist[source] = T::zero();

    let mut heap = BinaryHeap::new();
    heap.push(State {
        dist: T::zero(),
        node: source,
    });

    while let Some(State { dist: d, node: u }) = heap.pop() {
        if d > dist[u] {
            continue;
        }
        for &(v, w) in graph.adj_raw(u) {
            if w < T::zero() {
                return Err(GraphError::NegativeWeight);
            }
            let new_dist = dist[u] + w;
            if new_dist < dist[v] {
                dist[v] = new_dist;
                pred[v] = Some(u);
                heap.push(State {
                    dist: new_dist,
                    node: v,
                });
            }
        }
    }

    Ok(ShortestPathResult {
        distances: dist,
        predecessors: pred,
    })
}

/// Dijkstra's algorithm on a directed graph.
///
/// # Examples
///
/// ```
/// # use scivex_graph::{DiGraph, shortest};
/// let g = DiGraph::from_edges(&[(0, 1, 1.0_f64), (1, 2, 2.0)]).unwrap();
/// let result = shortest::dijkstra_directed(&g, 0).unwrap();
/// assert!((result.distances[2] - 3.0).abs() < 1e-10);
/// ```
pub fn dijkstra_directed<T: Float>(
    graph: &DiGraph<T>,
    source: usize,
) -> Result<ShortestPathResult<T>> {
    if !graph.is_active(source) {
        return Err(GraphError::NodeNotFound { id: source });
    }

    let n = graph.capacity();
    let mut dist = vec![T::infinity(); n];
    let mut pred: Vec<Option<usize>> = vec![None; n];
    dist[source] = T::zero();

    let mut heap = BinaryHeap::new();
    heap.push(State {
        dist: T::zero(),
        node: source,
    });

    while let Some(State { dist: d, node: u }) = heap.pop() {
        if d > dist[u] {
            continue;
        }
        for &(v, w) in graph.adj_raw(u) {
            if w < T::zero() {
                return Err(GraphError::NegativeWeight);
            }
            let new_dist = dist[u] + w;
            if new_dist < dist[v] {
                dist[v] = new_dist;
                pred[v] = Some(u);
                heap.push(State {
                    dist: new_dist,
                    node: v,
                });
            }
        }
    }

    Ok(ShortestPathResult {
        distances: dist,
        predecessors: pred,
    })
}

/// Bellman-Ford algorithm on an undirected graph. Handles negative weights.
///
/// Returns `Err(NegativeCycle)` if a negative-weight cycle is reachable.
///
/// # Examples
///
/// ```
/// # use scivex_graph::{Graph, shortest};
/// let g = Graph::from_edges(&[(0, 1, 1.0_f64), (1, 2, 2.0)]).unwrap();
/// let result = shortest::bellman_ford(&g, 0).unwrap();
/// assert!((result.distances[2] - 3.0).abs() < 1e-10);
/// ```
pub fn bellman_ford<T: Float>(graph: &Graph<T>, source: usize) -> Result<ShortestPathResult<T>> {
    if !graph.is_active(source) {
        return Err(GraphError::NodeNotFound { id: source });
    }

    let n = graph.capacity();
    let mut dist = vec![T::infinity(); n];
    let mut pred: Vec<Option<usize>> = vec![None; n];
    dist[source] = T::zero();

    let node_ids: Vec<usize> = graph.node_ids().collect();

    // Relax edges V-1 times
    for _ in 0..node_ids.len().saturating_sub(1) {
        let mut changed = false;
        for &u in &node_ids {
            if dist[u] == T::infinity() {
                continue;
            }
            for &(v, w) in graph.adj_raw(u) {
                let new_dist = dist[u] + w;
                if new_dist < dist[v] {
                    dist[v] = new_dist;
                    pred[v] = Some(u);
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }

    // Check for negative cycles
    for &u in &node_ids {
        if dist[u] == T::infinity() {
            continue;
        }
        for &(v, w) in graph.adj_raw(u) {
            if dist[u] + w < dist[v] {
                return Err(GraphError::NegativeCycle);
            }
        }
    }

    Ok(ShortestPathResult {
        distances: dist,
        predecessors: pred,
    })
}

/// Bellman-Ford algorithm on a directed graph.
///
/// # Examples
///
/// ```
/// # use scivex_graph::{DiGraph, shortest};
/// let g = DiGraph::from_edges(&[(0, 1, 1.0_f64), (1, 2, 2.0)]).unwrap();
/// let result = shortest::bellman_ford_directed(&g, 0).unwrap();
/// assert!((result.distances[2] - 3.0).abs() < 1e-10);
/// ```
pub fn bellman_ford_directed<T: Float>(
    graph: &DiGraph<T>,
    source: usize,
) -> Result<ShortestPathResult<T>> {
    if !graph.is_active(source) {
        return Err(GraphError::NodeNotFound { id: source });
    }

    let n = graph.capacity();
    let mut dist = vec![T::infinity(); n];
    let mut pred: Vec<Option<usize>> = vec![None; n];
    dist[source] = T::zero();

    let node_ids: Vec<usize> = graph.node_ids().collect();

    for _ in 0..node_ids.len().saturating_sub(1) {
        let mut changed = false;
        for &u in &node_ids {
            if dist[u] == T::infinity() {
                continue;
            }
            for &(v, w) in graph.adj_raw(u) {
                let new_dist = dist[u] + w;
                if new_dist < dist[v] {
                    dist[v] = new_dist;
                    pred[v] = Some(u);
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }

    for &u in &node_ids {
        if dist[u] == T::infinity() {
            continue;
        }
        for &(v, w) in graph.adj_raw(u) {
            if dist[u] + w < dist[v] {
                return Err(GraphError::NegativeCycle);
            }
        }
    }

    Ok(ShortestPathResult {
        distances: dist,
        predecessors: pred,
    })
}

/// Floyd-Warshall all-pairs shortest paths on an undirected graph.
///
/// Returns a 2-D distance matrix `dist[u][v]`.
///
/// # Examples
///
/// ```
/// # use scivex_graph::{Graph, shortest};
/// let g = Graph::from_edges(&[(0, 1, 1.0_f64), (1, 2, 2.0)]).unwrap();
/// let dist = shortest::floyd_warshall(&g).unwrap();
/// assert!((dist[0][2] - 3.0).abs() < 1e-10);
/// ```
pub fn floyd_warshall<T: Float>(graph: &Graph<T>) -> Result<Vec<Vec<T>>> {
    let n = graph.capacity();
    let mut dist = vec![vec![T::infinity(); n]; n];

    for u in graph.node_ids() {
        dist[u][u] = T::zero();
        for &(v, w) in graph.adj_raw(u) {
            dist[u][v] = w;
        }
    }

    for k in graph.node_ids() {
        for i in graph.node_ids() {
            if dist[i][k] == T::infinity() {
                continue;
            }
            for j in graph.node_ids() {
                let through_k = dist[i][k] + dist[k][j];
                if through_k < dist[i][j] {
                    dist[i][j] = through_k;
                }
            }
        }
    }

    Ok(dist)
}

/// Floyd-Warshall all-pairs shortest paths on a directed graph.
///
/// # Examples
///
/// ```
/// # use scivex_graph::{DiGraph, shortest};
/// let g = DiGraph::from_edges(&[(0, 1, 1.0_f64), (1, 2, 2.0)]).unwrap();
/// let dist = shortest::floyd_warshall_directed(&g).unwrap();
/// assert!((dist[0][2] - 3.0).abs() < 1e-10);
/// ```
pub fn floyd_warshall_directed<T: Float>(graph: &DiGraph<T>) -> Result<Vec<Vec<T>>> {
    let n = graph.capacity();
    let mut dist = vec![vec![T::infinity(); n]; n];

    for u in graph.node_ids() {
        dist[u][u] = T::zero();
        for &(v, w) in graph.adj_raw(u) {
            dist[u][v] = w;
        }
    }

    for k in graph.node_ids() {
        for i in graph.node_ids() {
            if dist[i][k] == T::infinity() {
                continue;
            }
            for j in graph.node_ids() {
                let through_k = dist[i][k] + dist[k][j];
                if through_k < dist[i][j] {
                    dist[i][j] = through_k;
                }
            }
        }
    }

    Ok(dist)
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn test_dijkstra_simple_path() {
        // 0 --1--> 1 --2--> 2
        let g = Graph::from_edges(&[(0, 1, 1.0), (1, 2, 2.0)]).unwrap();
        let result = dijkstra(&g, 0).unwrap();

        assert!((result.distances[0]).abs() < 1e-10);
        assert!((result.distances[1] - 1.0).abs() < 1e-10);
        assert!((result.distances[2] - 3.0).abs() < 1e-10);

        let path = result.path_to(0, 2).unwrap();
        assert_eq!(path, vec![0, 1, 2]);
    }

    #[test]
    fn test_dijkstra_unreachable() {
        let mut g = Graph::<f64>::new();
        let a = g.add_node();
        let _b = g.add_node(); // isolated
        g.add_node(); // another node
        g.add_edge(a, 2, 1.0).unwrap();

        let result = dijkstra(&g, 0).unwrap();
        assert_eq!(result.distances[1], f64::INFINITY);
        assert!(result.path_to(0, 1).is_none());
    }

    #[test]
    fn test_dijkstra_negative_weight_error() {
        let g = Graph::from_edges(&[(0, 1, -1.0)]).unwrap();
        let result = dijkstra(&g, 0);
        assert_eq!(result.unwrap_err(), GraphError::NegativeWeight);
    }

    #[test]
    fn test_bellman_ford_negative_edges() {
        // Directed: 0->1 (4), 0->2 (1), 2->1 (-2)
        let g = DiGraph::from_edges(&[(0, 1, 4.0), (0, 2, 1.0), (2, 1, -2.0)]).unwrap();
        let result = bellman_ford_directed(&g, 0).unwrap();

        assert!((result.distances[0]).abs() < 1e-10);
        assert!((result.distances[1] - (-1.0)).abs() < 1e-10);
        assert!((result.distances[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_bellman_ford_negative_cycle() {
        // Directed cycle with negative total: 0->1 (1), 1->2 (-3), 2->0 (1)
        let g = DiGraph::from_edges(&[(0, 1, 1.0), (1, 2, -3.0), (2, 0, 1.0)]).unwrap();
        let result = bellman_ford_directed(&g, 0);
        assert_eq!(result.unwrap_err(), GraphError::NegativeCycle);
    }

    #[test]
    fn test_floyd_warshall_agrees_with_dijkstra() {
        let g = Graph::from_edges(&[(0, 1, 1.0), (1, 2, 2.0), (0, 2, 5.0)]).unwrap();

        let fw = floyd_warshall(&g).unwrap();
        let dijk = dijkstra(&g, 0).unwrap();

        for (v, &fw_dist) in fw[0].iter().enumerate().take(3) {
            assert!(
                (fw_dist - dijk.distances[v]).abs() < 1e-10,
                "mismatch at node {v}"
            );
        }
    }

    #[test]
    fn test_path_reconstruction() {
        let g = Graph::from_edges(&[(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (0, 3, 10.0)]).unwrap();
        let result = dijkstra(&g, 0).unwrap();
        let path = result.path_to(0, 3).unwrap();
        assert_eq!(path, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_dijkstra_directed() {
        let g = DiGraph::from_edges(&[(0, 1, 1.0), (1, 2, 2.0)]).unwrap();
        let result = dijkstra_directed(&g, 0).unwrap();
        assert!((result.distances[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_floyd_warshall_directed() {
        let g = DiGraph::from_edges(&[(0, 1, 1.0), (1, 2, 2.0)]).unwrap();
        let fw = floyd_warshall_directed(&g).unwrap();
        assert!((fw[0][2] - 3.0).abs() < 1e-10);
        assert_eq!(fw[2][0], f64::INFINITY); // no reverse path
    }
}
