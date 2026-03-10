use std::collections::VecDeque;

use scivex_core::Float;

use crate::digraph::DiGraph;
use crate::error::{GraphError, Result};
use crate::graph::Graph;

/// Breadth-first search on an undirected graph starting from `start`.
///
/// Returns node IDs in the order they are visited.
pub fn bfs<T: Float>(graph: &Graph<T>, start: usize) -> Result<Vec<usize>> {
    if !graph.is_active(start) {
        return Err(GraphError::NodeNotFound { id: start });
    }

    let n = graph.capacity();
    let mut visited = vec![false; n];
    let mut order = Vec::new();
    let mut queue = VecDeque::new();

    visited[start] = true;
    queue.push_back(start);

    while let Some(u) = queue.pop_front() {
        order.push(u);
        for &(v, _) in graph.adj_raw(u) {
            if !visited[v] {
                visited[v] = true;
                queue.push_back(v);
            }
        }
    }

    Ok(order)
}

/// Breadth-first search on a directed graph starting from `start`.
pub fn bfs_directed<T: Float>(graph: &DiGraph<T>, start: usize) -> Result<Vec<usize>> {
    if !graph.is_active(start) {
        return Err(GraphError::NodeNotFound { id: start });
    }

    let n = graph.capacity();
    let mut visited = vec![false; n];
    let mut order = Vec::new();
    let mut queue = VecDeque::new();

    visited[start] = true;
    queue.push_back(start);

    while let Some(u) = queue.pop_front() {
        order.push(u);
        for &(v, _) in graph.adj_raw(u) {
            if !visited[v] {
                visited[v] = true;
                queue.push_back(v);
            }
        }
    }

    Ok(order)
}

/// Depth-first search on an undirected graph starting from `start`.
///
/// Uses an explicit stack (iterative, not recursive).
pub fn dfs<T: Float>(graph: &Graph<T>, start: usize) -> Result<Vec<usize>> {
    if !graph.is_active(start) {
        return Err(GraphError::NodeNotFound { id: start });
    }

    let n = graph.capacity();
    let mut visited = vec![false; n];
    let mut order = Vec::new();
    let mut stack = vec![start];

    while let Some(u) = stack.pop() {
        if visited[u] {
            continue;
        }
        visited[u] = true;
        order.push(u);

        // Push in reverse order so that lower-numbered neighbors are visited first
        let neighbors = graph.adj_raw(u);
        for &(v, _) in neighbors.iter().rev() {
            if !visited[v] {
                stack.push(v);
            }
        }
    }

    Ok(order)
}

/// Depth-first search on a directed graph starting from `start`.
pub fn dfs_directed<T: Float>(graph: &DiGraph<T>, start: usize) -> Result<Vec<usize>> {
    if !graph.is_active(start) {
        return Err(GraphError::NodeNotFound { id: start });
    }

    let n = graph.capacity();
    let mut visited = vec![false; n];
    let mut order = Vec::new();
    let mut stack = vec![start];

    while let Some(u) = stack.pop() {
        if visited[u] {
            continue;
        }
        visited[u] = true;
        order.push(u);

        let neighbors = graph.adj_raw(u);
        for &(v, _) in neighbors.iter().rev() {
            if !visited[v] {
                stack.push(v);
            }
        }
    }

    Ok(order)
}

/// Topological sort using Kahn's algorithm.
///
/// Returns `Err(CycleDetected)` if the graph contains a cycle.
pub fn topological_sort<T: Float>(graph: &DiGraph<T>) -> Result<Vec<usize>> {
    if graph.node_count() == 0 {
        return Ok(Vec::new());
    }

    let n = graph.capacity();
    let mut in_degree = vec![0usize; n];
    for u in graph.node_ids() {
        in_degree[u] = graph.in_adj_raw(u).len();
    }

    let mut queue = VecDeque::new();
    for u in graph.node_ids() {
        if in_degree[u] == 0 {
            queue.push_back(u);
        }
    }

    let mut order = Vec::new();
    while let Some(u) = queue.pop_front() {
        order.push(u);
        for &(v, _) in graph.adj_raw(u) {
            in_degree[v] -= 1;
            if in_degree[v] == 0 {
                queue.push_back(v);
            }
        }
    }

    if order.len() != graph.node_count() {
        return Err(GraphError::CycleDetected);
    }

    Ok(order)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bfs_star() {
        // Star graph: 0 connected to 1,2,3
        let g = Graph::from_edges(&[(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0)]).unwrap();
        let order = bfs(&g, 0).unwrap();
        assert_eq!(order[0], 0);
        // Neighbors 1,2,3 should all be at level 1
        assert_eq!(order.len(), 4);
        let level1: Vec<_> = order[1..].to_vec();
        assert!(level1.contains(&1));
        assert!(level1.contains(&2));
        assert!(level1.contains(&3));
    }

    #[test]
    fn test_dfs_path() {
        // Path: 0-1-2-3
        let g = Graph::from_edges(&[(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)]).unwrap();
        let order = dfs(&g, 0).unwrap();
        assert_eq!(order[0], 0);
        assert_eq!(order.len(), 4);
    }

    #[test]
    fn test_topological_sort_dag() {
        // DAG: 0->1, 0->2, 1->3, 2->3
        let g = DiGraph::from_edges(&[(0, 1, 1.0), (0, 2, 1.0), (1, 3, 1.0), (2, 3, 1.0)]).unwrap();

        let order = topological_sort(&g).unwrap();
        assert_eq!(order.len(), 4);

        // 0 must come before 1,2; both must come before 3
        let pos = |n: usize| order.iter().position(|&x| x == n).unwrap();
        assert!(pos(0) < pos(1));
        assert!(pos(0) < pos(2));
        assert!(pos(1) < pos(3));
        assert!(pos(2) < pos(3));
    }

    #[test]
    fn test_topological_sort_cycle_error() {
        // Cycle: 0->1->2->0
        let g = DiGraph::from_edges(&[(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)]).unwrap();
        let result = topological_sort(&g);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), GraphError::CycleDetected);
    }

    #[test]
    fn test_bfs_directed() {
        let g = DiGraph::from_edges(&[(0, 1, 1.0), (0, 2, 1.0), (1, 3, 1.0)]).unwrap();
        let order = bfs_directed(&g, 0).unwrap();
        assert_eq!(order[0], 0);
        assert_eq!(order.len(), 4);
    }

    #[test]
    fn test_dfs_directed() {
        let g = DiGraph::from_edges(&[(0, 1, 1.0), (1, 2, 1.0)]).unwrap();
        let order = dfs_directed(&g, 0).unwrap();
        assert_eq!(order, vec![0, 1, 2]);
    }
}
