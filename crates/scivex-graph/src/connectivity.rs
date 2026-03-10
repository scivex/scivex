use std::collections::VecDeque;

use scivex_core::Float;

use crate::digraph::DiGraph;
use crate::error::Result;
use crate::graph::Graph;

/// Find all connected components of an undirected graph.
///
/// Each component is a `Vec<usize>` of node IDs. Components are sorted by
/// their smallest node ID.
pub fn connected_components<T: Float>(graph: &Graph<T>) -> Vec<Vec<usize>> {
    let n = graph.capacity();
    let mut visited = vec![false; n];
    let mut components = Vec::new();

    for u in graph.node_ids() {
        if visited[u] {
            continue;
        }
        let mut component = Vec::new();
        let mut queue = VecDeque::new();
        visited[u] = true;
        queue.push_back(u);

        while let Some(v) = queue.pop_front() {
            component.push(v);
            for &(w, _) in graph.adj_raw(v) {
                if !visited[w] {
                    visited[w] = true;
                    queue.push_back(w);
                }
            }
        }

        component.sort_unstable();
        components.push(component);
    }

    components.sort_by_key(|c| c[0]);
    components
}

/// Check if an undirected graph is connected.
pub fn is_connected<T: Float>(graph: &Graph<T>) -> bool {
    if graph.node_count() <= 1 {
        return true;
    }
    let Some(start) = graph.node_ids().next() else {
        return true;
    };

    let n = graph.capacity();
    let mut visited = vec![false; n];
    let mut queue = VecDeque::new();
    let mut count = 0;

    visited[start] = true;
    queue.push_back(start);

    while let Some(u) = queue.pop_front() {
        count += 1;
        for &(v, _) in graph.adj_raw(u) {
            if !visited[v] {
                visited[v] = true;
                queue.push_back(v);
            }
        }
    }

    count == graph.node_count()
}

/// Find all strongly connected components of a directed graph using
/// Kosaraju's algorithm.
///
/// Each SCC is a `Vec<usize>` of node IDs.
pub fn strongly_connected_components<T: Float>(graph: &DiGraph<T>) -> Vec<Vec<usize>> {
    let n = graph.capacity();

    // Pass 1: compute finish order on the forward graph
    let mut visited = vec![false; n];
    let mut finish_order = Vec::new();

    for u in graph.node_ids() {
        if !visited[u] {
            dfs_finish(graph, u, &mut visited, &mut finish_order);
        }
    }

    // Pass 2: DFS on reverse graph in reverse finish order
    let mut visited = vec![false; n];
    let mut components = Vec::new();

    for &u in finish_order.iter().rev() {
        if !graph.is_active(u) || visited[u] {
            continue;
        }
        let mut component = Vec::new();
        dfs_reverse(graph, u, &mut visited, &mut component);
        component.sort_unstable();
        components.push(component);
    }

    components.sort_by_key(|c| c[0]);
    components
}

fn dfs_finish<T: Float>(
    graph: &DiGraph<T>,
    start: usize,
    visited: &mut [bool],
    finish_order: &mut Vec<usize>,
) {
    // Iterative DFS with explicit finish tracking
    let mut stack: Vec<(usize, bool)> = vec![(start, false)];

    while let Some((u, processed)) = stack.pop() {
        if processed {
            finish_order.push(u);
            continue;
        }
        if visited[u] {
            continue;
        }
        visited[u] = true;
        stack.push((u, true)); // push again to record finish

        for &(v, _) in graph.adj_raw(u) {
            if !visited[v] {
                stack.push((v, false));
            }
        }
    }
}

fn dfs_reverse<T: Float>(
    graph: &DiGraph<T>,
    start: usize,
    visited: &mut [bool],
    component: &mut Vec<usize>,
) {
    let mut stack = vec![start];
    visited[start] = true;

    while let Some(u) = stack.pop() {
        component.push(u);
        for &(v, _) in graph.in_adj_raw(u) {
            if !visited[v] && graph.is_active(v) {
                visited[v] = true;
                stack.push(v);
            }
        }
    }
}

/// Check if a directed graph is strongly connected.
pub fn is_strongly_connected<T: Float>(graph: &DiGraph<T>) -> bool {
    if graph.node_count() <= 1 {
        return true;
    }
    let sccs = strongly_connected_components(graph);
    sccs.len() == 1
}

/// Find weakly connected components of a directed graph.
///
/// Treats all edges as undirected.
pub fn weakly_connected_components<T: Float>(graph: &DiGraph<T>) -> Result<Vec<Vec<usize>>> {
    let undirected = graph.to_undirected()?;
    Ok(connected_components(&undirected))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_component() {
        let g = Graph::from_edges(&[(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)]).unwrap();
        let cc = connected_components(&g);
        assert_eq!(cc.len(), 1);
        assert_eq!(cc[0], vec![0, 1, 2]);
        assert!(is_connected(&g));
    }

    #[test]
    fn test_multiple_components() {
        let mut g = Graph::<f64>::new();
        for _ in 0..4 {
            g.add_node();
        }
        g.add_edge(0, 1, 1.0).unwrap();
        g.add_edge(2, 3, 1.0).unwrap();

        let cc = connected_components(&g);
        assert_eq!(cc.len(), 2);
        assert_eq!(cc[0], vec![0, 1]);
        assert_eq!(cc[1], vec![2, 3]);
        assert!(!is_connected(&g));
    }

    #[test]
    fn test_scc_cycle_is_one_scc() {
        // 0->1->2->0 is one SCC
        let g = DiGraph::from_edges(&[(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)]).unwrap();
        let sccs = strongly_connected_components(&g);
        assert_eq!(sccs.len(), 1);
        assert_eq!(sccs[0], vec![0, 1, 2]);
        assert!(is_strongly_connected(&g));
    }

    #[test]
    fn test_scc_dag_singletons() {
        // DAG: 0->1->2 — each node is its own SCC
        let g = DiGraph::from_edges(&[(0, 1, 1.0), (1, 2, 1.0)]).unwrap();
        let sccs = strongly_connected_components(&g);
        assert_eq!(sccs.len(), 3);
        assert!(!is_strongly_connected(&g));
    }

    #[test]
    fn test_is_connected_empty() {
        let g = Graph::<f64>::new();
        assert!(is_connected(&g));
    }

    #[test]
    fn test_is_connected_single_node() {
        let mut g = Graph::<f64>::new();
        g.add_node();
        assert!(is_connected(&g));
    }
}
