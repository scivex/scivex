use scivex_core::Float;
use scivex_core::linalg::sparse::{CooMatrix, CsrMatrix};

use crate::error::{GraphError, Result};

/// Undirected, weighted graph using adjacency lists.
///
/// Nodes are identified by `usize` indices. Removed nodes leave a tombstone
/// (empty adjacency list, `active[u] = false`) so that existing node IDs
/// remain stable.
///
/// # Examples
///
/// ```
/// # use scivex_graph::Graph;
/// let g = Graph::from_edges(&[(0, 1, 1.0_f64), (1, 2, 2.0)]).unwrap();
/// assert_eq!(g.node_count(), 3);
/// assert_eq!(g.edge_count(), 2);
/// ```
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct Graph<T: Float> {
    adj: Vec<Vec<(usize, T)>>,
    active: Vec<bool>,
    node_count: usize,
    edge_count: usize,
}

impl<T: Float> Graph<T> {
    /// Create an empty graph.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_graph::Graph;
    /// let g = Graph::<f64>::new();
    /// assert_eq!(g.node_count(), 0);
    /// assert_eq!(g.edge_count(), 0);
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self {
            adj: Vec::new(),
            active: Vec::new(),
            node_count: 0,
            edge_count: 0,
        }
    }

    /// Add a node and return its ID.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_graph::Graph;
    /// let mut g = Graph::<f64>::new();
    /// let a = g.add_node();
    /// let b = g.add_node();
    /// assert_eq!(g.node_count(), 2);
    /// ```
    pub fn add_node(&mut self) -> usize {
        let id = self.adj.len();
        self.adj.push(Vec::new());
        self.active.push(true);
        self.node_count += 1;
        id
    }

    /// Add an undirected edge between `u` and `v` with the given weight.
    ///
    /// If the edge already exists, its weight is overwritten.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_graph::Graph;
    /// let mut g = Graph::<f64>::new();
    /// let a = g.add_node();
    /// let b = g.add_node();
    /// g.add_edge(a, b, 1.5).unwrap();
    /// assert_eq!(g.edge_count(), 1);
    /// ```
    pub fn add_edge(&mut self, u: usize, v: usize, weight: T) -> Result<()> {
        self.check_node(u)?;
        self.check_node(v)?;

        let is_new = !self.adj[u].iter().any(|(n, _)| *n == v);

        Self::set_or_insert(&mut self.adj[u], v, weight);
        if u != v {
            Self::set_or_insert(&mut self.adj[v], u, weight);
        }

        if is_new {
            self.edge_count += 1;
        }

        Ok(())
    }

    /// Remove a node and all its edges.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_graph::Graph;
    /// let mut g = Graph::from_edges(&[(0, 1, 1.0_f64), (1, 2, 2.0)]).unwrap();
    /// g.remove_node(1).unwrap();
    /// assert_eq!(g.node_count(), 2);
    /// assert_eq!(g.edge_count(), 0);
    /// ```
    pub fn remove_node(&mut self, u: usize) -> Result<()> {
        self.check_node(u)?;

        // Remove all edges from neighbors pointing to u
        let neighbors: Vec<(usize, T)> = self.adj[u].clone();
        for (v, _) in &neighbors {
            if *v != u {
                self.adj[*v].retain(|(n, _)| *n != u);
            }
        }

        // Count edges being removed (self-loops counted once)
        let edge_removal_count = neighbors.len();
        // For undirected: neighbors list includes each neighbor once.
        // Each entry is one undirected edge.
        self.edge_count = self.edge_count.saturating_sub(edge_removal_count);

        self.adj[u].clear();
        self.active[u] = false;
        self.node_count -= 1;

        Ok(())
    }

    /// Remove the edge between `u` and `v`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_graph::Graph;
    /// let mut g = Graph::from_edges(&[(0, 1, 1.0_f64)]).unwrap();
    /// g.remove_edge(0, 1).unwrap();
    /// assert_eq!(g.edge_count(), 0);
    /// ```
    pub fn remove_edge(&mut self, u: usize, v: usize) -> Result<()> {
        self.check_node(u)?;
        self.check_node(v)?;

        let len_before = self.adj[u].len();
        self.adj[u].retain(|(n, _)| *n != v);
        if self.adj[u].len() == len_before {
            return Err(GraphError::EdgeNotFound { from: u, to: v });
        }
        if u != v {
            self.adj[v].retain(|(n, _)| *n != u);
        }
        self.edge_count -= 1;
        Ok(())
    }

    /// Return the neighbors of node `u` as `(neighbor_id, weight)` pairs.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_graph::Graph;
    /// let g = Graph::from_edges(&[(0, 1, 1.0_f64), (0, 2, 2.0)]).unwrap();
    /// let nbrs = g.neighbors(0).unwrap();
    /// assert_eq!(nbrs.len(), 2);
    /// ```
    pub fn neighbors(&self, u: usize) -> Result<&[(usize, T)]> {
        self.check_node(u)?;
        Ok(&self.adj[u])
    }

    /// Return the degree of node `u`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_graph::Graph;
    /// let g = Graph::from_edges(&[(0, 1, 1.0_f64), (0, 2, 2.0)]).unwrap();
    /// assert_eq!(g.degree(0).unwrap(), 2);
    /// assert_eq!(g.degree(1).unwrap(), 1);
    /// ```
    pub fn degree(&self, u: usize) -> Result<usize> {
        self.check_node(u)?;
        Ok(self.adj[u].len())
    }

    /// Number of active nodes.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_graph::Graph;
    /// let g = Graph::from_edges(&[(0, 1, 1.0_f64)]).unwrap();
    /// assert_eq!(g.node_count(), 2);
    /// ```
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.node_count
    }

    /// Number of undirected edges.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_graph::Graph;
    /// let g = Graph::from_edges(&[(0, 1, 1.0_f64), (1, 2, 2.0)]).unwrap();
    /// assert_eq!(g.edge_count(), 2);
    /// ```
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.edge_count
    }

    /// Check if an edge exists between `u` and `v`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_graph::Graph;
    /// let g = Graph::from_edges(&[(0, 1, 1.0_f64)]).unwrap();
    /// assert!(g.has_edge(0, 1).unwrap());
    /// assert!(!g.has_edge(0, 2).unwrap_or(false));
    /// ```
    pub fn has_edge(&self, u: usize, v: usize) -> Result<bool> {
        self.check_node(u)?;
        self.check_node(v)?;
        Ok(self.adj[u].iter().any(|(n, _)| *n == v))
    }

    /// Get the weight of the edge between `u` and `v`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_graph::Graph;
    /// let g = Graph::from_edges(&[(0, 1, 3.5_f64)]).unwrap();
    /// assert_eq!(g.get_weight(0, 1).unwrap(), 3.5);
    /// ```
    pub fn get_weight(&self, u: usize, v: usize) -> Result<T> {
        self.check_node(u)?;
        self.check_node(v)?;
        self.adj[u]
            .iter()
            .find(|(n, _)| *n == v)
            .map(|(_, w)| *w)
            .ok_or(GraphError::EdgeNotFound { from: u, to: v })
    }

    /// Build a CSR adjacency matrix. The matrix is `n x n` where `n` is the
    /// total capacity (including tombstoned nodes). Inactive nodes have empty
    /// rows/columns.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_graph::Graph;
    /// let g = Graph::from_edges(&[(0, 1, 1.0_f64), (1, 2, 2.0)]).unwrap();
    /// let mat = g.adjacency_matrix().unwrap();
    /// assert_eq!(mat.shape(), (3, 3));
    /// ```
    pub fn adjacency_matrix(&self) -> Result<CsrMatrix<T>> {
        let n = self.adj.len();
        let mut coo = CooMatrix::new(n, n);
        for u in 0..n {
            if !self.active[u] {
                continue;
            }
            for &(v, w) in &self.adj[u] {
                coo.push(u, v, w)?;
            }
        }
        Ok(coo.to_csr())
    }

    /// Build a graph from a symmetric CSR adjacency matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_graph::Graph;
    /// let g1 = Graph::from_edges(&[(0, 1, 1.0_f64), (1, 2, 2.0)]).unwrap();
    /// let mat = g1.adjacency_matrix().unwrap();
    /// let g2 = Graph::from_adjacency_matrix(&mat).unwrap();
    /// assert_eq!(g2.node_count(), 3);
    /// ```
    pub fn from_adjacency_matrix(mat: &CsrMatrix<T>) -> Result<Self> {
        let (nrows, ncols) = mat.shape();
        if nrows != ncols {
            return Err(GraphError::InvalidParameter {
                name: "matrix",
                reason: "adjacency matrix must be square",
            });
        }
        let mut g = Self::new();
        for _ in 0..nrows {
            g.add_node();
        }
        for u in 0..nrows {
            for v in u..ncols {
                #[allow(clippy::collapsible_if)]
                if let Some(&w) = mat.get(u, v) {
                    if w != T::zero() {
                        g.add_edge(u, v, w)?;
                    }
                }
            }
        }
        Ok(g)
    }

    /// Build a graph from a list of `(u, v, weight)` edges.
    ///
    /// Nodes are created as needed up to the maximum node ID referenced.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_graph::Graph;
    /// let g = Graph::from_edges(&[(0, 1, 1.0_f64), (1, 2, 2.0), (0, 2, 3.0)]).unwrap();
    /// assert_eq!(g.node_count(), 3);
    /// assert_eq!(g.edge_count(), 3);
    /// ```
    pub fn from_edges(edges: &[(usize, usize, T)]) -> Result<Self> {
        let max_node = edges
            .iter()
            .flat_map(|&(u, v, _)| [u, v])
            .max()
            .unwrap_or(0);
        let mut g = Self::new();
        for _ in 0..=max_node {
            g.add_node();
        }
        for &(u, v, w) in edges {
            g.add_edge(u, v, w)?;
        }
        Ok(g)
    }

    /// Iterate over all active node IDs.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_graph::Graph;
    /// let g = Graph::from_edges(&[(0, 1, 1.0_f64)]).unwrap();
    /// let ids: Vec<_> = g.node_ids().collect();
    /// assert_eq!(ids, vec![0, 1]);
    /// ```
    pub fn node_ids(&self) -> impl Iterator<Item = usize> + '_ {
        self.active
            .iter()
            .enumerate()
            .filter(|(_, a)| **a)
            .map(|(i, _)| i)
    }

    /// Iterate over all edges as `(u, v, weight)` — each undirected edge
    /// appears once with `u <= v`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_graph::Graph;
    /// let g = Graph::from_edges(&[(0, 1, 1.0_f64), (1, 2, 2.0)]).unwrap();
    /// let edges: Vec<_> = g.edges().collect();
    /// assert_eq!(edges.len(), 2);
    /// ```
    pub fn edges(&self) -> impl Iterator<Item = (usize, usize, T)> + '_ {
        self.adj
            .iter()
            .enumerate()
            .filter(move |(i, _)| self.active[*i])
            .flat_map(|(u, neighbors)| {
                neighbors
                    .iter()
                    .filter(move |(v, _)| u <= *v)
                    .map(move |&(v, w)| (u, v, w))
            })
    }

    /// Total number of slots (including tombstoned nodes).
    #[must_use]
    pub(crate) fn capacity(&self) -> usize {
        self.adj.len()
    }

    /// Whether a slot is active.
    #[must_use]
    pub(crate) fn is_active(&self, u: usize) -> bool {
        u < self.active.len() && self.active[u]
    }

    /// Raw adjacency list for a node (unchecked).
    pub(crate) fn adj_raw(&self, u: usize) -> &[(usize, T)] {
        &self.adj[u]
    }

    fn check_node(&self, u: usize) -> Result<()> {
        if u >= self.adj.len() || !self.active[u] {
            return Err(GraphError::NodeNotFound { id: u });
        }
        Ok(())
    }

    fn set_or_insert(list: &mut Vec<(usize, T)>, target: usize, weight: T) {
        if let Some(entry) = list.iter_mut().find(|(n, _)| *n == target) {
            entry.1 = weight;
        } else {
            list.push((target, weight));
        }
    }
}

impl<T: Float> Default for Graph<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn test_add_nodes_and_edges() {
        let mut g = Graph::<f64>::new();
        let a = g.add_node();
        let b = g.add_node();
        let c = g.add_node();
        assert_eq!(a, 0);
        assert_eq!(b, 1);
        assert_eq!(c, 2);

        g.add_edge(a, b, 1.0).unwrap();
        g.add_edge(b, c, 2.0).unwrap();

        assert_eq!(g.node_count(), 3);
        assert_eq!(g.edge_count(), 2);
        assert!(g.has_edge(a, b).unwrap());
        assert!(g.has_edge(b, a).unwrap());
        assert!(!g.has_edge(a, c).unwrap());
    }

    #[test]
    fn test_bidirectional_edges() {
        let mut g = Graph::<f64>::new();
        let a = g.add_node();
        let b = g.add_node();
        g.add_edge(a, b, 3.0).unwrap();

        assert_eq!(g.get_weight(a, b).unwrap(), 3.0);
        assert_eq!(g.get_weight(b, a).unwrap(), 3.0);
    }

    #[test]
    fn test_remove_node_cleans_edges() {
        let mut g = Graph::<f64>::new();
        let a = g.add_node();
        let b = g.add_node();
        let c = g.add_node();
        g.add_edge(a, b, 1.0).unwrap();
        g.add_edge(b, c, 2.0).unwrap();

        g.remove_node(b).unwrap();

        assert_eq!(g.node_count(), 2);
        assert_eq!(g.edge_count(), 0);
        assert!(g.neighbors(a).unwrap().is_empty());
        assert!(g.neighbors(c).unwrap().is_empty());
    }

    #[test]
    fn test_symmetric_adjacency_matrix() {
        let mut g = Graph::<f64>::new();
        let a = g.add_node();
        let b = g.add_node();
        let c = g.add_node();
        g.add_edge(a, b, 1.0).unwrap();
        g.add_edge(b, c, 2.0).unwrap();

        let mat = g.adjacency_matrix().unwrap();
        assert_eq!(mat.get(0, 1), Some(&1.0));
        assert_eq!(mat.get(1, 0), Some(&1.0));
        assert_eq!(mat.get(1, 2), Some(&2.0));
        assert_eq!(mat.get(2, 1), Some(&2.0));
        assert_eq!(mat.get(0, 2), None);
    }

    #[test]
    fn test_from_edges_roundtrip() {
        let edges = vec![(0, 1, 1.0), (1, 2, 2.0), (2, 0, 3.0)];
        let g = Graph::from_edges(&edges).unwrap();

        assert_eq!(g.node_count(), 3);
        assert_eq!(g.edge_count(), 3);
        assert_eq!(g.get_weight(0, 1).unwrap(), 1.0);
        assert_eq!(g.get_weight(1, 2).unwrap(), 2.0);
        assert_eq!(g.get_weight(2, 0).unwrap(), 3.0);
    }

    #[test]
    fn test_overwrite_edge_weight() {
        let mut g = Graph::<f64>::new();
        let a = g.add_node();
        let b = g.add_node();
        g.add_edge(a, b, 1.0).unwrap();
        g.add_edge(a, b, 5.0).unwrap();

        assert_eq!(g.get_weight(a, b).unwrap(), 5.0);
        assert_eq!(g.edge_count(), 1);
    }

    #[test]
    fn test_self_loop() {
        let mut g = Graph::<f64>::new();
        let a = g.add_node();
        g.add_edge(a, a, 1.0).unwrap();
        assert_eq!(g.edge_count(), 1);
        assert!(g.has_edge(a, a).unwrap());
    }

    #[test]
    fn test_node_ids_skips_removed() {
        let mut g = Graph::<f64>::new();
        g.add_node();
        let b = g.add_node();
        g.add_node();
        g.remove_node(b).unwrap();

        let ids: Vec<_> = g.node_ids().collect();
        assert_eq!(ids, vec![0, 2]);
    }

    #[test]
    fn test_edges_iterator() {
        let g = Graph::from_edges(&[(0, 1, 1.0), (1, 2, 2.0)]).unwrap();
        let mut edges: Vec<_> = g.edges().collect();
        edges.sort_by_key(|e| (e.0, e.1));
        assert_eq!(edges, vec![(0, 1, 1.0), (1, 2, 2.0)]);
    }

    #[test]
    fn test_empty_graph() {
        let g = Graph::<f64>::new();
        assert_eq!(g.node_count(), 0);
        assert_eq!(g.edge_count(), 0);
        assert_eq!(g.edges().count(), 0);
        assert_eq!(g.node_ids().count(), 0);
    }

    #[test]
    fn test_single_node_no_edges() {
        let mut g = Graph::<f64>::new();
        let a = g.add_node();
        assert_eq!(g.node_count(), 1);
        assert_eq!(g.edge_count(), 0);
        assert_eq!(g.degree(a).unwrap(), 0);
        assert!(g.neighbors(a).unwrap().is_empty());
    }

    #[test]
    fn test_invalid_node_errors() {
        let g = Graph::<f64>::new();
        assert_eq!(
            g.neighbors(0).unwrap_err(),
            GraphError::NodeNotFound { id: 0 }
        );
        assert_eq!(g.degree(5).unwrap_err(), GraphError::NodeNotFound { id: 5 });
        assert_eq!(
            g.has_edge(0, 1).unwrap_err(),
            GraphError::NodeNotFound { id: 0 }
        );
        assert_eq!(
            g.get_weight(0, 1).unwrap_err(),
            GraphError::NodeNotFound { id: 0 }
        );
    }

    #[test]
    fn test_add_edge_invalid_node() {
        let mut g = Graph::<f64>::new();
        g.add_node();
        let result = g.add_edge(0, 5, 1.0);
        assert_eq!(result.unwrap_err(), GraphError::NodeNotFound { id: 5 });
    }

    #[test]
    fn test_remove_edge_nonexistent() {
        let mut g = Graph::<f64>::new();
        let a = g.add_node();
        let b = g.add_node();
        let result = g.remove_edge(a, b);
        assert_eq!(
            result.unwrap_err(),
            GraphError::EdgeNotFound { from: a, to: b }
        );
    }

    #[test]
    fn test_remove_self_loop() {
        let mut g = Graph::<f64>::new();
        let a = g.add_node();
        g.add_edge(a, a, 2.0).unwrap();
        assert_eq!(g.edge_count(), 1);
        g.remove_edge(a, a).unwrap();
        assert_eq!(g.edge_count(), 0);
        assert!(!g.has_edge(a, a).unwrap());
    }

    #[test]
    fn test_get_weight_no_edge() {
        let mut g = Graph::<f64>::new();
        let a = g.add_node();
        let b = g.add_node();
        let result = g.get_weight(a, b);
        assert_eq!(
            result.unwrap_err(),
            GraphError::EdgeNotFound { from: a, to: b }
        );
    }

    #[test]
    fn test_from_edges_empty() {
        let g = Graph::<f64>::from_edges(&[]).unwrap();
        assert_eq!(g.node_count(), 1); // max_node defaults to 0, so one node created
        assert_eq!(g.edge_count(), 0);
    }

    #[test]
    fn test_remove_node_invalid() {
        let mut g = Graph::<f64>::new();
        let result = g.remove_node(0);
        assert_eq!(result.unwrap_err(), GraphError::NodeNotFound { id: 0 });
    }

    #[test]
    fn test_fully_connected_graph() {
        let mut g = Graph::<f64>::new();
        for _ in 0..4 {
            g.add_node();
        }
        // Add all 6 edges for a 4-node complete graph
        for i in 0..4 {
            for j in (i + 1)..4 {
                g.add_edge(i, j, 1.0).unwrap();
            }
        }
        assert_eq!(g.edge_count(), 6);
        for i in 0..4 {
            assert_eq!(g.degree(i).unwrap(), 3);
        }
    }

    #[test]
    fn test_default_trait() {
        let g = Graph::<f64>::default();
        assert_eq!(g.node_count(), 0);
        assert_eq!(g.edge_count(), 0);
    }

    #[test]
    fn test_remove_node_with_self_loop() {
        let mut g = Graph::<f64>::new();
        let a = g.add_node();
        let b = g.add_node();
        g.add_edge(a, b, 1.0).unwrap();
        g.add_edge(a, a, 2.0).unwrap();
        assert_eq!(g.edge_count(), 2);

        g.remove_node(a).unwrap();
        assert_eq!(g.node_count(), 1);
        assert_eq!(g.edge_count(), 0);
        assert!(g.neighbors(b).unwrap().is_empty());
    }

    #[test]
    fn test_remove_edge_then_readd() {
        let mut g = Graph::<f64>::new();
        let a = g.add_node();
        let b = g.add_node();
        g.add_edge(a, b, 1.0).unwrap();
        g.remove_edge(a, b).unwrap();
        assert_eq!(g.edge_count(), 0);
        assert!(!g.has_edge(a, b).unwrap());

        g.add_edge(a, b, 5.0).unwrap();
        assert_eq!(g.edge_count(), 1);
        assert_eq!(g.get_weight(a, b).unwrap(), 5.0);
    }

    #[test]
    fn test_edges_iterator_with_self_loop() {
        let mut g = Graph::<f64>::new();
        let a = g.add_node();
        let b = g.add_node();
        g.add_edge(a, a, 1.0).unwrap();
        g.add_edge(a, b, 2.0).unwrap();

        let edges: Vec<_> = g.edges().collect();
        // Self-loop: u==v so u<=v holds. Normal edge: (0,1).
        assert_eq!(edges.len(), 2);
    }

    #[test]
    fn test_node_ids_after_remove_and_add() {
        let mut g = Graph::<f64>::new();
        let a = g.add_node(); // 0
        let b = g.add_node(); // 1
        g.remove_node(a).unwrap();
        let c = g.add_node(); // 2 (not reusing 0)

        let ids: Vec<_> = g.node_ids().collect();
        assert_eq!(ids, vec![b, c]);
        assert_eq!(g.node_count(), 2);
    }

    #[test]
    fn test_degree_with_self_loop() {
        let mut g = Graph::<f64>::new();
        let a = g.add_node();
        g.add_edge(a, a, 1.0).unwrap();
        // Self-loop adds one entry to adj[a]
        assert_eq!(g.degree(a).unwrap(), 1);
    }

    #[test]
    fn test_adjacency_matrix_empty_graph() {
        let g = Graph::<f64>::new();
        let mat = g.adjacency_matrix().unwrap();
        assert_eq!(mat.shape(), (0, 0));
    }

    #[test]
    fn test_clone() {
        let mut g = Graph::<f64>::new();
        let a = g.add_node();
        let b = g.add_node();
        g.add_edge(a, b, 3.0).unwrap();

        let g2 = g.clone();
        assert_eq!(g2.node_count(), 2);
        assert_eq!(g2.edge_count(), 1);
        assert_eq!(g2.get_weight(a, b).unwrap(), 3.0);
    }

    #[test]
    fn test_f32_graph() {
        let mut g = Graph::<f32>::new();
        let a = g.add_node();
        let b = g.add_node();
        g.add_edge(a, b, 1.5f32).unwrap();
        assert_eq!(g.get_weight(a, b).unwrap(), 1.5f32);
    }
}
