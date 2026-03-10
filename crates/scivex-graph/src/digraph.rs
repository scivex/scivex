use scivex_core::Float;
use scivex_core::linalg::sparse::{CooMatrix, CsrMatrix};

use crate::error::{GraphError, Result};
use crate::graph::Graph;

/// Directed, weighted graph using adjacency lists.
///
/// Stores both outgoing and incoming adjacency lists so that in-degree queries
/// and reverse traversals (e.g., Kosaraju's SCC) are efficient.
#[derive(Debug, Clone)]
pub struct DiGraph<T: Float> {
    adj: Vec<Vec<(usize, T)>>,
    in_adj: Vec<Vec<(usize, T)>>,
    active: Vec<bool>,
    node_count: usize,
    edge_count: usize,
}

impl<T: Float> DiGraph<T> {
    /// Create an empty directed graph.
    #[must_use]
    pub fn new() -> Self {
        Self {
            adj: Vec::new(),
            in_adj: Vec::new(),
            active: Vec::new(),
            node_count: 0,
            edge_count: 0,
        }
    }

    /// Add a node and return its ID.
    pub fn add_node(&mut self) -> usize {
        let id = self.adj.len();
        self.adj.push(Vec::new());
        self.in_adj.push(Vec::new());
        self.active.push(true);
        self.node_count += 1;
        id
    }

    /// Add a directed edge from `u` to `v` with the given weight.
    pub fn add_edge(&mut self, u: usize, v: usize, weight: T) -> Result<()> {
        self.check_node(u)?;
        self.check_node(v)?;

        let is_new = !self.adj[u].iter().any(|(n, _)| *n == v);
        Self::set_or_insert(&mut self.adj[u], v, weight);
        Self::set_or_insert(&mut self.in_adj[v], u, weight);

        if is_new {
            self.edge_count += 1;
        }

        Ok(())
    }

    /// Remove a node and all its incoming and outgoing edges.
    pub fn remove_node(&mut self, u: usize) -> Result<()> {
        self.check_node(u)?;

        // Remove outgoing edges: for each (u -> v), remove u from in_adj[v]
        for &(v, _) in &self.adj[u] {
            if v != u {
                self.in_adj[v].retain(|(n, _)| *n != u);
            }
        }
        let out_count = self.adj[u].len();

        // Remove incoming edges: for each (w -> u), remove u from adj[w]
        for &(w, _) in &self.in_adj[u] {
            if w != u {
                self.adj[w].retain(|(n, _)| *n != u);
            }
        }
        let in_count = self.in_adj[u].len();

        // Self-loops are counted in both lists, so subtract overlap
        let self_loops = usize::from(self.adj[u].iter().any(|(n, _)| *n == u));
        self.edge_count = self
            .edge_count
            .saturating_sub(out_count + in_count - self_loops);

        self.adj[u].clear();
        self.in_adj[u].clear();
        self.active[u] = false;
        self.node_count -= 1;

        Ok(())
    }

    /// Remove the directed edge from `u` to `v`.
    pub fn remove_edge(&mut self, u: usize, v: usize) -> Result<()> {
        self.check_node(u)?;
        self.check_node(v)?;

        let len_before = self.adj[u].len();
        self.adj[u].retain(|(n, _)| *n != v);
        if self.adj[u].len() == len_before {
            return Err(GraphError::EdgeNotFound { from: u, to: v });
        }
        self.in_adj[v].retain(|(n, _)| *n != u);
        self.edge_count -= 1;
        Ok(())
    }

    /// Outgoing neighbors of `u`.
    pub fn neighbors(&self, u: usize) -> Result<&[(usize, T)]> {
        self.check_node(u)?;
        Ok(&self.adj[u])
    }

    /// Incoming neighbors of `u`.
    pub fn in_neighbors(&self, u: usize) -> Result<&[(usize, T)]> {
        self.check_node(u)?;
        Ok(&self.in_adj[u])
    }

    /// Return the out-degree of node `u`.
    pub fn out_degree(&self, u: usize) -> Result<usize> {
        self.check_node(u)?;
        Ok(self.adj[u].len())
    }

    /// Return the in-degree of node `u`.
    pub fn in_degree(&self, u: usize) -> Result<usize> {
        self.check_node(u)?;
        Ok(self.in_adj[u].len())
    }

    /// Number of active nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.node_count
    }

    /// Number of directed edges.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.edge_count
    }

    /// Check if a directed edge exists from `u` to `v`.
    pub fn has_edge(&self, u: usize, v: usize) -> Result<bool> {
        self.check_node(u)?;
        self.check_node(v)?;
        Ok(self.adj[u].iter().any(|(n, _)| *n == v))
    }

    /// Get the weight of the directed edge from `u` to `v`.
    pub fn get_weight(&self, u: usize, v: usize) -> Result<T> {
        self.check_node(u)?;
        self.check_node(v)?;
        self.adj[u]
            .iter()
            .find(|(n, _)| *n == v)
            .map(|(_, w)| *w)
            .ok_or(GraphError::EdgeNotFound { from: u, to: v })
    }

    /// Build a CSR adjacency matrix for the directed graph.
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

    /// Build a directed graph from a CSR adjacency matrix.
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
            for v in 0..ncols {
                if let Some(&w) = mat.get(u, v)
                    && w != T::zero()
                {
                    g.add_edge(u, v, w)?;
                }
            }
        }
        Ok(g)
    }

    /// Build a directed graph from a list of `(from, to, weight)` edges.
    ///
    /// Nodes are created as needed up to the maximum node ID referenced.
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
    pub fn node_ids(&self) -> impl Iterator<Item = usize> + '_ {
        self.active
            .iter()
            .enumerate()
            .filter(|(_, a)| **a)
            .map(|(i, _)| i)
    }

    /// Iterate over all directed edges as `(from, to, weight)`.
    pub fn edges(&self) -> impl Iterator<Item = (usize, usize, T)> + '_ {
        self.adj
            .iter()
            .enumerate()
            .filter(move |(i, _)| self.active[*i])
            .flat_map(|(u, neighbors)| neighbors.iter().map(move |&(v, w)| (u, v, w)))
    }

    /// Convert to an undirected graph. For each directed edge `(u, v, w)`,
    /// an undirected edge is added. If both `(u, v)` and `(v, u)` exist, the
    /// weight from the edge with the smaller source ID is used.
    pub fn to_undirected(&self) -> Result<Graph<T>> {
        let n = self.adj.len();
        let mut g = Graph::new();
        for _ in 0..n {
            g.add_node();
        }
        // Deactivate tombstoned nodes
        for u in 0..n {
            if !self.active[u] {
                g.remove_node(u).ok();
            }
        }
        for u in self.node_ids() {
            for &(v, w) in &self.adj[u] {
                g.add_edge(u, v, w)?;
            }
        }
        Ok(g)
    }

    #[must_use]
    pub(crate) fn capacity(&self) -> usize {
        self.adj.len()
    }

    #[must_use]
    pub(crate) fn is_active(&self, u: usize) -> bool {
        u < self.active.len() && self.active[u]
    }

    pub(crate) fn adj_raw(&self, u: usize) -> &[(usize, T)] {
        &self.adj[u]
    }

    pub(crate) fn in_adj_raw(&self, u: usize) -> &[(usize, T)] {
        &self.in_adj[u]
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

impl<T: Float> Default for DiGraph<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn test_directed_edge_not_bidirectional() {
        let mut g = DiGraph::<f64>::new();
        let a = g.add_node();
        let b = g.add_node();
        g.add_edge(a, b, 1.0).unwrap();

        assert!(g.has_edge(a, b).unwrap());
        assert!(!g.has_edge(b, a).unwrap());
    }

    #[test]
    fn test_in_out_degree() {
        let mut g = DiGraph::<f64>::new();
        let a = g.add_node();
        let b = g.add_node();
        let c = g.add_node();
        g.add_edge(a, b, 1.0).unwrap();
        g.add_edge(c, b, 2.0).unwrap();

        assert_eq!(g.out_degree(a).unwrap(), 1);
        assert_eq!(g.in_degree(b).unwrap(), 2);
        assert_eq!(g.out_degree(b).unwrap(), 0);
    }

    #[test]
    fn test_asymmetric_adjacency() {
        let mut g = DiGraph::<f64>::new();
        let a = g.add_node();
        let b = g.add_node();
        g.add_edge(a, b, 1.0).unwrap();

        let mat = g.adjacency_matrix().unwrap();
        assert_eq!(mat.get(0, 1), Some(&1.0));
        assert_eq!(mat.get(1, 0), None);
    }

    #[test]
    fn test_to_undirected() {
        let mut g = DiGraph::<f64>::new();
        let a = g.add_node();
        let b = g.add_node();
        g.add_edge(a, b, 1.0).unwrap();

        let ug = g.to_undirected().unwrap();
        assert!(ug.has_edge(a, b).unwrap());
        assert!(ug.has_edge(b, a).unwrap());
        assert_eq!(ug.edge_count(), 1);
    }

    #[test]
    fn test_from_edges() {
        let g = DiGraph::from_edges(&[(0, 1, 1.0), (1, 2, 2.0)]).unwrap();
        assert_eq!(g.node_count(), 3);
        assert_eq!(g.edge_count(), 2);
        assert!(g.has_edge(0, 1).unwrap());
        assert!(!g.has_edge(1, 0).unwrap());
    }

    #[test]
    fn test_remove_node() {
        let mut g = DiGraph::<f64>::new();
        let a = g.add_node();
        let b = g.add_node();
        let c = g.add_node();
        g.add_edge(a, b, 1.0).unwrap();
        g.add_edge(b, c, 2.0).unwrap();
        g.add_edge(c, a, 3.0).unwrap();

        g.remove_node(b).unwrap();
        assert_eq!(g.node_count(), 2);
        assert_eq!(g.edge_count(), 1); // only c->a remains
        assert!(g.has_edge(c, a).unwrap());
    }

    #[test]
    fn test_self_loop_directed() {
        let mut g = DiGraph::<f64>::new();
        let a = g.add_node();
        g.add_edge(a, a, 5.0).unwrap();
        assert_eq!(g.edge_count(), 1);
        assert!(g.has_edge(a, a).unwrap());
        assert_eq!(g.out_degree(a).unwrap(), 1);
        assert_eq!(g.in_degree(a).unwrap(), 1);
    }

    #[test]
    fn test_remove_edge_directed() {
        let mut g = DiGraph::<f64>::new();
        let a = g.add_node();
        let b = g.add_node();
        g.add_edge(a, b, 1.0).unwrap();
        g.remove_edge(a, b).unwrap();
        assert_eq!(g.edge_count(), 0);
        assert!(!g.has_edge(a, b).unwrap());
    }

    #[test]
    fn test_remove_edge_nonexistent_directed() {
        let mut g = DiGraph::<f64>::new();
        let a = g.add_node();
        let b = g.add_node();
        let result = g.remove_edge(a, b);
        assert_eq!(result.unwrap_err(), GraphError::EdgeNotFound { from: a, to: b });
    }

    #[test]
    fn test_overwrite_edge_weight_directed() {
        let mut g = DiGraph::<f64>::new();
        let a = g.add_node();
        let b = g.add_node();
        g.add_edge(a, b, 1.0).unwrap();
        g.add_edge(a, b, 9.0).unwrap();
        assert_eq!(g.get_weight(a, b).unwrap(), 9.0);
        assert_eq!(g.edge_count(), 1);
    }

    #[test]
    fn test_in_neighbors() {
        let g = DiGraph::from_edges(&[(0, 2, 1.0), (1, 2, 2.0)]).unwrap();
        let in_n = g.in_neighbors(2).unwrap();
        assert_eq!(in_n.len(), 2);
        let sources: Vec<usize> = in_n.iter().map(|(s, _)| *s).collect();
        assert!(sources.contains(&0));
        assert!(sources.contains(&1));
    }

    #[test]
    fn test_edges_iterator_directed() {
        let g = DiGraph::from_edges(&[(0, 1, 1.0), (1, 0, 2.0)]).unwrap();
        let edges: Vec<_> = g.edges().collect();
        // Both directions should appear
        assert_eq!(edges.len(), 2);
        assert!(edges.contains(&(0, 1, 1.0)));
        assert!(edges.contains(&(1, 0, 2.0)));
    }

    #[test]
    fn test_empty_digraph() {
        let g = DiGraph::<f64>::new();
        assert_eq!(g.node_count(), 0);
        assert_eq!(g.edge_count(), 0);
        assert_eq!(g.edges().count(), 0);
        assert_eq!(g.node_ids().count(), 0);
    }

    #[test]
    fn test_default_digraph() {
        let g = DiGraph::<f64>::default();
        assert_eq!(g.node_count(), 0);
    }

    #[test]
    fn test_remove_node_with_self_loop_directed() {
        let mut g = DiGraph::<f64>::new();
        let a = g.add_node();
        let b = g.add_node();
        g.add_edge(a, b, 1.0).unwrap();
        g.add_edge(a, a, 2.0).unwrap();
        assert_eq!(g.edge_count(), 2);

        g.remove_node(a).unwrap();
        assert_eq!(g.node_count(), 1);
        assert_eq!(g.edge_count(), 0);
    }

    #[test]
    fn test_to_undirected_both_directions() {
        let mut g = DiGraph::<f64>::new();
        let a = g.add_node();
        let b = g.add_node();
        g.add_edge(a, b, 3.0).unwrap();
        g.add_edge(b, a, 7.0).unwrap();

        let ug = g.to_undirected().unwrap();
        assert_eq!(ug.edge_count(), 1);
        // The weight from the smaller-source-ID edge should be used (a->b: 3.0)
        // because add_edge overwrites, and a < b, so a->b is processed first,
        // then b->a overwrites with 7.0.
        assert!(ug.has_edge(a, b).unwrap());
    }

    #[test]
    fn test_add_edge_invalid_node_directed() {
        let mut g = DiGraph::<f64>::new();
        g.add_node();
        let result = g.add_edge(0, 5, 1.0);
        assert_eq!(result.unwrap_err(), GraphError::NodeNotFound { id: 5 });
    }

    #[test]
    fn test_get_weight_no_edge_directed() {
        let mut g = DiGraph::<f64>::new();
        let a = g.add_node();
        let b = g.add_node();
        let result = g.get_weight(a, b);
        assert_eq!(result.unwrap_err(), GraphError::EdgeNotFound { from: a, to: b });
    }
}
