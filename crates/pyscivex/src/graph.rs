//! Python bindings for scivex-graph — graph & network analysis.

use pyo3::prelude::*;
use scivex_graph::prelude::*;
use scivex_graph::{centrality, community, connectivity, flow, mst, shortest, traversal};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn py_err(e: impl std::fmt::Display) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}

// ---------------------------------------------------------------------------
// Graph (undirected)
// ---------------------------------------------------------------------------

/// Undirected weighted graph.
#[pyclass(name = "Graph")]
pub struct PyGraph {
    inner: Graph<f64>,
}

#[pymethods]
impl PyGraph {
    /// Create a new empty undirected graph.
    #[new]
    fn new() -> Self {
        Self {
            inner: Graph::new(),
        }
    }

    /// Build from edge list: [(u, v, weight), ...].
    #[staticmethod]
    fn from_edges(edges: Vec<(usize, usize, f64)>) -> PyResult<Self> {
        let g = Graph::from_edges(&edges).map_err(py_err)?;
        Ok(Self { inner: g })
    }

    /// Add a new node, return its ID.
    fn add_node(&mut self) -> usize {
        self.inner.add_node()
    }

    /// Add a weighted edge between two nodes.
    fn add_edge(&mut self, u: usize, v: usize, weight: f64) -> PyResult<()> {
        self.inner.add_edge(u, v, weight).map_err(py_err)
    }

    /// Remove a node and all its edges.
    fn remove_node(&mut self, u: usize) -> PyResult<()> {
        self.inner.remove_node(u).map_err(py_err)
    }

    /// Remove the edge between two nodes.
    fn remove_edge(&mut self, u: usize, v: usize) -> PyResult<()> {
        self.inner.remove_edge(u, v).map_err(py_err)
    }

    /// Number of nodes in the graph.
    #[getter]
    fn node_count(&self) -> usize {
        self.inner.node_count()
    }

    /// Number of edges in the graph.
    #[getter]
    fn edge_count(&self) -> usize {
        self.inner.edge_count()
    }

    /// Return neighbors as [(node_id, weight), ...].
    fn neighbors(&self, u: usize) -> PyResult<Vec<(usize, f64)>> {
        let n = self.inner.neighbors(u).map_err(py_err)?;
        Ok(n.to_vec())
    }

    /// Return the degree of a node.
    fn degree(&self, u: usize) -> PyResult<usize> {
        self.inner.degree(u).map_err(py_err)
    }

    /// Check if an edge exists.
    fn has_edge(&self, u: usize, v: usize) -> PyResult<bool> {
        self.inner.has_edge(u, v).map_err(py_err)
    }

    /// Get the weight of an edge.
    fn get_weight(&self, u: usize, v: usize) -> PyResult<f64> {
        self.inner.get_weight(u, v).map_err(py_err)
    }

    /// Return all node IDs.
    fn node_ids(&self) -> Vec<usize> {
        self.inner.node_ids().collect()
    }

    /// Return all edges as [(u, v, weight), ...].
    fn edges(&self) -> Vec<(usize, usize, f64)> {
        self.inner.edges().collect()
    }

    // -- Traversals --

    /// Breadth-first search from a start node.
    fn bfs(&self, start: usize) -> PyResult<Vec<usize>> {
        traversal::bfs(&self.inner, start).map_err(py_err)
    }

    /// Depth-first search from a start node.
    fn dfs(&self, start: usize) -> PyResult<Vec<usize>> {
        traversal::dfs(&self.inner, start).map_err(py_err)
    }

    // -- Shortest paths --

    /// Shortest paths from source via Dijkstra's algorithm.
    fn dijkstra(&self, source: usize) -> PyResult<(Vec<f64>, Vec<Option<usize>>)> {
        let r = shortest::dijkstra(&self.inner, source).map_err(py_err)?;
        Ok((r.distances, r.predecessors))
    }

    /// Shortest paths via Bellman-Ford (supports negative weights).
    fn bellman_ford(&self, source: usize) -> PyResult<(Vec<f64>, Vec<Option<usize>>)> {
        let r = shortest::bellman_ford(&self.inner, source).map_err(py_err)?;
        Ok((r.distances, r.predecessors))
    }

    /// All-pairs shortest paths.
    fn floyd_warshall(&self) -> PyResult<Vec<Vec<f64>>> {
        shortest::floyd_warshall(&self.inner).map_err(py_err)
    }

    /// Find shortest path between two nodes.
    fn shortest_path(&self, source: usize, target: usize) -> PyResult<Option<Vec<usize>>> {
        let r = shortest::dijkstra(&self.inner, source).map_err(py_err)?;
        Ok(r.path_to(source, target))
    }

    // -- Connectivity --

    /// Find all connected components.
    fn connected_components(&self) -> Vec<Vec<usize>> {
        connectivity::connected_components(&self.inner)
    }

    /// Check if the graph is connected.
    fn is_connected(&self) -> bool {
        connectivity::is_connected(&self.inner)
    }

    // -- MST --

    /// Minimum spanning tree via Kruskal's algorithm.
    #[allow(clippy::type_complexity)]
    fn kruskal(&self) -> PyResult<(Vec<(usize, usize, f64)>, f64)> {
        let m = mst::kruskal(&self.inner).map_err(py_err)?;
        Ok((m.edges, m.total_weight))
    }

    /// Minimum spanning tree via Prim's algorithm.
    #[allow(clippy::type_complexity)]
    fn prim(&self) -> PyResult<(Vec<(usize, usize, f64)>, f64)> {
        let m = mst::prim(&self.inner).map_err(py_err)?;
        Ok((m.edges, m.total_weight))
    }

    // -- Centrality --

    /// Degree centrality for each node.
    fn degree_centrality(&self) -> Vec<f64> {
        centrality::degree_centrality(&self.inner)
    }

    /// Betweenness centrality for each node.
    fn betweenness_centrality(&self) -> Vec<f64> {
        centrality::betweenness_centrality(&self.inner)
    }

    // -- Community --

    /// Community detection via label propagation.
    #[pyo3(signature = (max_iter = 100))]
    fn label_propagation(&self, max_iter: usize) -> (Vec<usize>, usize) {
        let c = community::label_propagation(&self.inner, max_iter);
        (c.labels, c.n_communities)
    }

    /// Return a string representation of the graph.
    fn __repr__(&self) -> String {
        format!(
            "Graph(nodes={}, edges={})",
            self.inner.node_count(),
            self.inner.edge_count()
        )
    }
}

// ---------------------------------------------------------------------------
// DiGraph (directed)
// ---------------------------------------------------------------------------

/// Directed weighted graph.
#[pyclass(name = "DiGraph")]
pub struct PyDiGraph {
    inner: DiGraph<f64>,
}

#[pymethods]
impl PyDiGraph {
    /// Create a new empty directed graph.
    #[new]
    fn new() -> Self {
        Self {
            inner: DiGraph::new(),
        }
    }

    /// Build from edge list: [(u, v, weight), ...].
    #[staticmethod]
    fn from_edges(edges: Vec<(usize, usize, f64)>) -> PyResult<Self> {
        let g = DiGraph::from_edges(&edges).map_err(py_err)?;
        Ok(Self { inner: g })
    }

    /// Add a new node, return its ID.
    fn add_node(&mut self) -> usize {
        self.inner.add_node()
    }

    /// Add a weighted edge between two nodes.
    fn add_edge(&mut self, u: usize, v: usize, weight: f64) -> PyResult<()> {
        self.inner.add_edge(u, v, weight).map_err(py_err)
    }

    /// Remove a node and all its edges.
    fn remove_node(&mut self, u: usize) -> PyResult<()> {
        self.inner.remove_node(u).map_err(py_err)
    }

    /// Remove the edge between two nodes.
    fn remove_edge(&mut self, u: usize, v: usize) -> PyResult<()> {
        self.inner.remove_edge(u, v).map_err(py_err)
    }

    /// Number of nodes in the graph.
    #[getter]
    fn node_count(&self) -> usize {
        self.inner.node_count()
    }

    /// Number of edges in the graph.
    #[getter]
    fn edge_count(&self) -> usize {
        self.inner.edge_count()
    }

    /// Return out-neighbors as [(node_id, weight), ...].
    fn neighbors(&self, u: usize) -> PyResult<Vec<(usize, f64)>> {
        let n = self.inner.neighbors(u).map_err(py_err)?;
        Ok(n.to_vec())
    }

    /// Return in-neighbors as [(node_id, weight), ...].
    fn in_neighbors(&self, u: usize) -> PyResult<Vec<(usize, f64)>> {
        let n = self.inner.in_neighbors(u).map_err(py_err)?;
        Ok(n.to_vec())
    }

    /// Return the out-degree of a node.
    fn out_degree(&self, u: usize) -> PyResult<usize> {
        self.inner.out_degree(u).map_err(py_err)
    }

    /// Return the in-degree of a node.
    fn in_degree(&self, u: usize) -> PyResult<usize> {
        self.inner.in_degree(u).map_err(py_err)
    }

    /// Check if an edge exists.
    fn has_edge(&self, u: usize, v: usize) -> PyResult<bool> {
        self.inner.has_edge(u, v).map_err(py_err)
    }

    /// Get the weight of an edge.
    fn get_weight(&self, u: usize, v: usize) -> PyResult<f64> {
        self.inner.get_weight(u, v).map_err(py_err)
    }

    /// Return all node IDs.
    fn node_ids(&self) -> Vec<usize> {
        self.inner.node_ids().collect()
    }

    /// Return all edges as [(u, v, weight), ...].
    fn edges(&self) -> Vec<(usize, usize, f64)> {
        self.inner.edges().collect()
    }

    // -- Traversals --

    /// Breadth-first search from a start node.
    fn bfs(&self, start: usize) -> PyResult<Vec<usize>> {
        traversal::bfs_directed(&self.inner, start).map_err(py_err)
    }

    /// Depth-first search from a start node.
    fn dfs(&self, start: usize) -> PyResult<Vec<usize>> {
        traversal::dfs_directed(&self.inner, start).map_err(py_err)
    }

    /// Topological sort (DAG only).
    fn topological_sort(&self) -> PyResult<Vec<usize>> {
        traversal::topological_sort(&self.inner).map_err(py_err)
    }

    // -- Shortest paths --

    /// Shortest paths from source via Dijkstra's algorithm.
    fn dijkstra(&self, source: usize) -> PyResult<(Vec<f64>, Vec<Option<usize>>)> {
        let r = shortest::dijkstra_directed(&self.inner, source).map_err(py_err)?;
        Ok((r.distances, r.predecessors))
    }

    /// Shortest paths via Bellman-Ford (supports negative weights).
    fn bellman_ford(&self, source: usize) -> PyResult<(Vec<f64>, Vec<Option<usize>>)> {
        let r = shortest::bellman_ford_directed(&self.inner, source).map_err(py_err)?;
        Ok((r.distances, r.predecessors))
    }

    /// All-pairs shortest paths.
    fn floyd_warshall(&self) -> PyResult<Vec<Vec<f64>>> {
        shortest::floyd_warshall_directed(&self.inner).map_err(py_err)
    }

    /// Find shortest path between two nodes.
    fn shortest_path(&self, source: usize, target: usize) -> PyResult<Option<Vec<usize>>> {
        let r = shortest::dijkstra_directed(&self.inner, source).map_err(py_err)?;
        Ok(r.path_to(source, target))
    }

    // -- Connectivity --

    /// Find SCCs via Tarjan's algorithm.
    fn strongly_connected_components(&self) -> Vec<Vec<usize>> {
        connectivity::strongly_connected_components(&self.inner)
    }

    /// Check if the graph is strongly connected.
    fn is_strongly_connected(&self) -> bool {
        connectivity::is_strongly_connected(&self.inner)
    }

    /// Find all weakly connected components.
    fn weakly_connected_components(&self) -> PyResult<Vec<Vec<usize>>> {
        connectivity::weakly_connected_components(&self.inner).map_err(py_err)
    }

    // -- Centrality --

    /// In-degree centrality for each node.
    fn in_degree_centrality(&self) -> Vec<f64> {
        centrality::in_degree_centrality(&self.inner)
    }

    /// Out-degree centrality for each node.
    fn out_degree_centrality(&self) -> Vec<f64> {
        centrality::out_degree_centrality(&self.inner)
    }

    /// PageRank centrality scores.
    #[pyo3(signature = (damping = 0.85, max_iter = 100, tol = 1e-6))]
    fn pagerank(&self, damping: f64, max_iter: usize, tol: f64) -> PyResult<Vec<f64>> {
        centrality::pagerank(&self.inner, damping, max_iter, tol).map_err(py_err)
    }

    // -- Flow --

    /// Maximum flow from source to sink.
    #[allow(clippy::type_complexity)]
    fn max_flow(&self, source: usize, sink: usize) -> PyResult<(f64, Vec<(usize, usize, f64)>)> {
        let r = flow::max_flow(&self.inner, source, sink).map_err(py_err)?;
        // Return max_flow value and list of (from, to, flow) for non-zero flows
        let n = r.n;
        let mut flows = Vec::new();
        for i in 0..n {
            for j in 0..n {
                let f = r.edge_flow(i, j);
                if f > 0.0 {
                    flows.push((i, j, f));
                }
            }
        }
        Ok((r.max_flow, flows))
    }

    /// Return a string representation of the directed graph.
    fn __repr__(&self) -> String {
        format!(
            "DiGraph(nodes={}, edges={})",
            self.inner.node_count(),
            self.inner.edge_count()
        )
    }
}

// ---------------------------------------------------------------------------
// Module-level functions
// ---------------------------------------------------------------------------

/// Maximum bipartite matching.
#[pyfunction]
fn bipartite_matching(
    left_size: usize,
    right_size: usize,
    edges: Vec<(usize, usize)>,
) -> (Vec<(usize, usize)>, usize) {
    let r = flow::bipartite_matching(left_size, right_size, &edges);
    (r.pairs, r.size)
}

// ---------------------------------------------------------------------------
// Register submodule
// ---------------------------------------------------------------------------

/// Register the `graph` submodule with its classes and functions.
pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "graph")?;

    m.add_class::<PyGraph>()?;
    m.add_class::<PyDiGraph>()?;
    m.add_function(wrap_pyfunction!(bipartite_matching, &m)?)?;

    parent.add_submodule(&m)?;
    Ok(())
}
