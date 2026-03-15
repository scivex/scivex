//! Graph algorithm bindings for JavaScript.

use scivex_graph::{DiGraph, Graph};
use wasm_bindgen::prelude::*;

/// An undirected, weighted graph.
#[wasm_bindgen]
pub struct WasmGraph {
    inner: Graph<f64>,
}

impl Default for WasmGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl WasmGraph {
    /// Create an empty undirected graph.
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmGraph {
        WasmGraph {
            inner: Graph::new(),
        }
    }

    /// Add a node. Returns the node ID.
    #[wasm_bindgen(js_name = "addNode")]
    pub fn add_node(&mut self) -> usize {
        self.inner.add_node()
    }

    /// Add an undirected edge.
    #[wasm_bindgen(js_name = "addEdge")]
    pub fn add_edge(&mut self, u: usize, v: usize, weight: f64) -> Result<(), JsError> {
        self.inner
            .add_edge(u, v, weight)
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Number of nodes.
    #[wasm_bindgen(js_name = "nodeCount")]
    pub fn node_count(&self) -> usize {
        self.inner.node_count()
    }

    /// Number of edges.
    #[wasm_bindgen(js_name = "edgeCount")]
    pub fn edge_count(&self) -> usize {
        self.inner.edge_count()
    }

    /// BFS traversal from a starting node.
    pub fn bfs(&self, start: usize) -> Result<Vec<usize>, JsError> {
        scivex_graph::traversal::bfs(&self.inner, start).map_err(|e| JsError::new(&e.to_string()))
    }

    /// DFS traversal from a starting node.
    pub fn dfs(&self, start: usize) -> Result<Vec<usize>, JsError> {
        scivex_graph::traversal::dfs(&self.inner, start).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Shortest path distances from source (Dijkstra).
    pub fn dijkstra(&self, source: usize) -> Result<Vec<f64>, JsError> {
        let result = scivex_graph::shortest::dijkstra(&self.inner, source)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(result.distances)
    }

    /// Find connected components. Returns flat array: [component_id_for_node_0, ...].
    #[wasm_bindgen(js_name = "connectedComponents")]
    pub fn connected_components(&self) -> Vec<usize> {
        let components = scivex_graph::connectivity::connected_components(&self.inner);
        let n = self.inner.node_count();
        let mut labels = vec![0usize; n];
        for (comp_id, component) in components.iter().enumerate() {
            for &node in component {
                if node < n {
                    labels[node] = comp_id;
                }
            }
        }
        labels
    }

    /// Is the graph connected?
    #[wasm_bindgen(js_name = "isConnected")]
    pub fn is_connected(&self) -> bool {
        scivex_graph::connectivity::is_connected(&self.inner)
    }

    /// Minimum spanning tree total weight (Kruskal).
    #[wasm_bindgen(js_name = "mstWeight")]
    pub fn mst_weight(&self) -> Result<f64, JsError> {
        let mst =
            scivex_graph::mst::kruskal(&self.inner).map_err(|e| JsError::new(&e.to_string()))?;
        Ok(mst.total_weight)
    }
}

/// A directed, weighted graph.
#[wasm_bindgen]
pub struct WasmDiGraph {
    inner: DiGraph<f64>,
}

impl Default for WasmDiGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl WasmDiGraph {
    /// Create an empty directed graph.
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmDiGraph {
        WasmDiGraph {
            inner: DiGraph::new(),
        }
    }

    /// Add a node. Returns the node ID.
    #[wasm_bindgen(js_name = "addNode")]
    pub fn add_node(&mut self) -> usize {
        self.inner.add_node()
    }

    /// Add a directed edge from u to v.
    #[wasm_bindgen(js_name = "addEdge")]
    pub fn add_edge(&mut self, u: usize, v: usize, weight: f64) -> Result<(), JsError> {
        self.inner
            .add_edge(u, v, weight)
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Number of nodes.
    #[wasm_bindgen(js_name = "nodeCount")]
    pub fn node_count(&self) -> usize {
        self.inner.node_count()
    }

    /// Number of edges.
    #[wasm_bindgen(js_name = "edgeCount")]
    pub fn edge_count(&self) -> usize {
        self.inner.edge_count()
    }

    /// Topological sort (returns error if cycle detected).
    #[wasm_bindgen(js_name = "topologicalSort")]
    pub fn topological_sort(&self) -> Result<Vec<usize>, JsError> {
        scivex_graph::traversal::topological_sort(&self.inner)
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Max flow from source to sink (Edmonds-Karp).
    #[wasm_bindgen(js_name = "maxFlow")]
    pub fn max_flow(&self, source: usize, sink: usize) -> Result<f64, JsError> {
        let result = scivex_graph::flow::max_flow(&self.inner, source, sink)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(result.max_flow)
    }

    /// PageRank scores.
    pub fn pagerank(&self, damping: f64, iterations: usize) -> Result<Vec<f64>, JsError> {
        scivex_graph::centrality::pagerank(&self.inner, damping, iterations, 1e-6)
            .map_err(|e| JsError::new(&e.to_string()))
    }
}
