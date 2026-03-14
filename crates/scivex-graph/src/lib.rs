//! `scivex-graph` — Graph data structures, algorithms, and network analysis.
//!
//! Provides from-scratch implementations of:
//! - Undirected and directed weighted graphs (`Graph`, `DiGraph`)
//! - Traversals: BFS, DFS, topological sort
//! - Shortest paths: Dijkstra, Bellman-Ford, Floyd-Warshall
//! - Connectivity: connected components, strongly connected components
//! - Centrality: degree, betweenness, PageRank
//! - Minimum spanning trees: Kruskal, Prim
//! - Community detection: label propagation

/// Centrality measures (degree, betweenness, PageRank).
pub mod centrality;
/// Community detection (label propagation).
pub mod community;
/// Connectivity analysis (connected / strongly-connected components).
pub mod connectivity;
/// Directed weighted graph.
pub mod digraph;
/// Graph error types.
pub mod error;
/// Network flow algorithms (max-flow, bipartite matching).
pub mod flow;
/// Undirected weighted graph.
pub mod graph;
/// Minimum spanning trees (Kruskal, Prim).
pub mod mst;
/// Shortest path algorithms (Dijkstra, Bellman-Ford, Floyd-Warshall).
pub mod shortest;
/// Graph traversals (BFS, DFS, topological sort).
pub mod traversal;

pub use digraph::DiGraph;
pub use error::{GraphError, Result};
pub use graph::Graph;

/// Items intended for glob-import: `use scivex_graph::prelude::*;`
pub mod prelude {
    pub use crate::digraph::DiGraph;
    pub use crate::error::{GraphError, Result};
    pub use crate::graph::Graph;
    pub use crate::mst::Mst;
    pub use crate::shortest::ShortestPathResult;
}
