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

pub mod centrality;
pub mod community;
pub mod connectivity;
pub mod digraph;
pub mod error;
pub mod graph;
pub mod mst;
pub mod shortest;
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
