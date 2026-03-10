# scivex-graph

Graph data structures and algorithms for Scivex. Directed and undirected
weighted graphs with traversals, shortest paths, centrality, and more.

## Highlights

- **Graph types** — `Graph<T>` (undirected) and `DiGraph<T>` (directed), weighted edges
- **Traversals** — BFS, DFS (recursive + iterative), topological sort
- **Shortest paths** — Dijkstra, Bellman-Ford, Floyd-Warshall with path reconstruction
- **Centrality** — Degree, betweenness (Brandes), closeness, PageRank
- **Connectivity** — Connected components, strongly connected components (Tarjan)
- **MST** — Kruskal and Prim algorithms
- **Community** — Label propagation community detection

## Usage

```rust
use scivex_graph::prelude::*;

let mut g = Graph::<f64>::new();
let a = g.add_node();
let b = g.add_node();
let c = g.add_node();
g.add_edge(a, b, 1.0);
g.add_edge(b, c, 2.0);
g.add_edge(a, c, 4.0);

// Shortest path
let result = dijkstra(&g, a).unwrap();
let path = result.path_to(a, c);

// Centrality
let pr = pagerank(&g, 0.85, 100).unwrap();

// Minimum spanning tree
let mst = kruskal(&g).unwrap();
```

## License

MIT
