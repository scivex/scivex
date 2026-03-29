# scivex-graph

Graph data structures and algorithms for Scivex. Directed and undirected
weighted graphs with comprehensive algorithm support.

## Highlights

- **Graph types** — Undirected (`Graph`) and directed (`DiGraph`) with weighted edges
- **Traversals** — BFS, DFS, topological sort
- **Shortest paths** — Dijkstra, Bellman-Ford, Floyd-Warshall, A*
- **Connectivity** — Connected components, strongly connected components (Tarjan)
- **MST** — Kruskal's and Prim's minimum spanning tree
- **Centrality** — Degree, betweenness, PageRank
- **Community** — Label propagation community detection
- **Network flow** — Max flow (Edmonds-Karp), bipartite matching

## Usage

```rust
use scivex_graph::prelude::*;

let g = Graph::from_edges(&[(0, 1, 1.0), (1, 2, 2.0), (0, 2, 4.0)]).unwrap();

let result = shortest::dijkstra(&g, 0).unwrap();
let components = connectivity::connected_components(&g);
let mst = mst::kruskal(&g).unwrap();
let centrality = centrality::betweenness_centrality(&g);
```

## License

MIT
