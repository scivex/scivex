use std::collections::BinaryHeap;

use scivex_core::Float;

use crate::error::Result;
use crate::graph::Graph;

/// Minimum spanning tree result.
#[derive(Debug, Clone)]
pub struct Mst<T> {
    /// The edges in the minimum spanning tree as `(u, v, weight)` triples.
    pub edges: Vec<(usize, usize, T)>,
    /// Sum of all edge weights in the spanning tree.
    pub total_weight: T,
}

/// Union-Find (disjoint set) data structure with path compression and
/// union by rank.
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    fn union(&mut self, x: usize, y: usize) -> bool {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx == ry {
            return false;
        }
        match self.rank[rx].cmp(&self.rank[ry]) {
            std::cmp::Ordering::Less => self.parent[rx] = ry,
            std::cmp::Ordering::Greater => self.parent[ry] = rx,
            std::cmp::Ordering::Equal => {
                self.parent[ry] = rx;
                self.rank[rx] += 1;
            }
        }
        true
    }
}

/// Kruskal's MST algorithm. Sorts edges by weight and uses Union-Find.
///
/// For disconnected graphs, returns a minimum spanning forest.
pub fn kruskal<T: Float>(graph: &Graph<T>) -> Result<Mst<T>> {
    let mut edges: Vec<(usize, usize, T)> = graph.edges().collect();
    edges.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

    let n = graph.capacity();
    let mut uf = UnionFind::new(n);
    let mut mst_edges = Vec::new();
    let mut total = T::zero();

    for (u, v, w) in edges {
        if uf.union(u, v) {
            total += w;
            mst_edges.push((u, v, w));
        }
    }

    Ok(Mst {
        edges: mst_edges,
        total_weight: total,
    })
}

/// Min-heap state for Prim's algorithm.
#[derive(Debug, Clone, Copy)]
struct PrimState<T> {
    weight: T,
    node: usize,
    from: usize,
}

impl<T: Float> PartialEq for PrimState<T> {
    fn eq(&self, other: &Self) -> bool {
        self.node == other.node && self.weight == other.weight
    }
}

impl<T: Float> Eq for PrimState<T> {}

impl<T: Float> PartialOrd for PrimState<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Float> Ord for PrimState<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .weight
            .partial_cmp(&self.weight)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Prim's MST algorithm using a min-heap.
///
/// For disconnected graphs, returns a minimum spanning forest by restarting
/// from unvisited nodes.
pub fn prim<T: Float>(graph: &Graph<T>) -> Result<Mst<T>> {
    let n = graph.capacity();
    let mut in_mst = vec![false; n];
    let mut mst_edges = Vec::new();
    let mut total = T::zero();
    let mut heap = BinaryHeap::new();

    for start in graph.node_ids() {
        if in_mst[start] {
            continue;
        }

        in_mst[start] = true;
        for &(v, w) in graph.adj_raw(start) {
            if !in_mst[v] {
                heap.push(PrimState {
                    weight: w,
                    node: v,
                    from: start,
                });
            }
        }

        while let Some(PrimState { weight, node, from }) = heap.pop() {
            if in_mst[node] {
                continue;
            }
            in_mst[node] = true;
            total += weight;
            mst_edges.push((from, node, weight));

            for &(v, w) in graph.adj_raw(node) {
                if !in_mst[v] {
                    heap.push(PrimState {
                        weight: w,
                        node: v,
                        from: node,
                    });
                }
            }
        }
    }

    Ok(Mst {
        edges: mst_edges,
        total_weight: total,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kruskal_correct_weight() {
        // Triangle: 0-1 (1), 1-2 (2), 0-2 (3)
        // MST should pick edges with weight 1 and 2 => total 3
        let g = Graph::from_edges(&[(0, 1, 1.0), (1, 2, 2.0), (0, 2, 3.0)]).unwrap();
        let mst = kruskal(&g).unwrap();

        assert_eq!(mst.edges.len(), 2);
        assert!((mst.total_weight - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_prim_agrees_with_kruskal() {
        let g = Graph::from_edges(&[
            (0, 1, 4.0),
            (0, 2, 1.0),
            (1, 2, 2.0),
            (1, 3, 5.0),
            (2, 3, 8.0),
        ])
        .unwrap();

        let k = kruskal(&g).unwrap();
        let p = prim(&g).unwrap();

        assert!((k.total_weight - p.total_weight).abs() < 1e-10);
    }

    #[test]
    fn test_spanning_forest_disconnected() {
        let mut g = Graph::<f64>::new();
        for _ in 0..4 {
            g.add_node();
        }
        g.add_edge(0, 1, 1.0).unwrap();
        g.add_edge(2, 3, 2.0).unwrap();

        let mst = kruskal(&g).unwrap();
        assert_eq!(mst.edges.len(), 2);
        assert!((mst.total_weight - 3.0).abs() < 1e-10);

        let mst_p = prim(&g).unwrap();
        assert_eq!(mst_p.edges.len(), 2);
        assert!((mst_p.total_weight - 3.0).abs() < 1e-10);
    }
}
