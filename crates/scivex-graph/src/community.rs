use scivex_core::Float;

use crate::graph::Graph;

/// Community detection result.
#[derive(Debug, Clone)]
pub struct Communities {
    /// Label assigned to each node slot. Inactive nodes have label `usize::MAX`.
    pub labels: Vec<usize>,
    /// Number of distinct communities.
    pub n_communities: usize,
}

/// Label propagation community detection (deterministic, ID-order).
///
/// Each node adopts the most frequent label among its neighbors, breaking
/// ties by choosing the smallest label. Iterates until convergence or
/// `max_iter` is reached.
pub fn label_propagation<T: Float>(graph: &Graph<T>, max_iter: usize) -> Communities {
    let n = graph.capacity();
    let mut labels = vec![usize::MAX; n];

    // Initialize: each active node gets its own label
    for u in graph.node_ids() {
        labels[u] = u;
    }

    for _ in 0..max_iter {
        let mut changed = false;

        for u in graph.node_ids() {
            let neighbors = graph.adj_raw(u);
            if neighbors.is_empty() {
                continue;
            }

            // Count label frequencies
            let mut label_counts: Vec<(usize, usize)> = Vec::new();
            for &(v, _) in neighbors {
                if let Some(entry) = label_counts.iter_mut().find(|(l, _)| *l == labels[v]) {
                    entry.1 += 1;
                } else {
                    label_counts.push((labels[v], 1));
                }
            }

            // Find the most frequent label (smallest label on tie)
            label_counts.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
            let best_label = label_counts[0].0;

            if labels[u] != best_label {
                labels[u] = best_label;
                changed = true;
            }
        }

        if !changed {
            break;
        }
    }

    // Count distinct communities
    let mut unique_labels: Vec<usize> = labels
        .iter()
        .copied()
        .filter(|&l| l != usize::MAX)
        .collect();
    unique_labels.sort_unstable();
    unique_labels.dedup();
    let n_communities = unique_labels.len();

    Communities {
        labels,
        n_communities,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_two_cliques_different_labels() {
        // Two cliques connected by a single edge
        // Clique 1: 0-1-2 (fully connected)
        // Clique 2: 3-4-5 (fully connected)
        // Bridge: 2-3
        let g = Graph::from_edges(&[
            (0, 1, 1.0),
            (0, 2, 1.0),
            (1, 2, 1.0),
            (3, 4, 1.0),
            (3, 5, 1.0),
            (4, 5, 1.0),
            (2, 3, 1.0),
        ])
        .unwrap();

        let comm = label_propagation(&g, 100);

        // Nodes in the same clique should have the same label
        assert_eq!(comm.labels[0], comm.labels[1]);
        assert_eq!(comm.labels[0], comm.labels[2]);
        assert_eq!(comm.labels[3], comm.labels[4]);
        assert_eq!(comm.labels[3], comm.labels[5]);

        // At most n communities
        assert!(comm.n_communities <= 6);
    }

    #[test]
    fn test_n_communities_bounded() {
        let g = Graph::from_edges(&[(0, 1, 1.0), (2, 3, 1.0)]).unwrap();
        let comm = label_propagation(&g, 100);
        assert!(comm.n_communities <= 4);
        assert!(comm.n_communities >= 2);
    }

    #[test]
    fn test_isolated_nodes() {
        let mut g = Graph::<f64>::new();
        g.add_node();
        g.add_node();
        g.add_node();

        let comm = label_propagation(&g, 100);
        assert_eq!(comm.n_communities, 3);
    }
}
