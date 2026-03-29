"""Tests for pyscivex graph & network analysis — graph submodule."""

import math
import pyscivex as sv


# ===========================================================================
# UNDIRECTED GRAPH
# ===========================================================================


class TestGraph:
    def test_create_empty(self):
        g = sv.graph.Graph()
        assert g.node_count == 0
        assert g.edge_count == 0

    def test_add_nodes_edges(self):
        g = sv.graph.Graph()
        n0 = g.add_node()
        n1 = g.add_node()
        n2 = g.add_node()
        g.add_edge(n0, n1, 1.0)
        g.add_edge(n1, n2, 2.0)
        assert g.node_count == 3
        assert g.edge_count == 2

    def test_from_edges(self):
        g = sv.graph.Graph.from_edges([(0, 1, 1.0), (1, 2, 2.0), (2, 0, 3.0)])
        assert g.node_count == 3
        assert g.edge_count == 3

    def test_neighbors(self):
        g = sv.graph.Graph.from_edges([(0, 1, 1.0), (0, 2, 2.0)])
        n = g.neighbors(0)
        neighbor_ids = [x[0] for x in n]
        assert 1 in neighbor_ids
        assert 2 in neighbor_ids

    def test_degree(self):
        g = sv.graph.Graph.from_edges([(0, 1, 1.0), (0, 2, 2.0), (1, 2, 3.0)])
        assert g.degree(0) == 2
        assert g.degree(1) == 2

    def test_has_edge(self):
        g = sv.graph.Graph.from_edges([(0, 1, 1.0)])
        assert g.has_edge(0, 1)
        assert g.has_edge(1, 0)  # undirected

    def test_get_weight(self):
        g = sv.graph.Graph.from_edges([(0, 1, 3.14)])
        assert abs(g.get_weight(0, 1) - 3.14) < 1e-10

    def test_remove_edge(self):
        g = sv.graph.Graph.from_edges([(0, 1, 1.0), (1, 2, 2.0)])
        g.remove_edge(0, 1)
        assert g.edge_count == 1

    def test_remove_node(self):
        g = sv.graph.Graph.from_edges([(0, 1, 1.0), (1, 2, 2.0)])
        g.remove_node(1)
        assert g.node_count == 2

    def test_node_ids(self):
        g = sv.graph.Graph.from_edges([(0, 1, 1.0), (1, 2, 2.0)])
        ids = g.node_ids()
        assert 0 in ids
        assert 1 in ids
        assert 2 in ids

    def test_edges_list(self):
        g = sv.graph.Graph.from_edges([(0, 1, 1.0), (1, 2, 2.0)])
        edges = g.edges()
        assert len(edges) == 2

    def test_repr(self):
        g = sv.graph.Graph.from_edges([(0, 1, 1.0)])
        assert "Graph" in repr(g)


# ===========================================================================
# DIRECTED GRAPH
# ===========================================================================


class TestDiGraph:
    def test_create_empty(self):
        g = sv.graph.DiGraph()
        assert g.node_count == 0

    def test_add_directed_edge(self):
        g = sv.graph.DiGraph()
        g.add_node()
        g.add_node()
        g.add_edge(0, 1, 1.0)
        assert g.has_edge(0, 1)

    def test_from_edges(self):
        g = sv.graph.DiGraph.from_edges([(0, 1, 1.0), (1, 2, 2.0)])
        assert g.node_count == 3
        assert g.edge_count == 2

    def test_in_out_degree(self):
        g = sv.graph.DiGraph.from_edges([(0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0)])
        assert g.out_degree(0) == 2
        assert g.in_degree(2) == 2

    def test_in_neighbors(self):
        g = sv.graph.DiGraph.from_edges([(0, 2, 1.0), (1, 2, 1.0)])
        n = g.in_neighbors(2)
        sources = [x[0] for x in n]
        assert 0 in sources
        assert 1 in sources

    def test_repr(self):
        g = sv.graph.DiGraph.from_edges([(0, 1, 1.0)])
        assert "DiGraph" in repr(g)


# ===========================================================================
# TRAVERSALS
# ===========================================================================


class TestTraversals:
    def test_bfs(self):
        g = sv.graph.Graph.from_edges([(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)])
        order = g.bfs(0)
        assert order[0] == 0
        assert len(order) == 4

    def test_dfs(self):
        g = sv.graph.Graph.from_edges([(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)])
        order = g.dfs(0)
        assert order[0] == 0
        assert len(order) == 4

    def test_bfs_directed(self):
        g = sv.graph.DiGraph.from_edges([(0, 1, 1.0), (1, 2, 1.0)])
        order = g.bfs(0)
        assert order == [0, 1, 2]

    def test_dfs_directed(self):
        g = sv.graph.DiGraph.from_edges([(0, 1, 1.0), (1, 2, 1.0)])
        order = g.dfs(0)
        assert 0 in order
        assert len(order) == 3

    def test_topological_sort(self):
        g = sv.graph.DiGraph.from_edges([(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)])
        order = g.topological_sort()
        assert order.index(0) < order.index(1)
        assert order.index(1) < order.index(2)


# ===========================================================================
# SHORTEST PATHS
# ===========================================================================


class TestShortestPaths:
    def test_dijkstra(self):
        g = sv.graph.Graph.from_edges([
            (0, 1, 1.0), (1, 2, 2.0), (0, 2, 10.0)
        ])
        dists, preds = g.dijkstra(0)
        assert abs(dists[0]) < 1e-10  # dist to self
        assert abs(dists[1] - 1.0) < 1e-10
        assert abs(dists[2] - 3.0) < 1e-10  # via node 1

    def test_bellman_ford(self):
        g = sv.graph.Graph.from_edges([
            (0, 1, 1.0), (1, 2, 2.0), (0, 2, 10.0)
        ])
        dists, _ = g.bellman_ford(0)
        assert abs(dists[2] - 3.0) < 1e-10

    def test_floyd_warshall(self):
        g = sv.graph.Graph.from_edges([
            (0, 1, 1.0), (1, 2, 2.0), (0, 2, 10.0)
        ])
        dist_matrix = g.floyd_warshall()
        assert abs(dist_matrix[0][2] - 3.0) < 1e-10

    def test_shortest_path(self):
        g = sv.graph.Graph.from_edges([
            (0, 1, 1.0), (1, 2, 1.0), (0, 2, 10.0)
        ])
        path = g.shortest_path(0, 2)
        assert path is not None
        assert path[0] == 0
        assert path[-1] == 2

    def test_dijkstra_directed(self):
        g = sv.graph.DiGraph.from_edges([
            (0, 1, 1.0), (1, 2, 2.0)
        ])
        dists, _ = g.dijkstra(0)
        assert abs(dists[2] - 3.0) < 1e-10


# ===========================================================================
# CONNECTIVITY
# ===========================================================================


class TestConnectivity:
    def test_connected_components(self):
        g = sv.graph.Graph.from_edges([
            (0, 1, 1.0), (2, 3, 1.0)
        ])
        comps = g.connected_components()
        assert len(comps) == 2

    def test_is_connected(self):
        g1 = sv.graph.Graph.from_edges([(0, 1, 1.0), (1, 2, 1.0)])
        assert g1.is_connected()
        g2 = sv.graph.Graph.from_edges([(0, 1, 1.0), (2, 3, 1.0)])
        assert not g2.is_connected()

    def test_strongly_connected(self):
        # A cycle is strongly connected
        g = sv.graph.DiGraph.from_edges([
            (0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)
        ])
        assert g.is_strongly_connected()
        sccs = g.strongly_connected_components()
        assert len(sccs) == 1

    def test_not_strongly_connected(self):
        g = sv.graph.DiGraph.from_edges([(0, 1, 1.0), (1, 2, 1.0)])
        assert not g.is_strongly_connected()
        sccs = g.strongly_connected_components()
        assert len(sccs) == 3  # each node is its own SCC

    def test_weakly_connected(self):
        g = sv.graph.DiGraph.from_edges([(0, 1, 1.0), (1, 2, 1.0)])
        comps = g.weakly_connected_components()
        assert len(comps) == 1  # all reachable ignoring direction


# ===========================================================================
# MST
# ===========================================================================


class TestMst:
    def test_kruskal(self):
        g = sv.graph.Graph.from_edges([
            (0, 1, 1.0), (1, 2, 2.0), (0, 2, 3.0)
        ])
        edges, total = g.kruskal()
        assert len(edges) == 2  # n-1 edges for n=3
        assert abs(total - 3.0) < 1e-10  # 1 + 2

    def test_prim(self):
        g = sv.graph.Graph.from_edges([
            (0, 1, 1.0), (1, 2, 2.0), (0, 2, 3.0)
        ])
        edges, total = g.prim()
        assert len(edges) == 2
        assert abs(total - 3.0) < 1e-10


# ===========================================================================
# CENTRALITY
# ===========================================================================


class TestCentrality:
    def test_degree_centrality(self):
        g = sv.graph.Graph.from_edges([
            (0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0)
        ])
        dc = g.degree_centrality()
        # Node 0 has degree 3, others have degree 1
        assert dc[0] > dc[1]

    def test_betweenness_centrality(self):
        # Star graph: node 0 connects to 1,2,3
        g = sv.graph.Graph.from_edges([
            (0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0)
        ])
        bc = g.betweenness_centrality()
        assert bc[0] >= bc[1]  # center has highest betweenness

    def test_pagerank(self):
        g = sv.graph.DiGraph.from_edges([
            (0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)
        ])
        pr = g.pagerank(damping=0.85, max_iter=100, tol=1e-6)
        assert len(pr) > 0
        assert all(p >= 0 for p in pr)
        # In a cycle, all nodes should have equal rank
        assert abs(pr[0] - pr[1]) < 0.01

    def test_in_out_degree_centrality(self):
        g = sv.graph.DiGraph.from_edges([
            (0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0)
        ])
        idc = g.in_degree_centrality()
        odc = g.out_degree_centrality()
        assert len(idc) > 0
        assert len(odc) > 0


# ===========================================================================
# COMMUNITY
# ===========================================================================


class TestCommunity:
    def test_label_propagation(self):
        # Two clusters connected by one edge
        g = sv.graph.Graph.from_edges([
            (0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0),  # cluster 1
            (3, 4, 1.0), (4, 5, 1.0), (3, 5, 1.0),  # cluster 2
            (2, 3, 0.1),  # weak bridge
        ])
        labels, n_communities = g.label_propagation(100)
        assert n_communities >= 1
        assert len(labels) > 0


# ===========================================================================
# FLOW
# ===========================================================================


class TestFlow:
    def test_max_flow(self):
        g = sv.graph.DiGraph.from_edges([
            (0, 1, 10.0), (0, 2, 5.0),
            (1, 3, 5.0), (2, 3, 10.0),
        ])
        max_flow, flows = g.max_flow(0, 3)
        assert max_flow > 0
        assert abs(max_flow - 10.0) < 1e-10  # min(10,5) + min(5,10)

    def test_bipartite_matching(self):
        # 3 left nodes, 3 right nodes
        edges = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)]
        pairs, size = sv.graph.bipartite_matching(3, 3, edges)
        assert size == 3  # perfect matching possible
        assert len(pairs) == 3


# ===========================================================================
# INTEGRATION
# ===========================================================================


class TestIntegration:
    def test_all_accessible(self):
        items = [
            sv.graph.Graph,
            sv.graph.DiGraph,
            sv.graph.bipartite_matching,
        ]
        for item in items:
            assert item is not None

    def test_workflow(self):
        """Full graph analysis workflow."""
        g = sv.graph.Graph.from_edges([
            (0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0),
            (3, 4, 1.0), (4, 0, 1.0),
        ])
        # Traversal
        bfs_order = g.bfs(0)
        assert len(bfs_order) == 5
        # Shortest path
        path = g.shortest_path(0, 3)
        assert path is not None
        # MST
        _, total = g.kruskal()
        assert total > 0
        # Connectivity
        assert g.is_connected()
        # Centrality
        dc = g.degree_centrality()
        assert len(dc) == 5
