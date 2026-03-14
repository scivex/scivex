use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use scivex_graph::centrality;
use scivex_graph::shortest::dijkstra;
use scivex_graph::traversal;
use scivex_graph::{DiGraph, Graph};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn build_graph(n: usize) -> Graph<f64> {
    let mut g = Graph::new();
    for _ in 0..n {
        g.add_node();
    }
    // Create a connected graph with ~3*n edges
    for i in 0..n {
        let targets = [(i + 1) % n, (i + 3) % n, (i + 7) % n];
        for &t in &targets {
            let _ = g.add_edge(i, t, 1.0 + (i % 5) as f64);
        }
    }
    g
}

fn build_digraph(n: usize) -> DiGraph<f64> {
    let mut g = DiGraph::new();
    for _ in 0..n {
        g.add_node();
    }
    for i in 0..n {
        let targets = [(i + 1) % n, (i + 3) % n, (i + 7) % n];
        for &t in &targets {
            let _ = g.add_edge(i, t, 1.0 + (i % 5) as f64);
        }
    }
    g
}

// ---------------------------------------------------------------------------
// BFS / DFS
// ---------------------------------------------------------------------------

fn bench_bfs(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_bfs");
    for &n in &[100usize, 500, 1_000, 5_000] {
        let g = build_graph(n);
        group.bench_with_input(BenchmarkId::new("undirected", n), &n, |b, _| {
            b.iter(|| traversal::bfs(black_box(&g), 0).unwrap());
        });
    }
    group.finish();
}

fn bench_dfs(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_dfs");
    for &n in &[100usize, 500, 1_000, 5_000] {
        let g = build_graph(n);
        group.bench_with_input(BenchmarkId::new("undirected", n), &n, |b, _| {
            b.iter(|| traversal::dfs(black_box(&g), 0).unwrap());
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Dijkstra
// ---------------------------------------------------------------------------

fn bench_dijkstra(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_dijkstra");
    for &n in &[100usize, 500, 1_000] {
        let g = build_graph(n);
        group.bench_with_input(BenchmarkId::new("undirected", n), &n, |b, _| {
            b.iter(|| dijkstra(black_box(&g), 0).unwrap());
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// PageRank
// ---------------------------------------------------------------------------

fn bench_pagerank(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_pagerank");
    for &n in &[100usize, 500, 1_000] {
        let g = build_digraph(n);
        group.bench_with_input(BenchmarkId::new("directed", n), &n, |b, _| {
            b.iter(|| centrality::pagerank(black_box(&g), 0.85, 50, 1e-6).unwrap());
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Graph construction
// ---------------------------------------------------------------------------

fn bench_graph_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_build");
    for &n in &[100usize, 1_000, 5_000] {
        group.bench_with_input(BenchmarkId::new("nodes_3edges_each", n), &n, |b, &n| {
            b.iter(|| build_graph(black_box(n)));
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_bfs,
    bench_dfs,
    bench_dijkstra,
    bench_pagerank,
    bench_graph_build,
);
criterion_main!(benches);
