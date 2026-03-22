use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use scivex_frame::{DataFrame, JoinType, Series};

// ---------------------------------------------------------------------------
// DataFrame creation
// ---------------------------------------------------------------------------

fn bench_dataframe_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("dataframe_build");
    for &n in &[100usize, 1_000, 10_000] {
        group.bench_with_input(BenchmarkId::new("3_cols", n), &n, |b, &n| {
            let col1: Vec<f64> = (0..n).map(|i| i as f64).collect();
            let col2: Vec<f64> = (0..n).map(|i| (i * 2) as f64).collect();
            let col3: Vec<f64> = (0..n).map(|i| (i * 3) as f64).collect();
            b.iter(|| {
                DataFrame::builder()
                    .add_column("a", col1.clone())
                    .add_column("b", col2.clone())
                    .add_column("c", col3.clone())
                    .build()
                    .unwrap()
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Sort
// ---------------------------------------------------------------------------

fn bench_sort(c: &mut Criterion) {
    let mut group = c.benchmark_group("dataframe_sort");
    for &n in &[100usize, 1_000, 10_000] {
        let vals: Vec<f64> = (0..n).rev().map(|i| i as f64).collect();
        let df = DataFrame::builder().add_column("x", vals).build().unwrap();
        group.bench_with_input(BenchmarkId::new("single_col", n), &n, |b, _| {
            b.iter(|| df.sort_by(black_box("x"), true).unwrap());
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Filter
// ---------------------------------------------------------------------------

fn bench_filter(c: &mut Criterion) {
    let mut group = c.benchmark_group("dataframe_filter");
    for &n in &[1_000usize, 10_000] {
        let vals: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let df = DataFrame::builder()
            .add_column("x", vals.clone())
            .build()
            .unwrap();
        let half = n as f64 / 2.0;
        let mask: Vec<bool> = vals.iter().map(|v| *v > half).collect();
        group.bench_with_input(BenchmarkId::new("gt_half", n), &n, |b, _| {
            b.iter(|| df.filter(black_box(&mask)).unwrap());
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Join
// ---------------------------------------------------------------------------

fn bench_join(c: &mut Criterion) {
    let mut group = c.benchmark_group("dataframe_join");
    for &n in &[100usize, 1_000, 5_000] {
        let keys: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let vals_a: Vec<f64> = (0..n).map(|i| (i * 2) as f64).collect();
        let vals_b: Vec<f64> = (0..n).map(|i| (i * 3) as f64).collect();
        let left = DataFrame::builder()
            .add_column("key", keys.clone())
            .add_column("val_a", vals_a)
            .build()
            .unwrap();
        let right = DataFrame::builder()
            .add_column("key", keys)
            .add_column("val_b", vals_b)
            .build()
            .unwrap();
        group.bench_with_input(BenchmarkId::new("inner", n), &n, |b, _| {
            b.iter(|| {
                left.join(black_box(&right), &["key"], JoinType::Inner)
                    .unwrap()
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// GroupBy
// ---------------------------------------------------------------------------

fn bench_groupby(c: &mut Criterion) {
    let mut group = c.benchmark_group("dataframe_groupby");
    for &n in &[1_000usize, 10_000] {
        let groups: Vec<f64> = (0..n).map(|i| (i % 10) as f64).collect();
        let vals: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let df = DataFrame::builder()
            .add_column("grp", groups)
            .add_column("val", vals)
            .build()
            .unwrap();
        group.bench_with_input(BenchmarkId::new("sum_10groups", n), &n, |b, _| {
            b.iter(|| df.groupby(black_box(&["grp"])).unwrap().sum().unwrap());
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// GroupBy + mean aggregation (larger sizes)
// ---------------------------------------------------------------------------

fn bench_groupby_mean(c: &mut Criterion) {
    let mut group = c.benchmark_group("dataframe_groupby_mean");
    for &n in &[1_000usize, 10_000, 100_000] {
        let groups: Vec<f64> = (0..n).map(|i| (i % 20) as f64).collect();
        let vals: Vec<f64> = (0..n).map(|i| (i as f64) * 1.1).collect();
        let df = DataFrame::builder()
            .add_column("grp", groups)
            .add_column("val", vals)
            .build()
            .unwrap();
        group.bench_with_input(BenchmarkId::new("mean_20groups", n), &n, |b, _| {
            b.iter(|| df.groupby(black_box(&["grp"])).unwrap().mean().unwrap());
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// DataFrame creation at larger sizes
// ---------------------------------------------------------------------------

fn bench_dataframe_creation_large(c: &mut Criterion) {
    let mut group = c.benchmark_group("dataframe_creation_large");
    for &n in &[1_000usize, 10_000, 100_000] {
        let col_a: Vec<f64> = (0..n).map(|i| i as f64 * 0.5).collect();
        #[allow(clippy::cast_possible_wrap)]
        let col_b: Vec<i32> = (0..n).map(|i| (i % 100) as i32).collect();
        group.bench_with_input(BenchmarkId::new("two_cols", n), &n, |b, _| {
            b.iter(|| {
                DataFrame::builder()
                    .add_column("a", black_box(col_a.clone()))
                    .add_column("b", black_box(col_b.clone()))
                    .build()
                    .unwrap()
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Series operations
// ---------------------------------------------------------------------------

fn bench_series_create(c: &mut Criterion) {
    let mut group = c.benchmark_group("series_create");
    for &n in &[1_000usize, 10_000, 100_000] {
        let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
        group.bench_with_input(BenchmarkId::new("f64", n), &n, |b, _| {
            b.iter(|| Series::new("x", data.clone()));
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_dataframe_build,
    bench_dataframe_creation_large,
    bench_sort,
    bench_filter,
    bench_join,
    bench_groupby,
    bench_groupby_mean,
    bench_series_create,
);
criterion_main!(benches);
