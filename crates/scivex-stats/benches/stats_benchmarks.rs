#![allow(clippy::cast_lossless)]

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use scivex_core::Tensor;
use scivex_core::random::Rng;
use scivex_stats::correlation::{self, CorrelationMethod};
use scivex_stats::descriptive;
use scivex_stats::distributions::{Distribution, Normal, Uniform};
use scivex_stats::hypothesis;

// ---------------------------------------------------------------------------
// Descriptive statistics
// ---------------------------------------------------------------------------

fn bench_mean(c: &mut Criterion) {
    let mut group = c.benchmark_group("descriptive_mean");
    for &n in &[100, 1_000, 10_000, 100_000] {
        let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
        group.bench_with_input(BenchmarkId::new("f64", n), &data, |b, data| {
            b.iter(|| descriptive::mean(black_box(data)));
        });
    }
    group.finish();
}

fn bench_variance(c: &mut Criterion) {
    let mut group = c.benchmark_group("descriptive_variance");
    for &n in &[100, 1_000, 10_000, 100_000] {
        let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
        group.bench_with_input(BenchmarkId::new("f64", n), &data, |b, data| {
            b.iter(|| descriptive::variance(black_box(data)));
        });
    }
    group.finish();
}

fn bench_median(c: &mut Criterion) {
    let mut group = c.benchmark_group("descriptive_median");
    for &n in &[100, 1_000, 10_000] {
        let data: Vec<f64> = (0..n).rev().map(|i| i as f64).collect();
        group.bench_with_input(BenchmarkId::new("f64", n), &data, |b, data| {
            b.iter(|| descriptive::median(black_box(data)).unwrap());
        });
    }
    group.finish();
}

fn bench_describe(c: &mut Criterion) {
    let mut group = c.benchmark_group("descriptive_describe");
    for &n in &[1_000, 10_000] {
        let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
        group.bench_with_input(BenchmarkId::new("f64", n), &data, |b, data| {
            b.iter(|| descriptive::describe(black_box(data)).unwrap());
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Distributions
// ---------------------------------------------------------------------------

fn bench_normal_pdf(c: &mut Criterion) {
    let dist = Normal::new(0.0_f64, 1.0).unwrap();
    c.bench_function("normal_pdf_single", |b| {
        b.iter(|| dist.pdf(black_box(1.5)));
    });
}

fn bench_normal_cdf(c: &mut Criterion) {
    let dist = Normal::new(0.0_f64, 1.0).unwrap();
    c.bench_function("normal_cdf_single", |b| {
        b.iter(|| dist.cdf(black_box(1.5)));
    });
}

fn bench_normal_sample(c: &mut Criterion) {
    let dist = Normal::new(0.0_f64, 1.0).unwrap();
    let mut rng = Rng::new(42);
    let mut group = c.benchmark_group("normal_sample_n");
    for &n in &[100, 1_000, 10_000] {
        group.bench_with_input(BenchmarkId::new("f64", n), &n, |b, &n| {
            b.iter(|| dist.sample_n(&mut rng, n));
        });
    }
    group.finish();
}

fn bench_uniform_sample(c: &mut Criterion) {
    let dist = Uniform::new(0.0_f64, 1.0).unwrap();
    let mut rng = Rng::new(42);
    let mut group = c.benchmark_group("uniform_sample_n");
    for &n in &[100, 1_000, 10_000] {
        group.bench_with_input(BenchmarkId::new("f64", n), &n, |b, &n| {
            b.iter(|| dist.sample_n(&mut rng, n));
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Hypothesis tests
// ---------------------------------------------------------------------------

fn bench_t_test(c: &mut Criterion) {
    let mut group = c.benchmark_group("t_test_one_sample");
    for &n in &[100, 1_000, 10_000] {
        let data: Vec<f64> = (0..n).map(|i| i as f64 + 0.5).collect();
        group.bench_with_input(BenchmarkId::new("f64", n), &data, |b, data| {
            b.iter(|| hypothesis::t_test_one_sample(black_box(data), 0.0).unwrap());
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Correlation
// ---------------------------------------------------------------------------

fn bench_pearson(c: &mut Criterion) {
    let mut group = c.benchmark_group("correlation_pearson");
    for &n in &[100, 1_000, 10_000] {
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let y: Vec<f64> = (0..n).map(|i| (i * 2) as f64 + 1.0).collect();
        group.bench_with_input(BenchmarkId::new("f64", n), &n, |b, _| {
            b.iter(|| correlation::pearson(black_box(&x), black_box(&y)).unwrap());
        });
    }
    group.finish();
}

fn bench_corr_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("correlation_matrix");
    for &n in &[50, 100, 200] {
        let data: Vec<f64> = (0..n * 5).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![n, 5]).unwrap();
        group.bench_with_input(BenchmarkId::new("5_features", n), &n, |b, _| {
            b.iter(|| correlation::corr_matrix(black_box(&t), CorrelationMethod::Pearson).unwrap());
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_mean,
    bench_variance,
    bench_median,
    bench_describe,
    bench_normal_pdf,
    bench_normal_cdf,
    bench_normal_sample,
    bench_uniform_sample,
    bench_t_test,
    bench_pearson,
    bench_corr_matrix,
);
criterion_main!(benches);
