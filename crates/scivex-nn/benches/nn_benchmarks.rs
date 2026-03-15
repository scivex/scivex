#![allow(clippy::cast_lossless)]

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use scivex_core::Tensor;
use scivex_core::random::Rng;
use scivex_nn::Variable;
use scivex_nn::layer::{Layer, Linear, ReLU, Sequential};
use scivex_nn::loss::mse_loss;

// ---------------------------------------------------------------------------
// Forward pass through Linear layer
// ---------------------------------------------------------------------------

fn bench_linear_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("nn_linear_forward");
    for &(batch, in_f, out_f) in &[(32usize, 64, 32), (64, 128, 64), (128, 256, 128)] {
        let mut rng = Rng::new(42);
        let layer = Linear::new(in_f, out_f, true, &mut rng);
        let data: Vec<f64> = (0..batch * in_f).map(|i| (i as f64) * 0.01).collect();
        let input = Variable::new(Tensor::from_vec(data, vec![batch, in_f]).unwrap(), false);
        let label = format!("{batch}x{in_f}->{out_f}");
        group.bench_function(&label, |b| {
            b.iter(|| layer.forward(black_box(&input)).unwrap());
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Sequential model forward
// ---------------------------------------------------------------------------

fn bench_sequential_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("nn_sequential_forward");
    for &batch in &[32usize, 64, 128] {
        let mut rng = Rng::new(42);
        let layers: Vec<Box<dyn Layer<f64>>> = vec![
            Box::new(Linear::new(32, 64, true, &mut rng)),
            Box::new(ReLU),
            Box::new(Linear::new(64, 32, true, &mut rng)),
            Box::new(ReLU),
            Box::new(Linear::new(32, 1, true, &mut rng)),
        ];
        let model = Sequential::new(layers);
        let data: Vec<f64> = (0..batch * 32).map(|i| (i as f64) * 0.01).collect();
        let input = Variable::new(Tensor::from_vec(data, vec![batch, 32]).unwrap(), false);
        group.bench_with_input(BenchmarkId::new("3layer_32in", batch), &batch, |b, _| {
            b.iter(|| model.forward(black_box(&input)).unwrap());
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// MSE loss computation
// ---------------------------------------------------------------------------

fn bench_mse_loss(c: &mut Criterion) {
    let mut group = c.benchmark_group("nn_mse_loss");
    for &n in &[100usize, 1_000, 10_000] {
        let pred_data: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
        let target_data: Vec<f64> = (0..n).map(|i| i as f64 * 0.1 + 0.5).collect();
        let pred = Variable::new(Tensor::from_vec(pred_data, vec![n]).unwrap(), true);
        let target = Variable::new(Tensor::from_vec(target_data, vec![n]).unwrap(), false);
        group.bench_with_input(BenchmarkId::new("f64", n), &n, |b, _| {
            b.iter(|| mse_loss(black_box(&pred), black_box(&target)).unwrap());
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Variable backward
// ---------------------------------------------------------------------------

fn bench_backward(c: &mut Criterion) {
    let mut rng = Rng::new(42);
    let layers: Vec<Box<dyn Layer<f64>>> = vec![
        Box::new(Linear::new(16, 32, true, &mut rng)),
        Box::new(ReLU),
        Box::new(Linear::new(32, 1, true, &mut rng)),
    ];
    let model = Sequential::new(layers);
    let data: Vec<f64> = (0..32 * 16).map(|i| (i as f64) * 0.01).collect();
    let input = Variable::new(Tensor::from_vec(data, vec![32, 16]).unwrap(), true);
    let target = Variable::new(Tensor::from_vec(vec![0.5; 32], vec![32]).unwrap(), false);

    c.bench_function("nn_backward_2layer", |b| {
        b.iter(|| {
            let out = model.forward(black_box(&input)).unwrap();
            let loss = mse_loss(&out, &target).unwrap();
            loss.backward();
        });
    });
}

criterion_group!(
    benches,
    bench_linear_forward,
    bench_sequential_forward,
    bench_mse_loss,
    bench_backward,
);
criterion_main!(benches);
