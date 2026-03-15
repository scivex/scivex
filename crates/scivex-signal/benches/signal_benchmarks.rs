use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use scivex_core::Tensor;
use scivex_signal::convolution::{ConvolveMode, convolve};
use scivex_signal::spectral::{spectrogram, stft};
use scivex_signal::window;

// ---------------------------------------------------------------------------
// Convolution
// ---------------------------------------------------------------------------

fn bench_convolve(c: &mut Criterion) {
    let mut group = c.benchmark_group("convolve");
    for &n in &[256usize, 1024, 4096] {
        let signal: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        let kernel: Vec<f64> = (0..32).map(|i| (f64::from(i) * 0.2).cos()).collect();
        let a = Tensor::from_vec(signal, vec![n]).unwrap();
        let b = Tensor::from_vec(kernel, vec![32]).unwrap();
        group.bench_with_input(BenchmarkId::new("full_k32", n), &n, |bench, _| {
            bench.iter(|| convolve(black_box(&a), black_box(&b), ConvolveMode::Full).unwrap());
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Window functions
// ---------------------------------------------------------------------------

fn bench_window_hann(c: &mut Criterion) {
    let mut group = c.benchmark_group("window_hann");
    for &n in &[256usize, 1024, 4096] {
        group.bench_with_input(BenchmarkId::new("f64", n), &n, |b, &n| {
            b.iter(|| window::hann::<f64>(n).unwrap());
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// STFT
// ---------------------------------------------------------------------------

fn bench_stft(c: &mut Criterion) {
    let mut group = c.benchmark_group("stft");
    for &n in &[1024usize, 4096] {
        let signal: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        let t = Tensor::from_vec(signal, vec![n]).unwrap();
        group.bench_with_input(BenchmarkId::new("win256_hop128", n), &n, |b, _| {
            b.iter(|| stft(black_box(&t), 256, 128, None).unwrap());
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Spectrogram
// ---------------------------------------------------------------------------

fn bench_spectrogram(c: &mut Criterion) {
    let mut group = c.benchmark_group("spectrogram");
    for &n in &[1024usize, 4096] {
        let signal: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        let t = Tensor::from_vec(signal, vec![n]).unwrap();
        group.bench_with_input(BenchmarkId::new("win256_hop128", n), &n, |b, _| {
            b.iter(|| spectrogram(black_box(&t), 256, 128).unwrap());
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_convolve,
    bench_window_hann,
    bench_stft,
    bench_spectrogram,
);
criterion_main!(benches);
