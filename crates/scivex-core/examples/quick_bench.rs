#![allow(clippy::cast_lossless, clippy::uninlined_format_args)]
//! Quick benchmark script — run with: cargo run --release --example quick_bench -p scivex-core
//! (or as a standalone binary)

use std::time::Instant;

fn main() {
    use scivex_core::Tensor;
    use scivex_core::fft;
    use scivex_core::linalg::{dot, gemm};

    println!("=== Scivex Quick Benchmarks ===\n");

    // Dot product
    for &n in &[1_000usize, 10_000, 100_000] {
        let x = Tensor::<f64>::ones(vec![n]);
        let y = Tensor::<f64>::ones(vec![n]);
        let start = Instant::now();
        let iters = 1000;
        for _ in 0..iters {
            let _ = std::hint::black_box(dot(&x, &y).unwrap());
        }
        let elapsed = start.elapsed();
        println!(
            "dot f64 n={:<8} {:>10.1} ns/iter",
            n,
            elapsed.as_nanos() as f64 / iters as f64
        );
    }
    println!();

    // GEMM
    for &n in &[64usize, 128, 256] {
        let a = Tensor::<f64>::ones(vec![n, n]);
        let b = Tensor::<f64>::ones(vec![n, n]);
        let mut c = Tensor::<f64>::zeros(vec![n, n]);
        let iters = if n <= 128 { 100 } else { 10 };
        let start = Instant::now();
        for _ in 0..iters {
            gemm(1.0, &a, &b, 0.0, &mut c).unwrap();
        }
        let elapsed = start.elapsed();
        let ns_per = elapsed.as_nanos() as f64 / iters as f64;
        let gflops = 2.0 * (n as f64).powi(3) / ns_per;
        println!(
            "gemm f64 {0}x{0}  {1:>12.0} ns/iter  ({2:.2} GFLOP/s)",
            n, ns_per, gflops
        );
    }
    println!();

    // FFT
    for &n in &[256usize, 1024, 4096] {
        let data: Vec<f64> = (0..2 * n).map(|i| (i as f64).sin()).collect();
        let t = Tensor::from_vec(data, vec![n, 2]).unwrap();
        let iters = if n <= 1024 { 1000 } else { 100 };
        let start = Instant::now();
        for _ in 0..iters {
            let _ = std::hint::black_box(fft::fft(&t).unwrap());
        }
        let elapsed = start.elapsed();
        println!(
            "fft f64 n={:<6} {:>10.0} ns/iter",
            n,
            elapsed.as_nanos() as f64 / iters as f64
        );
    }
    println!();

    // Transpose
    for &n in &[64usize, 256] {
        let a = Tensor::<f64>::ones(vec![n, n]);
        let iters = 1000;
        let start = Instant::now();
        for _ in 0..iters {
            let _ = std::hint::black_box(a.transpose().unwrap());
        }
        let elapsed = start.elapsed();
        println!(
            "transpose f64 {0}x{0}  {1:>10.0} ns/iter",
            n,
            elapsed.as_nanos() as f64 / iters as f64
        );
    }
    println!();

    // Element-wise add
    for &n in &[10_000usize, 100_000] {
        let a = Tensor::<f64>::ones(vec![n]);
        let b = Tensor::<f64>::ones(vec![n]);
        let iters = 1000;
        let start = Instant::now();
        for _ in 0..iters {
            let _ = std::hint::black_box(&a + &b);
        }
        let elapsed = start.elapsed();
        println!(
            "add f64 n={:<8} {:>10.0} ns/iter",
            n,
            elapsed.as_nanos() as f64 / iters as f64
        );
    }
}
