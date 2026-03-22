use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use scivex_core::Tensor;
use scivex_core::fft;
use scivex_core::linalg::{
    CholeskyDecomposition, LuDecomposition, QrDecomposition, SvdDecomposition, dot, gemm,
};

// ---------------------------------------------------------------------------
// Tensor creation
// ---------------------------------------------------------------------------

fn bench_tensor_zeros(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_creation");
    for &n in &[100usize, 1_000, 10_000, 100_000] {
        group.bench_with_input(BenchmarkId::new("zeros", n), &n, |b, &n| {
            b.iter(|| Tensor::<f64>::zeros(vec![n]));
        });
    }
    group.finish();
}

fn bench_tensor_from_vec(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_from_vec");
    for &n in &[100usize, 1_000, 10_000] {
        let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
        group.bench_with_input(BenchmarkId::new("1d", n), &n, |b, _| {
            b.iter(|| Tensor::from_vec(black_box(data.clone()), vec![data.len()]).unwrap());
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Element-wise operations
// ---------------------------------------------------------------------------

fn bench_elementwise_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("elementwise_add");
    for &n in &[100usize, 1_000, 10_000, 100_000] {
        let a = Tensor::<f64>::ones(vec![n]);
        let b = Tensor::<f64>::ones(vec![n]);
        group.bench_with_input(BenchmarkId::new("f64", n), &n, |bench, _| {
            bench.iter(|| black_box(&a) + black_box(&b));
        });
    }
    group.finish();
}

fn bench_elementwise_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("elementwise_mul");
    for &n in &[100usize, 1_000, 10_000, 100_000] {
        let a = Tensor::<f64>::ones(vec![n]);
        let b = Tensor::<f64>::full(vec![n], 2.0);
        group.bench_with_input(BenchmarkId::new("f64", n), &n, |bench, _| {
            bench.iter(|| black_box(&a) * black_box(&b));
        });
    }
    group.finish();
}

fn bench_map(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_map");
    for &n in &[1_000usize, 10_000, 100_000] {
        let t = Tensor::<f64>::arange(n);
        group.bench_with_input(BenchmarkId::new("square", n), &n, |bench, _| {
            bench.iter(|| t.map(|x| x * x));
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// BLAS: dot product
// ---------------------------------------------------------------------------

fn bench_dot(c: &mut Criterion) {
    let mut group = c.benchmark_group("blas_dot");
    for &n in &[100usize, 1_000, 10_000, 100_000] {
        let x = Tensor::<f64>::ones(vec![n]);
        let y = Tensor::<f64>::ones(vec![n]);
        group.bench_with_input(BenchmarkId::new("f64", n), &n, |bench, _| {
            bench.iter(|| dot(black_box(&x), black_box(&y)).unwrap());
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Matrix multiply (matmul wrapping gemm)
// ---------------------------------------------------------------------------

fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul");
    for &n in &[16usize, 32, 64, 128, 256] {
        let a = Tensor::<f64>::ones(vec![n, n]);
        let b = Tensor::<f64>::ones(vec![n, n]);
        group.bench_with_input(BenchmarkId::new("f64_square", n), &n, |bench, _| {
            bench.iter(|| a.matmul(black_box(&b)).unwrap());
        });
    }
    group.finish();
}

fn bench_gemm(c: &mut Criterion) {
    let mut group = c.benchmark_group("blas_gemm");
    for &n in &[16usize, 32, 64, 128] {
        let a = Tensor::<f64>::ones(vec![n, n]);
        let b = Tensor::<f64>::ones(vec![n, n]);
        let mut out = Tensor::<f64>::zeros(vec![n, n]);
        group.bench_with_input(BenchmarkId::new("f64_square", n), &n, |bench, _| {
            bench.iter(|| {
                gemm(1.0, black_box(&a), black_box(&b), 0.0, &mut out).unwrap();
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Matrix decompositions
// ---------------------------------------------------------------------------

fn bench_lu(c: &mut Criterion) {
    let mut group = c.benchmark_group("decomp_lu");
    for &n in &[16usize, 32, 64, 128] {
        let data: Vec<f64> = (0..n * n).map(|i| ((i * 7 + 3) % 100) as f64).collect();
        let a = Tensor::from_vec(data, vec![n, n]).unwrap();
        group.bench_with_input(BenchmarkId::new("f64", n), &n, |bench, _| {
            bench.iter(|| LuDecomposition::decompose(black_box(&a)).unwrap());
        });
    }
    group.finish();
}

fn bench_qr(c: &mut Criterion) {
    let mut group = c.benchmark_group("decomp_qr");
    for &n in &[16usize, 32, 64, 128] {
        let data: Vec<f64> = (0..n * n).map(|i| ((i * 7 + 3) % 100) as f64).collect();
        let a = Tensor::from_vec(data, vec![n, n]).unwrap();
        group.bench_with_input(BenchmarkId::new("f64", n), &n, |bench, _| {
            bench.iter(|| QrDecomposition::decompose(black_box(&a)).unwrap());
        });
    }
    group.finish();
}

fn bench_svd(c: &mut Criterion) {
    let mut group = c.benchmark_group("decomp_svd");
    for &n in &[16usize, 32, 64] {
        let data: Vec<f64> = (0..n * n).map(|i| ((i * 7 + 3) % 100) as f64).collect();
        let a = Tensor::from_vec(data, vec![n, n]).unwrap();
        group.bench_with_input(BenchmarkId::new("f64", n), &n, |bench, _| {
            bench.iter(|| SvdDecomposition::decompose(black_box(&a)).unwrap());
        });
    }
    group.finish();
}

fn bench_cholesky(c: &mut Criterion) {
    let mut group = c.benchmark_group("decomp_cholesky");
    for &n in &[16usize, 32, 64, 128] {
        // Build a positive-definite matrix: A^T * A + n*I
        let raw: Vec<f64> = (0..n * n).map(|i| ((i * 7 + 3) % 100) as f64).collect();
        let a = Tensor::from_vec(raw, vec![n, n]).unwrap();
        let at = a.transpose().unwrap();
        let ata = at.matmul(&a).unwrap();
        let mut data = ata.into_vec();
        for i in 0..n {
            data[i * n + i] += n as f64;
        }
        let spd = Tensor::from_vec(data, vec![n, n]).unwrap();
        group.bench_with_input(BenchmarkId::new("f64", n), &n, |bench, _| {
            bench.iter(|| CholeskyDecomposition::decompose(black_box(&spd)).unwrap());
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// FFT
// ---------------------------------------------------------------------------

fn bench_fft(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft");
    for &n in &[64usize, 256, 1024, 4096] {
        // FFT expects interleaved [re, im, re, im, ...] so n rows x 2 cols
        let data: Vec<f64> = (0..2 * n).map(|i| (i as f64).sin()).collect();
        let t = Tensor::from_vec(data, vec![n, 2]).unwrap();
        group.bench_with_input(BenchmarkId::new("complex_f64", n), &n, |bench, _| {
            bench.iter(|| fft::fft(black_box(&t)).unwrap());
        });
    }
    group.finish();
}

fn bench_rfft(c: &mut Criterion) {
    let mut group = c.benchmark_group("rfft");
    for &n in &[64usize, 256, 1024, 4096] {
        let data: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
        let t = Tensor::from_vec(data, vec![n]).unwrap();
        group.bench_with_input(BenchmarkId::new("real_f64", n), &n, |bench, _| {
            bench.iter(|| fft::rfft(black_box(&t)).unwrap());
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Transpose
// ---------------------------------------------------------------------------

fn bench_transpose(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_transpose");
    for &n in &[32usize, 64, 128, 256] {
        let a = Tensor::<f64>::ones(vec![n, n]);
        group.bench_with_input(BenchmarkId::new("f64", n), &n, |bench, _| {
            bench.iter(|| a.transpose().unwrap());
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// SVD at larger sizes
// ---------------------------------------------------------------------------

fn bench_svd_large(c: &mut Criterion) {
    let mut group = c.benchmark_group("decomp_svd_large");
    for &n in &[32usize, 64, 128] {
        let data: Vec<f64> = (0..n * n).map(|i| ((i * 13 + 7) % 97) as f64).collect();
        let a = Tensor::from_vec(data, vec![n, n]).unwrap();
        group.bench_with_input(BenchmarkId::new("f64", n), &n, |bench, _| {
            bench.iter(|| SvdDecomposition::decompose(black_box(&a)).unwrap());
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// FFT 1-D at larger sizes
// ---------------------------------------------------------------------------

fn bench_fft_1d_large(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_1d_large");
    for &n in &[256usize, 1024, 4096, 16384] {
        let data: Vec<f64> = (0..2 * n).map(|i| (i as f64).sin()).collect();
        let t = Tensor::from_vec(data, vec![n, 2]).unwrap();
        group.bench_with_input(BenchmarkId::new("complex_f64", n), &n, |bench, _| {
            bench.iter(|| fft::fft(black_box(&t)).unwrap());
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Einsum: matrix multiplication via "ij,jk->ik"
// ---------------------------------------------------------------------------

fn bench_einsum(c: &mut Criterion) {
    use scivex_core::tensor::einsum::einsum;

    let mut group = c.benchmark_group("einsum_matmul");
    for &n in &[32usize, 64] {
        let a = Tensor::<f64>::ones(vec![n, n]);
        let b = Tensor::<f64>::ones(vec![n, n]);
        group.bench_with_input(BenchmarkId::new("ij_jk_ik", n), &n, |bench, _| {
            bench.iter(|| einsum("ij,jk->ik", &[black_box(&a), black_box(&b)]).unwrap());
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_tensor_zeros,
    bench_tensor_from_vec,
    bench_elementwise_add,
    bench_elementwise_mul,
    bench_map,
    bench_dot,
    bench_matmul,
    bench_gemm,
    bench_lu,
    bench_qr,
    bench_svd,
    bench_svd_large,
    bench_cholesky,
    bench_fft,
    bench_rfft,
    bench_fft_1d_large,
    bench_transpose,
    bench_einsum,
);
criterion_main!(benches);
