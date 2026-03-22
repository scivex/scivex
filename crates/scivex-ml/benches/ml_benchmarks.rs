use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use scivex_core::Tensor;
use scivex_ml::cluster::KMeans;
use scivex_ml::ensemble::{RandomForestClassifier, RandomForestRegressor};
use scivex_ml::linear::LinearRegression;
use scivex_ml::preprocessing::StandardScaler;
use scivex_ml::svm::{Kernel, SVC};
use scivex_ml::traits::{Predictor, Transformer};
use scivex_ml::tree::{DecisionTreeClassifier, DecisionTreeRegressor};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_classification(n: usize, features: usize) -> (Tensor<f64>, Tensor<f64>) {
    let mut xdata = Vec::with_capacity(n * features);
    let mut ydata = Vec::with_capacity(n);
    for i in 0..n {
        for f in 0..features {
            xdata.push(((i * 7 + f * 3) % 100) as f64 / 100.0);
        }
        ydata.push(if i % 2 == 0 { 0.0 } else { 1.0 });
    }
    let x = Tensor::from_vec(xdata, vec![n, features]).unwrap();
    let y = Tensor::from_vec(ydata, vec![n]).unwrap();
    (x, y)
}

fn make_regression(n: usize, features: usize) -> (Tensor<f64>, Tensor<f64>) {
    let mut xdata = Vec::with_capacity(n * features);
    let mut ydata = Vec::with_capacity(n);
    for i in 0..n {
        let mut val = 0.0;
        for f in 0..features {
            let x = ((i * 7 + f * 3) % 100) as f64 / 100.0;
            xdata.push(x);
            val += x * (f + 1) as f64;
        }
        ydata.push(val);
    }
    let x = Tensor::from_vec(xdata, vec![n, features]).unwrap();
    let y = Tensor::from_vec(ydata, vec![n]).unwrap();
    (x, y)
}

// ---------------------------------------------------------------------------
// Linear Regression
// ---------------------------------------------------------------------------

fn bench_linear_regression_fit(c: &mut Criterion) {
    let mut group = c.benchmark_group("linear_regression_fit");
    for &n in &[100usize, 500, 1_000] {
        let (x, y) = make_regression(n, 5);
        group.bench_with_input(BenchmarkId::new("5feat", n), &n, |b, _| {
            b.iter(|| {
                let mut model = LinearRegression::<f64>::new();
                model.fit(black_box(&x), black_box(&y)).unwrap();
            });
        });
    }
    group.finish();
}

fn bench_linear_regression_predict(c: &mut Criterion) {
    let (x, y) = make_regression(500, 5);
    let mut model = LinearRegression::<f64>::new();
    model.fit(&x, &y).unwrap();
    let mut group = c.benchmark_group("linear_regression_predict");
    for &n in &[100usize, 500, 1_000] {
        let (x_test, _) = make_regression(n, 5);
        group.bench_with_input(BenchmarkId::new("5feat", n), &n, |b, _| {
            b.iter(|| model.predict(black_box(&x_test)).unwrap());
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Decision Tree
// ---------------------------------------------------------------------------

fn bench_decision_tree_fit(c: &mut Criterion) {
    let mut group = c.benchmark_group("decision_tree_classifier_fit");
    for &n in &[100usize, 500, 1_000] {
        let (x, y) = make_classification(n, 4);
        group.bench_with_input(BenchmarkId::new("4feat", n), &n, |b, _| {
            b.iter(|| {
                let mut tree = DecisionTreeClassifier::<f64>::new(Some(5), 2);
                tree.fit(black_box(&x), black_box(&y)).unwrap();
            });
        });
    }
    group.finish();
}

fn bench_decision_tree_regressor_fit(c: &mut Criterion) {
    let mut group = c.benchmark_group("decision_tree_regressor_fit");
    for &n in &[100usize, 500, 1_000] {
        let (x, y) = make_regression(n, 4);
        group.bench_with_input(BenchmarkId::new("4feat", n), &n, |b, _| {
            b.iter(|| {
                let mut tree = DecisionTreeRegressor::<f64>::new(Some(5), 2);
                tree.fit(black_box(&x), black_box(&y)).unwrap();
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// KMeans
// ---------------------------------------------------------------------------

fn bench_kmeans_fit(c: &mut Criterion) {
    let mut group = c.benchmark_group("kmeans_fit");
    for &n in &[200usize, 500, 1_000] {
        let (x, _) = make_classification(n, 4);
        group.bench_with_input(BenchmarkId::new("k3_4feat", n), &n, |b, _| {
            b.iter(|| {
                let mut km = KMeans::<f64>::new(3, 20, 1e-4, 1, 42).unwrap();
                km.fit(black_box(&x)).unwrap();
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// SVC
// ---------------------------------------------------------------------------

fn bench_svc_fit(c: &mut Criterion) {
    let mut group = c.benchmark_group("svc_fit");
    for &n in &[50usize, 100, 200] {
        let (x, y) = make_classification(n, 4);
        group.bench_with_input(BenchmarkId::new("linear_4feat", n), &n, |b, _| {
            b.iter(|| {
                let mut svc = SVC::<f64>::new(Kernel::Linear, 1.0).unwrap();
                svc.set_max_iter(100);
                svc.fit(black_box(&x), black_box(&y)).unwrap();
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Preprocessing
// ---------------------------------------------------------------------------

fn bench_standard_scaler(c: &mut Criterion) {
    let mut group = c.benchmark_group("standard_scaler");
    for &n in &[100usize, 1_000, 10_000] {
        let (x, _) = make_regression(n, 5);
        group.bench_with_input(BenchmarkId::new("fit_transform_5feat", n), &n, |b, _| {
            b.iter(|| {
                let mut scaler = StandardScaler::<f64>::new();
                scaler.fit(black_box(&x)).unwrap();
                scaler.transform(black_box(&x)).unwrap()
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Random Forest Classifier
// ---------------------------------------------------------------------------

fn bench_random_forest_fit(c: &mut Criterion) {
    let mut group = c.benchmark_group("random_forest_classifier_fit");
    for &n in &[500usize, 1_000] {
        let (x, y) = make_classification(n, 10);
        group.bench_with_input(BenchmarkId::new("10feat", n), &n, |b, _| {
            b.iter(|| {
                let mut rf = RandomForestClassifier::<f64>::new(10, Some(5), None, 42).unwrap();
                rf.fit(black_box(&x), black_box(&y)).unwrap();
            });
        });
    }
    group.finish();
}

fn bench_random_forest_predict(c: &mut Criterion) {
    let (x_train, y_train) = make_classification(1_000, 10);
    let mut rf = RandomForestClassifier::<f64>::new(10, Some(5), None, 42).unwrap();
    rf.fit(&x_train, &y_train).unwrap();

    let mut group = c.benchmark_group("random_forest_classifier_predict");
    for &n in &[500usize, 1_000] {
        let (x_test, _) = make_classification(n, 10);
        group.bench_with_input(BenchmarkId::new("10feat", n), &n, |b, _| {
            b.iter(|| rf.predict(black_box(&x_test)).unwrap());
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Random Forest Regressor
// ---------------------------------------------------------------------------

fn bench_random_forest_regressor_fit(c: &mut Criterion) {
    let mut group = c.benchmark_group("random_forest_regressor_fit");
    for &n in &[500usize, 1_000] {
        let (x, y) = make_regression(n, 10);
        group.bench_with_input(BenchmarkId::new("10feat", n), &n, |b, _| {
            b.iter(|| {
                let mut rf = RandomForestRegressor::<f64>::new(10, Some(5), None, 42).unwrap();
                rf.fit(black_box(&x), black_box(&y)).unwrap();
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// KMeans on 2-D data (larger)
// ---------------------------------------------------------------------------

fn bench_kmeans_2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("kmeans_2d");
    for &n in &[500usize, 1_000] {
        let xdata: Vec<f64> = (0..n * 2)
            .map(|i| ((i * 13 + 7) % 97) as f64 / 97.0)
            .collect();
        let x = Tensor::from_vec(xdata, vec![n, 2]).unwrap();
        group.bench_with_input(BenchmarkId::new("k5", n), &n, |b, _| {
            b.iter(|| {
                let mut km = KMeans::<f64>::new(5, 30, 1e-4, 1, 42).unwrap();
                km.fit(black_box(&x)).unwrap();
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Linear regression at various sizes
// ---------------------------------------------------------------------------

fn bench_linear_regression_fit_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("linear_regression_fit_sizes");
    for &(n, feat) in &[(200usize, 5), (500, 10), (1_000, 10)] {
        let (x, y) = make_regression(n, feat);
        let label = format!("{n}x{feat}");
        group.bench_function(&label, |b| {
            b.iter(|| {
                let mut model = LinearRegression::<f64>::new();
                model.fit(black_box(&x), black_box(&y)).unwrap();
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_linear_regression_fit,
    bench_linear_regression_predict,
    bench_linear_regression_fit_sizes,
    bench_decision_tree_fit,
    bench_decision_tree_regressor_fit,
    bench_kmeans_fit,
    bench_kmeans_2d,
    bench_random_forest_fit,
    bench_random_forest_predict,
    bench_random_forest_regressor_fit,
    bench_svc_fit,
    bench_standard_scaler,
);
criterion_main!(benches);
