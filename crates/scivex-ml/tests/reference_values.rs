#![allow(clippy::uninlined_format_args)]
//! Reference-value tests comparing scivex-ml outputs against pre-computed
//! scikit-learn results.  Every expected value below was generated with
//! scikit-learn and NumPy so that we can catch regressions in numerical
//! accuracy.

use scivex_core::Tensor;
use scivex_ml::cluster::KMeans;
use scivex_ml::decomposition::PCA;
use scivex_ml::linear::LinearRegression;
use scivex_ml::metrics::{accuracy, mse};
use scivex_ml::preprocessing::{MinMaxScaler, StandardScaler};
use scivex_ml::traits::{Predictor, Transformer};

const TOL: f64 = 1e-4;

// ────────────────────────────────────────────────────────────────────
// 1. Linear Regression
//    sklearn: coef_ = [0.6], intercept_ = 2.2, predict([[6.0]]) = [5.8]
// ────────────────────────────────────────────────────────────────────

#[test]
fn linear_regression_coefficients() {
    let x = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0], vec![5, 1]).unwrap();
    let y = Tensor::from_vec(vec![2.0, 4.0, 5.0, 4.0, 5.0], vec![5]).unwrap();

    let mut lr = LinearRegression::<f64>::new();
    lr.fit(&x, &y).unwrap();

    let weights = lr.weights().expect("model should be fitted");
    let bias = lr.bias().expect("model should be fitted");

    assert!(
        (weights[0] - 0.6).abs() < TOL,
        "expected coef ~0.6, got {}",
        weights[0]
    );
    assert!(
        (bias - 2.2).abs() < TOL,
        "expected intercept ~2.2, got {}",
        bias
    );
}

#[test]
fn linear_regression_predict() {
    let x = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0], vec![5, 1]).unwrap();
    let y = Tensor::from_vec(vec![2.0, 4.0, 5.0, 4.0, 5.0], vec![5]).unwrap();

    let mut lr = LinearRegression::<f64>::new();
    lr.fit(&x, &y).unwrap();

    let x_new = Tensor::from_vec(vec![6.0_f64], vec![1, 1]).unwrap();
    let pred = lr.predict(&x_new).unwrap();

    assert!(
        (pred.as_slice()[0] - 5.8).abs() < TOL,
        "expected prediction ~5.8, got {}",
        pred.as_slice()[0]
    );
}

// ────────────────────────────────────────────────────────────────────
// 2. StandardScaler
//    sklearn: mean_ = [3.0, 4.0], scale_ = [1.6329932, 1.6329932]
//    transformed = [[-1.2247, -1.2247], [0, 0], [1.2247, 1.2247]]
// ────────────────────────────────────────────────────────────────────

#[test]
fn standard_scaler_transform_and_inverse() {
    let data = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]).unwrap();

    let mut scaler = StandardScaler::<f64>::new();
    scaler.fit(&data).unwrap();

    // Verify transform output matches sklearn
    let transformed = scaler.transform(&data).unwrap();
    let ts = transformed.as_slice();

    // sklearn: transformed = [[-1.2247, -1.2247], [0, 0], [1.2247, 1.2247]]
    let sqrt_1_5 = (1.5_f64).sqrt(); // ≈ 1.2247
    assert!((ts[0] - (-sqrt_1_5)).abs() < TOL, "t[0,0]");
    assert!((ts[1] - (-sqrt_1_5)).abs() < TOL, "t[0,1]");
    assert!((ts[2]).abs() < TOL, "t[1,0]");
    assert!((ts[3]).abs() < TOL, "t[1,1]");
    assert!((ts[4] - sqrt_1_5).abs() < TOL, "t[2,0]");
    assert!((ts[5] - sqrt_1_5).abs() < TOL, "t[2,1]");

    // Verify inverse_transform recovers original data
    let recovered = scaler.inverse_transform(&transformed).unwrap();
    for (a, b) in recovered.as_slice().iter().zip(data.as_slice()) {
        assert!((a - b).abs() < TOL, "inverse_transform roundtrip");
    }
}

#[test]
fn standard_scaler_transform() {
    let data = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]).unwrap();

    let mut scaler = StandardScaler::<f64>::new();
    let transformed = scaler.fit_transform(&data).unwrap();
    let t = transformed.as_slice();

    let expected = [
        -1.224_744_9,
        -1.224_744_9,
        0.0,
        0.0,
        1.224_744_9,
        1.224_744_9,
    ];
    for (i, (&got, &exp)) in t.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < TOL,
            "transformed[{}]: expected {}, got {}",
            i,
            exp,
            got
        );
    }
}

// ────────────────────────────────────────────────────────────────────
// 3. MinMaxScaler
//    sklearn: transformed = [[0, 0], [0.5, 0.5], [1.0, 1.0]]
// ────────────────────────────────────────────────────────────────────

#[test]
fn minmax_scaler_transform() {
    let data = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]).unwrap();

    let mut scaler = MinMaxScaler::<f64>::new();
    let transformed = scaler.fit_transform(&data).unwrap();
    let t = transformed.as_slice();

    let expected = [0.0, 0.0, 0.5, 0.5, 1.0, 1.0];
    for (i, (&got, &exp)) in t.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < TOL,
            "minmax[{}]: expected {}, got {}",
            i,
            exp,
            got
        );
    }
}

// ────────────────────────────────────────────────────────────────────
// 4. PCA
//    sklearn: explained_variance_ratio_ = [0.96318131] for the first PC
//    (first component captures ~96.3% of variance)
// ────────────────────────────────────────────────────────────────────

#[test]
fn pca_explained_variance_ratio() {
    #[rustfmt::skip]
    let data = Tensor::from_vec(
        vec![
            2.5, 2.4,
            0.5, 0.7,
            2.2, 2.9,
            1.9, 2.2,
            3.1, 3.0,
            2.3, 2.7,
            2.0, 1.6,
            1.0, 1.1,
            1.5, 1.6,
            1.1, 0.9_f64,
        ],
        vec![10, 2],
    )
    .unwrap();

    let mut pca = PCA::<f64>::new(1).unwrap();
    pca.fit(&data).unwrap();

    let ratio = pca
        .explained_variance_ratio()
        .expect("PCA should be fitted");
    assert!(
        (ratio[0] - 0.963_181_31).abs() < 1e-3,
        "expected explained variance ratio ~0.9632, got {}",
        ratio[0]
    );
}

// ────────────────────────────────────────────────────────────────────
// 5. KMeans
//    Two well-separated groups should be found.
//    Centroids near (1.33, 1.33) and (10.33, 10.33).
//    Inertia with k=2 < inertia with k=1.
// ────────────────────────────────────────────────────────────────────

#[test]
fn kmeans_two_clusters_sklearn_data() {
    #[rustfmt::skip]
    let x = Tensor::from_vec(
        vec![
            1.0, 1.0,
            1.0, 2.0,
            2.0, 1.0,
            10.0, 10.0,
            10.0, 11.0,
            11.0, 10.0_f64,
        ],
        vec![6, 2],
    )
    .unwrap();

    let mut km = KMeans::<f64>::new(2, 300, 1e-8, 10, 42).unwrap();
    km.fit(&x).unwrap();

    let labels = km.predict(&x).unwrap();
    let l = labels.as_slice();

    // First three points should share a label, last three should share a label.
    assert!(
        (l[0] - l[1]).abs() < 0.5 && (l[1] - l[2]).abs() < 0.5,
        "first three points should be in the same cluster: {:?}",
        &l[..3]
    );
    assert!(
        (l[3] - l[4]).abs() < 0.5 && (l[4] - l[5]).abs() < 0.5,
        "last three points should be in the same cluster: {:?}",
        &l[3..]
    );
    assert!(
        (l[0] - l[3]).abs() > 0.5,
        "the two groups should have different labels"
    );

    // Verify centroids are near the expected sklearn values.
    let centroids = km.centroids().expect("model should be fitted");
    let p = 2; // n_features
    let k = 2; // n_clusters

    // Determine which cluster index corresponds to the low group vs high group.
    let (low_idx, high_idx) = if centroids[0] < 5.0 { (0, 1) } else { (1, 0) };

    let low_cx = centroids[low_idx * p];
    let low_cy = centroids[low_idx * p + 1];
    let high_cx = centroids[high_idx * p];
    let high_cy = centroids[high_idx * p + 1];

    let expected_low = 4.0 / 3.0; // ~1.3333
    let expected_high = 31.0 / 3.0; // ~10.3333

    assert!(
        (low_cx - expected_low).abs() < 0.1 && (low_cy - expected_low).abs() < 0.1,
        "low centroid expected ~({}, {}), got ({}, {})",
        expected_low,
        expected_low,
        low_cx,
        low_cy
    );
    assert!(
        (high_cx - expected_high).abs() < 0.1 && (high_cy - expected_high).abs() < 0.1,
        "high centroid expected ~({}, {}), got ({}, {})",
        expected_high,
        expected_high,
        high_cx,
        high_cy
    );

    // Suppress unused variable warning
    let _ = k;
}

#[test]
fn kmeans_inertia_decreases_with_more_clusters() {
    #[rustfmt::skip]
    let x = Tensor::from_vec(
        vec![
            1.0, 1.0,
            1.0, 2.0,
            2.0, 1.0,
            10.0, 10.0,
            10.0, 11.0,
            11.0, 10.0_f64,
        ],
        vec![6, 2],
    )
    .unwrap();

    let mut km1 = KMeans::<f64>::new(1, 300, 1e-8, 5, 42).unwrap();
    km1.fit(&x).unwrap();
    let inertia1 = km1.inertia().unwrap();

    let mut km2 = KMeans::<f64>::new(2, 300, 1e-8, 10, 42).unwrap();
    km2.fit(&x).unwrap();
    let inertia2 = km2.inertia().unwrap();

    assert!(
        inertia2 < inertia1,
        "inertia with k=2 ({}) should be less than k=1 ({})",
        inertia2,
        inertia1
    );
}

// ────────────────────────────────────────────────────────────────────
// 6. Accuracy metric
//    sklearn: accuracy_score([0,1,1,0,1], [0,1,0,0,1]) = 0.8
// ────────────────────────────────────────────────────────────────────

#[test]
fn accuracy_sklearn_reference() {
    let y_true = [0.0_f64, 1.0, 1.0, 0.0, 1.0];
    let y_pred = [0.0_f64, 1.0, 0.0, 0.0, 1.0];

    let acc = accuracy(&y_true, &y_pred).unwrap();
    assert!(
        (acc - 0.8).abs() < TOL,
        "expected accuracy 0.8, got {}",
        acc
    );
}

// ────────────────────────────────────────────────────────────────────
// 7. MSE metric
//    sklearn: mean_squared_error([1,2,3], [1.1,2.2,2.8]) ≈ 0.03
// ────────────────────────────────────────────────────────────────────

#[test]
fn mse_sklearn_reference() {
    let y_true = [1.0_f64, 2.0, 3.0];
    let y_pred = [1.1_f64, 2.2, 2.8];

    let m = mse(&y_true, &y_pred).unwrap();
    // (0.01 + 0.04 + 0.04) / 3 = 0.03
    assert!((m - 0.03).abs() < TOL, "expected MSE ~0.03, got {}", m);
}
