//! Reference tests comparing scivex-ml against known sklearn/analytical values.

use scivex_ml::metrics::classification;
use scivex_ml::metrics::regression;

const TOL: f64 = 1e-10;

// ─── Classification metrics ──────────────────────────────────────────

#[test]
fn accuracy_perfect() {
    let y_true = [1.0_f64, 0.0, 1.0, 1.0, 0.0];
    let y_pred = [1.0_f64, 0.0, 1.0, 1.0, 0.0];
    let acc = classification::accuracy(&y_true, &y_pred).unwrap();
    assert!((acc - 1.0).abs() < TOL, "accuracy = {acc}");
}

#[test]
fn accuracy_half() {
    let y_true = [1.0_f64, 0.0, 1.0, 0.0];
    let y_pred = [1.0_f64, 1.0, 0.0, 0.0];
    let acc = classification::accuracy(&y_true, &y_pred).unwrap();
    assert!((acc - 0.5).abs() < TOL, "accuracy = {acc}");
}

#[test]
fn precision_known() {
    // True: [1, 1, 0, 0], Pred: [1, 0, 1, 0]
    // TP=1, FP=1, FN=1, TN=1
    // precision = TP/(TP+FP) = 1/2
    let y_true = [1.0_f64, 1.0, 0.0, 0.0];
    let y_pred = [1.0_f64, 0.0, 1.0, 0.0];
    let p = classification::precision(&y_true, &y_pred).unwrap();
    assert!((p - 0.5).abs() < TOL, "precision = {p}");
}

#[test]
fn recall_known() {
    // Same as above: recall = TP/(TP+FN) = 1/2
    let y_true = [1.0_f64, 1.0, 0.0, 0.0];
    let y_pred = [1.0_f64, 0.0, 1.0, 0.0];
    let r = classification::recall(&y_true, &y_pred).unwrap();
    assert!((r - 0.5).abs() < TOL, "recall = {r}");
}

#[test]
fn f1_score_known() {
    // precision = 0.5, recall = 0.5 → F1 = 2*0.5*0.5/(0.5+0.5) = 0.5
    let y_true = [1.0_f64, 1.0, 0.0, 0.0];
    let y_pred = [1.0_f64, 0.0, 1.0, 0.0];
    let f1 = classification::f1_score(&y_true, &y_pred).unwrap();
    assert!((f1 - 0.5).abs() < TOL, "f1 = {f1}");
}

// ─── Regression metrics ──────────────────────────────────────────────

#[test]
fn mse_known() {
    // MSE([1,2,3], [1.5, 2.5, 3.5]) = mean(0.25 + 0.25 + 0.25) = 0.25
    let y_true = [1.0_f64, 2.0, 3.0];
    let y_pred = [1.5_f64, 2.5, 3.5];
    let m = regression::mse(&y_true, &y_pred).unwrap();
    assert!((m - 0.25).abs() < TOL, "mse = {m}");
}

#[test]
fn mae_known() {
    // MAE([1,2,3], [1.5, 2.5, 3.5]) = mean(0.5 + 0.5 + 0.5) = 0.5
    let y_true = [1.0_f64, 2.0, 3.0];
    let y_pred = [1.5_f64, 2.5, 3.5];
    let m = regression::mae(&y_true, &y_pred).unwrap();
    assert!((m - 0.5).abs() < TOL, "mae = {m}");
}

#[test]
fn r2_perfect_fit() {
    let y_true = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let y_pred = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let r2 = regression::r2_score(&y_true, &y_pred).unwrap();
    assert!((r2 - 1.0).abs() < TOL, "r2 = {r2}");
}

#[test]
fn r2_mean_predictor() {
    // Predicting the mean always gives R² = 0
    let y_true = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let y_pred = [3.0_f64, 3.0, 3.0, 3.0, 3.0]; // mean = 3
    let r2 = regression::r2_score(&y_true, &y_pred).unwrap();
    assert!(r2.abs() < TOL, "r2 = {r2}");
}

#[test]
fn rmse_known() {
    // RMSE = sqrt(MSE) = sqrt(0.25) = 0.5
    let y_true = [1.0_f64, 2.0, 3.0];
    let y_pred = [1.5_f64, 2.5, 3.5];
    let r = regression::rmse(&y_true, &y_pred).unwrap();
    assert!((r - 0.5).abs() < TOL, "rmse = {r}");
}
