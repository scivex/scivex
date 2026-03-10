//! Reference tests comparing scivex-stats against known SciPy/analytical values.

use scivex_stats::correlation;
use scivex_stats::descriptive;
use scivex_stats::distributions::{Distribution, Exponential, Normal, Uniform};
use scivex_stats::regression;

use scivex_core::Tensor;

const TOL: f64 = 1e-6;

// ─── Normal distribution ─────────────────────────────────────────────

#[test]
fn normal_pdf_at_zero() {
    // scipy.stats.norm.pdf(0) = 1/sqrt(2*pi) ≈ 0.3989422804014327
    let n = Normal::new(0.0_f64, 1.0).unwrap();
    let pdf0 = n.pdf(0.0);
    assert!(
        (pdf0 - 0.398_942_280_401_432_7).abs() < TOL,
        "N(0,1).pdf(0) = {pdf0}"
    );
}

#[test]
fn normal_cdf_at_zero() {
    // scipy.stats.norm.cdf(0) = 0.5
    let n = Normal::new(0.0_f64, 1.0).unwrap();
    assert!((n.cdf(0.0) - 0.5).abs() < TOL);
}

#[test]
fn normal_cdf_at_1_96() {
    // scipy.stats.norm.cdf(1.96) ≈ 0.9750021048517795
    let n = Normal::new(0.0_f64, 1.0).unwrap();
    let c = n.cdf(1.96);
    assert!(
        (c - 0.975_002_104_851_78).abs() < 1e-4,
        "N(0,1).cdf(1.96) = {c}"
    );
}

#[test]
fn normal_ppf_0_975() {
    // scipy.stats.norm.ppf(0.975) ≈ 1.959963984540054
    let n = Normal::new(0.0_f64, 1.0).unwrap();
    let q = n.ppf(0.975).unwrap();
    assert!((q - 1.96).abs() < 0.01, "N(0,1).ppf(0.975) = {q}");
}

#[test]
fn normal_mean_variance() {
    let n = Normal::new(5.0_f64, 2.0).unwrap();
    assert!((n.mean() - 5.0).abs() < TOL);
    assert!((n.variance() - 4.0).abs() < TOL);
}

#[test]
fn normal_symmetry() {
    // PDF should be symmetric: pdf(x) == pdf(-x)
    let n = Normal::new(0.0_f64, 1.0).unwrap();
    for &x in &[0.5, 1.0, 2.0, 3.0] {
        assert!(
            (n.pdf(x) - n.pdf(-x)).abs() < 1e-14,
            "symmetry failed at x={x}"
        );
    }
}

// ─── Uniform distribution ────────────────────────────────────────────

#[test]
fn uniform_pdf_cdf() {
    // U(0,1): pdf = 1 everywhere in [0,1], cdf(0.5) = 0.5
    let u = Uniform::new(0.0_f64, 1.0).unwrap();
    assert!((u.pdf(0.5) - 1.0).abs() < TOL);
    assert!((u.cdf(0.5) - 0.5).abs() < TOL);
    assert!((u.cdf(0.0)).abs() < TOL);
    assert!((u.cdf(1.0) - 1.0).abs() < TOL);
}

// ─── Exponential distribution ────────────────────────────────────────

#[test]
fn exponential_pdf_cdf() {
    // Exp(λ=1): pdf(1) = e^(-1), cdf(1) = 1 - e^(-1)
    let e = Exponential::new(1.0_f64).unwrap();
    let pdf1 = e.pdf(1.0);
    let cdf1 = e.cdf(1.0);
    let e_inv = (-1.0_f64).exp(); // ≈ 0.3679
    assert!((pdf1 - e_inv).abs() < TOL, "Exp(1).pdf(1) = {pdf1}");
    assert!((cdf1 - (1.0 - e_inv)).abs() < TOL, "Exp(1).cdf(1) = {cdf1}");
}

#[test]
fn exponential_mean_variance() {
    // Exp(λ=2): mean = 1/λ = 0.5, variance = 1/λ² = 0.25
    let e = Exponential::new(2.0_f64).unwrap();
    assert!((e.mean() - 0.5).abs() < TOL);
    assert!((e.variance() - 0.25).abs() < TOL);
}

// ─── Descriptive statistics ──────────────────────────────────────────

#[test]
fn descriptive_mean() {
    // mean([1, 2, 3, 4, 5]) = 3.0
    let data = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let m = descriptive::mean(&data).unwrap();
    assert!((m - 3.0).abs() < TOL);
}

#[test]
fn descriptive_variance() {
    // np.var([1, 2, 3, 4, 5], ddof=1) = 2.5
    let data = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let v = descriptive::variance(&data).unwrap();
    assert!((v - 2.5).abs() < TOL, "variance = {v}");
}

#[test]
fn descriptive_std_dev() {
    // np.std([1, 2, 3, 4, 5], ddof=1) = sqrt(2.5) ≈ 1.5811
    let data = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let s = descriptive::std_dev(&data).unwrap();
    assert!((s - 2.5_f64.sqrt()).abs() < TOL, "std = {s}");
}

#[test]
fn descriptive_median() {
    // median([1, 3, 5, 7, 9]) = 5
    let data = [1.0_f64, 3.0, 5.0, 7.0, 9.0];
    let m = descriptive::median(&data).unwrap();
    assert!((m - 5.0).abs() < TOL);

    // median([1, 2, 3, 4]) = 2.5
    let data2 = [1.0_f64, 2.0, 3.0, 4.0];
    let m2 = descriptive::median(&data2).unwrap();
    assert!((m2 - 2.5).abs() < TOL);
}

// ─── Pearson correlation ─────────────────────────────────────────────

#[test]
fn pearson_perfect_positive() {
    // Perfect positive correlation
    let x = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let y = [2.0, 4.0, 6.0, 8.0, 10.0];
    let r = correlation::pearson(&x, &y).unwrap();
    assert!((r - 1.0).abs() < TOL, "pearson = {r}");
}

#[test]
fn pearson_perfect_negative() {
    let x = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let y = [10.0, 8.0, 6.0, 4.0, 2.0];
    let r = correlation::pearson(&x, &y).unwrap();
    assert!((r - (-1.0)).abs() < TOL, "pearson = {r}");
}

// ─── OLS regression ──────────────────────────────────────────────────

#[test]
fn ols_simple_linear() {
    // y = 2x + 1 exactly (no noise)
    // x = [1, 2, 3, 4, 5], y = [3, 5, 7, 9, 11]
    let x_data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let x = Tensor::from_vec(x_data, vec![5, 1]).unwrap();
    let y = [3.0_f64, 5.0, 7.0, 9.0, 11.0];
    let result = regression::ols(&x, &y).unwrap();

    // coefficients[0] = intercept ≈ 1.0, coefficients[1] = slope ≈ 2.0
    assert!(
        (result.coefficients[0] - 1.0).abs() < TOL,
        "intercept = {}",
        result.coefficients[0]
    );
    assert!(
        (result.coefficients[1] - 2.0).abs() < TOL,
        "slope = {}",
        result.coefficients[1]
    );
    // R² should be 1.0 for perfect fit
    assert!(
        (result.r_squared - 1.0).abs() < TOL,
        "R² = {}",
        result.r_squared
    );
}
