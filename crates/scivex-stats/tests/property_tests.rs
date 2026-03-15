//! Property-based tests for scivex-stats using proptest.
#![allow(clippy::manual_range_contains, clippy::cast_lossless)]

use proptest::prelude::*;
use scivex_core::random::Rng;
use scivex_stats::descriptive;
use scivex_stats::distributions::{Distribution, Exponential, Normal, Uniform};

// ---------------------------------------------------------------------------
// Descriptive statistics properties
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn mean_of_constant_is_constant(n in 5usize..200, v in -1000.0_f64..1000.0) {
        let data: Vec<f64> = vec![v; n];
        let m = descriptive::mean(&data).unwrap();
        prop_assert!((m - v).abs() < 1e-10, "mean of constant {} got {}", v, m);
    }

    #[test]
    fn mean_is_between_min_and_max(data in proptest::collection::vec(-1000.0_f64..1000.0, 2..200)) {
        let m = descriptive::mean(&data).unwrap();
        let min = data.iter().copied().fold(f64::INFINITY, f64::min);
        let max = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        prop_assert!(m >= min - 1e-10, "mean {} < min {}", m, min);
        prop_assert!(m <= max + 1e-10, "mean {} > max {}", m, max);
    }

    #[test]
    fn variance_is_nonnegative(data in proptest::collection::vec(-1000.0_f64..1000.0, 2..200)) {
        let v = descriptive::variance(&data).unwrap();
        prop_assert!(v >= -1e-10, "variance should be non-negative, got {}", v);
    }

    #[test]
    fn variance_of_constant_is_zero(n in 5usize..200, v in -1000.0_f64..1000.0) {
        let data: Vec<f64> = vec![v; n];
        let var = descriptive::variance(&data).unwrap();
        prop_assert!(var.abs() < 1e-10, "variance of constant should be 0, got {}", var);
    }

    #[test]
    fn std_dev_is_sqrt_of_variance(data in proptest::collection::vec(-100.0_f64..100.0, 5..100)) {
        let var = descriptive::variance(&data).unwrap();
        let sd = descriptive::std_dev(&data).unwrap();
        prop_assert!((sd * sd - var).abs() < 1e-6,
            "std_dev^2 ({}) should equal variance ({})", sd * sd, var);
    }

    #[test]
    fn median_is_between_min_and_max(data in proptest::collection::vec(-1000.0_f64..1000.0, 1..200)) {
        let med = descriptive::median(&data).unwrap();
        let min = data.iter().copied().fold(f64::INFINITY, f64::min);
        let max = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        prop_assert!(med >= min - 1e-10, "median {} < min {}", med, min);
        prop_assert!(med <= max + 1e-10, "median {} > max {}", med, max);
    }
}

// ---------------------------------------------------------------------------
// Distribution properties
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn normal_pdf_nonnegative(mu in -10.0_f64..10.0, sigma in 0.1_f64..10.0, x in -50.0_f64..50.0) {
        let dist = Normal::new(mu, sigma).unwrap();
        let p = dist.pdf(x);
        prop_assert!(p >= 0.0, "Normal PDF should be non-negative, got {}", p);
    }

    #[test]
    fn normal_cdf_monotonic(mu in -10.0_f64..10.0, sigma in 0.1_f64..10.0) {
        let dist = Normal::new(mu, sigma).unwrap();
        let points: Vec<f64> = (-20..=20).map(|i| i as f64).collect();
        for w in points.windows(2) {
            let c0 = dist.cdf(w[0]);
            let c1 = dist.cdf(w[1]);
            prop_assert!(c1 >= c0 - 1e-10,
                "CDF should be monotonic: CDF({}) = {} > CDF({}) = {}", w[0], c0, w[1], c1);
        }
    }

    #[test]
    fn normal_cdf_bounds(mu in -10.0_f64..10.0, sigma in 0.1_f64..10.0, x in -50.0_f64..50.0) {
        let dist = Normal::new(mu, sigma).unwrap();
        let c = dist.cdf(x);
        prop_assert!(c >= -1e-10, "CDF should be >= 0, got {}", c);
        prop_assert!(c <= 1.0 + 1e-10, "CDF should be <= 1, got {}", c);
    }

    #[test]
    fn normal_sample_mean_converges(mu in -5.0_f64..5.0, sigma in 0.5_f64..3.0) {
        let dist = Normal::new(mu, sigma).unwrap();
        let mut rng = Rng::new(42);
        let samples = dist.sample_n(&mut rng, 10_000);
        let sample_mean = descriptive::mean(&samples).unwrap();
        prop_assert!((sample_mean - mu).abs() < 0.2,
            "Sample mean {} should be close to {}", sample_mean, mu);
    }

    #[test]
    fn uniform_sample_in_range(a in -100.0_f64..0.0, width in 1.0_f64..100.0) {
        let b = a + width;
        let dist = Uniform::new(a, b).unwrap();
        let mut rng = Rng::new(123);
        let samples = dist.sample_n(&mut rng, 1000);
        for s in &samples {
            prop_assert!(*s >= a - 1e-10 && *s <= b + 1e-10,
                "Uniform sample {} out of range [{}, {}]", s, a, b);
        }
    }

    #[test]
    fn uniform_mean_correct(a in -10.0_f64..0.0, width in 1.0_f64..10.0) {
        let b = a + width;
        let dist = Uniform::new(a, b).unwrap();
        let theoretical_mean = f64::midpoint(a, b);
        let mut rng = Rng::new(42);
        let samples = dist.sample_n(&mut rng, 10_000);
        let sample_mean = descriptive::mean(&samples).unwrap();
        prop_assert!((sample_mean - theoretical_mean).abs() < 0.2,
            "Uniform sample mean {} should be close to {}", sample_mean, theoretical_mean);
    }

    #[test]
    fn exponential_pdf_nonnegative(lambda in 0.1_f64..10.0, x in 0.0_f64..50.0) {
        let dist = Exponential::new(lambda).unwrap();
        let p = dist.pdf(x);
        prop_assert!(p >= 0.0, "Exponential PDF should be non-negative, got {}", p);
    }

    #[test]
    fn exponential_cdf_monotonic(lambda in 0.1_f64..5.0) {
        let dist = Exponential::new(lambda).unwrap();
        let points: Vec<f64> = (0..=40).map(|i| i as f64 * 0.5).collect();
        for w in points.windows(2) {
            let c0 = dist.cdf(w[0]);
            let c1 = dist.cdf(w[1]);
            prop_assert!(c1 >= c0 - 1e-10, "Exp CDF not monotonic");
        }
    }
}

// ---------------------------------------------------------------------------
// Correlation properties
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn pearson_self_is_one(data in proptest::collection::vec(-100.0_f64..100.0, 10..100)) {
        let var = descriptive::variance(&data).unwrap();
        if var > 1e-10 {
            let r = scivex_stats::correlation::pearson(&data, &data).unwrap();
            prop_assert!((r - 1.0).abs() < 1e-6,
                "Pearson of data with itself should be 1.0, got {}", r);
        }
    }

    #[test]
    fn pearson_bounded(
        data_a in proptest::collection::vec(-100.0_f64..100.0, 20..100),
        data_b in proptest::collection::vec(-100.0_f64..100.0, 20..100),
    ) {
        let n = data_a.len().min(data_b.len());
        let a = &data_a[..n];
        let b = &data_b[..n];
        let va = descriptive::variance(a).unwrap();
        let vb = descriptive::variance(b).unwrap();
        if va > 1e-10 && vb > 1e-10 {
            let r = scivex_stats::correlation::pearson(a, b).unwrap();
            prop_assert!(r >= -1.0 - 1e-6 && r <= 1.0 + 1e-6,
                "Pearson should be in [-1, 1], got {}", r);
        }
    }
}
