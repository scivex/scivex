//! Property-based tests for scivex-sym using proptest.

use proptest::prelude::*;
use scivex_sym::{constant, diff, expand, simplify, var};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Simplification properties
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn simplify_idempotent(v in -100.0_f64..100.0) {
        let expr = constant(v) + constant(0.0);
        let s1 = simplify(&expr);
        let s2 = simplify(&s1);
        let vars = HashMap::new();
        let v1 = s1.eval(&vars).unwrap();
        let v2 = s2.eval(&vars).unwrap();
        prop_assert!((v1 - v2).abs() < 1e-10,
            "simplify should be idempotent: {} vs {}", v1, v2);
    }

    #[test]
    fn simplify_add_zero(v in -100.0_f64..100.0) {
        let expr = constant(v) + constant(0.0);
        let simplified = simplify(&expr);
        let result = simplified.eval(&HashMap::new()).unwrap();
        prop_assert!((result - v).abs() < 1e-10,
            "x + 0 should simplify to x: expected {}, got {}", v, result);
    }

    #[test]
    fn simplify_mul_one(v in -100.0_f64..100.0) {
        let expr = constant(v) * constant(1.0);
        let simplified = simplify(&expr);
        let result = simplified.eval(&HashMap::new()).unwrap();
        prop_assert!((result - v).abs() < 1e-10,
            "x * 1 should simplify to x: expected {}, got {}", v, result);
    }

    #[test]
    fn simplify_mul_zero(v in -100.0_f64..100.0) {
        let expr = constant(v) * constant(0.0);
        let simplified = simplify(&expr);
        let result = simplified.eval(&HashMap::new()).unwrap();
        prop_assert!(result.abs() < 1e-10,
            "x * 0 should simplify to 0, got {}", result);
    }

    #[test]
    fn constant_folding(a in -100.0_f64..100.0, b in -100.0_f64..100.0) {
        let expr = constant(a) + constant(b);
        let simplified = simplify(&expr);
        let result = simplified.eval(&HashMap::new()).unwrap();
        prop_assert!((result - (a + b)).abs() < 1e-10,
            "constant folding: {} + {} should be {}, got {}", a, b, a + b, result);
    }
}

// ---------------------------------------------------------------------------
// Differentiation properties
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn diff_of_constant_is_zero(v in -100.0_f64..100.0) {
        let expr = constant(v);
        let d = diff(&expr, "x");
        let simplified = simplify(&d);
        let result = simplified.eval(&HashMap::new()).unwrap();
        prop_assert!(result.abs() < 1e-10,
            "d/dx(constant) should be 0, got {}", result);
    }

    #[test]
    fn diff_of_x_is_one(x_val in -10.0_f64..10.0) {
        let x = var("x");
        let d = diff(&x, "x");
        let simplified = simplify(&d);
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), x_val);
        let result = simplified.eval(&vars).unwrap();
        prop_assert!((result - 1.0).abs() < 1e-10,
            "d/dx(x) should be 1, got {}", result);
    }

    #[test]
    fn diff_linearity_constant_multiple(c in -10.0_f64..10.0, x_val in -5.0_f64..5.0) {
        // d/dx(c*x) = c
        let expr = constant(c) * var("x");
        let d = simplify(&diff(&expr, "x"));
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), x_val);
        let result = d.eval(&vars).unwrap();
        prop_assert!((result - c).abs() < 1e-6,
            "d/dx({}*x) should be {}, got {}", c, c, result);
    }

    #[test]
    fn diff_sum_rule(x_val in -5.0_f64..5.0) {
        // d/dx(x + x) should equal d/dx(x) + d/dx(x) = 2
        let x = var("x");
        let expr = x.clone() + x;
        let d = simplify(&diff(&expr, "x"));
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), x_val);
        let result = d.eval(&vars).unwrap();
        prop_assert!((result - 2.0).abs() < 1e-6,
            "d/dx(x + x) should be 2, got {}", result);
    }

    #[test]
    fn diff_preserves_eval_via_finite_diff(x_val in -3.0_f64..3.0) {
        // Numerical check: d/dx(x^2) at x_val ≈ 2*x_val
        let x = var("x");
        let expr = x.clone() * x;
        let d = simplify(&diff(&expr, "x"));
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), x_val);
        let analytical = d.eval(&vars).unwrap();
        let expected = 2.0 * x_val;
        prop_assert!((analytical - expected).abs() < 1e-6,
            "d/dx(x^2) at {} should be {}, got {}", x_val, expected, analytical);
    }
}

// ---------------------------------------------------------------------------
// Expand properties
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn expand_preserves_value(x_val in -5.0_f64..5.0) {
        // (x + 1) * (x + 2) should equal its expansion
        let x = var("x");
        let expr = (x.clone() + constant(1.0)) * (x + constant(2.0));
        let expanded = expand(&expr);
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), x_val);
        let original_val = expr.eval(&vars).unwrap();
        let expanded_val = expanded.eval(&vars).unwrap();
        prop_assert!((original_val - expanded_val).abs() < 1e-6,
            "expand should preserve value: {} vs {}", original_val, expanded_val);
    }
}

// ---------------------------------------------------------------------------
// Eval properties
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn eval_addition_correct(a in -100.0_f64..100.0, b in -100.0_f64..100.0) {
        let expr = constant(a) + constant(b);
        let result = expr.eval(&HashMap::new()).unwrap();
        prop_assert!((result - (a + b)).abs() < 1e-10);
    }

    #[test]
    fn eval_multiplication_correct(a in -100.0_f64..100.0, b in -100.0_f64..100.0) {
        let expr = constant(a) * constant(b);
        let result = expr.eval(&HashMap::new()).unwrap();
        prop_assert!((result - (a * b)).abs() < 1e-8);
    }

    #[test]
    fn eval_substitution_correct(x_val in -10.0_f64..10.0) {
        let x = var("x");
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), x_val);
        let result = x.eval(&vars).unwrap();
        prop_assert!((result - x_val).abs() < 1e-10);
    }
}
