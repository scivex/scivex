//! Reference tests comparing scivex-sym symbolic math results against known
//! values computed by SymPy.
//!
//! Each test encodes a ground-truth result from SymPy (or elementary calculus)
//! and verifies that scivex-sym produces the same answer, either structurally
//! or by numerical evaluation at several sample points.

use std::collections::HashMap;

use scivex_sym::{
    constant, cos, diff, diff_n, exp, ln, simplify, sin, solve_linear, solve_quadratic, var, Expr,
    Polynomial,
};

const TOL: f64 = 1e-10;

/// Helper: evaluate an expression at a single variable binding.
fn eval_at(expr: &Expr, var_name: &str, val: f64) -> f64 {
    let mut vars = HashMap::new();
    vars.insert(var_name.to_owned(), val);
    expr.eval(&vars).unwrap()
}

// -----------------------------------------------------------------------
// 1. Differentiation: d/dx(x^3) = 3x^2
//    SymPy: diff(x**3, x) -> 3*x**2
// -----------------------------------------------------------------------
#[test]
fn diff_x_cubed_equals_3x_squared() {
    let x_cubed = Expr::Pow(Box::new(var("x")), Box::new(constant(3.0)));
    let derivative = diff(&x_cubed, "x");

    // Verify numerically at several points.
    for &x in &[-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 3.0] {
        let actual = eval_at(&derivative, "x", x);
        let expected = 3.0 * x * x;
        assert!(
            (actual - expected).abs() < TOL,
            "d/dx(x^3) at x={x}: expected {expected}, got {actual}"
        );
    }
}

// -----------------------------------------------------------------------
// 2. Differentiation: d/dx(sin(x)) = cos(x)
//    SymPy: diff(sin(x), x) -> cos(x)
// -----------------------------------------------------------------------
#[test]
fn diff_sin_equals_cos() {
    let sin_x = sin(var("x"));
    let derivative = diff(&sin_x, "x");

    let test_points = [
        0.0,
        std::f64::consts::FRAC_PI_4,
        std::f64::consts::FRAC_PI_2,
        std::f64::consts::PI,
        1.5,
    ];
    for &x in &test_points {
        let actual = eval_at(&derivative, "x", x);
        let expected = x.cos();
        assert!(
            (actual - expected).abs() < TOL,
            "d/dx(sin(x)) at x={x}: expected {expected}, got {actual}"
        );
    }
}

// -----------------------------------------------------------------------
// 3. Differentiation: d/dx(e^x) = e^x
//    SymPy: diff(exp(x), x) -> exp(x)
// -----------------------------------------------------------------------
#[test]
fn diff_exp_equals_exp() {
    let exp_x = exp(var("x"));
    let derivative = diff(&exp_x, "x");

    for &x in &[0.0, 1.0, -1.0, 2.0] {
        let actual = eval_at(&derivative, "x", x);
        let expected = x.exp();
        assert!(
            (actual - expected).abs() < TOL,
            "d/dx(e^x) at x={x}: expected {expected}, got {actual}"
        );
    }
}

// -----------------------------------------------------------------------
// 4. Differentiation: d/dx(ln(x)) = 1/x
//    SymPy: diff(ln(x), x) -> 1/x
// -----------------------------------------------------------------------
#[test]
fn diff_ln_equals_one_over_x() {
    let ln_x = ln(var("x"));
    let derivative = diff(&ln_x, "x");

    for &x in &[0.5, 1.0, 2.0, 10.0] {
        let actual = eval_at(&derivative, "x", x);
        let expected = 1.0 / x;
        assert!(
            (actual - expected).abs() < TOL,
            "d/dx(ln(x)) at x={x}: expected {expected}, got {actual}"
        );
    }
}

// -----------------------------------------------------------------------
// 5. Differentiation: d/dx(cos(x)) = -sin(x)
//    SymPy: diff(cos(x), x) -> -sin(x)
// -----------------------------------------------------------------------
#[test]
fn diff_cos_equals_neg_sin() {
    let cos_x = cos(var("x"));
    let derivative = diff(&cos_x, "x");

    for &x in &[0.0, std::f64::consts::FRAC_PI_2, 1.0, 3.0] {
        let actual = eval_at(&derivative, "x", x);
        let expected = -x.sin();
        assert!(
            (actual - expected).abs() < TOL,
            "d/dx(cos(x)) at x={x}: expected {expected}, got {actual}"
        );
    }
}

// -----------------------------------------------------------------------
// 6. Second derivative: d^2/dx^2(x^4) = 12x^2
//    SymPy: diff(x**4, x, 2) -> 12*x**2
// -----------------------------------------------------------------------
#[test]
fn second_derivative_x_to_the_4() {
    let x4 = Expr::Pow(Box::new(var("x")), Box::new(constant(4.0)));
    let d2 = diff_n(&x4, "x", 2);

    for &x in &[-2.0, -1.0, 0.0, 1.0, 2.0] {
        let actual = eval_at(&d2, "x", x);
        let expected = 12.0 * x * x;
        assert!(
            (actual - expected).abs() < TOL,
            "d^2/dx^2(x^4) at x={x}: expected {expected}, got {actual}"
        );
    }
}

// -----------------------------------------------------------------------
// 7. Solve linear: 2x - 6 = 0  =>  x = 3
//    SymPy: solve(2*x - 6, x) -> [3]
// -----------------------------------------------------------------------
#[test]
fn solve_linear_2x_minus_6() {
    let expr = constant(2.0) * var("x") + constant(-6.0);
    let sol = solve_linear(&expr, "x").unwrap();
    let val = sol.eval(&HashMap::new()).unwrap();
    assert!(
        (val - 3.0).abs() < TOL,
        "solve(2x - 6 = 0) expected 3.0, got {val}"
    );
}

// -----------------------------------------------------------------------
// 8. Solve quadratic: x^2 - 5x + 6 = 0  =>  {2, 3}
//    SymPy: solve(x**2 - 5*x + 6, x) -> [2, 3]
// -----------------------------------------------------------------------
#[test]
fn solve_quadratic_x2_minus_5x_plus_6() {
    let x = var("x");
    let expr = Expr::Pow(Box::new(x.clone()), Box::new(constant(2.0)))
        + constant(-5.0) * x
        + constant(6.0);
    let roots = solve_quadratic(&expr, "x").unwrap();
    assert_eq!(roots.len(), 2, "expected 2 roots");

    let empty = HashMap::new();
    let r0 = roots[0].eval(&empty).unwrap();
    let r1 = roots[1].eval(&empty).unwrap();
    // Roots should be 2 and 3 (sorted ascending).
    assert!(
        (r0 - 2.0).abs() < TOL,
        "first root: expected 2.0, got {r0}"
    );
    assert!(
        (r1 - 3.0).abs() < TOL,
        "second root: expected 3.0, got {r1}"
    );
}

// -----------------------------------------------------------------------
// 9. Polynomial evaluation: p(x) = x^3 - 2x + 1, p(2) = 5
//    SymPy: Poly([1, 0, -2, 1], x).eval(2) -> 5
// -----------------------------------------------------------------------
#[test]
fn polynomial_eval_cubic() {
    // Ascending power order: coeffs = [1, -2, 0, 1]  =>  1 - 2x + 0x^2 + x^3
    let p = Polynomial::<f64>::new(vec![1.0, -2.0, 0.0, 1.0]);
    let val = p.eval(2.0);
    // p(2) = 1 - 4 + 0 + 8 = 5
    assert!(
        (val - 5.0).abs() < TOL,
        "p(2) expected 5.0, got {val}"
    );

    // Also verify at x=0 and x=-1.
    // p(0) = 1
    assert!((p.eval(0.0) - 1.0_f64).abs() < TOL);
    // p(-1) = 1 + 2 + 0 - 1 = 2
    assert!((p.eval(-1.0) - 2.0_f64).abs() < TOL);
}

// -----------------------------------------------------------------------
// 10. Simplification identities:
//     simplify(x + 0) = x
//     simplify(x * 1) = x
//     simplify(x * 0) = 0
//     simplify(x ^ 1) = x
//     simplify(x ^ 0) = 1
//     SymPy: simplify(x + 0) -> x, simplify(x * 1) -> x, etc.
// -----------------------------------------------------------------------
#[test]
fn simplify_identity_rules() {
    // x + 0 = x
    let e = var("x") + constant(0.0);
    assert_eq!(simplify(&e), var("x"), "x + 0 should simplify to x");

    // 0 + x = x
    let e = constant(0.0) + var("x");
    assert_eq!(simplify(&e), var("x"), "0 + x should simplify to x");

    // x * 1 = x
    let e = var("x") * constant(1.0);
    assert_eq!(simplify(&e), var("x"), "x * 1 should simplify to x");

    // 1 * x = x
    let e = constant(1.0) * var("x");
    assert_eq!(simplify(&e), var("x"), "1 * x should simplify to x");

    // x * 0 = 0
    let e = var("x") * constant(0.0);
    assert_eq!(
        simplify(&e),
        constant(0.0),
        "x * 0 should simplify to 0"
    );

    // x ^ 1 = x
    let e = Expr::Pow(Box::new(var("x")), Box::new(constant(1.0)));
    assert_eq!(simplify(&e), var("x"), "x^1 should simplify to x");

    // x ^ 0 = 1
    let e = Expr::Pow(Box::new(var("x")), Box::new(constant(0.0)));
    assert_eq!(
        simplify(&e),
        constant(1.0),
        "x^0 should simplify to 1"
    );
}

// -----------------------------------------------------------------------
// 11. Constant folding: simplify(2 + 3) = 5, simplify(4 * 5) = 20
//     SymPy: simplify(2 + 3) -> 5
// -----------------------------------------------------------------------
#[test]
fn simplify_constant_folding() {
    let e = constant(2.0) + constant(3.0);
    assert_eq!(simplify(&e), constant(5.0));

    let e = constant(4.0) * constant(5.0);
    assert_eq!(simplify(&e), constant(20.0));

    let e = Expr::Pow(Box::new(constant(2.0)), Box::new(constant(3.0)));
    assert_eq!(simplify(&e), constant(8.0));
}

// -----------------------------------------------------------------------
// 12. Chain rule: d/dx(sin(x^2)) = 2x * cos(x^2)
//     SymPy: diff(sin(x**2), x) -> 2*x*cos(x**2)
// -----------------------------------------------------------------------
#[test]
fn diff_chain_rule_sin_x_squared() {
    let x_sq = Expr::Pow(Box::new(var("x")), Box::new(constant(2.0)));
    let expr = sin(x_sq);
    let derivative = diff(&expr, "x");

    for &x in &[0.0, 0.5, 1.0, 2.0, -1.0] {
        let actual = eval_at(&derivative, "x", x);
        let expected = 2.0 * x * (x * x).cos();
        assert!(
            (actual - expected).abs() < TOL,
            "d/dx(sin(x^2)) at x={x}: expected {expected}, got {actual}"
        );
    }
}

// -----------------------------------------------------------------------
// 13. Polynomial roots via the Polynomial type:
//     x^2 - 5x + 6 roots are 2.0 and 3.0
//     SymPy: Poly(x**2 - 5*x + 6).all_roots() -> [2, 3]
// -----------------------------------------------------------------------
#[test]
fn polynomial_roots_quadratic() {
    // coeffs in ascending order: c + bx + ax^2 => [6, -5, 1]
    let p = Polynomial::<f64>::new(vec![6.0, -5.0, 1.0]);
    let roots = p.roots().unwrap();
    assert_eq!(roots.len(), 2);
    let r0: f64 = roots[0];
    let r1: f64 = roots[1];
    assert!((r0 - 2.0).abs() < TOL, "root 0: {r0}");
    assert!((r1 - 3.0).abs() < TOL, "root 1: {r1}");
}

// -----------------------------------------------------------------------
// 14. Polynomial from_expr round-trip:
//     Build x^2 + 3x + 2 symbolically, extract polynomial, verify eval.
//     SymPy: Poly(x**2 + 3*x + 2).eval(1) -> 6
// -----------------------------------------------------------------------
#[test]
fn polynomial_from_expr_eval() {
    let x = var("x");
    let expr = Expr::Pow(Box::new(x.clone()), Box::new(constant(2.0)))
        + constant(3.0) * x
        + constant(2.0);
    let p = Polynomial::from_expr(&expr, "x").unwrap();

    // p(0) = 2, p(1) = 6, p(-1) = 0, p(2) = 12
    assert!((p.eval(0.0) - 2.0).abs() < TOL);
    assert!((p.eval(1.0) - 6.0).abs() < TOL);
    assert!((p.eval(-1.0) - 0.0).abs() < TOL);
    assert!((p.eval(2.0) - 12.0).abs() < TOL);
}

// -----------------------------------------------------------------------
// 15. Double negation: simplify(--x) = x
//     SymPy: simplify(-(-x)) -> x
// -----------------------------------------------------------------------
#[test]
fn simplify_double_negation() {
    let e = Expr::Neg(Box::new(Expr::Neg(Box::new(var("x")))));
    assert_eq!(simplify(&e), var("x"));
}
