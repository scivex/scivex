//! Reference value tests comparing scivex-sym outputs against pre-computed SymPy values.
//!
//! Every expected value in this file was computed with SymPy and hard-coded so
//! the test suite can detect regressions without requiring a Python install.

use std::collections::HashMap;

use scivex_sym::{
    Expr, Polynomial, constant, definite_integral, diff, diff_n, exp, expand, integrate, ln,
    maclaurin, simplify, sin, solve_linear, solve_quadratic, var,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const TOL: f64 = 1e-8;

fn vars(x: f64) -> HashMap<String, f64> {
    let mut m = HashMap::new();
    m.insert("x".into(), x);
    m
}

fn eval(e: &Expr, x: f64) -> f64 {
    e.eval(&vars(x)).unwrap()
}

fn assert_close(actual: f64, expected: f64, label: &str) {
    assert!(
        (actual - expected).abs() < TOL,
        "{label}: expected {expected}, got {actual} (delta = {})",
        (actual - expected).abs()
    );
}

// ===========================================================================
// 1. First derivatives
// ===========================================================================

#[test]
fn sympy_diff_x_cubed() {
    // diff(x**3, x) = 3*x**2  ->  at x=2: 12.0
    let x = var("x");
    let expr = Expr::Pow(Box::new(x), Box::new(constant(3.0)));
    let d = diff(&expr, "x");
    assert_close(eval(&d, 2.0), 12.0, "d/dx(x^3) at x=2");
}

#[test]
fn sympy_diff_sin() {
    // diff(sin(x), x) = cos(x)  ->  at x=0: 1.0
    let d = diff(&sin(var("x")), "x");
    assert_close(eval(&d, 0.0), 1.0, "d/dx(sin(x)) at x=0");
}

#[test]
fn sympy_diff_exp() {
    // diff(exp(x), x) = exp(x)  ->  at x=1: e
    let d = diff(&exp(var("x")), "x");
    assert_close(eval(&d, 1.0), std::f64::consts::E, "d/dx(exp(x)) at x=1");
}

#[test]
fn sympy_diff_ln() {
    // diff(ln(x), x) = 1/x  ->  at x=2: 0.5
    let d = diff(&ln(var("x")), "x");
    assert_close(eval(&d, 2.0), 0.5, "d/dx(ln(x)) at x=2");
}

#[test]
fn sympy_diff_product_x2_sinx() {
    // diff(x**2 * sin(x), x) = 2*x*sin(x) + x**2*cos(x)
    // At x=pi: 2*pi*sin(pi) + pi^2*cos(pi) = 0 + pi^2*(-1) = -pi^2
    let x = var("x");
    let x_sq = Expr::Pow(Box::new(x.clone()), Box::new(constant(2.0)));
    let expr = Expr::Mul(Box::new(x_sq), Box::new(sin(x)));
    let d = diff(&expr, "x");
    let expected = -(std::f64::consts::PI * std::f64::consts::PI); // -9.8696...
    assert_close(
        eval(&d, std::f64::consts::PI),
        expected,
        "d/dx(x^2*sin(x)) at x=pi",
    );
}

// ===========================================================================
// 2. Second derivatives
// ===========================================================================

#[test]
fn sympy_diff2_x4() {
    // diff(x**4, x, 2) = 12*x**2  ->  at x=3: 108.0
    let expr = Expr::Pow(Box::new(var("x")), Box::new(constant(4.0)));
    let d2 = diff_n(&expr, "x", 2);
    assert_close(eval(&d2, 3.0), 108.0, "d^2/dx^2(x^4) at x=3");
}

#[test]
fn sympy_diff2_sin() {
    // diff(sin(x), x, 2) = -sin(x)  ->  at x=pi/2: -1.0
    let d2 = diff_n(&sin(var("x")), "x", 2);
    assert_close(
        eval(&d2, std::f64::consts::FRAC_PI_2),
        -1.0,
        "d^2/dx^2(sin(x)) at x=pi/2",
    );
}

// ===========================================================================
// 3. Simplification identities
// ===========================================================================

#[test]
fn sympy_simplify_x_plus_0() {
    // simplify(x + 0) = x
    let s = simplify(&(var("x") + constant(0.0)));
    assert_eq!(s, var("x"), "simplify(x + 0) should be x");
}

#[test]
fn sympy_simplify_x_times_1() {
    // simplify(x * 1) = x
    let s = simplify(&(var("x") * constant(1.0)));
    assert_eq!(s, var("x"), "simplify(x * 1) should be x");
}

#[test]
fn sympy_simplify_x_times_0() {
    // simplify(x * 0) = 0
    let s = simplify(&(var("x") * constant(0.0)));
    assert_eq!(s, constant(0.0), "simplify(x * 0) should be 0");
}

#[test]
fn sympy_simplify_x_pow_1() {
    // simplify(x**1) = x
    let s = simplify(&Expr::Pow(Box::new(var("x")), Box::new(constant(1.0))));
    assert_eq!(s, var("x"), "simplify(x^1) should be x");
}

#[test]
fn sympy_simplify_x_pow_0() {
    // simplify(x**0) = 1
    let s = simplify(&Expr::Pow(Box::new(var("x")), Box::new(constant(0.0))));
    assert_eq!(s, constant(1.0), "simplify(x^0) should be 1");
}

// ===========================================================================
// 4. Expansion
// ===========================================================================

#[test]
fn sympy_expand_x_plus_1_times_x_plus_2() {
    // expand((x+1)*(x+2)) = x^2 + 3x + 2  ->  at x=3: 20.0
    let expr = (var("x") + constant(1.0)) * (var("x") + constant(2.0));
    let expanded = expand(&expr);
    assert_close(eval(&expanded, 3.0), 20.0, "expand((x+1)(x+2)) at x=3");
}

#[test]
fn sympy_expand_x_plus_1_squared() {
    // expand((x+1)**2) = x^2 + 2x + 1  ->  at x=5: 36.0
    let expr = Expr::Pow(Box::new(var("x") + constant(1.0)), Box::new(constant(2.0)));
    let expanded = expand(&expr);
    assert_close(eval(&expanded, 5.0), 36.0, "expand((x+1)^2) at x=5");
}

// ===========================================================================
// 5. Linear solving
// ===========================================================================

#[test]
fn sympy_solve_linear_2x_minus_6() {
    // 2*x - 6 = 0  ->  x = 3
    let expr = constant(2.0) * var("x") + constant(-6.0);
    let sol = solve_linear(&expr, "x").unwrap();
    let val = sol.eval(&HashMap::new()).unwrap();
    assert_close(val, 3.0, "solve 2x - 6 = 0");
}

#[test]
fn sympy_solve_linear_5x_plus_10() {
    // 5*x + 10 = 0  ->  x = -2
    let expr = constant(5.0) * var("x") + constant(10.0);
    let sol = solve_linear(&expr, "x").unwrap();
    let val = sol.eval(&HashMap::new()).unwrap();
    assert_close(val, -2.0, "solve 5x + 10 = 0");
}

// ===========================================================================
// 6. Quadratic solving
// ===========================================================================

#[test]
fn sympy_solve_quadratic_x2_minus_5x_plus_6() {
    // x^2 - 5x + 6 = 0  ->  x = 2, 3
    let x = var("x");
    let expr = Expr::Pow(Box::new(x.clone()), Box::new(constant(2.0)))
        + constant(-5.0) * x
        + constant(6.0);
    let roots = solve_quadratic(&expr, "x").unwrap();
    assert_eq!(roots.len(), 2, "should have 2 roots");
    let empty = HashMap::new();
    let r0 = roots[0].eval(&empty).unwrap();
    let r1 = roots[1].eval(&empty).unwrap();
    assert_close(r0, 2.0, "root 1 of x^2 - 5x + 6");
    assert_close(r1, 3.0, "root 2 of x^2 - 5x + 6");
}

#[test]
fn sympy_solve_quadratic_x2_minus_1() {
    // x^2 - 1 = 0  ->  x = -1, 1
    let expr = Expr::Pow(Box::new(var("x")), Box::new(constant(2.0))) + constant(-1.0);
    let roots = solve_quadratic(&expr, "x").unwrap();
    assert_eq!(roots.len(), 2, "should have 2 roots");
    let empty = HashMap::new();
    let r0 = roots[0].eval(&empty).unwrap();
    let r1 = roots[1].eval(&empty).unwrap();
    assert_close(r0, -1.0, "root 1 of x^2 - 1");
    assert_close(r1, 1.0, "root 2 of x^2 - 1");
}

// ===========================================================================
// 7. Integration (definite)
// ===========================================================================

#[test]
fn sympy_integrate_x2_definite_0_to_3() {
    // integrate(x**2, x) from 0 to 3 = 9.0
    let expr = Expr::Pow(Box::new(var("x")), Box::new(constant(2.0)));
    let result = definite_integral(&expr, "x", 0.0, 3.0).unwrap();
    assert_close(result, 9.0, "int(x^2, 0..3)");
}

#[test]
fn sympy_integrate_sin_definite_0_to_pi() {
    // integrate(sin(x), x) from 0 to pi = 2.0
    let result = definite_integral(&sin(var("x")), "x", 0.0, std::f64::consts::PI).unwrap();
    assert_close(result, 2.0, "int(sin(x), 0..pi)");
}

#[test]
fn sympy_integrate_exp_definite_0_to_1() {
    // integrate(exp(x), x) from 0 to 1 = e - 1 ≈ 1.71828
    let result = definite_integral(&exp(var("x")), "x", 0.0, 1.0).unwrap();
    assert_close(result, std::f64::consts::E - 1.0, "int(exp(x), 0..1)");
}

#[test]
fn sympy_integrate_x2_indefinite() {
    // integrate(x**2, x) = x^3/3  ->  verify antiderivative at x=3: 9.0
    let expr = Expr::Pow(Box::new(var("x")), Box::new(constant(2.0)));
    let anti = integrate(&expr, "x").unwrap();
    assert_close(eval(&anti, 3.0), 9.0, "antiderivative of x^2 at x=3");
}

#[test]
fn sympy_integrate_sin_indefinite() {
    // integrate(sin(x), x) = -cos(x)
    // -cos(pi) - (-cos(0)) = 1 - (-1) = 2
    let anti = integrate(&sin(var("x")), "x").unwrap();
    let at_pi = eval(&anti, std::f64::consts::PI);
    let at_0 = eval(&anti, 0.0);
    assert_close(at_pi - at_0, 2.0, "int(sin(x)) from 0 to pi via antideriv");
}

#[test]
fn sympy_integrate_exp_indefinite() {
    // integrate(exp(x), x) = exp(x)
    // exp(1) - exp(0) = e - 1
    let anti = integrate(&exp(var("x")), "x").unwrap();
    let at_1 = eval(&anti, 1.0);
    let at_0 = eval(&anti, 0.0);
    assert_close(
        at_1 - at_0,
        std::f64::consts::E - 1.0,
        "int(exp(x)) from 0 to 1 via antideriv",
    );
}

// ===========================================================================
// 8. Taylor / Maclaurin series
// ===========================================================================

#[test]
fn sympy_taylor_exp_order_3() {
    // e^x around 0, order 3: 1 + x + x^2/2 + x^3/6
    // At x=1: 1 + 1 + 0.5 + 0.16667 = 2.66667
    let t = maclaurin(&exp(var("x")), "x", 3).unwrap();
    let approx = eval(&t, 1.0);
    let expected = 1.0 + 1.0 + 0.5 + 1.0 / 6.0; // 2.66667
    assert_close(approx, expected, "Taylor(e^x, order=3) at x=1");
}

#[test]
fn sympy_taylor_exp_order_3_vs_actual() {
    // The order-3 Taylor approximation at x=1 should be close to (but less
    // than) e.  Verify it is within 0.06 of the true value.
    let t = maclaurin(&exp(var("x")), "x", 3).unwrap();
    let approx = eval(&t, 1.0);
    let exact = std::f64::consts::E;
    assert!(
        (approx - exact).abs() < 0.06,
        "Taylor(e^x, order=3) at x=1: approx={approx}, exact={exact}",
    );
}

// ===========================================================================
// 9. Polynomial evaluation (Horner)
// ===========================================================================

#[test]
fn sympy_polynomial_horner_at_2() {
    // p(x) = 2x^3 - 3x^2 + x - 5
    // coeffs ascending: [-5, 1, -3, 2]
    // p(2) = 16 - 12 + 2 - 5 = 1
    let p = Polynomial::<f64>::new(vec![-5.0, 1.0, -3.0, 2.0]);
    assert_close(p.eval(2.0), 1.0, "p(2) for 2x^3 - 3x^2 + x - 5");
}

#[test]
fn sympy_polynomial_horner_at_neg1() {
    // p(x) = 2x^3 - 3x^2 + x - 5
    // p(-1) = -2 - 3 - 1 - 5 = -11
    let p = Polynomial::<f64>::new(vec![-5.0, 1.0, -3.0, 2.0]);
    assert_close(p.eval(-1.0), -11.0, "p(-1) for 2x^3 - 3x^2 + x - 5");
}

// ===========================================================================
// 10. Round-trip consistency: differentiate then integrate
// ===========================================================================

#[test]
fn round_trip_diff_then_integrate_x3() {
    // d/dx(x^3) = 3x^2.  Integrating 3x^2 should give x^3 (up to constant).
    // Verify via definite integral: int(3x^2, 0..2) = 2^3 - 0^3 = 8.
    let x3 = Expr::Pow(Box::new(var("x")), Box::new(constant(3.0)));
    let deriv = diff(&x3, "x");
    let result = definite_integral(&deriv, "x", 0.0, 2.0).unwrap();
    assert_close(result, 8.0, "round-trip d/dx(x^3) then integrate 0..2");
}
