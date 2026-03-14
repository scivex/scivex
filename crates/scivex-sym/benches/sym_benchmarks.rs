use criterion::{Criterion, black_box, criterion_group, criterion_main};
use scivex_sym::{constant, diff, expand, simplify, var};

// ---------------------------------------------------------------------------
// Expression construction
// ---------------------------------------------------------------------------

fn bench_expr_build(c: &mut Criterion) {
    c.bench_function("sym_expr_build_polynomial", |b| {
        b.iter(|| {
            // Build: 3*x^2 + 2*x + 1
            let x = var("x");
            let _expr = constant(3.0) * x.clone() * x.clone() + constant(2.0) * x + constant(1.0);
        });
    });
}

// ---------------------------------------------------------------------------
// Simplify
// ---------------------------------------------------------------------------

fn bench_simplify_identity(c: &mut Criterion) {
    // x + 0 + x * 1 + 0 * y should simplify to x + x = 2x
    let x = var("x");
    let y = var("y");
    let expr = x.clone() + constant(0.0) + x * constant(1.0) + constant(0.0) * y;

    c.bench_function("sym_simplify_identities", |b| {
        b.iter(|| simplify(black_box(&expr)));
    });
}

fn bench_simplify_nested(c: &mut Criterion) {
    // Build a more complex expression: (x+1)*(x-1) + x^2 - 1
    let x = var("x");
    let expr = (x.clone() + constant(1.0)) * (x.clone() - constant(1.0)) + x.clone() * x.clone()
        - constant(1.0);

    c.bench_function("sym_simplify_nested", |b| {
        b.iter(|| simplify(black_box(&expr)));
    });
}

// ---------------------------------------------------------------------------
// Differentiation
// ---------------------------------------------------------------------------

fn bench_diff_polynomial(c: &mut Criterion) {
    // d/dx (3x^3 + 2x^2 + x + 5)
    let x = var("x");
    let expr = constant(3.0) * x.clone() * x.clone() * x.clone()
        + constant(2.0) * x.clone() * x.clone()
        + x
        + constant(5.0);

    c.bench_function("sym_diff_cubic", |b| {
        b.iter(|| diff(black_box(&expr), "x"));
    });
}

fn bench_diff_trig(c: &mut Criterion) {
    use scivex_sym::{cos, sin};
    // d/dx (sin(x) * cos(x))
    let x = var("x");
    let expr = sin(x.clone()) * cos(x);

    c.bench_function("sym_diff_sin_cos", |b| {
        b.iter(|| diff(black_box(&expr), "x"));
    });
}

fn bench_diff_chain_rule(c: &mut Criterion) {
    use scivex_sym::{exp, sin};
    // d/dx (exp(sin(x)))
    let x = var("x");
    let expr = exp(sin(x));

    c.bench_function("sym_diff_chain_rule", |b| {
        b.iter(|| diff(black_box(&expr), "x"));
    });
}

// ---------------------------------------------------------------------------
// Expand
// ---------------------------------------------------------------------------

fn bench_expand(c: &mut Criterion) {
    // expand((x+1)*(x+2)*(x+3))
    let x = var("x");
    let expr = (x.clone() + constant(1.0)) * (x.clone() + constant(2.0)) * (x + constant(3.0));

    c.bench_function("sym_expand_triple_product", |b| {
        b.iter(|| expand(black_box(&expr)));
    });
}

// ---------------------------------------------------------------------------
// Eval
// ---------------------------------------------------------------------------

fn bench_eval(c: &mut Criterion) {
    use scivex_sym::{exp, sin};
    use std::collections::HashMap;
    // Evaluate sin(x) * exp(x) + x^2 at x=1.5
    let x = var("x");
    let expr = sin(x.clone()) * exp(x.clone()) + x.clone() * x;
    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 1.5);

    c.bench_function("sym_eval_complex", |b| {
        b.iter(|| expr.eval(black_box(&vars)).unwrap());
    });
}

criterion_group!(
    benches,
    bench_expr_build,
    bench_simplify_identity,
    bench_simplify_nested,
    bench_diff_polynomial,
    bench_diff_trig,
    bench_diff_chain_rule,
    bench_expand,
    bench_eval,
);
criterion_main!(benches);
