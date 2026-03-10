# scivex-sym

Symbolic mathematics for Scivex. Expression trees, automatic differentiation,
algebraic simplification, and equation solving.

## Highlights

- **Expression AST** — `Expr` enum with `Const`, `Var`, `Add`, `Mul`, `Pow`, `Neg`, `Fn`
- **Operator overloading** — Write `var("x") + constant(1.0)` naturally
- **Evaluation** — `eval()` with variable bindings
- **Substitution** — Replace variables with expressions
- **Differentiation** — `diff()` with chain rule, `diff_n()` for higher orders
- **Simplification** — Constant folding, identity reduction (x+0, x*1, x^0, etc.)
- **Algebra** — `expand()` distributes products, `factor_out()` extracts common terms
- **Equation solving** — `solve_linear()`, `solve_quadratic()` with symbolic results
- **Polynomials** — `Polynomial<T>` with Horner evaluation, arithmetic, root finding

## Usage

```rust
use scivex_sym::prelude::*;

// Build expressions
let x = var("x");
let expr = x.clone() * x.clone() + constant(3.0) * x.clone() + constant(2.0);

// Differentiate: d/dx(x^2 + 3x + 2) = 2x + 3
let deriv = diff(&expr, "x");

// Simplify
let simplified = simplify(&(var("x") + zero()));  // → x

// Solve: x^2 - 5x + 6 = 0 → [2, 3]
let roots = solve_quadratic(&expr_eq, "x").unwrap();

// Polynomials
let p = Polynomial::new(vec![2.0, 3.0, 1.0]); // 2 + 3x + x^2
let val = p.eval(4.0); // Horner's method
```

## License

MIT
