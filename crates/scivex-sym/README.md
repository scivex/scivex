# scivex-sym

Symbolic mathematics for Scivex. Computer algebra system with expression
manipulation, calculus, and equation solving.

## Highlights

- **Expression AST** — Variables, constants, arithmetic, functions, powers
- **Simplification** — Algebraic simplification, constant folding, canonical forms
- **Differentiation** — Symbolic derivatives with chain rule
- **Integration** — Symbolic integration for common patterns
- **Equation solving** — Solve single-variable equations symbolically
- **Polynomials** — Polynomial arithmetic, roots, GCD, factoring
- **Substitution** — Variable substitution and evaluation
- **Pretty printing** — Human-readable expression formatting
- **LaTeX output** — Render expressions as LaTeX strings

## Usage

```rust
use scivex_sym::prelude::*;

let x = Expr::var("x");
let f = &x * &x + 2.0 * &x + 1.0;

// Differentiate
let df = f.diff("x");  // 2*x + 2

// Simplify
let simplified = f.simplify();  // (x + 1)^2

// Solve
let roots = solve(&f, "x").unwrap();
```

## License

MIT
