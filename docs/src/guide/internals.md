# Internals & Contributing Guide

This document explains Scivex's architecture and how to contribute.

## Architecture Overview

Scivex is a Cargo workspace of 19 crates. The dependency graph flows downward:

```
                      scivex (umbrella)
                            │
      ┌─────────┬───────┬───┴───┬────────┬─────────┐
      │         │       │       │        │         │
  scivex-nn scivex-ml scivex-nlp scivex-viz scivex-signal scivex-image
      │         │       │       │        │         │
      ├─────────┴───────┘       │        │         │
      │                         │        │         │
  scivex-optim            scivex-graph   │         │
      │                         │        │         │
      ├─────────────────────────┴────────┴─────────┘
      │
  scivex-stats
      │
  scivex-frame ── scivex-io
      │
  scivex-core  (foundation — zero external deps for math)
```

**Key rule:** All arrows point downward. A crate may only depend on crates below it.

## The Type Hierarchy

All numeric computation is generic over a trait hierarchy:

```
Scalar          — basic numeric operations (+, -, *, /, from_f64, abs)
  └── Float     — floating-point ops (sqrt, exp, ln, sin, cos, powi)
        └── Real    — real number ops (currently == Float for f32/f64)
              └── Complex  — complex number support
```

`Float` is the workhorse trait — most APIs are generic over `T: Float`:

```rust
pub fn mean<T: Float>(data: &[T]) -> Result<T> { ... }
pub fn matmul<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>> { ... }
```

The `Float` trait is implemented for `f32` and `f64`. The key method is `from_f64(f64) -> Self`
which allows writing constants in algorithms.

## Error Handling

Every crate has its own error type in `error.rs`:

```rust
// scivex-core
pub enum CoreError {
    DimensionMismatch { expected: Vec<usize>, got: Vec<usize> },
    SingularMatrix,
    InvalidShape { reason: &'static str },
    // ...
}

// scivex-ml
pub enum MlError {
    NotFitted,
    InvalidParameter { name: &'static str, reason: &'static str },
    ConvergenceFailure { iterations: usize, residual: f64 },
    // ...
}
```

**Rules:**
- Never panic in library code. Return `Result<T, CrateError>`.
- Error messages must be actionable: include expected vs. actual values.
- Use `&'static str` for compile-time-known messages to avoid allocation.

## `unsafe` Policy

`unsafe` is allowed **only** in `scivex-core`. Every `unsafe` block must have a
`// SAFETY: ...` comment explaining which invariant the code relies on:

```rust
// SAFETY: `i` is bounds-checked above, and `data` is a valid allocation
// of at least `len` elements.
unsafe { *self.data.add(i) }
```

All other crates build on `scivex-core`'s safe abstractions. If you need `unsafe`
outside `scivex-core`, discuss in the PR — there may be a safe alternative.

## Memory Model

- **Stack-first allocation:** Small tensors and matrices are stack-allocated by default.
- **No heap in hot paths** without opt-in. Use `into_boxed()` for explicit heap allocation.
- **Arena allocator:** `scivex_core::arena::Arena` provides bump allocation for temporary
  numeric slices. `SlabPool` recycles fixed-size buffers.

```rust
use scivex_core::arena::Arena;

let arena = Arena::new(1024 * 1024); // 1 MB
let slice = arena.alloc_slice::<f64>(256); // fast bump allocation
// slice is valid until arena is dropped
```

## Adding a New Crate

1. Create the directory structure:
   ```bash
   mkdir -p crates/scivex-{name}/src
   ```

2. Create `Cargo.toml`:
   ```toml
   [package]
   name = "scivex-{name}"
   version = "0.1.0"
   edition = "2024"
   license = "MIT"
   description = "Scivex — {description}"

   [dependencies]
   scivex-core = { path = "../scivex-core" }
   ```

3. Create `src/lib.rs` with module declarations and a prelude.

4. Create `src/error.rs` with the crate-specific error type.

5. Add to workspace `Cargo.toml` members list.

6. Add as optional dependency in umbrella `scivex/Cargo.toml`.

7. Add feature flag in umbrella crate.

8. Write unit tests.

9. Run the full verification suite:
   ```bash
   cargo fmt --all -- --check
   cargo clippy -p scivex-{name} --all-targets -- -D warnings
   cargo test -p scivex-{name}
   cargo doc -p scivex-{name} --no-deps
   cargo test --workspace
   ```

## Adding a New Algorithm

When adding a new algorithm to an existing crate:

1. **Read the existing patterns.** Look at similar algorithms in the crate.
2. **Create a new file** in the appropriate module directory.
3. **Add `mod` and `pub use`** in the module's `mod.rs`.
4. **Follow the type conventions:**
   - Generic over `T: Float`
   - Input data as `&[T]` or `&Tensor<T>`
   - Return `Result<OutputType<T>>`
   - Builder pattern for configuration (`.new()` then `.option(value)`)
5. **Write tests** — at least 3-4 unit tests covering:
   - Happy path with known answer
   - Edge cases (empty input, single element, etc.)
   - Error cases (invalid parameters)
6. **Add doc examples** — at least one `/// # Examples` block.

## Coding Conventions

- **Naming:** See CLAUDE.md for full naming conventions.
- **No panics:** `Result<T, Error>` everywhere.
- **No `unwrap()`** in library code (allowed in tests).
- **Feature gates:** Put optional functionality behind feature flags.
- **`#[must_use]`** on pure functions that return computed values.
- **Doc comments:** Every public item gets `///` documentation.

## CI Pipeline

Every PR runs:

1. **Formatting:** `cargo fmt --all -- --check`
2. **Clippy:** `cargo clippy --workspace --all-targets -- -D warnings`
3. **Tests:** `cargo test --workspace` on Ubuntu, macOS, Windows
4. **Documentation:** `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps`
5. **MSRV:** Tests on Rust 1.85 (minimum supported)
6. **Feature flags:** Tests with individual feature combinations
7. **CodeQL:** Security analysis

### Common CI Issues

- **MSRV compatibility:** Some stable Rust features are not available on 1.85.
  Check before using new APIs like `is_multiple_of()` (1.90+) or `if let` chains (1.87+).
- **Clippy pedantic:** We run with `-D warnings`. Use `#[allow(clippy::lint_name)]`
  sparingly with a comment explaining why.
- **Rustdoc links:** Use backtick references `` `TypeName` `` instead of `[TypeName]`
  in module-level `//!` docs to avoid broken intra-doc link errors.

## PR Workflow

1. Create a feature branch from `main`.
2. Make changes, run local verification (`fmt`, `clippy`, `test`, `doc`).
3. Push and open a PR.
4. CI must pass all checks.
5. One approval required for merge.
6. Squash-merge to keep history clean.
