# AI Reference

This document is designed for AI coding assistants (Claude, Copilot, etc.) to generate correct Scivex code. It contains the essential patterns, types, and anti-patterns.

## Project Overview

**Scivex** is a comprehensive Rust data science library that replaces the entire Python ecosystem (NumPy, Pandas, scikit-learn, PyTorch, Matplotlib, SciPy) with a single Cargo workspace. Everything is implemented from first principles in pure Rust.

- **Edition:** Rust 2024 (MSRV 1.85)
- **License:** MIT
- **Architecture:** Workspace of 15+ crates with `scivex` as the umbrella

## Import Patterns

```rust
// Umbrella import — gives you everything enabled by feature flags
use scivex::prelude::*;

// Individual crate imports — finer control
use scivex_core::prelude::*;
use scivex_frame::prelude::*;
use scivex_stats::prelude::*;
use scivex_ml::prelude::*;
use scivex_nn::prelude::*;
use scivex_viz::prelude::*;
use scivex_optim::prelude::*;
use scivex_io::prelude::*;

// Access sub-crate via umbrella
use scivex::core;
use scivex::frame;
use scivex::ml;
```

## Feature Flags

```toml
# Everything except GPU
scivex = { version = "0.1", features = ["full"] }

# Just what you need
scivex = { version = "0.1", features = ["core", "frame", "io", "viz"] }
```

| Feature | Crate | Default |
|---------|-------|---------|
| `core` | scivex-core | Yes |
| `frame` | scivex-frame | No |
| `stats` | scivex-stats | No |
| `io` | scivex-io | No |
| `optim` | scivex-optim | No |
| `viz` | scivex-viz | No |
| `ml` | scivex-ml | No |
| `nn` | scivex-nn | No |
| `image` | scivex-image | No |
| `signal` | scivex-signal | No |
| `graph` | scivex-graph | No |
| `nlp` | scivex-nlp | No |
| `sym` | scivex-sym | No |
| `gpu` | scivex-gpu | No |

Modifiers: `simd`, `parallel`, `serde-support`

## Core Types

### Tensor

```rust
// Creation
let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
let b = Tensor::<f64>::zeros(vec![3, 3]);
let c = Tensor::<f64>::ones(vec![2, 4]);
let d = Tensor::eye(3);                           // 3x3 identity
let e = Tensor::linspace(0.0, 1.0, 100).unwrap(); // 100 points
let f = Tensor::arange(10);                        // [0..10)

// Access
let shape: &[usize] = a.shape();
let val: &f64 = a.get(&[0, 1]).unwrap();
let slice: &[f64] = a.as_slice();

// Arithmetic (element-wise, returns Result)
let sum = (&a + &b)?;
let prod = (&a * &b)?;
let scaled = (&a * 2.0)?;

// Matrix multiplication
let result = a.matmul(&b).unwrap();

// Reductions
let total: f64 = a.sum();
let avg: f64 = a.mean();

// Reshape & transform
let flat = a.flatten();
let t = a.transpose().unwrap();
let r = a.reshape(vec![4, 1]).unwrap();
```

### DataFrame

```rust
use scivex_frame::prelude::*;

// Builder pattern
let df = DataFrame::builder()
    .add_column("name", vec!["Alice", "Bob", "Carol"])
    .add_column("age", vec![30_i64, 25, 35])
    .add_column("score", vec![85.0_f64, 92.0, 78.0])
    .build()
    .unwrap();

// Access
let col = df.column("age").unwrap();
let (rows, cols) = df.shape();

// GroupBy
let grouped = df.groupby(&["age"]).unwrap();
```

### Variable (Autograd)

```rust
use scivex_nn::prelude::*;

let x = Variable::new(Tensor::from_vec(vec![2.0_f64], vec![1]).unwrap(), true);
let y = Variable::new(Tensor::from_vec(vec![3.0_f64], vec![1]).unwrap(), true);
let z = mul(&x, &y);   // z = 6.0
z.backward();           // compute gradients
// x.grad() == 3.0, y.grad() == 2.0
```

## Error Handling

All Scivex functions return `Result<T, CrateError>`. Never unwrap in library code.

```rust
// Each crate has its own error type
use scivex_core::error::{CoreError, Result};
use scivex_frame::error::{FrameError, Result};
use scivex_ml::error::{MlError, Result};
use scivex_nn::error::{NnError, Result};

// Common error variants across crates
CoreError::DimensionMismatch { expected, got }
CoreError::InvalidParameter(String)
CoreError::ShapeMismatch { expected, got }
CoreError::SingularMatrix
```

Pattern: use `?` operator, return `Result`, propagate errors.

## Common Patterns

### ML: fit / predict

```rust
use scivex_ml::prelude::*;

let mut model = RandomForestClassifier::new(100); // 100 trees
model.fit(&x_train, &y_train)?;
let predictions = model.predict(&x_test)?;

// Evaluate
let acc = scivex_ml::metrics::accuracy(&y_test, &predictions);
```

Traits:
- `Transformer`: `fit(&data)`, `transform(&data)`, `fit_transform(&data)`
- `Predictor`: `fit(&x, &y)`, `predict(&x)`
- `Classifier`: extends `Predictor` + `predict_proba(&x)`

### NN: forward / backward

```rust
use scivex_nn::prelude::*;

let model = Sequential::new()
    .add(Linear::new(784, 128))
    .add(ReLU::new())
    .add(Linear::new(128, 10));

let mut optimizer = Adam::new(0.001);

for epoch in 0..10 {
    let output = model.forward(&x_batch);
    let loss = cross_entropy_loss(&output, &y_batch);
    loss.backward();
    optimizer.step(model.parameters());
    optimizer.zero_grad(model.parameters());
}
```

### Viz: Figure / Axes

```rust
use scivex_viz::prelude::*;

let fig = Figure::new().plot(
    Axes::new()
        .title("My Plot")
        .x_label("x")
        .y_label("y")
        .add_plot(LinePlot::new(x_data, y_data).color(Color::BLUE))
        .add_plot(ScatterPlot::new(x2, y2).color(Color::RED))
);

fig.save_svg("output.svg").unwrap();
```

### Stats: distributions

```rust
use scivex_stats::prelude::*;

let normal = Normal::new(0.0, 1.0);
let pdf_val = normal.pdf(1.5);
let cdf_val = normal.cdf(1.96);
let sample = normal.sample();

let (t_stat, p_value) = t_test_one_sample(&data, 3.0);
```

### Optimization

```rust
use scivex_optim::prelude::*;

let f = |x: &Tensor<f64>| -> f64 {
    let s = x.as_slice();
    (s[0] - 1.0).powi(2) + (s[1] - 2.5).powi(2)
};
let grad = |x: &Tensor<f64>| -> Tensor<f64> {
    let s = x.as_slice();
    Tensor::from_vec(vec![2.0 * (s[0] - 1.0), 2.0 * (s[1] - 2.5)], vec![2]).unwrap()
};

let x0 = Tensor::from_vec(vec![0.0, 0.0], vec![2]).unwrap();
let opts = MinimizeOptions::default();
let result = bfgs(f, grad, &x0, &opts)?;
```

## Anti-Patterns

**Do NOT:**

1. Use `ndarray`, `nalgebra`, or any external math crate — Scivex has its own `Tensor<T>`
2. Use `.unwrap()` in library code — always propagate with `?`
3. Use `panic!()` in library code — return `Err(...)`
4. Assume a specific numeric type — use generics with `Float` or `Scalar` bounds
5. Import from internal module paths — use `prelude::*` or crate-level re-exports
6. Use `unsafe` outside `scivex-core` — all other crates use safe abstractions
7. Forget feature flags — if using `DataFrame`, need `features = ["frame"]`
8. Call `backward()` on non-tracking Variables — set `requires_grad: true`

**Do:**

1. Use `Result<T>` everywhere
2. Use the trait hierarchy: `Scalar` > `Float` > `Real`
3. Use builder patterns where provided (DataFrame, CSV reader, etc.)
4. Check `.shape()` before operations when debugging dimension mismatches
5. Use `prelude::*` for ergonomic imports
6. Use feature flags to minimize compile times

## Quick Recipes

### Load CSV and Compute Statistics

```rust
use scivex::prelude::*;

let df = read_csv_path("data.csv")?;
let ages: &dyn AnySeries = df.column("age")?;
let stats = describe(&[2.0, 4.0, 6.0, 8.0]);
```

### Train a Classifier

```rust
use scivex::prelude::*;

let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, Some(42));
let mut model = RandomForestClassifier::new(100);
model.fit(&x_train, &y_train)?;
let preds = model.predict(&x_test)?;
let acc = scivex_ml::metrics::accuracy(&y_test, &preds);
```

### Build and Train a Neural Network

```rust
use scivex::prelude::*;

let model = Sequential::new()
    .add(Linear::new(10, 64))
    .add(ReLU::new())
    .add(Dropout::new(0.2))
    .add(Linear::new(64, 3));

let mut opt = Adam::new(0.001);
let dataset = TensorDataset::new(x_data, y_data);
let loader = DataLoader::new(dataset, 32, true);
```

### Create a Multi-Panel Plot

```rust
use scivex::prelude::*;

let fig = Figure::new()
    .layout(Layout::grid(1, 2))
    .plot(Axes::new().title("Line").add_plot(LinePlot::new(x.clone(), y1)))
    .plot(Axes::new().title("Scatter").add_plot(ScatterPlot::new(x, y2)));

fig.save_svg("panels.svg")?;
```

### Symbolic Differentiation

```rust
use scivex_sym::prelude::*;

let x = var("x");
let f = sin(x.clone()) * exp(x.clone()); // f(x) = sin(x) * e^x
let df = diff(&f, "x");                   // f'(x) = cos(x)*e^x + sin(x)*e^x
let simplified = simplify(&df);
```
