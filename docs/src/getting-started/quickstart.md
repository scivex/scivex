# Quickstart Tutorial

This tutorial walks through a complete data science workflow in Scivex: loading data, exploring it, training a model, and visualizing results.

## Setup

```toml
# Cargo.toml
[dependencies]
scivex = { version = "0.1", features = ["full"] }
```

```rust
use scivex::prelude::*;
```

## 1. Tensors

Tensors are the foundation of all numerical computing in Scivex.

```rust
// Create from a vector
let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();

// Zeros, ones, ranges
let zeros = Tensor::<f64>::zeros(vec![3, 3]);
let ones = Tensor::<f64>::ones(vec![2, 4]);
let range = Tensor::arange(0.0, 10.0, 1.0); // [0, 1, 2, ..., 9]

// Element-wise operations
let b = Tensor::from_vec(vec![6.0, 5.0, 4.0, 3.0, 2.0, 1.0], vec![2, 3]).unwrap();
let sum = a.add(&b).unwrap();
let product = a.mul(&b).unwrap();

// Matrix multiplication
let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
let y = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
let z = x.matmul(&y).unwrap();
// z = [[19, 22], [43, 50]]

// Reductions
let total = a.sum();
let mean = a.mean();
```

## 2. DataFrames

DataFrames provide tabular data operations similar to Pandas.

```rust
use scivex::frame::prelude::*;

// Create from columns
let df = DataFrame::new(vec![
    Series::new("name", vec!["Alice", "Bob", "Carol", "Dave"]),
    Series::new("age", vec![30i64, 25, 35, 28]),
    Series::new("score", vec![85.0, 92.0, 78.0, 95.0]),
]).unwrap();

// Select columns
let ages = df.column("age").unwrap();

// Filter rows
let high_scorers = df.filter("score", |v: &f64| *v > 80.0).unwrap();

// Group by and aggregate
let grouped = df.groupby(&["age"]).mean();

// Sort
let sorted = df.sort_by("score", false).unwrap(); // descending
```

## 3. Statistics

```rust
use scivex::stats::prelude::*;

let data = vec![2.1, 3.4, 2.8, 3.1, 2.5, 3.8, 2.9];

// Descriptive statistics
let desc = describe(&data);

// Hypothesis testing
let (t_stat, p_value) = t_test_1sample(&data, 3.0);

// Distributions
let normal = Normal::new(0.0, 1.0);
let sample = normal.sample();
let pdf = normal.pdf(1.5);
let cdf = normal.cdf(1.96);
```

## 4. Machine Learning

```rust
use scivex::ml::prelude::*;

// Prepare data
let x_train = Tensor::from_vec(/* feature data */, vec![100, 4]).unwrap();
let y_train = Tensor::from_vec(/* labels */, vec![100]).unwrap();

// Train a random forest
let mut rf = RandomForest::new(100); // 100 trees
rf.fit(&x_train, &y_train).unwrap();

// Predict
let predictions = rf.predict(&x_test).unwrap();

// Evaluate
let accuracy = accuracy_score(&y_test, &predictions);
```

## 5. Visualization

```rust
use scivex::viz::prelude::*;

let x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
let y = vec![0.0, 1.0, 4.0, 9.0, 16.0, 25.0];

let fig = Figure::new()
    .plot(
        Axes::new()
            .title("Quadratic Growth")
            .x_label("x")
            .y_label("y = x^2")
            .add_plot(LinePlot::new(x, y).color(Color::BLUE))
    );

// Save as SVG
fig.save_svg("plot.svg").unwrap();

// Or print to terminal
fig.show_terminal().unwrap();
```

## 6. Neural Networks

```rust
use scivex::nn::prelude::*;

// Define a model
let model = Sequential::new()
    .add(Dense::new(784, 128))
    .add(ReLU::new())
    .add(Dense::new(128, 10))
    .add(Softmax::new());

// Train
let optimizer = Adam::new(model.parameters(), 0.001);
let loss_fn = CrossEntropyLoss::new();

for epoch in 0..10 {
    let output = model.forward(&x_batch);
    let loss = loss_fn.compute(&output, &y_batch);
    loss.backward();
    optimizer.step();
    optimizer.zero_grad();
}
```

## Next Steps

- [Tensor User Guide](../guide/tensors.md) — deep dive into tensor operations
- [DataFrame User Guide](../guide/dataframes.md) — full DataFrame API
- [Migration Guides](../migration/numpy.md) — coming from Python?
- [Cookbook](../cookbook/recipes.md) — end-to-end examples
