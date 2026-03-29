# Scivex — Complete Use Case Guide

> **One crate. Every data science workflow. Pure Rust.**
>
> This document walks through every capability of Scivex — from basic tensor
> manipulation to training neural networks, building ML pipelines, and deploying
> models. Each section includes working code examples.

---

## Table of Contents

1. [Getting Started](#1-getting-started)
2. [Tensor Operations (NumPy equivalent)](#2-tensor-operations)
3. [DataFrames (Pandas/Polars equivalent)](#3-dataframes)
4. [Data I/O — Files, Databases, Cloud](#4-data-io)
5. [Statistics & Distributions](#5-statistics--distributions)
6. [Time Series Analysis](#6-time-series-analysis)
7. [Bayesian Inference & MCMC](#7-bayesian-inference--mcmc)
8. [Optimization & Solvers](#8-optimization--solvers)
9. [Signal Processing & Audio](#9-signal-processing--audio)
10. [Image Processing & Computer Vision](#10-image-processing--computer-vision)
11. [Natural Language Processing](#11-natural-language-processing)
12. [Graph & Network Analysis](#12-graph--network-analysis)
13. [Symbolic Mathematics](#13-symbolic-mathematics)
14. [Classical Machine Learning](#14-classical-machine-learning)
15. [Neural Networks & Deep Learning](#15-neural-networks--deep-learning)
16. [Reinforcement Learning](#16-reinforcement-learning)
17. [GPU Acceleration](#17-gpu-acceleration)
18. [Visualization & Plotting](#18-visualization--plotting)
19. [ML Pipelines & AutoML](#19-ml-pipelines--automl)
20. [Model Deployment & Interop](#20-model-deployment--interop)
21. [What Scivex Does NOT Do](#21-what-scivex-does-not-do)

---

## 1. Getting Started

### Installation

```toml
# Cargo.toml — full stack
[dependencies]
scivex = { version = "0.1", features = ["full"] }
```

Or pick only what you need:

```toml
[dependencies]
scivex = { version = "0.1", features = ["core", "frame", "io", "stats", "ml"] }
```

### Feature Flags

| Feature | What you get |
|---------|-------------|
| `core` | Tensors, linear algebra, FFT, PRNG, sparse matrices |
| `frame` | DataFrames, Series, GroupBy, joins, pivot, SQL |
| `io` | CSV, JSON, Parquet, Arrow, Excel, Avro, ORC, HDF5, NumPy, database connectors |
| `stats` | Distributions, hypothesis tests, regression, time series, Bayesian |
| `optim` | Minimization, root finding, integration, ODE/PDE, interpolation, LP/QP |
| `viz` | Line/scatter/bar/histogram/heatmap/box/violin/pie/contour plots, SVG/HTML/terminal |
| `ml` | Linear models, trees, random forests, SVM, k-NN, K-Means, Naive Bayes, pipelines |
| `nn` | Autograd, Linear/Conv2d/BatchNorm/Dropout/LSTM layers, SGD/Adam, ONNX runtime |
| `image` | Image types, transforms, filters, feature detection, augmentation, segmentation |
| `signal` | FIR/IIR filters, STFT, spectrograms, wavelets, MFCC, beat tracking |
| `graph` | Directed/undirected graphs, Dijkstra, BFS/DFS, PageRank, MST, max flow |
| `nlp` | Tokenizers, stemming, TF-IDF, Word2Vec, sentiment, POS tagging, NER, LDA |
| `sym` | Symbolic expressions, differentiation, integration, equation solving, polynomials |
| `rl` | Environments, DQN, PPO, A2C, SAC, TD3, replay buffers, HER |
| `gpu` | GPU tensors via wgpu (Vulkan/Metal/DX12), optional CUDA |
| `full` | Everything above |

### Hello World

```rust
use scivex::prelude::*;

fn main() -> scivex::core::Result<()> {
    // Create a 2×3 tensor
    let a = Tensor::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?;
    println!("Tensor: {:?}", a.shape()); // [2, 3]
    println!("Sum: {}", a.sum());        // 21.0

    // Quick stats
    let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
    let m = scivex::stats::descriptive::mean(&data)?;
    let s = scivex::stats::descriptive::std_dev(&data)?;
    println!("mean={m:.2}, std={s:.2}"); // mean=5.00, std=2.00

    Ok(())
}
```

---

## 2. Tensor Operations

> **Replaces:** NumPy, ndarray, nalgebra

### Creating Tensors

```rust
use scivex::core::prelude::*;

// From data
let a = Tensor::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
let b = Tensor::<f64>::from_slice(&[5.0, 6.0, 7.0, 8.0], vec![2, 2])?;

// Factory constructors
let zeros = Tensor::<f64>::zeros(vec![3, 4]);
let ones  = Tensor::<f64>::ones(vec![3, 4]);
let eye   = Tensor::<f64>::eye(3);                        // 3×3 identity
let range = Tensor::<f64>::arange(10);                    // [0, 1, ..., 9]
let lin   = Tensor::<f64>::linspace(0.0, 1.0, 100)?;     // 100 points in [0, 1]
let full  = Tensor::<f64>::full(vec![2, 3], 42.0);        // all 42s

// Random tensors
let mut rng = Rng::new(42);
let u = scivex::core::random::uniform::<f64>(&mut rng, vec![3, 3]);
let n = scivex::core::random::normal::<f64>(&mut rng, vec![3, 3], 0.0, 1.0);
let r = scivex::core::random::randint::<i64>(&mut rng, vec![5], 0, 10)?;
```

### Element-wise Arithmetic

```rust
let c = &a + &b;         // element-wise add
let d = &a * &b;         // element-wise multiply
let e = &a - &b;         // subtract
let f = &a / &b;         // divide
let g = -&a;             // negate
let h = &a + 2.0;        // scalar broadcast

// Safe (checked) variants
let c = a.add_checked(&b)?;
let d = a.mul_checked(&b)?;

// SIMD-accelerated (feature = "simd")
let c = a.add_simd(&b)?;
let d = a.mul_simd(&b)?;
```

### Reductions

```rust
let total = a.sum();              // sum all elements
let prod  = a.product();          // product of all
let mn    = a.min_element();      // Option<T>
let mx    = a.max_element();      // Option<T>
let avg   = a.mean();             // mean (Float types)
let col_sums = a.sum_axis(0)?;    // sum along axis 0
```

### Reshaping and Indexing

```rust
// Reshape
let flat = a.clone().flatten();          // 1-D
let r = a.clone().reshape(vec![1, 4])?;  // reshape

// Transpose and permute
let t = a.transpose()?;                  // 2-D transpose
let p = a.permute(&[1, 0])?;            // N-D axis permutation

// Indexing
let val = a.get(&[0, 1])?;              // single element
a.set(&[0, 1], 99.0)?;                  // set element

// Slicing
use scivex::core::tensor::SliceRange;
let row0 = a.slice(&[SliceRange::range(0, 1), SliceRange::full()])?;
let sub  = a.select(0, 0)?;             // select row 0

// Advanced indexing
let idx = a.index_select(1, &[0, 2])?;  // select columns 0, 2
let mask = a.masked_select(&[true, false, true, true])?;

// Concatenation and stacking
let cat = Tensor::concat(&[&a, &b], 0)?;  // along axis 0
let stk = Tensor::stack(&[&a, &b], 0)?;   // new axis at 0

// Squeeze / unsqueeze
let u = a.clone().unsqueeze(0)?;   // add dim at position 0
let s = u.squeeze();               // remove all size-1 dims
```

### Linear Algebra

```rust
use scivex::core::linalg::*;

// Matrix multiply
let c = a.matmul(&b)?;

// BLAS Level 1
let d = dot(&x, &y)?;            // dot product
axpy(2.0, &x, &mut y)?;          // y += 2·x
let n = nrm2(&x)?;               // L2 norm

// BLAS Level 2/3
gemv(1.0, &a, &x, 0.0, &mut y)?;    // y = A·x
gemm(1.0, &a, &b, 0.0, &mut c)?;    // C = A·B (NEON micro-kernel on Apple Silicon)

// Solve Ax = b
let x = solve(&a, &b)?;
let x = a.solve(&b)?;

// Inverse and determinant
let inv_a = inv(&a)?;
let det_a = det(&a)?;

// Least squares
let x = lstsq(&a, &b)?;

// Decompositions
let lu  = LuDecomposition::decompose(&a)?;
let qr  = QrDecomposition::decompose(&a)?;
let svd = SvdDecomposition::decompose(&a)?;
let eig = EigDecomposition::decompose_symmetric(&a)?;
let cho = CholeskyDecomposition::decompose(&a)?;

// Use decompositions
let x = lu.solve(&b)?;
let eigenvalues = eig.eigenvalues();       // sorted by |λ|
let singular_values = svd.singular_values();
let rank = svd.rank(1e-10);
let cond = svd.condition_number();
let log_det = cho.log_det();
```

### Einstein Summation

```rust
use scivex::core::tensor::einsum;

let c = einsum("ij,jk->ik", &[&a, &b])?;   // matmul
let t = einsum("ii->", &[&a])?;             // trace
let o = einsum("i,j->ij", &[&x, &y])?;     // outer product
let d = einsum("i,i->", &[&x, &y])?;        // dot product
```

### Sparse Matrices

```rust
use scivex::core::linalg::{CooMatrix, CsrMatrix, CscMatrix};

// Build sparse from triplets
let coo = CooMatrix::from_triplets(3, 3,
    &[0, 1, 2], &[0, 1, 2], &[1.0, 2.0, 3.0])?;

// Convert formats
let csr = coo.to_csr();
let csc = coo.to_csc();
let dense = csr.to_dense()?;

// Sparse matrix-vector multiply
let y = csr.matvec(&x)?;

// N-D sparse tensors
use scivex::core::SparseTensor;
let sp = SparseTensor::from_dense(&dense_tensor);
let y = sp.sparse_dense_matmul(&dense_mat)?;
```

### Tensor Decompositions (CP, Tucker, NTF)

```rust
use scivex::core::linalg::tensor_decomp::*;

// CP decomposition (PARAFAC)
let cp = CpDecomposition::decompose(&tensor_3d, 5, 100, 1e-6)?;
let reconstructed = cp.reconstruct()?;

// Tucker decomposition (HOSVD)
let tucker = TuckerDecomposition::decompose(&tensor_3d, &[3, 3, 3])?;
let core = tucker.core();
let factors = tucker.factors();

// Non-negative tensor factorization
let ntf = NtfDecomposition::decompose(&nonneg_tensor, 3, 200, 1e-6)?;
```

### FFT

```rust
use scivex::core::fft;

let spectrum = fft::rfft(&signal)?;       // real-to-complex FFT
let recovered = fft::irfft(&spectrum, n)?; // inverse
let spec_2d = fft::rfft2(&image)?;        // 2-D real FFT
```

### JIT Expression Compiler

```rust
use scivex::core::jit::Expr;

// Fused evaluation — no intermediate tensor allocations
let a = Tensor::<f64>::from_vec(vec![1.0, 2.0, 3.0], vec![3])?;
let b = Tensor::<f64>::from_vec(vec![4.0, 5.0, 6.0], vec![3])?;

let result = Expr::input(&a)
    .mul(Expr::scalar(2.0))
    .add(Expr::input(&b))
    .sqrt()
    .eval()?;
// Computes sqrt(2*a + b) in a single fused pass
```

---

## 3. DataFrames

> **Replaces:** Pandas, Polars

### Creating DataFrames

```rust
use scivex::frame::prelude::*;

// Builder pattern
let df = DataFrame::builder()
    .add_column("name", vec!["Alice", "Bob", "Carol", "Dave"])
    .add_column("age", vec![30_i32, 25, 35, 28])
    .add_column("salary", vec![95000.0_f64, 87500.0, 92000.0, 78000.0])
    .add_column("department", vec!["Engineering", "Sales", "Engineering", "Sales"])
    .build()?;

println!("Shape: {:?}", df.shape());      // (4, 4)
println!("Columns: {:?}", df.column_names()); // ["name", "age", "salary", "department"]
```

### Column Access and Types

```rust
// Type-erased access
let col = df.column("age")?;

// Typed access
let ages: &Series<i32> = df.column_typed::<i32>("age")?;
let mean_age = ages.mean();   // 29.5

// String column
let names = StringSeries::from_strs("name", &["Alice", "Bob"]);
let upper = names.to_uppercase();           // ["ALICE", "BOB"]
let has_a = names.contains("a");            // [false, false] (case-sensitive)
let lengths = names.len_chars();            // [5, 3]

// Categorical column
let cat = CategoricalSeries::from_strs("dept", &["eng", "sales", "eng", "hr"]);
println!("Categories: {:?}", cat.categories()); // ["eng", "hr", "sales"]
println!("N unique: {}", cat.n_categories());   // 3

// DateTime column
use scivex::frame::series::datetime::DateTime;
let dt = DateTime::from_ymd(2026, 3, 25)?;
println!("Year: {}, Month: {}", dt.year(), dt.month());
```

### Filtering and Selection

```rust
// Select columns
let subset = df.select(&["name", "salary"])?;

// Drop columns
let reduced = df.drop_columns(&["department"])?;

// Head / tail / slice
let first2 = df.head(2);
let last2 = df.tail(2);
let mid = df.slice(1, 2);   // offset=1, length=2

// Boolean mask filter
let mask: Vec<bool> = df.column_typed::<i32>("age")?
    .as_slice()
    .iter()
    .map(|&a| a > 28)
    .collect();
let adults = df.filter(&mask)?;

// Sort
let sorted = df.sort_by("salary", false)?;   // descending
```

### Null Handling

```rust
// Create series with nulls
let s = Series::with_nulls("x", vec![1.0, 0.0, 3.0, 0.0], vec![false, true, false, true])?;

// Fill strategies
let filled = s.fill_null(0.0);          // replace nulls with 0
let ffill = s.fill_forward();           // propagate last value
let bfill = s.fill_backward();          // propagate next value
let interp = s.interpolate();           // linear interpolation

// Drop nulls
let clean = s.drop_null();
let clean_df = df.drop_nulls()?;
let clean_sub = df.drop_nulls_subset(&["salary"])?;
```

### GroupBy and Aggregation

```rust
let grouped = df.groupby(&["department"])?;

let sums = grouped.sum()?;     // sum all numeric columns per group
let means = grouped.mean()?;   // mean per group
let counts = grouped.count()?; // row count per group

// Specific column aggregation
let max_salary = grouped.agg("salary", AggFunc::Max)?;
```

### Joins

```rust
let orders = DataFrame::builder()
    .add_column("order_id", vec![1_i32, 2, 3])
    .add_column("user_id", vec![1_i32, 2, 1])
    .add_column("amount", vec![100.0_f64, 250.0, 75.0])
    .build()?;

let users = DataFrame::builder()
    .add_column("user_id", vec![1_i32, 2, 3])
    .add_column("name", vec!["Alice", "Bob", "Carol"])
    .build()?;

// Join types: Inner, Left, Right, Outer
let joined = orders.join(&users, &["user_id"], JoinType::Inner)?;

// Different key names
let joined = orders.join_on(&users, &["user_id"], &["user_id"], JoinType::Left)?;
```

### Pivot Tables and Reshaping

```rust
// Pivot: long → wide
let pivoted = df.pivot(
    &["department"],   // index (row groups)
    "quarter",         // column values become column names
    "revenue",         // values to aggregate
    AggFunc::Sum,
)?;

// Melt: wide → long
let melted = df.melt(
    &["name"],                    // id columns
    &["q1_score", "q2_score"],    // value columns to unpivot
    Some("quarter"),              // variable name
    Some("score"),                // value name
)?;

// Cross-tabulation
let xtab = df.crosstab("department", "status")?;
```

### Rolling Windows and Expanding

```rust
use scivex::frame::series::window::RollingWindow;

let prices = Series::new("price", vec![10.0, 11.0, 12.0, 11.5, 13.0, 14.0, 13.5]);

let window = RollingWindow::new(3).min_periods(1);

let ma = prices.rolling_mean(&window)?;     // 3-period moving average
let rs = prices.rolling_std(&window)?;      // rolling std dev
let rmin = prices.rolling_min(&window)?;
let rmax = prices.rolling_max(&window)?;

// Expanding (cumulative)
let cum_sum = prices.expanding_sum();
let cum_mean = prices.expanding_mean();

// Exponentially weighted
let ewma = prices.ewm_mean(0.3)?;
```

### Lazy API (Deferred Execution)

```rust
use scivex::frame::lazy::{col, lit_f64};

let result = df.lazy()
    .filter(col("salary").gt(lit_f64(85000.0)))
    .select(&[col("name"), col("salary")])
    .sort("salary", false)
    .limit(10)
    .collect()?;
```

### SQL Queries

```rust
use scivex::frame::sql::{SqlContext, sql};

// Single-table convenience
let result = sql(&df, "SELECT name, salary FROM t WHERE age > 28 ORDER BY salary DESC")?;

// Multi-table context
let mut ctx = SqlContext::new();
ctx.register("employees", employees_df);
ctx.register("departments", departments_df);

let result = ctx.execute(
    "SELECT e.name, d.name AS dept
     FROM employees e
     INNER JOIN departments d ON e.dept_id = d.id
     WHERE e.salary > 80000
     ORDER BY e.salary DESC
     LIMIT 10"
)?;

// Supported SQL: SELECT, FROM, WHERE, JOIN (INNER/LEFT/RIGHT),
// GROUP BY with SUM/AVG/MIN/MAX/COUNT, ORDER BY, LIMIT, AS aliases
```

### Parallel Operations

```rust
use scivex::frame::parallel::*;

// Parallel filter
let filtered = par_filter(&df, "salary", |s| s > 80000.0)?;

// Parallel groupby
let grouped = par_groupby_agg(&df, &["department"], AggFunc::Mean)?;

// Parallel sort
let sorted = par_sort(&df, "salary", true)?;

// Parallel apply
let doubled: Vec<f64> = par_apply(&df, "salary", |s| s * 2.0)?;
```

---

## 4. Data I/O

> **Replaces:** Pandas I/O, SQLAlchemy, fsspec

### CSV

```rust
use scivex::io::csv::*;

// Read with defaults (auto type inference)
let df = read_csv_path("data.csv")?;

// Customized reader
let df = CsvReaderBuilder::new()
    .delimiter(b';')
    .has_header(true)
    .skip_rows(1)
    .max_rows(Some(10000))
    .null_values(vec!["NA".into(), "null".into()])
    .comment_char(Some(b'#'))
    .trim_whitespace(true)
    .read_path("data.csv")?;

// Write
CsvWriterBuilder::new()
    .delimiter(b',')
    .write_header(true)
    .write_path("output.csv", &df)?;
```

### JSON

```rust
use scivex::io::json::*;

// Read records format: [{"col": val}, ...]
let df = read_json_path("data.json")?;

// Read columns format: {"col": [val, ...]}
let df = JsonReaderBuilder::new()
    .orientation(JsonOrientation::Columns)
    .read_path("data.json")?;

// Write pretty JSON
JsonWriterBuilder::new()
    .pretty(true)
    .orientation(JsonOrientation::Records)
    .write_path("output.json", &df)?;
```

### SQL Databases

```rust
// PostgreSQL (feature = "postgres")
use scivex::io::postgres::PostgresConnection;

let conn = PostgresConnection::new("host=localhost dbname=mydb user=postgres")?;
let df = conn.query("SELECT * FROM users WHERE active = true")?;
conn.write_table("results", &df, IfExists::Replace)?;

// MySQL (feature = "mysql")
use scivex::io::mysql::MysqlConnection;

let conn = MysqlConnection::new("mysql://user:pass@localhost/mydb")?;
let df = conn.query("SELECT * FROM orders")?;

// SQL Server / MSSQL (feature = "mssql")
use scivex::io::mssql::{MssqlConfig, MssqlConnection};

let config = MssqlConfig::new("server", 1433, "user", "pass", "mydb");
let conn = MssqlConnection::new(config)?;
let df = conn.query("SELECT TOP 100 * FROM transactions")?;

// SQLite (feature = "sqlite")
use scivex::io::sqlite::SqliteConnection;

let conn = SqliteConnection::open("local.db")?;
let df = conn.query("SELECT * FROM events")?;
conn.write_table("summary", &df, IfExists::Append)?;

// DuckDB (feature = "duckdb") — embedded OLAP
use scivex::io::duckdb::DuckDbConnection;

let conn = DuckDbConnection::open("analytics.duckdb")?;
let df = conn.query("SELECT region, SUM(revenue) FROM sales GROUP BY region")?;
```

### Parquet, Arrow, and Other Formats

```rust
// Parquet (feature = "parquet")
let df = scivex::io::parquet::read_parquet("data.parquet")?;
scivex::io::parquet::write_parquet("output.parquet", &df)?;

// Arrow IPC (feature = "arrow")
let df = scivex::io::arrow::read_arrow("data.arrow")?;
scivex::io::arrow::write_arrow("output.arrow", &df)?;

// NumPy .npy/.npz (feature = "npy")
let tensor = scivex::io::npy::read_npy_path("weights.npy")?;
scivex::io::npy::write_npz_path("archive.npz", &[("x", &x), ("y", &y)])?;

// Excel (feature = "excel")
let df = scivex::io::excel::read_excel("report.xlsx")?;

// Avro, ORC, HDF5 (respective features)
let df = scivex::io::avro::read_avro("data.avro")?;
let df = scivex::io::orc::read_orc("data.orc")?;
let tensor = scivex::io::hdf5::read_hdf5_dataset("model.h5", "layer0/weights")?;

// Delta Lake (feature = "delta")
let df = scivex::io::delta::read_delta("s3://bucket/delta-table/")?;

// Memory-mapped tensors (feature = "mmap") — zero-copy
let tensor = scivex::io::mmap::mmap_npy("huge_matrix.npy")?;
```

### Database → DataFrame → ML Pipeline

```rust
// Complete workflow: query database, analyze, train model
let conn = PostgresConnection::new("host=db.example.com dbname=analytics user=etl")?;

// Pull training data
let df = conn.query("
    SELECT age, income, credit_score, loan_amount, defaulted
    FROM loan_applications
    WHERE application_date >= '2024-01-01'
")?;

// Convert to tensors for ML
let features = Tensor::<f64>::from_vec(/* extract numeric columns */, vec![n, 4])?;
let labels = Tensor::<f64>::from_vec(/* extract target column */, vec![n])?;

// Train model (see ML section)
let mut rf = RandomForestClassifier::new(100, 5);
rf.fit(&features, &labels)?;
```

---

## 5. Statistics & Distributions

> **Replaces:** SciPy.stats, statsmodels

### Descriptive Statistics

```rust
use scivex::stats::descriptive::*;

let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];

let m = mean(&data)?;                  // 5.0
let s = std_dev(&data)?;               // ~2.0
let v = variance(&data)?;              // ~4.57
let med = median(&data)?;              // 4.5
let q75 = quantile(&data, 0.75)?;      // ~6.5
let sk = skewness(&data)?;             // skewness
let ku = kurtosis(&data)?;             // excess kurtosis

// Full summary
let desc = describe(&data)?;
println!("count={}, mean={:.2}, std={:.2}, min={:.2}, max={:.2}",
    desc.count, desc.mean, desc.std_dev, desc.min, desc.max);
```

### Probability Distributions

```rust
use scivex::stats::distributions::*;

// Normal distribution
let normal = Normal::new(0.0, 1.0);
let p = normal.pdf(0.0);               // probability density
let c = normal.cdf(1.96);              // ≈ 0.975
let z = normal.ppf(0.975)?;            // ≈ 1.96 (inverse CDF)
let samples = normal.sample_n(1000, &mut rng);

// Student's t
let t = StudentT::new(10.0);           // 10 degrees of freedom
let p_value = 1.0 - t.cdf(2.228);

// Other distributions
let gamma = Gamma::new(2.0, 1.0);
let beta = Beta::new(2.0, 5.0);
let poisson = Poisson::new(3.5);
let binom = Binomial::new(100, 0.3);
let exp = Exponential::new(0.5);
let chi2 = ChiSquared::new(5.0);
let weibull = Weibull::new(1.5, 1.0);
let pareto = Pareto::new(2.0, 1.0);
```

### Hypothesis Tests

```rust
use scivex::stats::hypothesis::*;

// One-sample t-test: is the mean different from 5.0?
let result = t_test_one_sample(&data, 5.0)?;
println!("t={:.3}, p={:.4}", result.statistic, result.p_value);

// Two-sample t-test (Welch's)
let result = t_test_two_sample(&group_a, &group_b)?;

// Chi-square test
let observed = vec![16.0, 18.0, 16.0, 14.0, 12.0, 12.0];
let expected = vec![16.0, 16.0, 16.0, 16.0, 16.0, 8.0];
let result = chi_square_test(&observed, &expected)?;

// ANOVA
let result = anova_oneway(&[&group1, &group2, &group3])?;

// Kolmogorov-Smirnov test
let result = ks_test_two_sample(&sample1, &sample2)?;

// Mann-Whitney U test (non-parametric)
let result = mann_whitney_u(&x, &y)?;
```

### Correlation

```rust
use scivex::stats::correlation::*;

let r = pearson(&x, &y)?;              // Pearson correlation
let rho = spearman(&x, &y)?;           // Spearman rank correlation
let tau = kendall(&x, &y)?;            // Kendall's tau-b

// Correlation matrix
let corr = corr_matrix(&data_matrix, CorrelationMethod::Pearson)?;
```

### Regression

```rust
use scivex::stats::regression::*;

// OLS regression: y = β₀ + β₁x₁ + β₂x₂ + ε
let result = ols(&x_matrix, &y)?;

println!("Coefficients: {:?}", result.coefficients);
println!("R²: {:.4}", result.r_squared);
println!("Adjusted R²: {:.4}", result.r_squared_adj);
println!("F-statistic: {:.2}, p={:.4}", result.f_statistic, result.f_p_value);
println!("Std errors: {:?}", result.std_errors);
println!("P-values: {:?}", result.p_values);

// Generalized Linear Models
use scivex::stats::glm::*;

// Logistic regression
let result = glm(&x, &y, Family::Binomial)?;
println!("AIC: {:.2}", result.aic);
println!("Deviance: {:.2}", result.deviance);

// Poisson regression (count data)
let result = glm(&x, &counts, Family::Poisson)?;
```

### Confidence Intervals

```rust
use scivex::stats::confidence::*;

let ci = ci_mean(&data, 0.95)?;
println!("95% CI: [{:.2}, {:.2}]", ci.lower, ci.upper);

let ci = ci_proportion(45, 100, 0.99)?;   // 45 successes out of 100
```

### Effect Sizes

```rust
use scivex::stats::effect_size::*;

let d = cohens_d(&treatment, &control)?;
let interpretation = interpret_cohens_d(d);   // Small/Medium/Large
let g = hedges_g(&treatment, &control)?;      // bias-corrected
```

### Multiple Comparison Corrections

```rust
use scivex::stats::correction::*;

let p_values = vec![0.01, 0.04, 0.03, 0.20, 0.001];
let significant_bonf = bonferroni(&p_values, 0.05);        // conservative
let significant_bh = benjamini_hochberg(&p_values, 0.05);   // FDR control
```

### Survival Analysis

```rust
use scivex::stats::survival::*;

// Kaplan-Meier survival curve
let km = kaplan_meier(&times, &events)?;
let median = median_survival_time(&km);

// Log-rank test
let result = log_rank_test(&group1_records, &group2_records)?;

// Cox proportional hazards
let result = cox_ph(&covariates, &times, &events)?;
println!("Hazard ratios: {:?}", result.hazard_ratios);
println!("Concordance: {:.3}", result.concordance);
```

### Mixed Effects Models

```rust
use scivex::stats::mixed_effects::*;

let result = lmm(&fixed_x, &random_x, &y, &groups)?;
println!("Fixed effects: {:?}", result.fixed_effects);
println!("Random effects: {:?}", result.random_effects);
```

---

## 6. Time Series Analysis

> **Replaces:** statsmodels.tsa, Prophet, tsfresh

### Autocorrelation

```rust
use scivex::stats::timeseries::*;

let ac = acf(&series, 20)?;      // autocorrelation up to lag 20
let pac = pacf(&series, 20)?;    // partial autocorrelation
```

### Stationarity Tests

```rust
let adf = adf_test(&series)?;
println!("ADF statistic: {:.4}, p-value: {:.4}", adf.statistic, adf.p_value);
// p < 0.05 → series is stationary
```

### Seasonal Decomposition

```rust
let decomp = seasonal_decompose(&series, 12)?;  // period=12 (monthly data)
// decomp.trend, decomp.seasonal, decomp.residual
```

### ARIMA Forecasting

```rust
// ARIMA(p, d, q)
let mut model = Arima::new(2, 1, 1);   // AR(2), difference once, MA(1)
model.fit(&training_data)?;
let forecast = model.predict(30)?;     // 30-step forecast

// Seasonal ARIMA
let mut model = Sarimax::new(1, 1, 1, 1, 1, 1, 12);  // (p,d,q)(P,D,Q,s)
model.fit(&data)?;
let forecast = model.predict(12)?;
```

### Exponential Smoothing (Holt-Winters)

```rust
let mut es = ExponentialSmoothing::new(SmoothingMethod::Triple);
es.fit(&data, 12)?;  // seasonal period = 12
let forecast = es.predict(24)?;
```

### Prophet-style Forecasting

```rust
use scivex::stats::prophet::*;

let config = ProphetConfig {
    n_changepoints: 25,
    changepoint_prior_scale: 0.05,
    seasonality_prior_scale: 10.0,
    ..Default::default()
};

let mut model = Prophet::new(config);
model.fit(&timestamps, &values)?;
let forecast = model.predict(&future_timestamps)?;
// forecast.yhat, forecast.yhat_lower, forecast.yhat_upper, forecast.trend, forecast.seasonal
```

### Anomaly Detection

```rust
use scivex::stats::ts_anomaly::*;

let anomalies = zscore_anomaly(&series, 3.0);           // Z-score > 3
let anomalies = seasonal_anomaly(&series, 12, 2.5);     // seasonal residual outliers
let anomalies = ewma_anomaly(&series, 0.3, 3.0);        // EWMA-based
let anomalies = isolation_forest_anomaly(&series, 100, 0.05); // 5% contamination
```

### Feature Extraction

```rust
use scivex::stats::ts_features::*;

let features = extract_default_features(&series)?;
// Returns HashMap: mean, std, skewness, kurtosis, autocorr_lag1, trend, entropy, ...
```

### VAR and Granger Causality

```rust
use scivex::stats::var::*;

let mut model = VarModel::new();
model.fit(&multivariate_data, 3)?;   // 3 lags
let forecast = model.forecast(10)?;  // 10-step ahead

let result = granger_test(&x, &y, 5)?;
println!("F={:.2}, p={:.4} at lag {}", result.f_statistic, result.p_value, result.lag);
```

### Kalman Filter

```rust
use scivex::stats::kalman::*;

let mut kf = KalmanFilter::new(4, 2);  // 4 state dims, 2 measurement dims
// Set transition, observation, noise matrices...
kf.predict();
kf.update(&measurement);
let state = kf.state();
```

### GARCH Volatility Modeling

```rust
use scivex::stats::garch::Garch;

let mut model = Garch::new(1, 1);   // GARCH(1,1)
model.fit(&returns)?;
let vol_forecast = model.predict(10)?;
```

---

## 7. Bayesian Inference & MCMC

> **Replaces:** PyMC, Stan (sampling), emcee

### MCMC Sampling

```rust
use scivex::stats::bayesian::*;

// Define log-posterior
let log_prob = |params: &[f64]| -> f64 {
    let mu = params[0];
    let sigma = params[1];
    if sigma <= 0.0 { return f64::NEG_INFINITY; }
    // Prior: mu ~ N(0, 10), sigma ~ HalfNormal(5)
    let log_prior = -0.5 * (mu / 10.0).powi(2) - (sigma / 5.0).powi(2);
    // Likelihood: data ~ N(mu, sigma)
    let log_lik: f64 = data.iter()
        .map(|&x| -0.5 * ((x - mu) / sigma).powi(2) - sigma.ln())
        .sum();
    log_prior + log_lik
};

// Metropolis-Hastings
let config = McmcConfig::new(10000, 2000, 42, 1);  // samples, warmup, seed, thin
let mut mh = MetropolisHastings::new(vec![0.5, 0.1]);  // proposal stds
let result = mh.sample(log_prob, &[0.0, 1.0], &config)?;

// Hamiltonian Monte Carlo (requires gradient)
let mut hmc = HamiltonianMC::new(0.01, 10);  // step_size, n_leapfrog
let result = hmc.sample(log_prob, grad_log_prob, &[0.0, 1.0], &config)?;

// NUTS (No-U-Turn Sampler)
let mut nuts = Nuts::new(0.01);
let result = nuts.sample(log_prob, grad_log_prob, &[0.0, 1.0], &config)?;

// Diagnostics
let ess = effective_sample_size(&result.samples)?;
let r_hat = rhat(&result.samples)?;
let summary = trace_summary(&result);
println!("Acceptance rate: {:.2}", result.acceptance_rate[0]);
```

### Variational Inference

```rust
use scivex::stats::bayesian::*;

let config = ViConfig { n_iter: 1000, learning_rate: 0.01, seed: 42, ..Default::default() };
let result = mean_field_vi(log_prob, n_params, &config)?;
// result.mean, result.std — approximate posterior
```

### Bayesian Optimization

```rust
use scivex::stats::bayesian_optim::*;

// Optimize an expensive black-box function
let config = BayesOptConfig {
    n_init: 10,
    n_iter: 50,
    kernel: Kernel::Matern { nu: 2.5, length_scale: 1.0 },
    acquisition: AcquisitionFunction::ExpectedImprovement,
    seed: 42,
};

let optimizer = BayesianOptimizer::new(config);
let result = optimizer.optimize(
    |x| expensive_simulation(x),    // objective function
    &[(0.0, 10.0), (0.0, 5.0)],     // bounds for each parameter
    50,                              // n iterations
)?;

println!("Best params: {:?}, f(x) = {:.4}", result.x_best, result.f_best);
```

---

## 8. Optimization & Solvers

> **Replaces:** SciPy.optimize, cvxpy (basic LP/QP)

### Unconstrained Minimization

```rust
use scivex::optim::minimize::*;

// Rosenbrock function
let f = |x: &Tensor<f64>| {
    let a = x.get(&[0]).unwrap();
    let b = x.get(&[1]).unwrap();
    (1.0 - a).powi(2) + 100.0 * (b - a * a).powi(2)
};
let grad = |x: &Tensor<f64>| {
    let a = x.get(&[0]).unwrap();
    let b = x.get(&[1]).unwrap();
    Tensor::from_vec(vec![
        -2.0 * (1.0 - a) - 400.0 * a * (b - a * a),
        200.0 * (b - a * a),
    ], vec![2]).unwrap()
};

let x0 = Tensor::from_vec(vec![-1.0, 1.0], vec![2])?;
let opts = MinimizeOptions::default();

// BFGS (quasi-Newton)
let result = bfgs(f, grad, &x0, &opts)?;
println!("Minimum at {:?}, f={:.6}", result.x.as_slice(), result.f_val);

// Nelder-Mead (derivative-free)
let result = nelder_mead(f, &x0, &opts)?;
```

### Bounded Optimization

```rust
// L-BFGS-B: BFGS with box constraints
let bounds = vec![
    Bounds { lower: Some(0.0), upper: Some(10.0) },  // 0 ≤ x₀ ≤ 10
    Bounds { lower: Some(-5.0), upper: None },        // x₁ ≥ -5
];

let result = lbfgsb(f, grad, &x0, &bounds, &opts)?;
```

### Root Finding

```rust
use scivex::optim::roots::*;

let f = |x: f64| x * x - 2.0;   // find √2

let result = bisection(f, 1.0, 2.0, &RootOptions::default())?;
let result = brent_root(f, 1.0, 2.0, &RootOptions::default())?;
let result = newton(f, |x| 2.0 * x, 1.5, &RootOptions::default())?;

println!("Root: {:.10}", result.root);   // 1.4142135624
```

### Numerical Integration

```rust
use scivex::optim::integrate::*;

let f = |x: f64| x.sin();

// Simple quadrature
let result = trapezoid(f, 0.0, std::f64::consts::PI, 1000)?;
let result = simpson(f, 0.0, std::f64::consts::PI, 100)?;

// Adaptive Gauss-Kronrod (high accuracy)
let result = quad(f, 0.0, std::f64::consts::PI, &QuadOptions::default())?;
println!("∫sin(x)dx = {:.10} ± {:.2e}", result.value, result.error_estimate);
// ≈ 2.0
```

### Interpolation

```rust
use scivex::optim::interpolate::*;

let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
let y = vec![0.0, 1.0, 4.0, 9.0, 16.0];

// Linear interpolation
let val = interp1d(&x, &y, 2.5, Interp1dMethod::Linear)?;

// Cubic spline
let spline = CubicSpline::new(&x, &y, SplineBoundary::Natural)?;
let val = spline.eval(2.5)?;

// 2-D interpolation
let val = interp2d(&xg, &yg, &z_grid, 1.5, 2.5, Interp2dMethod::Bicubic)?;
```

### ODE Solvers

```rust
use scivex::optim::ode::*;

// Simple harmonic oscillator: y'' = -y
// Rewrite as system: y₀' = y₁, y₁' = -y₀
let f = |_t: f64, y: &[f64]| vec![y[1], -y[0]];

let result = solve_ivp(f, 0.0, 10.0, &[1.0, 0.0], OdeMethod::RK45,
    &OdeOptions::default())?;
// result.t = time points, result.y = solution at each time
```

### PDE Solvers

```rust
use scivex::optim::pde::*;

// 1-D heat equation
let u0 = vec![0.0; 100];  // initial temperature
let result = heat_equation_1d(&u0, 0.01, 0.0001, 1000,
    BoundaryCondition::Dirichlet(100.0))?;

// 2-D Laplace equation (steady-state heat)
let result = laplace_2d(&boundary, 1e-6, 10000)?;
```

### Linear and Quadratic Programming

```rust
use scivex::optim::linprog::*;

// Minimize c^T x subject to Ax ≤ b, x ≥ 0
let result = linprog(
    &[-1.0, -2.0],           // objective coefficients (minimize)
    &a_ub,                    // inequality constraint matrix
    &[10.0, 12.0],           // inequality RHS
    None, None,               // no equality constraints
    &bounds,                  // variable bounds
)?;

// Quadratic programming
use scivex::optim::quadprog::*;
let result = quadprog(&Q, &c, Some(&A_ineq), Some(&b_ineq), None, None)?;
```

### Curve Fitting

```rust
use scivex::optim::curve_fit::*;

// Fit: y = a * exp(-b * x) + c
let model = |x: f64, p: &[f64]| p[0] * (-p[1] * x).exp() + p[2];

let result = curve_fit(model, &x_data, &y_data, &[1.0, 0.5, 0.0])?;
println!("Parameters: {:?}", result.params);
println!("Cost: {:.6}", result.cost);
```

### Sparse Solvers

```rust
use scivex::optim::sparse_solve::*;

// Conjugate gradient for Ax = b (A sparse, symmetric positive definite)
let result = conjugate_gradient(&sparse_a, &b, &x0, 1e-10, 1000)?;

// BiCGSTAB for non-symmetric systems
let result = bicgstab(&sparse_a, &b, &x0, 1e-10, 1000)?;

// Preconditioned CG
let precond = JacobiPreconditioner::new(&sparse_a);
let result = preconditioned_cg(&sparse_a, &b, &x0, &precond, 1e-10, 1000)?;
```

---

## 9. Signal Processing & Audio

> **Replaces:** SciPy.signal, librosa

### Filtering

```rust
use scivex::signal::filter::*;

// Design FIR low-pass filter
let coeffs = FirFilter::low_pass::<f64>(0.3, 51)?;   // cutoff=0.3, 51 taps
let filtered = lfilter(&coeffs, &Tensor::from_vec(vec![1.0], vec![1])?, &signal)?;

// Zero-phase filtering (no phase distortion)
let filtered = filtfilt(&b, &a, &signal)?;

// Band-pass
let bp = FirFilter::band_pass::<f64>(0.1, 0.4, 101)?;
```

### Spectral Analysis

```rust
use scivex::signal::spectral::*;

// Short-Time Fourier Transform
let stft_result = stft(&signal, 1024, 256, None)?;   // window=1024, hop=256

// Inverse STFT
let recovered = istft(&stft_result, 1024, 256, None)?;

// Power spectrogram
let spec = spectrogram(&signal, 1024, 256)?;

// Power spectral density
let (freqs, psd) = periodogram(&signal)?;
let (freqs, psd) = welch(&signal, 256, 128)?;   // Welch's method
```

### Wavelets

```rust
use scivex::signal::wavelet::*;

let (approx, detail) = dwt(&signal, Wavelet::Haar)?;
let reconstructed = idwt(&approx, &detail, Wavelet::Haar)?;
```

### Audio Features

```rust
use scivex::signal::features::*;

let sr = 22050.0;  // sample rate

// Mel spectrogram
let mel = mel_spectrogram(&audio, sr, 128, 2048, 512)?;

// MFCCs (speech/audio features)
let mfcc = mfcc(&audio, sr, 13, 128, 2048, 512)?;

// Chroma features (pitch class)
let chroma = chroma_stft(&audio, sr, 12, 2048, 512)?;

// Pitch detection (YIN algorithm)
let pitch = pitch_yin(&audio, sr, 80.0, 800.0)?;
```

### Peak Detection

```rust
use scivex::signal::peak::*;

let peaks = find_peaks(&signal, Some(0.5), Some(10))?;   // min_height, min_distance
let prominences = peak_prominences(&signal, &peaks)?;
```

### Resampling

```rust
use scivex::signal::resample::*;

let resampled = resample(&audio, 44100)?;   // resample to 44100 samples
let decimated = decimate(&signal, 4)?;       // downsample by factor 4
```

### Window Functions

```rust
use scivex::signal::window::*;

let w = hann::<f64>(1024)?;
let w = hamming::<f64>(1024)?;
let w = blackman::<f64>(1024)?;
```

### Audio I/O

```rust
use scivex::signal::audio::*;

let audio = read_wav("recording.wav")?;
write_wav("output.wav", &processed_audio, sample_rate)?;
```

### Rhythm & Beat Tracking

```rust
use scivex::signal::rhythm::*;

let beats = detect_beats(&audio, sr)?;
println!("BPM: {:.1}", beats.bpm);
println!("Beat times: {:?}", beats.beat_times);
```

---

## 10. Image Processing & Computer Vision

> **Replaces:** Pillow, OpenCV, scikit-image, Albumentations

### Image Basics

```rust
use scivex::image::prelude::*;

// Load image
let img = read_bmp("photo.bmp")?;
let img = read_ppm("photo.ppm")?;

println!("{}×{}, {} channels", img.width(), img.height(), img.channels());

// Access pixels
let pixel = img.get_pixel(100, 200)?;   // [r, g, b]

// Convert to tensor for processing
let tensor = img.as_tensor();
```

### Transforms

```rust
use scivex::image::transform::*;

let resized = resize(&img, 224, 224, ResizeMethod::Bilinear)?;
let cropped = crop(&img, 50, 50, 200, 200)?;
let flipped = flip_horizontal(&img);
let rotated = rotate90(&img);
let padded = pad(&img, 10, 10, 10, 10, 0u8);

// High-quality Lanczos resampling
let hq = resize_lanczos(&img, 512, 512, 3)?;
```

### Filters & Edge Detection

```rust
use scivex::image::filter::*;

let blurred = gaussian_blur(&img, 1.5)?;
let edges = sobel_edges(&img)?;
let laplacian = laplacian(&img)?;
let smooth = median_filter(&img, 5)?;

// Custom convolution kernel
let kernel = Tensor::from_vec(vec![-1.0, -1.0, -1.0, -1.0, 8.0, -1.0, -1.0, -1.0, -1.0],
    vec![3, 3])?;
let filtered = convolve2d(&img, &kernel)?;
```

### Color Conversion

```rust
use scivex::image::color::*;

let gray = rgb_to_grayscale(&img)?;
let hsv = rgb_to_hsv(&img)?;
let rgb = hsv_to_rgb(&hsv)?;
```

### Feature Detection

```rust
use scivex::image::features::*;

// Harris corner detection
let corners = harris_corners(&gray, 0.04, 0.01, 3)?;

// FAST corner detection
let corners = fast_corners(&gray, 20.0, 9)?;

// ORB features (keypoints + descriptors)
use scivex::image::orb::*;
let detector = OrbDetector::new(500, 1.2, 8);   // 500 features, 8 levels
let descriptors = detector.detect_and_compute(&gray)?;

// Feature matching
use scivex::image::matching::*;
let matcher = BruteForceMatcher::new();
let matches = matcher.match_descriptors(&desc1, &desc2);
```

### Morphological Operations

```rust
use scivex::image::morphology::*;

let se = StructuringElement::rect(3, 3);
let eroded = erode(&binary_img, &se);
let dilated = dilate(&binary_img, &se);
let opened = opening(&binary_img, &se);
let closed = closing(&binary_img, &se);
```

### Segmentation

```rust
use scivex::image::segment::*;

let (labels, n_components) = connected_components(&binary, 128)?;
let region = region_growing(&img, 100, 100, 30.0)?;
let segments = watershed(&img, &markers)?;
```

### Contour Detection

```rust
use scivex::image::contour::*;

let contours = find_contours(&binary)?;
for c in &contours {
    println!("Area: {:.1}, Perimeter: {:.1}", c.area, c.perimeter);
    let (cx, cy) = contour_moments(c);
    println!("Centroid: ({:.1}, {:.1})", cx, cy);
}
```

### Hough Transform

```rust
use scivex::image::hough::*;

let lines = hough_lines(&edges, 1.0, 0.01, 100);     // rho_res, theta_res, threshold
let circles = hough_circles(&edges, 10, 100, 50);      // min_r, max_r, threshold
```

### Optical Flow

```rust
use scivex::image::optical_flow::*;

// Sparse flow (Lucas-Kanade)
let flow = lucas_kanade(&prev_frame, &next_frame, &keypoints, 21)?;

// Dense flow (Farneback)
let (flow_x, flow_y) = farneback(&prev_frame, &next_frame, 3)?;
```

### Histogram Operations

```rust
use scivex::image::histogram::*;

let hist = histogram(&gray, 256)?;
let equalized = equalize_histogram(&gray)?;
```

### Data Augmentation Pipeline

```rust
use scivex::image::augment::*;

let pipeline = AugmentPipeline::new()
    .add(AugmentStep::RandomFlipH { prob: 0.5 })
    .add(AugmentStep::RandomRotation { max_angle: 15.0 })
    .add(AugmentStep::ColorJitter {
        brightness: 0.2, contrast: 0.2, saturation: 0.2, hue: 0.1
    })
    .add(AugmentStep::CutOut { width: 32, height: 32 })
    .add(AugmentStep::Normalize {
        mean: vec![0.485, 0.456, 0.406],
        std: vec![0.229, 0.224, 0.225],
    });

let augmented = pipeline.apply(&img, &mut rng)?;
```

### Drawing

```rust
use scivex::image::draw::*;

draw_line(&mut img, 10, 10, 200, 200, &[255, 0, 0]);      // red line
draw_rect(&mut img, 50, 50, 100, 80, &[0, 255, 0]);        // green rect
draw_circle(&mut img, 150, 150, 40, &[0, 0, 255]);          // blue circle
fill_rect(&mut img, 50, 50, 100, 80, &[255, 255, 0]);       // filled yellow rect
```

---

## 11. Natural Language Processing

> **Replaces:** NLTK, spaCy (basic), scikit-learn text, Gensim

### Tokenization

```rust
use scivex::nlp::tokenize::*;

let tokenizer = WordTokenizer;
let tokens = tokenizer.tokenize("Hello, world! This is Scivex.");
// ["Hello", ",", "world", "!", "This", "is", "Scivex", "."]

// Subword tokenization (BERT-style)
use scivex::nlp::wordpiece::*;
let wp = WordPieceTokenizer::new(vocab, "[UNK]");
let tokens = wp.tokenize("unaffable");
let ids = wp.encode("Hello world");
let text = wp.decode(&ids);

// Character-level
let char_tok = CharTokenizer;
let chars = char_tok.tokenize("Hello");   // ["H", "e", "l", "l", "o"]

// N-grams
let ngram_tok = NGramTokenizer::new(2);   // bigrams
let bigrams = ngram_tok.tokenize("I love Rust");
```

### Text Preprocessing

```rust
use scivex::nlp::text::*;
use scivex::nlp::stem::*;

// Stopword removal
let filtered = remove_stopwords(&tokens);

// Stemming (Porter)
let stemmer = PorterStemmer::new();
let stemmed = stemmer.stem("running");   // "run"
let stems = stemmer.stem_tokens(&tokens);

// Edit distance
let dist = edit_distance("kitten", "sitting");   // 3

// Normalize text
let clean = normalize("Hello, World! 123");   // "hello world 123"
```

### TF-IDF Vectorization

```rust
use scivex::nlp::vectorize::*;

let corpus = vec![
    "the cat sat on the mat",
    "the dog sat on the log",
    "cats and dogs are friends",
];

// Count vectorizer (bag of words)
let mut cv = CountVectorizer::new();
cv.fit(&corpus);
let bow = cv.transform("the cat and dog");
let matrix = cv.transform_batch(&corpus);

// TF-IDF vectorizer
let mut tfidf = TfidfVectorizer::new();
tfidf.fit(&corpus);
let vector = tfidf.transform("the cat and dog");
let matrix = tfidf.transform_batch(&corpus);
```

### Word Embeddings (Word2Vec)

```rust
use scivex::nlp::word2vec::*;
use scivex::nlp::embeddings::*;

// Train Word2Vec
let config = Word2VecConfig {
    dim: 100,
    window: 5,
    n_negative: 5,
    epochs: 10,
    learning_rate: 0.025,
    min_count: 2,
    mode: Word2VecMode::SkipGram,
    seed: 42,
};

let trainer = Word2VecTrainer::new(config);
let model = trainer.train(&tokenized_corpus)?;

// Use embeddings
let embeddings = WordEmbeddings::from_word2vec(&model);
let vec = embeddings.get("king")?;
let sim = embeddings.similarity("king", "queen")?;
let similar = embeddings.most_similar("python", 10);
let analogy = embeddings.analogy("king", "queen", "man");  // → "woman"
```

### Sentiment Analysis

```rust
use scivex::nlp::sentiment::*;

let analyzer = SentimentAnalyzer::new();   // built-in lexicon
let result = analyzer.analyze("This product is absolutely wonderful!");

println!("Score: {:.2}", result.score);      // positive number
println!("Label: {:?}", result.label);       // Positive
println!("Positive words: {:?}", result.positive_words);
```

### POS Tagging

```rust
use scivex::nlp::pos::*;

let mut tagger = HmmPosTagger::new();
tagger.train(&labeled_sentences);
let tags = tagger.tag(&["The", "cat", "sat"]);
// [Determiner, Noun, Verb]
```

### Named Entity Recognition

```rust
use scivex::nlp::ner::*;

let mut ner = RuleBasedNer::new();
ner.add_rule(r"\b(New York|London|Paris)\b", EntityType::Location);
ner.add_rule(r"\$[\d,]+", EntityType::Money);

let entities = ner.recognize("I visited New York and spent $5,000");
// [Entity { text: "New York", entity_type: Location, ... },
//  Entity { text: "$5,000", entity_type: Money, ... }]
```

### Topic Modeling (LDA)

```rust
use scivex::nlp::lda::*;

let config = LdaConfig {
    n_topics: 5,
    n_iter: 1000,
    alpha: 0.1,
    beta: 0.01,
    seed: 42,
};

let model = LdaModel::fit(&tokenized_docs, config);
let top_words = model.topic_words(0, 10);       // top 10 words in topic 0
let doc_topics = model.doc_topics(&new_doc);     // topic distribution for a doc
```

### Text Similarity

```rust
use scivex::nlp::similarity::*;

let cos_sim = cosine_similarity(&vec_a, &vec_b);
let jaccard = jaccard_similarity(&set_a, &set_b);
let edit_sim = edit_distance_normalized("hello", "hallo");
```

---

## 12. Graph & Network Analysis

> **Replaces:** NetworkX, igraph, petgraph (higher-level API)

### Building Graphs

```rust
use scivex::graph::prelude::*;

// Undirected weighted graph
let mut g = Graph::<f64>::new();
let a = g.add_node();   // 0
let b = g.add_node();   // 1
let c = g.add_node();   // 2
g.add_edge(a, b, 1.0)?;
g.add_edge(b, c, 2.0)?;
g.add_edge(a, c, 4.0)?;

// Directed graph
let mut dg = DiGraph::<f64>::new();
dg.add_edge(0, 1, 1.0)?;
dg.add_edge(1, 2, 2.0)?;

// From edge list
let g = Graph::from_edges(&[(0, 1, 1.0), (1, 2, 2.0), (0, 2, 4.0)])?;

// From adjacency matrix
let g = Graph::from_adjacency_matrix(&adj_tensor)?;
let adj = g.adjacency_matrix()?;
```

### Shortest Paths

```rust
use scivex::graph::shortest::*;

// Dijkstra (non-negative weights)
let result = dijkstra(&g, 0)?;
println!("Distance to node 2: {}", result.distances[2]);
let path = result.path_to(0, 2);   // Some([0, 1, 2])

// Bellman-Ford (handles negative weights)
let result = bellman_ford(&g, 0)?;

// All-pairs (Floyd-Warshall)
let dist_matrix = floyd_warshall(&g)?;
```

### Traversals

```rust
use scivex::graph::traversal::*;

let bfs_order = bfs(&g, 0)?;                        // breadth-first
let dfs_order = dfs(&g, 0)?;                        // depth-first
let topo = topological_sort(&directed_graph)?;       // DAG ordering
```

### Connectivity

```rust
use scivex::graph::connectivity::*;

let components = connected_components(&g);           // Vec<Vec<usize>>
let is_conn = is_connected(&g);

let scc = strongly_connected_components(&dg);        // Kosaraju's
let wcc = weakly_connected_components(&dg)?;
```

### Minimum Spanning Tree

```rust
use scivex::graph::mst::*;

let mst = kruskal(&g)?;
println!("MST weight: {}", mst.total_weight);
println!("MST edges: {:?}", mst.edges);

let mst = prim(&g)?;   // alternative algorithm
```

### Centrality & PageRank

```rust
use scivex::graph::centrality::*;

let deg = degree_centrality(&g);
let bet = betweenness_centrality(&g);      // Brandes' algorithm

let pr = pagerank(&dg, 0.85, 100, 1e-6)?; // damping=0.85
for (node, score) in pr.iter().enumerate() {
    println!("Node {}: PageRank={:.4}", node, score);
}
```

### Community Detection

```rust
use scivex::graph::community::*;

let communities = label_propagation(&g, 100);   // max 100 iterations
// communities[i] = community label for node i
```

### Network Flow

```rust
use scivex::graph::flow::*;

// Max flow (Edmonds-Karp)
let result = max_flow(&dg, source, sink)?;
println!("Max flow: {}", result.max_flow);

// Bipartite matching (Hopcroft-Karp)
let matching = bipartite_matching(left_size, right_size, &edges);
println!("Matching size: {}", matching.size);
```

---

## 13. Symbolic Mathematics

> **Replaces:** SymPy

### Symbolic Expressions

```rust
use scivex::sym::prelude::*;

let x = var("x");
let y = var("y");

// Build expressions with operator overloading
let expr = &x * &x + constant(2.0) * &x + constant(1.0);   // x² + 2x + 1
let expr2 = sin(x.clone()) * cos(y.clone());

// Evaluate
let mut vars = HashMap::new();
vars.insert("x".into(), 3.0);
let result = expr.eval(&vars)?;   // 16.0

// Substitute
let substituted = expr.substitute("x", &constant(5.0));
```

### Differentiation

```rust
use scivex::sym::diff::*;

let x = var("x");
let expr = sin(x.clone()) * exp(x.clone());   // sin(x) * e^x

let derivative = diff(&expr, "x")?;
// cos(x) * e^x + sin(x) * e^x

let second = diff_n(&expr, "x", 2)?;          // second derivative
```

### Integration

```rust
use scivex::sym::integrate::*;

let x = var("x");
let expr = &x * &x;   // x²

let integral = integrate(&expr, "x")?;
// x³/3 (+ C implicit)

// Definite integral
let value = definite_integral(&expr, "x", 0.0, 1.0)?;
// 0.333...
```

### Simplification

```rust
use scivex::sym::simplify::*;

let x = var("x");
let messy = &x + constant(0.0);                     // x + 0
let clean = simplify(&messy);                         // x

let messy2 = &x * constant(1.0) + &x * constant(0.0);
let clean2 = simplify(&messy2);                       // x
```

### Equation Solving

```rust
use scivex::sym::solve::*;

let x = var("x");
let expr = constant(2.0) * &x + constant(6.0);     // 2x + 6 = 0

let solution = solve_linear(&expr, "x")?;            // x = -3

// Quadratic: ax² + bx + c = 0
let (r1, r2) = solve_quadratic(&constant(1.0), &constant(-5.0), &constant(6.0))?;
// r1 = 3, r2 = 2
```

### Polynomials

```rust
use scivex::sym::polynomial::*;

let p = Polynomial::new(vec![1.0, -5.0, 6.0]);   // x² - 5x + 6
let val = p.eval(3.0);                             // 0.0
let dp = p.derivative();                           // 2x - 5
let roots = p.roots();                             // [2.0, 3.0]
```

### Taylor Series

```rust
use scivex::sym::taylor::*;

let x = var("x");
let expr = exp(x.clone());

let series = taylor(&expr, "x", 0.0, 5)?;         // 5th-order Taylor at x=0
let series = maclaurin(&expr, "x", 5)?;            // same (center=0)
```

---

## 14. Classical Machine Learning

> **Replaces:** scikit-learn

### Data Preparation

```rust
use scivex::ml::prelude::*;

// Train/test split
let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, &mut rng);

// Cross-validation
let scores = cross_val_score(&mut model, &x, &y, 5)?;   // 5-fold CV
println!("Mean CV score: {:.4}", scores.iter().sum::<f64>() / scores.len() as f64);
```

### Preprocessing

```rust
use scivex::ml::preprocessing::*;

// StandardScaler: zero mean, unit variance
let mut scaler = StandardScaler::new();
scaler.fit(&x_train)?;
let x_train_scaled = scaler.transform(&x_train)?;
let x_test_scaled = scaler.transform(&x_test)?;

// MinMaxScaler: [0, 1] range
let mut scaler = MinMaxScaler::new();
scaler.fit(&x_train)?;
let x_scaled = scaler.transform(&x_train)?;

// One-hot encoding
let encoded = one_hot_encode(&labels, n_classes)?;

// Label encoding
let mut encoder = LabelEncoder::new();
encoder.fit(&string_labels)?;
let encoded = encoder.transform(&string_labels)?;
let decoded = encoder.inverse_transform(&encoded)?;
```

### Linear Models

```rust
use scivex::ml::linear::*;

// Linear Regression
let mut lr = LinearRegression::new();
lr.fit(&x_train, &y_train)?;
let predictions = lr.predict(&x_test)?;
println!("Coefficients: {:?}", lr.coefficients());

// Ridge Regression (L2 regularization)
let mut ridge = RidgeRegression::new(1.0);   // alpha=1.0
ridge.fit(&x_train, &y_train)?;

// Lasso Regression (L1 regularization)
let mut lasso = LassoRegression::new(0.1);
lasso.fit(&x_train, &y_train)?;

// Logistic Regression
let mut logit = LogisticRegression::new(0.01, 1000);   // lr, max_iter
logit.fit(&x_train, &y_train)?;
let probs = logit.predict_proba(&x_test)?;
```

### Decision Trees

```rust
use scivex::ml::tree::*;

// Classification tree
let mut tree = DecisionTreeClassifier::new(10);   // max_depth=10
tree.fit(&x_train, &y_train)?;
let preds = tree.predict(&x_test)?;

// Regression tree
let mut tree = DecisionTreeRegressor::new(10);
tree.fit(&x_train, &y_train)?;
let preds = tree.predict(&x_test)?;
```

### Random Forests

```rust
use scivex::ml::ensemble::*;

// Random Forest Classifier
let mut rf = RandomForestClassifier::new(100, 5);   // n_trees, max_depth
rf.fit(&x_train, &y_train)?;
let preds = rf.predict(&x_test)?;
let importances = rf.feature_importances();

// Random Forest Regressor
let mut rf = RandomForestRegressor::new(100, 10);
rf.fit(&x_train, &y_train)?;

// Gradient Boosted Trees
let mut gb = GradientBoostedClassifier::new(100, 3, 0.1);  // n_trees, depth, lr
gb.fit(&x_train, &y_train)?;
```

### Support Vector Machines

```rust
use scivex::ml::svm::*;

let mut svm = SvmClassifier::new(SvmKernel::Rbf { gamma: 0.1 }, 1.0);
svm.fit(&x_train, &y_train)?;
let preds = svm.predict(&x_test)?;

// Other kernels
let svm_linear = SvmClassifier::new(SvmKernel::Linear, 1.0);
let svm_poly = SvmClassifier::new(SvmKernel::Polynomial { degree: 3, gamma: 0.1, coef0: 0.0 }, 1.0);
```

### k-Nearest Neighbors

```rust
use scivex::ml::knn::*;

let mut knn = KnnClassifier::new(5);   // k=5
knn.fit(&x_train, &y_train)?;
let preds = knn.predict(&x_test)?;

// Regression
let mut knn = KnnRegressor::new(5);
knn.fit(&x_train, &y_train)?;
```

### Naive Bayes

```rust
use scivex::ml::naive_bayes::*;

let mut nb = GaussianNaiveBayes::new();
nb.fit(&x_train, &y_train)?;
let preds = nb.predict(&x_test)?;
let probs = nb.predict_proba(&x_test)?;
```

### Clustering

```rust
use scivex::ml::cluster::*;

// K-Means
let mut kmeans = KMeans::new(3, 100);   // k=3, max_iter=100
kmeans.fit(&x)?;
let labels = kmeans.predict(&x)?;
let centroids = kmeans.centroids();
let inertia = kmeans.inertia();

// DBSCAN
let mut dbscan = Dbscan::new(0.5, 5);   // eps=0.5, min_points=5
let labels = dbscan.fit_predict(&x)?;
```

### Dimensionality Reduction

```rust
use scivex::ml::decomposition::*;

// PCA
let mut pca = Pca::new(2);   // reduce to 2 dimensions
pca.fit(&x)?;
let x_2d = pca.transform(&x)?;
let variance_ratio = pca.explained_variance_ratio();
```

### Metrics

```rust
use scivex::ml::metrics::*;

// Classification metrics
let acc = accuracy(&y_true, &y_pred);
let prec = precision(&y_true, &y_pred);
let rec = recall(&y_true, &y_pred);
let f1 = f1_score(&y_true, &y_pred);
let auc = roc_auc(&y_true, &y_scores)?;
let cm = confusion_matrix(&y_true, &y_pred, n_classes);
let report = classification_report(&y_true, &y_pred, n_classes);

// Regression metrics
let mse = mean_squared_error(&y_true, &y_pred);
let rmse = mse.sqrt();
let mae = mean_absolute_error(&y_true, &y_pred);
let r2 = r2_score(&y_true, &y_pred);
```

### Approximate Nearest Neighbors

```rust
use scivex::ml::ann::*;

// Annoy-style random projection tree index
let mut index = AnnoyIndex::new(128, 10);   // 128-dim, 10 trees
index.add_items(&vectors)?;
index.build(&mut rng)?;

let (indices, distances) = index.query(&query_vec, 10)?;   // find 10 nearest
```

---

## 15. Neural Networks & Deep Learning

> **Replaces:** PyTorch, TensorFlow (training API)

### Autograd System

```rust
use scivex::nn::prelude::*;

// Variables track computation for automatic differentiation
let x = Variable::new(Tensor::from_vec(vec![2.0, 3.0], vec![2])?);
let w = Variable::new(Tensor::from_vec(vec![0.5, -0.3], vec![2])?);

// Forward pass — builds computation graph
let y = ops::add(&ops::mul(&w, &x)?, &Variable::new(Tensor::scalar(1.0)))?;

// Backward pass
y.backward(None)?;
let grad_w = w.grad();   // dy/dw
```

### Layers

```rust
use scivex::nn::layer::*;

let mut rng = Rng::new(42);

// Fully connected
let linear = Linear::new(784, 128, true, &mut rng);

// Convolutional
let conv = Conv2d::new(3, 64, 3, 1, 1, true, &mut rng);
// in_channels=3, out_channels=64, kernel=3, stride=1, padding=1

// Batch normalization
let bn = BatchNorm::new(64, 1e-5, 0.1);

// Dropout
let dropout = Dropout::new(0.5);

// LSTM
let lstm = Lstm::new(128, 256, true, &mut rng);
// input_size=128, hidden_size=256, bias=true

// Multi-head attention
let attn = MultiHeadAttention::new(512, 8, &mut rng);
// embed_dim=512, n_heads=8
```

### Building a Network

```rust
use scivex::nn::prelude::*;

let mut rng = Rng::new(42);

// Sequential model
let mut model = Sequential::new(vec![
    Box::new(Linear::new(784, 256, true, &mut rng)),
    Box::new(ReLU),
    Box::new(Dropout::new(0.2)),
    Box::new(Linear::new(256, 128, true, &mut rng)),
    Box::new(ReLU),
    Box::new(Dropout::new(0.2)),
    Box::new(Linear::new(128, 10, true, &mut rng)),
]);

// Forward pass
let output = model.forward(&input)?;
```

### Loss Functions

```rust
use scivex::nn::loss::*;

let mse_loss = mse_loss(&predictions, &targets)?;
let ce_loss = cross_entropy_loss(&logits, &targets)?;
let bce_loss = binary_cross_entropy_loss(&probs, &binary_targets)?;
```

### Optimizers

```rust
use scivex::nn::optim::*;

// SGD with momentum
let mut optimizer = Sgd::new(model.parameters(), 0.01, 0.9);

// Adam
let mut optimizer = Adam::new(model.parameters(), 0.001);

// Training loop
for epoch in 0..100 {
    let output = model.forward(&x_train)?;
    let loss = cross_entropy_loss(&output, &y_train)?;

    // Backward pass
    loss.backward(None)?;

    // Update weights
    optimizer.step();

    // Zero gradients for next iteration
    optimizer.zero_grad();

    println!("Epoch {}: loss={:.4}", epoch, loss.data().as_slice()[0]);
}
```

### Complete Training Example: MNIST Classifier

```rust
use scivex::prelude::*;

fn train_mnist() -> Result<()> {
    let mut rng = Rng::new(42);

    // Load data (from tensors — see I/O section for file loading)
    let x_train = Tensor::<f64>::from_vec(/* 60000×784 */, vec![60000, 784])?;
    let y_train = Tensor::<f64>::from_vec(/* 60000 labels */, vec![60000])?;

    // Build model
    let mut model = Sequential::new(vec![
        Box::new(Linear::new(784, 128, true, &mut rng)),
        Box::new(ReLU),
        Box::new(Dropout::new(0.2)),
        Box::new(Linear::new(128, 10, true, &mut rng)),
    ]);

    let mut optimizer = Adam::new(model.parameters(), 0.001);

    // Training loop
    for epoch in 0..20 {
        model.train();  // enable dropout
        let logits = model.forward(&Variable::new(x_train.clone()))?;
        let loss = cross_entropy_loss(&logits, &Variable::new(y_train.clone()))?;

        loss.backward(None)?;
        optimizer.step();
        optimizer.zero_grad();

        // Evaluate
        model.eval();  // disable dropout
        let test_logits = model.forward(&Variable::new(x_test.clone()))?;
        let preds = test_logits.data(); // argmax to get predictions
        let acc = accuracy(&y_test_vec, &pred_vec);

        println!("Epoch {}: loss={:.4}, test_acc={:.2}%", epoch, loss_val, acc * 100.0);
    }

    Ok(())
}
```

### Convolutional Neural Network

```rust
let mut model = Sequential::new(vec![
    // Conv block 1
    Box::new(Conv2d::new(1, 32, 3, 1, 1, true, &mut rng)),
    Box::new(BatchNorm::new(32, 1e-5, 0.1)),
    Box::new(ReLU),
    Box::new(MaxPool2d::new(2, 2)),

    // Conv block 2
    Box::new(Conv2d::new(32, 64, 3, 1, 1, true, &mut rng)),
    Box::new(BatchNorm::new(64, 1e-5, 0.1)),
    Box::new(ReLU),
    Box::new(MaxPool2d::new(2, 2)),

    // Classifier
    Box::new(Flatten),
    Box::new(Linear::new(64 * 7 * 7, 128, true, &mut rng)),
    Box::new(ReLU),
    Box::new(Dropout::new(0.5)),
    Box::new(Linear::new(128, 10, true, &mut rng)),
]);
```

### ONNX Model Import & Inference

```rust
use scivex::nn::onnx::*;

// Load ONNX model (supports ResNet, MobileNet, BERT, etc.)
let model_bytes = std::fs::read("model.onnx")?;
let model = OnnxModel::from_bytes(&model_bytes)?;

// Create executor with graph optimizations
let mut executor = OnnxExecutor::new(&model)?;
executor.optimize();   // constant folding, Conv+BN fusion

// Run inference
let input = Tensor::<f32>::from_vec(/* image data */, vec![1, 3, 224, 224])?;
let outputs = executor.run(&[("input", &input)])?;
let logits = &outputs["output"];

// Supported operators: Conv, BatchNorm, Relu, MaxPool, AveragePool, Flatten,
// Gemm, Add, Mul, Reshape, Transpose, Softmax, MatMul, Concat, Unsqueeze,
// Squeeze, Sigmoid, Tanh, Clip, Resize, Gather, Split, ReduceMean, ReduceSum,
// Cast, Where, and more.
```

---

## 16. Reinforcement Learning

> **Replaces:** Stable-Baselines3, RLlib (basic algorithms)

### Environments

```rust
use scivex::rl::prelude::*;

// CartPole
let mut env = CartPole::new();
let obs = env.reset();

// Step
let result = env.step(&1);   // action = push right
println!("Reward: {}, Done: {}", result.reward, result.done);

// Other environments
let mut env = MountainCar::new();
let mut env = GridWorld::new(10, 10, (0, 0), (9, 9));
```

### DQN (Deep Q-Network)

```rust
use scivex::rl::algorithms::dqn::*;

let config = DqnConfig::new()
    .with_learning_rate(0.001)
    .with_gamma(0.99)
    .with_epsilon(1.0)
    .with_epsilon_decay(0.995)
    .with_batch_size(32)
    .with_target_update_freq(100)
    .with_buffer_capacity(10000);

let mut agent = DqnAgent::new(4, 2, config);   // obs_dim=4, n_actions=2
let mut env = CartPole::new();

// Training loop
for episode in 0..1000 {
    let mut obs = env.reset();
    let mut total_reward = 0.0;

    loop {
        let action = agent.select_action(&obs);
        let result = env.step(&action);
        agent.push_experience(&obs, action, result.reward, &result.observation, result.done);
        let loss = agent.train(&mut rng)?;
        obs = result.observation;
        total_reward += result.reward;

        if result.done { break; }
    }

    agent.update_target();
    println!("Episode {}: reward={:.1}", episode, total_reward);
}
```

### PPO (Proximal Policy Optimization)

```rust
use scivex::rl::algorithms::ppo::*;

let config = PpoConfig::new()
    .with_learning_rate(0.0003)
    .with_gamma(0.99)
    .with_gae_lambda(0.95)
    .with_clip_epsilon(0.2)
    .with_n_epochs(4);

let mut agent = PpoAgent::new(4, 2, config);
let reward = agent.run_episode(&mut env)?;
```

### SAC (Soft Actor-Critic) — Continuous Actions

```rust
use scivex::rl::algorithms::sac::*;

let config = SacConfig::new()
    .with_learning_rate(0.0003)
    .with_gamma(0.99)
    .with_tau(0.005)
    .with_alpha(0.2);

let mut agent = SacAgent::new(obs_dim, action_dim, config);
```

### TD3 (Twin Delayed DDPG)

```rust
use scivex::rl::algorithms::td3::*;

let config = Td3Config::new();
let mut agent = Td3Agent::new(obs_dim, action_dim, config);
```

### Hindsight Experience Replay (Goal-Conditioned RL)

```rust
use scivex::rl::her::*;

let reward_fn = |achieved: &[f64], desired: &[f64]| -> f64 {
    let dist: f64 = achieved.iter().zip(desired).map(|(a, d)| (a - d).powi(2)).sum();
    if dist.sqrt() < 0.05 { 1.0 } else { -1.0 }
};

let mut buffer = HerReplayBuffer::new(10000, HerStrategy::Future(4), reward_fn);

// Push goal-conditioned transitions
buffer.push(GoalTransition {
    state: obs.clone(),
    action: action as f64,
    reward,
    next_state: next_obs.clone(),
    done: false,
    desired_goal: goal.clone(),
    achieved_goal: achieved.clone(),
});

buffer.end_episode(&mut rng);   // relabels with hindsight goals
let batch = buffer.sample(32, &mut rng)?;
```

### Episode Logging

```rust
use scivex::rl::logger::EpisodeLogger;

let mut logger = EpisodeLogger::new();
logger.log(episode, total_reward, steps, epsilon);

let mean_100 = logger.mean_reward(100);
logger.print_summary();
logger.to_csv("training_log.csv")?;
```

---

## 17. GPU Acceleration

> **Replaces:** CuPy, torch.cuda (basic ops)

```rust
use scivex::gpu::prelude::*;

// Initialize GPU (auto-detects Vulkan/Metal/DX12)
let device = GpuDevice::new()?;
println!("GPU: {}, Memory: {} MB", device.info().name, device.info().memory_bytes / 1_000_000);

// Transfer tensors to GPU
let a_gpu = GpuTensor::from_tensor(&device, &a_cpu)?;
let b_gpu = GpuTensor::from_tensor(&device, &b_cpu)?;

// GPU operations
let c_gpu = scivex::gpu::ops::matmul(&a_gpu, &b_gpu)?;
let d_gpu = scivex::gpu::ops::add(&a_gpu, &b_gpu)?;

// Transfer back to CPU
let c_cpu = c_gpu.to_tensor()?;

// GPU optimizers for neural network training
let mut optimizer = GpuAdam::new(0.001, 0.9, 0.999, 1e-8);
optimizer.step(&mut params_gpu, &grads_gpu);
```

### CUDA (Optional)

```rust
// feature = "cuda"
use scivex::gpu::cuda::*;

let ctx = CudaContext::new(0)?;   // device 0
let stream = CudaStream::new(&ctx)?;
let tensor = CudaTensor::from_tensor(&ctx, &cpu_tensor)?;
```

---

## 18. Visualization & Plotting

> **Replaces:** Matplotlib, Seaborn, Plotly (static)

### Basic Plots

```rust
use scivex::viz::prelude::*;

// Line plot
let fig = Figure::new().plot(
    Axes::new()
        .title("Training Loss")
        .x_label("Epoch")
        .y_label("Loss")
        .add_plot(LinePlot::new(epochs.clone(), losses.clone())
            .color(Color::blue())
            .label("Train"))
        .add_plot(LinePlot::new(epochs.clone(), val_losses.clone())
            .color(Color::red())
            .label("Validation"))
);

// Render
let svg = fig.to_svg()?;
std::fs::write("loss.svg", svg)?;

let html = fig.to_html()?;
std::fs::write("loss.html", html)?;

// Terminal output (braille art)
let term = fig.to_terminal()?;
println!("{}", term);
```

### Scatter Plot

```rust
let fig = Figure::new().plot(
    Axes::new()
        .title("Feature Space")
        .add_plot(ScatterPlot::new(x.clone(), y.clone())
            .color(Color::red())
            .marker(MarkerShape::Circle)
            .size(4.0))
);
```

### Bar Chart

```rust
let fig = Figure::new().plot(
    Axes::new()
        .title("Sales by Region")
        .add_plot(BarPlot::new(
            vec!["North", "South", "East", "West"],
            vec![120.0, 95.0, 140.0, 110.0])
            .color(Color::blue()))
);
```

### Histogram

```rust
let fig = Figure::new().plot(
    Axes::new()
        .title("Distribution")
        .add_plot(Histogram::new(data.clone())
            .bins(30)
            .color(Color::green())
            .density(true))
);
```

### Statistical Plots

```rust
// Box plot
let fig = Figure::new().plot(
    Axes::new()
        .add_plot(BoxPlotBuilder::new(vec![
            ("Group A", data_a.clone()),
            ("Group B", data_b.clone()),
            ("Group C", data_c.clone()),
        ]).show_outliers(true))
);

// Violin plot
let fig = Figure::new().plot(
    Axes::new()
        .add_plot(ViolinPlot::new(groups.clone()).show_box(true))
);

// Heatmap
let fig = Figure::new().plot(
    Axes::new()
        .title("Correlation Matrix")
        .add_plot(HeatmapBuilder::new(corr_data)
            .colormap(ColorMap::RdBu)
            .x_labels(col_names.clone())
            .y_labels(col_names.clone()))
);

// Pie chart
let fig = Figure::new().plot(
    Axes::new()
        .add_plot(PieChart::new(
            vec!["Rust", "Python", "Go", "Other"],
            vec![45.0, 30.0, 15.0, 10.0])
            .donut(true))
);

// Q-Q plot
let fig = Figure::new().plot(
    Axes::new()
        .add_plot(QQPlot::new(data.clone()))
);

// Regression plot
let fig = Figure::new().plot(
    Axes::new()
        .add_plot(RegressionPlot::new(x.clone(), y.clone())
            .show_ci(true))
);
```

### Pair Plot (Seaborn-style)

```rust
let fig = Figure::new().plot(
    Axes::new()
        .add_plot(PairPlot::new(&feature_matrix)
            .labels(vec!["Sepal L", "Sepal W", "Petal L", "Petal W"])
            .diag_mode(DiagMode::Histogram))
);
```

### Contour and Surface Plots

```rust
// Contour
let fig = Figure::new().plot(
    Axes::new()
        .add_plot(ContourPlot::new(x_grid, y_grid, &z_tensor)
            .levels(vec![0.1, 0.5, 1.0, 2.0])
            .filled(true))
);

// 3-D surface (wireframe)
let fig = Figure::new().plot(
    Axes::new()
        .add_plot(SurfacePlot::new(x_grid, y_grid, &z_tensor)
            .mode(SurfaceMode::Wireframe))
);
```

### Grammar-of-Graphics (Vega-Lite-style)

```rust
use scivex::viz::chart::*;

let chart = Chart::new()
    .data(columns)
    .mark(Mark::Point)
    .encode_x("age", ScaleType::Linear)
    .encode_y("salary", ScaleType::Linear)
    .encode_color("department")
    .encode_size("experience")
    .title("Employee Compensation")
    .width(800.0)
    .height(600.0);

let svg = chart.to_svg()?;
let html = chart.to_html()?;
```

### Animation

```rust
use scivex::viz::animation::*;

let mut anim = Animation::new().fps(30);

for i in 0..100 {
    let fig = Figure::new().plot(/* frame i */);
    anim.add_frame(fig);
}

let gif = anim.to_gif()?;
std::fs::write("animation.gif", gif)?;
```

### Themes

```rust
// Available themes: Default, Dark, Minimal, Publication
let fig = Figure::new()
    .theme(Theme::Publication)
    .plot(/* ... */);
```

---

## 19. ML Pipelines & AutoML

> **Replaces:** scikit-learn Pipeline, TPOT, auto-sklearn (basic)

### Pipeline Building

```rust
use scivex::ml::prelude::*;
use scivex::ml::automl::*;

// Define search space
let mut optimizer = PipelineOptimizer::new();

// Add preprocessing options
optimizer.add_transformer("standard_scaler", || Box::new(StandardScaler::new()));
optimizer.add_transformer("minmax_scaler", || Box::new(MinMaxScaler::new()));

// Add model options
optimizer.add_predictor("random_forest", || Box::new(RandomForestClassifier::new(100, 5)));
optimizer.add_predictor("logistic", || Box::new(LogisticRegression::new(0.01, 1000)));
optimizer.add_predictor("knn", || Box::new(KnnClassifier::new(5)));

// Search over all combinations with cross-validation
let result = optimizer.search(&x, &y, 5, &mut rng)?;   // 5-fold CV

println!("Best transformer: {}", result.best_transformer);
println!("Best predictor: {}", result.best_predictor);
println!("Best score: {:.4}", result.best_score);
```

### End-to-End ML Workflow

```rust
use scivex::prelude::*;

fn full_ml_pipeline() -> Result<()> {
    // 1. Load data from database
    let conn = PostgresConnection::new("host=localhost dbname=ml_data")?;
    let df = conn.query("SELECT * FROM training_data WHERE created_at > '2025-01-01'")?;

    // 2. Explore data
    let desc = describe(df.column_typed::<f64>("salary")?.as_slice())?;
    println!("Salary stats: mean={:.0}, std={:.0}", desc.mean, desc.std_dev);

    let corr = pearson(
        df.column_typed::<f64>("experience")?.as_slice(),
        df.column_typed::<f64>("salary")?.as_slice(),
    )?;
    println!("Experience-Salary correlation: {:.3}", corr);

    // 3. Prepare features
    let x = /* extract feature tensor */;
    let y = /* extract target tensor */;
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, &mut rng);

    // 4. Preprocess
    let mut scaler = StandardScaler::new();
    scaler.fit(&x_train)?;
    let x_train = scaler.transform(&x_train)?;
    let x_test = scaler.transform(&x_test)?;

    // 5. Train model
    let mut rf = RandomForestClassifier::new(200, 10);
    rf.fit(&x_train, &y_train)?;

    // 6. Evaluate
    let preds = rf.predict(&x_test)?;
    let acc = accuracy(&y_test_vec, &preds_vec);
    let f1 = f1_score(&y_test_vec, &preds_vec);
    println!("Accuracy: {:.2}%, F1: {:.4}", acc * 100.0, f1);

    // 7. Visualize results
    let fig = Figure::new().plot(
        Axes::new()
            .title("Confusion Matrix")
            .add_plot(HeatmapBuilder::new(confusion_matrix_data)
                .colormap(ColorMap::Blues))
    );
    std::fs::write("confusion.svg", fig.to_svg()?)?;

    // 8. Cross-validate
    let cv_scores = cross_val_score(&mut rf, &x, &y, 5)?;
    println!("CV scores: {:?}", cv_scores);

    Ok(())
}
```

---

## 20. Model Deployment & Interop

### ONNX Export/Import

Scivex can **import** ONNX models for inference. This means you can:
1. Train a model in PyTorch/TensorFlow
2. Export it as ONNX
3. Run inference in Scivex (pure Rust, no Python runtime)

```rust
use scivex::nn::onnx::*;

let model = OnnxModel::from_bytes(&std::fs::read("model.onnx")?)?;
let mut executor = OnnxExecutor::new(&model)?;
executor.optimize();

// Serve predictions
fn predict(executor: &mut OnnxExecutor, input: &[f64]) -> Result<Vec<f64>> {
    let tensor = Tensor::from_vec(input.to_vec(), vec![1, input.len()])?;
    let outputs = executor.run(&[("input", &tensor)])?;
    Ok(outputs["output"].as_slice().to_vec())
}
```

### C FFI (feature = "ffi")

Expose Scivex models to **any language** via C ABI:

```c
// C code calling Scivex
#include "scivex.h"

ScivexTensor* t = scivex_tensor_zeros(shape, 2);
ScivexDataFrame* df = scivex_frame_from_csv("data.csv");
ScivexModel* model = scivex_ml_random_forest(100, 5);
scivex_ml_fit(model, x, y);
ScivexTensor* preds = scivex_ml_predict(model, x_test);
scivex_tensor_free(t);
```

### WebAssembly (feature = "wasm")

Run Scivex **in the browser** or Node.js:

```javascript
import init, { WasmTensor, WasmLinearRegression, stats_mean } from 'scivex-wasm';

await init();

// Tensor operations
const t = WasmTensor.from_array(new Float64Array([1, 2, 3, 4]), [2, 2]);
const result = t.matmul(t);

// ML in the browser
const model = new WasmLinearRegression();
model.fit(x_tensor, y_tensor);
const predictions = model.predict(x_test_tensor);

// Statistics
const mean = stats_mean(new Float64Array([1, 2, 3, 4, 5]));
```

### Python Bindings (pyscivex)

Use Scivex from Python with near-native speed:

```python
import pyscivex as sv

# Tensors
t = sv.Tensor([1.0, 2.0, 3.0, 4.0], shape=[2, 2])
result = t.matmul(t)

# DataFrames
df = sv.read_csv("data.csv")
filtered = df.filter("age > 30")
```

### Building a REST API with Scivex

```rust
// Example using actix-web (external crate) + Scivex for model serving
use actix_web::{web, App, HttpServer, HttpResponse};
use scivex::nn::onnx::*;
use std::sync::Mutex;

struct AppState {
    executor: Mutex<OnnxExecutor>,
}

async fn predict(data: web::Json<Vec<f64>>, state: web::Data<AppState>) -> HttpResponse {
    let mut exec = state.executor.lock().unwrap();
    let input = Tensor::from_vec(data.into_inner(), vec![1, 784]).unwrap();
    let output = exec.run(&[("input", &input)]).unwrap();
    HttpResponse::Ok().json(output["output"].as_slice())
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let model = OnnxModel::from_bytes(&std::fs::read("model.onnx")?)?;
    let executor = OnnxExecutor::new(&model)?;

    let state = web::Data::new(AppState {
        executor: Mutex::new(executor),
    });

    HttpServer::new(move || {
        App::new()
            .app_data(state.clone())
            .route("/predict", web::post().to(predict))
    })
    .bind("0.0.0.0:8080")?
    .run()
    .await
}
```

---

## 21. What Scivex Does NOT Do

The following are **out of scope** for Scivex. These are application-level frameworks
rather than foundational libraries:

| Category | Library | Why Out of Scope |
|----------|---------|-----------------|
| LLM Inference | Transformers, llama.cpp | Requires pre-trained model weights and a model ecosystem. Scivex provides the building blocks (tensors, layers, ONNX) but not a model hub. |
| LLM Frameworks | LangChain, LlamaIndex, DSPy | Application frameworks for orchestrating LLM calls. These are API wrappers, not numerical computing. |
| RAG Pipelines | Haystack, LlamaIndex | Application-level search+generation pipelines. Scivex provides the components (embeddings, TF-IDF, vector search via ANN) but not the orchestration. |
| Distributed Training | Horovod, DeepSpeed | Requires cluster coordination, NCCL, etc. Scivex is single-node. |
| MLOps | MLflow, Kubeflow, W&B | Experiment tracking, model registries, deployment platforms — infrastructure, not math. |
| Data Orchestration | Airflow, Prefect, Dagster | Workflow orchestration — infrastructure concern. |
| Web Frameworks | FastAPI, Flask | Scivex is a library, not a web framework. Use actix-web, axum, etc. alongside Scivex. |
| Database Engines | PostgreSQL, DuckDB | Scivex **connects to** databases but doesn't implement one. |

### What You CAN Do Instead

- **"I want to run an LLM"** → Train a transformer architecture with `scivex-nn` layers (Linear, MultiHeadAttention, LayerNorm). Load pre-trained weights via ONNX. You handle the tokenizer and weights.
- **"I want vector search for RAG"** → Use `scivex-ml::AnnoyIndex` for approximate nearest neighbor search + `scivex-nlp::TfidfVectorizer` or `Word2Vec` for embeddings.
- **"I want to serve a model API"** → Use `scivex-nn::OnnxExecutor` inside an actix-web/axum handler (see example above).
- **"I want distributed training"** → Not supported. Scivex is designed for single-machine, multi-core (via Rayon + GPU).

---

## Summary: Python → Scivex Mapping

| Python | Scivex | Feature Flag |
|--------|--------|-------------|
| `numpy` | `scivex-core` (Tensor, linalg, FFT) | `core` |
| `pandas` / `polars` | `scivex-frame` (DataFrame, Series, SQL) | `frame` |
| `pandas.read_csv` / `sqlalchemy` | `scivex-io` (CSV, JSON, Parquet, SQL connectors) | `io` |
| `scipy.stats` / `statsmodels` | `scivex-stats` (distributions, tests, ARIMA, GLM, survival) | `stats` |
| `scipy.optimize` | `scivex-optim` (minimize, roots, integrate, ODE, LP) | `optim` |
| `matplotlib` / `seaborn` | `scivex-viz` (plots, charts, SVG/HTML/terminal) | `viz` |
| `scikit-learn` | `scivex-ml` (linear, trees, SVM, kNN, clustering, pipelines) | `ml` |
| `pytorch` / `tensorflow` | `scivex-nn` (autograd, layers, optimizers, ONNX) | `nn` |
| `pillow` / `opencv` / `albumentations` | `scivex-image` (transforms, filters, features, augmentation) | `image` |
| `scipy.signal` / `librosa` | `scivex-signal` (filters, STFT, wavelets, MFCC) | `signal` |
| `networkx` | `scivex-graph` (Dijkstra, PageRank, MST, flow) | `graph` |
| `nltk` / `spacy` / `gensim` | `scivex-nlp` (tokenizers, TF-IDF, Word2Vec, NER) | `nlp` |
| `sympy` | `scivex-sym` (differentiation, integration, solving) | `sym` |
| `stable-baselines3` | `scivex-rl` (DQN, PPO, SAC, TD3, HER) | `rl` |
| `cupy` / `torch.cuda` | `scivex-gpu` (wgpu + optional CUDA) | `gpu` |

---

*Built with Scivex v0.1.0 — MIT License*
