# Migrating from Pandas to Scivex

This guide provides a side-by-side reference for translating common Pandas
patterns into their Scivex equivalents.  Every Scivex API shown here exists in
the codebase -- nothing is hypothetical.

---

## Setup

**Python (Pandas)**

```python
import pandas as pd
```

**Rust (Scivex)**

```rust
use scivex_frame::prelude::*;
// For I/O:
use scivex_io::prelude::*;
```

---

## Quick-Reference Table

| Operation | Pandas | Scivex |
|-----------|--------|--------|
| Create DataFrame | `pd.DataFrame({"a": [1,2], "b": [3,4]})` | `DataFrame::builder().add_column("a", vec![1_i32, 2]).add_column("b", vec![3_i32, 4]).build()?` |
| Shape | `df.shape` | `df.shape()` |
| Row count | `len(df)` | `df.nrows()` |
| Column count | — | `df.ncols()` |
| Column names | `df.columns.tolist()` | `df.column_names()` |
| Column dtypes | `df.dtypes` | `df.dtypes()` |
| Select columns | `df[["a", "b"]]` | `df.select(&["a", "b"])?` |
| Drop columns | `df.drop(columns=["b"])` | `df.drop_columns(&["b"])?` |
| Rename column | `df.rename(columns={"a": "alpha"})` | `df.rename("a", "alpha")?` |
| Add column | `df["c"] = series` | `df.add_column(Box::new(Series::new("c", vec![...])))?` |
| Remove column | `df.pop("b")` | `df.remove_column("b")?` |
| Access column (type-erased) | `df["age"]` | `df.column("age")?` |
| Access column (typed) | — | `df.column_typed::<f64>("age")?` |
| Head | `df.head(5)` | `df.head(5)` |
| Tail | `df.tail(5)` | `df.tail(5)` |
| Slice rows | `df.iloc[10:20]` | `df.slice(10, 10)` |
| Boolean filter | `df[mask]` | `df.filter(&mask)?` |
| Sort | `df.sort_values("col")` | `df.sort_by("col", true)?` |
| Sort descending | `df.sort_values("col", ascending=False)` | `df.sort_by("col", false)?` |
| Drop null rows | `df.dropna()` | `df.drop_nulls()?` |
| Drop nulls (subset) | `df.dropna(subset=["a"])` | `df.drop_nulls_subset(&["a"])?` |
| Null counts | `df.isnull().sum()` | `df.null_count_per_column()` |
| GroupBy + sum | `df.groupby("g").sum()` | `df.groupby(&["g"])?.sum()?` |
| GroupBy + mean | `df.groupby("g").mean()` | `df.groupby(&["g"])?.mean()?` |
| GroupBy + count | `df.groupby("g").count()` | `df.groupby(&["g"])?.count()?` |
| GroupBy + min/max | `df.groupby("g").min()` | `df.groupby(&["g"])?.min()?` |
| GroupBy + agg one col | `df.groupby("g")["v"].sum()` | `df.groupby(&["g"])?.agg("v", AggFunc::Sum)?` |
| Inner join | `pd.merge(l, r, on="k")` | `left.join(&right, &["k"], JoinType::Inner)?` |
| Left join | `pd.merge(l, r, on="k", how="left")` | `left.join(&right, &["k"], JoinType::Left)?` |
| Right join | `pd.merge(l, r, on="k", how="right")` | `left.join(&right, &["k"], JoinType::Right)?` |
| Outer join | `pd.merge(l, r, on="k", how="outer")` | `left.join(&right, &["k"], JoinType::Outer)?` |
| Join on different keys | `pd.merge(l, r, left_on="a", right_on="b")` | `left.join_on(&right, &["a"], &["b"], JoinType::Inner)?` |
| Pivot | `df.pivot_table(index=..., columns=..., values=..., aggfunc=...)` | `df.pivot(&["idx"], "col", "val", AggFunc::Sum)?` |
| Read CSV | `pd.read_csv("f.csv")` | `read_csv_path("f.csv")?` |
| Read CSV (from reader) | — | `read_csv(reader)?` |
| Read CSV (builder) | `pd.read_csv("f.csv", sep="\t")` | `CsvReaderBuilder::new().delimiter(b'\t').read(reader)?` |
| Write CSV | `df.to_csv("f.csv")` | `write_csv(File::create("f.csv")?, &df)?` |
| Read JSON | `pd.read_json("f.json")` | `read_json_path("f.json")?` |
| Write JSON | `df.to_json("f.json")` | `write_json(File::create("f.json")?, &df)?` |
| Read Parquet | `pd.read_parquet("f.parquet")` | `read_parquet("f.parquet")?` |
| Write Parquet | `df.to_parquet("f.parquet")` | `write_parquet(&df, "f.parquet")?` |
| Series sum | `s.sum()` | `s.sum()` |
| Series mean | `s.mean()` | `s.mean()` |
| Series min / max | `s.min()` / `s.max()` | `s.min()` / `s.max()` |
| Series std / var | `s.std()` / `s.var()` | `s.std()` / `s.var()` |
| Series median | `s.median()` | `s.median()` |
| Series unique | `s.unique()` | `s.unique()` |
| Series sort | `s.sort_values()` | `s.sort(true)` |
| Series argsort | `s.argsort()` | `s.argsort(true)` |
| Series apply | `s.apply(fn)` | `s.apply(\|v\| ...)` |
| Series map type | `s.astype(float)` | `s.map(\|v\| v as f64)` |
| SQL on DataFrames | — | `sql("SELECT * FROM t WHERE x > 1", &ctx)?` |

---

## Code Examples

### Creating a DataFrame

**Pandas**

```python
df = pd.DataFrame({
    "name": ["Alice", "Bob", "Carol"],
    "age":  [30, 25, 35],
    "score": [9.1, 8.5, 7.8],
})
```

**Scivex**

```rust
use scivex_frame::prelude::*;
use scivex_frame::series::string::StringSeries;

let df = DataFrame::builder()
    .add_boxed(Box::new(StringSeries::from_strs(
        "name", &["Alice", "Bob", "Carol"],
    )))
    .add_column("age", vec![30_i32, 25, 35])
    .add_column("score", vec![9.1_f64, 8.5, 7.8])
    .build()?;

assert_eq!(df.shape(), (3, 3));
```

Note: string columns use `StringSeries`, which is added via `add_boxed`.
Numeric columns use the generic `add_column` method.

---

### Accessing and Selecting Columns

**Pandas**

```python
ages = df["age"]
subset = df[["name", "score"]]
```

**Scivex**

```rust
// Type-erased access (returns &dyn AnySeries):
let ages = df.column("age")?;

// Typed access (returns &Series<i32>):
let ages_typed = df.column_typed::<i32>("age")?;
assert_eq!(ages_typed.get(0), Some(30));

// Select multiple columns into a new DataFrame:
let subset = df.select(&["name", "score"])?;
assert_eq!(subset.ncols(), 2);
```

---

### Filtering Rows

**Pandas**

```python
adults = df[df["age"] >= 30]
```

**Scivex**

```rust
let ages = df.column_typed::<i32>("age")?;
let mask: Vec<bool> = ages.as_slice().iter().map(|&a| a >= 30).collect();
let adults = df.filter(&mask)?;
assert_eq!(adults.nrows(), 2); // Alice (30), Carol (35)
```

Scivex separates mask construction from filtering.  You build the boolean mask
yourself, then pass it to `DataFrame::filter`.  This is explicit but gives you
full control over the predicate logic.

---

### Sorting

**Pandas**

```python
sorted_df = df.sort_values("age")
sorted_desc = df.sort_values("age", ascending=False)
```

**Scivex**

```rust
let sorted_df = df.sort_by("age", true)?;   // ascending
let sorted_desc = df.sort_by("age", false)?; // descending
```

---

### GroupBy and Aggregation

**Pandas**

```python
df = pd.DataFrame({
    "city": ["NYC", "LA", "NYC", "LA"],
    "sales": [100.0, 200.0, 150.0, 250.0],
})

grouped = df.groupby("city").sum()
mean_df = df.groupby("city").mean()
per_col = df.groupby("city")["sales"].sum()
```

**Scivex**

```rust
let df = DataFrame::builder()
    .add_boxed(Box::new(StringSeries::from_strs(
        "city", &["NYC", "LA", "NYC", "LA"],
    )))
    .add_column("sales", vec![100.0_f64, 200.0, 150.0, 250.0])
    .build()?;

// Aggregate all numeric columns:
let summed = df.groupby(&["city"])?.sum()?;
let means  = df.groupby(&["city"])?.mean()?;

// Aggregate a single column:
let sales_sum = df.groupby(&["city"])?.agg("sales", AggFunc::Sum)?;

// Available: Sum, Mean, Min, Max, Count, First, Last
let counts = df.groupby(&["city"])?.count()?;
```

The `GroupBy` type supports multi-column grouping:

```rust
let grouped = df.groupby(&["city", "region"])?.sum()?;
```

---

### Joins

**Pandas**

```python
result = pd.merge(left, right, on="key", how="inner")
result = pd.merge(left, right, left_on="lk", right_on="rk", how="left")
```

**Scivex**

```rust
// Same key name in both DataFrames:
let result = left.join(&right, &["key"], JoinType::Inner)?;

// Different key names:
let result = left.join_on(&right, &["lk"], &["rk"], JoinType::Left)?;

// Multi-key join:
let result = left.join(&right, &["k1", "k2"], JoinType::Outer)?;
```

Join types: `JoinType::Inner`, `JoinType::Left`, `JoinType::Right`,
`JoinType::Outer`.

When both DataFrames have a non-key column with the same name, Scivex
automatically appends `_left` and `_right` suffixes (similar to Pandas'
`suffixes` parameter).

---

### Handling Missing Data

**Pandas**

```python
df.dropna()
df.dropna(subset=["col_a"])
df.isnull().sum()
```

**Scivex**

```rust
let clean = df.drop_nulls()?;
let clean = df.drop_nulls_subset(&["col_a"])?;

// Null counts per column:
let counts: Vec<(&str, usize)> = df.null_count_per_column();

// Null tracking on a Series:
let s = Series::with_nulls("x", vec![1.0, 0.0, 3.0], vec![false, true, false])?;
assert_eq!(s.null_count(), 1);
assert_eq!(s.get(1), None); // null element returns None
```

---

### Series Operations

**Pandas**

```python
s = pd.Series([1.0, 2.0, 3.0, 4.0], name="x")
s.sum()       # 10.0
s.mean()      # 2.5
s.std()       # population std
s.median()    # 2.5
s.min()       # 1.0
s.unique()
s.sort_values()
s.apply(lambda x: x * 2)
```

**Scivex**

```rust
let s = Series::new("x", vec![1.0_f64, 2.0, 3.0, 4.0]);

s.sum();            // 10.0
s.mean();           // 2.5
s.std();            // population standard deviation
s.var();            // population variance
s.median();         // 2.5
s.min();            // Some(1.0)
s.max();            // Some(4.0)
s.product();        // 24.0
s.unique();         // Series with deduplicated values
s.sort(true);       // sorted ascending
s.argsort(false);   // indices for descending sort
s.apply(|v| v * 2.0);  // element-wise transform, same type
s.map(|v| v as i32);   // element-wise transform, different type
```

---

### Reading and Writing CSV

**Pandas**

```python
df = pd.read_csv("data.csv")
df = pd.read_csv("data.tsv", sep="\t", skiprows=2, nrows=100)
df.to_csv("out.csv", index=False)
```

**Scivex**

```rust
use scivex_io::prelude::*;
use std::fs::File;

// Quick read:
let df = read_csv_path("data.csv")?;

// Builder with options:
let df = CsvReaderBuilder::new()
    .delimiter(b'\t')
    .skip_rows(2)
    .max_rows(100)
    .has_header(true)
    .read(File::open("data.tsv")?)?;

// Write:
write_csv(File::create("out.csv")?, &df)?;

// Builder with options:
CsvWriterBuilder::new()
    .delimiter(b'\t')
    .write_header(true)
    .write(&mut File::create("out.tsv")?, &df)?;
```

---

### Reading and Writing JSON

**Pandas**

```python
df = pd.read_json("data.json")
df.to_json("out.json")
```

**Scivex**

```rust
let df = read_json_path("data.json")?;
write_json(File::create("out.json")?, &df)?;
```

---

### Reading and Writing Parquet

**Pandas**

```python
df = pd.read_parquet("data.parquet")
df.to_parquet("out.parquet")
```

**Scivex**

```rust
let df = read_parquet("data.parquet")?;
write_parquet(&df, "out.parquet")?;
```

---

### SQL Queries on DataFrames

Scivex includes a built-in SQL engine that operates directly on DataFrames,
with no external database required.

```rust
use scivex_frame::prelude::*;

let df = DataFrame::builder()
    .add_column("x", vec![1_i32, 2, 3, 4])
    .add_column("y", vec![10.0_f64, 20.0, 30.0, 40.0])
    .build()?;

let mut ctx = SqlContext::new();
ctx.register("data", &df);

let result = sql("SELECT x, y FROM data WHERE x > 2", &ctx)?;
```

---

## Key Differences

### 1. Typed Columns

Pandas columns are dynamically typed (backed by NumPy arrays that default to
`object` dtype).  In Scivex, every column has a concrete Rust type known at
compile time.  `Series<T>` is generic over any type implementing the `Scalar`
trait.

```rust
// The compiler knows this column holds i32 values:
let s = Series::new("age", vec![30_i32, 25, 35]);

// Downcast from a type-erased column:
let typed: &Series<i32> = df.column_typed::<i32>("age")?;
```

If you request the wrong type, `column_typed` returns a `FrameError::TypeMismatch`
error rather than silently producing garbage.

### 2. No Implicit Index

Pandas DataFrames always have an index (either default `RangeIndex` or a
user-specified one).  Scivex has no implicit row index.  If you need an index
column, add it explicitly:

```rust
let df = DataFrame::builder()
    .add_column("id", vec![0_i64, 1, 2])
    .add_column("value", vec![10.0_f64, 20.0, 30.0])
    .build()?;
```

For hierarchical indexing, Scivex provides `MultiIndex` as a separate type.

### 3. Ownership and Borrowing

Pandas operations return views or copies depending on context (the notorious
copy-vs-view ambiguity).  In Scivex, the rules are clear and enforced by the
Rust compiler:

- Methods like `head`, `tail`, `filter`, `sort_by`, `select`, `join` return a
  **new** `DataFrame` (owned data).
- `column` and `column_typed` return **borrowed references** (`&dyn AnySeries`
  or `&Series<T>`).
- Mutation methods like `rename`, `add_column`, `remove_column` take `&mut self`.

There is never ambiguity about whether you are looking at a copy or a view.

### 4. Explicit Error Handling

Pandas typically raises exceptions.  Scivex returns `Result<T, FrameError>`.
Common error variants include:

- `FrameError::ColumnNotFound` -- column name does not exist
- `FrameError::TypeMismatch` -- wrong type in `column_typed`
- `FrameError::RowCountMismatch` -- mask or column length does not match
- `FrameError::DuplicateColumnName` -- adding a column that already exists

Use the `?` operator for concise propagation:

```rust
let sub = df.select(&["a", "b"])?;
let sorted = sub.sort_by("a", true)?;
```

### 5. String Columns are a Separate Type

Rust's `String` does not implement the `Scalar` trait (which requires `Copy`),
so string data uses `StringSeries` rather than `Series<String>`:

```rust
use scivex_frame::series::string::StringSeries;

let names = StringSeries::from_strs("name", &["Alice", "Bob"]);
// Add to a DataFrame via add_boxed:
let df = DataFrame::builder()
    .add_boxed(Box::new(names))
    .add_column("age", vec![30_i32, 25])
    .build()?;
```

### 6. Filtering is Two Steps

Pandas combines mask creation and row selection in a single expression
(`df[df["x"] > 5]`).  Scivex separates them:

```rust
// Step 1: build the mask
let col = df.column_typed::<f64>("x")?;
let mask: Vec<bool> = col.as_slice().iter().map(|&v| v > 5.0).collect();

// Step 2: apply it
let filtered = df.filter(&mask)?;
```

This is more verbose but avoids hidden copies and makes the predicate logic
composable (you can combine masks with standard boolean operations before
filtering).

### 7. Null Handling

Pandas uses `NaN` for float nulls and a separate `pd.NA` for other types.
Scivex uses a boolean null mask alongside the data vector.  Any element type
can have nulls:

```rust
let s = Series::with_nulls(
    "x",
    vec![1_i32, 0, 3],
    vec![false, true, false],  // second element is null
)?;
assert_eq!(s.get(1), None);
assert_eq!(s.null_count(), 1);
```

### 8. Lazy Evaluation (Optional)

Scivex provides `LazyFrame` for deferred, optimizable query plans -- similar
to Polars' lazy API.  For users coming from Pandas, the eager `DataFrame` API
shown throughout this guide is the natural starting point.
