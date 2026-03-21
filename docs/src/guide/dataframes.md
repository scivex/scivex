# DataFrames

This guide covers the `scivex-frame` and `scivex-io` crates: creating, querying,
transforming, and persisting tabular data.

## Creating DataFrames

### From columns with the builder

The most common way to create a `DataFrame` is the builder pattern. Each
`add_column` call appends a typed column:

```rust
use scivex_frame::prelude::*;

let df = DataFrame::builder()
    .add_column("name", vec![1_i32, 2, 3])
    .add_column("score", vec![9.5_f64, 8.0, 7.5])
    .build()
    .unwrap();

assert_eq!(df.shape(), (3, 2)); // (rows, columns)
```

All columns must have the same length, and column names must be unique.
Violations return a `FrameError`.

### From pre-built Series

You can construct `Series` objects independently and combine them:

```rust
use scivex_frame::prelude::*;

let ids = Series::new("id", vec![1_i64, 2, 3]);
let names = StringSeries::from_strs("name", &["Alice", "Bob", "Carol"]);

let df = DataFrame::new(vec![
    Box::new(ids),
    Box::new(names),
])
.unwrap();
```

`DataFrame::from_series` is an alias for `DataFrame::new`.

### Mixed-type columns with the builder

To add a `StringSeries` or `CategoricalSeries` alongside numeric columns, use
`add_boxed`:

```rust
use scivex_frame::prelude::*;

let names: Box<dyn AnySeries> =
    Box::new(StringSeries::from_strs("city", &["NYC", "LA", "NYC"]));

let df = DataFrame::builder()
    .add_boxed(names)
    .add_column("pop", vec![8_000_000_i64, 4_000_000, 8_000_000])
    .build()
    .unwrap();
```

### Empty DataFrames

```rust
use scivex_frame::prelude::*;

let df = DataFrame::empty();
assert!(df.is_empty());
assert_eq!(df.shape(), (0, 0));
```

---

## Series Types and Operations

A `Series<T>` is a named, typed column with optional null tracking. The
supported scalar types are `f64`, `f32`, `i64`, `i32`, `i16`, `i8`, `u64`,
`u32`, `u16`, `u8`, and `bool`. Non-numeric text is stored in `StringSeries`.

### Creating a Series

```rust
use scivex_frame::prelude::*;

// From a Vec
let s = Series::new("values", vec![10.0_f64, 20.0, 30.0]);

// From a slice
let s = Series::from_slice("values", &[10.0_f64, 20.0, 30.0]);

// String series
let s = StringSeries::from_strs("names", &["Alice", "Bob"]);
```

### Accessing elements

```rust
use scivex_frame::prelude::*;

let s = Series::new("x", vec![1_i32, 2, 3]);
assert_eq!(s.get(0), Some(1));
assert_eq!(s.get(99), None); // out of bounds returns None
assert_eq!(s.len(), 3);
assert_eq!(s.as_slice(), &[1, 2, 3]);
```

### Numeric aggregations

All aggregations skip null values automatically.

```rust
use scivex_frame::prelude::*;

let s = Series::new("x", vec![2.0_f64, 4.0, 6.0, 8.0]);

assert_eq!(s.sum(), 20.0);
assert_eq!(s.product(), 384.0);
assert_eq!(s.min(), Some(2.0));
assert_eq!(s.max(), Some(8.0));
assert_eq!(s.mean(), 5.0);
assert_eq!(s.median(), 5.0);

// variance and standard deviation (population)
let var = s.var();
let std = s.std();
```

Integer series support `sum`, `product`, `min`, and `max`. Float-specific
methods (`mean`, `var`, `std`, `median`) require `f32` or `f64`.

### Element-wise transforms

```rust
use scivex_frame::prelude::*;

let s = Series::new("x", vec![1_i32, 2, 3]);

// apply: same type out
let doubled = s.apply(|v| v * 2);
assert_eq!(doubled.as_slice(), &[2, 4, 6]);

// map: different type out
let floats: Series<f64> = s.map(|v| v as f64 * 1.5);
```

### Mutating a Series

```rust
use scivex_frame::prelude::*;

let mut s = Series::new("x", vec![1_i32, 2, 3]);
s.set(1, 20).unwrap();
s.push(4);
s.rename("y");
assert_eq!(s.name(), "y");
```

---

## Selecting Columns and Filtering Rows

### Selecting columns

```rust
use scivex_frame::prelude::*;

let df = DataFrame::builder()
    .add_column("a", vec![1_i32, 2, 3])
    .add_column("b", vec![4_i32, 5, 6])
    .add_column("c", vec![7_i32, 8, 9])
    .build()
    .unwrap();

// Pick specific columns (order is preserved as given)
let sub = df.select(&["c", "a"]).unwrap();
assert_eq!(sub.column_names(), vec!["c", "a"]);
```

### Accessing a single column

```rust
use scivex_frame::prelude::*;

let df = DataFrame::builder()
    .add_column("age", vec![25_i32, 30, 35])
    .build()
    .unwrap();

// Type-erased access
let col = df.column("age").unwrap();
assert_eq!(col.len(), 3);

// Typed access (downcast)
let ages: &Series<i32> = df.column_typed::<i32>("age").unwrap();
assert_eq!(ages.get(0), Some(25));
```

### Filtering rows with a boolean mask

```rust
use scivex_frame::prelude::*;

let df = DataFrame::builder()
    .add_column("x", vec![10_i32, 20, 30, 40])
    .build()
    .unwrap();

let mask = vec![true, false, true, false];
let filtered = df.filter(&mask).unwrap();
assert_eq!(filtered.nrows(), 2);
```

### head, tail, and slice

```rust
use scivex_frame::prelude::*;

let df = DataFrame::builder()
    .add_column("x", vec![10_i32, 20, 30, 40])
    .build()
    .unwrap();

let top2 = df.head(2);       // first 2 rows
let bottom2 = df.tail(2);    // last 2 rows
let middle = df.slice(1, 2); // offset=1, length=2
```

### Inspecting the DataFrame

```rust
use scivex_frame::prelude::*;

let df = DataFrame::builder()
    .add_column("x", vec![1_i32, 2])
    .add_column("y", vec![3.0_f64, 4.0])
    .build()
    .unwrap();

assert_eq!(df.nrows(), 2);
assert_eq!(df.ncols(), 2);
assert_eq!(df.shape(), (2, 2));
assert_eq!(df.column_names(), vec!["x", "y"]);
assert_eq!(df.dtypes(), vec![DType::I32, DType::F64]);
```

`DataFrame` implements `Display`, so `println!("{df}")` prints a formatted
table with automatic truncation for large frames (shows the first and last
10 rows when the total exceeds 20).

---

## Adding and Removing Columns

### Adding a column

```rust
use scivex_frame::prelude::*;

let mut df = DataFrame::builder()
    .add_column("x", vec![1_i32, 2, 3])
    .build()
    .unwrap();

df.add_column(Box::new(Series::new("y", vec![10_i32, 20, 30]))).unwrap();
assert_eq!(df.ncols(), 2);
```

The new column must match the existing row count. Duplicate names are rejected.

### Removing a column

```rust
use scivex_frame::prelude::*;

let mut df = DataFrame::builder()
    .add_column("a", vec![1_i32, 2])
    .add_column("b", vec![3_i32, 4])
    .build()
    .unwrap();

let removed = df.remove_column("b").unwrap();
assert_eq!(removed.name(), "b");
assert_eq!(df.ncols(), 1);
```

### Dropping columns (non-mutating)

```rust
use scivex_frame::prelude::*;

let df = DataFrame::builder()
    .add_column("a", vec![1_i32, 2])
    .add_column("b", vec![3_i32, 4])
    .add_column("c", vec![5_i32, 6])
    .build()
    .unwrap();

let smaller = df.drop_columns(&["b"]).unwrap();
assert_eq!(smaller.column_names(), vec!["a", "c"]);
```

### Renaming a column

```rust
use scivex_frame::prelude::*;

let mut df = DataFrame::builder()
    .add_column("old_name", vec![1_i32, 2])
    .build()
    .unwrap();

df.rename("old_name", "new_name").unwrap();
assert_eq!(df.column_names(), vec!["new_name"]);
```

---

## Sorting

### Sort by a single column

```rust
use scivex_frame::prelude::*;

let df = DataFrame::builder()
    .add_column("name", vec![3_i32, 1, 2])
    .add_column("score", vec![30.0_f64, 10.0, 20.0])
    .build()
    .unwrap();

let asc = df.sort_by("name", true).unwrap();   // ascending
let desc = df.sort_by("name", false).unwrap();  // descending

let col = asc.column_typed::<i32>("name").unwrap();
assert_eq!(col.as_slice(), &[1, 2, 3]);
```

Sorting reorders all columns consistently. It works for both numeric and
string columns.

---

## GroupBy and Aggregation

### Basic groupby

```rust
use scivex_frame::prelude::*;

let df = DataFrame::new(vec![
    Box::new(StringSeries::from_strs("city", &["NYC", "LA", "NYC", "LA", "NYC"])),
    Box::new(Series::new("sales", vec![100.0_f64, 200.0, 150.0, 250.0, 300.0])),
    Box::new(Series::new("units", vec![10_i32, 20, 15, 25, 30])),
])
.unwrap();

let grouped = df.groupby(&["city"]).unwrap();
assert_eq!(grouped.n_groups(), 2);
```

### Convenience aggregation methods

Each method aggregates all non-group numeric columns:

```rust
use scivex_frame::prelude::*;

# let df = DataFrame::new(vec![
#     Box::new(StringSeries::from_strs("city", &["NYC", "LA", "NYC", "LA", "NYC"])),
#     Box::new(Series::new("sales", vec![100.0_f64, 200.0, 150.0, 250.0, 300.0])),
# ]).unwrap();
let sums   = df.groupby(&["city"]).unwrap().sum().unwrap();
let means  = df.groupby(&["city"]).unwrap().mean().unwrap();
let mins   = df.groupby(&["city"]).unwrap().min().unwrap();
let maxes  = df.groupby(&["city"]).unwrap().max().unwrap();
let counts = df.groupby(&["city"]).unwrap().count().unwrap();
let firsts = df.groupby(&["city"]).unwrap().first().unwrap();
let lasts  = df.groupby(&["city"]).unwrap().last().unwrap();
```

### Aggregating a specific column

```rust
use scivex_frame::prelude::*;

# let df = DataFrame::new(vec![
#     Box::new(StringSeries::from_strs("city", &["NYC", "LA", "NYC"])),
#     Box::new(Series::new("sales", vec![100.0_f64, 200.0, 150.0])),
# ]).unwrap();
let result = df.groupby(&["city"]).unwrap().agg("sales", AggFunc::Mean).unwrap();
```

### Multi-column groupby

```rust
use scivex_frame::prelude::*;

let df = DataFrame::new(vec![
    Box::new(StringSeries::from_strs("region", &["East", "East", "West", "West"])),
    Box::new(StringSeries::from_strs("product", &["A", "B", "A", "B"])),
    Box::new(Series::new("revenue", vec![100_i32, 200, 300, 400])),
])
.unwrap();

let result = df.groupby(&["region", "product"]).unwrap().sum().unwrap();
assert_eq!(result.nrows(), 4);
```

---

## Joins

Four join types are supported: `Inner`, `Left`, `Right`, and `Outer`.

### Inner join

```rust
use scivex_frame::prelude::*;

let left = DataFrame::new(vec![
    Box::new(StringSeries::from_strs("key", &["a", "b", "c", "d"])),
    Box::new(Series::new("lval", vec![1_i32, 2, 3, 4])),
])
.unwrap();

let right = DataFrame::new(vec![
    Box::new(StringSeries::from_strs("key", &["b", "c", "c", "e"])),
    Box::new(Series::new("rval", vec![20_i32, 30, 31, 50])),
])
.unwrap();

let result = left.join(&right, &["key"], JoinType::Inner).unwrap();
// b matches once, c matches twice => 3 result rows
assert_eq!(result.nrows(), 3);
```

### Left, Right, and Outer joins

```rust
use scivex_frame::prelude::*;

# let left = DataFrame::new(vec![
#     Box::new(StringSeries::from_strs("key", &["a", "b", "c"])),
#     Box::new(Series::new("v", vec![1_i32, 2, 3])),
# ]).unwrap();
# let right = DataFrame::new(vec![
#     Box::new(StringSeries::from_strs("key", &["b", "d"])),
#     Box::new(Series::new("w", vec![20_i32, 40])),
# ]).unwrap();
// Left join: all left rows, nulls for non-matching right columns
let lj = left.join(&right, &["key"], JoinType::Left).unwrap();

// Right join: all right rows, nulls for non-matching left columns
let rj = left.join(&right, &["key"], JoinType::Right).unwrap();

// Outer join: all rows from both sides
let oj = left.join(&right, &["key"], JoinType::Outer).unwrap();
```

### Joining on differently-named columns

When the key columns have different names in the two frames, use `join_on`:

```rust
use scivex_frame::prelude::*;

let left = DataFrame::new(vec![
    Box::new(StringSeries::from_strs("left_id", &["a", "b"])),
    Box::new(Series::new("lval", vec![1_i32, 2])),
])
.unwrap();

let right = DataFrame::new(vec![
    Box::new(StringSeries::from_strs("right_id", &["b", "a"])),
    Box::new(Series::new("rval", vec![20_i32, 10])),
])
.unwrap();

let result = left
    .join_on(&right, &["left_id"], &["right_id"], JoinType::Inner)
    .unwrap();
assert_eq!(result.nrows(), 2);
```

### Duplicate column name handling

When both frames have a non-key column with the same name, Scivex
automatically appends `_left` and `_right` suffixes:

```rust
use scivex_frame::prelude::*;

let left = DataFrame::new(vec![
    Box::new(StringSeries::from_strs("key", &["a", "b"])),
    Box::new(Series::new("val", vec![1_i32, 2])),
]).unwrap();
let right = DataFrame::new(vec![
    Box::new(StringSeries::from_strs("key", &["a", "b"])),
    Box::new(Series::new("val", vec![10_i32, 20])),
]).unwrap();

let result = left.join(&right, &["key"], JoinType::Inner).unwrap();
let names = result.column_names();
assert!(names.contains(&"val_left"));
assert!(names.contains(&"val_right"));
```

---

## Lazy Evaluation

The lazy API builds a logical plan without executing anything until
`collect()` is called. This enables plan-level optimizations.

### Basic lazy workflow

```rust
use scivex_frame::prelude::*;
use scivex_frame::lazy::expr::{col, lit_f64, lit_i64};

let df = DataFrame::builder()
    .add_column("x", vec![1.0_f64, 2.0, 3.0, 4.0])
    .add_column("y", vec![10.0_f64, 20.0, 30.0, 40.0])
    .build()
    .unwrap();

let result = df
    .lazy()
    .filter(col("x").gt(lit_f64(1.5)))
    .select(&[col("y")])
    .collect()
    .unwrap();

assert_eq!(result.nrows(), 3);
assert_eq!(result.ncols(), 1);
```

### Available expression builders

| Function | Description |
|----------|-------------|
| `col("name")` | Reference a column |
| `lit_f64(val)` | Literal `f64` |
| `lit_i64(val)` | Literal `i64` |
| `lit_str(val)` | Literal string |
| `lit_bool(val)` | Literal boolean |

### Expression methods

Comparison: `eq`, `neq`, `lt`, `lt_eq`, `gt`, `gt_eq`

Logic: `and`, `or`, `not`

Arithmetic: `add`, `sub`, `mul`, `div`

Aggregation: `sum`, `mean`, `min`, `max`, `count`, `first`, `last`

Other: `alias`, `sort_asc`, `sort_desc`

### Chaining operations

```rust
use scivex_frame::prelude::*;
use scivex_frame::lazy::expr::{col, lit_i64};

let df = DataFrame::builder()
    .add_column("a", vec![5_i32, 3, 1, 4, 2])
    .add_column("b", vec![50.0_f64, 30.0, 10.0, 40.0, 20.0])
    .build()
    .unwrap();

let result = df
    .lazy()
    .filter(col("a").gt(lit_i64(1)))           // keep a > 1
    .sort("b", true)                            // sort by b ascending
    .limit(3)                                   // take first 3
    .collect()
    .unwrap();

assert_eq!(result.nrows(), 3);
```

### Computed columns

```rust
use scivex_frame::prelude::*;
use scivex_frame::lazy::expr::col;

let df = DataFrame::builder()
    .add_column("x", vec![1.0_f64, 2.0, 3.0])
    .add_column("y", vec![10.0_f64, 20.0, 30.0])
    .build()
    .unwrap();

let result = df
    .lazy()
    .select(&[col("x"), col("x").add(col("y")).alias("sum")])
    .collect()
    .unwrap();

let sum_col = result.column_typed::<f64>("sum").unwrap();
assert_eq!(sum_col.as_slice(), &[11.0, 22.0, 33.0]);
```

### Lazy groupby aggregation

```rust
use scivex_frame::prelude::*;
use scivex_frame::lazy::expr::col;

let df = DataFrame::builder()
    .add_column("group", vec![1_i32, 1, 2, 2, 2])
    .add_column("value", vec![10.0_f64, 20.0, 30.0, 40.0, 50.0])
    .build()
    .unwrap();

let result = df
    .lazy()
    .groupby_agg(&["group"], &[col("value").mean().alias("avg")])
    .collect()
    .unwrap();

assert_eq!(result.nrows(), 2);
```

---

## IO: Reading and Writing Data

The `scivex-io` crate provides readers and writers for multiple formats. Each
format is behind a feature flag.

### CSV

Enabled by default with the `csv` feature.

**Reading:**

```rust,no_run
use scivex_io::prelude::*;

// Quick one-liner from a file
let df = read_csv_path("data.csv").unwrap();

// From bytes / any Read source
let csv_data = "name,age\nAlice,30\nBob,25\n";
let df = read_csv(csv_data.as_bytes()).unwrap();

// With options
let df = CsvReaderBuilder::new()
    .delimiter(b'\t')
    .has_header(true)
    .skip_rows(1)
    .max_rows(Some(1000))
    .null_values(vec!["NA".into(), "".into()])
    .comment_char(Some(b'#'))
    .trim_whitespace(true)
    .infer_sample_size(500)
    .read(csv_data.as_bytes())
    .unwrap();
```

**Writing:**

```rust,no_run
use scivex_io::prelude::*;
use scivex_frame::prelude::*;

# let df = DataFrame::builder()
#     .add_column("x", vec![1_i32, 2]).build().unwrap();
// Quick one-liner
let mut buf = Vec::new();
write_csv(&mut buf, &df).unwrap();

// With options
CsvWriterBuilder::new()
    .delimiter(b'\t')
    .write_header(true)
    .null_representation("NA".into())
    .quote_style(QuoteStyle::Always)
    .write_path("output.tsv", &df)
    .unwrap();
```

**Streaming (chunked) reading** for large files:

```rust,no_run
use scivex_io::prelude::*;

let mut reader = CsvChunkReader::open("large_file.csv", 10_000).unwrap();
while let Some(chunk) = reader.next_chunk().unwrap() {
    println!("chunk with {} rows", chunk.nrows());
    // process each chunk without loading the entire file
}
```

### JSON

Requires the `json` feature.

**Reading:**

```rust,no_run
use scivex_io::prelude::*;

// Records orientation (array of objects) -- default
let json = r#"[{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]"#;
let df = read_json(json.as_bytes()).unwrap();

// Columns orientation (object of arrays)
let json = r#"{"name": ["Alice", "Bob"], "age": [30, 25]}"#;
let df = JsonReaderBuilder::new()
    .orientation(JsonOrientation::Columns)
    .read(json.as_bytes())
    .unwrap();

// From a file
let df = read_json_path("data.json").unwrap();
```

**Writing:**

```rust,no_run
use scivex_io::prelude::*;
use scivex_frame::prelude::*;

# let df = DataFrame::builder()
#     .add_column("x", vec![1_i32, 2]).build().unwrap();
let mut buf = Vec::new();
write_json(&mut buf, &df).unwrap();
```

### Parquet

Requires the `parquet` feature.

```rust,no_run
use scivex_io::prelude::*;
use scivex_frame::prelude::*;

// Reading
let df = read_parquet("data.parquet").unwrap();

// Reading with options
let df = ParquetReaderBuilder::new()
    .read_path("data.parquet")
    .unwrap();

// Writing (Snappy compression by default)
# let df = DataFrame::builder()
#     .add_column("x", vec![1_i32, 2]).build().unwrap();
write_parquet(&df, "output.parquet").unwrap();

// Writing with specific compression
ParquetWriterBuilder::new()
    .compression(ParquetCompression::None)
    .write_path(&df, "output.parquet")
    .unwrap();
```

---

## Null Handling

### Creating Series with nulls

A null mask is a `Vec<bool>` where `true` marks a null position:

```rust
use scivex_frame::prelude::*;

let s = Series::with_nulls(
    "x",
    vec![1.0_f64, 0.0, 3.0],    // 0.0 is a placeholder for the null
    vec![false, true, false],     // index 1 is null
)
.unwrap();

assert_eq!(s.null_count(), 1);
assert_eq!(s.get(1), None);  // null returns None
assert_eq!(s.count(), 2);    // non-null count
```

Nulls can also be appended incrementally:

```rust
use scivex_frame::prelude::*;

let mut s = Series::new("x", vec![1_i32, 2]);
s.push_null();
assert_eq!(s.len(), 3);
assert_eq!(s.null_count(), 1);
```

### Inspecting nulls

```rust
use scivex_frame::prelude::*;

let s = Series::with_nulls("x", vec![1_i32, 0, 3], vec![false, true, false]).unwrap();

assert!(s.is_null_at(1));
assert_eq!(s.is_null_mask(), vec![false, true, false]);
assert_eq!(s.is_not_null_mask(), vec![true, false, true]);
```

### Filling nulls

```rust
use scivex_frame::prelude::*;

let s = Series::with_nulls("x", vec![1.0_f64, 0.0, 3.0], vec![false, true, false]).unwrap();

// Fill with a constant
let filled = s.fill_null(999.0);

// Forward fill (last observation carried forward)
let ffill = s.fill_forward();

// Backward fill (next observation carried backward)
let bfill = s.fill_backward();

// Linear interpolation (Float types only)
let interp = s.interpolate();
```

### Dropping nulls

```rust
use scivex_frame::prelude::*;

let s = Series::with_nulls("x", vec![1_i32, 0, 3], vec![false, true, false]).unwrap();
let clean = s.drop_null();
assert_eq!(clean.len(), 2);
```

### DataFrame-level null handling

```rust
use scivex_frame::prelude::*;

let s1 = Series::with_nulls("a", vec![1_i32, 0, 3], vec![false, true, false]).unwrap();
let s2 = Series::new("b", vec![10_i32, 20, 30]);
let df = DataFrame::new(vec![Box::new(s1), Box::new(s2)]).unwrap();

// Drop rows where ANY column is null
let clean = df.drop_nulls().unwrap();
assert_eq!(clean.nrows(), 2);

// Drop rows where specific columns are null
let clean = df.drop_nulls_subset(&["a"]).unwrap();

// Inspect null counts per column
let counts = df.null_count_per_column();
// returns vec![("a", 1), ("b", 0)]
```

---

## String Operations

`StringSeries` provides text-specific methods that return either new series or
boolean masks suitable for filtering.

```rust
use scivex_frame::prelude::*;

let s = StringSeries::from_strs("text", &["  Hello World  ", "foo bar", "RUST"]);

// Case conversion
let upper = s.to_uppercase();  // ["  HELLO WORLD  ", "FOO BAR", "RUST"]
let lower = s.to_lowercase();  // ["  hello world  ", "foo bar", "rust"]

// Whitespace trimming
let trimmed = s.strip();       // ["Hello World", "foo bar", "RUST"]

// Search (returns Vec<bool> masks)
let mask = s.contains("bar");       // [false, true, false]
let mask = s.starts_with("  ");     // [true, false, false]
let mask = s.ends_with("ST");       // [false, false, true]

// Character lengths
let lens = s.len_chars();           // [15, 7, 4]

// Replace
let replaced = s.replace_all("o", "0");  // ["  Hell0 W0rld  ", "f00 bar", "RUST"]
```

### Regex operations (requires `regex` feature)

```rust,ignore
let mask = s.regex_contains(r"\d+").unwrap();
let extracted = s.regex_extract(r"(\d+)").unwrap();
```

---

## Rolling Window Operations

`Series<T: Float>` supports rolling window computations:

```rust
use scivex_frame::prelude::*;

let s = Series::new("price", vec![1.0_f64, 2.0, 3.0, 4.0, 5.0]);

let window = RollingWindow::new(3);  // window size of 3
let rolling_avg = s.rolling_mean(&window).unwrap();
// First 2 values are null (not enough data), then [2.0, 3.0, 4.0]
```

The `RollingWindow` builder supports:

```rust
use scivex_frame::prelude::*;

let window = RollingWindow::new(5)
    .min_periods(3)   // produce a result with at least 3 non-null values
    .center(true);    // center the window on the current element
```

---

## Performance Tips

### Use typed column access

When you know a column's type, use `column_typed::<T>` instead of the
type-erased `column`. This avoids dynamic dispatch on every element access:

```rust
use scivex_frame::prelude::*;

# let df = DataFrame::builder()
#     .add_column("x", vec![1.0_f64, 2.0]).build().unwrap();
// Preferred for hot loops
let col = df.column_typed::<f64>("x").unwrap();
for &val in col.as_slice() {
    // direct slice access, no vtable dispatch
}
```

### Use the lazy API for multi-step pipelines

The lazy API can optimize chains of filter/select/sort operations. Prefer
`df.lazy().filter(...).select(...).collect()` over eagerly materializing
intermediate DataFrames.

### Stream large CSV files

For CSV files that do not fit in memory, use `CsvChunkReader` to process
data in fixed-size chunks instead of loading the entire file at once.

### Use CategoricalSeries for groupby keys

When a string column has few unique values and is used repeatedly as a groupby
key, convert it to a `CategoricalSeries`. The groupby engine uses a fast path
with integer codes instead of string hashing.

### Avoid unnecessary clones

Methods like `select`, `drop_columns`, and `filter` return new DataFrames
with cloned columns. If you only need a subset of data, select the columns you
need before filtering to reduce the amount of data cloned.

### Parquet over CSV for large datasets

Parquet files are columnar, compressed, and encode type information. Reading
Parquet skips the type-inference step required by CSV and typically loads
faster for large datasets.
