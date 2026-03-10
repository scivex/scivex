# scivex-frame

Column-oriented DataFrames for Scivex. Provides typed Series, type-erased
DataFrames, group-by aggregation, joins, pivot tables, and window functions.

## Highlights

- **DataFrame** — Type-erased columnar table with mixed column types
- **Series<T>** — Typed column with null tracking
- **StringSeries** — String column with case, contains, split operations
- **CategoricalSeries** — Dictionary-encoded categorical data
- **Joins** — Inner, Left, Right, Outer joins on one or more keys
- **GroupBy** — Aggregation (sum, mean, min, max, count, first, last)
- **Pivot/Melt** — Long-to-wide and wide-to-long reshaping
- **Rolling windows** — mean, sum, min, max, median, std, var, EWM
- **Filtering** — Boolean mask and expression-based row filtering

## Usage

```rust
use scivex_frame::prelude::*;

let df = DataFrameBuilder::new()
    .add_column(Series::new("city", vec!["NYC", "LA", "NYC", "LA"]))
    .add_column(Series::new("sales", vec![100.0f64, 200.0, 150.0, 250.0]))
    .build()
    .unwrap();

// Group and aggregate
let summary = df.group_by(&["city"]).unwrap().mean();

// Join two DataFrames
let merged = left.join(&right, &["id"], JoinType::Inner).unwrap();

// Rolling window
let series: &Series<f64> = df.column_typed("sales").unwrap();
let rolling_avg = series.rolling_mean(3);
```

## License

MIT
