# scivex-frame

DataFrames for Scivex. Column-oriented tabular data with rich operations
for data manipulation and analysis.

## Highlights

- **DataFrame** — Column-oriented storage with heterogeneous types
- **Series** — Typed 1-D arrays (f64, i64, String, Bool)
- **Joins** — Inner, left, right, outer, cross joins on any column
- **GroupBy** — Group and aggregate with sum, mean, min, max, count
- **Pivot** — Pivot tables and cross-tabulations
- **Rolling windows** — Rolling mean, sum, std, min, max with configurable window size
- **String ops** — contains, starts_with, replace, split, regex matching
- **LazyFrame** — Deferred execution with query optimization
- **Sorting** — Multi-column sort with ascending/descending per column
- **Filtering** — Boolean masks, expression-based filtering

## Usage

```rust
use scivex_frame::prelude::*;

let df = DataFrameBuilder::new()
    .add_column(Series::new("city", vec!["NYC", "LA", "NYC", "LA"]))
    .add_column(Series::new("sales", vec![100.0, 200.0, 150.0, 250.0]))
    .build()
    .unwrap();

let grouped = df.group_by(&["city"]).unwrap();
let totals = grouped.sum().unwrap();
```

## License

MIT
