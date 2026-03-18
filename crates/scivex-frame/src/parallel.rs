//! Parallel DataFrame operations using Rayon.
//!
//! All public functions are feature-gated behind the `parallel` feature flag.
//! They mirror sequential operations but distribute work across threads.

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(feature = "parallel")]
use std::collections::HashMap;

#[cfg(feature = "parallel")]
use scivex_core::Scalar;

#[cfg(feature = "parallel")]
use crate::dataframe::DataFrame;
#[cfg(feature = "parallel")]
use crate::dtype::{DType, HasDType};
#[cfg(feature = "parallel")]
use crate::error::{FrameError, Result};
#[cfg(feature = "parallel")]
use crate::groupby::AggFunc;
#[cfg(feature = "parallel")]
use crate::series::string::StringSeries;
#[cfg(feature = "parallel")]
use crate::series::{AnySeries, Series};

// ---------------------------------------------------------------------------
// Parallel GroupBy Aggregation
// ---------------------------------------------------------------------------

/// Parallel groupby aggregation -- partitions groups across threads.
///
/// Builds groups sequentially (hash step), then computes per-group aggregates
/// in parallel using Rayon's `par_iter`.
#[cfg(feature = "parallel")]
pub fn par_groupby_agg(df: &DataFrame, by: &[&str], agg: AggFunc) -> Result<DataFrame> {
    // Validate group columns exist.
    for &col in by {
        df.column(col)?;
    }

    let nrows = df.nrows();

    // Sequential hash step: build composite key -> row indices.
    let mut map: HashMap<Vec<String>, Vec<usize>> = HashMap::new();
    let mut order: Vec<Vec<String>> = Vec::new();

    for row in 0..nrows {
        let key: Vec<String> = by
            .iter()
            .map(|&c| {
                df.column(c)
                    .expect("groupby column validated above")
                    .display_value(row)
            })
            .collect();
        if !map.contains_key(&key) {
            order.push(key.clone());
        }
        map.entry(key).or_default().push(row);
    }

    let groups: Vec<(Vec<String>, Vec<usize>)> = order
        .into_iter()
        .map(|k| {
            let indices = map.remove(&k).expect("key guaranteed by iteration order");
            (k, indices)
        })
        .collect();

    // Build group-key columns for the result.
    let group_columns: Vec<String> = by.iter().map(|s| (*s).to_string()).collect();
    let mut result_columns: Vec<Box<dyn AnySeries>> = group_columns
        .iter()
        .enumerate()
        .map(|(ki, name)| {
            let data: Vec<String> = groups.iter().map(|(keys, _)| keys[ki].clone()).collect();
            Box::new(StringSeries::new(name.clone(), data)) as Box<dyn AnySeries>
        })
        .collect();

    // Identify non-group columns to aggregate.
    let value_cols: Vec<&dyn AnySeries> = df
        .columns()
        .iter()
        .filter(|c| !group_columns.contains(&c.name().to_string()))
        .map(AsRef::as_ref)
        .collect();

    // Aggregate each value column, parallelizing across groups.
    for src_col in value_cols {
        match agg {
            AggFunc::Count => {
                #[allow(clippy::cast_possible_wrap)]
                let counts: Vec<i64> = groups
                    .par_iter()
                    .map(|(_, indices)| indices.len() as i64)
                    .collect();
                result_columns.push(Box::new(Series::new(
                    format!("{}_count", src_col.name()),
                    counts,
                )));
            }
            AggFunc::First => {
                let indices: Vec<usize> = groups.par_iter().map(|(_, idx)| idx[0]).collect();
                result_columns.push(src_col.take_indices(&indices));
            }
            AggFunc::Last => {
                let indices: Vec<usize> = groups
                    .par_iter()
                    .map(|(_, idx)| *idx.last().expect("group indices guaranteed non-empty"))
                    .collect();
                result_columns.push(src_col.take_indices(&indices));
            }
            _ => {
                if let Ok(col) = par_aggregate_typed_column(src_col, &groups, agg) {
                    result_columns.push(col);
                }
            }
        }
    }

    DataFrame::new(result_columns)
}

/// Typed parallel aggregation dispatch for numeric columns.
#[cfg(feature = "parallel")]
fn par_aggregate_typed_column(
    col: &dyn AnySeries,
    groups: &[(Vec<String>, Vec<usize>)],
    func: AggFunc,
) -> Result<Box<dyn AnySeries>> {
    let dtype = col.dtype();

    macro_rules! dispatch_par_agg {
        ($ty:ty, $dtype_variant:ident) => {{
            if dtype == DType::$dtype_variant {
                let typed =
                    col.as_any()
                        .downcast_ref::<Series<$ty>>()
                        .ok_or(FrameError::TypeMismatch {
                            expected: stringify!($ty),
                            got: "unknown",
                        })?;
                return par_agg_typed(typed, groups, func);
            }
        }};
    }

    dispatch_par_agg!(f64, F64);
    dispatch_par_agg!(f32, F32);
    dispatch_par_agg!(i64, I64);
    dispatch_par_agg!(i32, I32);
    dispatch_par_agg!(i16, I16);
    dispatch_par_agg!(i8, I8);
    dispatch_par_agg!(u64, U64);
    dispatch_par_agg!(u32, U32);
    dispatch_par_agg!(u16, U16);
    dispatch_par_agg!(u8, U8);

    Err(FrameError::InvalidArgument {
        reason: "parallel aggregation not supported for this column type",
    })
}

/// Compute aggregation for a typed series in parallel across groups.
#[cfg(feature = "parallel")]
fn par_agg_typed<T: Scalar + HasDType + 'static>(
    col: &Series<T>,
    groups: &[(Vec<String>, Vec<usize>)],
    func: AggFunc,
) -> Result<Box<dyn AnySeries>> {
    let name = col.name().to_string();
    let slice = col.as_slice();

    let data: Vec<T> = match func {
        AggFunc::Sum => groups
            .par_iter()
            .map(|(_, indices)| indices.iter().fold(T::zero(), |acc, &i| acc + slice[i]))
            .collect(),
        AggFunc::Min => groups
            .par_iter()
            .map(|(_, indices)| {
                indices
                    .iter()
                    .map(|&i| slice[i])
                    .reduce(|a, b| if b < a { b } else { a })
                    .unwrap_or_else(T::zero)
            })
            .collect(),
        AggFunc::Max => groups
            .par_iter()
            .map(|(_, indices)| {
                indices
                    .iter()
                    .map(|&i| slice[i])
                    .reduce(|a, b| if b > a { b } else { a })
                    .unwrap_or_else(T::zero)
            })
            .collect(),
        AggFunc::Mean => groups
            .par_iter()
            .map(|(_, indices)| {
                let sum = indices.iter().fold(T::zero(), |acc, &i| acc + slice[i]);
                sum / T::from_usize(indices.len())
            })
            .collect(),
        AggFunc::Count | AggFunc::First | AggFunc::Last => {
            // Already handled in the caller.
            return Err(FrameError::InvalidArgument {
                reason: "Count/First/Last handled separately",
            });
        }
    };

    Ok(Box::new(Series::new(name, data)))
}

// ---------------------------------------------------------------------------
// Parallel Filter
// ---------------------------------------------------------------------------

/// Parallel predicate evaluation on rows.
///
/// Evaluates `predicate` on every element of the named `column` in parallel
/// (using `par_chunks`), then builds a filtered `DataFrame` containing only
/// matching rows.
#[cfg(feature = "parallel")]
pub fn par_filter(
    df: &DataFrame,
    column: &str,
    predicate: impl Fn(f64) -> bool + Sync,
) -> Result<DataFrame> {
    let col = df.column(column)?;
    let typed = col
        .as_any()
        .downcast_ref::<Series<f64>>()
        .ok_or(FrameError::TypeMismatch {
            expected: "Series<f64>",
            got: "other",
        })?;
    let slice = typed.as_slice();

    // Evaluate predicate in parallel chunks, producing a bool mask.
    let chunk_size = (slice.len() / rayon::current_num_threads()).max(1);
    let mask: Vec<bool> = slice
        .par_chunks(chunk_size)
        .flat_map_iter(|chunk| chunk.iter().map(|&v| predicate(v)))
        .collect();

    df.filter(&mask)
}

// ---------------------------------------------------------------------------
// Parallel Apply (map)
// ---------------------------------------------------------------------------

/// Apply a function to each element of a numeric (`f64`) column in parallel.
///
/// Returns a `Vec<f64>` with the mapped values.
#[cfg(feature = "parallel")]
pub fn par_apply(df: &DataFrame, column: &str, f: impl Fn(f64) -> f64 + Sync) -> Result<Vec<f64>> {
    let col = df.column(column)?;
    let typed = col
        .as_any()
        .downcast_ref::<Series<f64>>()
        .ok_or(FrameError::TypeMismatch {
            expected: "Series<f64>",
            got: "other",
        })?;
    let slice = typed.as_slice();

    let result: Vec<f64> = slice.par_iter().map(|&v| f(v)).collect();
    Ok(result)
}

// ---------------------------------------------------------------------------
// Parallel Sort
// ---------------------------------------------------------------------------

/// Parallel sort by column using Rayon's `par_sort_by`.
///
/// Builds sort indices in parallel, then reindexes all columns.
#[cfg(feature = "parallel")]
pub fn par_sort(df: &DataFrame, by: &str, ascending: bool) -> Result<DataFrame> {
    let col = df.column(by)?;
    let n = col.len();
    let mut indices: Vec<usize> = (0..n).collect();

    // Build sortable keys: try f64 parse, fall back to string comparison.
    let keys: Vec<String> = (0..n).map(|i| col.display_value(i)).collect();
    let parsed: Vec<Option<f64>> = keys.par_iter().map(|k| k.parse::<f64>().ok()).collect();

    indices.par_sort_by(|&a, &b| {
        let cmp = match (parsed[a], parsed[b]) {
            (Some(fa), Some(fb)) => fa.partial_cmp(&fb).unwrap_or(core::cmp::Ordering::Equal),
            _ => keys[a].cmp(&keys[b]),
        };
        if ascending { cmp } else { cmp.reverse() }
    });

    let cols = df
        .columns()
        .iter()
        .map(|c| c.take_indices(&indices))
        .collect();
    DataFrame::new(cols)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[cfg(feature = "parallel")]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use crate::series::string::StringSeries;

    fn sample_df() -> DataFrame {
        DataFrame::new(vec![
            Box::new(StringSeries::from_strs(
                "city",
                &["NYC", "LA", "NYC", "LA", "NYC"],
            )),
            Box::new(Series::new(
                "sales",
                vec![100.0_f64, 200.0, 150.0, 250.0, 300.0],
            )),
            Box::new(Series::new("units", vec![10_i32, 20, 15, 25, 30])),
        ])
        .unwrap()
    }

    #[test]
    fn test_par_groupby_sum() {
        let df = sample_df();
        let result = par_groupby_agg(&df, &["city"], AggFunc::Sum).unwrap();
        assert_eq!(result.nrows(), 2);

        let sales = result.column_typed::<f64>("sales").unwrap();
        // NYC: 100+150+300 = 550, LA: 200+250 = 450
        assert_eq!(sales.as_slice(), &[550.0, 450.0]);

        // Compare with sequential groupby.
        let seq = df.groupby(&["city"]).unwrap().sum().unwrap();
        let seq_sales = seq.column_typed::<f64>("sales").unwrap();
        assert_eq!(sales.as_slice(), seq_sales.as_slice());
    }

    #[test]
    fn test_par_filter() {
        let df = sample_df();
        let result = par_filter(&df, "sales", |v| v > 150.0).unwrap();
        assert_eq!(result.nrows(), 3);

        let sales = result.column_typed::<f64>("sales").unwrap();
        assert_eq!(sales.as_slice(), &[200.0, 250.0, 300.0]);

        // Compare with sequential filter.
        let seq_sales_col = df.column_typed::<f64>("sales").unwrap();
        let mask: Vec<bool> = seq_sales_col
            .as_slice()
            .iter()
            .map(|&v| v > 150.0)
            .collect();
        let seq = df.filter(&mask).unwrap();
        assert_eq!(result.nrows(), seq.nrows());
    }

    #[test]
    fn test_par_apply() {
        let df = sample_df();
        let result = par_apply(&df, "sales", |v| v * 2.0).unwrap();
        assert_eq!(result, vec![200.0, 400.0, 300.0, 500.0, 600.0]);
    }

    #[test]
    fn test_par_sort() {
        let df = sample_df();
        let sorted = par_sort(&df, "sales", true).unwrap();
        let sales = sorted.column_typed::<f64>("sales").unwrap();
        assert_eq!(sales.as_slice(), &[100.0, 150.0, 200.0, 250.0, 300.0]);

        // Compare with sequential sort.
        let seq = df.sort_by("sales", true).unwrap();
        let seq_sales = seq.column_typed::<f64>("sales").unwrap();
        assert_eq!(sales.as_slice(), seq_sales.as_slice());

        // Descending.
        let sorted_desc = par_sort(&df, "sales", false).unwrap();
        let desc_sales = sorted_desc.column_typed::<f64>("sales").unwrap();
        assert_eq!(desc_sales.as_slice(), &[300.0, 250.0, 200.0, 150.0, 100.0]);
    }
}
