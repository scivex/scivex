//! Group-by aggregation for [`DataFrame`].

use std::collections::HashMap;

use scivex_core::Scalar;

use crate::dataframe::DataFrame;
use crate::dtype::{DType, HasDType};
use crate::error::{FrameError, Result};
use crate::series::string::StringSeries;
use crate::series::{AnySeries, Series};

/// Aggregation function selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggFunc {
    Sum,
    Mean,
    Min,
    Max,
    Count,
    First,
    Last,
}

/// A grouped `DataFrame` ready for aggregation.
pub struct GroupBy<'a> {
    df: &'a DataFrame,
    group_columns: Vec<String>,
    /// Maps composite key → row indices in the original `DataFrame`.
    groups: Vec<(Vec<String>, Vec<usize>)>,
}

impl<'a> GroupBy<'a> {
    /// Group `df` by the given columns.
    pub(crate) fn new(df: &'a DataFrame, cols: &[&str]) -> Result<Self> {
        // Validate that all group columns exist.
        for &col in cols {
            df.column(col)?;
        }

        let group_columns: Vec<String> = cols.iter().map(|s| (*s).to_string()).collect();
        let nrows = df.nrows();

        // Build groups by hashing composite string keys.
        let mut map: HashMap<Vec<String>, Vec<usize>> = HashMap::new();
        let mut order: Vec<Vec<String>> = Vec::new();

        for row in 0..nrows {
            let key: Vec<String> = cols
                .iter()
                .map(|&c| df.column(c).unwrap().display_value(row))
                .collect();
            if !map.contains_key(&key) {
                order.push(key.clone());
            }
            map.entry(key).or_default().push(row);
        }

        let groups: Vec<(Vec<String>, Vec<usize>)> = order
            .into_iter()
            .map(|k| {
                let indices = map.remove(&k).unwrap();
                (k, indices)
            })
            .collect();

        Ok(Self {
            df,
            group_columns,
            groups,
        })
    }

    /// Number of groups.
    pub fn n_groups(&self) -> usize {
        self.groups.len()
    }

    // -- Convenience aggregation methods ------------------------------------

    /// Sum of all numeric columns per group.
    pub fn sum(&self) -> Result<DataFrame> {
        self.agg_all(AggFunc::Sum)
    }

    /// Mean of all float columns per group.
    pub fn mean(&self) -> Result<DataFrame> {
        self.agg_all(AggFunc::Mean)
    }

    /// Min of all numeric columns per group.
    pub fn min(&self) -> Result<DataFrame> {
        self.agg_all(AggFunc::Min)
    }

    /// Max of all numeric columns per group.
    pub fn max(&self) -> Result<DataFrame> {
        self.agg_all(AggFunc::Max)
    }

    /// Count of rows per group.
    pub fn count(&self) -> Result<DataFrame> {
        self.agg_all(AggFunc::Count)
    }

    /// First row per group.
    pub fn first(&self) -> Result<DataFrame> {
        self.agg_all(AggFunc::First)
    }

    /// Last row per group.
    pub fn last(&self) -> Result<DataFrame> {
        self.agg_all(AggFunc::Last)
    }

    /// Apply an aggregation function to a specific column.
    pub fn agg(&self, col: &str, func: AggFunc) -> Result<DataFrame> {
        self.df.column(col)?; // validate column exists

        let mut result_columns: Vec<Box<dyn AnySeries>> = self.build_key_columns();

        let src_col = self.df.column(col)?;
        let agg_col = self.aggregate_column(src_col, func)?;
        result_columns.push(agg_col);

        DataFrame::new(result_columns)
    }

    // -- Internal -----------------------------------------------------------

    /// Apply aggregation to all non-group columns.
    fn agg_all(&self, func: AggFunc) -> Result<DataFrame> {
        let mut result_columns: Vec<Box<dyn AnySeries>> = self.build_key_columns();

        for src_col in self.df.columns() {
            if self.group_columns.contains(&src_col.name().to_string()) {
                continue;
            }
            if let Ok(col) = self.aggregate_column(src_col.as_ref(), func) {
                result_columns.push(col);
            }
        }

        DataFrame::new(result_columns)
    }

    /// Build the group-key columns for the result `DataFrame`.
    fn build_key_columns(&self) -> Vec<Box<dyn AnySeries>> {
        self.group_columns
            .iter()
            .enumerate()
            .map(|(ki, name)| {
                let data: Vec<String> = self
                    .groups
                    .iter()
                    .map(|(keys, _)| keys[ki].clone())
                    .collect();
                Box::new(StringSeries::new(name.clone(), data)) as Box<dyn AnySeries>
            })
            .collect()
    }

    /// Aggregate a single column across all groups.
    fn aggregate_column(&self, col: &dyn AnySeries, func: AggFunc) -> Result<Box<dyn AnySeries>> {
        let dtype = col.dtype();

        // For Count, First, Last we can operate generically.
        match func {
            AggFunc::Count => {
                #[allow(clippy::cast_possible_wrap)]
                let counts: Vec<i64> = self
                    .groups
                    .iter()
                    .map(|(_, indices)| indices.len() as i64)
                    .collect();
                return Ok(Box::new(Series::new(
                    format!("{}_count", col.name()),
                    counts,
                )));
            }
            AggFunc::First => {
                let indices: Vec<usize> = self.groups.iter().map(|(_, idx)| idx[0]).collect();
                let result = col.take_indices(&indices);
                return Ok(result);
            }
            AggFunc::Last => {
                let indices: Vec<usize> = self
                    .groups
                    .iter()
                    .map(|(_, idx)| *idx.last().unwrap())
                    .collect();
                let result = col.take_indices(&indices);
                return Ok(result);
            }
            _ => {}
        }

        // Numeric aggregations need typed dispatch.
        macro_rules! dispatch_agg {
            ($ty:ty, $dtype_variant:ident) => {{
                if dtype == DType::$dtype_variant {
                    let typed = col.as_any().downcast_ref::<Series<$ty>>().ok_or(
                        FrameError::TypeMismatch {
                            expected: stringify!($ty),
                            got: "unknown",
                        },
                    )?;
                    return self.agg_typed(typed, func);
                }
            }};
        }

        dispatch_agg!(f64, F64);
        dispatch_agg!(f32, F32);
        dispatch_agg!(i64, I64);
        dispatch_agg!(i32, I32);
        dispatch_agg!(i16, I16);
        dispatch_agg!(i8, I8);
        dispatch_agg!(u64, U64);
        dispatch_agg!(u32, U32);
        dispatch_agg!(u16, U16);
        dispatch_agg!(u8, U8);

        Err(FrameError::InvalidArgument {
            reason: "aggregation not supported for this column type",
        })
    }

    /// Typed aggregation helper for Scalar types.
    fn agg_typed<T: Scalar + HasDType + 'static>(
        &self,
        col: &Series<T>,
        func: AggFunc,
    ) -> Result<Box<dyn AnySeries>> {
        let name = col.name().to_string();
        let slice = col.as_slice();

        match func {
            AggFunc::Sum => {
                let data: Vec<T> = self
                    .groups
                    .iter()
                    .map(|(_, indices)| indices.iter().fold(T::zero(), |acc, &i| acc + slice[i]))
                    .collect();
                Ok(Box::new(Series::new(name, data)))
            }
            AggFunc::Min => {
                let data: Vec<T> = self
                    .groups
                    .iter()
                    .map(|(_, indices)| {
                        indices
                            .iter()
                            .map(|&i| slice[i])
                            .reduce(|a, b| if b < a { b } else { a })
                            .unwrap_or_else(T::zero)
                    })
                    .collect();
                Ok(Box::new(Series::new(name, data)))
            }
            AggFunc::Max => {
                let data: Vec<T> = self
                    .groups
                    .iter()
                    .map(|(_, indices)| {
                        indices
                            .iter()
                            .map(|&i| slice[i])
                            .reduce(|a, b| if b > a { b } else { a })
                            .unwrap_or_else(T::zero)
                    })
                    .collect();
                Ok(Box::new(Series::new(name, data)))
            }
            AggFunc::Mean => {
                // Mean only makes sense for Float types. We'll compute as
                // sum / count using Scalar arithmetic (works for floats; for
                // integers it gives integer division which is acceptable MVP).
                let data: Vec<T> = self
                    .groups
                    .iter()
                    .map(|(_, indices)| {
                        let sum = indices.iter().fold(T::zero(), |acc, &i| acc + slice[i]);
                        sum / T::from_usize(indices.len())
                    })
                    .collect();
                Ok(Box::new(Series::new(name, data)))
            }
            AggFunc::Count | AggFunc::First | AggFunc::Last => {
                // Already handled in aggregate_column.
                unreachable!()
            }
        }
    }
}

// ---------------------------------------------------------------------------
// DataFrame extension
// ---------------------------------------------------------------------------

impl DataFrame {
    /// Group by one or more columns.
    pub fn groupby(&self, cols: &[&str]) -> Result<GroupBy<'_>> {
        GroupBy::new(self, cols)
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

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
    fn test_groupby_count() {
        let df = sample_df();
        let result = df.groupby(&["city"]).unwrap().count().unwrap();
        assert_eq!(result.nrows(), 2);
        // NYC = 3 rows, LA = 2 rows
        let counts = result.column_typed::<i64>("sales_count").unwrap();
        // NYC comes first (first occurrence order)
        assert_eq!(counts.as_slice(), &[3, 2]);
    }

    #[test]
    fn test_groupby_sum() {
        let df = sample_df();
        let result = df.groupby(&["city"]).unwrap().sum().unwrap();
        let sales = result.column_typed::<f64>("sales").unwrap();
        // NYC: 100+150+300 = 550, LA: 200+250 = 450
        assert_eq!(sales.as_slice(), &[550.0, 450.0]);
    }

    #[test]
    fn test_groupby_mean() {
        let df = sample_df();
        let result = df.groupby(&["city"]).unwrap().mean().unwrap();
        let sales = result.column_typed::<f64>("sales").unwrap();
        // NYC: 550/3 ≈ 183.33, LA: 450/2 = 225
        assert!((sales.get(0).unwrap() - 550.0 / 3.0).abs() < 1e-10);
        assert_eq!(sales.get(1).unwrap(), 225.0);
    }

    #[test]
    fn test_groupby_min_max() {
        let df = sample_df();
        let min_result = df.groupby(&["city"]).unwrap().min().unwrap();
        let max_result = df.groupby(&["city"]).unwrap().max().unwrap();
        let min_sales = min_result.column_typed::<f64>("sales").unwrap();
        let max_sales = max_result.column_typed::<f64>("sales").unwrap();
        // NYC min=100, max=300; LA min=200, max=250
        assert_eq!(min_sales.as_slice(), &[100.0, 200.0]);
        assert_eq!(max_sales.as_slice(), &[300.0, 250.0]);
    }

    #[test]
    fn test_groupby_first_last() {
        let df = sample_df();
        let first_result = df.groupby(&["city"]).unwrap().first().unwrap();
        let last_result = df.groupby(&["city"]).unwrap().last().unwrap();
        let first_sales = first_result.column_typed::<f64>("sales").unwrap();
        let last_sales = last_result.column_typed::<f64>("sales").unwrap();
        // NYC first=100, last=300; LA first=200, last=250
        assert_eq!(first_sales.as_slice(), &[100.0, 200.0]);
        assert_eq!(last_sales.as_slice(), &[300.0, 250.0]);
    }

    #[test]
    fn test_groupby_agg_single_column() {
        let df = sample_df();
        let result = df
            .groupby(&["city"])
            .unwrap()
            .agg("sales", AggFunc::Sum)
            .unwrap();
        assert_eq!(result.ncols(), 2); // city + sales
        let sales = result.column_typed::<f64>("sales").unwrap();
        assert_eq!(sales.as_slice(), &[550.0, 450.0]);
    }

    #[test]
    fn test_groupby_n_groups() {
        let df = sample_df();
        let gb = df.groupby(&["city"]).unwrap();
        assert_eq!(gb.n_groups(), 2);
    }

    #[test]
    fn test_groupby_all_same_group() {
        let df = DataFrame::new(vec![
            Box::new(StringSeries::from_strs("g", &["A", "A", "A"])),
            Box::new(Series::new("v", vec![1.0_f64, 2.0, 3.0])),
        ])
        .unwrap();
        let result = df.groupby(&["g"]).unwrap().sum().unwrap();
        assert_eq!(result.nrows(), 1);
        let v = result.column_typed::<f64>("v").unwrap();
        assert_eq!(v.as_slice(), &[6.0]);
    }
}
