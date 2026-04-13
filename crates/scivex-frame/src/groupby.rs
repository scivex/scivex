//! Group-by aggregation for [`DataFrame`].

use std::collections::HashMap;

#[cfg(feature = "parallel")]
use rayon::prelude::*;
use scivex_core::Scalar;

use crate::dataframe::DataFrame;
use crate::dtype::{DType, HasDType};
use crate::error::{FrameError, Result};
use crate::series::categorical::CategoricalSeries;
use crate::series::string::StringSeries;
use crate::series::{AnySeries, Series};

/// Return type for internal grouping functions: (groups, group_ids, num_groups).
type GroupResult = (Vec<(Vec<String>, Vec<usize>)>, Vec<u32>, usize);

/// Aggregation function selector.
///
/// # Examples
///
/// ```
/// # use scivex_frame::prelude::*;
/// # use scivex_frame::groupby::AggFunc;
/// let df = DataFrame::builder()
///     .add_boxed(Box::new(StringSeries::from_strs("g", &["A", "A", "B"])))
///     .add_column("v", vec![10.0_f64, 20.0, 30.0])
///     .build()
///     .unwrap();
/// let result = df.groupby(&["g"]).unwrap().agg("v", AggFunc::Sum).unwrap();
/// assert_eq!(result.nrows(), 2);
/// ```
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
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
///
/// # Examples
///
/// ```
/// # use scivex_frame::prelude::*;
/// let df = DataFrame::builder()
///     .add_boxed(Box::new(StringSeries::from_strs("city", &["NYC", "LA", "NYC"])))
///     .add_column("sales", vec![100.0_f64, 200.0, 150.0])
///     .build()
///     .unwrap();
/// let gb = df.groupby(&["city"]).unwrap();
/// assert_eq!(gb.n_groups(), 2);
/// let totals = gb.sum().unwrap();
/// assert_eq!(totals.nrows(), 2);
/// ```
pub struct GroupBy<'a> {
    df: &'a DataFrame,
    group_columns: Vec<String>,
    /// Maps composite key → row indices in the original `DataFrame`.
    groups: Vec<(Vec<String>, Vec<usize>)>,
    /// Per-row group id for cache-friendly aggregation (group_ids[row] = group index).
    group_ids: Vec<u32>,
    /// Number of groups.
    num_groups: usize,
}

impl<'a> GroupBy<'a> {
    /// Group `df` by the given columns.
    ///
    /// When all group columns are [`CategoricalSeries`], uses a fast path that
    /// indexes by integer codes instead of hashing display strings.
    pub(crate) fn new(df: &'a DataFrame, cols: &[&str]) -> Result<Self> {
        // Validate that all group columns exist.
        for &col in cols {
            df.column(col)?;
        }

        let group_columns: Vec<String> = cols.iter().map(|s| (*s).to_string()).collect();
        let nrows = df.nrows();

        // Try categorical fast path: all group cols must be CategoricalSeries.
        let cat_cols: Vec<&CategoricalSeries> = cols
            .iter()
            .filter_map(|&c| {
                df.column(c)
                    .ok()?
                    .as_any()
                    .downcast_ref::<CategoricalSeries>()
            })
            .collect();

        let (groups, group_ids, num_groups) = if cat_cols.len() == cols.len() {
            Self::group_by_codes(nrows, &cat_cols)
        } else {
            // Try sorted run-length encoding first (no hashing, O(n) scan).
            let series: Vec<&dyn AnySeries> = cols
                .iter()
                .map(|&c| df.column(c).expect("groupby column exists"))
                .collect();
            if Self::is_sorted(&series, nrows) {
                Self::group_by_sorted_runs(&series, nrows)
            } else {
                // Use parallel groupby for large datasets.
                #[cfg(feature = "parallel")]
                {
                    if nrows >= 10_000 {
                        Self::group_by_parallel(df, cols, nrows)
                    } else {
                        Self::group_by_strings(df, cols, nrows)
                    }
                }
                #[cfg(not(feature = "parallel"))]
                {
                    Self::group_by_strings(df, cols, nrows)
                }
            }
        };

        Ok(Self {
            df,
            group_columns,
            groups,
            group_ids,
            num_groups,
        })
    }

    /// Check if data is already sorted by the group columns.
    fn is_sorted(series: &[&dyn AnySeries], nrows: usize) -> bool {
        if nrows <= 1 {
            return true;
        }
        for row in 1..nrows {
            for &col in series {
                match col.compare_at(row - 1, row) {
                    core::cmp::Ordering::Less => break,
                    core::cmp::Ordering::Greater => return false,
                    core::cmp::Ordering::Equal => {}
                }
            }
        }
        true
    }

    /// Fast grouping when data is already sorted: scan for run boundaries.
    /// O(n) with no hashing — just detect where group keys change.
    fn group_by_sorted_runs(series: &[&dyn AnySeries], nrows: usize) -> GroupResult {
        let mut group_ids = vec![0u32; nrows];
        let mut group_starts: Vec<usize> = vec![0];
        let mut gid = 0u32;

        for (row, gid_slot) in group_ids.iter_mut().enumerate().skip(1) {
            let same = series.iter().all(|col| col.hash_value(row - 1) == col.hash_value(row));
            if !same {
                gid += 1;
                group_starts.push(row);
            }
            *gid_slot = gid;
        }

        let ng = gid as usize + 1;
        let mut group_indices: Vec<Vec<usize>> = vec![Vec::new(); ng];
        for (row, &g) in group_ids.iter().enumerate() {
            group_indices[g as usize].push(row);
        }

        let groups: Vec<(Vec<String>, Vec<usize>)> = group_starts
            .into_iter()
            .zip(group_indices)
            .map(|(first_row, indices)| {
                let str_key: Vec<String> = series.iter().map(|s| s.display_value(first_row)).collect();
                (str_key, indices)
            })
            .collect();

        (groups, group_ids, ng)
    }

    /// Parallel groupby: split rows into chunks, build per-chunk hash maps, merge.
    ///
    /// Only used when `parallel` feature is enabled and nrows > threshold.
    #[cfg(feature = "parallel")]
    fn group_by_parallel(df: &DataFrame, cols: &[&str], nrows: usize) -> GroupResult {
        let series: Vec<&dyn AnySeries> = cols
            .iter()
            .map(|&c| df.column(c).expect("groupby column exists"))
            .collect();

        // Each thread builds: HashMap<Vec<u64>, (u32, usize)> mapping hash_key → (local_gid, first_row)
        let n_threads = rayon::current_num_threads().max(1);
        let chunk_size = nrows.div_ceil(n_threads);

        // Phase 1: per-chunk grouping in parallel.
        #[allow(clippy::type_complexity)]
        let chunk_results: Vec<(HashMap<Vec<u64>, u32>, Vec<u32>, Vec<usize>)> = (0..n_threads)
            .into_par_iter()
            .map(|t| {
                let start = t * chunk_size;
                let end = (start + chunk_size).min(nrows);
                let mut map: HashMap<Vec<u64>, u32> = HashMap::new();
                let mut local_ids = vec![0u32; end - start];
                let mut first_rows: Vec<usize> = Vec::new();
                let mut next_gid = 0u32;

                for row in start..end {
                    let hash_key: Vec<u64> = series.iter().map(|s| s.hash_value(row)).collect();
                    let gid = *map.entry(hash_key).or_insert_with(|| {
                        let g = next_gid;
                        next_gid += 1;
                        first_rows.push(row);
                        g
                    });
                    local_ids[row - start] = gid;
                }
                (map, local_ids, first_rows)
            })
            .collect();

        // Phase 2: merge per-chunk maps into global map.
        let mut global_map: HashMap<Vec<u64>, u32> = HashMap::new();
        let mut global_first_rows: Vec<usize> = Vec::new();
        let mut next_global = 0u32;
        // For each chunk, store the local→global gid mapping.
        let mut remap: Vec<Vec<u32>> = Vec::with_capacity(n_threads);

        for (local_map, _, local_firsts) in &chunk_results {
            let mut local_remap = vec![0u32; local_map.len()];
            for (hash_key, &local_gid) in local_map {
                let global_gid = *global_map.entry(hash_key.clone()).or_insert_with(|| {
                    let g = next_global;
                    next_global += 1;
                    global_first_rows.push(local_firsts[local_gid as usize]);
                    g
                });
                local_remap[local_gid as usize] = global_gid;
            }
            remap.push(local_remap);
        }

        // Phase 3: build global group_ids by remapping local ids.
        let ng = next_global as usize;
        let mut group_ids = vec![0u32; nrows];
        for (t, (_, local_ids, _)) in chunk_results.iter().enumerate() {
            let start = t * chunk_size;
            let mapping = &remap[t];
            for (i, &local_gid) in local_ids.iter().enumerate() {
                group_ids[start + i] = mapping[local_gid as usize];
            }
        }

        // Build groups vec.
        let mut group_indices: Vec<Vec<usize>> = vec![Vec::new(); ng];
        for (row, &gid) in group_ids.iter().enumerate() {
            group_indices[gid as usize].push(row);
        }

        let groups: Vec<(Vec<String>, Vec<usize>)> = global_first_rows
            .into_iter()
            .zip(group_indices)
            .map(|(first_row, indices)| {
                let str_key: Vec<String> =
                    series.iter().map(|s| s.display_value(first_row)).collect();
                (str_key, indices)
            })
            .collect();

        (groups, group_ids, ng)
    }

    /// Fast grouping via integer category codes.
    /// Returns (groups, group_ids, num_groups).
    fn group_by_codes(nrows: usize, cat_cols: &[&CategoricalSeries]) -> GroupResult {
        let mut map: HashMap<Vec<u32>, u32> = HashMap::new();
        let mut first_rows: Vec<usize> = Vec::new();
        let mut code_keys: Vec<Vec<u32>> = Vec::new();
        let mut group_ids = vec![0u32; nrows];
        let mut next_gid = 0u32;

        for (row, gid_slot) in group_ids.iter_mut().enumerate().take(nrows) {
            let code_key: Vec<u32> = cat_cols.iter().map(|c| c.codes()[row]).collect();
            let gid = *map.entry(code_key.clone()).or_insert_with(|| {
                let g = next_gid;
                next_gid += 1;
                first_rows.push(row);
                code_keys.push(code_key);
                g
            });
            *gid_slot = gid;
        }

        let ng = next_gid as usize;
        let mut group_indices: Vec<Vec<usize>> = vec![Vec::new(); ng];
        for (row, &gid) in group_ids.iter().enumerate() {
            group_indices[gid as usize].push(row);
        }

        let groups: Vec<(Vec<String>, Vec<usize>)> = code_keys
            .into_iter()
            .zip(group_indices)
            .map(|(code_key, indices)| {
                let str_key: Vec<String> = code_key
                    .iter()
                    .zip(cat_cols.iter())
                    .map(|(&code, col)| col.categories()[code as usize].clone())
                    .collect();
                (str_key, indices)
            })
            .collect();

        (groups, group_ids, ng)
    }

    /// Fallback grouping using hash-based keys (avoids string formatting for numeric types).
    /// Returns (groups, group_ids, num_groups).
    fn group_by_strings(df: &DataFrame, cols: &[&str], nrows: usize) -> GroupResult {
        // Resolve column references once.
        let series: Vec<&dyn AnySeries> = cols
            .iter()
            .map(|&c| df.column(c).expect("groupby column exists"))
            .collect();

        if series.len() == 1 {
            // Single-column fast path: use a single u64 hash key (no Vec alloc per row).
            let col = series[0];
            let mut map: HashMap<u64, u32> = HashMap::new(); // hash → group_id
            let mut first_rows: Vec<usize> = Vec::new();
            let mut group_ids = vec![0u32; nrows];
            let mut next_gid = 0u32;

            for (row, gid_slot) in group_ids.iter_mut().enumerate().take(nrows) {
                let hk = col.hash_value(row);
                let gid = *map.entry(hk).or_insert_with(|| {
                    let g = next_gid;
                    next_gid += 1;
                    first_rows.push(row);
                    g
                });
                *gid_slot = gid;
            }

            // Build the groups Vec from group_ids for compatibility.
            let ng = next_gid as usize;
            let mut group_indices: Vec<Vec<usize>> = vec![Vec::new(); ng];
            for (row, &gid) in group_ids.iter().enumerate() {
                group_indices[gid as usize].push(row);
            }

            let groups: Vec<(Vec<String>, Vec<usize>)> = first_rows
                .into_iter()
                .zip(group_indices)
                .map(|(first_row, indices)| {
                    let str_key = vec![col.display_value(first_row)];
                    (str_key, indices)
                })
                .collect();

            return (groups, group_ids, ng);
        }

        // Multi-column path: composite hash key.
        let mut map: HashMap<Vec<u64>, u32> = HashMap::new();
        let mut first_rows: Vec<usize> = Vec::new();
        let mut group_ids = vec![0u32; nrows];
        let mut next_gid = 0u32;

        for (row, gid_slot) in group_ids.iter_mut().enumerate().take(nrows) {
            let hash_key: Vec<u64> = series.iter().map(|s| s.hash_value(row)).collect();
            let gid = *map.entry(hash_key).or_insert_with(|| {
                let g = next_gid;
                next_gid += 1;
                first_rows.push(row);
                g
            });
            *gid_slot = gid;
        }

        let ng = next_gid as usize;
        let mut group_indices: Vec<Vec<usize>> = vec![Vec::new(); ng];
        for (row, &gid) in group_ids.iter().enumerate() {
            group_indices[gid as usize].push(row);
        }

        let groups: Vec<(Vec<String>, Vec<usize>)> = first_rows
            .into_iter()
            .zip(group_indices)
            .map(|(first_row, indices)| {
                let str_key: Vec<String> =
                    series.iter().map(|s| s.display_value(first_row)).collect();
                (str_key, indices)
            })
            .collect();

        (groups, group_ids, ng)
    }

    /// Number of groups.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::prelude::*;
    /// let df = DataFrame::builder()
    ///     .add_boxed(Box::new(StringSeries::from_strs("g", &["A", "B", "A"])))
    ///     .add_column("v", vec![1_i32, 2, 3])
    ///     .build()
    ///     .unwrap();
    /// assert_eq!(df.groupby(&["g"]).unwrap().n_groups(), 2);
    /// ```
    pub fn n_groups(&self) -> usize {
        self.groups.len()
    }

    // -- Convenience aggregation methods ------------------------------------

    /// Sum of all numeric columns per group.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::prelude::*;
    /// let df = DataFrame::builder()
    ///     .add_boxed(Box::new(StringSeries::from_strs("g", &["A", "A", "B"])))
    ///     .add_column("v", vec![10.0_f64, 20.0, 30.0])
    ///     .build()
    ///     .unwrap();
    /// let result = df.groupby(&["g"]).unwrap().sum().unwrap();
    /// assert_eq!(result.nrows(), 2);
    /// ```
    pub fn sum(&self) -> Result<DataFrame> {
        self.agg_all(AggFunc::Sum)
    }

    /// Mean of all float columns per group.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::prelude::*;
    /// let df = DataFrame::builder()
    ///     .add_boxed(Box::new(StringSeries::from_strs("g", &["A", "A", "B"])))
    ///     .add_column("v", vec![10.0_f64, 20.0, 30.0])
    ///     .build()
    ///     .unwrap();
    /// let result = df.groupby(&["g"]).unwrap().mean().unwrap();
    /// assert_eq!(result.nrows(), 2);
    /// ```
    pub fn mean(&self) -> Result<DataFrame> {
        self.agg_all(AggFunc::Mean)
    }

    /// Min of all numeric columns per group.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::prelude::*;
    /// let df = DataFrame::builder()
    ///     .add_boxed(Box::new(StringSeries::from_strs("g", &["A", "A", "B"])))
    ///     .add_column("v", vec![10.0_f64, 20.0, 30.0])
    ///     .build()
    ///     .unwrap();
    /// let result = df.groupby(&["g"]).unwrap().min().unwrap();
    /// assert_eq!(result.nrows(), 2);
    /// ```
    pub fn min(&self) -> Result<DataFrame> {
        self.agg_all(AggFunc::Min)
    }

    /// Max of all numeric columns per group.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::prelude::*;
    /// let df = DataFrame::builder()
    ///     .add_boxed(Box::new(StringSeries::from_strs("g", &["A", "A", "B"])))
    ///     .add_column("v", vec![10.0_f64, 20.0, 30.0])
    ///     .build()
    ///     .unwrap();
    /// let result = df.groupby(&["g"]).unwrap().max().unwrap();
    /// assert_eq!(result.nrows(), 2);
    /// ```
    pub fn max(&self) -> Result<DataFrame> {
        self.agg_all(AggFunc::Max)
    }

    /// Count of rows per group.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::prelude::*;
    /// let df = DataFrame::builder()
    ///     .add_boxed(Box::new(StringSeries::from_strs("g", &["A", "A", "B"])))
    ///     .add_column("v", vec![1_i32, 2, 3])
    ///     .build()
    ///     .unwrap();
    /// let result = df.groupby(&["g"]).unwrap().count().unwrap();
    /// assert_eq!(result.nrows(), 2);
    /// ```
    pub fn count(&self) -> Result<DataFrame> {
        self.agg_all(AggFunc::Count)
    }

    /// First row per group.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::prelude::*;
    /// let df = DataFrame::builder()
    ///     .add_boxed(Box::new(StringSeries::from_strs("g", &["A", "A", "B"])))
    ///     .add_column("v", vec![10_i32, 20, 30])
    ///     .build()
    ///     .unwrap();
    /// let result = df.groupby(&["g"]).unwrap().first().unwrap();
    /// assert_eq!(result.nrows(), 2);
    /// ```
    pub fn first(&self) -> Result<DataFrame> {
        self.agg_all(AggFunc::First)
    }

    /// Last row per group.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::prelude::*;
    /// let df = DataFrame::builder()
    ///     .add_boxed(Box::new(StringSeries::from_strs("g", &["A", "A", "B"])))
    ///     .add_column("v", vec![10_i32, 20, 30])
    ///     .build()
    ///     .unwrap();
    /// let result = df.groupby(&["g"]).unwrap().last().unwrap();
    /// assert_eq!(result.nrows(), 2);
    /// ```
    pub fn last(&self) -> Result<DataFrame> {
        self.agg_all(AggFunc::Last)
    }

    /// Apply an aggregation function to a specific column.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::prelude::*;
    /// # use scivex_frame::groupby::AggFunc;
    /// let df = DataFrame::builder()
    ///     .add_boxed(Box::new(StringSeries::from_strs("g", &["A", "A", "B"])))
    ///     .add_column("v", vec![10.0_f64, 20.0, 30.0])
    ///     .build()
    ///     .unwrap();
    /// let result = df.groupby(&["g"]).unwrap().agg("v", AggFunc::Sum).unwrap();
    /// assert_eq!(result.ncols(), 2); // g + v
    /// ```
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
                    .map(|(_, idx)| *idx.last().expect("group indices guaranteed non-empty"))
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
    ///
    /// Uses a single sequential scan over the data with `group_ids` lookup
    /// into per-group accumulators, which is much more cache-friendly than
    /// random-access gather per group.
    fn agg_typed<T: Scalar + HasDType + 'static>(
        &self,
        col: &Series<T>,
        func: AggFunc,
    ) -> Result<Box<dyn AnySeries>> {
        let name = col.name().to_string();
        let slice = col.as_slice();
        let ng = self.num_groups;

        match func {
            AggFunc::Sum => {
                let mut acc = vec![T::zero(); ng];
                for (row, &val) in slice.iter().enumerate() {
                    acc[self.group_ids[row] as usize] += val;
                }
                Ok(Box::new(Series::new(name, acc)))
            }
            AggFunc::Min => {
                let mut acc = vec![T::zero(); ng];
                let mut init = vec![false; ng];
                for (row, &val) in slice.iter().enumerate() {
                    let g = self.group_ids[row] as usize;
                    if !init[g] || val < acc[g] {
                        acc[g] = val;
                        init[g] = true;
                    }
                }
                Ok(Box::new(Series::new(name, acc)))
            }
            AggFunc::Max => {
                let mut acc = vec![T::zero(); ng];
                let mut init = vec![false; ng];
                for (row, &val) in slice.iter().enumerate() {
                    let g = self.group_ids[row] as usize;
                    if !init[g] || val > acc[g] {
                        acc[g] = val;
                        init[g] = true;
                    }
                }
                Ok(Box::new(Series::new(name, acc)))
            }
            AggFunc::Mean => {
                let mut sums = vec![T::zero(); ng];
                let mut counts = vec![0usize; ng];
                for (row, &val) in slice.iter().enumerate() {
                    let g = self.group_ids[row] as usize;
                    sums[g] += val;
                    counts[g] += 1;
                }
                let data: Vec<T> = sums
                    .into_iter()
                    .zip(counts)
                    .map(|(s, c)| s / T::from_usize(c))
                    .collect();
                Ok(Box::new(Series::new(name, data)))
            }
            AggFunc::Count | AggFunc::First | AggFunc::Last => {
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
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::prelude::*;
    /// let df = DataFrame::builder()
    ///     .add_boxed(Box::new(StringSeries::from_strs("city", &["NYC", "LA", "NYC"])))
    ///     .add_column("sales", vec![100.0_f64, 200.0, 150.0])
    ///     .build()
    ///     .unwrap();
    /// let gb = df.groupby(&["city"]).unwrap();
    /// assert_eq!(gb.n_groups(), 2);
    /// ```
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

    #[test]
    fn test_groupby_all_unique_values() {
        let df = DataFrame::new(vec![
            Box::new(StringSeries::from_strs("g", &["A", "B", "C", "D"])),
            Box::new(Series::new("v", vec![10.0_f64, 20.0, 30.0, 40.0])),
        ])
        .unwrap();
        let gb = df.groupby(&["g"]).unwrap();
        assert_eq!(gb.n_groups(), 4);
        let result = gb.sum().unwrap();
        assert_eq!(result.nrows(), 4);
        let v = result.column_typed::<f64>("v").unwrap();
        assert_eq!(v.as_slice(), &[10.0, 20.0, 30.0, 40.0]);
    }

    #[test]
    fn test_groupby_nonexistent_column() {
        let df = DataFrame::new(vec![Box::new(Series::new("x", vec![1_i32, 2]))]).unwrap();
        assert!(df.groupby(&["nonexistent"]).is_err());
    }

    #[test]
    fn test_groupby_agg_nonexistent_column() {
        let df = DataFrame::new(vec![
            Box::new(StringSeries::from_strs("g", &["A", "B"])),
            Box::new(Series::new("v", vec![1.0_f64, 2.0])),
        ])
        .unwrap();
        let gb = df.groupby(&["g"]).unwrap();
        assert!(gb.agg("nonexistent", AggFunc::Sum).is_err());
    }

    #[test]
    fn test_groupby_categorical_fast_path() {
        use crate::series::categorical::CategoricalSeries;

        let df = DataFrame::new(vec![
            Box::new(CategoricalSeries::from_strs(
                "region",
                &["East", "West", "East", "West", "East"],
            )),
            Box::new(Series::new(
                "revenue",
                vec![100.0_f64, 200.0, 150.0, 250.0, 300.0],
            )),
        ])
        .unwrap();

        let result = df.groupby(&["region"]).unwrap().sum().unwrap();
        assert_eq!(result.nrows(), 2);
        let rev = result.column_typed::<f64>("revenue").unwrap();
        // East: 100+150+300 = 550, West: 200+250 = 450
        assert_eq!(rev.as_slice(), &[550.0, 450.0]);
    }

    #[test]
    fn test_groupby_sorted_runs_fast_path() {
        // Data is already sorted by group column — should use RLE path.
        let df = DataFrame::new(vec![
            Box::new(StringSeries::from_strs(
                "g",
                &["A", "A", "A", "B", "B", "C"],
            )),
            Box::new(Series::new("v", vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])),
        ])
        .unwrap();
        let gb = df.groupby(&["g"]).unwrap();
        assert_eq!(gb.n_groups(), 3);
        let result = gb.sum().unwrap();
        let v = result.column_typed::<f64>("v").unwrap();
        assert_eq!(v.as_slice(), &[6.0, 9.0, 6.0]);
    }

    #[test]
    fn test_groupby_single_row() {
        let df = DataFrame::new(vec![
            Box::new(StringSeries::from_strs("g", &["A"])),
            Box::new(Series::new("v", vec![42.0_f64])),
        ])
        .unwrap();
        let gb = df.groupby(&["g"]).unwrap();
        assert_eq!(gb.n_groups(), 1);
        let result = gb.mean().unwrap();
        let v = result.column_typed::<f64>("v").unwrap();
        assert_eq!(v.get(0), Some(42.0));
    }
}
