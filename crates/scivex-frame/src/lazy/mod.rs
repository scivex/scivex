//! Lazy DataFrame evaluation with expression trees and logical plans.
//!
//! The lazy API allows building a chain of operations without executing them
//! until [`LazyFrame::collect`] is called. This enables plan-level
//! optimizations such as predicate pushdown and projection pruning.
//!
//! # Example
//!
//! ```
//! use scivex_frame::prelude::*;
//! use scivex_frame::lazy::expr::{col, lit_f64};
//!
//! let df = DataFrame::builder()
//!     .add_column("x", vec![1.0_f64, 2.0, 3.0, 4.0])
//!     .add_column("y", vec![10.0_f64, 20.0, 30.0, 40.0])
//!     .build()
//!     .unwrap();
//!
//! let result = df.lazy()
//!     .filter(col("x").gt(lit_f64(1.5)))
//!     .select(&[col("y")])
//!     .collect()
//!     .unwrap();
//!
//! assert_eq!(result.nrows(), 3);
//! assert_eq!(result.ncols(), 1);
//! ```

mod executor;
pub mod expr;
pub mod plan;

pub use expr::{BinaryOp, Expr};
pub use plan::LogicalPlan;

use crate::dataframe::DataFrame;
use crate::error::Result;

/// A lazy wrapper around a [`DataFrame`] that builds a [`LogicalPlan`]
/// without executing any operations until [`collect`](Self::collect).
#[derive(Debug, Clone)]
pub struct LazyFrame {
    plan: LogicalPlan,
}

impl LazyFrame {
    /// Create a new `LazyFrame` from an existing `DataFrame`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::prelude::*;
    /// # use scivex_frame::lazy::LazyFrame;
    /// let df = DataFrame::builder().add_column("x", vec![1_i32]).build().unwrap();
    /// let lazy = LazyFrame::new(df);
    /// let result = lazy.collect().unwrap();
    /// assert_eq!(result.nrows(), 1);
    /// ```
    pub fn new(df: DataFrame) -> Self {
        Self {
            plan: LogicalPlan::Scan(df),
        }
    }

    /// Filter rows by a boolean predicate expression.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::prelude::*;
    /// # use scivex_frame::lazy::expr::{col, lit_i64};
    /// let df = DataFrame::builder().add_column("x", vec![1_i32, 2, 3]).build().unwrap();
    /// let result = df.lazy().filter(col("x").gt(lit_i64(1))).collect().unwrap();
    /// assert_eq!(result.nrows(), 2);
    /// ```
    pub fn filter(self, predicate: Expr) -> Self {
        Self {
            plan: LogicalPlan::Filter {
                input: Box::new(self.plan),
                predicate,
            },
        }
    }

    /// Select expressions (column projections or computed columns).
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::prelude::*;
    /// # use scivex_frame::lazy::expr::col;
    /// let df = DataFrame::builder()
    ///     .add_column("a", vec![1_i32, 2])
    ///     .add_column("b", vec![3_i32, 4])
    ///     .build().unwrap();
    /// let result = df.lazy().select(&[col("a")]).collect().unwrap();
    /// assert_eq!(result.ncols(), 1);
    /// ```
    pub fn select(self, exprs: &[Expr]) -> Self {
        Self {
            plan: LogicalPlan::Select {
                input: Box::new(self.plan),
                exprs: exprs.to_vec(),
            },
        }
    }

    /// Sort by a column.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::prelude::*;
    /// let df = DataFrame::builder().add_column("x", vec![3_i32, 1, 2]).build().unwrap();
    /// let result = df.lazy().sort("x", true).collect().unwrap();
    /// let col = result.column_typed::<i32>("x").unwrap();
    /// assert_eq!(col.as_slice(), &[1, 2, 3]);
    /// ```
    pub fn sort(self, column: &str, ascending: bool) -> Self {
        Self {
            plan: LogicalPlan::Sort {
                input: Box::new(self.plan),
                by_column: column.to_string(),
                ascending,
            },
        }
    }

    /// Limit to the first `n` rows.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::prelude::*;
    /// let df = DataFrame::builder().add_column("x", vec![1_i32, 2, 3, 4]).build().unwrap();
    /// let result = df.lazy().limit(2).collect().unwrap();
    /// assert_eq!(result.nrows(), 2);
    /// ```
    pub fn limit(self, n: usize) -> Self {
        Self {
            plan: LogicalPlan::Limit {
                input: Box::new(self.plan),
                n,
            },
        }
    }

    /// Group by columns and aggregate with the given expressions.
    ///
    /// Each expression in `agg_exprs` should be an `Expr::Agg` or
    /// `Expr::Alias(Expr::Agg(...))`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::prelude::*;
    /// # use scivex_frame::lazy::expr::col;
    /// let df = DataFrame::builder()
    ///     .add_column("g", vec![1_i32, 1, 2])
    ///     .add_column("v", vec![10.0_f64, 20.0, 30.0])
    ///     .build().unwrap();
    /// let result = df.lazy()
    ///     .groupby_agg(&["g"], &[col("v").sum()])
    ///     .collect().unwrap();
    /// assert_eq!(result.nrows(), 2);
    /// ```
    pub fn groupby_agg(self, group_cols: &[&str], agg_exprs: &[Expr]) -> Self {
        Self {
            plan: LogicalPlan::GroupByAgg {
                input: Box::new(self.plan),
                group_cols: group_cols.iter().map(|s| (*s).to_string()).collect(),
                agg_exprs: agg_exprs.to_vec(),
            },
        }
    }

    /// Access the underlying logical plan.
    pub fn plan(&self) -> &LogicalPlan {
        &self.plan
    }

    /// Execute the plan and return the materialized `DataFrame`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::prelude::*;
    /// let df = DataFrame::builder().add_column("x", vec![1_i32]).build().unwrap();
    /// let result = df.lazy().collect().unwrap();
    /// assert_eq!(result.nrows(), 1);
    /// ```
    pub fn collect(self) -> Result<DataFrame> {
        executor::execute(&self.plan)
    }
}

// ---------------------------------------------------------------------------
// DataFrame → LazyFrame bridge
// ---------------------------------------------------------------------------

impl DataFrame {
    /// Convert this `DataFrame` into a [`LazyFrame`] for lazy evaluation.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::prelude::*;
    /// let df = DataFrame::builder().add_column("x", vec![1_i32, 2]).build().unwrap();
    /// let result = df.lazy().collect().unwrap();
    /// assert_eq!(result.nrows(), 2);
    /// ```
    pub fn lazy(self) -> LazyFrame {
        LazyFrame::new(self)
    }
}

#[cfg(test)]
mod tests {
    use super::expr::{col, lit_f64, lit_i64};
    use super::*;
    use crate::series::Series;

    fn sample_df() -> DataFrame {
        DataFrame::new(vec![
            Box::new(Series::new("a", vec![1_i32, 2, 3, 4, 5])),
            Box::new(Series::new("b", vec![10.0_f64, 20.0, 30.0, 40.0, 50.0])),
            Box::new(Series::new("group", vec![1_i32, 1, 2, 2, 2])),
        ])
        .unwrap()
    }

    #[test]
    fn test_lazy_collect_identity() {
        let df = sample_df();
        let result = df.clone().lazy().collect().unwrap();
        assert_eq!(result.nrows(), df.nrows());
        assert_eq!(result.ncols(), df.ncols());
    }

    #[test]
    fn test_lazy_filter() {
        let df = sample_df();
        let result = df
            .lazy()
            .filter(col("b").gt(lit_f64(25.0)))
            .collect()
            .unwrap();
        assert_eq!(result.nrows(), 3);
        let col_b = result.column_typed::<f64>("b").unwrap();
        assert!(col_b.as_slice().iter().all(|&v| v > 25.0));
    }

    #[test]
    fn test_lazy_select() {
        let df = sample_df();
        let result = df.lazy().select(&[col("b")]).collect().unwrap();
        assert_eq!(result.ncols(), 1);
        assert_eq!(result.column_names(), vec!["b"]);
        assert_eq!(result.nrows(), 5);
    }

    #[test]
    fn test_lazy_filter_then_select() {
        let df = sample_df();
        let result = df
            .lazy()
            .filter(col("a").gt(lit_i64(2)))
            .select(&[col("b")])
            .collect()
            .unwrap();
        assert_eq!(result.nrows(), 3);
        assert_eq!(result.ncols(), 1);
    }

    #[test]
    fn test_lazy_sort() {
        let df = sample_df();
        let result = df.lazy().sort("b", false).collect().unwrap();
        let col_b = result.column_typed::<f64>("b").unwrap();
        assert_eq!(col_b.as_slice(), &[50.0, 40.0, 30.0, 20.0, 10.0]);
    }

    #[test]
    fn test_lazy_limit() {
        let df = sample_df();
        let result = df.lazy().limit(3).collect().unwrap();
        assert_eq!(result.nrows(), 3);
    }

    #[test]
    fn test_lazy_groupby_sum() {
        let df = sample_df();
        let result = df
            .lazy()
            .groupby_agg(&["group"], &[col("b").sum()])
            .collect()
            .unwrap();
        assert_eq!(result.nrows(), 2);
        assert!(result.column("group").is_ok());
        assert!(result.column("b").is_ok());
    }

    #[test]
    fn test_lazy_chained_operations() {
        let df = sample_df();
        let result = df
            .lazy()
            .filter(col("a").gt(lit_i64(1)))
            .sort("b", true)
            .limit(2)
            .collect()
            .unwrap();
        assert_eq!(result.nrows(), 2);
        let col_b = result.column_typed::<f64>("b").unwrap();
        assert_eq!(col_b.as_slice(), &[20.0, 30.0]);
    }

    #[test]
    fn test_lazy_select_with_alias() {
        let df = sample_df();
        let result = df
            .lazy()
            .select(&[col("b").alias("beta")])
            .collect()
            .unwrap();
        assert_eq!(result.column_names(), vec!["beta"]);
    }

    #[test]
    fn test_lazy_filter_compound_predicate() {
        let df = sample_df();
        let result = df
            .lazy()
            .filter(col("a").gt(lit_i64(1)).and(col("a").lt(lit_i64(5))))
            .collect()
            .unwrap();
        assert_eq!(result.nrows(), 3); // a = 2, 3, 4
    }

    #[test]
    fn test_lazy_select_computed_column() {
        let df = DataFrame::new(vec![
            Box::new(Series::new("x", vec![1.0_f64, 2.0, 3.0])),
            Box::new(Series::new("y", vec![10.0_f64, 20.0, 30.0])),
        ])
        .unwrap();

        let result = df
            .lazy()
            .select(&[col("x"), col("x").add(col("y")).alias("sum")])
            .collect()
            .unwrap();
        assert_eq!(result.ncols(), 2);
        let sum_col = result.column_typed::<f64>("sum").unwrap();
        assert_eq!(sum_col.as_slice(), &[11.0, 22.0, 33.0]);
    }

    #[test]
    fn test_lazy_groupby_mean() {
        let df = sample_df();
        let result = df
            .lazy()
            .groupby_agg(&["group"], &[col("b").mean().alias("avg_b")])
            .collect()
            .unwrap();
        assert_eq!(result.nrows(), 2);
        assert!(result.column("avg_b").is_ok());
    }
}
