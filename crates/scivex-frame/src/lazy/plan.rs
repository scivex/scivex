//! Logical plan nodes for lazy DataFrame evaluation.
//!
//! A [`LogicalPlan`] is a tree of operations built lazily and executed only
//! when [`LazyFrame::collect`](super::LazyFrame::collect) is called.

use super::expr::Expr;
use crate::dataframe::DataFrame;

/// A node in the logical query plan.
#[derive(Debug, Clone)]
pub enum LogicalPlan {
    /// Scan: the root data source (an existing eager `DataFrame`).
    Scan(DataFrame),

    /// Select specific expressions (column projections and computed columns).
    Select {
        input: Box<LogicalPlan>,
        exprs: Vec<Expr>,
    },

    /// Filter rows by a boolean predicate expression.
    Filter {
        input: Box<LogicalPlan>,
        predicate: Expr,
    },

    /// Sort by one or more expressions.
    Sort {
        input: Box<LogicalPlan>,
        by_column: String,
        ascending: bool,
    },

    /// Limit to first N rows.
    Limit { input: Box<LogicalPlan>, n: usize },

    /// Group by columns and aggregate.
    GroupByAgg {
        input: Box<LogicalPlan>,
        group_cols: Vec<String>,
        agg_exprs: Vec<Expr>,
    },
}
