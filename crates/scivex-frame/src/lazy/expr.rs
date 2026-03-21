//! Expression tree for lazy DataFrame evaluation.
//!
//! Expressions describe *what* to compute without executing immediately.
//! They are composed into a [`LogicalPlan`](super::plan::LogicalPlan) and
//! evaluated when [`LazyFrame::collect`](super::LazyFrame::collect) is called.

use crate::groupby::AggFunc;

/// A column expression that can be evaluated against a [`DataFrame`](crate::DataFrame).
#[derive(Debug, Clone)]
pub enum Expr {
    /// Reference a column by name.
    Col(String),
    /// A literal f64 value broadcast to all rows.
    LitF64(f64),
    /// A literal i64 value broadcast to all rows.
    LitI64(i64),
    /// A literal string value broadcast to all rows.
    LitStr(String),
    /// A literal boolean value broadcast to all rows.
    LitBool(bool),
    /// Arithmetic or comparison between two expressions.
    BinaryOp {
        left: Box<Expr>,
        op: BinaryOp,
        right: Box<Expr>,
    },
    /// Negate a boolean expression.
    Not(Box<Expr>),
    /// Aggregation applied to an expression.
    Agg { expr: Box<Expr>, func: AggFunc },
    /// Alias (rename) the output of an expression.
    Alias { expr: Box<Expr>, name: String },
    /// Sort by this expression.
    Sort { expr: Box<Expr>, ascending: bool },
}

/// Binary operations for expressions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Eq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
    And,
    Or,
}

// ---------------------------------------------------------------------------
// Expression builder functions
// ---------------------------------------------------------------------------

/// Create a column reference expression.
///
/// # Examples
///
/// ```
/// # use scivex_frame::lazy::expr::col;
/// let e = col("x");
/// assert!(matches!(e, scivex_frame::lazy::expr::Expr::Col(_)));
/// ```
pub fn col(name: &str) -> Expr {
    Expr::Col(name.to_string())
}

/// Create a literal f64 expression.
///
/// # Examples
///
/// ```
/// # use scivex_frame::lazy::expr::lit_f64;
/// let e = lit_f64(3.14);
/// assert!(matches!(e, scivex_frame::lazy::expr::Expr::LitF64(_)));
/// ```
pub fn lit_f64(val: f64) -> Expr {
    Expr::LitF64(val)
}

/// Create a literal i64 expression.
///
/// # Examples
///
/// ```
/// # use scivex_frame::lazy::expr::lit_i64;
/// let e = lit_i64(42);
/// assert!(matches!(e, scivex_frame::lazy::expr::Expr::LitI64(_)));
/// ```
pub fn lit_i64(val: i64) -> Expr {
    Expr::LitI64(val)
}

/// Create a literal string expression.
///
/// # Examples
///
/// ```
/// # use scivex_frame::lazy::expr::lit_str;
/// let e = lit_str("hello");
/// assert!(matches!(e, scivex_frame::lazy::expr::Expr::LitStr(_)));
/// ```
pub fn lit_str(val: &str) -> Expr {
    Expr::LitStr(val.to_string())
}

/// Create a literal boolean expression.
///
/// # Examples
///
/// ```
/// # use scivex_frame::lazy::expr::lit_bool;
/// let e = lit_bool(true);
/// assert!(matches!(e, scivex_frame::lazy::expr::Expr::LitBool(true)));
/// ```
pub fn lit_bool(val: bool) -> Expr {
    Expr::LitBool(val)
}

impl Expr {
    /// Alias this expression (rename the output column).
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::lazy::expr::col;
    /// let e = col("x").alias("renamed");
    /// assert!(matches!(e, scivex_frame::lazy::expr::Expr::Alias { .. }));
    /// ```
    pub fn alias(self, name: &str) -> Self {
        Self::Alias {
            expr: Box::new(self),
            name: name.to_string(),
        }
    }

    /// Aggregate: sum.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::lazy::expr::col;
    /// let e = col("x").sum();
    /// assert!(matches!(e, scivex_frame::lazy::expr::Expr::Agg { .. }));
    /// ```
    pub fn sum(self) -> Self {
        Self::Agg {
            expr: Box::new(self),
            func: AggFunc::Sum,
        }
    }

    /// Aggregate: mean.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::lazy::expr::col;
    /// let e = col("x").mean();
    /// assert!(matches!(e, scivex_frame::lazy::expr::Expr::Agg { .. }));
    /// ```
    pub fn mean(self) -> Self {
        Self::Agg {
            expr: Box::new(self),
            func: AggFunc::Mean,
        }
    }

    /// Aggregate: min.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::lazy::expr::col;
    /// let e = col("x").min();
    /// assert!(matches!(e, scivex_frame::lazy::expr::Expr::Agg { .. }));
    /// ```
    pub fn min(self) -> Self {
        Self::Agg {
            expr: Box::new(self),
            func: AggFunc::Min,
        }
    }

    /// Aggregate: max.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::lazy::expr::col;
    /// let e = col("x").max();
    /// assert!(matches!(e, scivex_frame::lazy::expr::Expr::Agg { .. }));
    /// ```
    pub fn max(self) -> Self {
        Self::Agg {
            expr: Box::new(self),
            func: AggFunc::Max,
        }
    }

    /// Aggregate: count.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::lazy::expr::col;
    /// let e = col("x").count();
    /// assert!(matches!(e, scivex_frame::lazy::expr::Expr::Agg { .. }));
    /// ```
    pub fn count(self) -> Self {
        Self::Agg {
            expr: Box::new(self),
            func: AggFunc::Count,
        }
    }

    /// Aggregate: first.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::lazy::expr::col;
    /// let e = col("x").first();
    /// assert!(matches!(e, scivex_frame::lazy::expr::Expr::Agg { .. }));
    /// ```
    pub fn first(self) -> Self {
        Self::Agg {
            expr: Box::new(self),
            func: AggFunc::First,
        }
    }

    /// Aggregate: last.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::lazy::expr::col;
    /// let e = col("x").last();
    /// assert!(matches!(e, scivex_frame::lazy::expr::Expr::Agg { .. }));
    /// ```
    pub fn last(self) -> Self {
        Self::Agg {
            expr: Box::new(self),
            func: AggFunc::Last,
        }
    }

    /// Sort ascending.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::lazy::expr::col;
    /// let e = col("x").sort_asc();
    /// assert!(matches!(e, scivex_frame::lazy::expr::Expr::Sort { ascending: true, .. }));
    /// ```
    pub fn sort_asc(self) -> Self {
        Self::Sort {
            expr: Box::new(self),
            ascending: true,
        }
    }

    /// Sort descending.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::lazy::expr::col;
    /// let e = col("x").sort_desc();
    /// assert!(matches!(e, scivex_frame::lazy::expr::Expr::Sort { ascending: false, .. }));
    /// ```
    pub fn sort_desc(self) -> Self {
        Self::Sort {
            expr: Box::new(self),
            ascending: false,
        }
    }

    /// Equality comparison.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::lazy::expr::{col, lit_i64};
    /// let e = col("x").eq(lit_i64(10));
    /// assert!(matches!(e, scivex_frame::lazy::expr::Expr::BinaryOp { .. }));
    /// ```
    pub fn eq(self, other: Self) -> Self {
        Self::BinaryOp {
            left: Box::new(self),
            op: BinaryOp::Eq,
            right: Box::new(other),
        }
    }

    /// Not-equal comparison.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::lazy::expr::{col, lit_i64};
    /// let e = col("x").neq(lit_i64(0));
    /// assert!(matches!(e, scivex_frame::lazy::expr::Expr::BinaryOp { .. }));
    /// ```
    pub fn neq(self, other: Self) -> Self {
        Self::BinaryOp {
            left: Box::new(self),
            op: BinaryOp::NotEq,
            right: Box::new(other),
        }
    }

    /// Less-than comparison.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::lazy::expr::{col, lit_f64};
    /// let e = col("x").lt(lit_f64(3.0));
    /// assert!(matches!(e, scivex_frame::lazy::expr::Expr::BinaryOp { .. }));
    /// ```
    pub fn lt(self, other: Self) -> Self {
        Self::BinaryOp {
            left: Box::new(self),
            op: BinaryOp::Lt,
            right: Box::new(other),
        }
    }

    /// Less-than-or-equal comparison.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::lazy::expr::{col, lit_i64};
    /// let e = col("x").lt_eq(lit_i64(5));
    /// assert!(matches!(e, scivex_frame::lazy::expr::Expr::BinaryOp { .. }));
    /// ```
    pub fn lt_eq(self, other: Self) -> Self {
        Self::BinaryOp {
            left: Box::new(self),
            op: BinaryOp::LtEq,
            right: Box::new(other),
        }
    }

    /// Greater-than comparison.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::lazy::expr::{col, lit_i64};
    /// let e = col("x").gt(lit_i64(5));
    /// assert!(matches!(e, scivex_frame::lazy::expr::Expr::BinaryOp { .. }));
    /// ```
    pub fn gt(self, other: Self) -> Self {
        Self::BinaryOp {
            left: Box::new(self),
            op: BinaryOp::Gt,
            right: Box::new(other),
        }
    }

    /// Greater-than-or-equal comparison.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::lazy::expr::{col, lit_i64};
    /// let e = col("x").gt_eq(lit_i64(0));
    /// assert!(matches!(e, scivex_frame::lazy::expr::Expr::BinaryOp { .. }));
    /// ```
    pub fn gt_eq(self, other: Self) -> Self {
        Self::BinaryOp {
            left: Box::new(self),
            op: BinaryOp::GtEq,
            right: Box::new(other),
        }
    }

    /// Logical AND.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::lazy::expr::{col, lit_i64};
    /// let e = col("x").gt(lit_i64(0)).and(col("x").lt(lit_i64(10)));
    /// assert!(matches!(e, scivex_frame::lazy::expr::Expr::BinaryOp { .. }));
    /// ```
    pub fn and(self, other: Self) -> Self {
        Self::BinaryOp {
            left: Box::new(self),
            op: BinaryOp::And,
            right: Box::new(other),
        }
    }

    /// Logical OR.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::lazy::expr::{col, lit_i64};
    /// let e = col("x").lt(lit_i64(0)).or(col("x").gt(lit_i64(100)));
    /// assert!(matches!(e, scivex_frame::lazy::expr::Expr::BinaryOp { .. }));
    /// ```
    pub fn or(self, other: Self) -> Self {
        Self::BinaryOp {
            left: Box::new(self),
            op: BinaryOp::Or,
            right: Box::new(other),
        }
    }

    /// Logical NOT.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::lazy::expr::lit_bool;
    /// let e = lit_bool(true).not();
    /// assert!(matches!(e, scivex_frame::lazy::expr::Expr::Not(_)));
    /// ```
    #[allow(clippy::should_implement_trait)]
    pub fn not(self) -> Self {
        Self::Not(Box::new(self))
    }

    /// Add two expressions.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::lazy::expr::{col, lit_f64};
    /// let e = col("x").add(lit_f64(1.0));
    /// assert!(matches!(e, scivex_frame::lazy::expr::Expr::BinaryOp { .. }));
    /// ```
    #[allow(clippy::should_implement_trait)]
    pub fn add(self, other: Self) -> Self {
        Self::BinaryOp {
            left: Box::new(self),
            op: BinaryOp::Add,
            right: Box::new(other),
        }
    }

    /// Subtract two expressions.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::lazy::expr::{col, lit_f64};
    /// let e = col("x").sub(lit_f64(1.0));
    /// assert!(matches!(e, scivex_frame::lazy::expr::Expr::BinaryOp { .. }));
    /// ```
    #[allow(clippy::should_implement_trait)]
    pub fn sub(self, other: Self) -> Self {
        Self::BinaryOp {
            left: Box::new(self),
            op: BinaryOp::Sub,
            right: Box::new(other),
        }
    }

    /// Multiply two expressions.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::lazy::expr::{col, lit_f64};
    /// let e = col("x").mul(lit_f64(2.0));
    /// assert!(matches!(e, scivex_frame::lazy::expr::Expr::BinaryOp { .. }));
    /// ```
    #[allow(clippy::should_implement_trait)]
    pub fn mul(self, other: Self) -> Self {
        Self::BinaryOp {
            left: Box::new(self),
            op: BinaryOp::Mul,
            right: Box::new(other),
        }
    }

    /// Divide two expressions.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::lazy::expr::{col, lit_f64};
    /// let e = col("x").div(lit_f64(2.0));
    /// assert!(matches!(e, scivex_frame::lazy::expr::Expr::BinaryOp { .. }));
    /// ```
    #[allow(clippy::should_implement_trait)]
    pub fn div(self, other: Self) -> Self {
        Self::BinaryOp {
            left: Box::new(self),
            op: BinaryOp::Div,
            right: Box::new(other),
        }
    }
}
