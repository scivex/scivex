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
pub fn col(name: &str) -> Expr {
    Expr::Col(name.to_string())
}

/// Create a literal f64 expression.
pub fn lit_f64(val: f64) -> Expr {
    Expr::LitF64(val)
}

/// Create a literal i64 expression.
pub fn lit_i64(val: i64) -> Expr {
    Expr::LitI64(val)
}

/// Create a literal string expression.
pub fn lit_str(val: &str) -> Expr {
    Expr::LitStr(val.to_string())
}

/// Create a literal boolean expression.
pub fn lit_bool(val: bool) -> Expr {
    Expr::LitBool(val)
}

impl Expr {
    /// Alias this expression (rename the output column).
    pub fn alias(self, name: &str) -> Self {
        Self::Alias {
            expr: Box::new(self),
            name: name.to_string(),
        }
    }

    /// Aggregate: sum.
    pub fn sum(self) -> Self {
        Self::Agg {
            expr: Box::new(self),
            func: AggFunc::Sum,
        }
    }

    /// Aggregate: mean.
    pub fn mean(self) -> Self {
        Self::Agg {
            expr: Box::new(self),
            func: AggFunc::Mean,
        }
    }

    /// Aggregate: min.
    pub fn min(self) -> Self {
        Self::Agg {
            expr: Box::new(self),
            func: AggFunc::Min,
        }
    }

    /// Aggregate: max.
    pub fn max(self) -> Self {
        Self::Agg {
            expr: Box::new(self),
            func: AggFunc::Max,
        }
    }

    /// Aggregate: count.
    pub fn count(self) -> Self {
        Self::Agg {
            expr: Box::new(self),
            func: AggFunc::Count,
        }
    }

    /// Aggregate: first.
    pub fn first(self) -> Self {
        Self::Agg {
            expr: Box::new(self),
            func: AggFunc::First,
        }
    }

    /// Aggregate: last.
    pub fn last(self) -> Self {
        Self::Agg {
            expr: Box::new(self),
            func: AggFunc::Last,
        }
    }

    /// Sort ascending.
    pub fn sort_asc(self) -> Self {
        Self::Sort {
            expr: Box::new(self),
            ascending: true,
        }
    }

    /// Sort descending.
    pub fn sort_desc(self) -> Self {
        Self::Sort {
            expr: Box::new(self),
            ascending: false,
        }
    }

    /// Equality comparison.
    pub fn eq(self, other: Self) -> Self {
        Self::BinaryOp {
            left: Box::new(self),
            op: BinaryOp::Eq,
            right: Box::new(other),
        }
    }

    /// Not-equal comparison.
    pub fn neq(self, other: Self) -> Self {
        Self::BinaryOp {
            left: Box::new(self),
            op: BinaryOp::NotEq,
            right: Box::new(other),
        }
    }

    /// Less-than comparison.
    pub fn lt(self, other: Self) -> Self {
        Self::BinaryOp {
            left: Box::new(self),
            op: BinaryOp::Lt,
            right: Box::new(other),
        }
    }

    /// Less-than-or-equal comparison.
    pub fn lt_eq(self, other: Self) -> Self {
        Self::BinaryOp {
            left: Box::new(self),
            op: BinaryOp::LtEq,
            right: Box::new(other),
        }
    }

    /// Greater-than comparison.
    pub fn gt(self, other: Self) -> Self {
        Self::BinaryOp {
            left: Box::new(self),
            op: BinaryOp::Gt,
            right: Box::new(other),
        }
    }

    /// Greater-than-or-equal comparison.
    pub fn gt_eq(self, other: Self) -> Self {
        Self::BinaryOp {
            left: Box::new(self),
            op: BinaryOp::GtEq,
            right: Box::new(other),
        }
    }

    /// Logical AND.
    pub fn and(self, other: Self) -> Self {
        Self::BinaryOp {
            left: Box::new(self),
            op: BinaryOp::And,
            right: Box::new(other),
        }
    }

    /// Logical OR.
    pub fn or(self, other: Self) -> Self {
        Self::BinaryOp {
            left: Box::new(self),
            op: BinaryOp::Or,
            right: Box::new(other),
        }
    }

    /// Logical NOT.
    #[allow(clippy::should_implement_trait)]
    pub fn not(self) -> Self {
        Self::Not(Box::new(self))
    }

    /// Add two expressions.
    #[allow(clippy::should_implement_trait)]
    pub fn add(self, other: Self) -> Self {
        Self::BinaryOp {
            left: Box::new(self),
            op: BinaryOp::Add,
            right: Box::new(other),
        }
    }

    /// Subtract two expressions.
    #[allow(clippy::should_implement_trait)]
    pub fn sub(self, other: Self) -> Self {
        Self::BinaryOp {
            left: Box::new(self),
            op: BinaryOp::Sub,
            right: Box::new(other),
        }
    }

    /// Multiply two expressions.
    #[allow(clippy::should_implement_trait)]
    pub fn mul(self, other: Self) -> Self {
        Self::BinaryOp {
            left: Box::new(self),
            op: BinaryOp::Mul,
            right: Box::new(other),
        }
    }

    /// Divide two expressions.
    #[allow(clippy::should_implement_trait)]
    pub fn div(self, other: Self) -> Self {
        Self::BinaryOp {
            left: Box::new(self),
            op: BinaryOp::Div,
            right: Box::new(other),
        }
    }
}
