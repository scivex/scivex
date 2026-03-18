//! Abstract syntax tree types for the SQL subset.

/// A parsed `SELECT` statement.
#[derive(Debug, Clone)]
pub struct SelectStatement {
    pub projections: Vec<SelectItem>,
    pub from: Vec<TableRef>,
    pub joins: Vec<JoinClause>,
    pub where_clause: Option<SqlExpr>,
    pub group_by: Vec<SqlExpr>,
    pub having: Option<SqlExpr>,
    pub order_by: Vec<OrderItem>,
    pub limit: Option<usize>,
}

/// A single item in the SELECT list.
#[derive(Debug, Clone)]
pub enum SelectItem {
    /// `SELECT *`
    Wildcard,
    /// An expression, optionally aliased: `expr AS alias`.
    Expr {
        expr: SqlExpr,
        alias: Option<String>,
    },
}

/// A table reference in the FROM clause.
#[derive(Debug, Clone)]
pub struct TableRef {
    pub name: String,
    pub alias: Option<String>,
}

/// A JOIN clause.
#[derive(Debug, Clone)]
pub struct JoinClause {
    pub join_type: JoinKind,
    pub table: TableRef,
    pub on: SqlExpr,
}

/// Kind of join.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinKind {
    Inner,
    Left,
    Right,
}

/// An SQL expression.
#[derive(Debug, Clone)]
pub enum SqlExpr {
    /// A bare column name.
    Column(String),
    /// A qualified column: `table.column`.
    QualifiedColumn { table: String, column: String },
    /// A literal value.
    Literal(SqlLiteral),
    /// A binary operation.
    BinaryOp {
        left: Box<SqlExpr>,
        op: BinaryOp,
        right: Box<SqlExpr>,
    },
    /// `NOT expr`
    UnaryNot(Box<SqlExpr>),
    /// A function call: `SUM(x)`, `COUNT(*)`, etc.
    Function { name: String, args: Vec<SqlExpr> },
    /// `expr IS NULL`
    IsNull(Box<SqlExpr>),
    /// `*` (used inside `COUNT(*)`)
    Wildcard,
}

/// A literal value.
#[derive(Debug, Clone)]
pub enum SqlLiteral {
    Integer(i64),
    Float(f64),
    String(String),
    Null,
}

/// A binary operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Eq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
    Plus,
    Minus,
    Mul,
    Div,
    And,
    Or,
}

/// An item in the ORDER BY clause.
#[derive(Debug, Clone)]
pub struct OrderItem {
    pub expr: SqlExpr,
    pub ascending: bool,
}
