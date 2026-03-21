//! SQL executor — evaluates a parsed `SelectStatement` against DataFrames.

use std::collections::HashMap;

use crate::dataframe::DataFrame;
use crate::dtype::DType;
use crate::error::{FrameError, Result};
use crate::series::string::StringSeries;
use crate::series::{AnySeries, Series};

use super::ast::{BinaryOp, JoinKind, SelectItem, SelectStatement, SqlExpr, SqlLiteral, TableRef};

/// Execute a `SelectStatement` against a table context.
#[allow(clippy::implicit_hasher)]
pub fn execute(stmt: &SelectStatement, tables: &HashMap<String, DataFrame>) -> Result<DataFrame> {
    // 1. Resolve FROM tables
    if stmt.from.is_empty() {
        return Err(FrameError::InvalidValue {
            reason: "SELECT requires at least one table in FROM".to_string(),
        });
    }

    let first_ref = &stmt.from[0];
    let mut df = resolve_table(tables, first_ref)?;

    // Build alias map: alias -> original table name
    let mut alias_map: HashMap<String, String> = HashMap::new();
    alias_map.insert(
        first_ref
            .alias
            .as_deref()
            .unwrap_or(&first_ref.name)
            .to_string(),
        first_ref.name.clone(),
    );
    // Also map original name
    alias_map.insert(first_ref.name.clone(), first_ref.name.clone());

    // 2. Execute JOINs
    for join in &stmt.joins {
        let right = resolve_table(tables, &join.table)?;

        // Register alias for right table
        let right_alias = join
            .table
            .alias
            .as_deref()
            .unwrap_or(&join.table.name)
            .to_string();
        alias_map.insert(right_alias, join.table.name.clone());
        alias_map.insert(join.table.name.clone(), join.table.name.clone());

        // Extract ON condition — we support `t1.col = t2.col` patterns
        let (left_col, right_col) = extract_join_columns(&join.on)?;

        // Resolve qualified column names
        let left_col_name = resolve_qualified_col(&left_col, &df)?;
        let right_col_name = resolve_qualified_col(&right_col, &right)?;

        let join_type = match join.join_type {
            JoinKind::Inner => crate::dataframe::join::JoinType::Inner,
            JoinKind::Left => crate::dataframe::join::JoinType::Left,
            JoinKind::Right => crate::dataframe::join::JoinType::Right,
        };

        df = df.join_on(
            &right,
            &[left_col_name.as_str()],
            &[right_col_name.as_str()],
            join_type,
        )?;
    }

    // 3. Apply WHERE filter
    if let Some(ref where_expr) = stmt.where_clause {
        let mask = eval_expr_to_bool_mask(where_expr, &df)?;
        df = df.filter(&mask)?;
    }

    // 4. GROUP BY + aggregation
    if stmt.group_by.is_empty() {
        // Check if there are aggregate functions without GROUP BY (full-table agg)
        let has_agg = stmt
            .projections
            .iter()
            .any(|item| matches!(item, SelectItem::Expr { expr, .. } if expr_has_aggregate(expr)));

        if has_agg {
            df = execute_full_aggregate(&df, &stmt.projections)?;
        } else {
            // 6. Compute SELECT projections (non-aggregate)
            df = execute_projection(&df, &stmt.projections)?;
        }
    } else {
        df = execute_group_by(&df, &stmt.group_by, &stmt.projections)?;

        // 5. Apply HAVING filter
        if let Some(ref having_expr) = stmt.having {
            let mask = eval_expr_to_bool_mask(having_expr, &df)?;
            df = df.filter(&mask)?;
        }
    }

    // 7. Apply ORDER BY
    if !stmt.order_by.is_empty() {
        for order_item in stmt.order_by.iter().rev() {
            let col_name = expr_to_column_name(&order_item.expr)?;
            df = df.sort_by(&col_name, order_item.ascending)?;
        }
    }

    // 8. Apply LIMIT
    if let Some(limit) = stmt.limit {
        df = df.head(limit);
    }

    Ok(df)
}

// ---------------------------------------------------------------------------
// Table resolution
// ---------------------------------------------------------------------------

fn resolve_table(tables: &HashMap<String, DataFrame>, table_ref: &TableRef) -> Result<DataFrame> {
    tables
        .get(&table_ref.name)
        .cloned()
        .ok_or_else(|| FrameError::InvalidValue {
            reason: format!("table not found: {:?}", table_ref.name),
        })
}

// ---------------------------------------------------------------------------
// Join column extraction
// ---------------------------------------------------------------------------

/// Extract the left and right column references from a join ON condition
/// like `t1.col = t2.col` or `col1 = col2`.
fn extract_join_columns(expr: &SqlExpr) -> Result<(SqlExpr, SqlExpr)> {
    if let SqlExpr::BinaryOp {
        left, op, right, ..
    } = expr
    {
        if *op != BinaryOp::Eq {
            return Err(FrameError::InvalidValue {
                reason: "JOIN ON condition must use = operator".to_string(),
            });
        }
        Ok((*left.clone(), *right.clone()))
    } else {
        Err(FrameError::InvalidValue {
            reason: "JOIN ON must be a simple equality condition".to_string(),
        })
    }
}

/// Resolve a column expression to a column name in a DataFrame.
fn resolve_qualified_col(expr: &SqlExpr, df: &DataFrame) -> Result<String> {
    match expr {
        SqlExpr::Column(name) => {
            df.column(name)?;
            Ok(name.clone())
        }
        SqlExpr::QualifiedColumn { column, .. } => {
            df.column(column)?;
            Ok(column.clone())
        }
        _ => Err(FrameError::InvalidValue {
            reason: "JOIN ON operand must be a column reference".to_string(),
        }),
    }
}

// ---------------------------------------------------------------------------
// Projection
// ---------------------------------------------------------------------------

fn execute_projection(df: &DataFrame, projections: &[SelectItem]) -> Result<DataFrame> {
    // Wildcard shortcut: return all columns
    if projections.len() == 1 && matches!(&projections[0], SelectItem::Wildcard) {
        return Ok(df.clone());
    }

    let mut columns: Vec<Box<dyn AnySeries>> = Vec::new();
    for item in projections {
        match item {
            SelectItem::Wildcard => {
                for col in df.columns() {
                    columns.push(col.clone_box());
                }
            }
            SelectItem::Expr { expr, alias } => {
                let col = eval_expr_to_series(expr, df)?;
                if let Some(alias_name) = alias {
                    columns.push(col.rename_box(alias_name));
                } else {
                    columns.push(col);
                }
            }
        }
    }
    DataFrame::new(columns)
}

// ---------------------------------------------------------------------------
// GROUP BY
// ---------------------------------------------------------------------------

fn execute_group_by(
    df: &DataFrame,
    group_exprs: &[SqlExpr],
    projections: &[SelectItem],
) -> Result<DataFrame> {
    // Resolve group-by column names
    let group_cols: Vec<String> = group_exprs
        .iter()
        .map(expr_to_column_name)
        .collect::<Result<_>>()?;

    let group_col_refs: Vec<&str> = group_cols.iter().map(String::as_str).collect();

    // Build groups using display_value-based composite key (same as GroupBy)
    let nrows = df.nrows();
    let mut group_map: HashMap<Vec<String>, Vec<usize>> = HashMap::new();
    let mut group_order: Vec<Vec<String>> = Vec::new();

    for row in 0..nrows {
        let key: Vec<String> = group_col_refs
            .iter()
            .map(|&c| {
                df.column(c)
                    .expect("group column exists")
                    .display_value(row)
            })
            .collect();
        if !group_map.contains_key(&key) {
            group_order.push(key.clone());
        }
        group_map.entry(key).or_default().push(row);
    }

    let groups: Vec<(Vec<String>, Vec<usize>)> = group_order
        .into_iter()
        .map(|k| {
            let indices = group_map.remove(&k).expect("key present");
            (k, indices)
        })
        .collect();

    // Build result columns from projection list
    let mut result_columns: Vec<Box<dyn AnySeries>> = Vec::new();

    for item in projections {
        match item {
            SelectItem::Wildcard => {
                return Err(FrameError::InvalidValue {
                    reason: "SELECT * not supported with GROUP BY".to_string(),
                });
            }
            SelectItem::Expr { expr, alias } => {
                let col = eval_group_expr(expr, df, &groups, &group_col_refs)?;
                if let Some(alias_name) = alias {
                    result_columns.push(col.rename_box(alias_name));
                } else {
                    result_columns.push(col);
                }
            }
        }
    }

    DataFrame::new(result_columns)
}

/// Evaluate an expression in a GROUP BY context.
fn eval_group_expr(
    expr: &SqlExpr,
    df: &DataFrame,
    groups: &[(Vec<String>, Vec<usize>)],
    group_cols: &[&str],
) -> Result<Box<dyn AnySeries>> {
    match expr {
        SqlExpr::Column(name) => {
            // If it's a group column, produce the group key values
            if group_cols.contains(&name.as_str()) {
                let ki = group_cols.iter().position(|&g| g == name).unwrap();
                let data: Vec<String> = groups.iter().map(|(keys, _)| keys[ki].clone()).collect();
                // Try to produce the same type as the source column
                let src = df.column(name)?;
                Ok(rebuild_group_key_column(name, &data, src.dtype()))
            } else {
                Err(FrameError::InvalidValue {
                    reason: format!(
                        "column {name:?} must appear in GROUP BY or be in an aggregate"
                    ),
                })
            }
        }
        SqlExpr::Function { name, args } => {
            let func_name = name.to_uppercase();
            eval_aggregate_function(&func_name, args, df, groups)
        }
        _ => Err(FrameError::InvalidValue {
            reason: "unsupported expression in GROUP BY projection".to_string(),
        }),
    }
}

/// Rebuild a group key column with the correct dtype.
fn rebuild_group_key_column(
    name: &str,
    string_values: &[String],
    dtype: DType,
) -> Box<dyn AnySeries> {
    macro_rules! try_parse_series {
        ($ty:ty, $dtype_variant:ident) => {
            if dtype == DType::$dtype_variant {
                let data: Vec<$ty> = string_values
                    .iter()
                    .map(|s| s.parse::<$ty>().unwrap_or(<$ty>::default()))
                    .collect();
                return Box::new(Series::new(name.to_string(), data));
            }
        };
    }
    try_parse_series!(i32, I32);
    try_parse_series!(i64, I64);
    try_parse_series!(f64, F64);
    try_parse_series!(f32, F32);
    try_parse_series!(i16, I16);
    try_parse_series!(i8, I8);
    try_parse_series!(u64, U64);
    try_parse_series!(u32, U32);
    try_parse_series!(u16, U16);
    try_parse_series!(u8, U8);

    // Fallback: string series
    Box::new(StringSeries::new(name.to_string(), string_values.to_vec()))
}

/// Evaluate an aggregate function over groups.
fn eval_aggregate_function(
    func_name: &str,
    args: &[SqlExpr],
    df: &DataFrame,
    groups: &[(Vec<String>, Vec<usize>)],
) -> Result<Box<dyn AnySeries>> {
    match func_name {
        "COUNT" => {
            // COUNT(*) or COUNT(col)
            let col_name = if args.len() == 1 {
                if let SqlExpr::Wildcard = &args[0] {
                    "count".to_string()
                } else {
                    let n = expr_to_column_name(&args[0])?;
                    format!("COUNT({n})")
                }
            } else {
                "count".to_string()
            };

            #[allow(clippy::cast_possible_wrap)]
            let counts: Vec<i64> = groups.iter().map(|(_, idx)| idx.len() as i64).collect();
            Ok(Box::new(Series::new(col_name, counts)))
        }
        "SUM" | "AVG" | "MIN" | "MAX" => {
            if args.len() != 1 {
                return Err(FrameError::InvalidValue {
                    reason: format!("{func_name} requires exactly one argument"),
                });
            }
            let src_col_name = expr_to_column_name(&args[0])?;
            let src = df.column(&src_col_name)?;
            let result_name = format!("{func_name}({src_col_name})");
            aggregate_typed_column(func_name, &result_name, src, groups)
        }
        _ => Err(FrameError::InvalidValue {
            reason: format!("unknown aggregate function: {func_name}"),
        }),
    }
}

/// Aggregate a typed column by dispatching on its DType.
fn aggregate_typed_column(
    func_name: &str,
    result_name: &str,
    col: &dyn AnySeries,
    groups: &[(Vec<String>, Vec<usize>)],
) -> Result<Box<dyn AnySeries>> {
    let dtype = col.dtype();

    macro_rules! dispatch {
        ($ty:ty, $dtype_variant:ident) => {
            if dtype == DType::$dtype_variant {
                let typed =
                    col.as_any()
                        .downcast_ref::<Series<$ty>>()
                        .ok_or(FrameError::TypeMismatch {
                            expected: stringify!($ty),
                            got: "unknown",
                        })?;
                return agg_series_typed::<$ty>(func_name, result_name, typed, groups);
            }
        };
    }

    dispatch!(f64, F64);
    dispatch!(f32, F32);
    dispatch!(i64, I64);
    dispatch!(i32, I32);
    dispatch!(i16, I16);
    dispatch!(i8, I8);
    dispatch!(u64, U64);
    dispatch!(u32, U32);
    dispatch!(u16, U16);
    dispatch!(u8, U8);

    Err(FrameError::InvalidValue {
        reason: format!("cannot aggregate column of type {dtype}"),
    })
}

fn agg_series_typed<T: scivex_core::Scalar + crate::dtype::HasDType + 'static>(
    func_name: &str,
    result_name: &str,
    series: &Series<T>,
    groups: &[(Vec<String>, Vec<usize>)],
) -> Result<Box<dyn AnySeries>> {
    let data = series.as_slice();
    let values: Vec<T> = match func_name {
        "SUM" => groups
            .iter()
            .map(|(_, idx)| idx.iter().fold(T::zero(), |acc, &i| acc + data[i]))
            .collect(),
        "MIN" => groups
            .iter()
            .map(|(_, idx)| {
                idx.iter()
                    .map(|&i| data[i])
                    .reduce(|a, b| if b < a { b } else { a })
                    .unwrap_or_else(T::zero)
            })
            .collect(),
        "MAX" => groups
            .iter()
            .map(|(_, idx)| {
                idx.iter()
                    .map(|&i| data[i])
                    .reduce(|a, b| if b > a { b } else { a })
                    .unwrap_or_else(T::zero)
            })
            .collect(),
        "AVG" => groups
            .iter()
            .map(|(_, idx)| {
                let sum = idx.iter().fold(T::zero(), |acc, &i| acc + data[i]);
                sum / T::from_usize(idx.len())
            })
            .collect(),
        _ => {
            return Err(FrameError::InvalidValue {
                reason: format!("unknown aggregate: {func_name}"),
            });
        }
    };
    Ok(Box::new(Series::new(result_name.to_string(), values)))
}

// ---------------------------------------------------------------------------
// Full-table aggregation (no GROUP BY)
// ---------------------------------------------------------------------------

fn execute_full_aggregate(df: &DataFrame, projections: &[SelectItem]) -> Result<DataFrame> {
    // Treat the entire table as one group
    let all_indices: Vec<usize> = (0..df.nrows()).collect();
    let groups = vec![(vec![], all_indices)];
    let group_cols: Vec<&str> = Vec::new();

    let mut result_columns: Vec<Box<dyn AnySeries>> = Vec::new();
    for item in projections {
        match item {
            SelectItem::Wildcard => {
                return Err(FrameError::InvalidValue {
                    reason: "SELECT * not supported with aggregate functions".to_string(),
                });
            }
            SelectItem::Expr { expr, alias } => {
                let col = eval_group_expr(expr, df, &groups, &group_cols)?;
                if let Some(alias_name) = alias {
                    result_columns.push(col.rename_box(alias_name));
                } else {
                    result_columns.push(col);
                }
            }
        }
    }
    DataFrame::new(result_columns)
}

// ---------------------------------------------------------------------------
// Expression evaluation (column-at-a-time)
// ---------------------------------------------------------------------------

/// Evaluate an expression to a boolean mask for WHERE / HAVING filtering.
fn eval_expr_to_bool_mask(expr: &SqlExpr, df: &DataFrame) -> Result<Vec<bool>> {
    let nrows = df.nrows();
    match expr {
        SqlExpr::BinaryOp {
            left, op, right, ..
        } => match op {
            BinaryOp::And => {
                let l = eval_expr_to_bool_mask(left, df)?;
                let r = eval_expr_to_bool_mask(right, df)?;
                Ok(l.iter().zip(r.iter()).map(|(&a, &b)| a && b).collect())
            }
            BinaryOp::Or => {
                let l = eval_expr_to_bool_mask(left, df)?;
                let r = eval_expr_to_bool_mask(right, df)?;
                Ok(l.iter().zip(r.iter()).map(|(&a, &b)| a || b).collect())
            }
            _ => {
                // Comparison: evaluate left and right as display values
                let l_vals = eval_expr_display_values(left, df)?;
                let r_vals = eval_expr_display_values(right, df)?;
                let mut mask = Vec::with_capacity(nrows);
                for i in 0..nrows {
                    let cmp = compare_values(&l_vals[i], &r_vals[i]);
                    let result = match op {
                        BinaryOp::Eq => cmp == std::cmp::Ordering::Equal,
                        BinaryOp::NotEq => cmp != std::cmp::Ordering::Equal,
                        BinaryOp::Lt => cmp == std::cmp::Ordering::Less,
                        BinaryOp::LtEq => cmp != std::cmp::Ordering::Greater,
                        BinaryOp::Gt => cmp == std::cmp::Ordering::Greater,
                        BinaryOp::GtEq => cmp != std::cmp::Ordering::Less,
                        _ => unreachable!(),
                    };
                    mask.push(result);
                }
                Ok(mask)
            }
        },
        SqlExpr::UnaryNot(inner) => {
            let m = eval_expr_to_bool_mask(inner, df)?;
            Ok(m.iter().map(|&v| !v).collect())
        }
        SqlExpr::IsNull(inner) => {
            let col_name = expr_to_column_name(inner)?;
            let col = df.column(&col_name)?;
            Ok((0..nrows).map(|i| col.is_null(i)).collect())
        }
        _ => Err(FrameError::InvalidValue {
            reason: "unsupported expression in WHERE clause".to_string(),
        }),
    }
}

/// A value that can be compared: either numeric or string.
#[derive(Clone)]
enum CmpValue {
    Numeric(f64),
    Text(String),
}

/// Evaluate an expression to comparable values, one per row.
fn eval_expr_display_values(expr: &SqlExpr, df: &DataFrame) -> Result<Vec<CmpValue>> {
    let nrows = df.nrows();
    match expr {
        SqlExpr::Column(name) => {
            let col = df.column(name)?;
            Ok(display_values_for_column(col, nrows))
        }
        SqlExpr::QualifiedColumn { column, .. } => {
            let col = df.column(column)?;
            Ok(display_values_for_column(col, nrows))
        }
        SqlExpr::Literal(lit) => {
            let val = match lit {
                SqlLiteral::Integer(n) => CmpValue::Numeric(*n as f64),
                SqlLiteral::Float(f) => CmpValue::Numeric(*f),
                SqlLiteral::String(s) => CmpValue::Text(s.clone()),
                SqlLiteral::Null => CmpValue::Text("null".to_string()),
            };
            Ok(vec![val; nrows])
        }
        _ => Err(FrameError::InvalidValue {
            reason: "unsupported expression type in comparison".to_string(),
        }),
    }
}

fn display_values_for_column(col: &dyn AnySeries, nrows: usize) -> Vec<CmpValue> {
    (0..nrows)
        .map(|i| {
            let s = col.display_value(i);
            match s.parse::<f64>() {
                Ok(f) => CmpValue::Numeric(f),
                Err(_) => CmpValue::Text(s),
            }
        })
        .collect()
}

fn compare_values(a: &CmpValue, b: &CmpValue) -> std::cmp::Ordering {
    match (a, b) {
        (CmpValue::Numeric(fa), CmpValue::Numeric(fb)) => {
            fa.partial_cmp(fb).unwrap_or(std::cmp::Ordering::Equal)
        }
        (CmpValue::Text(sa), CmpValue::Text(sb)) => sa.cmp(sb),
        // Mixed: fall back to string comparison
        (CmpValue::Numeric(f), CmpValue::Text(s)) => {
            let fs = format!("{f}");
            fs.cmp(s)
        }
        (CmpValue::Text(s), CmpValue::Numeric(f)) => {
            let fs = format!("{f}");
            s.cmp(&fs)
        }
    }
}

/// Evaluate an expression to a Series (for SELECT projections).
fn eval_expr_to_series(expr: &SqlExpr, df: &DataFrame) -> Result<Box<dyn AnySeries>> {
    match expr {
        SqlExpr::Column(name) => Ok(df.column(name)?.clone_box()),
        SqlExpr::QualifiedColumn { column, .. } => Ok(df.column(column)?.clone_box()),
        SqlExpr::Literal(lit) => {
            let nrows = df.nrows();
            match lit {
                SqlLiteral::Integer(n) => Ok(Box::new(Series::new(
                    "literal".to_string(),
                    vec![*n; nrows],
                ))),
                SqlLiteral::Float(f) => Ok(Box::new(Series::new(
                    "literal".to_string(),
                    vec![*f; nrows],
                ))),
                SqlLiteral::String(s) => Ok(Box::new(StringSeries::new(
                    "literal".to_string(),
                    vec![s.clone(); nrows],
                ))),
                SqlLiteral::Null => Ok(Box::new(Series::<i64>::with_nulls(
                    "literal".to_string(),
                    vec![0; nrows],
                    vec![true; nrows],
                )?)),
            }
        }
        SqlExpr::Function { name, args } => eval_aggregate_function(
            &name.to_uppercase(),
            args,
            df,
            &[(vec![], (0..df.nrows()).collect())],
        ),
        _ => Err(FrameError::InvalidValue {
            reason: "unsupported expression in SELECT".to_string(),
        }),
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract a column name from a simple column expression.
fn expr_to_column_name(expr: &SqlExpr) -> Result<String> {
    match expr {
        SqlExpr::Column(name) => Ok(name.clone()),
        SqlExpr::QualifiedColumn { column, .. } => Ok(column.clone()),
        SqlExpr::Function { name, args } => {
            // For ORDER BY on aggregated result, the column name is e.g. "SUM(val)"
            if args.len() == 1 {
                if let SqlExpr::Wildcard = &args[0] {
                    Ok(format!("{name}(*)"))
                } else {
                    let inner = expr_to_column_name(&args[0])?;
                    Ok(format!("{name}({inner})"))
                }
            } else if args.is_empty() {
                Ok(format!("{name}(*)"))
            } else {
                Err(FrameError::InvalidValue {
                    reason: "cannot derive column name from multi-arg function".to_string(),
                })
            }
        }
        _ => Err(FrameError::InvalidValue {
            reason: "expected a column name".to_string(),
        }),
    }
}

/// Check whether an expression contains an aggregate function call.
fn expr_has_aggregate(expr: &SqlExpr) -> bool {
    match expr {
        SqlExpr::Function { name, .. } => {
            matches!(
                name.to_uppercase().as_str(),
                "SUM" | "AVG" | "MIN" | "MAX" | "COUNT"
            )
        }
        SqlExpr::BinaryOp { left, right, .. } => {
            expr_has_aggregate(left) || expr_has_aggregate(right)
        }
        SqlExpr::UnaryNot(inner) => expr_has_aggregate(inner),
        _ => false,
    }
}
