//! Execute a [`LogicalPlan`] into a materialized [`DataFrame`].

use crate::dataframe::DataFrame;
use crate::error::{FrameError, Result};
use crate::groupby::AggFunc;
use crate::series::string::StringSeries;
use crate::series::{AnySeries, Series};

use super::expr::{BinaryOp, Expr};
use super::plan::LogicalPlan;

/// Execute a logical plan and return the resulting `DataFrame`.
pub fn execute(plan: &LogicalPlan) -> Result<DataFrame> {
    match plan {
        LogicalPlan::Scan(df) => Ok(df.clone()),
        LogicalPlan::Filter { input, predicate } => exec_filter(input, predicate),
        LogicalPlan::Select { input, exprs } => exec_select(input, exprs),
        LogicalPlan::Sort {
            input,
            by_column,
            ascending,
        } => {
            let df = execute(input)?;
            df.sort_by(by_column, *ascending)
        }
        LogicalPlan::Limit { input, n } => {
            let df = execute(input)?;
            Ok(df.head(*n))
        }
        LogicalPlan::GroupByAgg {
            input,
            group_cols,
            agg_exprs,
        } => exec_groupby(input, group_cols, agg_exprs),
    }
}

// ---------------------------------------------------------------------------
// Filter execution
// ---------------------------------------------------------------------------

fn exec_filter(input: &LogicalPlan, predicate: &Expr) -> Result<DataFrame> {
    let df = execute(input)?;
    let mask = eval_bool_expr(&df, predicate)?;
    df.filter(&mask)
}

// ---------------------------------------------------------------------------
// Select execution
// ---------------------------------------------------------------------------

fn exec_select(input: &LogicalPlan, exprs: &[Expr]) -> Result<DataFrame> {
    let df = execute(input)?;
    let mut columns: Vec<Box<dyn AnySeries>> = Vec::with_capacity(exprs.len());
    for expr in exprs {
        columns.push(eval_to_series(&df, expr)?);
    }
    DataFrame::new(columns)
}

// ---------------------------------------------------------------------------
// GroupBy execution
// ---------------------------------------------------------------------------

fn exec_groupby(
    input: &LogicalPlan,
    group_cols: &[String],
    agg_exprs: &[Expr],
) -> Result<DataFrame> {
    let df = execute(input)?;
    let keys: Vec<&str> = group_cols.iter().map(String::as_str).collect();
    let gb = df.groupby(&keys)?;

    // Start with the group key columns from the grouped result.
    let count_df = gb.count()?;
    let mut result_cols: Vec<Box<dyn AnySeries>> = Vec::new();
    for key_name in &keys {
        let col = count_df.column(key_name)?;
        result_cols.push(col.clone_box());
    }

    // Apply each aggregation expression.
    for agg_expr in agg_exprs {
        let (col_name, func, alias) = extract_agg_info(agg_expr)?;
        let agg_df = apply_agg_func(&gb, &col_name, func)?;
        let out_name = alias.as_deref().unwrap_or(&col_name);
        let col = agg_df.column(&col_name)?;
        result_cols.push(col.rename_box(out_name));
    }

    DataFrame::new(result_cols)
}

/// Extract (column_name, agg_func, optional_alias) from an Agg expression.
fn extract_agg_info(expr: &Expr) -> Result<(String, AggFunc, Option<String>)> {
    match expr {
        Expr::Agg { expr: inner, func } => {
            let col_name = extract_col_name(inner)?;
            Ok((col_name, *func, None))
        }
        Expr::Alias { expr: inner, name } => {
            let (col_name, func, _) = extract_agg_info(inner)?;
            Ok((col_name, func, Some(name.clone())))
        }
        _ => Err(FrameError::InvalidArgument {
            reason: "groupby agg_exprs must be Agg or Alias(Agg) expressions",
        }),
    }
}

fn extract_col_name(expr: &Expr) -> Result<String> {
    match expr {
        Expr::Col(name) => Ok(name.clone()),
        _ => Err(FrameError::InvalidArgument {
            reason: "expected a column reference inside aggregation",
        }),
    }
}

fn apply_agg_func(
    gb: &crate::groupby::GroupBy<'_>,
    col_name: &str,
    func: AggFunc,
) -> Result<DataFrame> {
    match func {
        AggFunc::Sum => gb.agg(col_name, AggFunc::Sum),
        AggFunc::Mean => gb.agg(col_name, AggFunc::Mean),
        AggFunc::Min => gb.agg(col_name, AggFunc::Min),
        AggFunc::Max => gb.agg(col_name, AggFunc::Max),
        AggFunc::Count => gb.agg(col_name, AggFunc::Count),
        AggFunc::First => gb.agg(col_name, AggFunc::First),
        AggFunc::Last => gb.agg(col_name, AggFunc::Last),
    }
}

// ---------------------------------------------------------------------------
// Expression evaluation
// ---------------------------------------------------------------------------

/// Evaluate a boolean-producing expression to a mask vector.
fn eval_bool_expr(df: &DataFrame, expr: &Expr) -> Result<Vec<bool>> {
    let n = df.nrows();
    match expr {
        Expr::LitBool(v) => Ok(vec![*v; n]),
        Expr::Not(inner) => {
            let mask = eval_bool_expr(df, inner)?;
            Ok(mask.into_iter().map(|b| !b).collect())
        }
        Expr::BinaryOp { left, op, right } => eval_binary_bool(df, left, *op, right),
        Expr::Col(name) => {
            // Try to interpret as a bool column
            let col = df.column(name)?;
            let mut mask = Vec::with_capacity(n);
            for i in 0..n {
                let val = col.display_value(i);
                mask.push(val == "true" || val == "1");
            }
            Ok(mask)
        }
        _ => Err(FrameError::InvalidArgument {
            reason: "expression does not produce a boolean result",
        }),
    }
}

/// Evaluate a binary operation that produces booleans.
fn eval_binary_bool(df: &DataFrame, left: &Expr, op: BinaryOp, right: &Expr) -> Result<Vec<bool>> {
    let n = df.nrows();

    // Handle logical AND/OR on bool sub-expressions
    if op == BinaryOp::And {
        let l = eval_bool_expr(df, left)?;
        let r = eval_bool_expr(df, right)?;
        return Ok(l.into_iter().zip(r).map(|(a, b)| a && b).collect());
    }
    if op == BinaryOp::Or {
        let l = eval_bool_expr(df, left)?;
        let r = eval_bool_expr(df, right)?;
        return Ok(l.into_iter().zip(r).map(|(a, b)| a || b).collect());
    }

    // For comparison ops, evaluate both sides as f64 values
    let lv = eval_f64_vec(df, left, n)?;
    let rv = eval_f64_vec(df, right, n)?;

    let result = lv
        .into_iter()
        .zip(rv)
        .map(|(a, b)| match op {
            BinaryOp::Eq => (a - b).abs() < f64::EPSILON,
            BinaryOp::NotEq => (a - b).abs() >= f64::EPSILON,
            BinaryOp::Lt => a < b,
            BinaryOp::LtEq => a <= b,
            BinaryOp::Gt => a > b,
            BinaryOp::GtEq => a >= b,
            BinaryOp::And | BinaryOp::Or => unreachable!(),
            BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div => false,
        })
        .collect();
    Ok(result)
}

/// Evaluate an expression to a vector of f64 values.
fn eval_f64_vec(df: &DataFrame, expr: &Expr, n: usize) -> Result<Vec<f64>> {
    match expr {
        Expr::Col(name) => {
            let col = df.column(name)?;
            let mut vals = Vec::with_capacity(n);
            for i in 0..n {
                let v: f64 =
                    col.display_value(i)
                        .parse()
                        .map_err(|_| FrameError::InvalidArgument {
                            reason: "cannot convert column value to f64 for comparison",
                        })?;
                vals.push(v);
            }
            Ok(vals)
        }
        Expr::LitF64(v) => Ok(vec![*v; n]),
        Expr::LitI64(v) => Ok(vec![*v as f64; n]),
        Expr::BinaryOp { left, op, right } => {
            let lv = eval_f64_vec(df, left, n)?;
            let rv = eval_f64_vec(df, right, n)?;
            let result = lv
                .into_iter()
                .zip(rv)
                .map(|(a, b)| match op {
                    BinaryOp::Add => a + b,
                    BinaryOp::Sub => a - b,
                    BinaryOp::Mul => a * b,
                    BinaryOp::Div => a / b,
                    _ => f64::NAN,
                })
                .collect();
            Ok(result)
        }
        _ => Err(FrameError::InvalidArgument {
            reason: "expression cannot be evaluated as f64",
        }),
    }
}

/// Evaluate an expression into a named `Series` (boxed as `AnySeries`).
fn eval_to_series(df: &DataFrame, expr: &Expr) -> Result<Box<dyn AnySeries>> {
    let n = df.nrows();
    match expr {
        Expr::Col(name) => Ok(df.column(name)?.clone_box()),
        Expr::Alias { expr: inner, name } => {
            let series = eval_to_series(df, inner)?;
            Ok(series.rename_box(name))
        }
        Expr::LitF64(v) => Ok(Box::new(Series::new("literal", vec![*v; n]))),
        Expr::LitI64(v) => Ok(Box::new(Series::new("literal", vec![*v; n]))),
        Expr::LitStr(v) => Ok(Box::new(StringSeries::from_strs(
            "literal",
            &vec![v.as_str(); n],
        ))),
        Expr::LitBool(v) => {
            let int_val = i32::from(*v);
            Ok(Box::new(Series::new("literal", vec![int_val; n])))
        }
        Expr::BinaryOp { left, op, right } => {
            let vals = eval_f64_vec(
                df,
                &Expr::BinaryOp {
                    left: left.clone(),
                    op: *op,
                    right: right.clone(),
                },
                n,
            )?;
            let name = format_binary_name(left, *op, right);
            Ok(Box::new(Series::new(name, vals)))
        }
        _ => Err(FrameError::InvalidArgument {
            reason: "expression cannot be evaluated as a Series in select context",
        }),
    }
}

/// Generate a default column name for a binary expression.
fn format_binary_name(left: &Expr, op: BinaryOp, right: &Expr) -> String {
    let l = expr_name(left);
    let r = expr_name(right);
    let op_str = match op {
        BinaryOp::Add => "+",
        BinaryOp::Sub => "-",
        BinaryOp::Mul => "*",
        BinaryOp::Div => "/",
        BinaryOp::Eq => "==",
        BinaryOp::NotEq => "!=",
        BinaryOp::Lt => "<",
        BinaryOp::LtEq => "<=",
        BinaryOp::Gt => ">",
        BinaryOp::GtEq => ">=",
        BinaryOp::And => "&&",
        BinaryOp::Or => "||",
    };
    format!("{l} {op_str} {r}")
}

/// Extract a human-readable name from an expression.
fn expr_name(expr: &Expr) -> String {
    match expr {
        Expr::Col(name) | Expr::Alias { name, .. } => name.clone(),
        Expr::LitF64(v) => v.to_string(),
        Expr::LitI64(v) => v.to_string(),
        Expr::LitStr(v) => format!("\"{v}\""),
        Expr::LitBool(v) => v.to_string(),
        _ => "expr".to_string(),
    }
}
