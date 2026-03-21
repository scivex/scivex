//! SQL query engine for DataFrames.
//!
//! Provides a minimal SQL subset that operates directly on [`DataFrame`]s:
//!
//! - `SELECT` with column selection, expressions, and aliases
//! - `FROM` with table references
//! - `JOIN` (INNER, LEFT, RIGHT)
//! - `WHERE` filtering
//! - `GROUP BY` with aggregate functions (SUM, AVG, MIN, MAX, COUNT)
//! - `HAVING`
//! - `ORDER BY`
//! - `LIMIT`
//!
//! # Example
//!
//! ```
//! use scivex_frame::sql::SqlContext;
//! use scivex_frame::{DataFrame, Series};
//!
//! let df = DataFrame::builder()
//!     .add_column("x", vec![1_i32, 2, 3])
//!     .build()
//!     .unwrap();
//!
//! let mut ctx = SqlContext::new();
//! ctx.register("t", df);
//! let result = ctx.execute("SELECT * FROM t WHERE x > 1").unwrap();
//! assert_eq!(result.nrows(), 2);
//! ```

pub mod ast;
pub mod executor;
pub mod parser;
pub mod tokenizer;

use std::collections::HashMap;

use crate::dataframe::DataFrame;
use crate::error::Result;

/// A context holding named tables that SQL queries can reference.
pub struct SqlContext {
    tables: HashMap<String, DataFrame>,
}

impl SqlContext {
    /// Create an empty context.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::sql::SqlContext;
    /// let ctx = SqlContext::new();
    /// ```
    pub fn new() -> Self {
        Self {
            tables: HashMap::new(),
        }
    }

    /// Register a `DataFrame` under the given name.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::sql::SqlContext;
    /// # use scivex_frame::{DataFrame, Series};
    /// let mut ctx = SqlContext::new();
    /// let df = DataFrame::builder().add_column("x", vec![1_i32]).build().unwrap();
    /// ctx.register("t", df);
    /// ```
    pub fn register(&mut self, name: &str, df: DataFrame) {
        self.tables.insert(name.to_string(), df);
    }

    /// Execute a SQL query and return the result as a `DataFrame`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::sql::SqlContext;
    /// # use scivex_frame::{DataFrame, Series};
    /// let mut ctx = SqlContext::new();
    /// let df = DataFrame::builder().add_column("x", vec![1_i32, 2, 3]).build().unwrap();
    /// ctx.register("t", df);
    /// let result = ctx.execute("SELECT * FROM t WHERE x > 1").unwrap();
    /// assert_eq!(result.nrows(), 2);
    /// ```
    pub fn execute(&self, sql: &str) -> Result<DataFrame> {
        let tokens = tokenizer::tokenize(sql)?;
        let stmt = parser::parse(&tokens)?;
        executor::execute(&stmt, &self.tables)
    }
}

impl Default for SqlContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function: query a single DataFrame (registered as `"t"`).
///
/// # Examples
///
/// ```
/// # use scivex_frame::sql::sql;
/// # use scivex_frame::{DataFrame, Series};
/// let df = DataFrame::builder().add_column("x", vec![1_i32, 2, 3]).build().unwrap();
/// let result = sql(&df, "SELECT * FROM t WHERE x > 1").unwrap();
/// assert_eq!(result.nrows(), 2);
/// ```
pub fn sql(df: &DataFrame, query: &str) -> Result<DataFrame> {
    let mut ctx = SqlContext::new();
    ctx.register("t", df.clone());
    ctx.execute(query)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::series::Series;
    use crate::series::string::StringSeries;

    fn sample_df() -> DataFrame {
        DataFrame::new(vec![
            Box::new(Series::new("a", vec![1_i32, 2, 3, 4, 5])),
            Box::new(Series::new("b", vec![10.0_f64, 20.0, 30.0, 40.0, 50.0])),
            Box::new(StringSeries::from_strs(
                "name",
                &["foo", "bar", "foo", "baz", "bar"],
            )),
        ])
        .unwrap()
    }

    #[test]
    fn test_select_star() {
        let df = sample_df();
        let result = sql(&df, "SELECT * FROM t").unwrap();
        assert_eq!(result.nrows(), 5);
        assert_eq!(result.ncols(), 3);
        assert_eq!(result.column_names(), vec!["a", "b", "name"]);
    }

    #[test]
    fn test_select_columns() {
        let df = sample_df();
        let result = sql(&df, "SELECT a, b FROM t").unwrap();
        assert_eq!(result.ncols(), 2);
        assert_eq!(result.column_names(), vec!["a", "b"]);
        assert_eq!(result.nrows(), 5);
    }

    #[test]
    fn test_where_numeric_filter() {
        let df = sample_df();
        let result = sql(&df, "SELECT * FROM t WHERE a > 3").unwrap();
        assert_eq!(result.nrows(), 2);
        let col_a = result.column_typed::<i32>("a").unwrap();
        assert_eq!(col_a.as_slice(), &[4, 5]);
    }

    #[test]
    fn test_where_string_filter() {
        let df = sample_df();
        let result = sql(&df, "SELECT * FROM t WHERE name = 'foo'").unwrap();
        assert_eq!(result.nrows(), 2);
        let names = result
            .column("name")
            .unwrap()
            .as_any()
            .downcast_ref::<StringSeries>()
            .unwrap();
        assert_eq!(names.get(0), Some("foo"));
        assert_eq!(names.get(1), Some("foo"));
    }

    #[test]
    fn test_group_by_sum() {
        let df = sample_df();
        let result = sql(&df, "SELECT name, SUM(a) FROM t GROUP BY name").unwrap();
        // 3 unique names: foo, bar, baz (in first-occurrence order)
        assert_eq!(result.nrows(), 3);
        assert_eq!(result.ncols(), 2);

        // Verify the sum values
        let names = result
            .column("name")
            .unwrap()
            .as_any()
            .downcast_ref::<StringSeries>()
            .unwrap();
        let sums = result.column_typed::<i32>("SUM(a)").unwrap();

        // foo: 1+3=4, bar: 2+5=7, baz: 4
        assert_eq!(names.get(0), Some("foo"));
        assert_eq!(sums.get(0), Some(4));
        assert_eq!(names.get(1), Some("bar"));
        assert_eq!(sums.get(1), Some(7));
        assert_eq!(names.get(2), Some("baz"));
        assert_eq!(sums.get(2), Some(4));
    }

    #[test]
    fn test_group_by_count_star() {
        let df = sample_df();
        let result = sql(&df, "SELECT name, COUNT(*) FROM t GROUP BY name").unwrap();
        assert_eq!(result.nrows(), 3);

        let counts = result.column_typed::<i64>("count").unwrap();
        // foo: 2, bar: 2, baz: 1
        assert_eq!(counts.get(0), Some(2));
        assert_eq!(counts.get(1), Some(2));
        assert_eq!(counts.get(2), Some(1));
    }

    #[test]
    fn test_order_by_desc() {
        let df = sample_df();
        let result = sql(&df, "SELECT * FROM t ORDER BY a DESC").unwrap();
        assert_eq!(result.nrows(), 5);
        let col_a = result.column_typed::<i32>("a").unwrap();
        assert_eq!(col_a.as_slice(), &[5, 4, 3, 2, 1]);
    }

    #[test]
    fn test_limit() {
        let df = sample_df();
        let result = sql(&df, "SELECT * FROM t LIMIT 3").unwrap();
        assert_eq!(result.nrows(), 3);
        let col_a = result.column_typed::<i32>("a").unwrap();
        assert_eq!(col_a.as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn test_join() {
        let left = DataFrame::new(vec![
            Box::new(Series::new("id", vec![1_i32, 2, 3])),
            Box::new(StringSeries::from_strs("val", &["a", "b", "c"])),
        ])
        .unwrap();

        let right = DataFrame::new(vec![
            Box::new(Series::new("id", vec![2_i32, 3, 4])),
            Box::new(Series::new("score", vec![100.0_f64, 200.0, 300.0])),
        ])
        .unwrap();

        let mut ctx = SqlContext::new();
        ctx.register("t1", left);
        ctx.register("t2", right);

        let result = ctx
            .execute("SELECT * FROM t1 INNER JOIN t2 ON t1.id = t2.id")
            .unwrap();
        assert_eq!(result.nrows(), 2); // ids 2 and 3
        // Should have columns: id, val, score
        assert!(result.column("id").is_ok());
        assert!(result.column("val").is_ok());
        assert!(result.column("score").is_ok());
    }

    #[test]
    fn test_error_invalid_sql() {
        let df = sample_df();
        // Missing FROM
        assert!(sql(&df, "SELECT *").is_err());
    }

    #[test]
    fn test_error_unknown_column() {
        let df = sample_df();
        assert!(sql(&df, "SELECT nonexistent FROM t").is_err());
    }

    #[test]
    fn test_error_unknown_table() {
        let ctx = SqlContext::new();
        assert!(ctx.execute("SELECT * FROM missing_table").is_err());
    }

    #[test]
    fn test_where_and_limit() {
        let df = sample_df();
        let result = sql(&df, "SELECT * FROM t WHERE a > 1 ORDER BY a ASC LIMIT 2").unwrap();
        assert_eq!(result.nrows(), 2);
        let col_a = result.column_typed::<i32>("a").unwrap();
        assert_eq!(col_a.as_slice(), &[2, 3]);
    }

    #[test]
    fn test_select_with_alias() {
        let df = sample_df();
        let result = sql(&df, "SELECT a AS x FROM t").unwrap();
        assert_eq!(result.column_names(), vec!["x"]);
        assert_eq!(result.nrows(), 5);
    }

    #[test]
    fn test_left_join() {
        let left = DataFrame::new(vec![
            Box::new(Series::new("id", vec![1_i32, 2, 3])),
            Box::new(StringSeries::from_strs("val", &["a", "b", "c"])),
        ])
        .unwrap();

        let right = DataFrame::new(vec![
            Box::new(Series::new("id", vec![2_i32, 3, 4])),
            Box::new(Series::new("score", vec![100.0_f64, 200.0, 300.0])),
        ])
        .unwrap();

        let mut ctx = SqlContext::new();
        ctx.register("t1", left);
        ctx.register("t2", right);

        let result = ctx
            .execute("SELECT * FROM t1 LEFT JOIN t2 ON t1.id = t2.id")
            .unwrap();
        // All 3 left rows kept
        assert_eq!(result.nrows(), 3);
    }

    #[test]
    fn test_case_insensitive_keywords() {
        let df = sample_df();
        let result = sql(&df, "select * from t where a > 3").unwrap();
        assert_eq!(result.nrows(), 2);
    }
}
