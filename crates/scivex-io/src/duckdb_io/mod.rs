//! DuckDB embedded OLAP database connectivity for DataFrames.
//!
//! Provides reading query results into DataFrames, writing DataFrames to
//! DuckDB tables, and leveraging DuckDB's built-in Parquet/CSV readers.
//!
//! Gated behind the `duckdb` feature flag.

use ::duckdb::{self, types::Value};
use scivex_frame::DataFrame;

use crate::error::{IoError, Result};
use crate::sql::{ColumnData, IfExists, SqlValue, build_dataframe, extract_value};

// ---------------------------------------------------------------------------
// DType → DuckDB SQL type mapping
// ---------------------------------------------------------------------------

use scivex_frame::DType;

fn dtype_to_duckdb_sql(dtype: DType) -> &'static str {
    match dtype {
        DType::I64 => "BIGINT",
        DType::I32 => "INTEGER",
        DType::I16 => "SMALLINT",
        DType::I8 => "TINYINT",
        DType::U64 => "UBIGINT",
        DType::U32 => "UINTEGER",
        DType::U16 => "USMALLINT",
        DType::U8 => "UTINYINT",
        DType::F64 => "DOUBLE",
        DType::F32 => "FLOAT",
        DType::Bool => "BOOLEAN",
        DType::Str | DType::Categorical | DType::DateTime => "VARCHAR",
    }
}

fn create_table_sql(df: &DataFrame, table: &str) -> String {
    let cols: Vec<String> = df
        .columns()
        .iter()
        .map(|col| {
            let sql_type = dtype_to_duckdb_sql(col.dtype());
            format!("\"{}\" {sql_type}", col.name())
        })
        .collect();
    format!("CREATE TABLE \"{table}\" ({})", cols.join(", "))
}

fn insert_sql(table: &str, column_names: &[&str]) -> String {
    let cols: String = column_names
        .iter()
        .map(|n| format!("\"{n}\""))
        .collect::<Vec<_>>()
        .join(", ");
    let placeholders: String = vec!["?"; column_names.len()].join(", ");
    format!("INSERT INTO \"{table}\" ({cols}) VALUES ({placeholders})")
}

// ---------------------------------------------------------------------------
// DuckDbConnection
// ---------------------------------------------------------------------------

/// A connection to a DuckDB database (file-backed or in-memory).
pub struct DuckDbConnection {
    conn: duckdb::Connection,
}

impl DuckDbConnection {
    /// Open a DuckDB database file. Creates the file if it does not exist.
    pub fn open(path: &str) -> Result<Self> {
        let conn = duckdb::Connection::open(path).map_err(|e| IoError::SqlError(e.to_string()))?;
        Ok(Self { conn })
    }

    /// Open an in-memory DuckDB database.
    pub fn open_in_memory() -> Result<Self> {
        let conn =
            duckdb::Connection::open_in_memory().map_err(|e| IoError::SqlError(e.to_string()))?;
        Ok(Self { conn })
    }

    /// Get a reference to the underlying `duckdb::Connection`.
    pub fn inner(&self) -> &duckdb::Connection {
        &self.conn
    }

    /// Execute a SQL statement that returns no rows (DDL, DML).
    ///
    /// Returns the number of rows affected.
    pub fn execute(&self, sql: &str) -> Result<usize> {
        self.conn
            .execute(sql, [])
            .map_err(|e| IoError::SqlError(e.to_string()))
    }

    /// Execute a SQL query and return the results as a [`DataFrame`].
    pub fn read_sql(&self, query: &str) -> Result<DataFrame> {
        let mut stmt = self
            .conn
            .prepare(query)
            .map_err(|e| IoError::SqlError(e.to_string()))?;

        let mut rows = stmt
            .query([])
            .map_err(|e| IoError::SqlError(e.to_string()))?;

        // column_count/column_names must be called after query() in duckdb crate
        #[allow(clippy::redundant_closure_for_method_calls)]
        let col_count = rows.as_ref().map_or(0, |r| r.column_count());
        #[allow(clippy::redundant_closure_for_method_calls)]
        let col_names: Vec<String> = rows.as_ref().map(|r| r.column_names()).unwrap_or_default();

        // Collect all rows first, since we need to inspect types dynamically.
        let mut raw_rows: Vec<Vec<Value>> = Vec::new();

        while let Some(row) = rows.next().map_err(|e| IoError::SqlError(e.to_string()))? {
            let mut values = Vec::with_capacity(col_count);
            for i in 0..col_count {
                let val: Value = row.get(i).map_err(|e| IoError::SqlError(e.to_string()))?;
                values.push(val);
            }
            raw_rows.push(values);
        }

        if raw_rows.is_empty() {
            // Return empty DataFrame with column names but no rows.
            let columns: Vec<Box<dyn scivex_frame::AnySeries>> = col_names
                .iter()
                .map(|name| -> Box<dyn scivex_frame::AnySeries> {
                    Box::new(scivex_frame::StringSeries::new(name, Vec::<String>::new()))
                })
                .collect();
            return Ok(DataFrame::new(columns)?);
        }

        // Determine column types from first non-null value in each column.
        let mut collectors: Vec<ColumnData> = Vec::with_capacity(col_count);
        for col_idx in 0..col_count {
            let col_type = raw_rows.iter().find_map(|row| match &row[col_idx] {
                Value::TinyInt(_)
                | Value::SmallInt(_)
                | Value::Int(_)
                | Value::BigInt(_)
                | Value::UTinyInt(_)
                | Value::USmallInt(_)
                | Value::UInt(_)
                | Value::UBigInt(_)
                | Value::Boolean(_) => Some("int"),
                Value::Float(_) | Value::Double(_) | Value::HugeInt(_) => Some("real"),
                Value::Null => None,
                _ => Some("text"),
            });
            collectors.push(match col_type.unwrap_or("text") {
                "int" => ColumnData::new_i64(),
                "real" => ColumnData::new_f64(),
                _ => ColumnData::new_str(),
            });
        }

        // Fill column data.
        for row in &raw_rows {
            for (col_idx, val) in row.iter().enumerate() {
                push_value(&mut collectors[col_idx], val);
            }
        }

        build_dataframe(&col_names, collectors)
    }

    /// Write a [`DataFrame`] to a DuckDB table.
    ///
    /// - `table`: the target table name.
    /// - `if_exists`: behavior when the table already exists.
    pub fn write_table(&self, table: &str, df: &DataFrame, if_exists: IfExists) -> Result<()> {
        match if_exists {
            IfExists::Replace => {
                let _ = self
                    .conn
                    .execute(&format!("DROP TABLE IF EXISTS \"{table}\""), []);
                self.conn
                    .execute(&create_table_sql(df, table), [])
                    .map_err(|e| IoError::SqlError(e.to_string()))?;
            }
            IfExists::Fail => {
                self.conn
                    .execute(&create_table_sql(df, table), [])
                    .map_err(|e| IoError::SqlError(e.to_string()))?;
            }
            IfExists::Append => {
                // Table must already exist; we just insert.
            }
        }

        let nrows = df.nrows();
        if nrows == 0 {
            return Ok(());
        }

        let col_names: Vec<&str> = df.column_names();
        let sql = insert_sql(table, &col_names);

        let columns = df.columns();

        for row in 0..nrows {
            let params: Vec<Value> = columns
                .iter()
                .map(|col| sql_value_to_duckdb(extract_value(col.as_ref(), row)))
                .collect();

            self.conn
                .execute(&sql, duckdb::params_from_iter(params))
                .map_err(|e| IoError::SqlError(e.to_string()))?;
        }

        Ok(())
    }

    /// Use DuckDB's built-in Parquet reader to load a Parquet file into a
    /// [`DataFrame`].
    ///
    /// This leverages DuckDB's native Parquet support, so neither the
    /// `parquet` nor `arrow` feature flags are required.
    pub fn read_parquet(&self, path: &str) -> Result<DataFrame> {
        let query = format!("SELECT * FROM read_parquet('{path}')");
        self.read_sql(&query)
    }

    /// Use DuckDB's built-in CSV reader to load a CSV file into a
    /// [`DataFrame`].
    ///
    /// This leverages DuckDB's native CSV support.
    pub fn read_csv_duckdb(&self, path: &str) -> Result<DataFrame> {
        let query = format!("SELECT * FROM read_csv_auto('{path}')");
        self.read_sql(&query)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Push a DuckDB `Value` into the appropriate `ColumnData` collector.
#[allow(clippy::cast_possible_wrap)]
fn push_value(collector: &mut ColumnData, val: &Value) {
    match val {
        Value::Null => collector.push_null(),
        Value::Boolean(v) => match collector {
            ColumnData::I64 { .. } => collector.push_i64(i64::from(*v)),
            ColumnData::F64 { .. } => collector.push_f64(if *v { 1.0 } else { 0.0 }),
            ColumnData::Str { .. } => collector.push_str(v.to_string()),
        },
        Value::TinyInt(v) => match collector {
            ColumnData::I64 { .. } => collector.push_i64(i64::from(*v)),
            ColumnData::F64 { .. } => collector.push_f64(f64::from(*v)),
            ColumnData::Str { .. } => collector.push_str(v.to_string()),
        },
        Value::SmallInt(v) => match collector {
            ColumnData::I64 { .. } => collector.push_i64(i64::from(*v)),
            ColumnData::F64 { .. } => collector.push_f64(f64::from(*v)),
            ColumnData::Str { .. } => collector.push_str(v.to_string()),
        },
        Value::Int(v) => match collector {
            ColumnData::I64 { .. } => collector.push_i64(i64::from(*v)),
            ColumnData::F64 { .. } => collector.push_f64(f64::from(*v)),
            ColumnData::Str { .. } => collector.push_str(v.to_string()),
        },
        Value::BigInt(v) => match collector {
            ColumnData::I64 { .. } => collector.push_i64(*v),
            ColumnData::F64 { .. } => collector.push_f64(*v as f64),
            ColumnData::Str { .. } => collector.push_str(v.to_string()),
        },
        Value::UTinyInt(v) => match collector {
            ColumnData::I64 { .. } => collector.push_i64(i64::from(*v)),
            ColumnData::F64 { .. } => collector.push_f64(f64::from(*v)),
            ColumnData::Str { .. } => collector.push_str(v.to_string()),
        },
        Value::USmallInt(v) => match collector {
            ColumnData::I64 { .. } => collector.push_i64(i64::from(*v)),
            ColumnData::F64 { .. } => collector.push_f64(f64::from(*v)),
            ColumnData::Str { .. } => collector.push_str(v.to_string()),
        },
        Value::UInt(v) => match collector {
            ColumnData::I64 { .. } => collector.push_i64(i64::from(*v)),
            ColumnData::F64 { .. } => collector.push_f64(f64::from(*v)),
            ColumnData::Str { .. } => collector.push_str(v.to_string()),
        },
        Value::UBigInt(v) => match collector {
            ColumnData::I64 { .. } => collector.push_i64(*v as i64),
            ColumnData::F64 { .. } => collector.push_f64(*v as f64),
            ColumnData::Str { .. } => collector.push_str(v.to_string()),
        },
        Value::HugeInt(v) => match collector {
            ColumnData::I64 { .. } => collector.push_i64(*v as i64),
            ColumnData::F64 { .. } => collector.push_f64(*v as f64),
            ColumnData::Str { .. } => collector.push_str(v.to_string()),
        },
        Value::Float(v) => match collector {
            ColumnData::F64 { .. } => collector.push_f64(f64::from(*v)),
            ColumnData::I64 { .. } => collector.push_i64(*v as i64),
            ColumnData::Str { .. } => collector.push_str(v.to_string()),
        },
        Value::Double(v) => match collector {
            ColumnData::F64 { .. } => collector.push_f64(*v),
            ColumnData::I64 { .. } => collector.push_i64(*v as i64),
            ColumnData::Str { .. } => collector.push_str(v.to_string()),
        },
        Value::Text(v) => collector.push_str(v.clone()),
        // For any other DuckDB types (Blob, Date, Time, Timestamp, etc.),
        // convert to string representation.
        other => collector.push_str(format!("{other:?}")),
    }
}

/// Convert our internal [`SqlValue`] to a DuckDB [`Value`].
fn sql_value_to_duckdb(v: SqlValue) -> Value {
    match v {
        SqlValue::Null => Value::Null,
        SqlValue::I64(i) => Value::BigInt(i),
        SqlValue::F64(f) => Value::Double(f),
        SqlValue::Str(s) => Value::Text(s),
        SqlValue::Bool(b) => Value::Boolean(b),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scivex_frame::{AnySeries, DataFrame, Series, StringSeries};

    #[test]
    fn test_duckdb_create_and_read() {
        let conn = DuckDbConnection::open_in_memory().unwrap();

        conn.execute("CREATE TABLE test (id INTEGER, value BIGINT)")
            .unwrap();
        conn.execute("INSERT INTO test VALUES (1, 10)").unwrap();
        conn.execute("INSERT INTO test VALUES (2, 20)").unwrap();
        conn.execute("INSERT INTO test VALUES (3, 30)").unwrap();

        let df = conn.read_sql("SELECT * FROM test").unwrap();
        assert_eq!(df.nrows(), 3);
        assert_eq!(df.ncols(), 2);

        let id_col = df.column_typed::<i64>("id").unwrap();
        assert_eq!(id_col.as_slice(), &[1, 2, 3]);

        let val_col = df.column_typed::<i64>("value").unwrap();
        assert_eq!(val_col.as_slice(), &[10, 20, 30]);
    }

    #[test]
    fn test_duckdb_write_and_read_back() {
        let conn = DuckDbConnection::open_in_memory().unwrap();

        let ids: Box<dyn AnySeries> = Box::new(Series::new("id", vec![1_i64, 2, 3]));
        let names: Box<dyn AnySeries> = Box::new(StringSeries::from_strs(
            "name",
            &["alice", "bob", "charlie"],
        ));
        let scores: Box<dyn AnySeries> = Box::new(Series::new("score", vec![1.1_f64, 2.2, 3.3]));

        let df = DataFrame::new(vec![ids, names, scores]).unwrap();
        conn.write_table("results", &df, IfExists::Fail).unwrap();

        let read_back = conn.read_sql("SELECT * FROM results").unwrap();
        assert_eq!(read_back.nrows(), 3);
        assert_eq!(read_back.ncols(), 3);

        let id_col = read_back.column_typed::<i64>("id").unwrap();
        assert_eq!(id_col.as_slice(), &[1, 2, 3]);

        let score_col = read_back.column_typed::<f64>("score").unwrap();
        assert!((score_col.as_slice()[0] - 1.1).abs() < 1e-10);
        assert!((score_col.as_slice()[1] - 2.2).abs() < 1e-10);
    }

    #[test]
    fn test_duckdb_if_exists_replace() {
        let conn = DuckDbConnection::open_in_memory().unwrap();

        let df1 = DataFrame::new(vec![Box::new(Series::new("x", vec![1_i64, 2])) as _]).unwrap();
        conn.write_table("tbl", &df1, IfExists::Fail).unwrap();

        let df2 =
            DataFrame::new(vec![Box::new(Series::new("x", vec![10_i64, 20, 30])) as _]).unwrap();
        conn.write_table("tbl", &df2, IfExists::Replace).unwrap();

        let result = conn.read_sql("SELECT * FROM tbl").unwrap();
        assert_eq!(result.nrows(), 3);

        let col = result.column_typed::<i64>("x").unwrap();
        assert_eq!(col.as_slice(), &[10, 20, 30]);
    }

    #[test]
    fn test_duckdb_if_exists_append() {
        let conn = DuckDbConnection::open_in_memory().unwrap();

        let df1 = DataFrame::new(vec![Box::new(Series::new("x", vec![1_i64, 2])) as _]).unwrap();
        conn.write_table("tbl", &df1, IfExists::Fail).unwrap();

        let df2 = DataFrame::new(vec![Box::new(Series::new("x", vec![3_i64, 4])) as _]).unwrap();
        conn.write_table("tbl", &df2, IfExists::Append).unwrap();

        let result = conn.read_sql("SELECT * FROM tbl").unwrap();
        assert_eq!(result.nrows(), 4);

        let col = result.column_typed::<i64>("x").unwrap();
        assert_eq!(col.as_slice(), &[1, 2, 3, 4]);
    }

    #[test]
    fn test_duckdb_sql_where_clause() {
        let conn = DuckDbConnection::open_in_memory().unwrap();

        conn.execute("CREATE TABLE data (id INTEGER, value DOUBLE)")
            .unwrap();
        conn.execute("INSERT INTO data VALUES (1, 10.0)").unwrap();
        conn.execute("INSERT INTO data VALUES (2, 20.0)").unwrap();
        conn.execute("INSERT INTO data VALUES (3, 30.0)").unwrap();
        conn.execute("INSERT INTO data VALUES (4, 40.0)").unwrap();

        let df = conn
            .read_sql("SELECT * FROM data WHERE value > 15.0")
            .unwrap();
        assert_eq!(df.nrows(), 3);

        let id_col = df.column_typed::<i64>("id").unwrap();
        assert_eq!(id_col.as_slice(), &[2, 3, 4]);
    }

    #[test]
    fn test_duckdb_multiple_column_types() {
        let conn = DuckDbConnection::open_in_memory().unwrap();

        conn.execute("CREATE TABLE multi (id INTEGER, score DOUBLE, name VARCHAR, active BOOLEAN)")
            .unwrap();
        conn.execute("INSERT INTO multi VALUES (1, 95.5, 'alice', true)")
            .unwrap();
        conn.execute("INSERT INTO multi VALUES (2, 87.3, 'bob', false)")
            .unwrap();
        conn.execute("INSERT INTO multi VALUES (3, 91.0, 'carol', true)")
            .unwrap();

        let df = conn.read_sql("SELECT * FROM multi").unwrap();
        assert_eq!(df.nrows(), 3);
        assert_eq!(df.ncols(), 4);

        // Integer column
        let id_col = df.column_typed::<i64>("id").unwrap();
        assert_eq!(id_col.as_slice(), &[1, 2, 3]);

        // Float column
        let score_col = df.column_typed::<f64>("score").unwrap();
        assert!((score_col.as_slice()[0] - 95.5).abs() < 1e-10);

        // Boolean mapped to i64 (1 / 0)
        let active_col = df.column_typed::<i64>("active").unwrap();
        assert_eq!(active_col.as_slice(), &[1, 0, 1]);
    }

    #[test]
    fn test_duckdb_empty_result() {
        let conn = DuckDbConnection::open_in_memory().unwrap();
        conn.execute("CREATE TABLE empty (id INTEGER)").unwrap();

        let df = conn.read_sql("SELECT * FROM empty").unwrap();
        assert_eq!(df.nrows(), 0);
        assert_eq!(df.ncols(), 1);
    }

    #[test]
    fn test_duckdb_write_fail_exists() {
        let conn = DuckDbConnection::open_in_memory().unwrap();

        let df = DataFrame::new(vec![Box::new(Series::new("x", vec![1_i64])) as _]).unwrap();
        conn.write_table("tbl", &df, IfExists::Fail).unwrap();

        let result = conn.write_table("tbl", &df, IfExists::Fail);
        assert!(result.is_err());
    }
}
