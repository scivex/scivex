//! SQL database connectivity for DataFrames.
//!
//! Supports reading query results into DataFrames and writing DataFrames to
//! database tables. Each backend is behind its own feature flag:
//!
//! | Feature    | Backend    | Crate        |
//! |------------|------------|--------------|
//! | `sqlite`   | SQLite     | `rusqlite`   |
//! | `postgres` | PostgreSQL | `postgres`   |
//! | `mysql`    | MySQL      | `mysql`      |
//! | `mssql`    | SQL Server | `tiberius`   |

use scivex_frame::{AnySeries, DType, DataFrame, Series, StringSeries};

use crate::error::Result;

/// Behavior when writing to a table that already exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IfExists {
    /// Raise an error if the table exists.
    Fail,
    /// Drop the existing table and create a new one.
    Replace,
    /// Append rows to the existing table.
    Append,
}

#[cfg(feature = "sqlite")]
pub mod sqlite;

#[cfg(feature = "postgres")]
pub mod postgres;

#[cfg(feature = "mysql")]
pub mod mysql;

#[cfg(feature = "mssql")]
pub mod mssql;

// ---------------------------------------------------------------------------
// Shared column collection utilities
// ---------------------------------------------------------------------------

/// Intermediate storage for collecting query result values per column.
pub(crate) enum ColumnData {
    /// Column of 64-bit integers with null mask.
    I64 { data: Vec<i64>, nulls: Vec<bool> },
    /// Column of 64-bit floats with null mask.
    F64 { data: Vec<f64>, nulls: Vec<bool> },
    /// Column of strings with null mask.
    Str { data: Vec<String>, nulls: Vec<bool> },
}

impl ColumnData {
    pub(crate) fn new_i64() -> Self {
        Self::I64 {
            data: Vec::new(),
            nulls: Vec::new(),
        }
    }

    pub(crate) fn new_f64() -> Self {
        Self::F64 {
            data: Vec::new(),
            nulls: Vec::new(),
        }
    }

    pub(crate) fn new_str() -> Self {
        Self::Str {
            data: Vec::new(),
            nulls: Vec::new(),
        }
    }

    pub(crate) fn push_i64(&mut self, v: i64) {
        if let Self::I64 { data, nulls } = self {
            data.push(v);
            nulls.push(false);
        }
    }

    pub(crate) fn push_f64(&mut self, v: f64) {
        if let Self::F64 { data, nulls } = self {
            data.push(v);
            nulls.push(false);
        }
    }

    pub(crate) fn push_str(&mut self, v: String) {
        if let Self::Str { data, nulls } = self {
            data.push(v);
            nulls.push(false);
        }
    }

    pub(crate) fn push_null(&mut self) {
        match self {
            Self::I64 { data, nulls } => {
                data.push(0);
                nulls.push(true);
            }
            Self::F64 { data, nulls } => {
                data.push(0.0);
                nulls.push(true);
            }
            Self::Str { data, nulls } => {
                data.push(String::new());
                nulls.push(true);
            }
        }
    }

    /// Convert into a boxed [`AnySeries`].
    pub(crate) fn into_series(self, name: &str) -> Result<Box<dyn AnySeries>> {
        match self {
            Self::I64 { data, nulls } => {
                if nulls.iter().any(|&n| n) {
                    Ok(Box::new(Series::with_nulls(name, data, nulls)?))
                } else {
                    Ok(Box::new(Series::new(name, data)))
                }
            }
            Self::F64 { data, nulls } => {
                if nulls.iter().any(|&n| n) {
                    Ok(Box::new(Series::with_nulls(name, data, nulls)?))
                } else {
                    Ok(Box::new(Series::new(name, data)))
                }
            }
            Self::Str { data, nulls } => {
                if nulls.iter().any(|&n| n) {
                    Ok(Box::new(StringSeries::with_nulls(name, data, nulls)?))
                } else {
                    Ok(Box::new(StringSeries::new(name, data)))
                }
            }
        }
    }
}

/// Build a [`DataFrame`] from column names and collected data.
pub(crate) fn build_dataframe(names: &[String], columns: Vec<ColumnData>) -> Result<DataFrame> {
    let series: Vec<Box<dyn AnySeries>> = names
        .iter()
        .zip(columns)
        .map(|(name, col)| col.into_series(name))
        .collect::<Result<Vec<_>>>()?;
    Ok(DataFrame::new(series)?)
}

/// SQL dialect for type mapping.
#[derive(Debug, Clone, Copy)]
pub(crate) enum SqlDialect {
    #[cfg(feature = "sqlite")]
    Sqlite,
    #[cfg(feature = "postgres")]
    Postgres,
    #[cfg(feature = "mysql")]
    Mysql,
    #[cfg(feature = "mssql")]
    Mssql,
}

/// Map a [`DType`] to a SQL column type string for the given dialect.
#[allow(unreachable_patterns)]
pub(crate) fn dtype_to_sql(dtype: DType, dialect: SqlDialect) -> &'static str {
    match dialect {
        #[cfg(feature = "sqlite")]
        SqlDialect::Sqlite => match dtype {
            DType::I64
            | DType::I32
            | DType::I16
            | DType::I8
            | DType::U64
            | DType::U32
            | DType::U16
            | DType::U8
            | DType::Bool => "INTEGER",
            DType::F64 | DType::F32 => "REAL",
            DType::Str | DType::Categorical | DType::DateTime => "TEXT",
        },
        #[cfg(feature = "postgres")]
        SqlDialect::Postgres => match dtype {
            DType::I64 | DType::U64 | DType::U32 => "BIGINT",
            DType::I32 | DType::U16 | DType::U8 => "INTEGER",
            DType::I16 | DType::I8 => "SMALLINT",
            DType::F64 => "DOUBLE PRECISION",
            DType::F32 => "REAL",
            DType::Bool => "BOOLEAN",
            DType::Str | DType::Categorical | DType::DateTime => "TEXT",
        },
        #[cfg(feature = "mysql")]
        SqlDialect::Mysql => match dtype {
            DType::I64 => "BIGINT",
            DType::I32 => "INT",
            DType::I16 => "SMALLINT",
            DType::I8 => "TINYINT",
            DType::U64 => "BIGINT UNSIGNED",
            DType::U32 => "INT UNSIGNED",
            DType::U16 => "SMALLINT UNSIGNED",
            DType::U8 => "TINYINT UNSIGNED",
            DType::F64 => "DOUBLE",
            DType::F32 => "FLOAT",
            DType::Bool => "TINYINT(1)",
            DType::Str | DType::Categorical | DType::DateTime => "TEXT",
        },
        #[cfg(feature = "mssql")]
        SqlDialect::Mssql => match dtype {
            DType::I64 | DType::U64 | DType::U32 => "BIGINT",
            DType::I32 | DType::U16 | DType::U8 => "INT",
            DType::I16 => "SMALLINT",
            DType::I8 => "TINYINT",
            DType::F64 => "FLOAT",
            DType::F32 => "REAL",
            DType::Bool => "BIT",
            DType::Str | DType::Categorical | DType::DateTime => "NVARCHAR(MAX)",
        },
    }
}

/// Generate a `CREATE TABLE` statement from a DataFrame's schema.
pub(crate) fn create_table_sql(df: &DataFrame, table: &str, dialect: SqlDialect) -> String {
    let cols: Vec<String> = df
        .columns()
        .iter()
        .map(|col| {
            let sql_type = dtype_to_sql(col.dtype(), dialect);
            format!("\"{}\" {sql_type}", col.name())
        })
        .collect();
    format!("CREATE TABLE \"{table}\" ({})", cols.join(", "))
}

/// Generate a parameterized `INSERT` statement.
#[allow(dead_code, unreachable_patterns)]
pub(crate) fn insert_sql(table: &str, column_names: &[&str], dialect: SqlDialect) -> String {
    let cols: String = column_names
        .iter()
        .map(|n| format!("\"{n}\""))
        .collect::<Vec<_>>()
        .join(", ");
    let placeholders: String = match dialect {
        #[cfg(feature = "postgres")]
        SqlDialect::Postgres => (1..=column_names.len())
            .map(|i| format!("${i}"))
            .collect::<Vec<_>>()
            .join(", "),
        #[cfg(feature = "mssql")]
        SqlDialect::Mssql => (1..=column_names.len())
            .map(|i| format!("@P{i}"))
            .collect::<Vec<_>>()
            .join(", "),
        _ => vec!["?"; column_names.len()].join(", "),
    };
    format!("INSERT INTO \"{table}\" ({cols}) VALUES ({placeholders})")
}

/// Dynamically-typed SQL value for parameterized queries.
pub(crate) enum SqlValue {
    /// SQL NULL.
    Null,
    /// 64-bit integer value.
    I64(i64),
    /// 64-bit float value.
    F64(f64),
    /// String value.
    Str(String),
    /// Boolean value.
    Bool(bool),
}

/// Extract the SQL value from a column at the given row index.
#[allow(clippy::cast_possible_wrap)]
pub(crate) fn extract_value(col: &dyn AnySeries, row: usize) -> SqlValue {
    if col.is_null(row) {
        return SqlValue::Null;
    }
    match col.dtype() {
        DType::I64 => {
            let s = col.as_any().downcast_ref::<Series<i64>>().unwrap();
            SqlValue::I64(s.as_slice()[row])
        }
        DType::I32 => {
            let s = col.as_any().downcast_ref::<Series<i32>>().unwrap();
            SqlValue::I64(i64::from(s.as_slice()[row]))
        }
        DType::I16 => {
            let s = col.as_any().downcast_ref::<Series<i16>>().unwrap();
            SqlValue::I64(i64::from(s.as_slice()[row]))
        }
        DType::I8 => {
            let s = col.as_any().downcast_ref::<Series<i8>>().unwrap();
            SqlValue::I64(i64::from(s.as_slice()[row]))
        }
        DType::U64 => {
            let s = col.as_any().downcast_ref::<Series<u64>>().unwrap();
            SqlValue::I64(s.as_slice()[row] as i64)
        }
        DType::U32 => {
            let s = col.as_any().downcast_ref::<Series<u32>>().unwrap();
            SqlValue::I64(i64::from(s.as_slice()[row]))
        }
        DType::U16 => {
            let s = col.as_any().downcast_ref::<Series<u16>>().unwrap();
            SqlValue::I64(i64::from(s.as_slice()[row]))
        }
        DType::U8 => {
            let s = col.as_any().downcast_ref::<Series<u8>>().unwrap();
            SqlValue::I64(i64::from(s.as_slice()[row]))
        }
        DType::F64 => {
            let s = col.as_any().downcast_ref::<Series<f64>>().unwrap();
            SqlValue::F64(s.as_slice()[row])
        }
        DType::F32 => {
            let s = col.as_any().downcast_ref::<Series<f32>>().unwrap();
            SqlValue::F64(f64::from(s.as_slice()[row]))
        }
        DType::Bool => {
            // Booleans are stored as Series<u8> (0 = false, 1 = true).
            let s = col.as_any().downcast_ref::<Series<u8>>().unwrap();
            SqlValue::Bool(s.as_slice()[row] != 0)
        }
        DType::Str | DType::Categorical | DType::DateTime => SqlValue::Str(col.display_value(row)),
    }
}
