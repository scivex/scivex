//! SQLite database connectivity via `rusqlite`.

use ::rusqlite::{self, params_from_iter, types::Value};
use scivex_frame::DataFrame;

use crate::error::{IoError, Result};

use super::{
    ColumnData, IfExists, SqlDialect, SqlValue, build_dataframe, create_table_sql, extract_value,
    insert_sql,
};

/// A connection to a SQLite database.
pub struct SqliteConnection {
    conn: rusqlite::Connection,
}

impl SqliteConnection {
    /// Open a SQLite database file. Creates the file if it does not exist.
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let conn =
            rusqlite::Connection::open(path).map_err(|e| IoError::SqlError(e.to_string()))?;
        Ok(Self { conn })
    }

    /// Open an in-memory SQLite database.
    pub fn open_in_memory() -> Result<Self> {
        let conn =
            rusqlite::Connection::open_in_memory().map_err(|e| IoError::SqlError(e.to_string()))?;
        Ok(Self { conn })
    }

    /// Get a reference to the underlying `rusqlite::Connection`.
    pub fn inner(&self) -> &rusqlite::Connection {
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

        let col_count = stmt.column_count();
        let col_names: Vec<String> = stmt.column_names().into_iter().map(String::from).collect();

        // We'll collect all rows first to determine types, since SQLite is
        // dynamically typed and the first row's types inform the column types.
        let mut raw_rows: Vec<Vec<Value>> = Vec::new();

        let mut rows = stmt
            .query([])
            .map_err(|e| IoError::SqlError(e.to_string()))?;

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
                Value::Integer(_) => Some("int"),
                Value::Real(_) => Some("real"),
                Value::Text(_) | Value::Blob(_) => Some("text"),
                Value::Null => None,
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
                match val {
                    Value::Null => collectors[col_idx].push_null(),
                    Value::Integer(v) => match &mut collectors[col_idx] {
                        ColumnData::I64 { .. } => collectors[col_idx].push_i64(*v),
                        ColumnData::F64 { .. } => collectors[col_idx].push_f64(*v as f64),
                        ColumnData::Str { .. } => collectors[col_idx].push_str(v.to_string()),
                    },
                    Value::Real(v) => match &mut collectors[col_idx] {
                        ColumnData::F64 { .. } => collectors[col_idx].push_f64(*v),
                        ColumnData::I64 { .. } => collectors[col_idx].push_i64(*v as i64),
                        ColumnData::Str { .. } => collectors[col_idx].push_str(v.to_string()),
                    },
                    Value::Text(v) => {
                        collectors[col_idx].push_str(v.clone());
                    }
                    Value::Blob(v) => {
                        collectors[col_idx].push_str(format!("{v:?}"));
                    }
                }
            }
        }

        build_dataframe(&col_names, collectors)
    }

    /// Write a [`DataFrame`] to a SQLite table.
    ///
    /// - `table`: the target table name.
    /// - `if_exists`: behavior when the table already exists.
    pub fn write_sql(&self, df: &DataFrame, table: &str, if_exists: IfExists) -> Result<()> {
        match if_exists {
            IfExists::Replace => {
                let _ = self
                    .conn
                    .execute(&format!("DROP TABLE IF EXISTS \"{table}\""), []);
                self.conn
                    .execute(&create_table_sql(df, table, SqlDialect::Sqlite), [])
                    .map_err(|e| IoError::SqlError(e.to_string()))?;
            }
            IfExists::Fail => {
                self.conn
                    .execute(&create_table_sql(df, table, SqlDialect::Sqlite), [])
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
        let sql = insert_sql(table, &col_names, SqlDialect::Sqlite);

        let columns = df.columns();
        let tx = self
            .conn
            .unchecked_transaction()
            .map_err(|e| IoError::SqlError(e.to_string()))?;

        for row in 0..nrows {
            let params: Vec<Value> = columns
                .iter()
                .map(|col| match extract_value(col.as_ref(), row) {
                    SqlValue::Null => Value::Null,
                    SqlValue::I64(v) => Value::Integer(v),
                    SqlValue::F64(v) => Value::Real(v),
                    SqlValue::Str(v) => Value::Text(v),
                    SqlValue::Bool(v) => Value::Integer(i64::from(v)),
                })
                .collect();

            self.conn
                .execute(&sql, params_from_iter(params))
                .map_err(|e| IoError::SqlError(e.to_string()))?;
        }

        tx.commit().map_err(|e| IoError::SqlError(e.to_string()))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scivex_frame::{DataFrame, Series, StringSeries};

    #[test]
    fn test_sqlite_roundtrip_integers() {
        let conn = SqliteConnection::open_in_memory().unwrap();

        conn.execute("CREATE TABLE test (id INTEGER, value INTEGER)")
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
    fn test_sqlite_roundtrip_mixed_types() {
        let conn = SqliteConnection::open_in_memory().unwrap();

        conn.execute("CREATE TABLE mixed (name TEXT, score REAL, rank INTEGER)")
            .unwrap();
        conn.execute("INSERT INTO mixed VALUES ('alice', 95.5, 1)")
            .unwrap();
        conn.execute("INSERT INTO mixed VALUES ('bob', 87.3, 2)")
            .unwrap();

        let df = conn.read_sql("SELECT * FROM mixed").unwrap();
        assert_eq!(df.nrows(), 2);
        assert_eq!(df.ncols(), 3);
    }

    #[test]
    fn test_sqlite_write_df() {
        let conn = SqliteConnection::open_in_memory().unwrap();

        let ids: Box<dyn scivex_frame::AnySeries> = Box::new(Series::new("id", vec![1_i64, 2, 3]));
        let names: Box<dyn scivex_frame::AnySeries> =
            Box::new(StringSeries::from_strs("name", &["a", "b", "c"]));
        let scores: Box<dyn scivex_frame::AnySeries> =
            Box::new(Series::new("score", vec![1.1_f64, 2.2, 3.3]));

        let df = DataFrame::new(vec![ids, names, scores]).unwrap();

        conn.write_sql(&df, "results", IfExists::Fail).unwrap();

        let read_back = conn.read_sql("SELECT * FROM results").unwrap();
        assert_eq!(read_back.nrows(), 3);
        assert_eq!(read_back.ncols(), 3);

        let id_col = read_back.column_typed::<i64>("id").unwrap();
        assert_eq!(id_col.as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn test_sqlite_write_replace() {
        let conn = SqliteConnection::open_in_memory().unwrap();

        let df1 = DataFrame::new(vec![Box::new(Series::new("x", vec![1_i64, 2])) as _]).unwrap();
        conn.write_sql(&df1, "tbl", IfExists::Fail).unwrap();

        let df2 =
            DataFrame::new(vec![Box::new(Series::new("x", vec![10_i64, 20, 30])) as _]).unwrap();
        conn.write_sql(&df2, "tbl", IfExists::Replace).unwrap();

        let result = conn.read_sql("SELECT * FROM tbl").unwrap();
        assert_eq!(result.nrows(), 3);
    }

    #[test]
    fn test_sqlite_write_append() {
        let conn = SqliteConnection::open_in_memory().unwrap();

        let df1 = DataFrame::new(vec![Box::new(Series::new("x", vec![1_i64, 2])) as _]).unwrap();
        conn.write_sql(&df1, "tbl", IfExists::Fail).unwrap();

        let df2 = DataFrame::new(vec![Box::new(Series::new("x", vec![3_i64, 4])) as _]).unwrap();
        conn.write_sql(&df2, "tbl", IfExists::Append).unwrap();

        let result = conn.read_sql("SELECT * FROM tbl").unwrap();
        assert_eq!(result.nrows(), 4);
    }

    #[test]
    fn test_sqlite_write_fail_exists() {
        let conn = SqliteConnection::open_in_memory().unwrap();

        let df = DataFrame::new(vec![Box::new(Series::new("x", vec![1_i64])) as _]).unwrap();
        conn.write_sql(&df, "tbl", IfExists::Fail).unwrap();

        let result = conn.write_sql(&df, "tbl", IfExists::Fail);
        assert!(result.is_err());
    }

    #[test]
    fn test_sqlite_nulls() {
        let conn = SqliteConnection::open_in_memory().unwrap();

        conn.execute("CREATE TABLE nulltest (a INTEGER, b TEXT)")
            .unwrap();
        conn.execute("INSERT INTO nulltest VALUES (1, 'hello')")
            .unwrap();
        conn.execute("INSERT INTO nulltest VALUES (NULL, NULL)")
            .unwrap();
        conn.execute("INSERT INTO nulltest VALUES (3, 'world')")
            .unwrap();

        let df = conn.read_sql("SELECT * FROM nulltest").unwrap();
        assert_eq!(df.nrows(), 3);

        let a = df.column("a").unwrap();
        assert!(!a.is_null(0));
        assert!(a.is_null(1));
        assert!(!a.is_null(2));
    }

    #[test]
    fn test_sqlite_empty_result() {
        let conn = SqliteConnection::open_in_memory().unwrap();
        conn.execute("CREATE TABLE empty (id INTEGER)").unwrap();

        let df = conn.read_sql("SELECT * FROM empty").unwrap();
        assert_eq!(df.nrows(), 0);
        assert_eq!(df.ncols(), 1);
    }
}
