//! PostgreSQL database connectivity via the `postgres` crate.

use ::postgres as pg;
use scivex_frame::DataFrame;

use crate::error::{IoError, Result};

use super::{
    ColumnData, IfExists, SqlDialect, SqlValue, build_dataframe, create_table_sql, extract_value,
    insert_sql,
};

/// A connection to a PostgreSQL database.
pub struct PostgresConnection {
    client: pg::Client,
}

impl PostgresConnection {
    /// Connect to a PostgreSQL database.
    ///
    /// `params` is a connection string, e.g.
    /// `"host=localhost user=postgres password=secret dbname=mydb"`.
    pub fn connect(params: &str) -> Result<Self> {
        let client =
            pg::Client::connect(params, pg::NoTls).map_err(|e| IoError::SqlError(e.to_string()))?;
        Ok(Self { client })
    }

    /// Connect with TLS support using a custom TLS connector.
    pub fn connect_tls<T>(params: &str, tls: T) -> Result<Self>
    where
        T: pg::tls::MakeTlsConnect<pg::Socket> + Send + 'static,
        T::Stream: Send,
        T::TlsConnect: Send,
        <T::TlsConnect as pg::tls::TlsConnect<pg::Socket>>::Future: Send,
    {
        let client =
            pg::Client::connect(params, tls).map_err(|e| IoError::SqlError(e.to_string()))?;
        Ok(Self { client })
    }

    /// Wrap an existing `postgres::Client`.
    pub fn from_client(client: pg::Client) -> Self {
        Self { client }
    }

    /// Get a mutable reference to the underlying `postgres::Client`.
    pub fn inner(&mut self) -> &mut pg::Client {
        &mut self.client
    }

    /// Execute a SQL statement that returns no rows.
    ///
    /// Returns the number of rows affected.
    pub fn execute(&mut self, sql: &str) -> Result<u64> {
        self.client
            .execute(sql, &[])
            .map_err(|e| IoError::SqlError(e.to_string()))
    }

    /// Execute a SQL query and return the results as a [`DataFrame`].
    pub fn read_sql(&mut self, query: &str) -> Result<DataFrame> {
        let rows = self
            .client
            .query(query, &[])
            .map_err(|e| IoError::SqlError(e.to_string()))?;

        if rows.is_empty() {
            return Ok(DataFrame::empty());
        }

        let columns = rows[0].columns();
        let col_count = columns.len();
        let col_names: Vec<String> = columns.iter().map(|c| c.name().to_string()).collect();

        // Determine column data types from Postgres type info.
        let mut collectors: Vec<ColumnData> = Vec::with_capacity(col_count);
        for col in columns {
            let collector = match *col.type_() {
                pg::types::Type::INT2 | pg::types::Type::INT4 | pg::types::Type::INT8 => {
                    ColumnData::new_i64()
                }
                pg::types::Type::FLOAT4 | pg::types::Type::FLOAT8 | pg::types::Type::NUMERIC => {
                    ColumnData::new_f64()
                }
                pg::types::Type::BOOL => ColumnData::new_i64(),
                _ => ColumnData::new_str(),
            };
            collectors.push(collector);
        }

        // Process rows.
        for row in &rows {
            for (col_idx, collector) in collectors.iter_mut().enumerate() {
                let col_type = row.columns()[col_idx].type_();
                match *col_type {
                    pg::types::Type::INT2 => match row.try_get::<_, i16>(col_idx) {
                        Ok(v) => collector.push_i64(i64::from(v)),
                        Err(_) => collector.push_null(),
                    },
                    pg::types::Type::INT4 => match row.try_get::<_, i32>(col_idx) {
                        Ok(v) => collector.push_i64(i64::from(v)),
                        Err(_) => collector.push_null(),
                    },
                    pg::types::Type::INT8 => match row.try_get::<_, i64>(col_idx) {
                        Ok(v) => collector.push_i64(v),
                        Err(_) => collector.push_null(),
                    },
                    pg::types::Type::FLOAT4 => match row.try_get::<_, f32>(col_idx) {
                        Ok(v) => collector.push_f64(f64::from(v)),
                        Err(_) => collector.push_null(),
                    },
                    pg::types::Type::FLOAT8 => match row.try_get::<_, f64>(col_idx) {
                        Ok(v) => collector.push_f64(v),
                        Err(_) => collector.push_null(),
                    },
                    pg::types::Type::BOOL => match row.try_get::<_, bool>(col_idx) {
                        Ok(v) => collector.push_i64(i64::from(v)),
                        Err(_) => collector.push_null(),
                    },
                    _ => match row.try_get::<_, String>(col_idx) {
                        Ok(v) => collector.push_str(v),
                        Err(_) => collector.push_null(),
                    },
                }
            }
        }

        build_dataframe(&col_names, collectors)
    }

    /// Write a [`DataFrame`] to a PostgreSQL table.
    ///
    /// - `table`: the target table name.
    /// - `if_exists`: behavior when the table already exists.
    pub fn write_sql(&mut self, df: &DataFrame, table: &str, if_exists: IfExists) -> Result<()> {
        match if_exists {
            IfExists::Replace => {
                let _ = self
                    .client
                    .execute(&format!("DROP TABLE IF EXISTS \"{table}\""), &[]);
                self.client
                    .execute(&create_table_sql(df, table, SqlDialect::Postgres), &[])
                    .map_err(|e| IoError::SqlError(e.to_string()))?;
            }
            IfExists::Fail => {
                self.client
                    .execute(&create_table_sql(df, table, SqlDialect::Postgres), &[])
                    .map_err(|e| IoError::SqlError(e.to_string()))?;
            }
            IfExists::Append => {}
        }

        let nrows = df.nrows();
        if nrows == 0 {
            return Ok(());
        }

        let col_names: Vec<&str> = df.column_names();
        let sql = insert_sql(table, &col_names, SqlDialect::Postgres);

        let columns = df.columns();

        for row in 0..nrows {
            let params: Vec<Box<dyn pg::types::ToSql + Sync>> = columns
                .iter()
                .map(|col| -> Box<dyn pg::types::ToSql + Sync> {
                    match extract_value(col.as_ref(), row) {
                        SqlValue::Null => Box::new(Option::<i64>::None),
                        SqlValue::I64(v) => Box::new(v),
                        SqlValue::F64(v) => Box::new(v),
                        SqlValue::Str(v) => Box::new(v),
                        SqlValue::Bool(v) => Box::new(v),
                    }
                })
                .collect();

            let param_refs: Vec<&(dyn pg::types::ToSql + Sync)> =
                params.iter().map(AsRef::as_ref).collect();

            self.client
                .execute(&sql, &param_refs)
                .map_err(|e| IoError::SqlError(e.to_string()))?;
        }

        Ok(())
    }
}
