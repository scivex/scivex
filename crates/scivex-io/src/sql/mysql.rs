//! MySQL database connectivity via the `mysql` crate.

use ::mysql as my;
use ::mysql::prelude::*;
use scivex_frame::DataFrame;

use crate::error::{IoError, Result};

use super::{
    ColumnData, IfExists, SqlDialect, SqlValue, build_dataframe, create_table_sql, extract_value,
    insert_sql,
};

/// A connection to a MySQL database.
pub struct MysqlConnection {
    conn: my::Conn,
}

impl MysqlConnection {
    /// Connect to a MySQL database via a URL.
    ///
    /// Example URL: `"mysql://user:password@localhost:3306/database"`.
    pub fn connect(url: &str) -> Result<Self> {
        let opts = my::Opts::from_url(url).map_err(|e| IoError::SqlError(e.to_string()))?;
        let conn = my::Conn::new(opts).map_err(|e| IoError::SqlError(e.to_string()))?;
        Ok(Self { conn })
    }

    /// Wrap an existing `mysql::Conn`.
    pub fn from_conn(conn: my::Conn) -> Self {
        Self { conn }
    }

    /// Get a mutable reference to the underlying `mysql::Conn`.
    pub fn inner(&mut self) -> &mut my::Conn {
        &mut self.conn
    }

    /// Execute a SQL statement that returns no rows.
    pub fn execute(&mut self, sql: &str) -> Result<()> {
        self.conn
            .query_drop(sql)
            .map_err(|e| IoError::SqlError(e.to_string()))
    }

    /// Execute a SQL query and return the results as a [`DataFrame`].
    pub fn read_sql(&mut self, query: &str) -> Result<DataFrame> {
        let result = self
            .conn
            .query_iter(query)
            .map_err(|e| IoError::SqlError(e.to_string()))?;

        // Get column metadata.
        let columns_meta: Vec<(String, my::consts::ColumnType)> = result
            .columns()
            .as_ref()
            .iter()
            .map(|c| (c.name_str().to_string(), c.column_type()))
            .collect();

        let col_names: Vec<String> = columns_meta.iter().map(|(n, _)| n.clone()).collect();

        // Create column collectors based on column types.
        let mut collectors: Vec<ColumnData> = columns_meta
            .iter()
            .map(|(_, ct)| match ct {
                my::consts::ColumnType::MYSQL_TYPE_TINY
                | my::consts::ColumnType::MYSQL_TYPE_SHORT
                | my::consts::ColumnType::MYSQL_TYPE_LONG
                | my::consts::ColumnType::MYSQL_TYPE_LONGLONG
                | my::consts::ColumnType::MYSQL_TYPE_INT24 => ColumnData::new_i64(),
                my::consts::ColumnType::MYSQL_TYPE_FLOAT
                | my::consts::ColumnType::MYSQL_TYPE_DOUBLE
                | my::consts::ColumnType::MYSQL_TYPE_DECIMAL
                | my::consts::ColumnType::MYSQL_TYPE_NEWDECIMAL => ColumnData::new_f64(),
                _ => ColumnData::new_str(),
            })
            .collect();

        // Process rows.
        #[allow(clippy::cast_possible_wrap)]
        for row_result in result {
            let row = row_result.map_err(|e| IoError::SqlError(e.to_string()))?;
            for (col_idx, collector) in collectors.iter_mut().enumerate() {
                let val: my::Value = row.get(col_idx).unwrap_or(my::Value::NULL);

                match val {
                    my::Value::NULL => collector.push_null(),
                    my::Value::Int(v) => match collector {
                        ColumnData::I64 { .. } => collector.push_i64(v),
                        ColumnData::F64 { .. } => collector.push_f64(v as f64),
                        ColumnData::Str { .. } => collector.push_str(v.to_string()),
                    },
                    my::Value::UInt(v) => match collector {
                        ColumnData::I64 { .. } => collector.push_i64(v as i64),
                        ColumnData::F64 { .. } => collector.push_f64(v as f64),
                        ColumnData::Str { .. } => collector.push_str(v.to_string()),
                    },
                    my::Value::Float(v) => match collector {
                        ColumnData::F64 { .. } => collector.push_f64(f64::from(v)),
                        ColumnData::I64 { .. } => collector.push_i64(v as i64),
                        ColumnData::Str { .. } => collector.push_str(v.to_string()),
                    },
                    my::Value::Double(v) => match collector {
                        ColumnData::F64 { .. } => collector.push_f64(v),
                        ColumnData::I64 { .. } => collector.push_i64(v as i64),
                        ColumnData::Str { .. } => collector.push_str(v.to_string()),
                    },
                    my::Value::Bytes(ref b) => {
                        let s = String::from_utf8_lossy(b).to_string();
                        match collector {
                            ColumnData::Str { .. } => collector.push_str(s),
                            ColumnData::I64 { .. } => {
                                if let Ok(v) = s.parse::<i64>() {
                                    collector.push_i64(v);
                                } else {
                                    collector.push_null();
                                }
                            }
                            ColumnData::F64 { .. } => {
                                if let Ok(v) = s.parse::<f64>() {
                                    collector.push_f64(v);
                                } else {
                                    collector.push_null();
                                }
                            }
                        }
                    }
                    my::Value::Date(..) | my::Value::Time(..) => {
                        let s: String = my::from_value_opt(val).unwrap_or_default();
                        collector.push_str(s);
                    }
                }
            }
        }

        build_dataframe(&col_names, collectors)
    }

    /// Write a [`DataFrame`] to a MySQL table.
    ///
    /// - `table`: the target table name.
    /// - `if_exists`: behavior when the table already exists.
    pub fn write_sql(&mut self, df: &DataFrame, table: &str, if_exists: IfExists) -> Result<()> {
        match if_exists {
            IfExists::Replace => {
                self.conn
                    .query_drop(format!("DROP TABLE IF EXISTS `{table}`"))
                    .map_err(|e| IoError::SqlError(e.to_string()))?;
                self.conn
                    .query_drop(create_table_sql(df, table, SqlDialect::Mysql))
                    .map_err(|e| IoError::SqlError(e.to_string()))?;
            }
            IfExists::Fail => {
                self.conn
                    .query_drop(create_table_sql(df, table, SqlDialect::Mysql))
                    .map_err(|e| IoError::SqlError(e.to_string()))?;
            }
            IfExists::Append => {}
        }

        let nrows = df.nrows();
        if nrows == 0 {
            return Ok(());
        }

        let col_names: Vec<&str> = df.column_names();
        let sql = insert_sql(table, &col_names, SqlDialect::Mysql);
        let columns = df.columns();

        for row in 0..nrows {
            let params: Vec<my::Value> = columns
                .iter()
                .map(|col| match extract_value(col.as_ref(), row) {
                    SqlValue::Null => my::Value::NULL,
                    SqlValue::I64(v) => my::Value::Int(v),
                    SqlValue::F64(v) => my::Value::Double(v),
                    SqlValue::Str(v) => my::Value::Bytes(v.into_bytes()),
                    SqlValue::Bool(v) => my::Value::Int(i64::from(v)),
                })
                .collect();

            self.conn
                .exec_drop(&sql, params)
                .map_err(|e| IoError::SqlError(e.to_string()))?;
        }

        Ok(())
    }
}
