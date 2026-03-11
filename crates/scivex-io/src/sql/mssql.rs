//! Microsoft SQL Server connectivity via `tiberius`.
//!
//! Uses a blocking wrapper around the async `tiberius` client by embedding
//! a minimal Tokio runtime. The runtime is created once per connection and
//! reused for all operations.

use ::tiberius::{AuthMethod, Client, ColumnType, Config};
use ::tokio::net::TcpStream;
use ::tokio::runtime::Runtime;
use ::tokio_util::compat::{Compat, TokioAsyncWriteCompatExt};
use scivex_frame::DataFrame;

use crate::error::{IoError, Result};

use super::{
    ColumnData, IfExists, SqlDialect, SqlValue, build_dataframe, create_table_sql, extract_value,
};

/// Configuration for connecting to a SQL Server instance.
pub struct MssqlConfig {
    /// Server hostname or IP.
    pub host: String,
    /// Server port (default: 1433).
    pub port: u16,
    /// Username for SQL Server authentication.
    pub username: String,
    /// Password for SQL Server authentication.
    pub password: String,
    /// Optional database name.
    pub database: Option<String>,
    /// Whether to trust the server certificate (for development).
    pub trust_cert: bool,
}

impl MssqlConfig {
    /// Create a new configuration with the given host and credentials.
    pub fn new(
        host: impl Into<String>,
        username: impl Into<String>,
        password: impl Into<String>,
    ) -> Self {
        Self {
            host: host.into(),
            port: 1433,
            username: username.into(),
            password: password.into(),
            database: None,
            trust_cert: false,
        }
    }

    /// Set the port number.
    pub fn port(mut self, port: u16) -> Self {
        self.port = port;
        self
    }

    /// Set the database name.
    pub fn database(mut self, db: impl Into<String>) -> Self {
        self.database = Some(db.into());
        self
    }

    /// Trust the server certificate (useful for development/testing).
    pub fn trust_certificate(mut self) -> Self {
        self.trust_cert = true;
        self
    }
}

/// A connection to a Microsoft SQL Server database.
pub struct MssqlConnection {
    runtime: Runtime,
    client: Client<Compat<TcpStream>>,
}

impl MssqlConnection {
    /// Connect to a SQL Server instance.
    pub fn connect(config: &MssqlConfig) -> Result<Self> {
        let runtime = ::tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| IoError::SqlError(e.to_string()))?;

        let client = runtime.block_on(async {
            let mut tib_config = Config::new();
            tib_config.host(&config.host);
            tib_config.port(config.port);
            tib_config.authentication(AuthMethod::sql_server(&config.username, &config.password));

            if config.trust_cert {
                tib_config.trust_cert();
            }

            if let Some(ref db) = config.database {
                tib_config.database(db);
            }

            let tcp = TcpStream::connect(format!("{}:{}", config.host, config.port))
                .await
                .map_err(|e| IoError::SqlError(format!("TCP connect failed: {e}")))?;
            tcp.set_nodelay(true)
                .map_err(|e| IoError::SqlError(e.to_string()))?;

            let client = Client::connect(tib_config, tcp.compat_write())
                .await
                .map_err(|e| IoError::SqlError(e.to_string()))?;

            Ok::<_, IoError>(client)
        })?;

        Ok(Self { runtime, client })
    }

    /// Execute a SQL statement that returns no rows.
    pub fn execute(&mut self, sql: &str) -> Result<u64> {
        self.runtime.block_on(async {
            let result = self
                .client
                .execute(sql, &[])
                .await
                .map_err(|e| IoError::SqlError(e.to_string()))?;
            Ok(result.total())
        })
    }

    /// Execute a SQL query and return the results as a [`DataFrame`].
    pub fn read_sql(&mut self, query: &str) -> Result<DataFrame> {
        self.runtime.block_on(async {
            let stream = self
                .client
                .query(query, &[])
                .await
                .map_err(|e| IoError::SqlError(e.to_string()))?;

            let rows = stream
                .into_results()
                .await
                .map_err(|e| IoError::SqlError(e.to_string()))?;

            // Take the first result set.
            let Some(first_set) = rows.into_iter().next() else {
                return Ok(DataFrame::empty());
            };

            if first_set.is_empty() {
                return Ok(DataFrame::empty());
            }

            // Extract column info from first row.
            let col_names: Vec<String> = first_set[0]
                .columns()
                .iter()
                .map(|c| c.name().to_string())
                .collect();

            // Create collectors based on column types.
            let mut collectors: Vec<ColumnData> = first_set[0]
                .columns()
                .iter()
                .map(|c| match c.column_type() {
                    ColumnType::Int1
                    | ColumnType::Int2
                    | ColumnType::Int4
                    | ColumnType::Int8
                    | ColumnType::Intn
                    | ColumnType::Bit
                    | ColumnType::Bitn => ColumnData::new_i64(),
                    ColumnType::Float4
                    | ColumnType::Float8
                    | ColumnType::Floatn
                    | ColumnType::Money
                    | ColumnType::Money4
                    | ColumnType::Numericn
                    | ColumnType::Decimaln => ColumnData::new_f64(),
                    _ => ColumnData::new_str(),
                })
                .collect();

            // Process rows.
            for row in &first_set {
                for (col_idx, collector) in collectors.iter_mut().enumerate() {
                    match collector {
                        ColumnData::I64 { .. } => {
                            if let Some(v) = row.try_get::<i64, _>(col_idx).ok().flatten() {
                                collector.push_i64(v);
                            } else if let Some(v) = row.try_get::<i32, _>(col_idx).ok().flatten() {
                                collector.push_i64(i64::from(v));
                            } else if let Some(v) = row.try_get::<i16, _>(col_idx).ok().flatten() {
                                collector.push_i64(i64::from(v));
                            } else if let Some(v) = row.try_get::<bool, _>(col_idx).ok().flatten() {
                                collector.push_i64(i64::from(v));
                            } else {
                                collector.push_null();
                            }
                        }
                        ColumnData::F64 { .. } => {
                            if let Some(v) = row.try_get::<f64, _>(col_idx).ok().flatten() {
                                collector.push_f64(v);
                            } else if let Some(v) = row.try_get::<f32, _>(col_idx).ok().flatten() {
                                collector.push_f64(f64::from(v));
                            } else {
                                collector.push_null();
                            }
                        }
                        ColumnData::Str { .. } => {
                            if let Some(v) = row.try_get::<&str, _>(col_idx).ok().flatten() {
                                collector.push_str(v.to_string());
                            } else {
                                collector.push_null();
                            }
                        }
                    }
                }
            }

            build_dataframe(&col_names, collectors)
        })
    }

    /// Write a [`DataFrame`] to a SQL Server table.
    ///
    /// - `table`: the target table name.
    /// - `if_exists`: behavior when the table already exists.
    pub fn write_sql(&mut self, df: &DataFrame, table: &str, if_exists: IfExists) -> Result<()> {
        self.runtime.block_on(async {
            match if_exists {
                IfExists::Replace => {
                    let drop_sql =
                        format!("IF OBJECT_ID('{table}', 'U') IS NOT NULL DROP TABLE [{table}]");
                    let _ = self.client.execute(drop_sql.as_str(), &[]).await;
                    self.client
                        .execute(create_table_sql(df, table, SqlDialect::Mssql).as_str(), &[])
                        .await
                        .map_err(|e| IoError::SqlError(e.to_string()))?;
                }
                IfExists::Fail => {
                    self.client
                        .execute(create_table_sql(df, table, SqlDialect::Mssql).as_str(), &[])
                        .await
                        .map_err(|e| IoError::SqlError(e.to_string()))?;
                }
                IfExists::Append => {}
            }

            let nrows = df.nrows();
            if nrows == 0 {
                return Ok(());
            }

            let col_names: Vec<&str> = df.column_names();
            let columns = df.columns();

            // Build INSERT statements with literal values since tiberius
            // parameterized queries with dynamic types are complex.
            for row in 0..nrows {
                let values: Vec<String> = columns
                    .iter()
                    .map(|col| match extract_value(col.as_ref(), row) {
                        SqlValue::Null => "NULL".to_string(),
                        SqlValue::I64(v) => v.to_string(),
                        SqlValue::F64(v) => v.to_string(),
                        SqlValue::Str(v) => {
                            // Escape single quotes by doubling them.
                            format!("N'{}'", v.replace('\'', "''"))
                        }
                        SqlValue::Bool(v) => if v { "1" } else { "0" }.to_string(),
                    })
                    .collect();

                let col_list: String = col_names
                    .iter()
                    .map(|n| format!("[{n}]"))
                    .collect::<Vec<_>>()
                    .join(", ");
                let val_list = values.join(", ");
                let sql = format!("INSERT INTO [{table}] ({col_list}) VALUES ({val_list})");

                self.client
                    .execute(sql.as_str(), &[])
                    .await
                    .map_err(|e| IoError::SqlError(e.to_string()))?;
            }

            Ok(())
        })
    }
}
