# scivex-io

Data I/O for Scivex. Read and write DataFrames in CSV, JSON, Parquet, Arrow,
Excel, SQL databases, and more.

## Highlights

- **CSV** — High-performance reader/writer with type inference
- **JSON** — Row-oriented and column-oriented JSON support
- **Parquet** — Apache Parquet columnar format (Snappy, Zstd, LZ4)
- **Arrow** — Apache Arrow IPC format for zero-copy interop
- **Excel** — Read `.xlsx`/`.xls` via calamine, write `.xlsx` via rust_xlsxwriter
- **SQL** — SQLite, PostgreSQL, MySQL, MSSQL, DuckDB backends
- **NPY/NPZ** — NumPy binary array format
- **HDF5** — Hierarchical Data Format 5 datasets
- **ORC** — Apache ORC columnar format
- **Avro** — Apache Avro serialization format
- **Memory-mapped I/O** — Large file support via mmap
- **Cloud storage** — S3/GCS/Azure support (planned)

## Usage

```rust
use scivex_io::prelude::*;

// CSV
let df = read_csv("data.csv", &CsvOptions::default()).unwrap();
write_csv(&df, "output.csv", &CsvOptions::default()).unwrap();

// Parquet
let df = read_parquet("data.parquet").unwrap();

// SQLite
let df = read_sql("SELECT * FROM users", &SqlBackend::Sqlite("db.sqlite")).unwrap();
```

## License

MIT
