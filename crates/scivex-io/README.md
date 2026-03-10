# scivex-io

Data I/O for the Scivex ecosystem. Read and write DataFrames from CSV and JSON
with automatic type inference, configurable parsing, and null handling.

## Highlights

- **CSV** — Custom parser with delimiter, quoting, header, null, comment support
- **JSON** — Records and Split orientations
- **Auto type inference** — Detects Int32, Float64, String, Categorical from data
- **Builder pattern** — Fluent configuration for readers and writers
- **Pluggable** — Works with any `impl Read` / `impl Write`

## Usage

```rust
use scivex_io::prelude::*;

// Read CSV from file
let df = read_csv_path("data.csv").unwrap();

// Read with options
let df = CsvReaderBuilder::new()
    .delimiter(b';')
    .has_header(true)
    .null_values(vec!["NA".into(), "".into()])
    .skip_rows(1)
    .read(reader)
    .unwrap();

// Write JSON
write_json(&df, &mut writer, JsonOrientation::Records).unwrap();
```

## Feature Flags

| Flag | Enables |
|------|---------|
| `csv` *(default)* | CSV reading and writing |
| `json` | JSON reading and writing |
| `full` | All formats |

## License

MIT
