//! JSON writer that serialises a [`DataFrame`] to JSON format.

use std::io::Write;
use std::path::Path;

use serde_json::{Map, Value};

use scivex_frame::{AnySeries, DType, DataFrame, Series, StringSeries};

use super::reader::JsonOrientation;
use crate::error::Result;

/// Builder for writing a [`DataFrame`] as JSON.
///
/// # Example
///
/// ```no_run
/// use scivex_io::json::JsonWriterBuilder;
/// use scivex_frame::DataFrame;
///
/// fn example(df: &DataFrame) {
///     let mut buf = Vec::new();
///     JsonWriterBuilder::new()
///         .write(&mut buf, df)
///         .unwrap();
/// }
/// ```
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct JsonWriterBuilder {
    orientation: JsonOrientation,
    pretty: bool,
}

impl Default for JsonWriterBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl JsonWriterBuilder {
    /// Create a new builder with default settings (records, not pretty).
    pub fn new() -> Self {
        Self {
            orientation: JsonOrientation::Records,
            pretty: false,
        }
    }

    /// Set the JSON orientation.
    pub fn orientation(mut self, orientation: JsonOrientation) -> Self {
        self.orientation = orientation;
        self
    }

    /// Whether to pretty-print the JSON output.
    pub fn pretty(mut self, pretty: bool) -> Self {
        self.pretty = pretty;
        self
    }

    /// Write the `DataFrame` as JSON to a writer.
    pub fn write<W: Write>(self, mut writer: W, df: &DataFrame) -> Result<()> {
        let value = match self.orientation {
            JsonOrientation::Records => Self::to_records_value(df),
            JsonOrientation::Columns => Self::to_columns_value(df),
        };

        let json_str = if self.pretty {
            serde_json::to_string_pretty(&value)
        } else {
            serde_json::to_string(&value)
        }
        .map_err(|e| crate::error::IoError::JsonError(e.to_string()))?;

        writer.write_all(json_str.as_bytes())?;
        Ok(())
    }

    /// Write the `DataFrame` as JSON to a file.
    pub fn write_path<P: AsRef<Path>>(self, path: P, df: &DataFrame) -> Result<()> {
        let file = std::fs::File::create(path)?;
        let buf = std::io::BufWriter::new(file);
        self.write(buf, df)
    }

    fn to_records_value(df: &DataFrame) -> Value {
        let columns = df.columns();
        let names = df.column_names();
        let nrows = df.nrows();

        let mut rows: Vec<Value> = Vec::with_capacity(nrows);
        for row in 0..nrows {
            let mut obj = Map::new();
            for (col_idx, col) in columns.iter().enumerate() {
                let name = names[col_idx];
                let val = column_value_at(col.as_ref(), row);
                obj.insert(name.to_string(), val);
            }
            rows.push(Value::Object(obj));
        }
        Value::Array(rows)
    }

    fn to_columns_value(df: &DataFrame) -> Value {
        let columns = df.columns();
        let names = df.column_names();

        let mut obj = Map::new();
        for (col_idx, col) in columns.iter().enumerate() {
            let name = names[col_idx];
            let nrows = col.len();
            let mut vals: Vec<Value> = Vec::with_capacity(nrows);
            for row in 0..nrows {
                vals.push(column_value_at(col.as_ref(), row));
            }
            obj.insert(name.to_string(), Value::Array(vals));
        }
        Value::Object(obj)
    }
}

/// Convert a single cell to a typed [`serde_json::Value`].
fn column_value_at(col: &dyn AnySeries, row: usize) -> Value {
    if col.is_null(row) {
        return Value::Null;
    }

    match col.dtype() {
        DType::I64 => {
            if let Some(s) = col.as_any().downcast_ref::<Series<i64>>() {
                s.get(row).map_or(Value::Null, |v| Value::Number(v.into()))
            } else {
                Value::String(col.display_value(row))
            }
        }
        DType::I32 => {
            if let Some(s) = col.as_any().downcast_ref::<Series<i32>>() {
                s.get(row)
                    .map_or(Value::Null, |v| Value::Number(i64::from(v).into()))
            } else {
                Value::String(col.display_value(row))
            }
        }
        DType::I16 => {
            if let Some(s) = col.as_any().downcast_ref::<Series<i16>>() {
                s.get(row)
                    .map_or(Value::Null, |v| Value::Number(i64::from(v).into()))
            } else {
                Value::String(col.display_value(row))
            }
        }
        DType::I8 => {
            if let Some(s) = col.as_any().downcast_ref::<Series<i8>>() {
                s.get(row)
                    .map_or(Value::Null, |v| Value::Number(i64::from(v).into()))
            } else {
                Value::String(col.display_value(row))
            }
        }
        DType::U64 => {
            if let Some(s) = col.as_any().downcast_ref::<Series<u64>>() {
                s.get(row).map_or(Value::Null, |v| Value::Number(v.into()))
            } else {
                Value::String(col.display_value(row))
            }
        }
        DType::U32 => {
            if let Some(s) = col.as_any().downcast_ref::<Series<u32>>() {
                s.get(row)
                    .map_or(Value::Null, |v| Value::Number(u64::from(v).into()))
            } else {
                Value::String(col.display_value(row))
            }
        }
        DType::U16 => {
            if let Some(s) = col.as_any().downcast_ref::<Series<u16>>() {
                s.get(row)
                    .map_or(Value::Null, |v| Value::Number(u64::from(v).into()))
            } else {
                Value::String(col.display_value(row))
            }
        }
        DType::U8 => {
            if let Some(s) = col.as_any().downcast_ref::<Series<u8>>() {
                s.get(row)
                    .map_or(Value::Null, |v| Value::Number(u64::from(v).into()))
            } else {
                Value::String(col.display_value(row))
            }
        }
        DType::F64 => {
            if let Some(s) = col.as_any().downcast_ref::<Series<f64>>() {
                s.get(row).map_or(Value::Null, |v| {
                    serde_json::Number::from_f64(v)
                        .map_or_else(|| Value::String(v.to_string()), Value::Number)
                })
            } else {
                Value::String(col.display_value(row))
            }
        }
        DType::F32 => {
            if let Some(s) = col.as_any().downcast_ref::<Series<f32>>() {
                s.get(row).map_or(Value::Null, |v| {
                    serde_json::Number::from_f64(f64::from(v))
                        .map_or_else(|| Value::String(v.to_string()), Value::Number)
                })
            } else {
                Value::String(col.display_value(row))
            }
        }
        DType::Str => {
            if let Some(s) = col.as_any().downcast_ref::<StringSeries>() {
                s.get(row)
                    .map_or(Value::Null, |v| Value::String(v.to_string()))
            } else {
                Value::String(col.display_value(row))
            }
        }
        DType::Bool | DType::Categorical | DType::DateTime => Value::String(col.display_value(row)),
    }
}

/// Write a `DataFrame` as JSON to a writer with default settings.
pub fn write_json<W: Write>(writer: W, df: &DataFrame) -> Result<()> {
    JsonWriterBuilder::new().write(writer, df)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scivex_frame::{DataFrame, Series, StringSeries};

    fn test_df() -> DataFrame {
        let name: Box<dyn AnySeries> = Box::new(StringSeries::from_strs("name", &["Alice", "Bob"]));
        let age: Box<dyn AnySeries> = Box::new(Series::new("age", vec![30_i64, 25]));
        DataFrame::new(vec![name, age]).unwrap()
    }

    #[test]
    fn test_write_json_records() {
        let df = test_df();
        let mut buf = Vec::new();
        write_json(&mut buf, &df).unwrap();
        let output = String::from_utf8(buf).unwrap();
        let parsed: Value = serde_json::from_str(&output).unwrap();
        let arr = parsed.as_array().unwrap();
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0]["name"], "Alice");
        assert_eq!(arr[0]["age"], 30);
    }

    #[test]
    fn test_write_json_columns() {
        let df = test_df();
        let mut buf = Vec::new();
        JsonWriterBuilder::new()
            .orientation(JsonOrientation::Columns)
            .write(&mut buf, &df)
            .unwrap();
        let output = String::from_utf8(buf).unwrap();
        let parsed: Value = serde_json::from_str(&output).unwrap();
        let obj = parsed.as_object().unwrap();
        assert!(obj.contains_key("name"));
        assert!(obj.contains_key("age"));
        assert_eq!(obj["name"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn test_write_json_pretty() {
        let df = test_df();
        let mut buf = Vec::new();
        JsonWriterBuilder::new()
            .pretty(true)
            .write(&mut buf, &df)
            .unwrap();
        let output = String::from_utf8(buf).unwrap();
        assert!(output.contains('\n'));
        assert!(output.contains("  "));
    }

    #[test]
    fn test_write_json_with_nulls() {
        let col: Box<dyn AnySeries> =
            Box::new(Series::with_nulls("x", vec![1_i64, 0, 3], vec![false, true, false]).unwrap());
        let df = DataFrame::new(vec![col]).unwrap();
        let mut buf = Vec::new();
        write_json(&mut buf, &df).unwrap();
        let output = String::from_utf8(buf).unwrap();
        let parsed: Value = serde_json::from_str(&output).unwrap();
        let arr = parsed.as_array().unwrap();
        assert_eq!(arr[0]["x"], 1);
        assert!(arr[1]["x"].is_null());
        assert_eq!(arr[2]["x"], 3);
    }

    #[test]
    fn test_write_json_f64() {
        let col: Box<dyn AnySeries> = Box::new(Series::new("val", vec![1.5_f64, 2.0]));
        let df = DataFrame::new(vec![col]).unwrap();
        let mut buf = Vec::new();
        write_json(&mut buf, &df).unwrap();
        let output = String::from_utf8(buf).unwrap();
        let parsed: Value = serde_json::from_str(&output).unwrap();
        let arr = parsed.as_array().unwrap();
        assert_eq!(arr[0]["val"], 1.5);
    }

    #[test]
    fn test_roundtrip_records() {
        let df = test_df();
        let mut buf = Vec::new();
        write_json(&mut buf, &df).unwrap();
        let df2 = crate::json::read_json(buf.as_slice()).unwrap();
        assert_eq!(df.nrows(), df2.nrows());
        assert_eq!(df.ncols(), df2.ncols());
    }

    #[test]
    fn test_roundtrip_columns() {
        let df = test_df();
        let mut buf = Vec::new();
        JsonWriterBuilder::new()
            .orientation(JsonOrientation::Columns)
            .write(&mut buf, &df)
            .unwrap();
        let df2 = crate::json::JsonReaderBuilder::new()
            .orientation(JsonOrientation::Columns)
            .read(buf.as_slice())
            .unwrap();
        assert_eq!(df.nrows(), df2.nrows());
        assert_eq!(df.ncols(), df2.ncols());
    }
}
