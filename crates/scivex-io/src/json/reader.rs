//! JSON reader that produces a [`DataFrame`].

use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;

use serde_json::Value;

use scivex_frame::{AnySeries, DataFrame, Series, StringSeries};

use crate::common::is_null_sentinel;
use crate::error::{IoError, Result};

/// Orientation of JSON data.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JsonOrientation {
    /// Array of row objects: `[{"col": val}, ...]`
    Records,
    /// Object of column arrays: `{"col": [val, ...], ...}`
    Columns,
}

/// Builder for reading JSON data into a [`DataFrame`].
///
/// # Example
///
/// ```no_run
/// use scivex_io::json::JsonReaderBuilder;
///
/// let json = r#"[{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]"#;
/// let df = JsonReaderBuilder::new()
///     .read(json.as_bytes())
///     .unwrap();
/// ```
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct JsonReaderBuilder {
    orientation: JsonOrientation,
}

impl Default for JsonReaderBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl JsonReaderBuilder {
    /// Create a new builder with default settings (records orientation).
    pub fn new() -> Self {
        Self {
            orientation: JsonOrientation::Records,
        }
    }

    /// Set the JSON orientation.
    pub fn orientation(mut self, orientation: JsonOrientation) -> Self {
        self.orientation = orientation;
        self
    }

    /// Read JSON from any [`Read`] implementation.
    pub fn read<R: Read>(self, mut reader: R) -> Result<DataFrame> {
        let mut buf = String::new();
        reader.read_to_string(&mut buf)?;
        if buf.trim().is_empty() {
            return Err(IoError::EmptyInput);
        }
        let value: Value =
            serde_json::from_str(&buf).map_err(|e| IoError::JsonError(e.to_string()))?;
        match self.orientation {
            JsonOrientation::Records => parse_records(&value),
            JsonOrientation::Columns => parse_columns(&value),
        }
    }

    /// Read JSON from a file path.
    pub fn read_path<P: AsRef<Path>>(self, path: P) -> Result<DataFrame> {
        let file = std::fs::File::open(path)?;
        self.read(file)
    }
}

/// Read JSON with default settings (records orientation).
pub fn read_json<R: Read>(reader: R) -> Result<DataFrame> {
    JsonReaderBuilder::new().read(reader)
}

/// Read JSON from a file path with default settings.
pub fn read_json_path<P: AsRef<Path>>(path: P) -> Result<DataFrame> {
    JsonReaderBuilder::new().read_path(path)
}

/// Parse records orientation: `[{"col": val}, ...]`
fn parse_records(value: &Value) -> Result<DataFrame> {
    let array = value.as_array().ok_or_else(|| {
        IoError::JsonError("expected a JSON array for records orientation".into())
    })?;

    if array.is_empty() {
        return Err(IoError::EmptyInput);
    }

    // Collect all unique keys in order of first appearance.
    let mut col_names: Vec<String> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for row in array {
        if let Some(obj) = row.as_object() {
            for key in obj.keys() {
                if seen.insert(key.clone()) {
                    col_names.push(key.clone());
                }
            }
        }
    }

    if col_names.is_empty() {
        return Err(IoError::JsonError(
            "no columns found in JSON records".into(),
        ));
    }

    // Collect column values.
    let nrows = array.len();
    let mut col_values: BTreeMap<&str, Vec<&Value>> = BTreeMap::new();
    for name in &col_names {
        col_values.insert(name.as_str(), Vec::with_capacity(nrows));
    }

    let null_val = Value::Null;
    for row in array {
        if let Some(obj) = row.as_object() {
            for name in &col_names {
                let val = obj.get(name.as_str()).unwrap_or(&null_val);
                col_values
                    .get_mut(name.as_str())
                    .expect("key was inserted from col_names above")
                    .push(val);
            }
        } else {
            // Non-object row — fill with nulls.
            for name in &col_names {
                col_values
                    .get_mut(name.as_str())
                    .expect("key was inserted from col_names above")
                    .push(&null_val);
            }
        }
    }

    // Build series for each column.
    let mut series: Vec<Box<dyn AnySeries>> = Vec::with_capacity(col_names.len());
    for name in &col_names {
        let values = &col_values[name.as_str()];
        let s = build_series_from_json_values(name, values)?;
        series.push(s);
    }

    Ok(DataFrame::new(series)?)
}

/// Parse columns orientation: `{"col": [val, ...], ...}`
fn parse_columns(value: &Value) -> Result<DataFrame> {
    let obj = value.as_object().ok_or_else(|| {
        IoError::JsonError("expected a JSON object for columns orientation".into())
    })?;

    if obj.is_empty() {
        return Err(IoError::EmptyInput);
    }

    let mut series: Vec<Box<dyn AnySeries>> = Vec::with_capacity(obj.len());

    for (name, col_val) in obj {
        let array = col_val
            .as_array()
            .ok_or_else(|| IoError::JsonError(format!("expected array for column {name:?}")))?;
        let val_refs: Vec<&Value> = array.iter().collect();
        let s = build_series_from_json_values(name, &val_refs)?;
        series.push(s);
    }

    Ok(DataFrame::new(series)?)
}

/// Infer the type from a list of JSON values and build the appropriate series.
fn build_series_from_json_values(name: &str, values: &[&Value]) -> Result<Box<dyn AnySeries>> {
    // Determine the best type:
    // - all integers → I64
    // - any float (or mix of int+float) → F64
    // - all bools → Bool (stored as u8)
    // - else → Str
    let mut has_int = false;
    let mut has_float = false;
    let mut has_bool = false;
    let mut has_string = false;

    for v in values {
        match v {
            Value::Null => {}
            Value::Bool(_) => has_bool = true,
            Value::Number(n) => {
                if n.is_f64() && n.as_i64().is_none() {
                    has_float = true;
                } else {
                    has_int = true;
                }
            }
            Value::String(_) | Value::Array(_) | Value::Object(_) => has_string = true,
        }
    }

    if has_string || (has_bool && (has_int || has_float)) {
        build_json_string_series(name, values)
    } else if has_bool {
        build_json_bool_series(name, values)
    } else if has_float {
        build_json_f64_series(name, values)
    } else if has_int {
        build_json_i64_series(name, values)
    } else {
        // All nulls.
        build_json_string_series(name, values)
    }
}

fn build_json_i64_series(name: &str, values: &[&Value]) -> Result<Box<dyn AnySeries>> {
    let mut data = Vec::with_capacity(values.len());
    let mut nulls = Vec::with_capacity(values.len());
    let mut has_nulls = false;

    for v in values {
        if let Value::Number(n) = v {
            data.push(n.as_i64().unwrap_or(0));
            nulls.push(false);
        } else {
            data.push(0);
            nulls.push(true);
            has_nulls = true;
        }
    }

    if has_nulls {
        Ok(Box::new(Series::with_nulls(name, data, nulls)?))
    } else {
        Ok(Box::new(Series::new(name, data)))
    }
}

fn build_json_f64_series(name: &str, values: &[&Value]) -> Result<Box<dyn AnySeries>> {
    let mut data = Vec::with_capacity(values.len());
    let mut nulls = Vec::with_capacity(values.len());
    let mut has_nulls = false;

    for v in values {
        if let Value::Number(n) = v {
            data.push(n.as_f64().unwrap_or(0.0));
            nulls.push(false);
        } else {
            data.push(0.0);
            nulls.push(true);
            has_nulls = true;
        }
    }

    if has_nulls {
        Ok(Box::new(Series::with_nulls(name, data, nulls)?))
    } else {
        Ok(Box::new(Series::new(name, data)))
    }
}

fn build_json_bool_series(name: &str, values: &[&Value]) -> Result<Box<dyn AnySeries>> {
    let mut data = Vec::with_capacity(values.len());
    let mut nulls = Vec::with_capacity(values.len());
    let mut has_nulls = false;

    for v in values {
        if let Value::Bool(b) = v {
            data.push(u8::from(*b));
            nulls.push(false);
        } else {
            data.push(0);
            nulls.push(true);
            has_nulls = true;
        }
    }

    if has_nulls {
        Ok(Box::new(Series::with_nulls(name, data, nulls)?))
    } else {
        Ok(Box::new(Series::new(name, data)))
    }
}

fn build_json_string_series(name: &str, values: &[&Value]) -> Result<Box<dyn AnySeries>> {
    let mut data = Vec::with_capacity(values.len());
    let mut nulls = Vec::with_capacity(values.len());
    let mut has_nulls = false;

    for v in values {
        match v {
            Value::Null => {
                data.push(String::new());
                nulls.push(true);
                has_nulls = true;
            }
            Value::String(s) => {
                if is_null_sentinel(s) {
                    data.push(String::new());
                    nulls.push(true);
                    has_nulls = true;
                } else {
                    data.push(s.clone());
                    nulls.push(false);
                }
            }
            Value::Bool(b) => {
                data.push(b.to_string());
                nulls.push(false);
            }
            Value::Number(n) => {
                data.push(n.to_string());
                nulls.push(false);
            }
            other => {
                data.push(other.to_string());
                nulls.push(false);
            }
        }
    }

    if has_nulls {
        Ok(Box::new(StringSeries::with_nulls(name, data, nulls)?))
    } else {
        Ok(Box::new(StringSeries::new(name, data)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scivex_frame::DType;

    #[test]
    fn test_read_json_records() {
        let json = r#"[{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]"#;
        let df = read_json(json.as_bytes()).unwrap();
        assert_eq!(df.nrows(), 2);
        assert_eq!(df.ncols(), 2);
    }

    #[test]
    fn test_read_json_records_types() {
        let json = r#"[{"name": "Alice", "age": 30, "score": 95.5}, {"name": "Bob", "age": 25, "score": 87.0}]"#;
        let df = read_json(json.as_bytes()).unwrap();
        assert_eq!(df.column("name").unwrap().dtype(), DType::Str);
        assert_eq!(df.column("age").unwrap().dtype(), DType::I64);
        assert_eq!(df.column("score").unwrap().dtype(), DType::F64);
    }

    #[test]
    fn test_read_json_records_with_nulls() {
        let json = r#"[{"a": 1, "b": "x"}, {"a": null, "b": "y"}, {"a": 3}]"#;
        let df = read_json(json.as_bytes()).unwrap();
        assert_eq!(df.nrows(), 3);
        let col_a = df.column("a").unwrap();
        assert!(col_a.is_null(1));
        let col_b = df.column("b").unwrap();
        assert!(col_b.is_null(2)); // missing key
    }

    #[test]
    fn test_read_json_columns() {
        let json = r#"{"name": ["Alice", "Bob"], "age": [30, 25]}"#;
        let df = JsonReaderBuilder::new()
            .orientation(JsonOrientation::Columns)
            .read(json.as_bytes())
            .unwrap();
        assert_eq!(df.nrows(), 2);
        assert_eq!(df.ncols(), 2);
    }

    #[test]
    fn test_read_json_bool_column() {
        let json = r#"[{"flag": true}, {"flag": false}, {"flag": null}]"#;
        let df = read_json(json.as_bytes()).unwrap();
        let col = df.column("flag").unwrap();
        assert_eq!(col.dtype(), DType::U8);
        assert!(!col.is_null(0));
        assert!(col.is_null(2));
    }

    #[test]
    fn test_read_json_empty_array() {
        let json = "[]";
        let result = read_json(json.as_bytes());
        assert!(result.is_err());
    }

    #[test]
    fn test_read_json_empty_input() {
        let result = read_json("".as_bytes());
        assert!(result.is_err());
    }

    #[test]
    fn test_read_json_mixed_int_float() {
        let json = r#"[{"x": 1}, {"x": 2.5}]"#;
        let df = read_json(json.as_bytes()).unwrap();
        assert_eq!(df.column("x").unwrap().dtype(), DType::F64);
    }

    #[test]
    fn test_read_json_all_nulls() {
        let json = r#"[{"x": null}, {"x": null}]"#;
        let df = read_json(json.as_bytes()).unwrap();
        let col = df.column("x").unwrap();
        assert_eq!(col.dtype(), DType::Str);
        assert!(col.is_null(0));
        assert!(col.is_null(1));
    }
}
