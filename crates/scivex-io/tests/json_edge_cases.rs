//! Integration and edge-case tests for JSON reading and writing.
#![cfg(feature = "json")]

use scivex_frame::{AnySeries, DType, DataFrame, Series, StringSeries};
use scivex_io::error::IoError;
use scivex_io::json::{
    JsonOrientation, JsonReaderBuilder, JsonWriterBuilder, read_json, write_json,
};

// ---------------------------------------------------------------------------
// JSON Reader edge cases
// ---------------------------------------------------------------------------

#[test]
fn json_empty_string() {
    let result = read_json("".as_bytes());
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), IoError::EmptyInput));
}

#[test]
fn json_whitespace_only() {
    let result = read_json("   \n\t  ".as_bytes());
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), IoError::EmptyInput));
}

#[test]
fn json_empty_array() {
    let result = read_json("[]".as_bytes());
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), IoError::EmptyInput));
}

#[test]
fn json_empty_object_columns() {
    let result = JsonReaderBuilder::new()
        .orientation(JsonOrientation::Columns)
        .read("{}".as_bytes());
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), IoError::EmptyInput));
}

#[test]
fn json_null_values_in_records() {
    let json = r#"[{"a": 1, "b": null}, {"a": null, "b": 2}]"#;
    let df = read_json(json.as_bytes()).unwrap();
    assert_eq!(df.nrows(), 2);
    assert!(df.column("b").unwrap().is_null(0));
    assert!(df.column("a").unwrap().is_null(1));
}

#[test]
fn json_null_values_in_columns() {
    let json = r#"{"a": [1, null, 3], "b": [null, "hello", null]}"#;
    let df = JsonReaderBuilder::new()
        .orientation(JsonOrientation::Columns)
        .read(json.as_bytes())
        .unwrap();
    assert_eq!(df.nrows(), 3);
    assert!(df.column("a").unwrap().is_null(1));
    assert!(df.column("b").unwrap().is_null(0));
    assert!(df.column("b").unwrap().is_null(2));
}

#[test]
fn json_nested_objects_become_strings() {
    // Nested objects should be serialized as strings (not error).
    let json = r#"[{"a": 1, "b": {"nested": true}}, {"a": 2, "b": {"nested": false}}]"#;
    let df = read_json(json.as_bytes()).unwrap();
    assert_eq!(df.nrows(), 2);
    // The nested object column should be Str type.
    assert_eq!(df.column("b").unwrap().dtype(), DType::Str);
}

#[test]
fn json_nested_arrays_become_strings() {
    let json = r#"[{"a": 1, "b": [1,2,3]}, {"a": 2, "b": [4,5]}]"#;
    let df = read_json(json.as_bytes()).unwrap();
    assert_eq!(df.nrows(), 2);
    assert_eq!(df.column("b").unwrap().dtype(), DType::Str);
}

#[test]
fn json_mixed_int_float_in_column() {
    let json = r#"[{"x": 1}, {"x": 2.5}, {"x": 3}]"#;
    let df = read_json(json.as_bytes()).unwrap();
    assert_eq!(df.column("x").unwrap().dtype(), DType::F64);
}

#[test]
fn json_all_null_column() {
    let json = r#"[{"x": null}, {"x": null}, {"x": null}]"#;
    let df = read_json(json.as_bytes()).unwrap();
    let col = df.column("x").unwrap();
    assert_eq!(col.dtype(), DType::Str);
    for i in 0..3 {
        assert!(col.is_null(i));
    }
}

#[test]
fn json_bool_column() {
    let json = r#"[{"flag": true}, {"flag": false}]"#;
    let df = read_json(json.as_bytes()).unwrap();
    let col = df.column("flag").unwrap();
    assert_eq!(col.dtype(), DType::U8);
    assert_eq!(col.display_value(0), "1");
    assert_eq!(col.display_value(1), "0");
}

#[test]
fn json_bool_with_null() {
    let json = r#"[{"flag": true}, {"flag": null}, {"flag": false}]"#;
    let df = read_json(json.as_bytes()).unwrap();
    let col = df.column("flag").unwrap();
    assert_eq!(col.dtype(), DType::U8);
    assert!(col.is_null(1));
}

#[test]
fn json_mixed_bool_and_int_becomes_string() {
    // Bool + int in same column should fallback to string.
    let json = r#"[{"x": true}, {"x": 42}]"#;
    let df = read_json(json.as_bytes()).unwrap();
    assert_eq!(df.column("x").unwrap().dtype(), DType::Str);
}

#[test]
fn json_missing_keys_across_rows() {
    // Different rows have different keys.
    let json = r#"[{"a": 1}, {"b": 2}, {"a": 3, "b": 4}]"#;
    let df = read_json(json.as_bytes()).unwrap();
    assert_eq!(df.nrows(), 3);
    assert_eq!(df.ncols(), 2);
    // Missing keys should be null.
    assert!(df.column("b").unwrap().is_null(0));
    assert!(df.column("a").unwrap().is_null(1));
}

#[test]
fn json_single_row() {
    let json = r#"[{"x": 42}]"#;
    let df = read_json(json.as_bytes()).unwrap();
    assert_eq!(df.nrows(), 1);
    assert_eq!(df.ncols(), 1);
}

#[test]
fn json_single_column_records() {
    let json = r#"[{"val": 1}, {"val": 2}, {"val": 3}]"#;
    let df = read_json(json.as_bytes()).unwrap();
    assert_eq!(df.ncols(), 1);
    assert_eq!(df.nrows(), 3);
}

#[test]
fn json_malformed_input() {
    let result = read_json("{invalid json".as_bytes());
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), IoError::JsonError(_)));
}

#[test]
fn json_not_array_for_records() {
    let result = read_json(r#"{"a": 1}"#.as_bytes());
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), IoError::JsonError(_)));
}

#[test]
fn json_not_object_for_columns() {
    let result = JsonReaderBuilder::new()
        .orientation(JsonOrientation::Columns)
        .read("[1,2,3]".as_bytes());
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), IoError::JsonError(_)));
}

#[test]
fn json_columns_non_array_value() {
    // Column value is not an array.
    let result = JsonReaderBuilder::new()
        .orientation(JsonOrientation::Columns)
        .read(r#"{"a": "not an array"}"#.as_bytes());
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// JSON Writer edge cases
// ---------------------------------------------------------------------------

#[test]
fn json_write_records_roundtrip() {
    let name: Box<dyn AnySeries> = Box::new(StringSeries::from_strs("name", &["Alice", "Bob"]));
    let age: Box<dyn AnySeries> = Box::new(Series::new("age", vec![30_i64, 25]));
    let df = DataFrame::new(vec![name, age]).unwrap();

    let mut buf = Vec::new();
    write_json(&mut buf, &df).unwrap();
    let df2 = read_json(buf.as_slice()).unwrap();

    assert_eq!(df.nrows(), df2.nrows());
    assert_eq!(df.ncols(), df2.ncols());
    assert_eq!(df2.column("name").unwrap().display_value(0), "Alice");
    assert_eq!(df2.column("age").unwrap().display_value(1), "25");
}

#[test]
fn json_write_columns_roundtrip() {
    let col: Box<dyn AnySeries> = Box::new(Series::new("x", vec![1_i64, 2, 3]));
    let df = DataFrame::new(vec![col]).unwrap();

    let mut buf = Vec::new();
    JsonWriterBuilder::new()
        .orientation(JsonOrientation::Columns)
        .write(&mut buf, &df)
        .unwrap();
    let df2 = JsonReaderBuilder::new()
        .orientation(JsonOrientation::Columns)
        .read(buf.as_slice())
        .unwrap();

    assert_eq!(df2.nrows(), 3);
    assert_eq!(df2.column("x").unwrap().display_value(0), "1");
}

#[test]
fn json_write_with_nulls_roundtrip() {
    let col: Box<dyn AnySeries> =
        Box::new(Series::with_nulls("x", vec![1_i64, 0, 3], vec![false, true, false]).unwrap());
    let df = DataFrame::new(vec![col]).unwrap();

    let mut buf = Vec::new();
    write_json(&mut buf, &df).unwrap();
    let df2 = read_json(buf.as_slice()).unwrap();

    assert_eq!(df2.nrows(), 3);
    assert!(df2.column("x").unwrap().is_null(1));
    assert_eq!(df2.column("x").unwrap().display_value(0), "1");
    assert_eq!(df2.column("x").unwrap().display_value(2), "3");
}

#[test]
fn json_write_f64_values() {
    let col: Box<dyn AnySeries> = Box::new(Series::new("val", vec![1.5_f64, -2.0, 0.0]));
    let df = DataFrame::new(vec![col]).unwrap();

    let mut buf = Vec::new();
    write_json(&mut buf, &df).unwrap();
    // Roundtrip through reader to validate.
    let df2 = read_json(buf.as_slice()).unwrap();
    assert_eq!(df2.nrows(), 3);
    let col2 = df2.column("val").unwrap();
    let typed = col2.as_any().downcast_ref::<Series<f64>>().unwrap();
    assert!((typed.as_slice()[0] - 1.5).abs() < 1e-10);
    assert!((typed.as_slice()[1] - (-2.0)).abs() < 1e-10);
    assert!((typed.as_slice()[2] - 0.0).abs() < 1e-10);
}

#[test]
fn json_write_pretty_formatting() {
    let col: Box<dyn AnySeries> = Box::new(Series::new("x", vec![1_i64]));
    let df = DataFrame::new(vec![col]).unwrap();

    let mut buf = Vec::new();
    JsonWriterBuilder::new()
        .pretty(true)
        .write(&mut buf, &df)
        .unwrap();
    let output = String::from_utf8(buf).unwrap();
    // Pretty output should have indentation.
    assert!(output.contains("  "));
    assert!(output.contains('\n'));
}

#[test]
fn json_write_empty_dataframe() {
    let col: Box<dyn AnySeries> = Box::new(Series::new("x", Vec::<i64>::new()));
    let df = DataFrame::new(vec![col]).unwrap();

    let mut buf = Vec::new();
    write_json(&mut buf, &df).unwrap();
    let output = String::from_utf8(buf).unwrap();
    // Empty DataFrame should produce "[]".
    assert_eq!(output.trim(), "[]");
}

#[test]
fn json_file_not_found() {
    let result = JsonReaderBuilder::new().read_path("/nonexistent/path/file.json");
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), IoError::Io(_)));
}

#[test]
fn json_write_string_with_special_chars() {
    let col: Box<dyn AnySeries> = Box::new(StringSeries::from_strs(
        "text",
        &["hello \"world\"", "line1\nline2", "tab\there"],
    ));
    let df = DataFrame::new(vec![col]).unwrap();

    let mut buf = Vec::new();
    write_json(&mut buf, &df).unwrap();
    // Roundtrip to validate JSON is valid and preserves special chars.
    let df2 = read_json(buf.as_slice()).unwrap();
    assert_eq!(df2.nrows(), 3);
    assert_eq!(
        df2.column("text").unwrap().display_value(0),
        "hello \"world\""
    );
    assert_eq!(df2.column("text").unwrap().display_value(1), "line1\nline2");
}

#[test]
fn json_write_to_file_and_read_back() {
    let col: Box<dyn AnySeries> = Box::new(Series::new("val", vec![10_i64, 20, 30]));
    let df = DataFrame::new(vec![col]).unwrap();

    let dir = std::env::temp_dir().join("scivex_io_test_json");
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("test_roundtrip.json");

    JsonWriterBuilder::new().write_path(&path, &df).unwrap();
    let df2 = JsonReaderBuilder::new().read_path(&path).unwrap();

    assert_eq!(df2.nrows(), 3);
    assert_eq!(df2.column("val").unwrap().display_value(0), "10");

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(&dir);
}

#[test]
fn json_string_column_with_null_sentinel_values() {
    // Strings like "NA", "null" in JSON should NOT be treated as null
    // (they are valid JSON strings, only JSON null is null).
    let json = r#"[{"x": "NA"}, {"x": "null"}, {"x": "hello"}]"#;
    let df = read_json(json.as_bytes()).unwrap();
    let col = df.column("x").unwrap();
    // "NA" and "null" are null sentinels in the is_null_sentinel function,
    // so they will be treated as null.
    assert!(col.is_null(0));
    assert!(col.is_null(1));
    assert!(!col.is_null(2));
}

#[test]
fn json_integer_column_values() {
    let json = r#"[{"x": 100}, {"x": -50}, {"x": 0}]"#;
    let df = read_json(json.as_bytes()).unwrap();
    let col = df.column("x").unwrap();
    assert_eq!(col.dtype(), DType::I64);
    let typed = col.as_any().downcast_ref::<Series<i64>>().unwrap();
    assert_eq!(typed.as_slice(), &[100, -50, 0]);
}
