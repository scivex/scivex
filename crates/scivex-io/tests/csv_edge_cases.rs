//! Integration and edge-case tests for CSV reading and writing.

use scivex_frame::{AnySeries, DType, DataFrame, Series, StringSeries};
use scivex_io::common::InferredType;
use scivex_io::csv::{CsvReaderBuilder, CsvWriterBuilder, QuoteStyle, read_csv, write_csv};
use scivex_io::error::IoError;

// ---------------------------------------------------------------------------
// CSV Reader edge cases
// ---------------------------------------------------------------------------

#[test]
fn csv_empty_file() {
    let result = read_csv("".as_bytes());
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), IoError::EmptyInput));
}

#[test]
fn csv_only_whitespace() {
    // A file that is only whitespace lines should still be parseable
    // (whitespace line is a record with one empty field).
    let csv = "   \n";
    // This has one line with content "   " which is a valid row.
    let result = CsvReaderBuilder::new()
        .has_header(false)
        .read(csv.as_bytes());
    // Should succeed -- whitespace is a valid field value.
    assert!(result.is_ok());
}

#[test]
fn csv_only_header_no_data_rows() {
    let csv = "name,age,score\n";
    let result = read_csv(csv.as_bytes());
    // Header exists but no data rows -- should be EmptyInput.
    assert!(result.is_err());
}

#[test]
fn csv_missing_values_empty_cells() {
    let csv = "a,b,c\n1,,3\n,2,\n,,\n";
    let df = read_csv(csv.as_bytes()).unwrap();
    assert_eq!(df.nrows(), 3);
    assert_eq!(df.ncols(), 3);
    // Empty cells should be null.
    let col_b = df.column("b").unwrap();
    assert!(col_b.is_null(0)); // ""
    let col_a = df.column("a").unwrap();
    assert!(col_a.is_null(1)); // ""
}

#[test]
fn csv_quoted_fields_containing_commas() {
    let csv = r#"city,description
"New York","Big city, lots of people"
"Paris","City of light, romance"
"#;
    let df = read_csv(csv.as_bytes()).unwrap();
    assert_eq!(df.nrows(), 2);
    let desc = df.column("description").unwrap();
    assert_eq!(desc.display_value(0), "Big city, lots of people");
    assert_eq!(desc.display_value(1), "City of light, romance");
}

#[test]
fn csv_quoted_fields_containing_newlines() {
    let csv = "a,b\n\"line1\nline2\",1\n\"single\",2\n";
    let df = read_csv(csv.as_bytes()).unwrap();
    assert_eq!(df.nrows(), 2);
    let col_a = df.column("a").unwrap();
    assert_eq!(col_a.display_value(0), "line1\nline2");
}

#[test]
fn csv_quoted_fields_containing_quotes() {
    let csv = "a\n\"he said \"\"hello\"\"\"\n\"normal\"\n";
    let df = read_csv(csv.as_bytes()).unwrap();
    assert_eq!(df.nrows(), 2);
    let col_a = df.column("a").unwrap();
    assert_eq!(col_a.display_value(0), "he said \"hello\"");
}

#[test]
fn csv_tab_delimiter() {
    let csv = "name\tvalue\nAlice\t42\nBob\t99\n";
    let df = CsvReaderBuilder::new()
        .delimiter(b'\t')
        .read(csv.as_bytes())
        .unwrap();
    assert_eq!(df.nrows(), 2);
    assert_eq!(df.column_names(), vec!["name", "value"]);
}

#[test]
fn csv_semicolon_delimiter() {
    let csv = "a;b;c\n1;2;3\n4;5;6\n";
    let df = CsvReaderBuilder::new()
        .delimiter(b';')
        .read(csv.as_bytes())
        .unwrap();
    assert_eq!(df.nrows(), 2);
    assert_eq!(df.ncols(), 3);
}

#[test]
fn csv_pipe_delimiter() {
    let csv = "x|y\n10|20\n30|40\n";
    let df = CsvReaderBuilder::new()
        .delimiter(b'|')
        .read(csv.as_bytes())
        .unwrap();
    assert_eq!(df.nrows(), 2);
    assert_eq!(df.column("x").unwrap().display_value(0), "10");
}

#[test]
fn csv_many_columns() {
    // 50 columns
    let ncols = 50;
    let header: String = (0..ncols)
        .map(|i| format!("c{i}"))
        .collect::<Vec<_>>()
        .join(",");
    let row: String = (0..ncols)
        .map(|i| i.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let csv = format!("{header}\n{row}\n");
    let df = read_csv(csv.as_bytes()).unwrap();
    assert_eq!(df.ncols(), ncols);
    assert_eq!(df.nrows(), 1);
    assert_eq!(df.column("c0").unwrap().display_value(0), "0");
    assert_eq!(df.column("c49").unwrap().display_value(0), "49");
}

#[test]
fn csv_single_column() {
    let csv = "value\n1\n2\n3\n";
    let df = read_csv(csv.as_bytes()).unwrap();
    assert_eq!(df.ncols(), 1);
    assert_eq!(df.nrows(), 3);
    assert_eq!(df.column_names(), vec!["value"]);
}

#[test]
fn csv_single_value() {
    let csv = "x\n42\n";
    let df = read_csv(csv.as_bytes()).unwrap();
    assert_eq!(df.nrows(), 1);
    assert_eq!(df.ncols(), 1);
}

#[test]
fn csv_infer_integers() {
    let csv = "x\n1\n2\n3\n";
    let df = read_csv(csv.as_bytes()).unwrap();
    assert_eq!(df.column("x").unwrap().dtype(), DType::I64);
}

#[test]
fn csv_infer_floats() {
    let csv = "x\n1.5\n2.0\n3.14\n";
    let df = read_csv(csv.as_bytes()).unwrap();
    assert_eq!(df.column("x").unwrap().dtype(), DType::F64);
}

#[test]
fn csv_infer_mixed_int_float() {
    let csv = "x\n1\n2.5\n3\n";
    let df = read_csv(csv.as_bytes()).unwrap();
    // Mix of int and float text should infer as F64.
    assert_eq!(df.column("x").unwrap().dtype(), DType::F64);
}

#[test]
fn csv_infer_booleans() {
    let csv = "flag\ntrue\nfalse\nTrue\n";
    let df = read_csv(csv.as_bytes()).unwrap();
    assert_eq!(df.column("flag").unwrap().dtype(), DType::U8);
}

#[test]
fn csv_infer_strings_fallback() {
    let csv = "x\nhello\nworld\n";
    let df = read_csv(csv.as_bytes()).unwrap();
    assert_eq!(df.column("x").unwrap().dtype(), DType::Str);
}

#[test]
fn csv_negative_integers() {
    let csv = "x\n-1\n-200\n0\n";
    let df = read_csv(csv.as_bytes()).unwrap();
    assert_eq!(df.column("x").unwrap().dtype(), DType::I64);
    let col = df.column("x").unwrap();
    let typed = col.as_any().downcast_ref::<Series<i64>>().unwrap();
    assert_eq!(typed.as_slice(), &[-1, -200, 0]);
}

#[test]
fn csv_negative_floats() {
    let csv = "x\n-1.5\n-0.001\n3.14\n";
    let df = read_csv(csv.as_bytes()).unwrap();
    assert_eq!(df.column("x").unwrap().dtype(), DType::F64);
}

#[test]
fn csv_write_and_readback_roundtrip() {
    let name: Box<dyn AnySeries> = Box::new(StringSeries::from_strs(
        "name",
        &["Alice", "Bob", "Charlie"],
    ));
    let age: Box<dyn AnySeries> = Box::new(Series::new("age", vec![30_i64, 25, 40]));
    let score: Box<dyn AnySeries> = Box::new(Series::new("score", vec![95.5_f64, 87.0, 92.3]));
    let df = DataFrame::new(vec![name, age, score]).unwrap();

    let mut buf = Vec::new();
    write_csv(&mut buf, &df).unwrap();
    let df2 = read_csv(buf.as_slice()).unwrap();

    assert_eq!(df.nrows(), df2.nrows());
    assert_eq!(df.ncols(), df2.ncols());
    assert_eq!(df.column_names(), df2.column_names());
    // Verify actual values survived the roundtrip.
    assert_eq!(df2.column("name").unwrap().display_value(0), "Alice");
    assert_eq!(df2.column("age").unwrap().display_value(1), "25");
}

#[test]
fn csv_roundtrip_with_special_chars() {
    let col: Box<dyn AnySeries> = Box::new(StringSeries::from_strs(
        "text",
        &["hello, world", "line1\nline2", "he said \"hi\""],
    ));
    let df = DataFrame::new(vec![col]).unwrap();

    let mut buf = Vec::new();
    write_csv(&mut buf, &df).unwrap();
    let df2 = read_csv(buf.as_slice()).unwrap();

    assert_eq!(df2.nrows(), 3);
    assert_eq!(df2.column("text").unwrap().display_value(0), "hello, world");
    assert_eq!(df2.column("text").unwrap().display_value(1), "line1\nline2");
    assert_eq!(
        df2.column("text").unwrap().display_value(2),
        "he said \"hi\""
    );
}

#[test]
fn csv_roundtrip_with_nulls() {
    let col: Box<dyn AnySeries> =
        Box::new(Series::with_nulls("x", vec![1_i64, 0, 3], vec![false, true, false]).unwrap());
    let df = DataFrame::new(vec![col]).unwrap();

    let mut buf = Vec::new();
    CsvWriterBuilder::new()
        .null_representation("NA".to_string())
        .write(&mut buf, &df)
        .unwrap();
    let df2 = CsvReaderBuilder::new()
        .null_values(vec!["NA".to_string()])
        .read(buf.as_slice())
        .unwrap();

    assert_eq!(df2.nrows(), 3);
    assert!(df2.column("x").unwrap().is_null(1));
}

#[test]
fn csv_custom_null_values() {
    let csv = "x\n1\nMISSING\n3\n";
    let df = CsvReaderBuilder::new()
        .null_values(vec!["MISSING".to_string()])
        .read(csv.as_bytes())
        .unwrap();
    assert!(df.column("x").unwrap().is_null(1));
}

#[test]
fn csv_skip_rows_beyond_data() {
    let csv = "only one line\n";
    let result = CsvReaderBuilder::new().skip_rows(5).read(csv.as_bytes());
    assert!(result.is_err());
}

#[test]
fn csv_max_rows_zero() {
    let csv = "x\n1\n2\n3\n";
    // max_rows(Some(0)) means read 0 data rows.
    let result = CsvReaderBuilder::new()
        .max_rows(Some(0))
        .read(csv.as_bytes());
    // No data rows -> EmptyInput.
    assert!(result.is_err());
}

#[test]
fn csv_max_rows_one() {
    let csv = "x\n1\n2\n3\n";
    let df = CsvReaderBuilder::new()
        .max_rows(Some(1))
        .read(csv.as_bytes())
        .unwrap();
    assert_eq!(df.nrows(), 1);
}

#[test]
fn csv_comment_char_hash() {
    let csv = "x\n# this is a comment\n1\n# another comment\n2\n";
    let df = CsvReaderBuilder::new()
        .comment_char(Some(b'#'))
        .read(csv.as_bytes())
        .unwrap();
    assert_eq!(df.nrows(), 2);
}

#[test]
fn csv_comment_char_semicolon() {
    let csv = "x\n; comment\n42\n";
    let df = CsvReaderBuilder::new()
        .comment_char(Some(b';'))
        .read(csv.as_bytes())
        .unwrap();
    assert_eq!(df.nrows(), 1);
    assert_eq!(df.column("x").unwrap().display_value(0), "42");
}

#[test]
fn csv_rows_with_fewer_fields_than_header() {
    // Row with fewer fields should fill missing columns with empty/default.
    let csv = "a,b,c\n1,2,3\n4\n";
    let df = read_csv(csv.as_bytes()).unwrap();
    assert_eq!(df.nrows(), 2);
    // Missing fields get "" which is null sentinel.
    assert!(df.column("b").unwrap().is_null(1));
    assert!(df.column("c").unwrap().is_null(1));
}

#[test]
fn csv_no_header_auto_column_names() {
    let csv = "1,2,3,4,5\n6,7,8,9,10\n";
    let df = CsvReaderBuilder::new()
        .has_header(false)
        .read(csv.as_bytes())
        .unwrap();
    assert_eq!(
        df.column_names(),
        vec!["column_0", "column_1", "column_2", "column_3", "column_4"]
    );
}

#[test]
fn csv_explicit_column_types_force_float() {
    let csv = "a,b\n1,hello\n2,world\n";
    let df = CsvReaderBuilder::new()
        .column_types(vec![("a".into(), InferredType::F64)])
        .read(csv.as_bytes())
        .unwrap();
    assert_eq!(df.column("a").unwrap().dtype(), DType::F64);
    assert_eq!(df.column("b").unwrap().dtype(), DType::Str);
}

#[test]
fn csv_explicit_column_types_force_string() {
    let csv = "a\n1\n2\n3\n";
    let df = CsvReaderBuilder::new()
        .column_types(vec![("a".into(), InferredType::Str)])
        .read(csv.as_bytes())
        .unwrap();
    assert_eq!(df.column("a").unwrap().dtype(), DType::Str);
}

#[test]
fn csv_writer_never_quote() {
    let col: Box<dyn AnySeries> = Box::new(StringSeries::from_strs("x", &["hello", "world"]));
    let df = DataFrame::new(vec![col]).unwrap();
    let mut buf = Vec::new();
    CsvWriterBuilder::new()
        .quote_style(QuoteStyle::Never)
        .write(&mut buf, &df)
        .unwrap();
    let output = String::from_utf8(buf).unwrap();
    assert!(!output.contains('"'));
}

#[test]
fn csv_writer_always_quote() {
    let col: Box<dyn AnySeries> = Box::new(StringSeries::from_strs("x", &["simple"]));
    let df = DataFrame::new(vec![col]).unwrap();
    let mut buf = Vec::new();
    CsvWriterBuilder::new()
        .quote_style(QuoteStyle::Always)
        .write(&mut buf, &df)
        .unwrap();
    let output = String::from_utf8(buf).unwrap();
    assert!(output.contains("\"x\""));
    assert!(output.contains("\"simple\""));
}

#[test]
fn csv_writer_empty_dataframe_no_rows() {
    // A DataFrame with columns but 0 rows.
    let col: Box<dyn AnySeries> = Box::new(Series::new("x", Vec::<i64>::new()));
    let df = DataFrame::new(vec![col]).unwrap();
    let mut buf = Vec::new();
    write_csv(&mut buf, &df).unwrap();
    let output = String::from_utf8(buf).unwrap();
    // Should just have the header.
    assert_eq!(output.trim(), "x");
}

#[test]
fn csv_file_not_found() {
    let result = CsvReaderBuilder::new().read_path("/nonexistent/path/file.csv");
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), IoError::Io(_)));
}

#[test]
fn csv_write_to_file_and_read_back() {
    let col: Box<dyn AnySeries> = Box::new(Series::new("val", vec![10_i64, 20, 30]));
    let df = DataFrame::new(vec![col]).unwrap();

    let dir = std::env::temp_dir().join("scivex_io_test_csv");
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("test_roundtrip.csv");

    CsvWriterBuilder::new().write_path(&path, &df).unwrap();
    let df2 = CsvReaderBuilder::new().read_path(&path).unwrap();

    assert_eq!(df2.nrows(), 3);
    assert_eq!(df2.column("val").unwrap().display_value(0), "10");

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(&dir);
}

#[test]
fn csv_large_integers() {
    let csv = "x\n9999999999\n-9999999999\n0\n";
    let df = read_csv(csv.as_bytes()).unwrap();
    assert_eq!(df.column("x").unwrap().dtype(), DType::I64);
    let col = df.column("x").unwrap();
    let typed = col.as_any().downcast_ref::<Series<i64>>().unwrap();
    assert_eq!(typed.as_slice(), &[9_999_999_999_i64, -9_999_999_999, 0]);
}

#[test]
fn csv_scientific_notation_floats() {
    let csv = "x\n1.5e2\n3.14e-1\n";
    let df = read_csv(csv.as_bytes()).unwrap();
    assert_eq!(df.column("x").unwrap().dtype(), DType::F64);
    let col = df.column("x").unwrap();
    let typed = col.as_any().downcast_ref::<Series<f64>>().unwrap();
    assert!((typed.as_slice()[0] - 150.0).abs() < 1e-10);
    assert!((typed.as_slice()[1] - 0.314).abs() < 1e-10);
}

#[test]
fn csv_trim_whitespace_values() {
    let csv = " a , b \n 1 , hello \n 2 , world \n";
    let df = CsvReaderBuilder::new()
        .trim_whitespace(true)
        .read(csv.as_bytes())
        .unwrap();
    assert_eq!(df.column_names(), vec!["a", "b"]);
    assert_eq!(df.column("b").unwrap().display_value(0), "hello");
    assert_eq!(df.column("a").unwrap().display_value(0), "1");
}

#[test]
fn csv_all_null_sentinels() {
    let csv = "x\nNA\nN/A\nnull\nNULL\nNone\nNaN\nnan\n.\n-\n";
    let df = read_csv(csv.as_bytes()).unwrap();
    let col = df.column("x").unwrap();
    // All values are null sentinels, so all should be null.
    for i in 0..col.len() {
        assert!(col.is_null(i), "row {i} should be null");
    }
}
