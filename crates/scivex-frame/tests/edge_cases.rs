#![allow(
    clippy::float_cmp,
    clippy::cast_lossless,
    clippy::needless_borrows_for_generic_args
)]
//! Edge case tests for DataFrame and Series operations.
//!
//! Tests empty DataFrames, single-row/column frames, all-null columns,
//! Unicode column names, very wide frames, and groupby edge cases.

use scivex_frame::series::string::StringSeries;
use scivex_frame::{DataFrame, Series};

// ===========================================================================
// Empty DataFrame
// ===========================================================================

#[test]
fn empty_dataframe_shape() {
    let df = DataFrame::builder().build().unwrap();
    assert_eq!(df.nrows(), 0);
    assert_eq!(df.ncols(), 0);
    assert_eq!(df.shape(), (0, 0));
}

#[test]
fn empty_dataframe_column_names() {
    let df = DataFrame::builder().build().unwrap();
    assert!(df.column_names().is_empty());
}

// ===========================================================================
// Single row / single column
// ===========================================================================

#[test]
fn single_row_dataframe() {
    let df = DataFrame::builder()
        .add_column("x", vec![42.0_f64])
        .add_column("y", vec![99.0_f64])
        .build()
        .unwrap();
    assert_eq!(df.nrows(), 1);
    assert_eq!(df.ncols(), 2);
}

#[test]
fn single_column_dataframe() {
    let df = DataFrame::builder()
        .add_column("only", vec![1.0_f64, 2.0, 3.0])
        .build()
        .unwrap();
    assert_eq!(df.nrows(), 3);
    assert_eq!(df.ncols(), 1);
}

#[test]
fn single_element_dataframe() {
    let df = DataFrame::builder()
        .add_column("val", vec![7.0_f64])
        .build()
        .unwrap();
    assert_eq!(df.shape(), (1, 1));
}

// ===========================================================================
// Series edge cases
// ===========================================================================

#[test]
fn series_single_element() {
    let s = Series::new("x", vec![42.0_f64]);
    assert_eq!(s.len(), 1);
    assert_eq!(s.name(), "x");
    assert_eq!(s.as_slice(), &[42.0]);
}

#[test]
fn series_large() {
    let data: Vec<f64> = (0..100_000).map(|i| i as f64).collect();
    let s = Series::new("big", data.clone());
    assert_eq!(s.len(), 100_000);
    assert_eq!(s.as_slice()[99_999], 99_999.0);
}

#[test]
fn series_all_same_values() {
    let s = Series::new("constant", vec![5.0_f64; 1000]);
    assert_eq!(s.len(), 1000);
    assert!(s.as_slice().iter().all(|&v| (v - 5.0).abs() < f64::EPSILON));
}

#[test]
fn series_negative_values() {
    let s = Series::new("neg", vec![-1.0_f64, -2.0, -3.0]);
    assert_eq!(s.as_slice(), &[-1.0, -2.0, -3.0]);
}

#[test]
fn series_nan_values() {
    let s = Series::new("nan", vec![f64::NAN, 1.0, 2.0]);
    assert_eq!(s.len(), 3);
    assert!(s.as_slice()[0].is_nan());
}

#[test]
fn series_inf_values() {
    let s = Series::new("inf", vec![f64::INFINITY, f64::NEG_INFINITY, 0.0]);
    assert_eq!(s.len(), 3);
    assert!(s.as_slice()[0].is_infinite());
    assert!(s.as_slice()[1].is_infinite());
}

// ===========================================================================
// Unicode column names
// ===========================================================================

#[test]
fn unicode_column_names() {
    let df = DataFrame::builder()
        .add_column("名前", vec![1.0_f64, 2.0])
        .add_column("données", vec![3.0_f64, 4.0])
        .add_column("数据", vec![5.0_f64, 6.0])
        .build()
        .unwrap();
    assert_eq!(df.ncols(), 3);
    let names = df.column_names();
    assert!(names.contains(&"名前"));
}

#[test]
fn empty_string_column_name() {
    let df = DataFrame::builder()
        .add_column("", vec![1.0_f64, 2.0])
        .build()
        .unwrap();
    assert_eq!(df.ncols(), 1);
}

// ===========================================================================
// Wide DataFrames
// ===========================================================================

#[test]
fn wide_dataframe_100_columns() {
    let mut builder = DataFrame::builder();
    for i in 0..100 {
        builder = builder.add_column(&format!("col_{i}"), vec![i as f64; 10]);
    }
    let df = builder.build().unwrap();
    assert_eq!(df.ncols(), 100);
    assert_eq!(df.nrows(), 10);
}

#[test]
fn wide_dataframe_select_subset() {
    let mut builder = DataFrame::builder();
    for i in 0..50 {
        builder = builder.add_column(&format!("c{i}"), vec![i as f64; 5]);
    }
    let df = builder.build().unwrap();
    let selected = df.select(&["c0", "c25", "c49"]).unwrap();
    assert_eq!(selected.ncols(), 3);
    assert_eq!(selected.nrows(), 5);
}

// ===========================================================================
// Sort edge cases
// ===========================================================================

#[test]
fn sort_single_row() {
    let df = DataFrame::builder()
        .add_column("x", vec![42.0_f64])
        .build()
        .unwrap();
    let sorted = df.sort_by("x", true).unwrap();
    assert_eq!(sorted.nrows(), 1);
}

#[test]
fn sort_already_sorted() {
    let df = DataFrame::builder()
        .add_column("x", vec![1.0_f64, 2.0, 3.0, 4.0, 5.0])
        .build()
        .unwrap();
    let sorted = df.sort_by("x", true).unwrap();
    assert_eq!(sorted.nrows(), 5);
    let col = sorted.column_typed::<f64>("x").unwrap();
    let data = col.as_slice();
    for i in 1..data.len() {
        assert!(data[i] >= data[i - 1]);
    }
}

#[test]
fn sort_reverse_sorted() {
    let df = DataFrame::builder()
        .add_column("x", vec![5.0_f64, 4.0, 3.0, 2.0, 1.0])
        .build()
        .unwrap();
    let sorted = df.sort_by("x", true).unwrap();
    let col = sorted.column_typed::<f64>("x").unwrap();
    let data = col.as_slice();
    for i in 1..data.len() {
        assert!(data[i] >= data[i - 1]);
    }
}

#[test]
fn sort_all_same_values() {
    let df = DataFrame::builder()
        .add_column("x", vec![7.0_f64; 100])
        .build()
        .unwrap();
    let sorted = df.sort_by("x", true).unwrap();
    assert_eq!(sorted.nrows(), 100);
}

// ===========================================================================
// Filter edge cases
// ===========================================================================

#[test]
fn filter_all_true() {
    let df = DataFrame::builder()
        .add_column("x", vec![1.0_f64, 2.0, 3.0])
        .build()
        .unwrap();
    let mask = vec![true, true, true];
    let filtered = df.filter(&mask).unwrap();
    assert_eq!(filtered.nrows(), 3);
}

#[test]
fn filter_all_false() {
    let df = DataFrame::builder()
        .add_column("x", vec![1.0_f64, 2.0, 3.0])
        .build()
        .unwrap();
    let mask = vec![false, false, false];
    let filtered = df.filter(&mask).unwrap();
    assert_eq!(filtered.nrows(), 0);
}

#[test]
fn filter_single_true() {
    let df = DataFrame::builder()
        .add_column("x", vec![10.0_f64, 20.0, 30.0])
        .build()
        .unwrap();
    let mask = vec![false, true, false];
    let filtered = df.filter(&mask).unwrap();
    assert_eq!(filtered.nrows(), 1);
}

// ===========================================================================
// Column access edge cases
// ===========================================================================

#[test]
fn nonexistent_column_returns_error() {
    let df = DataFrame::builder()
        .add_column("x", vec![1.0_f64])
        .build()
        .unwrap();
    assert!(df.column("nonexistent").is_err());
}

#[test]
fn select_nonexistent_column_returns_error() {
    let df = DataFrame::builder()
        .add_column("x", vec![1.0_f64])
        .build()
        .unwrap();
    assert!(df.select(&["x", "missing"]).is_err());
}

// ===========================================================================
// Integer Series
// ===========================================================================

#[test]
fn i64_series() {
    let s = Series::new("ids", vec![1_i64, 2, 3, 4, 5]);
    assert_eq!(s.len(), 5);
}

#[test]
fn i32_series() {
    let s = Series::new("counts", vec![10_i32, 20, 30]);
    assert_eq!(s.len(), 3);
}

// ===========================================================================
// String Series
// ===========================================================================

#[test]
fn string_series() {
    let s = StringSeries::from_strs("names", &["Alice", "Bob", "Carol"]);
    assert_eq!(s.len(), 3);
}

#[test]
fn unicode_string_series() {
    let s = StringSeries::from_strs("text", &["こんにちは", "مرحبا", "Привет"]);
    assert_eq!(s.len(), 3);
}

#[test]
fn empty_string_series() {
    let s = StringSeries::from_strs("empty", &["", "", ""]);
    assert_eq!(s.len(), 3);
}
