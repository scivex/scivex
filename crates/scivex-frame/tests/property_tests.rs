#![allow(clippy::stable_sort_primitive)]
//! Property-based tests for scivex-frame using proptest.

use proptest::prelude::*;
use scivex_frame::{DataFrame, Series};

// ---------------------------------------------------------------------------
// Series properties
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn series_preserves_data(data in proptest::collection::vec(-1000.0_f64..1000.0, 1..500)) {
        let s = Series::new("test", data.clone());
        prop_assert_eq!(s.len(), data.len());
        prop_assert_eq!(s.as_slice(), &data[..]);
        prop_assert_eq!(s.name(), "test");
    }

    #[test]
    fn series_filter_reduces_length(data in proptest::collection::vec(-100.0_f64..100.0, 5..200)) {
        let s = Series::new("x", data.clone());
        let mask: Vec<bool> = data.iter().map(|v| *v > 0.0).collect();
        let filtered = s.filter(&mask).unwrap();
        let expected_count = mask.iter().filter(|&&b| b).count();
        prop_assert_eq!(filtered.len(), expected_count);
    }
}

// ---------------------------------------------------------------------------
// DataFrame properties
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn dataframe_shape_correct(n in 1usize..200) {
        let col_a: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let col_b: Vec<f64> = (0..n).map(|i| (i * 2) as f64).collect();
        let df = DataFrame::builder()
            .add_column("a", col_a)
            .add_column("b", col_b)
            .build()
            .unwrap();
        prop_assert_eq!(df.nrows(), n);
        prop_assert_eq!(df.ncols(), 2);
        prop_assert_eq!(df.shape(), (n, 2));
    }

    #[test]
    fn dataframe_filter_preserves_subset(
        data in proptest::collection::vec(-100.0_f64..100.0, 10..100)
    ) {
        let df = DataFrame::builder()
            .add_column("x", data.clone())
            .build()
            .unwrap();
        let mask: Vec<bool> = data.iter().map(|v| *v > 0.0).collect();
        let filtered = df.filter(&mask).unwrap();
        let expected = mask.iter().filter(|&&b| b).count();
        prop_assert_eq!(filtered.nrows(), expected);
        prop_assert_eq!(filtered.ncols(), 1);
    }

    #[test]
    fn dataframe_select_columns(n in 1usize..100) {
        let col_a: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let col_b: Vec<f64> = (0..n).map(|i| (i * 2) as f64).collect();
        let col_c: Vec<f64> = (0..n).map(|i| (i * 3) as f64).collect();
        let df = DataFrame::builder()
            .add_column("a", col_a)
            .add_column("b", col_b)
            .add_column("c", col_c)
            .build()
            .unwrap();
        let selected = df.select(&["a", "c"]).unwrap();
        prop_assert_eq!(selected.ncols(), 2);
        prop_assert_eq!(selected.nrows(), n);
    }

    #[test]
    fn dataframe_sort_preserves_length(
        data in proptest::collection::vec(-100.0_f64..100.0, 5..200)
    ) {
        let n = data.len();
        let df = DataFrame::builder()
            .add_column("x", data)
            .build()
            .unwrap();
        let sorted = df.sort_by("x", true).unwrap();
        prop_assert_eq!(sorted.nrows(), n);
        prop_assert_eq!(sorted.ncols(), 1);
    }

    #[test]
    fn dataframe_sort_produces_ordered_output(
        data in proptest::collection::vec(-100.0_f64..100.0, 5..100)
    ) {
        let df = DataFrame::builder()
            .add_column("x", data)
            .build()
            .unwrap();
        let sorted = df.sort_by("x", true).unwrap();
        let col = sorted.column_typed::<f64>("x").unwrap();
        let vals = col.as_slice();
        for i in 1..vals.len() {
            prop_assert!(vals[i] >= vals[i - 1], "sort not ascending at index {}", i);
        }
    }

    #[test]
    fn dataframe_filter_then_nrows(
        data in proptest::collection::vec(-100.0_f64..100.0, 10..100),
        threshold in -100.0_f64..100.0,
    ) {
        let expected = data.iter().filter(|&&v| v > threshold).count();
        let mask: Vec<bool> = data.iter().map(|v| *v > threshold).collect();
        let df = DataFrame::builder()
            .add_column("x", data)
            .build()
            .unwrap();
        let filtered = df.filter(&mask).unwrap();
        prop_assert_eq!(filtered.nrows(), expected);
    }

    #[test]
    fn dataframe_multi_column_consistency(n in 2usize..200) {
        let col_a: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let col_b: Vec<f64> = (0..n).map(|i| (i * 2) as f64).collect();
        let col_c: Vec<f64> = (0..n).map(|i| (i * 3) as f64).collect();
        let df = DataFrame::builder()
            .add_column("a", col_a)
            .add_column("b", col_b)
            .add_column("c", col_c)
            .build()
            .unwrap();
        // All columns same length after any operation
        prop_assert_eq!(df.nrows(), n);
        prop_assert_eq!(df.ncols(), 3);
        // Column names unique
        let names = df.column_names();
        let mut unique = names.clone();
        unique.sort();
        unique.dedup();
        prop_assert_eq!(names.len(), unique.len(), "column names not unique");
    }
}
