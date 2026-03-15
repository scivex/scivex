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
}
