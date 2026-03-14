//! Missing data handling for [`Series`] and [`StringSeries`].

use scivex_core::{Float, Scalar};

use super::Series;
use crate::series::string::StringSeries;

// ---------------------------------------------------------------------------
// Series<T: Scalar> — missing data methods
// ---------------------------------------------------------------------------

impl<T: Scalar> Series<T> {
    /// Replace null values with a constant.
    pub fn fill_null(&self, value: T) -> Series<T> {
        let data: Vec<T> = self
            .data
            .iter()
            .enumerate()
            .map(|(i, &v)| if self.is_null_at(i) { value } else { v })
            .collect();
        Series::new(self.name.clone(), data)
    }

    /// Forward-fill: replace each null with the last non-null value.
    /// Leading nulls remain null.
    pub fn fill_forward(&self) -> Series<T> {
        let mut data = self.data.clone();
        let mut null_mask = self
            .null_mask
            .clone()
            .unwrap_or_else(|| vec![false; data.len()]);
        let mut last_valid: Option<T> = None;
        for i in 0..data.len() {
            if null_mask[i] {
                if let Some(v) = last_valid {
                    data[i] = v;
                    null_mask[i] = false;
                }
            } else {
                last_valid = Some(data[i]);
            }
        }
        let has_nulls = null_mask.iter().any(|&v| v);
        Series {
            name: self.name.clone(),
            data,
            null_mask: if has_nulls { Some(null_mask) } else { None },
        }
    }

    /// Backward-fill: replace each null with the next non-null value.
    /// Trailing nulls remain null.
    pub fn fill_backward(&self) -> Series<T> {
        let mut data = self.data.clone();
        let mut null_mask = self
            .null_mask
            .clone()
            .unwrap_or_else(|| vec![false; data.len()]);
        let mut next_valid: Option<T> = None;
        for i in (0..data.len()).rev() {
            if null_mask[i] {
                if let Some(v) = next_valid {
                    data[i] = v;
                    null_mask[i] = false;
                }
            } else {
                next_valid = Some(data[i]);
            }
        }
        let has_nulls = null_mask.iter().any(|&v| v);
        Series {
            name: self.name.clone(),
            data,
            null_mask: if has_nulls { Some(null_mask) } else { None },
        }
    }

    /// Drop all null values, returning a shorter series.
    pub fn drop_null(&self) -> Series<T> {
        if self.null_mask.is_none() {
            return self.clone();
        }
        let mask = self
            .null_mask
            .as_ref()
            .expect("null_mask present when has_nulls is true");
        let data: Vec<T> = self
            .data
            .iter()
            .enumerate()
            .filter(|(i, _)| !mask[*i])
            .map(|(_, &v)| v)
            .collect();
        Series::new(self.name.clone(), data)
    }

    /// Boolean mask: `true` where the value is null.
    pub fn is_null_mask(&self) -> Vec<bool> {
        self.null_mask
            .clone()
            .unwrap_or_else(|| vec![false; self.data.len()])
    }

    /// Boolean mask: `true` where the value is NOT null.
    pub fn is_not_null_mask(&self) -> Vec<bool> {
        self.is_null_mask().iter().map(|&v| !v).collect()
    }
}

// ---------------------------------------------------------------------------
// Series<T: Float> — interpolation
// ---------------------------------------------------------------------------

impl<T: Float> Series<T> {
    /// Linear interpolation between non-null values.
    /// Leading/trailing nulls remain null.
    pub fn interpolate(&self) -> Series<T> {
        if self.null_mask.is_none() {
            return self.clone();
        }
        let mask = self
            .null_mask
            .as_ref()
            .expect("null_mask present when has_nulls is true");
        let mut data = self.data.clone();
        let mut new_mask = mask.clone();
        let n = data.len();

        // Find non-null positions
        let non_null: Vec<usize> = (0..n).filter(|&i| !mask[i]).collect();
        if non_null.len() < 2 {
            return self.clone();
        }

        // Interpolate between consecutive non-null positions
        for w in non_null.windows(2) {
            let (start, end) = (w[0], w[1]);
            if end - start <= 1 {
                continue;
            }
            let v_start = data[start];
            let v_end = data[end];
            let span = T::from_usize(end - start);
            for i in (start + 1)..end {
                let frac = T::from_usize(i - start) / span;
                data[i] = v_start + (v_end - v_start) * frac;
                new_mask[i] = false;
            }
        }

        let has_nulls = new_mask.iter().any(|&v| v);
        Series {
            name: self.name.clone(),
            data,
            null_mask: if has_nulls { Some(new_mask) } else { None },
        }
    }
}

// ---------------------------------------------------------------------------
// StringSeries — missing data methods
// ---------------------------------------------------------------------------

impl StringSeries {
    /// Replace null values with a constant string.
    pub fn fill_null(&self, value: &str) -> StringSeries {
        let data: Vec<String> = self
            .data
            .iter()
            .enumerate()
            .map(|(i, v)| {
                if self.is_null_at(i) {
                    value.to_string()
                } else {
                    v.clone()
                }
            })
            .collect();
        StringSeries::new(self.name.clone(), data)
    }

    /// Forward-fill: replace each null with the last non-null value.
    pub fn fill_forward(&self) -> StringSeries {
        let mut data = self.data.clone();
        let mut null_mask = self
            .null_mask
            .clone()
            .unwrap_or_else(|| vec![false; data.len()]);
        let mut last_valid: Option<String> = None;
        for i in 0..data.len() {
            if null_mask[i] {
                if let Some(ref v) = last_valid {
                    data[i].clone_from(v);
                    null_mask[i] = false;
                }
            } else {
                last_valid = Some(data[i].clone());
            }
        }
        let has_nulls = null_mask.iter().any(|&v| v);
        StringSeries {
            name: self.name.clone(),
            data,
            null_mask: if has_nulls { Some(null_mask) } else { None },
        }
    }

    /// Backward-fill: replace each null with the next non-null value.
    pub fn fill_backward(&self) -> StringSeries {
        let mut data = self.data.clone();
        let mut null_mask = self
            .null_mask
            .clone()
            .unwrap_or_else(|| vec![false; data.len()]);
        let mut next_valid: Option<String> = None;
        for i in (0..data.len()).rev() {
            if null_mask[i] {
                if let Some(ref v) = next_valid {
                    data[i].clone_from(v);
                    null_mask[i] = false;
                }
            } else {
                next_valid = Some(data[i].clone());
            }
        }
        let has_nulls = null_mask.iter().any(|&v| v);
        StringSeries {
            name: self.name.clone(),
            data,
            null_mask: if has_nulls { Some(null_mask) } else { None },
        }
    }

    /// Drop all null values, returning a shorter series.
    pub fn drop_null(&self) -> StringSeries {
        if self.null_mask.is_none() {
            return self.clone();
        }
        let mask = self
            .null_mask
            .as_ref()
            .expect("null_mask present when has_nulls is true");
        let data: Vec<String> = self
            .data
            .iter()
            .enumerate()
            .filter(|(i, _)| !mask[*i])
            .map(|(_, v)| v.clone())
            .collect();
        StringSeries::new(self.name.clone(), data)
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn test_fill_null_constant() {
        let s = Series::with_nulls("x", vec![1.0_f64, 0.0, 3.0], vec![false, true, false]).unwrap();
        let filled = s.fill_null(99.0);
        assert_eq!(filled.get(1), Some(99.0));
        assert_eq!(filled.null_count(), 0);
    }

    #[test]
    fn test_fill_forward() {
        let s = Series::with_nulls(
            "x",
            vec![1.0_f64, 0.0, 0.0, 4.0],
            vec![false, true, true, false],
        )
        .unwrap();
        let filled = s.fill_forward();
        assert_eq!(filled.get(0), Some(1.0));
        assert_eq!(filled.get(1), Some(1.0));
        assert_eq!(filled.get(2), Some(1.0));
        assert_eq!(filled.get(3), Some(4.0));
    }

    #[test]
    fn test_fill_backward() {
        let s = Series::with_nulls(
            "x",
            vec![0.0_f64, 0.0, 3.0, 4.0],
            vec![true, true, false, false],
        )
        .unwrap();
        let filled = s.fill_backward();
        assert_eq!(filled.get(0), Some(3.0));
        assert_eq!(filled.get(1), Some(3.0));
    }

    #[test]
    fn test_fill_forward_leading_null_stays() {
        let s = Series::with_nulls("x", vec![0.0_f64, 2.0, 0.0], vec![true, false, true]).unwrap();
        let filled = s.fill_forward();
        assert!(filled.is_null_at(0));
        assert_eq!(filled.get(1), Some(2.0));
        assert_eq!(filled.get(2), Some(2.0));
    }

    #[test]
    fn test_drop_null_series() {
        let s =
            Series::with_nulls("x", vec![1_i32, 0, 3, 0], vec![false, true, false, true]).unwrap();
        let dropped = s.drop_null();
        assert_eq!(dropped.len(), 2);
        assert_eq!(dropped.as_slice(), &[1, 3]);
    }

    #[test]
    fn test_interpolate() {
        let s = Series::with_nulls(
            "x",
            vec![1.0_f64, 0.0, 0.0, 4.0],
            vec![false, true, true, false],
        )
        .unwrap();
        let interp = s.interpolate();
        assert_eq!(interp.get(0), Some(1.0));
        assert!((interp.get(1).unwrap() - 2.0).abs() < 1e-10);
        assert!((interp.get(2).unwrap() - 3.0).abs() < 1e-10);
        assert_eq!(interp.get(3), Some(4.0));
    }

    #[test]
    fn test_string_fill_null() {
        let s = StringSeries::with_nulls(
            "s",
            vec!["a".into(), String::new(), "c".into()],
            vec![false, true, false],
        )
        .unwrap();
        let filled = s.fill_null("X");
        assert_eq!(filled.get(1), Some("X"));
    }

    #[test]
    fn test_is_null_mask() {
        let s = Series::with_nulls("x", vec![1.0_f64, 0.0, 3.0], vec![false, true, false]).unwrap();
        assert_eq!(s.is_null_mask(), vec![false, true, false]);
        assert_eq!(s.is_not_null_mask(), vec![true, false, true]);
    }

    #[test]
    fn test_fill_null_no_nulls_is_noop() {
        let s = Series::new("x", vec![1.0_f64, 2.0, 3.0]);
        let filled = s.fill_null(99.0);
        assert_eq!(filled.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_fill_forward_all_nulls() {
        let s = Series::with_nulls("x", vec![0.0_f64, 0.0, 0.0], vec![true, true, true]).unwrap();
        let filled = s.fill_forward();
        // All leading nulls remain null
        assert!(filled.is_null_at(0));
        assert!(filled.is_null_at(1));
        assert!(filled.is_null_at(2));
    }

    #[test]
    fn test_fill_backward_all_nulls() {
        let s = Series::with_nulls("x", vec![0.0_f64, 0.0, 0.0], vec![true, true, true]).unwrap();
        let filled = s.fill_backward();
        // All trailing nulls remain null
        assert!(filled.is_null_at(0));
        assert!(filled.is_null_at(1));
        assert!(filled.is_null_at(2));
    }

    #[test]
    fn test_interpolate_no_nulls() {
        let s = Series::new("x", vec![1.0_f64, 2.0, 3.0]);
        let interp = s.interpolate();
        assert_eq!(interp.as_slice(), &[1.0, 2.0, 3.0]);
        assert_eq!(interp.null_count(), 0);
    }

    #[test]
    fn test_drop_null_no_nulls() {
        let s = Series::new("x", vec![1_i32, 2, 3]);
        let dropped = s.drop_null();
        assert_eq!(dropped.len(), 3);
        assert_eq!(dropped.as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn test_string_fill_forward() {
        let s = StringSeries::with_nulls(
            "s",
            vec!["a".into(), String::new(), String::new()],
            vec![false, true, true],
        )
        .unwrap();
        let filled = s.fill_forward();
        assert_eq!(filled.get(0), Some("a"));
        assert_eq!(filled.get(1), Some("a"));
        assert_eq!(filled.get(2), Some("a"));
    }

    #[test]
    fn test_string_fill_backward() {
        let s = StringSeries::with_nulls(
            "s",
            vec![String::new(), String::new(), "c".into()],
            vec![true, true, false],
        )
        .unwrap();
        let filled = s.fill_backward();
        assert_eq!(filled.get(0), Some("c"));
        assert_eq!(filled.get(1), Some("c"));
        assert_eq!(filled.get(2), Some("c"));
    }

    #[test]
    fn test_string_drop_null() {
        let s = StringSeries::with_nulls(
            "s",
            vec!["a".into(), String::new(), "c".into()],
            vec![false, true, false],
        )
        .unwrap();
        let dropped = s.drop_null();
        assert_eq!(dropped.len(), 2);
        assert_eq!(dropped.get(0), Some("a"));
        assert_eq!(dropped.get(1), Some("c"));
    }

    #[test]
    fn test_is_null_mask_no_nulls() {
        let s = Series::new("x", vec![1.0_f64, 2.0]);
        assert_eq!(s.is_null_mask(), vec![false, false]);
        assert_eq!(s.is_not_null_mask(), vec![true, true]);
    }

    #[test]
    fn test_interpolate_leading_trailing_nulls() {
        let s = Series::with_nulls(
            "x",
            vec![0.0_f64, 2.0, 0.0, 4.0, 0.0],
            vec![true, false, true, false, true],
        )
        .unwrap();
        let interp = s.interpolate();
        // Leading null stays null
        assert!(interp.is_null_at(0));
        assert_eq!(interp.get(1), Some(2.0));
        // Interpolated between 2.0 and 4.0
        assert!((interp.get(2).unwrap() - 3.0).abs() < 1e-10);
        assert_eq!(interp.get(3), Some(4.0));
        // Trailing null stays null
        assert!(interp.is_null_at(4));
    }
}
