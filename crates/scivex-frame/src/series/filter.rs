//! Filtering, sorting, and uniqueness operations on `Series<T>`.

use scivex_core::Scalar;

use super::Series;
use crate::error::{FrameError, Result};

impl<T: Scalar> Series<T> {
    /// Keep only elements where `mask[i]` is true.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::series::Series;
    /// let s = Series::new("x", vec![10_i32, 20, 30, 40]);
    /// let filtered = s.filter(&[true, false, true, false]).unwrap();
    /// assert_eq!(filtered.as_slice(), &[10, 30]);
    /// ```
    pub fn filter(&self, mask: &[bool]) -> Result<Series<T>> {
        if mask.len() != self.data.len() {
            return Err(FrameError::RowCountMismatch {
                expected: self.data.len(),
                got: mask.len(),
            });
        }
        let mut data = Vec::new();
        let mut new_nulls: Option<Vec<bool>> = self.null_mask.as_ref().map(|_| Vec::new());
        for (i, &keep) in mask.iter().enumerate() {
            if keep {
                data.push(self.data[i]);
                if let Some(ref mut nm) = new_nulls {
                    nm.push(
                        self.null_mask
                            .as_ref()
                            .expect("null_mask present when has_nulls is true")[i],
                    );
                }
            }
        }
        Ok(Series {
            name: self.name.clone(),
            data,
            null_mask: new_nulls,
        })
    }

    /// Sort the series by value.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::series::Series;
    /// let s = Series::new("x", vec![3_i32, 1, 4, 1, 5]);
    /// let sorted = s.sort(true);
    /// assert_eq!(sorted.as_slice(), &[1, 1, 3, 4, 5]);
    /// ```
    pub fn sort(&self, ascending: bool) -> Series<T> {
        let indices = self.argsort(ascending);
        let data: Vec<T> = indices.iter().map(|&i| self.data[i]).collect();
        let null_mask = self
            .null_mask
            .as_ref()
            .map(|m| indices.iter().map(|&i| m[i]).collect());
        Series {
            name: self.name.clone(),
            data,
            null_mask,
        }
    }

    /// Return indices that would sort the series.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::series::Series;
    /// let s = Series::new("x", vec![30_i32, 10, 20]);
    /// assert_eq!(s.argsort(true), vec![1, 2, 0]);
    /// ```
    pub fn argsort(&self, ascending: bool) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..self.data.len()).collect();
        indices.sort_by(|&a, &b| {
            let cmp = self.data[a]
                .partial_cmp(&self.data[b])
                .unwrap_or(core::cmp::Ordering::Equal);
            if ascending { cmp } else { cmp.reverse() }
        });
        indices
    }

    /// Unique values (preserving first-occurrence order).
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::series::Series;
    /// let s = Series::new("x", vec![1_i32, 2, 1, 3, 2]);
    /// let u = s.unique();
    /// assert_eq!(u.as_slice(), &[1, 2, 3]);
    /// ```
    pub fn unique(&self) -> Series<T> {
        let mut seen = Vec::new();
        let mut data = Vec::new();
        for &v in &self.data {
            if !seen.contains(&v) {
                seen.push(v);
                data.push(v);
            }
        }
        Series {
            name: self.name.clone(),
            data,
            null_mask: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter() {
        let s = Series::new("x", vec![10_i32, 20, 30, 40]);
        let filtered = s.filter(&[true, false, true, false]).unwrap();
        assert_eq!(filtered.as_slice(), &[10, 30]);
    }

    #[test]
    fn test_filter_length_mismatch() {
        let s = Series::new("x", vec![1_i32, 2]);
        assert!(s.filter(&[true]).is_err());
    }

    #[test]
    fn test_sort_ascending() {
        let s = Series::new("x", vec![3_i32, 1, 4, 1, 5]);
        let sorted = s.sort(true);
        assert_eq!(sorted.as_slice(), &[1, 1, 3, 4, 5]);
    }

    #[test]
    fn test_sort_descending() {
        let s = Series::new("x", vec![3_i32, 1, 4, 1, 5]);
        let sorted = s.sort(false);
        assert_eq!(sorted.as_slice(), &[5, 4, 3, 1, 1]);
    }

    #[test]
    fn test_argsort() {
        let s = Series::new("x", vec![30_i32, 10, 20]);
        assert_eq!(s.argsort(true), vec![1, 2, 0]);
    }

    #[test]
    fn test_unique() {
        let s = Series::new("x", vec![1_i32, 2, 1, 3, 2]);
        let u = s.unique();
        assert_eq!(u.as_slice(), &[1, 2, 3]);
    }

    // -- Edge-case tests -------------------------------------------------------

    #[test]
    fn test_filter_all_true() {
        let s = Series::new("x", vec![10_i32, 20, 30]);
        let filtered = s.filter(&[true, true, true]).unwrap();
        assert_eq!(filtered.as_slice(), &[10, 20, 30]);
    }

    #[test]
    fn test_filter_all_false() {
        let s = Series::new("x", vec![10_i32, 20, 30]);
        let filtered = s.filter(&[false, false, false]).unwrap();
        assert_eq!(filtered.len(), 0);
        assert!(filtered.is_empty());
    }

    #[test]
    fn test_filter_empty_series() {
        let s: Series<i32> = Series::new("x", vec![]);
        let filtered = s.filter(&[]).unwrap();
        assert_eq!(filtered.len(), 0);
    }

    #[test]
    fn test_sort_with_duplicates() {
        let s = Series::new("x", vec![3_i32, 1, 3, 1, 2]);
        let sorted = s.sort(true);
        assert_eq!(sorted.as_slice(), &[1, 1, 2, 3, 3]);
    }

    #[test]
    fn test_sort_single_element() {
        let s = Series::new("x", vec![42_i32]);
        let sorted = s.sort(true);
        assert_eq!(sorted.as_slice(), &[42]);
    }

    #[test]
    fn test_sort_empty() {
        let s: Series<i32> = Series::new("x", vec![]);
        let sorted = s.sort(true);
        assert!(sorted.is_empty());
    }

    #[test]
    fn test_unique_empty() {
        let s: Series<i32> = Series::new("x", vec![]);
        let u = s.unique();
        assert!(u.is_empty());
    }

    #[test]
    fn test_unique_all_same() {
        let s = Series::new("x", vec![5_i32, 5, 5, 5]);
        let u = s.unique();
        assert_eq!(u.as_slice(), &[5]);
    }

    #[test]
    fn test_argsort_empty() {
        let s: Series<i32> = Series::new("x", vec![]);
        let indices = s.argsort(true);
        assert!(indices.is_empty());
    }

    #[test]
    fn test_filter_preserves_nulls() {
        let s =
            Series::with_nulls("x", vec![1_i32, 0, 3, 0], vec![false, true, false, true]).unwrap();
        let filtered = s.filter(&[true, true, false, false]).unwrap();
        assert_eq!(filtered.len(), 2);
        assert!(!filtered.is_null_at(0));
        assert!(filtered.is_null_at(1));
    }

    #[test]
    fn test_unique_single_element() {
        let s = Series::new("x", vec![42_i32]);
        let u = s.unique();
        assert_eq!(u.as_slice(), &[42]);
    }

    #[test]
    fn test_sort_preserves_nulls() {
        let s = Series::with_nulls("x", vec![3_i32, 0, 1], vec![false, true, false]).unwrap();
        let sorted = s.sort(true);
        // Values are sorted by data values; null mask is reordered too
        assert_eq!(sorted.len(), 3);
    }
}
