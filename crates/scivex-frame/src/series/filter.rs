//! Filtering, sorting, and uniqueness operations on `Series<T>`.

use scivex_core::Scalar;

use super::Series;
use crate::error::{FrameError, Result};

impl<T: Scalar> Series<T> {
    /// Keep only elements where `mask[i]` is true.
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
                    nm.push(self.null_mask.as_ref().unwrap()[i]);
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
}
