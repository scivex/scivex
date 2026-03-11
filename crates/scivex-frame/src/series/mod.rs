//! Typed and type-erased columnar series.

pub mod categorical;
pub mod datetime;
mod filter;
pub mod missing;
mod ops;
pub mod string;
pub mod window;

use std::any::Any;
use std::fmt;

use scivex_core::Scalar;

use crate::dtype::{DType, HasDType};
use crate::error::{FrameError, Result};

// ---------------------------------------------------------------------------
// AnySeries — type-erased column interface
// ---------------------------------------------------------------------------

/// A type-erased column that can be stored in a [`DataFrame`](crate::DataFrame).
pub trait AnySeries: Send + Sync + fmt::Debug {
    /// Column name.
    fn name(&self) -> &str;

    /// Runtime element type.
    fn dtype(&self) -> DType;

    /// Number of rows (including nulls).
    fn len(&self) -> usize;

    /// Whether the series is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of null entries.
    fn null_count(&self) -> usize;

    /// Whether the element at `index` is null.
    fn is_null(&self, index: usize) -> bool;

    /// Downcast to a concrete type.
    fn as_any(&self) -> &dyn Any;

    /// Clone into a boxed trait object.
    fn clone_box(&self) -> Box<dyn AnySeries>;

    /// Format the value at `index` for display.
    fn display_value(&self, index: usize) -> String;

    /// Return a new series keeping only rows where `mask[i]` is true.
    fn filter_mask(&self, mask: &[bool]) -> Box<dyn AnySeries>;

    /// Return a new series with rows at the given indices.
    fn take_indices(&self, indices: &[usize]) -> Box<dyn AnySeries>;

    /// Return a contiguous sub-series.
    fn slice(&self, offset: usize, length: usize) -> Box<dyn AnySeries>;

    /// Rename the column (returns a clone with the new name).
    fn rename_box(&self, name: &str) -> Box<dyn AnySeries>;

    /// Return a new series with null rows removed.
    fn drop_nulls(&self) -> Box<dyn AnySeries>;

    /// Return a boolean vector where `true` means the value is null.
    fn null_mask_vec(&self) -> Vec<bool>;

    /// Create a series of this dtype filled with nulls.
    fn null_series(&self, name: &str, len: usize) -> Box<dyn AnySeries>;

    /// Build a new series by picking values from optional row indices.
    /// `None` entries produce null values.
    fn take_optional(&self, indices: &[Option<usize>]) -> Box<dyn AnySeries>;
}

impl Clone for Box<dyn AnySeries> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

// ---------------------------------------------------------------------------
// Series<T> — generic typed column
// ---------------------------------------------------------------------------

/// A named, typed columnar array with optional null tracking.
#[derive(Debug, Clone)]
pub struct Series<T: Scalar> {
    pub(crate) name: String,
    pub(crate) data: Vec<T>,
    pub(crate) null_mask: Option<Vec<bool>>, // true = null
}

impl<T: Scalar> Series<T> {
    /// Create a new series from a `Vec<T>`.
    pub fn new(name: impl Into<String>, data: Vec<T>) -> Self {
        Self {
            name: name.into(),
            data,
            null_mask: None,
        }
    }

    /// Create a new series from a slice.
    pub fn from_slice(name: impl Into<String>, data: &[T]) -> Self {
        Self::new(name, data.to_vec())
    }

    /// Create a series with explicit null positions.
    ///
    /// `null_mask[i] == true` means the i-th element is null.
    pub fn with_nulls(name: impl Into<String>, data: Vec<T>, null_mask: Vec<bool>) -> Result<Self> {
        if data.len() != null_mask.len() {
            return Err(FrameError::RowCountMismatch {
                expected: data.len(),
                got: null_mask.len(),
            });
        }
        Ok(Self {
            name: name.into(),
            data,
            null_mask: Some(null_mask),
        })
    }

    // -- Accessors ----------------------------------------------------------

    /// Column name.
    #[inline]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Number of elements (including nulls).
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the series is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Underlying data as a slice.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Get the element at `index`, or `None` if out of bounds or null.
    pub fn get(&self, index: usize) -> Option<T> {
        if index >= self.data.len() {
            return None;
        }
        if self.is_null_at(index) {
            return None;
        }
        Some(self.data[index])
    }

    /// Whether the element at `index` is null.
    #[inline]
    pub fn is_null_at(&self, index: usize) -> bool {
        self.null_mask
            .as_ref()
            .is_some_and(|m| m.get(index).copied().unwrap_or(false))
    }

    /// Number of null entries.
    pub fn null_count(&self) -> usize {
        self.null_mask
            .as_ref()
            .map_or(0, |m| m.iter().filter(|&&v| v).count())
    }

    /// Number of non-null entries.
    pub fn count(&self) -> usize {
        self.len() - self.null_count()
    }

    // -- Mutators -----------------------------------------------------------

    /// Rename this series in place.
    pub fn rename(&mut self, name: impl Into<String>) {
        self.name = name.into();
    }

    /// Set the element at `index`. Clears the null flag for that position.
    pub fn set(&mut self, index: usize, value: T) -> Result<()> {
        if index >= self.data.len() {
            return Err(FrameError::IndexOutOfBounds {
                index,
                length: self.data.len(),
            });
        }
        self.data[index] = value;
        if let Some(ref mut mask) = self.null_mask {
            mask[index] = false;
        }
        Ok(())
    }

    /// Append a value.
    pub fn push(&mut self, value: T) {
        self.data.push(value);
        if let Some(ref mut mask) = self.null_mask {
            mask.push(false);
        }
    }

    /// Append a null.
    pub fn push_null(&mut self) {
        self.data.push(T::zero());
        if let Some(ref mut mask) = self.null_mask {
            mask.push(true);
        } else {
            let mut mask = vec![false; self.data.len() - 1];
            mask.push(true);
            self.null_mask = Some(mask);
        }
    }
}

// ---------------------------------------------------------------------------
// HasDType-bound methods (need the DType mapping)
// ---------------------------------------------------------------------------

impl<T: Scalar + HasDType> Series<T> {
    /// Runtime element type.
    pub fn dtype(&self) -> DType {
        T::dtype()
    }
}

// ---------------------------------------------------------------------------
// AnySeries implementation for Series<T>
// ---------------------------------------------------------------------------

impl<T: Scalar + HasDType + 'static> AnySeries for Series<T> {
    fn name(&self) -> &str {
        &self.name
    }

    fn dtype(&self) -> DType {
        T::dtype()
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    fn null_count(&self) -> usize {
        self.null_count()
    }

    fn is_null(&self, index: usize) -> bool {
        self.is_null_at(index)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn AnySeries> {
        Box::new(self.clone())
    }

    fn display_value(&self, index: usize) -> String {
        if self.is_null_at(index) {
            "null".to_string()
        } else if index < self.data.len() {
            format!("{}", self.data[index])
        } else {
            String::new()
        }
    }

    fn filter_mask(&self, mask: &[bool]) -> Box<dyn AnySeries> {
        let mut data = Vec::new();
        let mut new_nulls: Option<Vec<bool>> = self.null_mask.as_ref().map(|_| Vec::new());
        for (i, &keep) in mask.iter().enumerate() {
            if keep && i < self.data.len() {
                data.push(self.data[i]);
                if let Some(ref mut nm) = new_nulls {
                    nm.push(self.null_mask.as_ref().unwrap()[i]);
                }
            }
        }
        Box::new(Self {
            name: self.name.clone(),
            data,
            null_mask: new_nulls,
        })
    }

    fn take_indices(&self, indices: &[usize]) -> Box<dyn AnySeries> {
        let data: Vec<T> = indices.iter().map(|&i| self.data[i]).collect();
        let null_mask = self
            .null_mask
            .as_ref()
            .map(|m| indices.iter().map(|&i| m[i]).collect());
        Box::new(Self {
            name: self.name.clone(),
            data,
            null_mask,
        })
    }

    fn slice(&self, offset: usize, length: usize) -> Box<dyn AnySeries> {
        let end = (offset + length).min(self.data.len());
        let data = self.data[offset..end].to_vec();
        let null_mask = self.null_mask.as_ref().map(|m| m[offset..end].to_vec());
        Box::new(Self {
            name: self.name.clone(),
            data,
            null_mask,
        })
    }

    fn rename_box(&self, name: &str) -> Box<dyn AnySeries> {
        let mut cloned = self.clone();
        cloned.name = name.to_string();
        Box::new(cloned)
    }

    fn drop_nulls(&self) -> Box<dyn AnySeries> {
        if self.null_mask.is_none() {
            return self.clone_box();
        }
        let mask = self.null_mask.as_ref().unwrap();
        let keep: Vec<bool> = mask.iter().map(|&is_null| !is_null).collect();
        self.filter_mask(&keep)
    }

    fn null_mask_vec(&self) -> Vec<bool> {
        self.null_mask
            .clone()
            .unwrap_or_else(|| vec![false; self.data.len()])
    }

    fn null_series(&self, name: &str, len: usize) -> Box<dyn AnySeries> {
        Box::new(Self {
            name: name.to_string(),
            data: vec![T::zero(); len],
            null_mask: Some(vec![true; len]),
        })
    }

    fn take_optional(&self, indices: &[Option<usize>]) -> Box<dyn AnySeries> {
        let mut data = Vec::with_capacity(indices.len());
        let mut nulls = Vec::with_capacity(indices.len());
        for opt in indices {
            if let Some(i) = opt {
                data.push(self.data[*i]);
                nulls.push(self.is_null_at(*i));
            } else {
                data.push(T::zero());
                nulls.push(true);
            }
        }
        let has_nulls = nulls.iter().any(|&v| v);
        Box::new(Self {
            name: self.name.clone(),
            data,
            null_mask: if has_nulls { Some(nulls) } else { None },
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_series_new() {
        let s = Series::new("x", vec![1.0_f64, 2.0, 3.0]);
        assert_eq!(s.name(), "x");
        assert_eq!(s.len(), 3);
        assert!(!s.is_empty());
        assert_eq!(s.get(0), Some(1.0));
        assert_eq!(s.get(3), None);
    }

    #[test]
    fn test_series_from_slice() {
        let s = Series::from_slice("y", &[10_i32, 20, 30]);
        assert_eq!(s.as_slice(), &[10, 20, 30]);
    }

    #[test]
    fn test_series_with_nulls() {
        let s = Series::with_nulls("z", vec![1.0_f64, 0.0, 3.0], vec![false, true, false]).unwrap();
        assert_eq!(s.null_count(), 1);
        assert!(!s.is_null_at(0));
        assert!(s.is_null_at(1));
        assert_eq!(s.get(1), None);
        assert_eq!(s.count(), 2);
    }

    #[test]
    fn test_series_with_nulls_length_mismatch() {
        let result = Series::with_nulls("z", vec![1.0_f64, 2.0], vec![false]);
        assert!(result.is_err());
    }

    #[test]
    fn test_series_mutators() {
        let mut s = Series::new("m", vec![1_i64, 2, 3]);
        s.set(1, 20).unwrap();
        assert_eq!(s.get(1), Some(20));
        s.push(4);
        assert_eq!(s.len(), 4);
        s.push_null();
        assert_eq!(s.len(), 5);
        assert!(s.is_null_at(4));
    }

    #[test]
    fn test_series_dtype() {
        let s = Series::new("a", vec![1.0_f64]);
        assert_eq!(s.dtype(), DType::F64);
        let s = Series::new("b", vec![1_i32]);
        assert_eq!(s.dtype(), DType::I32);
    }

    #[test]
    fn test_any_series_downcast() {
        let s = Series::new("x", vec![1.0_f64, 2.0, 3.0]);
        let boxed: Box<dyn AnySeries> = Box::new(s);
        assert_eq!(boxed.dtype(), DType::F64);
        let downcasted = boxed.as_any().downcast_ref::<Series<f64>>().unwrap();
        assert_eq!(downcasted.get(0), Some(1.0));
    }

    #[test]
    fn test_any_series_display_value() {
        let s = Series::with_nulls("x", vec![1.0_f64, 0.0], vec![false, true]).unwrap();
        let boxed: Box<dyn AnySeries> = Box::new(s);
        assert_eq!(boxed.display_value(0), "1");
        assert_eq!(boxed.display_value(1), "null");
    }

    #[test]
    fn test_any_series_clone_box() {
        let s = Series::new("x", vec![1_i32, 2, 3]);
        let boxed: Box<dyn AnySeries> = Box::new(s);
        let cloned = boxed.clone_box();
        assert_eq!(cloned.name(), "x");
        assert_eq!(cloned.len(), 3);
    }

    #[test]
    fn test_any_series_filter_mask() {
        let s = Series::new("x", vec![10_i32, 20, 30]);
        let boxed: Box<dyn AnySeries> = Box::new(s);
        let filtered = boxed.filter_mask(&[true, false, true]);
        assert_eq!(filtered.len(), 2);
        let downcasted = filtered.as_any().downcast_ref::<Series<i32>>().unwrap();
        assert_eq!(downcasted.as_slice(), &[10, 30]);
    }

    #[test]
    fn test_any_series_take_indices() {
        let s = Series::new("x", vec![10_i32, 20, 30, 40]);
        let boxed: Box<dyn AnySeries> = Box::new(s);
        let taken = boxed.take_indices(&[3, 0, 2]);
        let downcasted = taken.as_any().downcast_ref::<Series<i32>>().unwrap();
        assert_eq!(downcasted.as_slice(), &[40, 10, 30]);
    }

    #[test]
    fn test_any_series_slice() {
        let s = Series::new("x", vec![10_i32, 20, 30, 40, 50]);
        let boxed: Box<dyn AnySeries> = Box::new(s);
        let sliced = boxed.slice(1, 3);
        let downcasted = sliced.as_any().downcast_ref::<Series<i32>>().unwrap();
        assert_eq!(downcasted.as_slice(), &[20, 30, 40]);
    }
}
