//! Slicing and advanced indexing for [`Tensor`].

use crate::Scalar;
use crate::error::{CoreError, Result};

use super::{Tensor, compute_strides};

/// A range specification for one axis when slicing a tensor.
///
/// Mirrors Python's `start:stop:step` slice notation.
#[derive(Debug, Clone, Copy)]
pub struct SliceRange {
    pub start: usize,
    pub stop: usize,
    pub step: usize,
}

impl SliceRange {
    /// Create a new slice range.
    ///
    /// # Panics
    ///
    /// Panics in debug mode if `step == 0`.
    #[allow(clippy::similar_names)]
    pub fn new(start: usize, stop: usize, step: usize) -> Self {
        debug_assert!(step > 0, "slice step must be > 0");
        Self { start, stop, step }
    }

    /// Shorthand for `start..stop` with step 1.
    pub fn range(start: usize, stop: usize) -> Self {
        Self::new(start, stop, 1)
    }

    /// Select the full extent of an axis. Requires knowing the axis length.
    pub fn full(len: usize) -> Self {
        Self::new(0, len, 1)
    }

    /// The number of elements this range selects.
    fn len(&self) -> usize {
        if self.stop <= self.start {
            0
        } else {
            (self.stop - self.start).div_ceil(self.step)
        }
    }
}

impl<T: Scalar> Tensor<T> {
    /// Extract a sub-tensor by slicing along each axis.
    ///
    /// `ranges` must have exactly `ndim` elements. Each [`SliceRange`]
    /// specifies which indices to take along that axis.
    ///
    /// Returns a new tensor with copied data.
    pub fn slice(&self, ranges: &[SliceRange]) -> Result<Self> {
        if ranges.len() != self.ndim() {
            return Err(CoreError::InvalidArgument {
                reason: "number of slice ranges must match tensor rank",
            });
        }

        // Validate ranges
        for (d, r) in ranges.iter().enumerate() {
            if r.stop > self.shape[d] {
                return Err(CoreError::IndexOutOfBounds {
                    index: vec![r.stop],
                    shape: self.shape.clone(),
                });
            }
            if r.step == 0 {
                return Err(CoreError::InvalidArgument {
                    reason: "slice step must be > 0",
                });
            }
        }

        let new_shape: Vec<usize> = ranges.iter().map(SliceRange::len).collect();
        let new_numel: usize = new_shape.iter().product();

        if new_numel == 0 {
            return Tensor::from_vec(vec![], new_shape);
        }

        let mut data = Vec::with_capacity(new_numel);
        let mut index = vec![0usize; self.ndim()];

        // Initialize index to start positions
        for (d, r) in ranges.iter().enumerate() {
            index[d] = r.start;
        }

        // Iterate over all output elements (odometer on the sliced indices)
        for _ in 0..new_numel {
            let flat = index
                .iter()
                .zip(self.strides.iter())
                .map(|(&i, &s)| i * s)
                .sum::<usize>();
            data.push(self.data[flat]);

            // Advance the odometer
            for d in (0..self.ndim()).rev() {
                index[d] += ranges[d].step;
                if index[d] < ranges[d].stop {
                    break;
                }
                index[d] = ranges[d].start;
            }
        }

        Tensor::from_vec(data, new_shape)
    }

    /// Select a single index along the given axis, reducing dimensionality by 1.
    ///
    /// For a 2-D tensor, `select(0, i)` returns row `i` as a 1-D tensor.
    pub fn select(&self, axis: usize, index: usize) -> Result<Self> {
        if axis >= self.ndim() {
            return Err(CoreError::AxisOutOfBounds {
                axis,
                ndim: self.ndim(),
            });
        }
        if index >= self.shape[axis] {
            return Err(CoreError::IndexOutOfBounds {
                index: vec![index],
                shape: self.shape.clone(),
            });
        }

        let mut ranges: Vec<SliceRange> = self
            .shape
            .iter()
            .map(|&len| SliceRange::full(len))
            .collect();
        ranges[axis] = SliceRange::new(index, index + 1, 1);

        let sliced = self.slice(&ranges)?;
        // Remove the axis that was selected (it has size 1)
        let mut new_shape: Vec<usize> = sliced.shape().to_vec();
        new_shape.remove(axis);
        if new_shape.is_empty() {
            Ok(Tensor::scalar(sliced.data[0]))
        } else {
            let strides = compute_strides(&new_shape);
            Ok(Tensor {
                data: sliced.data,
                shape: new_shape,
                strides,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slice_basic() {
        // [[1, 2, 3],
        //  [4, 5, 6],
        //  [7, 8, 9]]
        let t = Tensor::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9], vec![3, 3]).unwrap();

        // Slice rows 0..2, cols 1..3 -> [[2, 3], [5, 6]]
        let s = t
            .slice(&[SliceRange::range(0, 2), SliceRange::range(1, 3)])
            .unwrap();
        assert_eq!(s.shape(), &[2, 2]);
        assert_eq!(s.as_slice(), &[2, 3, 5, 6]);
    }

    #[test]
    fn test_slice_with_step() {
        let t = Tensor::<i32>::arange(10);
        // [0, 1, 2, ..., 9] with step 3 -> [0, 3, 6, 9]
        let s = t.slice(&[SliceRange::new(0, 10, 3)]).unwrap();
        assert_eq!(s.shape(), &[4]);
        assert_eq!(s.as_slice(), &[0, 3, 6, 9]);
    }

    #[test]
    fn test_slice_full() {
        let t = Tensor::from_vec(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
        let s = t
            .slice(&[SliceRange::full(2), SliceRange::full(2)])
            .unwrap();
        assert_eq!(s, t);
    }

    #[test]
    fn test_select_row() {
        let t = Tensor::from_vec(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
        let row = t.select(0, 1).unwrap();
        assert_eq!(row.shape(), &[3]);
        assert_eq!(row.as_slice(), &[4, 5, 6]);
    }

    #[test]
    fn test_select_col() {
        let t = Tensor::from_vec(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
        let col = t.select(1, 0).unwrap();
        assert_eq!(col.shape(), &[2]);
        assert_eq!(col.as_slice(), &[1, 4]);
    }

    #[test]
    fn test_select_to_scalar() {
        let t = Tensor::from_vec(vec![42], vec![1]).unwrap();
        let s = t.select(0, 0).unwrap();
        assert_eq!(s.ndim(), 0);
        assert_eq!(s.as_slice(), &[42]);
    }

    #[test]
    fn test_slice_out_of_bounds() {
        let t = Tensor::from_vec(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
        assert!(
            t.slice(&[SliceRange::range(0, 3), SliceRange::full(2)])
                .is_err()
        );
    }

    #[test]
    fn test_select_axis_oob() {
        let t = Tensor::from_vec(vec![1, 2, 3], vec![3]).unwrap();
        assert!(t.select(1, 0).is_err());
    }

    #[test]
    fn test_select_index_oob() {
        let t = Tensor::from_vec(vec![1, 2, 3], vec![3]).unwrap();
        assert!(t.select(0, 5).is_err());
    }
}
