//! Slicing and advanced indexing for [`Tensor`].

use crate::Scalar;
use crate::error::{CoreError, Result};

use super::{Tensor, compute_strides};

/// A range specification for one axis when slicing a tensor.
///
/// Mirrors Python's `start:stop:step` slice notation.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
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

// ======================================================================
// Fancy indexing: integer array indexing, boolean mask indexing, gather/scatter
// ======================================================================

impl<T: Scalar> Tensor<T> {
    /// Select elements along an axis using an array of indices.
    ///
    /// Like numpy `np.take(arr, indices, axis)`. For a tensor of shape
    /// `[d0, d1, ..., dk, ...]`, selecting along axis `k` with `indices` of
    /// length `m` produces a tensor of shape `[d0, ..., m, ..., dn]`.
    pub fn index_select(&self, axis: usize, indices: &[usize]) -> Result<Self> {
        if axis >= self.ndim() {
            return Err(CoreError::AxisOutOfBounds {
                axis,
                ndim: self.ndim(),
            });
        }
        for &idx in indices {
            if idx >= self.shape[axis] {
                return Err(CoreError::IndexOutOfBounds {
                    index: vec![idx],
                    shape: self.shape.clone(),
                });
            }
        }

        let mut new_shape = self.shape.clone();
        new_shape[axis] = indices.len();
        let new_numel: usize = new_shape.iter().product();

        if new_numel == 0 {
            return Tensor::from_vec(vec![], new_shape);
        }

        let mut data = Vec::with_capacity(new_numel);

        // Iterate over every output position
        let ndim = self.ndim();
        let mut out_idx = vec![0usize; ndim];

        for _ in 0..new_numel {
            // Map output index to source index
            let mut src_flat = 0;
            for d in 0..ndim {
                let src_coord = if d == axis {
                    indices[out_idx[d]]
                } else {
                    out_idx[d]
                };
                src_flat += src_coord * self.strides[d];
            }
            data.push(self.data[src_flat]);

            // Advance odometer
            for d in (0..ndim).rev() {
                out_idx[d] += 1;
                if out_idx[d] < new_shape[d] {
                    break;
                }
                out_idx[d] = 0;
            }
        }

        Tensor::from_vec(data, new_shape)
    }

    /// Select elements where `mask` is true, returning a flat 1-D tensor.
    ///
    /// Like numpy `arr[mask]`. The mask length must equal the total number of
    /// elements in the tensor.
    pub fn masked_select(&self, mask: &[bool]) -> Result<Self> {
        if mask.len() != self.numel() {
            return Err(CoreError::InvalidArgument {
                reason: "mask length must equal tensor element count",
            });
        }

        let data: Vec<T> = self
            .data
            .iter()
            .zip(mask.iter())
            .filter(|&(_, &m)| m)
            .map(|(&v, _)| v)
            .collect();

        let len = data.len();
        Tensor::from_vec(data, vec![len])
    }

    /// Select rows (slices along axis 0) where `mask` is true.
    ///
    /// `mask.len()` must equal `self.shape()[0]`. The result has the same
    /// number of dimensions, with dimension 0 reduced to the count of `true`
    /// entries.
    pub fn masked_select_along(&self, mask: &[bool]) -> Result<Self> {
        if self.ndim() == 0 {
            return Err(CoreError::InvalidArgument {
                reason: "cannot mask-select along axis 0 of a scalar tensor",
            });
        }
        if mask.len() != self.shape[0] {
            return Err(CoreError::InvalidArgument {
                reason: "mask length must equal shape[0]",
            });
        }

        let row_size: usize = self.strides[0]; // number of elements per row
        let selected: usize = mask.iter().filter(|&&m| m).count();

        let mut new_shape = self.shape.clone();
        new_shape[0] = selected;
        let new_numel: usize = new_shape.iter().product();

        let mut data = Vec::with_capacity(new_numel);
        for (i, &m) in mask.iter().enumerate() {
            if m {
                let start = i * row_size;
                let end = start + row_size;
                data.extend_from_slice(&self.data[start..end]);
            }
        }

        Tensor::from_vec(data, new_shape)
    }

    /// Set elements along an axis at the given indices.
    ///
    /// `values` must have the same shape as the result of
    /// `self.index_select(axis, indices)`.
    pub fn index_put(&mut self, axis: usize, indices: &[usize], values: &Tensor<T>) -> Result<()> {
        if axis >= self.ndim() {
            return Err(CoreError::AxisOutOfBounds {
                axis,
                ndim: self.ndim(),
            });
        }
        // Validate expected shape
        let mut expected_shape = self.shape.clone();
        expected_shape[axis] = indices.len();
        if values.shape() != expected_shape.as_slice() {
            return Err(CoreError::DimensionMismatch {
                expected: expected_shape,
                got: values.shape().to_vec(),
            });
        }
        for &idx in indices {
            if idx >= self.shape[axis] {
                return Err(CoreError::IndexOutOfBounds {
                    index: vec![idx],
                    shape: self.shape.clone(),
                });
            }
        }

        let ndim = self.ndim();
        let val_numel = values.numel();

        if val_numel == 0 {
            return Ok(());
        }

        let mut out_idx = vec![0usize; ndim];

        for vi in 0..val_numel {
            // Map output index to source index in self
            let mut dst_flat = 0;
            for d in 0..ndim {
                let dst_coord = if d == axis {
                    indices[out_idx[d]]
                } else {
                    out_idx[d]
                };
                dst_flat += dst_coord * self.strides[d];
            }
            self.data[dst_flat] = values.data[vi];

            // Advance odometer using expected_shape
            for d in (0..ndim).rev() {
                out_idx[d] += 1;
                if out_idx[d] < expected_shape[d] {
                    break;
                }
                out_idx[d] = 0;
            }
        }

        Ok(())
    }

    /// Set elements where `mask` is true to corresponding values from `values`.
    ///
    /// `mask.len()` must equal `self.numel()`, and `values.len()` must equal
    /// the number of `true` entries in the mask.
    pub fn masked_put(&mut self, mask: &[bool], values: &[T]) -> Result<()> {
        if mask.len() != self.numel() {
            return Err(CoreError::InvalidArgument {
                reason: "mask length must equal tensor element count",
            });
        }
        let true_count = mask.iter().filter(|&&m| m).count();
        if values.len() != true_count {
            return Err(CoreError::InvalidArgument {
                reason: "values length must equal number of true entries in mask",
            });
        }

        let mut vi = 0;
        for (i, &m) in mask.iter().enumerate() {
            if m {
                self.data[i] = values[vi];
                vi += 1;
            }
        }

        Ok(())
    }

    /// Gather values along `axis` using an index tensor.
    ///
    /// The `indices` tensor must have the same number of dimensions as `self`,
    /// and all dimensions except `axis` must match `self`'s shape. The output
    /// has the same shape as `indices`.
    ///
    /// This is equivalent to PyTorch's `torch.gather`.
    pub fn gather(&self, axis: usize, indices: &Tensor<usize>) -> Result<Tensor<T>> {
        if axis >= self.ndim() {
            return Err(CoreError::AxisOutOfBounds {
                axis,
                ndim: self.ndim(),
            });
        }
        if indices.ndim() != self.ndim() {
            return Err(CoreError::DimensionMismatch {
                expected: self.shape.clone(),
                got: indices.shape().to_vec(),
            });
        }
        // All dims except axis must match
        for d in 0..self.ndim() {
            if d != axis && indices.shape()[d] != self.shape[d] {
                return Err(CoreError::DimensionMismatch {
                    expected: self.shape.clone(),
                    got: indices.shape().to_vec(),
                });
            }
        }

        let out_shape = indices.shape().to_vec();
        let out_numel: usize = out_shape.iter().product();

        if out_numel == 0 {
            return Tensor::from_vec(vec![], out_shape);
        }

        let ndim = self.ndim();
        let mut data = Vec::with_capacity(out_numel);
        let mut out_idx = vec![0usize; ndim];
        let idx_strides = compute_strides(&out_shape);

        for _ in 0..out_numel {
            // Read the index value from the indices tensor
            let idx_flat: usize = out_idx
                .iter()
                .zip(idx_strides.iter())
                .map(|(&i, &s)| i * s)
                .sum();
            let gather_idx = indices.data[idx_flat];

            if gather_idx >= self.shape[axis] {
                return Err(CoreError::IndexOutOfBounds {
                    index: vec![gather_idx],
                    shape: self.shape.clone(),
                });
            }

            // Compute flat index in self
            let src_flat: usize = out_idx
                .iter()
                .enumerate()
                .zip(self.strides.iter())
                .map(|((d, &oi), &s)| {
                    let coord = if d == axis { gather_idx } else { oi };
                    coord * s
                })
                .sum();
            data.push(self.data[src_flat]);

            // Advance odometer
            for d in (0..ndim).rev() {
                out_idx[d] += 1;
                if out_idx[d] < out_shape[d] {
                    break;
                }
                out_idx[d] = 0;
            }
        }

        Tensor::from_vec(data, out_shape)
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

    // ------------------------------------------------------------------
    // Fancy indexing tests
    // ------------------------------------------------------------------

    #[test]
    fn test_index_select_1d() {
        let t = Tensor::from_vec(vec![10, 20, 30, 40, 50], vec![5]).unwrap();
        let s = t.index_select(0, &[4, 0, 2]).unwrap();
        assert_eq!(s.shape(), &[3]);
        assert_eq!(s.as_slice(), &[50, 10, 30]);
    }

    #[test]
    fn test_index_select_2d_axis0() {
        // [[1, 2, 3],
        //  [4, 5, 6],
        //  [7, 8, 9]]
        let t = Tensor::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9], vec![3, 3]).unwrap();
        // Select rows 2, 0 -> [[7, 8, 9], [1, 2, 3]]
        let s = t.index_select(0, &[2, 0]).unwrap();
        assert_eq!(s.shape(), &[2, 3]);
        assert_eq!(s.as_slice(), &[7, 8, 9, 1, 2, 3]);
    }

    #[test]
    fn test_index_select_2d_axis1() {
        // [[1, 2, 3],
        //  [4, 5, 6]]
        let t = Tensor::from_vec(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
        // Select cols 2, 0 -> [[3, 1], [6, 4]]
        let s = t.index_select(1, &[2, 0]).unwrap();
        assert_eq!(s.shape(), &[2, 2]);
        assert_eq!(s.as_slice(), &[3, 1, 6, 4]);
    }

    #[test]
    fn test_masked_select_flat() {
        let t = Tensor::from_vec(vec![10, 20, 30, 40, 50], vec![5]).unwrap();
        let mask = vec![true, false, true, false, true];
        let s = t.masked_select(&mask).unwrap();
        assert_eq!(s.shape(), &[3]);
        assert_eq!(s.as_slice(), &[10, 30, 50]);
    }

    #[test]
    fn test_masked_select_along_rows() {
        // [[1, 2],
        //  [3, 4],
        //  [5, 6]]
        let t = Tensor::from_vec(vec![1, 2, 3, 4, 5, 6], vec![3, 2]).unwrap();
        let mask = vec![false, true, true];
        let s = t.masked_select_along(&mask).unwrap();
        assert_eq!(s.shape(), &[2, 2]);
        assert_eq!(s.as_slice(), &[3, 4, 5, 6]);
    }

    #[test]
    fn test_index_put() {
        // [[1, 2, 3],
        //  [4, 5, 6],
        //  [7, 8, 9]]
        let mut t = Tensor::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9], vec![3, 3]).unwrap();
        // Replace rows 0 and 2
        let vals = Tensor::from_vec(vec![10, 20, 30, 70, 80, 90], vec![2, 3]).unwrap();
        t.index_put(0, &[0, 2], &vals).unwrap();
        assert_eq!(t.as_slice(), &[10, 20, 30, 4, 5, 6, 70, 80, 90]);
    }

    #[test]
    fn test_masked_put() {
        let mut t = Tensor::from_vec(vec![1, 2, 3, 4, 5], vec![5]).unwrap();
        let mask = vec![false, true, false, true, false];
        t.masked_put(&mask, &[99, 88]).unwrap();
        assert_eq!(t.as_slice(), &[1, 99, 3, 88, 5]);
    }

    #[test]
    fn test_index_out_of_bounds() {
        let t = Tensor::from_vec(vec![1, 2, 3], vec![3]).unwrap();
        assert!(t.index_select(0, &[5]).is_err());
        assert!(t.index_select(1, &[0]).is_err()); // axis OOB
    }
}
