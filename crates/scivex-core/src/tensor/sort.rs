//! Sorting operations for tensors.
//!
//! Provides `sort`, `argsort` (flat), and axis-based `sort_axis` / `argsort_axis`.

use crate::Scalar;
use crate::error::{CoreError, Result};

use super::Tensor;

impl<T: Scalar> Tensor<T> {
    /// Sort all elements, returning a 1-D tensor in ascending order.
    pub fn sort(&self) -> Tensor<T> {
        let mut data = self.data.clone();
        data.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
        Tensor {
            data,
            shape: vec![self.numel()],
            strides: vec![1],
        }
    }

    /// Return indices that would sort all elements (flat), as a 1-D `Tensor<usize>`.
    pub fn argsort(&self) -> Tensor<usize> {
        let mut indices: Vec<usize> = (0..self.numel()).collect();
        indices.sort_unstable_by(|&a, &b| {
            self.data[a]
                .partial_cmp(&self.data[b])
                .unwrap_or(core::cmp::Ordering::Equal)
        });
        Tensor {
            data: indices,
            shape: vec![self.numel()],
            strides: vec![1],
        }
    }

    /// Sort along a given axis, returning a new tensor with the same shape.
    ///
    /// Each 1-D slice along `axis` is sorted independently in ascending order.
    pub fn sort_axis(&self, axis: usize) -> Result<Tensor<T>> {
        if axis >= self.ndim() {
            return Err(CoreError::AxisOutOfBounds {
                axis,
                ndim: self.ndim(),
            });
        }

        let mut result = self.data.clone();
        let outer: usize = self.shape[..axis].iter().product();
        let axis_len = self.shape[axis];
        let inner: usize = self.shape[axis + 1..].iter().product();

        let mut slice_buf = vec![T::zero(); axis_len];

        for o in 0..outer {
            for i in 0..inner {
                // Extract the 1-D slice along the axis
                let base = o * axis_len * inner + i;
                for (k, slot) in slice_buf.iter_mut().enumerate() {
                    *slot = self.data[base + k * inner];
                }

                // Sort the slice
                slice_buf.sort_unstable_by(|a, b| {
                    a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal)
                });

                // Write back
                for (k, &val) in slice_buf.iter().enumerate() {
                    result[base + k * inner] = val;
                }
            }
        }

        Tensor::from_vec(result, self.shape.clone())
    }

    /// Return indices that would sort each slice along `axis`.
    ///
    /// The result has the same shape as `self` but element type `usize`.
    pub fn argsort_axis(&self, axis: usize) -> Result<Tensor<usize>> {
        if axis >= self.ndim() {
            return Err(CoreError::AxisOutOfBounds {
                axis,
                ndim: self.ndim(),
            });
        }

        let outer: usize = self.shape[..axis].iter().product();
        let axis_len = self.shape[axis];
        let inner: usize = self.shape[axis + 1..].iter().product();

        let numel = self.numel();
        let mut result = vec![0usize; numel];
        let mut idx_buf: Vec<usize> = (0..axis_len).collect();

        for o in 0..outer {
            for i in 0..inner {
                let base = o * axis_len * inner + i;

                // Reset index buffer
                for (k, slot) in idx_buf.iter_mut().enumerate() {
                    *slot = k;
                }

                // Sort indices by comparing elements along the axis
                idx_buf.sort_unstable_by(|&a, &b| {
                    let va = self.data[base + a * inner];
                    let vb = self.data[base + b * inner];
                    va.partial_cmp(&vb).unwrap_or(core::cmp::Ordering::Equal)
                });

                // Write back
                for k in 0..axis_len {
                    result[base + k * inner] = idx_buf[k];
                }
            }
        }

        Tensor::from_vec(result, self.shape.clone())
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn test_sort_1d() {
        let t = Tensor::from_vec(vec![3, 1, 4, 1, 5, 9], vec![6]).unwrap();
        let s = t.sort();
        assert_eq!(s.as_slice(), &[1, 1, 3, 4, 5, 9]);
        assert_eq!(s.shape(), &[6]);
    }

    #[test]
    fn test_argsort_1d() {
        let t = Tensor::from_vec(vec![3, 1, 4, 1, 5, 9], vec![6]).unwrap();
        let idx = t.argsort();
        // Values at sorted indices should be ascending
        let sorted: Vec<i32> = idx.as_slice().iter().map(|&i| t.as_slice()[i]).collect();
        assert_eq!(sorted, &[1, 1, 3, 4, 5, 9]);
    }

    #[test]
    fn test_sort_axis0_2d() {
        // [[3, 1],
        //  [1, 4]]
        let t = Tensor::from_vec(vec![3, 1, 1, 4], vec![2, 2]).unwrap();
        let s = t.sort_axis(0).unwrap();
        // Sort columns: col0=[3,1]->[1,3], col1=[1,4]->[1,4]
        assert_eq!(s.as_slice(), &[1, 1, 3, 4]);
    }

    #[test]
    fn test_sort_axis1_2d() {
        // [[3, 1],
        //  [4, 2]]
        let t = Tensor::from_vec(vec![3, 1, 4, 2], vec![2, 2]).unwrap();
        let s = t.sort_axis(1).unwrap();
        // Sort rows: row0=[3,1]->[1,3], row1=[4,2]->[2,4]
        assert_eq!(s.as_slice(), &[1, 3, 2, 4]);
    }

    #[test]
    fn test_argsort_axis() {
        let t = Tensor::from_vec(vec![3.0_f64, 1.0, 4.0, 2.0], vec![2, 2]).unwrap();
        let idx = t.argsort_axis(1).unwrap();
        // Row 0: [3, 1] -> argsort = [1, 0]
        // Row 1: [4, 2] -> argsort = [1, 0]
        assert_eq!(idx.as_slice(), &[1, 0, 1, 0]);
        assert_eq!(idx.shape(), &[2, 2]);
    }

    #[test]
    fn test_flat_sort_from_2d() {
        let t = Tensor::from_vec(vec![5, 2, 8, 1], vec![2, 2]).unwrap();
        let s = t.sort();
        assert_eq!(s.as_slice(), &[1, 2, 5, 8]);
        assert_eq!(s.shape(), &[4]);
    }

    #[test]
    fn test_sort_axis_out_of_bounds() {
        let t = Tensor::from_vec(vec![1, 2, 3], vec![3]).unwrap();
        assert!(t.sort_axis(1).is_err());
        assert!(t.argsort_axis(1).is_err());
    }

    #[test]
    fn test_already_sorted() {
        let t = Tensor::from_vec(vec![1, 2, 3, 4, 5], vec![5]).unwrap();
        let s = t.sort();
        assert_eq!(s.as_slice(), &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_reverse_sorted() {
        let t = Tensor::from_vec(vec![5, 4, 3, 2, 1], vec![5]).unwrap();
        let s = t.sort();
        assert_eq!(s.as_slice(), &[1, 2, 3, 4, 5]);
    }
}
