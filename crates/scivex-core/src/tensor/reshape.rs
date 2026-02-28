//! Shape manipulation: reshape, transpose, flatten, squeeze, unsqueeze,
//! concatenate, and stack.

use crate::Scalar;
use crate::error::{CoreError, Result};

use super::{Tensor, compute_strides};

impl<T: Scalar> Tensor<T> {
    /// Reshape the tensor to a new shape without copying data.
    ///
    /// The total number of elements must remain the same.
    pub fn reshape(mut self, new_shape: Vec<usize>) -> Result<Self> {
        let new_numel: usize = new_shape.iter().product();
        if new_numel != self.numel() {
            return Err(CoreError::InvalidShape {
                shape: new_shape,
                reason: "new shape has different number of elements",
            });
        }
        self.strides = compute_strides(&new_shape);
        self.shape = new_shape;
        Ok(self)
    }

    /// Return a reshaped view without consuming the tensor (copies data).
    pub fn reshaped(&self, new_shape: Vec<usize>) -> Result<Self> {
        self.clone().reshape(new_shape)
    }

    /// Flatten the tensor into a 1-D tensor (consumes self, no copy).
    pub fn flatten(self) -> Self {
        let n = self.numel();
        Tensor {
            data: self.data,
            shape: vec![n],
            strides: vec![1],
        }
    }

    /// Return a flattened copy of the tensor.
    pub fn flattened(&self) -> Self {
        let n = self.numel();
        Tensor {
            data: self.data.clone(),
            shape: vec![n],
            strides: vec![1],
        }
    }

    /// Transpose a 2-D tensor (matrix). Returns a new tensor with copied data.
    ///
    /// For higher-rank tensors, use [`permute`](Self::permute).
    pub fn transpose(&self) -> Result<Self> {
        if self.ndim() != 2 {
            return Err(CoreError::InvalidArgument {
                reason: "transpose() requires a 2-D tensor; use permute() for higher ranks",
            });
        }
        let (rows, cols) = (self.shape[0], self.shape[1]);
        let mut data = vec![T::zero(); self.numel()];

        for r in 0..rows {
            for c in 0..cols {
                data[c * rows + r] = self.data[r * cols + c];
            }
        }

        Tensor::from_vec(data, vec![cols, rows])
    }

    /// Permute the dimensions of the tensor according to the given axes.
    ///
    /// `axes` must be a permutation of `0..ndim`.
    pub fn permute(&self, axes: &[usize]) -> Result<Self> {
        if axes.len() != self.ndim() {
            return Err(CoreError::InvalidArgument {
                reason: "axes length must match tensor rank",
            });
        }

        // Validate it's a valid permutation
        let mut seen = vec![false; self.ndim()];
        for &a in axes {
            if a >= self.ndim() {
                return Err(CoreError::AxisOutOfBounds {
                    axis: a,
                    ndim: self.ndim(),
                });
            }
            if seen[a] {
                return Err(CoreError::InvalidArgument {
                    reason: "duplicate axis in permutation",
                });
            }
            seen[a] = true;
        }

        let new_shape: Vec<usize> = axes.iter().map(|&a| self.shape[a]).collect();
        let new_strides = compute_strides(&new_shape);
        let new_numel: usize = new_shape.iter().product();
        let mut data = vec![T::zero(); new_numel];

        // Iterate over every element in the output
        let mut out_index = vec![0usize; self.ndim()];
        for item in &mut data {
            // Map output index back to input index
            let mut flat_in = 0;
            for (out_ax, &in_ax) in axes.iter().enumerate() {
                flat_in += out_index[out_ax] * self.strides[in_ax];
            }
            *item = self.data[flat_in];

            // Increment the output index (odometer style)
            for d in (0..self.ndim()).rev() {
                out_index[d] += 1;
                if out_index[d] < new_shape[d] {
                    break;
                }
                out_index[d] = 0;
            }
        }

        Ok(Tensor {
            data,
            shape: new_shape,
            strides: new_strides,
        })
    }

    /// Insert a dimension of size 1 at the given axis.
    pub fn unsqueeze(mut self, axis: usize) -> Result<Self> {
        if axis > self.ndim() {
            return Err(CoreError::AxisOutOfBounds {
                axis,
                ndim: self.ndim(),
            });
        }
        self.shape.insert(axis, 1);
        self.strides = compute_strides(&self.shape);
        Ok(self)
    }

    /// Remove all dimensions of size 1.
    pub fn squeeze(mut self) -> Self {
        self.shape.retain(|&d| d != 1);
        if self.shape.is_empty() && self.numel() == 1 {
            self.shape = vec![];
        }
        self.strides = compute_strides(&self.shape);
        self
    }

    /// Concatenate a list of tensors along the given axis.
    ///
    /// All tensors must have the same shape except along the concatenation axis.
    pub fn concat(tensors: &[&Tensor<T>], axis: usize) -> Result<Self> {
        if tensors.is_empty() {
            return Err(CoreError::InvalidArgument {
                reason: "cannot concatenate zero tensors",
            });
        }

        let ndim = tensors[0].ndim();
        if axis >= ndim {
            return Err(CoreError::AxisOutOfBounds { axis, ndim });
        }

        // Validate shapes match on all axes except `axis`
        for t in &tensors[1..] {
            if t.ndim() != ndim {
                return Err(CoreError::DimensionMismatch {
                    expected: tensors[0].shape.clone(),
                    got: t.shape.clone(),
                });
            }
            for (d, (&a, &b)) in tensors[0].shape.iter().zip(t.shape.iter()).enumerate() {
                if d != axis && a != b {
                    return Err(CoreError::DimensionMismatch {
                        expected: tensors[0].shape.clone(),
                        got: t.shape.clone(),
                    });
                }
            }
        }

        let mut new_shape = tensors[0].shape.clone();
        new_shape[axis] = tensors.iter().map(|t| t.shape[axis]).sum();

        let outer: usize = new_shape[..axis].iter().product();
        let inner: usize = new_shape[axis + 1..].iter().product();
        let total: usize = new_shape.iter().product();

        let mut data = Vec::with_capacity(total);

        for o in 0..outer {
            for t in tensors {
                let axis_len = t.shape[axis];
                let src_start = o * axis_len * inner;
                let src_end = src_start + axis_len * inner;
                data.extend_from_slice(&t.data[src_start..src_end]);
            }
        }

        Tensor::from_vec(data, new_shape)
    }

    /// Stack tensors along a new axis inserted at position `axis`.
    ///
    /// All tensors must have identical shapes.
    pub fn stack(tensors: &[&Tensor<T>], axis: usize) -> Result<Self> {
        if tensors.is_empty() {
            return Err(CoreError::InvalidArgument {
                reason: "cannot stack zero tensors",
            });
        }

        let base_shape = &tensors[0].shape;
        if axis > base_shape.len() {
            return Err(CoreError::AxisOutOfBounds {
                axis,
                ndim: base_shape.len() + 1,
            });
        }

        for t in &tensors[1..] {
            if t.shape != *base_shape {
                return Err(CoreError::DimensionMismatch {
                    expected: base_shape.clone(),
                    got: t.shape.clone(),
                });
            }
        }

        // Unsqueeze each tensor along the new axis, then concat
        let expanded: Vec<Tensor<T>> = tensors
            .iter()
            .map(|t| (*t).clone().unsqueeze(axis).unwrap())
            .collect();
        let refs: Vec<&Tensor<T>> = expanded.iter().collect();
        Tensor::concat(&refs, axis)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reshape() {
        let t = Tensor::from_vec(vec![1, 2, 3, 4, 5, 6], vec![6]).unwrap();
        let t = t.reshape(vec![2, 3]).unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.strides(), &[3, 1]);
        assert_eq!(*t.get(&[1, 0]).unwrap(), 4);
    }

    #[test]
    fn test_reshape_invalid() {
        let t = Tensor::from_vec(vec![1, 2, 3, 4], vec![4]).unwrap();
        assert!(t.reshape(vec![3, 2]).is_err());
    }

    #[test]
    fn test_flatten() {
        let t = Tensor::from_vec(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
        let flat = t.flatten();
        assert_eq!(flat.shape(), &[6]);
        assert_eq!(flat.as_slice(), &[1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_transpose() {
        // [[1, 2, 3],
        //  [4, 5, 6]]
        let t = Tensor::from_vec(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
        let tt = t.transpose().unwrap();
        assert_eq!(tt.shape(), &[3, 2]);
        assert_eq!(*tt.get(&[0, 0]).unwrap(), 1);
        assert_eq!(*tt.get(&[0, 1]).unwrap(), 4);
        assert_eq!(*tt.get(&[2, 0]).unwrap(), 3);
        assert_eq!(*tt.get(&[2, 1]).unwrap(), 6);
    }

    #[test]
    fn test_transpose_not_2d() {
        let t = Tensor::from_vec(vec![1, 2, 3], vec![3]).unwrap();
        assert!(t.transpose().is_err());
    }

    #[test]
    fn test_permute() {
        // Shape [2, 3, 4] -> permute [2, 0, 1] -> shape [4, 2, 3]
        let t = Tensor::<i32>::arange(24).reshape(vec![2, 3, 4]).unwrap();
        let p = t.permute(&[2, 0, 1]).unwrap();
        assert_eq!(p.shape(), &[4, 2, 3]);
        // Element at [0, 0, 0] in original is at [0, 0, 0] in permuted
        assert_eq!(*p.get(&[0, 0, 0]).unwrap(), 0);
        // Element at [1, 2, 3] in original -> permuted[3, 1, 2]
        assert_eq!(*p.get(&[3, 1, 2]).unwrap(), *t.get(&[1, 2, 3]).unwrap());
    }

    #[test]
    fn test_unsqueeze_squeeze() {
        let t = Tensor::from_vec(vec![1, 2, 3], vec![3]).unwrap();
        let t = t.unsqueeze(0).unwrap();
        assert_eq!(t.shape(), &[1, 3]);
        let t = t.squeeze();
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_concat() {
        let a = Tensor::from_vec(vec![1, 2, 3], vec![1, 3]).unwrap();
        let b = Tensor::from_vec(vec![4, 5, 6], vec![1, 3]).unwrap();
        let c = Tensor::concat(&[&a, &b], 0).unwrap();
        assert_eq!(c.shape(), &[2, 3]);
        assert_eq!(c.as_slice(), &[1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_concat_axis1() {
        let a = Tensor::from_vec(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
        let b = Tensor::from_vec(vec![5, 6, 7, 8], vec![2, 2]).unwrap();
        let c = Tensor::concat(&[&a, &b], 1).unwrap();
        assert_eq!(c.shape(), &[2, 4]);
        assert_eq!(c.as_slice(), &[1, 2, 5, 6, 3, 4, 7, 8]);
    }

    #[test]
    fn test_stack() {
        let a = Tensor::from_vec(vec![1, 2, 3], vec![3]).unwrap();
        let b = Tensor::from_vec(vec![4, 5, 6], vec![3]).unwrap();
        let c = Tensor::stack(&[&a, &b], 0).unwrap();
        assert_eq!(c.shape(), &[2, 3]);
        assert_eq!(c.as_slice(), &[1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_stack_axis1() {
        let a = Tensor::from_vec(vec![1, 2, 3], vec![3]).unwrap();
        let b = Tensor::from_vec(vec![4, 5, 6], vec![3]).unwrap();
        let c = Tensor::stack(&[&a, &b], 1).unwrap();
        assert_eq!(c.shape(), &[3, 2]);
        assert_eq!(c.as_slice(), &[1, 4, 2, 5, 3, 6]);
    }

    #[test]
    fn test_concat_shape_mismatch() {
        let a = Tensor::from_vec(vec![1, 2, 3], vec![1, 3]).unwrap();
        let b = Tensor::from_vec(vec![4, 5], vec![1, 2]).unwrap();
        assert!(Tensor::concat(&[&a, &b], 0).is_err());
    }
}
