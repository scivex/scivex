//! Named dimensions for tensors.
//!
//! [`NamedTensor`] wraps a [`Tensor`] and associates optional string names
//! with each dimension, enabling dimension lookup and reordering by name.

use crate::Scalar;
use crate::dtype::Float;
use crate::error::{CoreError, Result};

use super::{Tensor, compute_strides};

/// A tensor with optional names attached to each dimension.
///
/// Unnamed dimensions use `None`. Named dimensions must be unique within
/// a single tensor.
///
/// # Examples
///
/// ```
/// # use scivex_core::tensor::named::NamedTensor;
/// # use scivex_core::Tensor;
/// let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
/// let nt = NamedTensor::new(t, vec![Some("batch".into()), Some("feature".into())]).unwrap();
/// assert_eq!(nt.dim_index("batch").unwrap(), 0);
/// assert_eq!(nt.dim_index("feature").unwrap(), 1);
/// ```
#[derive(Debug, Clone)]
pub struct NamedTensor<T: Scalar> {
    tensor: Tensor<T>,
    names: Vec<Option<String>>,
}

impl<T: Scalar> NamedTensor<T> {
    /// Create a named tensor, validating that the number of names matches
    /// the tensor's rank and that named dimensions are unique.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::InvalidArgument`] if `names.len() != tensor.ndim()`
    /// or if duplicate dimension names are found.
    pub fn new(tensor: Tensor<T>, names: Vec<Option<String>>) -> Result<Self> {
        if names.len() != tensor.ndim() {
            return Err(CoreError::InvalidArgument {
                reason: "number of dimension names must match tensor rank",
            });
        }
        // Check for duplicate named dimensions.
        let named: Vec<&str> = names.iter().filter_map(|n| n.as_deref()).collect();
        let mut sorted = named.clone();
        sorted.sort_unstable();
        for window in sorted.windows(2) {
            if window[0] == window[1] {
                return Err(CoreError::InvalidArgument {
                    reason: "duplicate dimension names are not allowed",
                });
            }
        }
        Ok(Self { tensor, names })
    }

    /// Wrap a tensor with all dimensions unnamed.
    pub fn from_tensor(tensor: Tensor<T>) -> Self {
        let ndim = tensor.ndim();
        Self {
            tensor,
            names: vec![None; ndim],
        }
    }

    /// Borrow the inner tensor.
    #[inline]
    pub fn tensor(&self) -> &Tensor<T> {
        &self.tensor
    }

    /// Get the dimension names.
    #[inline]
    pub fn names(&self) -> &[Option<String>] {
        &self.names
    }

    /// Consume the wrapper and return the plain tensor.
    #[inline]
    pub fn into_tensor(self) -> Tensor<T> {
        self.tensor
    }

    /// Look up the axis index for a named dimension.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::InvalidArgument`] if no dimension has the given name.
    pub fn dim_index(&self, name: &str) -> Result<usize> {
        self.names
            .iter()
            .position(|n| n.as_deref() == Some(name))
            .ok_or(CoreError::InvalidArgument {
                reason: "dimension name not found",
            })
    }

    /// Rename a dimension from `old` to `new`.
    ///
    /// # Errors
    ///
    /// Returns an error if `old` is not found or `new` already exists.
    pub fn rename(&mut self, old: &str, new: &str) -> Result<()> {
        // Check that new name doesn't already exist.
        if self.names.iter().any(|n| n.as_deref() == Some(new)) {
            return Err(CoreError::InvalidArgument {
                reason: "new dimension name already exists",
            });
        }
        let idx = self.dim_index(old)?;
        self.names[idx] = Some(new.to_string());
        Ok(())
    }

    /// Replace all dimension names at once.
    ///
    /// # Errors
    ///
    /// Returns an error if the length does not match the tensor rank.
    pub fn set_names(&mut self, names: Vec<Option<String>>) -> Result<()> {
        if names.len() != self.tensor.ndim() {
            return Err(CoreError::InvalidArgument {
                reason: "number of dimension names must match tensor rank",
            });
        }
        self.names = names;
        Ok(())
    }

    /// Reorder dimensions by name, producing a permuted copy.
    ///
    /// Every named dimension in the current tensor must appear in `target_names`,
    /// and the lengths must match the tensor rank.
    ///
    /// # Errors
    ///
    /// Returns an error if a name is not found or if the number of names does
    /// not match the rank.
    pub fn align_to(&self, target_names: &[&str]) -> Result<NamedTensor<T>> {
        if target_names.len() != self.tensor.ndim() {
            return Err(CoreError::InvalidArgument {
                reason: "target names length must match tensor rank",
            });
        }

        // Build permutation: perm[i] = source axis for target axis i.
        let perm: Vec<usize> = target_names
            .iter()
            .map(|name| self.dim_index(name))
            .collect::<Result<Vec<_>>>()?;

        let src_shape = self.tensor.shape();
        let src_strides = self.tensor.strides();
        let src_data = self.tensor.as_slice();

        // New shape and names according to the permutation.
        let new_shape: Vec<usize> = perm.iter().map(|&p| src_shape[p]).collect();
        let new_names: Vec<Option<String>> = perm.iter().map(|&p| self.names[p].clone()).collect();
        let new_strides = compute_strides(&new_shape);

        let numel: usize = new_shape.iter().product();
        let mut new_data = vec![T::zero(); numel];

        // Iterate over every element in the output tensor, compute source index.
        for (out_flat, dest) in new_data.iter_mut().enumerate() {
            // Convert out_flat to multi-dim index in the output.
            let mut remaining = out_flat;
            let mut src_flat = 0usize;
            for (dim, &stride) in new_strides.iter().enumerate() {
                let idx = remaining / stride;
                remaining %= stride;
                src_flat += idx * src_strides[perm[dim]];
            }
            *dest = src_data[src_flat];
        }

        let new_tensor = Tensor::from_vec(new_data, new_shape)?;
        Ok(NamedTensor {
            tensor: new_tensor,
            names: new_names,
        })
    }

    /// Select a single index along a named dimension, reducing the rank by one.
    ///
    /// # Errors
    ///
    /// Returns an error if the name is not found or the index is out of bounds.
    pub fn select(&self, name: &str, index: usize) -> Result<NamedTensor<T>> {
        let axis = self.dim_index(name)?;
        let shape = self.tensor.shape();
        if index >= shape[axis] {
            return Err(CoreError::IndexOutOfBounds {
                index: vec![index],
                shape: shape.to_vec(),
            });
        }

        let ndim = shape.len();
        let strides = self.tensor.strides();
        let src_data = self.tensor.as_slice();

        // New shape: remove the selected axis.
        let new_shape: Vec<usize> = shape
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != axis)
            .map(|(_, &s)| s)
            .collect();
        let new_names: Vec<Option<String>> = self
            .names
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != axis)
            .map(|(_, n)| n.clone())
            .collect();

        let numel: usize = new_shape.iter().product();
        let new_strides = compute_strides(&new_shape);
        let mut new_data = vec![T::zero(); numel];

        // Build a mapping from output dim to source dim (skipping `axis`).
        let dim_map: Vec<usize> = (0..ndim).filter(|&d| d != axis).collect();

        for (out_flat, dest) in new_data.iter_mut().enumerate() {
            let mut remaining = out_flat;
            let mut src_flat = index * strides[axis];
            for (out_dim, &src_dim) in dim_map.iter().enumerate() {
                let idx = if out_dim < new_strides.len() {
                    let i = remaining / new_strides[out_dim];
                    remaining %= new_strides[out_dim];
                    i
                } else {
                    remaining
                };
                src_flat += idx * strides[src_dim];
            }
            *dest = src_data[src_flat];
        }

        let new_tensor = Tensor::from_vec(new_data, new_shape)?;
        Ok(NamedTensor {
            tensor: new_tensor,
            names: new_names,
        })
    }
}

impl<T: Scalar + Float> NamedTensor<T> {
    /// Sum along a named dimension, removing it from the result.
    ///
    /// # Errors
    ///
    /// Returns an error if the name is not found.
    pub fn sum_dim(&self, name: &str) -> Result<NamedTensor<T>> {
        let axis = self.dim_index(name)?;
        let shape = self.tensor.shape();
        let strides = self.tensor.strides();
        let src_data = self.tensor.as_slice();
        let ndim = shape.len();
        let axis_len = shape[axis];

        let new_shape: Vec<usize> = shape
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != axis)
            .map(|(_, &s)| s)
            .collect();
        let new_names: Vec<Option<String>> = self
            .names
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != axis)
            .map(|(_, n)| n.clone())
            .collect();

        let numel: usize = new_shape.iter().product();
        let new_strides = compute_strides(&new_shape);
        let mut new_data = vec![T::zero(); numel];

        // Build a mapping from output dim to source dim (skipping `axis`).
        let dim_map: Vec<usize> = (0..ndim).filter(|&d| d != axis).collect();

        for (out_flat, dest) in new_data.iter_mut().enumerate() {
            let mut remaining = out_flat;
            // Decode the output flat index into per-source-dim indices (skipping axis).
            let mut out_indices = vec![0usize; ndim];
            for (out_dim, &src_dim) in dim_map.iter().enumerate() {
                let idx = if out_dim < new_strides.len() {
                    let i = remaining / new_strides[out_dim];
                    remaining %= new_strides[out_dim];
                    i
                } else {
                    remaining
                };
                out_indices[src_dim] = idx;
            }

            let mut acc = T::zero();
            for k in 0..axis_len {
                out_indices[axis] = k;
                let src_flat: usize = out_indices
                    .iter()
                    .zip(strides.iter())
                    .map(|(&idx, &s)| idx * s)
                    .sum();
                acc += src_data[src_flat];
            }
            *dest = acc;
        }

        let new_tensor = Tensor::from_vec(new_data, new_shape)?;
        Ok(NamedTensor {
            tensor: new_tensor,
            names: new_names,
        })
    }

    /// Mean along a named dimension, removing it from the result.
    ///
    /// # Errors
    ///
    /// Returns an error if the name is not found.
    pub fn mean_dim(&self, name: &str) -> Result<NamedTensor<T>> {
        let axis = self.dim_index(name)?;
        let axis_len = self.tensor.shape()[axis];
        let summed = self.sum_dim(name)?;
        let divisor = T::from_usize(axis_len);
        let result_tensor = summed.tensor.map(|x| x / divisor);
        Ok(NamedTensor {
            tensor: result_tensor,
            names: summed.names,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_named_tensor_basic() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let nt = NamedTensor::new(t, vec![Some("batch".into()), Some("feature".into())]).unwrap();
        assert_eq!(nt.names().len(), 2);
        assert_eq!(nt.names()[0].as_deref(), Some("batch"));
        assert_eq!(nt.names()[1].as_deref(), Some("feature"));
        assert_eq!(nt.tensor().shape(), &[2, 3]);
    }

    #[test]
    fn test_rename_dimension() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let mut nt = NamedTensor::new(t, vec![Some("rows".into()), Some("cols".into())]).unwrap();
        nt.rename("rows", "samples").unwrap();
        assert_eq!(nt.names()[0].as_deref(), Some("samples"));
        assert_eq!(nt.dim_index("samples").unwrap(), 0);
        assert!(nt.dim_index("rows").is_err());
    }

    #[test]
    fn test_align_to() {
        // 2x3x4 tensor with named dims
        let numel = 2 * 3 * 4;
        let data: Vec<f64> = (0..numel).map(f64::from).collect();
        let t = Tensor::from_vec(data, vec![2, 3, 4]).unwrap();
        let nt = NamedTensor::new(
            t.clone(),
            vec![
                Some("batch".into()),
                Some("channel".into()),
                Some("width".into()),
            ],
        )
        .unwrap();

        // Reorder to (channel, width, batch) = (3, 4, 2)
        let aligned = nt.align_to(&["channel", "width", "batch"]).unwrap();
        assert_eq!(aligned.tensor().shape(), &[3, 4, 2]);
        assert_eq!(aligned.names()[0].as_deref(), Some("channel"));
        assert_eq!(aligned.names()[1].as_deref(), Some("width"));
        assert_eq!(aligned.names()[2].as_deref(), Some("batch"));

        // Verify a specific element: original [1, 2, 3] should appear at aligned [2, 3, 1]
        let original_val = *t.get(&[1, 2, 3]).unwrap();
        let aligned_val = *aligned.tensor().get(&[2, 3, 1]).unwrap();
        assert!((original_val - aligned_val).abs() < 1e-15);
    }

    #[test]
    fn test_dim_index() {
        let t = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0], vec![3]).unwrap();
        let nt = NamedTensor::new(t, vec![Some("time".into())]).unwrap();
        assert_eq!(nt.dim_index("time").unwrap(), 0);
        assert!(nt.dim_index("space").is_err());
    }

    #[test]
    fn test_sum_dim() {
        // 2x3 tensor, sum along "rows" (axis 0)
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let nt = NamedTensor::new(t, vec![Some("rows".into()), Some("cols".into())]).unwrap();
        let summed = nt.sum_dim("rows").unwrap();
        assert_eq!(summed.tensor().shape(), &[3]);
        let data = summed.tensor().as_slice();
        assert!((data[0] - 5.0).abs() < 1e-15); // 1 + 4
        assert!((data[1] - 7.0).abs() < 1e-15); // 2 + 5
        assert!((data[2] - 9.0).abs() < 1e-15); // 3 + 6
        assert_eq!(summed.names()[0].as_deref(), Some("cols"));
    }

    #[test]
    fn test_select() {
        // 2x3 tensor, select row 1 along "rows"
        let t = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let nt = NamedTensor::new(t, vec![Some("rows".into()), Some("cols".into())]).unwrap();
        let selected = nt.select("rows", 1).unwrap();
        assert_eq!(selected.tensor().shape(), &[3]);
        assert_eq!(selected.tensor().as_slice(), &[4.0, 5.0, 6.0]);
        assert_eq!(selected.names()[0].as_deref(), Some("cols"));
    }

    #[test]
    fn test_invalid_names_length() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let result = NamedTensor::new(t, vec![Some("a".into()), Some("b".into())]);
        assert!(result.is_err());
    }
}
