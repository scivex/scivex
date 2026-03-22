//! N-dimensional sparse tensor in COO (coordinate) format.
//!
//! Unlike [`crate::linalg::sparse::CooMatrix`], which is a 2D sparse matrix
//! for linear algebra, [`SparseTensor`] generalises to arbitrary dimensions
//! while still supporting efficient 2D matrix operations (matmul, transpose).

use crate::Scalar;
use crate::error::{CoreError, Result};
use crate::tensor::Tensor;

use std::collections::HashMap;

/// An N-dimensional sparse tensor stored in COO (coordinate) format.
///
/// Each non-zero element is represented by its coordinates (one index per
/// dimension) and the corresponding value.
///
/// # Examples
///
/// ```
/// # use scivex_core::tensor::sparse::SparseTensor;
/// let st = SparseTensor::new(
///     vec![vec![0, 1], vec![1, 2]],
///     vec![3.0_f64, 7.0],
///     vec![3, 4],
/// ).unwrap();
/// assert_eq!(st.nnz(), 2);
/// assert_eq!(st.ndim(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct SparseTensor<T: Scalar> {
    /// `indices[dim][k]` is the coordinate in dimension `dim` for the k-th
    /// stored element.
    indices: Vec<Vec<usize>>,
    /// Values of the stored (non-zero) elements.
    values: Vec<T>,
    /// Shape of the full dense tensor.
    shape: Vec<usize>,
    /// Number of stored elements.
    nnz: usize,
}

impl<T: Scalar> SparseTensor<T> {
    // ------------------------------------------------------------------
    // Constructors
    // ------------------------------------------------------------------

    /// Create a sparse tensor from explicit index lists, values, and shape.
    ///
    /// `indices` must have one entry per dimension, and each inner vec must
    /// have the same length as `values`. All indices must be in-bounds for the
    /// corresponding dimension of `shape`.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::InvalidArgument`] when lengths are inconsistent or
    /// indices are out of bounds.
    pub fn new(indices: Vec<Vec<usize>>, values: Vec<T>, shape: Vec<usize>) -> Result<Self> {
        let ndim = shape.len();
        if indices.len() != ndim {
            return Err(CoreError::InvalidArgument {
                reason: "indices length must equal number of dimensions",
            });
        }
        let nnz = values.len();
        for (dim, idx_vec) in indices.iter().enumerate() {
            if idx_vec.len() != nnz {
                return Err(CoreError::InvalidArgument {
                    reason: "all index vectors must have the same length as values",
                });
            }
            for &idx in idx_vec {
                if idx >= shape[dim] {
                    return Err(CoreError::IndexOutOfBounds {
                        index: vec![idx],
                        shape: shape.clone(),
                    });
                }
            }
        }
        Ok(Self {
            indices,
            values,
            shape,
            nnz,
        })
    }

    /// Build a sparse tensor from a dense [`Tensor`], keeping only non-zero
    /// entries (where value != `T::zero()`).
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_core::{Tensor, tensor::sparse::SparseTensor};
    /// let t = Tensor::from_vec(vec![0.0, 1.0, 0.0, 2.0], vec![2, 2]).unwrap();
    /// let st = SparseTensor::from_dense(&t);
    /// assert_eq!(st.nnz(), 2);
    /// ```
    pub fn from_dense(tensor: &Tensor<T>) -> Self {
        let shape = tensor.shape().to_vec();
        let ndim = shape.len();
        let strides = tensor.strides().to_vec();

        let mut indices: Vec<Vec<usize>> = vec![Vec::new(); ndim];
        let mut values = Vec::new();

        for (flat, &val) in tensor.as_slice().iter().enumerate() {
            if val != T::zero() {
                // Convert flat index to multi-dimensional coordinates.
                let mut remaining = flat;
                for (dim, &stride) in strides.iter().enumerate() {
                    let coord = if stride == 0 { 0 } else { remaining / stride };
                    indices[dim].push(coord);
                    if stride != 0 {
                        remaining %= stride;
                    }
                }
                values.push(val);
            }
        }

        let nnz = values.len();
        Self {
            indices,
            values,
            shape,
            nnz,
        }
    }

    /// Create an empty sparse tensor (no stored elements) with the given shape.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_core::tensor::sparse::SparseTensor;
    /// let st = SparseTensor::<f64>::zeros(vec![4, 5]);
    /// assert_eq!(st.nnz(), 0);
    /// assert_eq!(st.shape(), &[4, 5]);
    /// ```
    pub fn zeros(shape: Vec<usize>) -> Self {
        let ndim = shape.len();
        Self {
            indices: vec![Vec::new(); ndim],
            values: Vec::new(),
            shape,
            nnz: 0,
        }
    }

    // ------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------

    /// Number of stored (non-zero) elements.
    #[inline]
    pub fn nnz(&self) -> usize {
        self.nnz
    }

    /// Shape of the full dense tensor.
    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Number of dimensions (rank).
    #[inline]
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Fraction of elements that are stored: `nnz / total_elements`.
    ///
    /// Returns `0.0` for a tensor with zero total elements.
    pub fn density(&self) -> f64 {
        let total: usize = self.shape.iter().product();
        if total == 0 {
            return 0.0;
        }
        self.nnz as f64 / total as f64
    }

    /// Slice of stored values.
    #[inline]
    pub fn values(&self) -> &[T] {
        &self.values
    }

    /// Slice of index vectors (one per dimension).
    #[inline]
    pub fn indices(&self) -> &[Vec<usize>] {
        &self.indices
    }

    // ------------------------------------------------------------------
    // Conversion
    // ------------------------------------------------------------------

    /// Expand this sparse tensor into a fully materialised dense [`Tensor`].
    ///
    /// Duplicate coordinates are summed (call [`coalesce`](Self::coalesce)
    /// first if you want canonical form).
    ///
    /// # Errors
    ///
    /// Returns an error if the resulting dense tensor would be invalid (should
    /// not happen for a well-formed `SparseTensor`).
    pub fn to_dense(&self) -> Result<Tensor<T>> {
        let mut dense = Tensor::zeros(self.shape.clone());
        let strides = dense.strides().to_vec();
        for k in 0..self.nnz {
            let flat: usize = self
                .indices
                .iter()
                .enumerate()
                .map(|(dim, idx_vec)| idx_vec[k] * strides[dim])
                .sum();
            let data = dense.as_mut_slice();
            data[flat] += self.values[k];
        }
        Ok(dense)
    }

    // ------------------------------------------------------------------
    // Element-wise operations
    // ------------------------------------------------------------------

    /// Element-wise addition of two sparse tensors with the same shape.
    ///
    /// The result may contain duplicate indices; call [`coalesce`](Self::coalesce)
    /// to merge them.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::DimensionMismatch`] if shapes differ.
    pub fn add(&self, other: &SparseTensor<T>) -> Result<SparseTensor<T>> {
        if self.shape != other.shape {
            return Err(CoreError::DimensionMismatch {
                expected: self.shape.clone(),
                got: other.shape.clone(),
            });
        }
        let new_nnz = self.nnz + other.nnz;
        let mut indices: Vec<Vec<usize>> = self
            .indices
            .iter()
            .zip(other.indices.iter())
            .map(|(a, b)| {
                let mut v = a.clone();
                v.extend_from_slice(b);
                v
            })
            .collect();
        // Handle 0-dim tensors (no index vectors to concatenate).
        if indices.is_empty() && self.shape.is_empty() {
            indices = Vec::new();
        }
        let mut values = self.values.clone();
        values.extend_from_slice(&other.values);
        Ok(SparseTensor {
            indices,
            values,
            shape: self.shape.clone(),
            nnz: new_nnz,
        })
    }

    /// Multiply every stored value by a scalar.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_core::tensor::sparse::SparseTensor;
    /// let st = SparseTensor::new(
    ///     vec![vec![0, 1], vec![0, 1]],
    ///     vec![2.0_f64, 3.0],
    ///     vec![2, 2],
    /// ).unwrap();
    /// let scaled = st.scalar_mul(10.0);
    /// assert_eq!(scaled.values(), &[20.0, 30.0]);
    /// ```
    pub fn scalar_mul(&self, scalar: T) -> SparseTensor<T> {
        SparseTensor {
            indices: self.indices.clone(),
            values: self.values.iter().map(|&v| v * scalar).collect(),
            shape: self.shape.clone(),
            nnz: self.nnz,
        }
    }

    // ------------------------------------------------------------------
    // Matrix operations (2D only)
    // ------------------------------------------------------------------

    /// Sparse-sparse matrix multiplication (2D tensors only).
    ///
    /// Computes `self @ other` where `self` is `(M, K)` and `other` is `(K, N)`.
    /// The result is an `(M, N)` sparse tensor.
    ///
    /// # Errors
    ///
    /// Returns an error if either tensor is not 2D or inner dimensions do not
    /// match.
    pub fn sparse_matmul(&self, other: &SparseTensor<T>) -> Result<SparseTensor<T>> {
        if self.ndim() != 2 || other.ndim() != 2 {
            return Err(CoreError::InvalidArgument {
                reason: "sparse_matmul requires 2D tensors",
            });
        }
        let m = self.shape[0];
        let k_self = self.shape[1];
        let k_other = other.shape[0];
        let n = other.shape[1];
        if k_self != k_other {
            return Err(CoreError::DimensionMismatch {
                expected: self.shape.clone(),
                got: other.shape.clone(),
            });
        }

        // Build a map from column index in `other` -> list of (row, value).
        // Actually, build row-index of other -> list of (col, value) for
        // efficient inner-product style multiplication.
        let mut other_row_map: HashMap<usize, Vec<(usize, T)>> = HashMap::new();
        for k in 0..other.nnz {
            other_row_map
                .entry(other.indices[0][k])
                .or_default()
                .push((other.indices[1][k], other.values[k]));
        }

        let mut result_map: HashMap<(usize, usize), T> = HashMap::new();
        for k in 0..self.nnz {
            let row = self.indices[0][k];
            let col = self.indices[1][k];
            let val = self.values[k];
            if let Some(entries) = other_row_map.get(&col) {
                for &(other_col, other_val) in entries {
                    let entry = result_map.entry((row, other_col)).or_insert(T::zero());
                    *entry += val * other_val;
                }
            }
        }

        // Collect results, filtering out zeros.
        let mut row_indices = Vec::new();
        let mut col_indices = Vec::new();
        let mut values = Vec::new();
        for (&(r, c), &v) in &result_map {
            if v != T::zero() {
                row_indices.push(r);
                col_indices.push(c);
                values.push(v);
            }
        }
        let nnz = values.len();
        Ok(SparseTensor {
            indices: vec![row_indices, col_indices],
            values,
            shape: vec![m, n],
            nnz,
        })
    }

    /// Multiply a sparse 2D tensor by a dense [`Tensor`] (matrix-matrix product).
    ///
    /// `self` is `(M, K)` and `dense` is `(K, N)`. Returns a dense `(M, N)`
    /// tensor.
    ///
    /// # Errors
    ///
    /// Returns an error if `self` is not 2D, `dense` is not 2D, or inner
    /// dimensions do not match.
    pub fn sparse_dense_matmul(&self, dense: &Tensor<T>) -> Result<Tensor<T>> {
        if self.ndim() != 2 {
            return Err(CoreError::InvalidArgument {
                reason: "sparse_dense_matmul requires a 2D sparse tensor",
            });
        }
        if dense.ndim() != 2 {
            return Err(CoreError::InvalidArgument {
                reason: "sparse_dense_matmul requires a 2D dense tensor",
            });
        }
        let m = self.shape[0];
        let k_self = self.shape[1];
        let k_dense = dense.shape()[0];
        let n = dense.shape()[1];
        if k_self != k_dense {
            return Err(CoreError::DimensionMismatch {
                expected: self.shape.clone(),
                got: dense.shape().to_vec(),
            });
        }

        let mut result = Tensor::zeros(vec![m, n]);
        let result_data = result.as_mut_slice();
        let dense_data = dense.as_slice();

        for k in 0..self.nnz {
            let row = self.indices[0][k];
            let col = self.indices[1][k];
            let val = self.values[k];
            for j in 0..n {
                let idx = row * n + j;
                result_data[idx] += val * dense_data[col * n + j];
            }
        }

        Ok(result)
    }

    /// Transpose a 2D sparse tensor (swap row and column indices).
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::InvalidArgument`] if the tensor is not 2D.
    pub fn transpose(&self) -> Result<SparseTensor<T>> {
        if self.ndim() != 2 {
            return Err(CoreError::InvalidArgument {
                reason: "transpose requires a 2D sparse tensor",
            });
        }
        Ok(SparseTensor {
            indices: vec![self.indices[1].clone(), self.indices[0].clone()],
            values: self.values.clone(),
            shape: vec![self.shape[1], self.shape[0]],
            nnz: self.nnz,
        })
    }

    // ------------------------------------------------------------------
    // Utility
    // ------------------------------------------------------------------

    /// Merge duplicate indices by summing their values, and sort entries
    /// in lexicographic order of their coordinates.
    ///
    /// After coalescing, every coordinate tuple appears at most once and entries
    /// with a value of zero are removed.
    pub fn coalesce(&mut self) {
        if self.nnz == 0 {
            return;
        }
        let ndim = self.shape.len();

        // Build a (coordinate-tuple -> summed value) map.
        let mut map: HashMap<Vec<usize>, T> = HashMap::new();
        for k in 0..self.nnz {
            let coord: Vec<usize> = (0..ndim).map(|d| self.indices[d][k]).collect();
            let entry = map.entry(coord).or_insert(T::zero());
            *entry += self.values[k];
        }

        // Sort coordinates lexicographically.
        let mut entries: Vec<(Vec<usize>, T)> =
            map.into_iter().filter(|(_, v)| *v != T::zero()).collect();
        entries.sort_by(|(a, _), (b, _)| a.cmp(b));

        // Rebuild storage.
        let new_nnz = entries.len();
        let mut new_indices: Vec<Vec<usize>> = vec![Vec::with_capacity(new_nnz); ndim];
        let mut new_values = Vec::with_capacity(new_nnz);
        for (coord, val) in entries {
            for (dim, &c) in coord.iter().enumerate() {
                new_indices[dim].push(c);
            }
            new_values.push(val);
        }

        self.indices = new_indices;
        self.values = new_values;
        self.nnz = new_nnz;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_from_dense_roundtrip() {
        // 2x3 matrix with some zeros.
        let dense = Tensor::from_vec(vec![0.0_f64, 1.0, 0.0, 2.0, 0.0, 3.0], vec![2, 3]).unwrap();
        let sparse = SparseTensor::from_dense(&dense);
        assert_eq!(sparse.nnz(), 3);
        let recovered = sparse.to_dense().unwrap();
        assert_eq!(dense, recovered);
    }

    #[test]
    fn test_sparse_matmul() {
        // A = [[1, 2], [0, 3]]  B = [[4, 0], [5, 6]]
        // A*B = [[14, 12], [15, 18]]
        let a = SparseTensor::new(
            vec![vec![0, 0, 1], vec![0, 1, 1]],
            vec![1.0_f64, 2.0, 3.0],
            vec![2, 2],
        )
        .unwrap();
        let b = SparseTensor::new(
            vec![vec![0, 1, 1], vec![0, 0, 1]],
            vec![4.0_f64, 5.0, 6.0],
            vec![2, 2],
        )
        .unwrap();
        let mut c = a.sparse_matmul(&b).unwrap();
        c.coalesce();
        let dense_c = c.to_dense().unwrap();
        let expected = Tensor::from_vec(vec![14.0, 12.0, 15.0, 18.0], vec![2, 2]).unwrap();
        assert_eq!(dense_c, expected);
    }

    #[test]
    fn test_sparse_dense_matmul() {
        // A (sparse) = [[1, 0], [0, 2]]  B (dense) = [[3, 4], [5, 6]]
        // A*B = [[3, 4], [10, 12]]
        let a = SparseTensor::new(vec![vec![0, 1], vec![0, 1]], vec![1.0_f64, 2.0], vec![2, 2])
            .unwrap();
        let b = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]).unwrap();
        let result = a.sparse_dense_matmul(&b).unwrap();
        let expected = Tensor::from_vec(vec![3.0, 4.0, 10.0, 12.0], vec![2, 2]).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_sparse_add() {
        let a = SparseTensor::new(vec![vec![0, 1], vec![0, 1]], vec![1.0_f64, 2.0], vec![2, 2])
            .unwrap();
        let b = SparseTensor::new(vec![vec![0, 1], vec![1, 0]], vec![3.0_f64, 4.0], vec![2, 2])
            .unwrap();
        let mut c = a.add(&b).unwrap();
        c.coalesce();
        let dense_c = c.to_dense().unwrap();
        // Result: [[1, 3], [4, 2]]
        let expected = Tensor::from_vec(vec![1.0, 3.0, 4.0, 2.0], vec![2, 2]).unwrap();
        assert_eq!(dense_c, expected);
    }

    #[test]
    fn test_scalar_mul() {
        let st = SparseTensor::new(vec![vec![0, 1], vec![0, 1]], vec![2.0_f64, 3.0], vec![2, 2])
            .unwrap();
        let scaled = st.scalar_mul(10.0);
        assert_eq!(scaled.values(), &[20.0, 30.0]);
        let dense = scaled.to_dense().unwrap();
        let expected = Tensor::from_vec(vec![20.0, 0.0, 0.0, 30.0], vec![2, 2]).unwrap();
        assert_eq!(dense, expected);
    }

    #[test]
    fn test_coalesce() {
        // Duplicate index (0,0) with values 1.0 and 2.0 should merge to 3.0.
        let mut st = SparseTensor::new(
            vec![vec![0, 0, 1], vec![0, 0, 1]],
            vec![1.0_f64, 2.0, 5.0],
            vec![2, 2],
        )
        .unwrap();
        assert_eq!(st.nnz(), 3);
        st.coalesce();
        assert_eq!(st.nnz(), 2);
        let dense = st.to_dense().unwrap();
        let expected = Tensor::from_vec(vec![3.0, 0.0, 0.0, 5.0], vec![2, 2]).unwrap();
        assert_eq!(dense, expected);
    }

    #[test]
    fn test_transpose() {
        // [[1, 0], [0, 2]] -> [[1, 0], [0, 2]] (diagonal, same result)
        // Use non-symmetric to verify: [[0, 3], [0, 0]]
        let st = SparseTensor::new(vec![vec![0], vec![1]], vec![3.0_f64], vec![2, 2]).unwrap();
        let tr = st.transpose().unwrap();
        assert_eq!(tr.shape(), &[2, 2]);
        // Transposed: value 3.0 moves from (0,1) to (1,0).
        assert_eq!(tr.indices()[0], vec![1]); // row indices
        assert_eq!(tr.indices()[1], vec![0]); // col indices
        let dense = tr.to_dense().unwrap();
        let expected = Tensor::from_vec(vec![0.0, 0.0, 3.0, 0.0], vec![2, 2]).unwrap();
        assert_eq!(dense, expected);
    }
}
