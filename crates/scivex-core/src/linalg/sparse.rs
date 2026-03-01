//! Sparse matrix formats: COO, CSR, CSC.
//!
//! Three standard sparse representations with conversions between them and
//! interoperation with dense [`Tensor`] for I/O and matrix-vector products.

use crate::Scalar;
use crate::error::{CoreError, Result};
use crate::tensor::Tensor;

// ======================================================================
// COO (Coordinate) format
// ======================================================================

/// Sparse matrix in COO (coordinate / triplet) format.
///
/// Stores (row, col, value) triplets. Duplicate entries are summed during
/// conversion to CSR/CSC.
#[derive(Debug, Clone)]
pub struct CooMatrix<T: Scalar> {
    rows: Vec<usize>,
    cols: Vec<usize>,
    values: Vec<T>,
    nrows: usize,
    ncols: usize,
}

impl<T: Scalar> CooMatrix<T> {
    /// Create an empty COO matrix with the given dimensions.
    pub fn new(nrows: usize, ncols: usize) -> Self {
        Self {
            rows: Vec::new(),
            cols: Vec::new(),
            values: Vec::new(),
            nrows,
            ncols,
        }
    }

    /// Build a COO matrix from triplet arrays.
    pub fn from_triplets(
        nrows: usize,
        ncols: usize,
        rows: Vec<usize>,
        cols: Vec<usize>,
        values: Vec<T>,
    ) -> Result<Self> {
        if rows.len() != cols.len() || rows.len() != values.len() {
            return Err(CoreError::InvalidArgument {
                reason: "rows, cols, and values must have the same length",
            });
        }
        for (&r, &c) in rows.iter().zip(cols.iter()) {
            if r >= nrows || c >= ncols {
                return Err(CoreError::InvalidArgument {
                    reason: "index out of bounds for matrix dimensions",
                });
            }
        }
        Ok(Self {
            rows,
            cols,
            values,
            nrows,
            ncols,
        })
    }

    /// Append a single entry.
    pub fn push(&mut self, row: usize, col: usize, value: T) -> Result<()> {
        if row >= self.nrows || col >= self.ncols {
            return Err(CoreError::InvalidArgument {
                reason: "index out of bounds for matrix dimensions",
            });
        }
        self.rows.push(row);
        self.cols.push(col);
        self.values.push(value);
        Ok(())
    }

    /// Number of rows.
    #[inline]
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Number of columns.
    #[inline]
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Number of stored entries (may include duplicates).
    #[inline]
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Shape as `(nrows, ncols)`.
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }

    /// Convert to a dense 2-D tensor. Duplicate entries are summed.
    pub fn to_dense(&self) -> Tensor<T> {
        let mut data = vec![T::zero(); self.nrows * self.ncols];
        for ((&r, &c), &v) in self
            .rows
            .iter()
            .zip(self.cols.iter())
            .zip(self.values.iter())
        {
            data[r * self.ncols + c] += v;
        }
        Tensor::from_vec(data, vec![self.nrows, self.ncols]).unwrap()
    }

    /// Convert to CSR format. Duplicate entries at the same position are summed.
    pub fn to_csr(&self) -> CsrMatrix<T> {
        // Count entries per row
        let mut row_counts = vec![0usize; self.nrows + 1];
        for &r in &self.rows {
            row_counts[r + 1] += 1;
        }
        // Prefix sum -> row_ptr
        for i in 1..=self.nrows {
            row_counts[i] += row_counts[i - 1];
        }

        let nnz = self.values.len();
        let mut col_idx = vec![0usize; nnz];
        let mut values = vec![T::zero(); nnz];
        let mut offset = row_counts.clone();

        for ((&r, &c), &v) in self
            .rows
            .iter()
            .zip(self.cols.iter())
            .zip(self.values.iter())
        {
            let pos = offset[r];
            col_idx[pos] = c;
            values[pos] = v;
            offset[r] += 1;
        }

        // Sort within each row by column index and sum duplicates
        let mut result = CsrMatrix {
            row_ptr: row_counts,
            col_idx,
            values,
            nrows: self.nrows,
            ncols: self.ncols,
        };
        result.sort_and_sum_duplicates();
        result
    }

    /// Convert to CSC format. Duplicate entries at the same position are summed.
    pub fn to_csc(&self) -> CscMatrix<T> {
        // Count entries per column
        let mut col_counts = vec![0usize; self.ncols + 1];
        for &c in &self.cols {
            col_counts[c + 1] += 1;
        }
        for i in 1..=self.ncols {
            col_counts[i] += col_counts[i - 1];
        }

        let nnz = self.values.len();
        let mut row_idx = vec![0usize; nnz];
        let mut values = vec![T::zero(); nnz];
        let mut offset = col_counts.clone();

        for ((&r, &c), &v) in self
            .rows
            .iter()
            .zip(self.cols.iter())
            .zip(self.values.iter())
        {
            let pos = offset[c];
            row_idx[pos] = r;
            values[pos] = v;
            offset[c] += 1;
        }

        let mut result = CscMatrix {
            col_ptr: col_counts,
            row_idx,
            values,
            nrows: self.nrows,
            ncols: self.ncols,
        };
        result.sort_and_sum_duplicates();
        result
    }
}

// ======================================================================
// CSR (Compressed Sparse Row) format
// ======================================================================

/// Sparse matrix in CSR (Compressed Sparse Row) format.
#[derive(Debug, Clone)]
pub struct CsrMatrix<T: Scalar> {
    row_ptr: Vec<usize>,
    col_idx: Vec<usize>,
    values: Vec<T>,
    nrows: usize,
    ncols: usize,
}

impl<T: Scalar> CsrMatrix<T> {
    /// Create an empty CSR matrix.
    pub fn new(nrows: usize, ncols: usize) -> Self {
        Self {
            row_ptr: vec![0; nrows + 1],
            col_idx: Vec::new(),
            values: Vec::new(),
            nrows,
            ncols,
        }
    }

    /// Build CSR from triplet data.
    pub fn from_triplets(
        nrows: usize,
        ncols: usize,
        rows: Vec<usize>,
        cols: Vec<usize>,
        values: Vec<T>,
    ) -> Result<Self> {
        let coo = CooMatrix::from_triplets(nrows, ncols, rows, cols, values)?;
        Ok(coo.to_csr())
    }

    /// Build CSR from a dense 2-D tensor, dropping zero entries.
    pub fn from_dense(tensor: &Tensor<T>) -> Result<Self> {
        if tensor.ndim() != 2 {
            return Err(CoreError::InvalidArgument {
                reason: "from_dense requires a 2-D tensor",
            });
        }
        let nrows = tensor.shape()[0];
        let ncols = tensor.shape()[1];
        let data = tensor.as_slice();

        let mut row_ptr = vec![0usize; nrows + 1];
        let mut col_idx = Vec::new();
        let mut values = Vec::new();

        for r in 0..nrows {
            for c in 0..ncols {
                let v = data[r * ncols + c];
                if v != T::zero() {
                    col_idx.push(c);
                    values.push(v);
                }
            }
            row_ptr[r + 1] = values.len();
        }

        Ok(Self {
            row_ptr,
            col_idx,
            values,
            nrows,
            ncols,
        })
    }

    /// Number of rows.
    #[inline]
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Number of columns.
    #[inline]
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Number of stored non-zero entries.
    #[inline]
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Shape as `(nrows, ncols)`.
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }

    /// Get the value at `(row, col)`, or `None` if not stored.
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        if row >= self.nrows || col >= self.ncols {
            return None;
        }
        let start = self.row_ptr[row];
        let end = self.row_ptr[row + 1];
        self.col_idx[start..end]
            .binary_search(&col)
            .ok()
            .map(|pos| &self.values[start + pos])
    }

    /// Convert to a dense 2-D tensor.
    pub fn to_dense(&self) -> Tensor<T> {
        let mut data = vec![T::zero(); self.nrows * self.ncols];
        for r in 0..self.nrows {
            let start = self.row_ptr[r];
            let end = self.row_ptr[r + 1];
            for idx in start..end {
                let c = self.col_idx[idx];
                data[r * self.ncols + c] = self.values[idx];
            }
        }
        Tensor::from_vec(data, vec![self.nrows, self.ncols]).unwrap()
    }

    /// Sparse matrix × dense vector multiplication.
    ///
    /// `x` must be a 1-D tensor of length `ncols`.
    pub fn matvec(&self, x: &Tensor<T>) -> Result<Tensor<T>> {
        if x.ndim() != 1 || x.numel() != self.ncols {
            return Err(CoreError::DimensionMismatch {
                expected: vec![self.ncols],
                got: x.shape().to_vec(),
            });
        }
        let xdata = x.as_slice();
        let mut result = vec![T::zero(); self.nrows];

        for (r, dest) in result.iter_mut().enumerate() {
            let start = self.row_ptr[r];
            let end = self.row_ptr[r + 1];
            let mut acc = T::zero();
            for idx in start..end {
                acc += self.values[idx] * xdata[self.col_idx[idx]];
            }
            *dest = acc;
        }

        Tensor::from_vec(result, vec![self.nrows])
    }

    /// Transpose, returning a CSC matrix.
    pub fn transpose(&self) -> CscMatrix<T> {
        CscMatrix {
            col_ptr: self.row_ptr.clone(),
            row_idx: self.col_idx.clone(),
            values: self.values.clone(),
            nrows: self.ncols,
            ncols: self.nrows,
        }
    }

    /// Convert to COO format.
    pub fn to_coo(&self) -> CooMatrix<T> {
        let mut rows = Vec::with_capacity(self.nnz());
        let mut cols = Vec::with_capacity(self.nnz());
        let mut values = Vec::with_capacity(self.nnz());

        for r in 0..self.nrows {
            let start = self.row_ptr[r];
            let end = self.row_ptr[r + 1];
            for idx in start..end {
                rows.push(r);
                cols.push(self.col_idx[idx]);
                values.push(self.values[idx]);
            }
        }

        CooMatrix {
            rows,
            cols,
            values,
            nrows: self.nrows,
            ncols: self.ncols,
        }
    }

    /// Convert to CSC format.
    pub fn to_csc(&self) -> CscMatrix<T> {
        self.to_coo().to_csc()
    }

    /// Sort column indices within each row and sum duplicate entries.
    fn sort_and_sum_duplicates(&mut self) {
        for r in 0..self.nrows {
            let start = self.row_ptr[r];
            let end = self.row_ptr[r + 1];
            if start == end {
                continue;
            }

            // Sort by column index using a permutation
            let len = end - start;
            let mut perm: Vec<usize> = (0..len).collect();
            perm.sort_unstable_by_key(|&i| self.col_idx[start + i]);

            let old_cols: Vec<usize> = self.col_idx[start..end].to_vec();
            let old_vals: Vec<T> = self.values[start..end].to_vec();
            for (j, &p) in perm.iter().enumerate() {
                self.col_idx[start + j] = old_cols[p];
                self.values[start + j] = old_vals[p];
            }

            // Sum duplicates in-place
            let mut write = start;
            for read in (start + 1)..end {
                if self.col_idx[read] == self.col_idx[write] {
                    let v = self.values[read];
                    self.values[write] += v;
                } else {
                    write += 1;
                    self.col_idx[write] = self.col_idx[read];
                    self.values[write] = self.values[read];
                }
            }
            let new_end = write + 1;

            // Shift subsequent data if duplicates were removed
            if new_end < end {
                let removed = end - new_end;
                let total_nnz = self.col_idx.len();
                self.col_idx.copy_within(end..total_nnz, new_end);
                self.col_idx.truncate(total_nnz - removed);
                let total_vals = self.values.len();
                self.values.copy_within(end..total_vals, new_end);
                self.values.truncate(total_vals - removed);

                for i in (r + 1)..=self.nrows {
                    self.row_ptr[i] -= removed;
                }
            }
        }
    }
}

// ======================================================================
// CSC (Compressed Sparse Column) format
// ======================================================================

/// Sparse matrix in CSC (Compressed Sparse Column) format.
#[derive(Debug, Clone)]
pub struct CscMatrix<T: Scalar> {
    col_ptr: Vec<usize>,
    row_idx: Vec<usize>,
    values: Vec<T>,
    nrows: usize,
    ncols: usize,
}

impl<T: Scalar> CscMatrix<T> {
    /// Create an empty CSC matrix.
    pub fn new(nrows: usize, ncols: usize) -> Self {
        Self {
            col_ptr: vec![0; ncols + 1],
            row_idx: Vec::new(),
            values: Vec::new(),
            nrows,
            ncols,
        }
    }

    /// Build CSC from triplet data.
    pub fn from_triplets(
        nrows: usize,
        ncols: usize,
        rows: Vec<usize>,
        cols: Vec<usize>,
        values: Vec<T>,
    ) -> Result<Self> {
        let coo = CooMatrix::from_triplets(nrows, ncols, rows, cols, values)?;
        Ok(coo.to_csc())
    }

    /// Number of rows.
    #[inline]
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Number of columns.
    #[inline]
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Number of stored non-zero entries.
    #[inline]
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Shape as `(nrows, ncols)`.
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }

    /// Convert to a dense 2-D tensor.
    pub fn to_dense(&self) -> Tensor<T> {
        let mut data = vec![T::zero(); self.nrows * self.ncols];
        for c in 0..self.ncols {
            let start = self.col_ptr[c];
            let end = self.col_ptr[c + 1];
            for idx in start..end {
                let r = self.row_idx[idx];
                data[r * self.ncols + c] = self.values[idx];
            }
        }
        Tensor::from_vec(data, vec![self.nrows, self.ncols]).unwrap()
    }

    /// Sparse matrix × dense vector multiplication.
    ///
    /// `x` must be a 1-D tensor of length `ncols`.
    pub fn matvec(&self, x: &Tensor<T>) -> Result<Tensor<T>> {
        if x.ndim() != 1 || x.numel() != self.ncols {
            return Err(CoreError::DimensionMismatch {
                expected: vec![self.ncols],
                got: x.shape().to_vec(),
            });
        }
        let xdata = x.as_slice();
        let mut result = vec![T::zero(); self.nrows];

        for (c, &xc) in xdata.iter().enumerate().take(self.ncols) {
            let start = self.col_ptr[c];
            let end = self.col_ptr[c + 1];
            for idx in start..end {
                result[self.row_idx[idx]] += self.values[idx] * xc;
            }
        }

        Tensor::from_vec(result, vec![self.nrows])
    }

    /// Transpose, returning a CSR matrix.
    pub fn transpose(&self) -> CsrMatrix<T> {
        CsrMatrix {
            row_ptr: self.col_ptr.clone(),
            col_idx: self.row_idx.clone(),
            values: self.values.clone(),
            nrows: self.ncols,
            ncols: self.nrows,
        }
    }

    /// Convert to COO format.
    pub fn to_coo(&self) -> CooMatrix<T> {
        let mut rows = Vec::with_capacity(self.nnz());
        let mut cols = Vec::with_capacity(self.nnz());
        let mut values = Vec::with_capacity(self.nnz());

        for c in 0..self.ncols {
            let start = self.col_ptr[c];
            let end = self.col_ptr[c + 1];
            for idx in start..end {
                rows.push(self.row_idx[idx]);
                cols.push(c);
                values.push(self.values[idx]);
            }
        }

        CooMatrix {
            rows,
            cols,
            values,
            nrows: self.nrows,
            ncols: self.ncols,
        }
    }

    /// Convert to CSR format.
    pub fn to_csr(&self) -> CsrMatrix<T> {
        self.to_coo().to_csr()
    }

    /// Sort row indices within each column and sum duplicate entries.
    fn sort_and_sum_duplicates(&mut self) {
        for c in 0..self.ncols {
            let start = self.col_ptr[c];
            let end = self.col_ptr[c + 1];
            if start == end {
                continue;
            }

            let len = end - start;
            let mut perm: Vec<usize> = (0..len).collect();
            perm.sort_unstable_by_key(|&i| self.row_idx[start + i]);

            let old_rows: Vec<usize> = self.row_idx[start..end].to_vec();
            let old_vals: Vec<T> = self.values[start..end].to_vec();
            for (j, &p) in perm.iter().enumerate() {
                self.row_idx[start + j] = old_rows[p];
                self.values[start + j] = old_vals[p];
            }

            // Sum duplicates
            let mut write = start;
            for read in (start + 1)..end {
                if self.row_idx[read] == self.row_idx[write] {
                    let v = self.values[read];
                    self.values[write] += v;
                } else {
                    write += 1;
                    self.row_idx[write] = self.row_idx[read];
                    self.values[write] = self.values[read];
                }
            }
            let new_end = write + 1;

            if new_end < end {
                let removed = end - new_end;
                let total_idx = self.row_idx.len();
                self.row_idx.copy_within(end..total_idx, new_end);
                self.row_idx.truncate(total_idx - removed);
                let total_vals = self.values.len();
                self.values.copy_within(end..total_vals, new_end);
                self.values.truncate(total_vals - removed);

                for i in (c + 1)..=self.ncols {
                    self.col_ptr[i] -= removed;
                }
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    // Helper: 3x3 matrix
    // [[1, 0, 2],
    //  [0, 3, 0],
    //  [4, 0, 5]]
    fn sample_coo() -> CooMatrix<f64> {
        CooMatrix::from_triplets(
            3,
            3,
            vec![0, 0, 1, 2, 2],
            vec![0, 2, 1, 0, 2],
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
        )
        .unwrap()
    }

    #[test]
    fn test_coo_to_dense() {
        let coo = sample_coo();
        let dense = coo.to_dense();
        assert_eq!(dense.shape(), &[3, 3]);
        assert_eq!(
            dense.as_slice(),
            &[1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0]
        );
    }

    #[test]
    fn test_csr_from_dense_roundtrip() {
        let dense = Tensor::from_vec(
            vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0],
            vec![3, 3],
        )
        .unwrap();
        let csr = CsrMatrix::from_dense(&dense).unwrap();
        assert_eq!(csr.nnz(), 5);
        let back = csr.to_dense();
        assert_eq!(dense, back);
    }

    #[test]
    fn test_csr_matvec() {
        let csr = sample_coo().to_csr();
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let y = csr.matvec(&x).unwrap();
        // [1*1 + 0*2 + 2*3, 0*1 + 3*2 + 0*3, 4*1 + 0*2 + 5*3] = [7, 6, 19]
        assert_eq!(y.as_slice(), &[7.0, 6.0, 19.0]);
    }

    #[test]
    fn test_csc_matvec() {
        let csc = sample_coo().to_csc();
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let y = csc.matvec(&x).unwrap();
        assert_eq!(y.as_slice(), &[7.0, 6.0, 19.0]);
    }

    #[test]
    fn test_coo_csr_csc_dense_roundtrip() {
        let coo = sample_coo();
        let expected = coo.to_dense();

        let csr = coo.to_csr();
        assert_eq!(csr.to_dense(), expected);

        let csc = csr.to_csc();
        assert_eq!(csc.to_dense(), expected);

        let coo2 = csc.to_coo();
        assert_eq!(coo2.to_dense(), expected);
    }

    #[test]
    fn test_identity_matrix() {
        let csr = CsrMatrix::from_dense(&Tensor::<f64>::eye(4)).unwrap();
        assert_eq!(csr.nnz(), 4);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let y = csr.matvec(&x).unwrap();
        assert_eq!(y, x);
    }

    #[test]
    fn test_empty_matrix() {
        let csr = CsrMatrix::<f64>::new(3, 3);
        assert_eq!(csr.nnz(), 0);
        let dense = csr.to_dense();
        assert_eq!(dense, Tensor::<f64>::zeros(vec![3, 3]));
    }

    #[test]
    fn test_dimension_mismatch() {
        let csr = sample_coo().to_csr();
        let x = Tensor::from_vec(vec![1.0, 2.0], vec![2]).unwrap();
        assert!(csr.matvec(&x).is_err());
    }

    #[test]
    fn test_duplicate_coo_entries_summed() {
        // Two entries at (0, 0): 1.0 + 2.0 = 3.0
        let coo = CooMatrix::from_triplets(2, 2, vec![0, 0, 1], vec![0, 0, 1], vec![1.0, 2.0, 5.0])
            .unwrap();
        let csr = coo.to_csr();
        assert_eq!(*csr.get(0, 0).unwrap(), 3.0);
        assert_eq!(*csr.get(1, 1).unwrap(), 5.0);
        assert_eq!(csr.nnz(), 2);
    }

    #[test]
    fn test_csr_transpose() {
        let csr = sample_coo().to_csr();
        let csc = csr.transpose();
        // Transposed: nrows/ncols swap
        assert_eq!(csc.nrows(), 3);
        assert_eq!(csc.ncols(), 3);
        // The transposed matrix's dense form should be the transpose of the original
        let orig = csr.to_dense();
        let trans = csc.to_dense();
        // Check (i,j) of trans == (j,i) of orig
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(*trans.get(&[i, j]).unwrap(), *orig.get(&[j, i]).unwrap());
            }
        }
    }

    #[test]
    fn test_csr_get() {
        let csr = sample_coo().to_csr();
        assert_eq!(*csr.get(0, 0).unwrap(), 1.0);
        assert_eq!(*csr.get(0, 2).unwrap(), 2.0);
        assert!(csr.get(0, 1).is_none()); // zero entry
        assert!(csr.get(5, 0).is_none()); // out of bounds
    }

    #[test]
    fn test_coo_push() {
        let mut coo = CooMatrix::<f64>::new(2, 2);
        coo.push(0, 0, 1.0).unwrap();
        coo.push(1, 1, 2.0).unwrap();
        assert_eq!(coo.nnz(), 2);
        assert!(coo.push(2, 0, 1.0).is_err()); // out of bounds
    }
}
