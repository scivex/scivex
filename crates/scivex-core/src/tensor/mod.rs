//! N-dimensional tensor type with dynamic shape and contiguous storage.
//!
//! The [`Tensor`] type is the fundamental data structure in Scivex, analogous
//! to `NumPy`'s `ndarray`. It stores elements in row-major (C) order by default
//! and is generic over any type implementing [`Scalar`].

mod create;
mod display;
mod ops;
mod reshape;
mod sort;

pub mod einsum;
pub mod einsum_path;
pub mod indexing;
pub mod named;
pub mod sparse;

pub use indexing::SliceRange;

use crate::Scalar;
use crate::dtype::Float;
use crate::error::{CoreError, Result};

/// An N-dimensional tensor with dynamic shape.
///
/// Data is stored contiguously in row-major (C) order. The tensor owns its
/// data and cloning performs a deep copy.
///
/// # Type Parameters
///
/// - `T`: The element type, which must implement [`Scalar`].
///
/// # Examples
///
/// ```
/// # use scivex_core::Tensor;
/// let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
/// assert_eq!(t.shape(), &[2, 2]);
/// assert_eq!(t.numel(), 4);
/// ```
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct Tensor<T: Scalar> {
    data: Vec<T>,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

impl<T: Scalar> Tensor<T> {
    // ------------------------------------------------------------------
    // Construction from raw parts
    // ------------------------------------------------------------------

    /// Create a tensor from a flat data vector and a shape.
    ///
    /// Returns an error if the product of `shape` does not equal `data.len()`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_core::Tensor;
    /// let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
    /// assert_eq!(t.shape(), &[2, 3]);
    /// assert_eq!(t.numel(), 6);
    /// ```
    pub fn from_vec(data: Vec<T>, shape: Vec<usize>) -> Result<Self> {
        let numel: usize = shape.iter().product();
        if numel != data.len() {
            return Err(CoreError::InvalidShape {
                shape: shape.clone(),
                reason: "shape product does not match data length",
            });
        }
        let strides = compute_strides(&shape);
        Ok(Self {
            data,
            shape,
            strides,
        })
    }

    /// Create a tensor from a flat slice and a shape (copies the data).
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_core::Tensor;
    /// let data = [1, 2, 3, 4];
    /// let t = Tensor::from_slice(&data, vec![2, 2]).unwrap();
    /// assert_eq!(t.shape(), &[2, 2]);
    /// assert_eq!(*t.get(&[1, 0]).unwrap(), 3);
    /// ```
    pub fn from_slice(data: &[T], shape: Vec<usize>) -> Result<Self> {
        Self::from_vec(data.to_vec(), shape)
    }

    /// Create a scalar (0-dimensional) tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_core::Tensor;
    /// let t = Tensor::scalar(42.0_f64);
    /// assert_eq!(t.ndim(), 0);
    /// assert_eq!(t.numel(), 1);
    /// assert_eq!(t.as_slice(), &[42.0]);
    /// ```
    pub fn scalar(value: T) -> Self {
        Self {
            data: vec![value],
            shape: vec![],
            strides: vec![],
        }
    }

    // ------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------

    /// The shape of the tensor as a slice.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_core::Tensor;
    /// let t = Tensor::from_vec(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
    /// assert_eq!(t.shape(), &[2, 3]);
    /// ```
    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// The strides of the tensor as a slice (in number of elements).
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_core::Tensor;
    /// let t = Tensor::from_vec(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
    /// assert_eq!(t.strides(), &[3, 1]);
    /// ```
    #[inline]
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// The number of dimensions (rank) of the tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_core::Tensor;
    /// let t = Tensor::from_vec(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
    /// assert_eq!(t.ndim(), 2);
    /// ```
    #[inline]
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// The total number of elements.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_core::Tensor;
    /// let t = Tensor::from_vec(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
    /// assert_eq!(t.numel(), 6);
    /// ```
    #[inline]
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    /// Whether the tensor has zero elements.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_core::Tensor;
    /// let empty = Tensor::<f64>::zeros(vec![0]);
    /// assert!(empty.is_empty());
    /// let nonempty = Tensor::<f64>::ones(vec![3]);
    /// assert!(!nonempty.is_empty());
    /// ```
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// A flat slice of all elements in storage order.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_core::Tensor;
    /// let t = Tensor::from_vec(vec![1, 2, 3], vec![3]).unwrap();
    /// assert_eq!(t.as_slice(), &[1, 2, 3]);
    /// ```
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// A mutable flat slice of all elements in storage order.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_core::Tensor;
    /// let mut t = Tensor::from_vec(vec![1, 2, 3], vec![3]).unwrap();
    /// t.as_mut_slice()[0] = 99;
    /// assert_eq!(t.as_slice(), &[99, 2, 3]);
    /// ```
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Consume the tensor and return the underlying `Vec<T>`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_core::Tensor;
    /// let t = Tensor::from_vec(vec![1, 2, 3], vec![3]).unwrap();
    /// let v: Vec<i32> = t.into_vec();
    /// assert_eq!(v, vec![1, 2, 3]);
    /// ```
    #[inline]
    pub fn into_vec(self) -> Vec<T> {
        self.data
    }

    // ------------------------------------------------------------------
    // Element access
    // ------------------------------------------------------------------

    /// Compute the flat index for a multi-dimensional index.
    fn flat_index(&self, index: &[usize]) -> Result<usize> {
        if index.len() != self.ndim() {
            return Err(CoreError::IndexOutOfBounds {
                index: index.to_vec(),
                shape: self.shape.clone(),
            });
        }
        let mut flat = 0;
        for (i, (&idx, &dim)) in index.iter().zip(self.shape.iter()).enumerate() {
            if idx >= dim {
                return Err(CoreError::IndexOutOfBounds {
                    index: index.to_vec(),
                    shape: self.shape.clone(),
                });
            }
            flat += idx * self.strides[i];
        }
        Ok(flat)
    }

    /// Get a reference to the element at the given multi-dimensional index.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_core::Tensor;
    /// let t = Tensor::from_vec(vec![10, 20, 30, 40], vec![2, 2]).unwrap();
    /// assert_eq!(*t.get(&[0, 1]).unwrap(), 20);
    /// assert_eq!(*t.get(&[1, 0]).unwrap(), 30);
    /// ```
    pub fn get(&self, index: &[usize]) -> Result<&T> {
        let flat = self.flat_index(index)?;
        Ok(&self.data[flat])
    }

    /// Get a mutable reference to the element at the given index.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_core::Tensor;
    /// let mut t = Tensor::from_vec(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
    /// *t.get_mut(&[0, 0]).unwrap() = 42;
    /// assert_eq!(*t.get(&[0, 0]).unwrap(), 42);
    /// ```
    pub fn get_mut(&mut self, index: &[usize]) -> Result<&mut T> {
        let flat = self.flat_index(index)?;
        Ok(&mut self.data[flat])
    }

    /// Set the element at the given multi-dimensional index.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_core::Tensor;
    /// let mut t = Tensor::from_vec(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
    /// t.set(&[0, 1], 99).unwrap();
    /// assert_eq!(*t.get(&[0, 1]).unwrap(), 99);
    /// ```
    pub fn set(&mut self, index: &[usize], value: T) -> Result<()> {
        let flat = self.flat_index(index)?;
        self.data[flat] = value;
        Ok(())
    }

    // ------------------------------------------------------------------
    // Iterators
    // ------------------------------------------------------------------

    /// Iterate over all elements in storage order.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_core::Tensor;
    /// let t = Tensor::from_vec(vec![10, 20, 30], vec![3]).unwrap();
    /// let sum: i32 = t.iter().sum();
    /// assert_eq!(sum, 60);
    /// ```
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.data.iter()
    }

    /// Iterate mutably over all elements in storage order.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_core::Tensor;
    /// let mut t = Tensor::from_vec(vec![1, 2, 3], vec![3]).unwrap();
    /// for x in t.iter_mut() {
    ///     *x *= 10;
    /// }
    /// assert_eq!(t.as_slice(), &[10, 20, 30]);
    /// ```
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.data.iter_mut()
    }

    // ------------------------------------------------------------------
    // Map / apply
    // ------------------------------------------------------------------

    /// Apply a function to every element, returning a new tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_core::Tensor;
    /// let t = Tensor::from_vec(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
    /// let doubled = t.map(|x| x * 2);
    /// assert_eq!(doubled.as_slice(), &[2, 4, 6, 8]);
    /// ```
    pub fn map<F>(&self, f: F) -> Tensor<T>
    where
        F: Fn(T) -> T,
    {
        Tensor {
            data: self.data.iter().map(|&x| f(x)).collect(),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        }
    }

    /// Cast every element to a different scalar type, preserving shape.
    ///
    /// Uses `to_f64()` / `from_f64()` for the conversion, which is lossless for
    /// f32→f64 and lossy (but intentionally so) for f64→f32 or f32→f16.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_core::Tensor;
    /// let t = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0], vec![3]).unwrap();
    /// let t32: Tensor<f32> = t.cast();
    /// assert_eq!(t32.as_slice(), &[1.0_f32, 2.0, 3.0]);
    /// ```
    pub fn cast<U: Scalar + Float>(&self) -> Tensor<U>
    where
        T: Float,
    {
        Tensor {
            data: self.data.iter().map(|&x| U::from_f64(x.to_f64())).collect(),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        }
    }

    /// Apply a function element-wise to two tensors of the same shape.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_core::Tensor;
    /// let a = Tensor::from_vec(vec![1, 2, 3], vec![3]).unwrap();
    /// let b = Tensor::from_vec(vec![10, 20, 30], vec![3]).unwrap();
    /// let c = a.zip_map(&b, |x, y| x + y).unwrap();
    /// assert_eq!(c.as_slice(), &[11, 22, 33]);
    /// ```
    pub fn zip_map<F>(&self, other: &Tensor<T>, f: F) -> Result<Tensor<T>>
    where
        F: Fn(T, T) -> T,
    {
        if self.shape != other.shape {
            return Err(CoreError::DimensionMismatch {
                expected: self.shape.clone(),
                got: other.shape.clone(),
            });
        }
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| f(a, b))
            .collect();
        Ok(Tensor {
            data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        })
    }

    /// Apply a function to every element in place.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_core::Tensor;
    /// let mut t = Tensor::from_vec(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
    /// t.apply(|x| x * x);
    /// assert_eq!(t.as_slice(), &[1, 4, 9, 16]);
    /// ```
    pub fn apply<F>(&mut self, f: F)
    where
        F: Fn(T) -> T,
    {
        for x in &mut self.data {
            *x = f(*x);
        }
    }
}

impl<T: Scalar> PartialEq for Tensor<T> {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape && self.data == other.data
    }
}

// ======================================================================
// Utility functions
// ======================================================================

/// Compute row-major (C-order) strides from a shape.
pub(crate) fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let ndim = shape.len();
    if ndim == 0 {
        return vec![];
    }
    let mut strides = vec![1usize; ndim];
    for i in (0..ndim - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_vec() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.strides(), &[3, 1]);
        assert_eq!(t.ndim(), 2);
        assert_eq!(t.numel(), 6);
    }

    #[test]
    fn test_from_vec_shape_mismatch() {
        let r = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![2, 3]);
        assert!(r.is_err());
    }

    #[test]
    fn test_scalar_tensor() {
        let t = Tensor::scalar(42.0_f64);
        assert_eq!(t.ndim(), 0);
        assert_eq!(t.numel(), 1);
        assert_eq!(t.as_slice(), &[42.0]);
    }

    #[test]
    fn test_get_set() {
        let mut t = Tensor::from_vec(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
        assert_eq!(*t.get(&[0, 0]).unwrap(), 1);
        assert_eq!(*t.get(&[1, 2]).unwrap(), 6);
        t.set(&[0, 1], 99).unwrap();
        assert_eq!(*t.get(&[0, 1]).unwrap(), 99);
    }

    #[test]
    fn test_get_out_of_bounds() {
        let t = Tensor::from_vec(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
        assert!(t.get(&[2, 0]).is_err());
        assert!(t.get(&[0]).is_err());
    }

    #[test]
    fn test_compute_strides() {
        assert_eq!(compute_strides(&[2, 3, 4]), vec![12, 4, 1]);
        assert_eq!(compute_strides(&[5]), vec![1]);
        assert_eq!(compute_strides(&[]), Vec::<usize>::new());
    }

    #[test]
    fn test_map() {
        let t = Tensor::from_vec(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
        let t2 = t.map(|x| x * 10);
        assert_eq!(t2.as_slice(), &[10, 20, 30, 40]);
        assert_eq!(t2.shape(), &[2, 2]);
    }

    #[test]
    fn test_zip_map() {
        let a = Tensor::from_vec(vec![1, 2, 3], vec![3]).unwrap();
        let b = Tensor::from_vec(vec![10, 20, 30], vec![3]).unwrap();
        let c = a.zip_map(&b, |x, y| x + y).unwrap();
        assert_eq!(c.as_slice(), &[11, 22, 33]);
    }

    #[test]
    fn test_zip_map_shape_mismatch() {
        let a = Tensor::from_vec(vec![1, 2, 3], vec![3]).unwrap();
        let b = Tensor::from_vec(vec![1, 2], vec![2]).unwrap();
        assert!(a.zip_map(&b, |x, y| x + y).is_err());
    }

    #[test]
    fn test_partial_eq() {
        let a = Tensor::from_vec(vec![1, 2, 3], vec![3]).unwrap();
        let b = Tensor::from_vec(vec![1, 2, 3], vec![3]).unwrap();
        let c = Tensor::from_vec(vec![1, 2, 4], vec![3]).unwrap();
        assert_eq!(a, b);
        assert_ne!(a, c);
    }
}
