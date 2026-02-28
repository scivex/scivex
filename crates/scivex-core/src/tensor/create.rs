//! Tensor creation functions analogous to `np.zeros`, `np.ones`, etc.

use crate::error::{CoreError, Result};
use crate::{Float, Scalar};

use super::{Tensor, compute_strides};

impl<T: Scalar> Tensor<T> {
    /// Create a tensor filled with zeros.
    ///
    /// ```
    /// # use scivex_core::tensor::Tensor;
    /// let t = Tensor::<f64>::zeros(vec![2, 3]);
    /// assert_eq!(t.shape(), &[2, 3]);
    /// assert!(t.iter().all(|&x| x == 0.0));
    /// ```
    pub fn zeros(shape: Vec<usize>) -> Self {
        let numel: usize = shape.iter().product();
        let strides = compute_strides(&shape);
        Self {
            data: vec![T::zero(); numel],
            shape,
            strides,
        }
    }

    /// Create a tensor filled with ones.
    pub fn ones(shape: Vec<usize>) -> Self {
        let numel: usize = shape.iter().product();
        let strides = compute_strides(&shape);
        Self {
            data: vec![T::one(); numel],
            shape,
            strides,
        }
    }

    /// Create a tensor filled with a constant value.
    pub fn full(shape: Vec<usize>, value: T) -> Self {
        let numel: usize = shape.iter().product();
        let strides = compute_strides(&shape);
        Self {
            data: vec![value; numel],
            shape,
            strides,
        }
    }

    /// Create a 1-D tensor with values `[0, 1, 2, ..., n-1]`.
    ///
    /// ```
    /// # use scivex_core::tensor::Tensor;
    /// let t = Tensor::<i32>::arange(5);
    /// assert_eq!(t.as_slice(), &[0, 1, 2, 3, 4]);
    /// ```
    pub fn arange(n: usize) -> Self {
        let data: Vec<T> = (0..n).map(T::from_usize).collect();
        let strides = compute_strides(&[n]);
        Self {
            data,
            shape: vec![n],
            strides,
        }
    }

    /// Create an identity matrix of size `n x n`.
    ///
    /// ```
    /// # use scivex_core::tensor::Tensor;
    /// let eye = Tensor::<f64>::eye(3);
    /// assert_eq!(eye.shape(), &[3, 3]);
    /// assert_eq!(*eye.get(&[0, 0]).unwrap(), 1.0);
    /// assert_eq!(*eye.get(&[0, 1]).unwrap(), 0.0);
    /// ```
    pub fn eye(n: usize) -> Self {
        let mut data = vec![T::zero(); n * n];
        for i in 0..n {
            data[i * n + i] = T::one();
        }
        let strides = compute_strides(&[n, n]);
        Self {
            data,
            shape: vec![n, n],
            strides,
        }
    }
}

impl<T: Float> Tensor<T> {
    /// Create a 1-D tensor with `n` evenly spaced values from `start` to `end`
    /// (inclusive).
    ///
    /// Returns an error if `n < 2`.
    ///
    /// ```
    /// # use scivex_core::tensor::Tensor;
    /// let t = Tensor::<f64>::linspace(0.0, 1.0, 5).unwrap();
    /// assert_eq!(t.shape(), &[5]);
    /// ```
    pub fn linspace(start: T, end: T, n: usize) -> Result<Self> {
        if n < 2 {
            return Err(CoreError::InvalidArgument {
                reason: "linspace requires n >= 2",
            });
        }
        let step = (end - start) / T::from_usize(n - 1);
        let data: Vec<T> = (0..n).map(|i| start + step * T::from_usize(i)).collect();
        let strides = compute_strides(&[n]);
        Ok(Self {
            data,
            shape: vec![n],
            strides,
        })
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros() {
        let t = Tensor::<f64>::zeros(vec![3, 4]);
        assert_eq!(t.shape(), &[3, 4]);
        assert_eq!(t.numel(), 12);
        assert!(t.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_ones() {
        let t = Tensor::<f32>::ones(vec![2, 2]);
        assert!(t.iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_full() {
        let t = Tensor::full(vec![2, 3], 7_i32);
        assert!(t.iter().all(|&x| x == 7));
    }

    #[test]
    fn test_arange() {
        let t = Tensor::<i32>::arange(5);
        assert_eq!(t.as_slice(), &[0, 1, 2, 3, 4]);
        assert_eq!(t.shape(), &[5]);
    }

    #[test]
    fn test_arange_zero() {
        let t = Tensor::<i32>::arange(0);
        assert!(t.is_empty());
        assert_eq!(t.shape(), &[0]);
    }

    #[test]
    fn test_eye() {
        let t = Tensor::<f64>::eye(3);
        assert_eq!(t.shape(), &[3, 3]);
        assert_eq!(*t.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(*t.get(&[1, 1]).unwrap(), 1.0);
        assert_eq!(*t.get(&[2, 2]).unwrap(), 1.0);
        assert_eq!(*t.get(&[0, 1]).unwrap(), 0.0);
        assert_eq!(*t.get(&[1, 0]).unwrap(), 0.0);
    }

    #[test]
    fn test_linspace() {
        let t = Tensor::<f64>::linspace(0.0, 1.0, 5).unwrap();
        assert_eq!(t.shape(), &[5]);
        assert_eq!(*t.get(&[0]).unwrap(), 0.0);
        assert_eq!(*t.get(&[4]).unwrap(), 1.0);
        assert!((t.as_slice()[2] - 0.5).abs() < 1e-15);
    }

    #[test]
    fn test_linspace_invalid() {
        assert!(Tensor::<f64>::linspace(0.0, 1.0, 1).is_err());
    }
}
