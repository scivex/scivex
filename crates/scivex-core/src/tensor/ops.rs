//! Element-wise arithmetic operators for [`Tensor`].
//!
//! Implements `Add`, `Sub`, `Mul`, `Div` for:
//! - `Tensor<T> op Tensor<T>` (element-wise, same shape)
//! - `Tensor<T> op T` (broadcast scalar to every element)
//! - `Neg` for `Float` tensors

use core::ops::{Add, Div, Mul, Neg, Sub};

use crate::error::CoreError;
use crate::{Float, Scalar};

use super::Tensor;

// ======================================================================
// Tensor + Tensor  (element-wise, same shape — panics on mismatch)
// ======================================================================

macro_rules! impl_tensor_binop {
    ($trait:ident, $method:ident, $op:tt) => {
        impl<T: Scalar> $trait for Tensor<T> {
            type Output = Tensor<T>;

            fn $method(self, rhs: Tensor<T>) -> Tensor<T> {
                assert_eq!(
                    self.shape, rhs.shape,
                    "shape mismatch in element-wise {}: {:?} vs {:?}",
                    stringify!($method), self.shape, rhs.shape,
                );
                let data = self.data.iter()
                    .zip(rhs.data.iter())
                    .map(|(&a, &b)| a $op b)
                    .collect();
                Tensor {
                    data,
                    shape: self.shape,
                    strides: self.strides,
                }
            }
        }

        impl<T: Scalar> $trait for &Tensor<T> {
            type Output = Tensor<T>;

            fn $method(self, rhs: &Tensor<T>) -> Tensor<T> {
                assert_eq!(
                    self.shape, rhs.shape,
                    "shape mismatch in element-wise {}: {:?} vs {:?}",
                    stringify!($method), self.shape, rhs.shape,
                );
                let data = self.data.iter()
                    .zip(rhs.data.iter())
                    .map(|(&a, &b)| a $op b)
                    .collect();
                Tensor {
                    data,
                    shape: self.shape.clone(),
                    strides: self.strides.clone(),
                }
            }
        }
    };
}

impl_tensor_binop!(Add, add, +);
impl_tensor_binop!(Sub, sub, -);
impl_tensor_binop!(Mul, mul, *);
impl_tensor_binop!(Div, div, /);

// ======================================================================
// Tensor + scalar  (broadcast scalar to every element)
// ======================================================================

macro_rules! impl_scalar_binop {
    ($trait:ident, $method:ident, $op:tt) => {
        impl<T: Scalar> $trait<T> for Tensor<T> {
            type Output = Tensor<T>;

            fn $method(self, rhs: T) -> Tensor<T> {
                let data = self.data.iter().map(|&a| a $op rhs).collect();
                Tensor {
                    data,
                    shape: self.shape,
                    strides: self.strides,
                }
            }
        }

        impl<T: Scalar> $trait<T> for &Tensor<T> {
            type Output = Tensor<T>;

            fn $method(self, rhs: T) -> Tensor<T> {
                let data = self.data.iter().map(|&a| a $op rhs).collect();
                Tensor {
                    data,
                    shape: self.shape.clone(),
                    strides: self.strides.clone(),
                }
            }
        }
    };
}

impl_scalar_binop!(Add, add, +);
impl_scalar_binop!(Sub, sub, -);
impl_scalar_binop!(Mul, mul, *);
impl_scalar_binop!(Div, div, /);

// ======================================================================
// Negation
// ======================================================================

impl<T: Float> Neg for Tensor<T> {
    type Output = Tensor<T>;

    fn neg(self) -> Tensor<T> {
        let data = self.data.iter().map(|&a| -a).collect();
        Tensor {
            data,
            shape: self.shape,
            strides: self.strides,
        }
    }
}

impl<T: Float> Neg for &Tensor<T> {
    type Output = Tensor<T>;

    fn neg(self) -> Tensor<T> {
        let data = self.data.iter().map(|&a| -a).collect();
        Tensor {
            data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        }
    }
}

// ======================================================================
// Fallible (Result-returning) arithmetic for non-panicking callers
// ======================================================================

impl<T: Scalar> Tensor<T> {
    /// Element-wise addition, returning `Err` on shape mismatch.
    pub fn add_checked(&self, other: &Tensor<T>) -> crate::Result<Tensor<T>> {
        self.zip_map(other, |a, b| a + b)
    }

    /// Element-wise subtraction, returning `Err` on shape mismatch.
    pub fn sub_checked(&self, other: &Tensor<T>) -> crate::Result<Tensor<T>> {
        self.zip_map(other, |a, b| a - b)
    }

    /// Element-wise multiplication, returning `Err` on shape mismatch.
    pub fn mul_checked(&self, other: &Tensor<T>) -> crate::Result<Tensor<T>> {
        self.zip_map(other, |a, b| a * b)
    }

    /// Element-wise division, returning `Err` on shape mismatch.
    pub fn div_checked(&self, other: &Tensor<T>) -> crate::Result<Tensor<T>> {
        self.zip_map(other, |a, b| a / b)
    }
}

// ======================================================================
// Reductions
// ======================================================================

impl<T: Scalar> Tensor<T> {
    /// Sum of all elements.
    pub fn sum(&self) -> T {
        self.data.iter().copied().sum()
    }

    /// Product of all elements.
    pub fn product(&self) -> T {
        self.data.iter().copied().fold(T::one(), |acc, x| acc * x)
    }

    /// Minimum element. Returns `None` for empty tensors.
    pub fn min_element(&self) -> Option<T> {
        self.data
            .iter()
            .copied()
            .reduce(|a, b| if b < a { b } else { a })
    }

    /// Maximum element. Returns `None` for empty tensors.
    pub fn max_element(&self) -> Option<T> {
        self.data
            .iter()
            .copied()
            .reduce(|a, b| if b > a { b } else { a })
    }

    /// Sum along a given axis, producing a tensor with that axis removed.
    pub fn sum_axis(&self, axis: usize) -> crate::Result<Tensor<T>> {
        if axis >= self.ndim() {
            return Err(CoreError::AxisOutOfBounds {
                axis,
                ndim: self.ndim(),
            });
        }

        let mut new_shape: Vec<usize> = self.shape.clone();
        let axis_len = new_shape.remove(axis);

        // Handle reduction to scalar
        if new_shape.is_empty() {
            return Ok(Tensor::scalar(self.sum()));
        }

        let new_numel: usize = new_shape.iter().product();
        let mut result_data = vec![T::zero(); new_numel];

        let outer: usize = self.shape[..axis].iter().product();
        let inner: usize = self.shape[axis + 1..].iter().product();

        for o in 0..outer {
            for k in 0..axis_len {
                let src_offset = (o * axis_len + k) * inner;
                let dst_offset = o * inner;
                for i in 0..inner {
                    result_data[dst_offset + i] += self.data[src_offset + i];
                }
            }
        }

        Tensor::from_vec(result_data, new_shape)
    }
}

impl<T: Float> Tensor<T> {
    /// Mean of all elements.
    pub fn mean(&self) -> T {
        self.sum() / T::from_usize(self.numel())
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn test_add_tensors() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let b = Tensor::from_vec(vec![10.0, 20.0, 30.0], vec![3]).unwrap();
        let c = a + b;
        assert_eq!(c.as_slice(), &[11.0, 22.0, 33.0]);
    }

    #[test]
    fn test_sub_tensors() {
        let a = Tensor::from_vec(vec![10.0, 20.0], vec![2]).unwrap();
        let b = Tensor::from_vec(vec![1.0, 2.0], vec![2]).unwrap();
        let c = &a - &b;
        assert_eq!(c.as_slice(), &[9.0, 18.0]);
    }

    #[test]
    fn test_mul_scalar() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let c = a * 10.0;
        assert_eq!(c.as_slice(), &[10.0, 20.0, 30.0]);
    }

    #[test]
    fn test_div_scalar() {
        let a = Tensor::from_vec(vec![10.0, 20.0, 30.0], vec![3]).unwrap();
        let c = &a / 10.0;
        assert_eq!(c.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_neg() {
        let a = Tensor::from_vec(vec![1.0_f64, -2.0, 3.0], vec![3]).unwrap();
        let b = -a;
        assert_eq!(b.as_slice(), &[-1.0, 2.0, -3.0]);
    }

    #[test]
    fn test_checked_add_mismatch() {
        let a = Tensor::from_vec(vec![1.0, 2.0], vec![2]).unwrap();
        let b = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        assert!(a.add_checked(&b).is_err());
    }

    #[test]
    fn test_sum() {
        let t = Tensor::from_vec(vec![1, 2, 3, 4], vec![4]).unwrap();
        assert_eq!(t.sum(), 10);
    }

    #[test]
    fn test_product() {
        let t = Tensor::from_vec(vec![1, 2, 3, 4], vec![4]).unwrap();
        assert_eq!(t.product(), 24);
    }

    #[test]
    fn test_min_max() {
        let t = Tensor::from_vec(vec![3, 1, 4, 1, 5, 9], vec![6]).unwrap();
        assert_eq!(t.min_element(), Some(1));
        assert_eq!(t.max_element(), Some(9));
    }

    #[test]
    fn test_mean() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        assert_eq!(t.mean(), 2.5);
    }

    #[test]
    fn test_sum_axis() {
        // [[1, 2, 3],
        //  [4, 5, 6]]
        let t = Tensor::from_vec(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();

        // Sum along axis 0 -> [5, 7, 9]
        let s0 = t.sum_axis(0).unwrap();
        assert_eq!(s0.shape(), &[3]);
        assert_eq!(s0.as_slice(), &[5, 7, 9]);

        // Sum along axis 1 -> [6, 15]
        let s1 = t.sum_axis(1).unwrap();
        assert_eq!(s1.shape(), &[2]);
        assert_eq!(s1.as_slice(), &[6, 15]);
    }

    #[test]
    fn test_sum_axis_out_of_bounds() {
        let t = Tensor::from_vec(vec![1, 2, 3], vec![3]).unwrap();
        assert!(t.sum_axis(1).is_err());
    }

    #[test]
    #[should_panic(expected = "shape mismatch")]
    fn test_add_panics_on_mismatch() {
        let a = Tensor::from_vec(vec![1.0, 2.0], vec![2]).unwrap();
        let b = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let _ = a + b;
    }
}
