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

// SIMD-accelerated overrides for f64 and f32 element-wise add/mul.
// These override the generic macro implementations for concrete float types.
#[cfg(feature = "simd")]
impl Tensor<f64> {
    /// SIMD-accelerated element-wise addition.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// # use scivex_core::Tensor;
    /// let a = Tensor::from_vec(vec![1.0_f64, 2.0], vec![2]).unwrap();
    /// let b = Tensor::from_vec(vec![3.0, 4.0], vec![2]).unwrap();
    /// let c = a.add_simd(&b);
    /// assert_eq!(c.as_slice(), &[4.0, 6.0]);
    /// ```
    pub fn add_simd(&self, other: &Tensor<f64>) -> Tensor<f64> {
        assert_eq!(self.shape, other.shape, "shape mismatch in simd add");
        let mut out = vec![0.0_f64; self.data.len()];
        crate::simd::f64_ops::add_f64(&self.data, &other.data, &mut out);
        Tensor {
            data: out,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        }
    }

    /// SIMD-accelerated element-wise multiplication.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// # use scivex_core::Tensor;
    /// let a = Tensor::from_vec(vec![2.0_f64, 3.0], vec![2]).unwrap();
    /// let b = Tensor::from_vec(vec![4.0, 5.0], vec![2]).unwrap();
    /// let c = a.mul_simd(&b);
    /// assert_eq!(c.as_slice(), &[8.0, 15.0]);
    /// ```
    pub fn mul_simd(&self, other: &Tensor<f64>) -> Tensor<f64> {
        assert_eq!(self.shape, other.shape, "shape mismatch in simd mul");
        let mut out = vec![0.0_f64; self.data.len()];
        crate::simd::f64_ops::mul_f64(&self.data, &other.data, &mut out);
        Tensor {
            data: out,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        }
    }
}

#[cfg(feature = "simd")]
impl Tensor<f32> {
    /// SIMD-accelerated element-wise addition.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// # use scivex_core::Tensor;
    /// let a = Tensor::from_vec(vec![1.0_f32, 2.0], vec![2]).unwrap();
    /// let b = Tensor::from_vec(vec![3.0, 4.0], vec![2]).unwrap();
    /// let c = a.add_simd(&b);
    /// assert_eq!(c.as_slice(), &[4.0, 6.0]);
    /// ```
    pub fn add_simd(&self, other: &Tensor<f32>) -> Tensor<f32> {
        assert_eq!(self.shape, other.shape, "shape mismatch in simd add");
        let mut out = vec![0.0_f32; self.data.len()];
        crate::simd::f32_ops::add_f32(&self.data, &other.data, &mut out);
        Tensor {
            data: out,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        }
    }

    /// SIMD-accelerated element-wise multiplication.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// # use scivex_core::Tensor;
    /// let a = Tensor::from_vec(vec![2.0_f32, 3.0], vec![2]).unwrap();
    /// let b = Tensor::from_vec(vec![4.0, 5.0], vec![2]).unwrap();
    /// let c = a.mul_simd(&b);
    /// assert_eq!(c.as_slice(), &[8.0, 15.0]);
    /// ```
    pub fn mul_simd(&self, other: &Tensor<f32>) -> Tensor<f32> {
        assert_eq!(self.shape, other.shape, "shape mismatch in simd mul");
        let mut out = vec![0.0_f32; self.data.len()];
        crate::simd::f32_ops::mul_f32(&self.data, &other.data, &mut out);
        Tensor {
            data: out,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        }
    }
}

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
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_core::Tensor;
    /// let a = Tensor::from_vec(vec![1, 2, 3], vec![3]).unwrap();
    /// let b = Tensor::from_vec(vec![4, 5, 6], vec![3]).unwrap();
    /// let c = a.add_checked(&b).unwrap();
    /// assert_eq!(c.as_slice(), &[5, 7, 9]);
    /// ```
    pub fn add_checked(&self, other: &Tensor<T>) -> crate::Result<Tensor<T>> {
        self.zip_map(other, |a, b| a + b)
    }

    /// Element-wise subtraction, returning `Err` on shape mismatch.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_core::Tensor;
    /// let a = Tensor::from_vec(vec![10, 20, 30], vec![3]).unwrap();
    /// let b = Tensor::from_vec(vec![1, 2, 3], vec![3]).unwrap();
    /// let c = a.sub_checked(&b).unwrap();
    /// assert_eq!(c.as_slice(), &[9, 18, 27]);
    /// ```
    pub fn sub_checked(&self, other: &Tensor<T>) -> crate::Result<Tensor<T>> {
        self.zip_map(other, |a, b| a - b)
    }

    /// Element-wise multiplication, returning `Err` on shape mismatch.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_core::Tensor;
    /// let a = Tensor::from_vec(vec![2, 3, 4], vec![3]).unwrap();
    /// let b = Tensor::from_vec(vec![5, 6, 7], vec![3]).unwrap();
    /// let c = a.mul_checked(&b).unwrap();
    /// assert_eq!(c.as_slice(), &[10, 18, 28]);
    /// ```
    pub fn mul_checked(&self, other: &Tensor<T>) -> crate::Result<Tensor<T>> {
        self.zip_map(other, |a, b| a * b)
    }

    /// Element-wise division, returning `Err` on shape mismatch.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_core::Tensor;
    /// let a = Tensor::from_vec(vec![10, 20, 30], vec![3]).unwrap();
    /// let b = Tensor::from_vec(vec![2, 5, 6], vec![3]).unwrap();
    /// let c = a.div_checked(&b).unwrap();
    /// assert_eq!(c.as_slice(), &[5, 4, 5]);
    /// ```
    pub fn div_checked(&self, other: &Tensor<T>) -> crate::Result<Tensor<T>> {
        self.zip_map(other, |a, b| a / b)
    }
}

// ======================================================================
// Reductions
// ======================================================================

impl<T: Scalar> Tensor<T> {
    /// Sum of all elements.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_core::Tensor;
    /// let t = Tensor::from_vec(vec![1, 2, 3, 4], vec![4]).unwrap();
    /// assert_eq!(t.sum(), 10);
    /// ```
    pub fn sum(&self) -> T {
        #[cfg(feature = "simd")]
        {
            use crate::simd;
            use std::any::TypeId;
            if TypeId::of::<T>() == TypeId::of::<f64>() {
                // SAFETY: T is f64 confirmed by TypeId.
                let result =
                    unsafe { simd::f64_ops::sum_f64(simd::slice_as_f64(self.data.as_slice())) };
                return unsafe { simd::f64_to_t(result) };
            }
            if TypeId::of::<T>() == TypeId::of::<f32>() {
                // SAFETY: T is f32 confirmed by TypeId.
                let result =
                    unsafe { simd::f32_ops::sum_f32(simd::slice_as_f32(self.data.as_slice())) };
                return unsafe { simd::f32_to_t(result) };
            }
        }
        self.data.iter().copied().sum()
    }

    /// Product of all elements.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_core::Tensor;
    /// let t = Tensor::from_vec(vec![1, 2, 3, 4], vec![4]).unwrap();
    /// assert_eq!(t.product(), 24);
    /// ```
    pub fn product(&self) -> T {
        self.data.iter().copied().fold(T::one(), |acc, x| acc * x)
    }

    /// Minimum element. Returns `None` for empty tensors.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_core::Tensor;
    /// let t = Tensor::from_vec(vec![3, 1, 4, 1, 5], vec![5]).unwrap();
    /// assert_eq!(t.min_element(), Some(1));
    /// let empty = Tensor::<i32>::zeros(vec![0]);
    /// assert_eq!(empty.min_element(), None);
    /// ```
    pub fn min_element(&self) -> Option<T> {
        if self.data.is_empty() {
            return None;
        }
        #[cfg(feature = "simd")]
        {
            use crate::simd;
            use std::any::TypeId;
            if TypeId::of::<T>() == TypeId::of::<f64>() {
                // SAFETY: T is f64 confirmed by TypeId.
                let result =
                    unsafe { simd::f64_ops::min_f64(simd::slice_as_f64(self.data.as_slice())) };
                return Some(unsafe { simd::f64_to_t(result) });
            }
            if TypeId::of::<T>() == TypeId::of::<f32>() {
                // SAFETY: T is f32 confirmed by TypeId.
                let result =
                    unsafe { simd::f32_ops::min_f32(simd::slice_as_f32(self.data.as_slice())) };
                return Some(unsafe { simd::f32_to_t(result) });
            }
        }
        self.data
            .iter()
            .copied()
            .reduce(|a, b| if b < a { b } else { a })
    }

    /// Maximum element. Returns `None` for empty tensors.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_core::Tensor;
    /// let t = Tensor::from_vec(vec![3, 1, 4, 1, 5], vec![5]).unwrap();
    /// assert_eq!(t.max_element(), Some(5));
    /// ```
    pub fn max_element(&self) -> Option<T> {
        if self.data.is_empty() {
            return None;
        }
        #[cfg(feature = "simd")]
        {
            use crate::simd;
            use std::any::TypeId;
            if TypeId::of::<T>() == TypeId::of::<f64>() {
                // SAFETY: T is f64 confirmed by TypeId.
                let result =
                    unsafe { simd::f64_ops::max_f64(simd::slice_as_f64(self.data.as_slice())) };
                return Some(unsafe { simd::f64_to_t(result) });
            }
            if TypeId::of::<T>() == TypeId::of::<f32>() {
                // SAFETY: T is f32 confirmed by TypeId.
                let result =
                    unsafe { simd::f32_ops::max_f32(simd::slice_as_f32(self.data.as_slice())) };
                return Some(unsafe { simd::f32_to_t(result) });
            }
        }
        self.data
            .iter()
            .copied()
            .reduce(|a, b| if b > a { b } else { a })
    }

    /// Sum along a given axis, producing a tensor with that axis removed.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_core::Tensor;
    /// let t = Tensor::from_vec(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
    /// let s = t.sum_axis(0).unwrap();
    /// assert_eq!(s.as_slice(), &[5, 7, 9]);
    /// ```
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
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_core::Tensor;
    /// let t = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0], vec![4]).unwrap();
    /// assert_eq!(t.mean(), 2.5_f64);
    /// ```
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
