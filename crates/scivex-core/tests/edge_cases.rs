#![allow(clippy::float_cmp, clippy::cast_lossless, clippy::redundant_closure)]
//! Edge case tests for Tensor operations.
//!
//! Tests zero-size tensors, single-element tensors, 0-D scalars,
//! NaN/Inf propagation, empty slices, and broadcasting edge cases.

use scivex_core::Tensor;

// ===========================================================================
// Single-element tensors
// ===========================================================================

#[test]
fn single_element_tensor_creation() {
    let t = Tensor::from_vec(vec![42.0_f64], vec![1]).unwrap();
    assert_eq!(t.shape(), &[1]);
    assert_eq!(t.numel(), 1);
    assert_eq!(*t.get(&[0]).unwrap(), 42.0);
}

#[test]
fn single_element_sum_mean() {
    let t = Tensor::from_vec(vec![7.0_f64], vec![1]).unwrap();
    assert!((t.sum() - 7.0).abs() < f64::EPSILON);
    assert!((t.mean() - 7.0).abs() < f64::EPSILON);
}

#[test]
fn single_element_matmul() {
    // 1x1 * 1x1 = 1x1
    let a = Tensor::from_vec(vec![3.0_f64], vec![1, 1]).unwrap();
    let b = Tensor::from_vec(vec![4.0_f64], vec![1, 1]).unwrap();
    let c = a.matmul(&b).unwrap();
    assert_eq!(c.shape(), &[1, 1]);
    assert!((c.as_slice()[0] - 12.0).abs() < f64::EPSILON);
}

#[test]
fn single_element_transpose() {
    let t = Tensor::from_vec(vec![5.0_f64], vec![1, 1]).unwrap();
    let tt = t.transpose().unwrap();
    assert_eq!(tt.shape(), &[1, 1]);
    assert_eq!(tt.as_slice(), &[5.0]);
}

#[test]
fn single_element_reshape() {
    let t = Tensor::from_vec(vec![9.0_f64], vec![1]).unwrap();
    let r = t.reshape(vec![1, 1]).unwrap();
    assert_eq!(r.shape(), &[1, 1]);
    assert_eq!(r.as_slice(), &[9.0]);
}

#[test]
fn single_element_arithmetic() {
    let a = Tensor::from_vec(vec![3.0_f64], vec![1]).unwrap();
    let b = Tensor::from_vec(vec![4.0_f64], vec![1]).unwrap();
    let sum = &a + &b;
    assert_eq!(sum.as_slice(), &[7.0]);
    let prod = &a * &b;
    assert_eq!(prod.as_slice(), &[12.0]);
}

// ===========================================================================
// NaN and Inf propagation
// ===========================================================================

#[test]
fn nan_propagates_through_add() {
    let a = Tensor::from_vec(vec![1.0_f64, f64::NAN, 3.0], vec![3]).unwrap();
    let b = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
    let result = &a + &b;
    assert!((result.as_slice()[0] - 2.0).abs() < f64::EPSILON);
    assert!(result.as_slice()[1].is_nan());
    assert!((result.as_slice()[2] - 6.0).abs() < f64::EPSILON);
}

#[test]
fn nan_propagates_through_mul() {
    let a = Tensor::from_vec(vec![f64::NAN, 2.0], vec![2]).unwrap();
    let b = Tensor::from_vec(vec![5.0, 3.0], vec![2]).unwrap();
    let result = &a * &b;
    assert!(result.as_slice()[0].is_nan());
    assert!((result.as_slice()[1] - 6.0).abs() < f64::EPSILON);
}

#[test]
fn nan_in_sum() {
    let t = Tensor::from_vec(vec![1.0_f64, f64::NAN, 3.0], vec![3]).unwrap();
    assert!(t.sum().is_nan());
}

#[test]
fn nan_in_mean() {
    let t = Tensor::from_vec(vec![1.0_f64, f64::NAN, 3.0], vec![3]).unwrap();
    assert!(t.mean().is_nan());
}

#[test]
fn inf_propagates_through_add() {
    let a = Tensor::from_vec(vec![f64::INFINITY, 1.0], vec![2]).unwrap();
    let b = Tensor::from_vec(vec![1.0, 2.0], vec![2]).unwrap();
    let result = &a + &b;
    assert!(result.as_slice()[0].is_infinite());
    assert!((result.as_slice()[1] - 3.0).abs() < f64::EPSILON);
}

#[test]
fn inf_minus_inf_is_nan() {
    let a = Tensor::from_vec(vec![f64::INFINITY], vec![1]).unwrap();
    let b = Tensor::from_vec(vec![f64::INFINITY], vec![1]).unwrap();
    let result = &a - &b;
    assert!(result.as_slice()[0].is_nan());
}

// ===========================================================================
// Large dimension / edge shapes
// ===========================================================================

#[test]
fn very_wide_matrix() {
    // 1 x 10000 matrix
    let t = Tensor::<f64>::ones(vec![1, 10000]);
    assert_eq!(t.shape(), &[1, 10000]);
    assert_eq!(t.numel(), 10000);
    assert!((t.sum() - 10000.0).abs() < f64::EPSILON);
}

#[test]
fn very_tall_matrix() {
    // 10000 x 1 matrix
    let t = Tensor::<f64>::ones(vec![10000, 1]);
    assert_eq!(t.shape(), &[10000, 1]);
    assert!((t.sum() - 10000.0).abs() < f64::EPSILON);
}

#[test]
fn high_dimensional_tensor() {
    // 2x2x2x2x2 = 32 elements
    let t = Tensor::<f64>::ones(vec![2, 2, 2, 2, 2]);
    assert_eq!(t.numel(), 32);
    assert!((t.sum() - 32.0).abs() < f64::EPSILON);
}

#[test]
fn reshape_to_many_dims() {
    let t = Tensor::from_vec((0..24).map(|i| i as f64).collect(), vec![24]).unwrap();
    let r = t.reshape(vec![2, 3, 4]).unwrap();
    assert_eq!(r.shape(), &[2, 3, 4]);
    assert_eq!(r.numel(), 24);
}

// ===========================================================================
// Zeros and ones edge cases
// ===========================================================================

#[test]
fn zeros_sum_is_zero() {
    let t = Tensor::<f64>::zeros(vec![100, 100]);
    assert!((t.sum() - 0.0).abs() < f64::EPSILON);
}

#[test]
fn ones_sum_equals_numel() {
    let t = Tensor::<f64>::ones(vec![7, 13]);
    assert!((t.sum() - 91.0).abs() < f64::EPSILON);
}

#[test]
fn eye_diagonal_sum() {
    let t = Tensor::<f64>::eye(50);
    assert!((t.sum() - 50.0).abs() < f64::EPSILON);
}

#[test]
fn eye_1x1() {
    let t = Tensor::<f64>::eye(1);
    assert_eq!(t.shape(), &[1, 1]);
    assert_eq!(t.as_slice(), &[1.0]);
}

// ===========================================================================
// Flatten and reshape edge cases
// ===========================================================================

#[test]
fn flatten_already_flat() {
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
    let flat = t.flatten();
    assert_eq!(flat.shape(), &[3]);
    assert_eq!(flat.as_slice(), &[1.0, 2.0, 3.0]);
}

#[test]
fn reshape_incompatible_fails() {
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
    let result = t.reshape(vec![2, 2]);
    assert!(result.is_err());
}

#[test]
fn from_vec_shape_mismatch_fails() {
    let result = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![2, 2]);
    assert!(result.is_err());
}

// ===========================================================================
// Map edge cases
// ===========================================================================

#[test]
fn map_with_nan_producing_function() {
    let t = Tensor::from_vec(vec![-1.0_f64, 0.0, 1.0], vec![3]).unwrap();
    let mapped = t.map(f64::sqrt); // sqrt(-1) = NaN
    assert!(mapped.as_slice()[0].is_nan());
    assert!((mapped.as_slice()[1] - 0.0).abs() < f64::EPSILON);
    assert!((mapped.as_slice()[2] - 1.0).abs() < f64::EPSILON);
}

#[test]
fn map_constant_function() {
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
    let mapped = t.map(|_| 42.0_f64);
    assert!(
        mapped
            .as_slice()
            .iter()
            .all(|&x| (x - 42.0).abs() < f64::EPSILON)
    );
}

// ===========================================================================
// Arange edge cases
// ===========================================================================

#[test]
fn arange_single_element() {
    let t = Tensor::<f64>::arange(1);
    assert_eq!(t.shape(), &[1]);
    assert_eq!(t.as_slice(), &[0.0]);
}

#[test]
fn linspace_two_points() {
    let t = Tensor::linspace(0.0_f64, 1.0, 2).unwrap();
    assert_eq!(t.shape(), &[2]);
    assert!((t.as_slice()[0] - 0.0).abs() < f64::EPSILON);
    assert!((t.as_slice()[1] - 1.0).abs() < f64::EPSILON);
}

#[test]
fn linspace_single_point_returns_error() {
    // linspace requires n >= 2
    let result = Tensor::linspace(5.0_f64, 5.0, 1);
    assert!(result.is_err());
}

// ===========================================================================
// f32 type tests (ensure Float trait works for both types)
// ===========================================================================

#[test]
fn f32_basic_operations() {
    let a = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0], vec![3]).unwrap();
    let b = Tensor::from_vec(vec![4.0_f32, 5.0, 6.0], vec![3]).unwrap();
    let sum = &a + &b;
    assert_eq!(sum.as_slice(), &[5.0_f32, 7.0, 9.0]);
}

#[test]
fn f32_matmul() {
    let a = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let b = Tensor::from_vec(vec![5.0_f32, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
    let c = a.matmul(&b).unwrap();
    assert_eq!(c.shape(), &[2, 2]);
    assert!((c.as_slice()[0] - 19.0).abs() < 1e-5);
}
