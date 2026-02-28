//! Numeric type hierarchy for generic math.
//!
//! The trait hierarchy is:
//! ```text
//! Scalar
//!   ├── Integer
//!   └── Float
//!         └── Real  (f32, f64)
//! ```
//!
//! All tensor operations and linear algebra routines are generic over these
//! traits so users can work with `f32`, `f64`, or integer types seamlessly.

use core::fmt;
use core::iter::Sum;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

// ---------------------------------------------------------------------------
// Scalar — the root trait for every numeric element type
// ---------------------------------------------------------------------------

/// Base trait for all numeric types storable in a tensor.
///
/// This intentionally does *not* require floating-point operations so that
/// integer tensors remain first-class citizens.
pub trait Scalar:
    Copy
    + Clone
    + fmt::Debug
    + fmt::Display
    + PartialEq
    + PartialOrd
    + Send
    + Sync
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + Sum
    + Default
    + 'static
{
    /// The additive identity (`0`).
    fn zero() -> Self;

    /// The multiplicative identity (`1`).
    fn one() -> Self;

    /// Convert from `usize` (used for index / shape arithmetic).
    fn from_usize(v: usize) -> Self;
}

// ---------------------------------------------------------------------------
// Integer
// ---------------------------------------------------------------------------

/// Marker trait for integer scalar types.
pub trait Integer: Scalar {
    /// Remainder after division.
    fn rem(self, rhs: Self) -> Self;
}

// ---------------------------------------------------------------------------
// Float — adds operations that only make sense for floating-point numbers
// ---------------------------------------------------------------------------

/// Trait for floating-point scalar types (`f32`, `f64`).
pub trait Float: Scalar + Neg<Output = Self> {
    /// Mathematical constant pi.
    fn pi() -> Self;

    /// Machine epsilon.
    fn epsilon() -> Self;

    /// Positive infinity.
    fn infinity() -> Self;

    /// Negative infinity.
    fn neg_infinity() -> Self;

    /// Not-a-number.
    fn nan() -> Self;

    fn abs(self) -> Self;
    fn sqrt(self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn tan(self) -> Self;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn log2(self) -> Self;
    fn log10(self) -> Self;
    fn powf(self, n: Self) -> Self;
    fn powi(self, n: i32) -> Self;
    fn floor(self) -> Self;
    fn ceil(self) -> Self;
    fn round(self) -> Self;
    fn recip(self) -> Self;
    fn is_nan(self) -> bool;
    fn is_finite(self) -> bool;
    fn min(self, other: Self) -> Self;
    fn max(self, other: Self) -> Self;

    /// Fused multiply-add: `self * a + b` with a single rounding step.
    fn mul_add(self, a: Self, b: Self) -> Self;

    /// Convert from an `f64` literal (used for constants).
    fn from_f64(v: f64) -> Self;
}

/// Trait alias for real-valued floats (non-complex).
///
/// Currently identical to [`Float`]; exists so that future complex-number
/// support can distinguish `Float` (the full set) from `Real` (the reals).
pub trait Real: Float {}

// ===========================================================================
// Blanket / macro implementations
// ===========================================================================

macro_rules! impl_scalar_float {
    ($ty:ty) => {
        impl Scalar for $ty {
            #[inline]
            fn zero() -> Self {
                0.0
            }
            #[inline]
            fn one() -> Self {
                1.0
            }
            #[inline]
            fn from_usize(v: usize) -> Self {
                v as Self
            }
        }

        impl Float for $ty {
            #[inline]
            fn pi() -> Self {
                Self::from_f64(std::f64::consts::PI)
            }
            #[inline]
            fn epsilon() -> Self {
                <$ty>::EPSILON
            }
            #[inline]
            fn infinity() -> Self {
                <$ty>::INFINITY
            }
            #[inline]
            fn neg_infinity() -> Self {
                <$ty>::NEG_INFINITY
            }
            #[inline]
            fn nan() -> Self {
                <$ty>::NAN
            }
            #[inline]
            fn abs(self) -> Self {
                <$ty>::abs(self)
            }
            #[inline]
            fn sqrt(self) -> Self {
                <$ty>::sqrt(self)
            }
            #[inline]
            fn sin(self) -> Self {
                <$ty>::sin(self)
            }
            #[inline]
            fn cos(self) -> Self {
                <$ty>::cos(self)
            }
            #[inline]
            fn tan(self) -> Self {
                <$ty>::tan(self)
            }
            #[inline]
            fn exp(self) -> Self {
                <$ty>::exp(self)
            }
            #[inline]
            fn ln(self) -> Self {
                <$ty>::ln(self)
            }
            #[inline]
            fn log2(self) -> Self {
                <$ty>::log2(self)
            }
            #[inline]
            fn log10(self) -> Self {
                <$ty>::log10(self)
            }
            #[inline]
            fn powf(self, n: Self) -> Self {
                <$ty>::powf(self, n)
            }
            #[inline]
            fn powi(self, n: i32) -> Self {
                <$ty>::powi(self, n)
            }
            #[inline]
            fn floor(self) -> Self {
                <$ty>::floor(self)
            }
            #[inline]
            fn ceil(self) -> Self {
                <$ty>::ceil(self)
            }
            #[inline]
            fn round(self) -> Self {
                <$ty>::round(self)
            }
            #[inline]
            fn recip(self) -> Self {
                <$ty>::recip(self)
            }
            #[inline]
            fn is_nan(self) -> bool {
                <$ty>::is_nan(self)
            }
            #[inline]
            fn is_finite(self) -> bool {
                <$ty>::is_finite(self)
            }
            #[inline]
            fn min(self, other: Self) -> Self {
                <$ty>::min(self, other)
            }
            #[inline]
            fn max(self, other: Self) -> Self {
                <$ty>::max(self, other)
            }
            #[inline]
            fn mul_add(self, a: Self, b: Self) -> Self {
                <$ty>::mul_add(self, a, b)
            }
            #[inline]
            fn from_f64(v: f64) -> Self {
                v as Self
            }
        }

        impl Real for $ty {}
    };
}

impl_scalar_float!(f32);
impl_scalar_float!(f64);

macro_rules! impl_scalar_int {
    ($ty:ty) => {
        impl Scalar for $ty {
            #[inline]
            fn zero() -> Self {
                0
            }
            #[inline]
            fn one() -> Self {
                1
            }
            #[inline]
            #[allow(clippy::cast_possible_wrap)]
            fn from_usize(v: usize) -> Self {
                v as Self
            }
        }

        impl Integer for $ty {
            #[inline]
            fn rem(self, rhs: Self) -> Self {
                self % rhs
            }
        }
    };
}

impl_scalar_int!(i8);
impl_scalar_int!(i16);
impl_scalar_int!(i32);
impl_scalar_int!(i64);
impl_scalar_int!(u8);
impl_scalar_int!(u16);
impl_scalar_int!(u32);
impl_scalar_int!(u64);
impl_scalar_int!(usize);
impl_scalar_int!(isize);

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_zero_one() {
        assert_eq!(f64::zero(), 0.0);
        assert_eq!(f64::one(), 1.0);
        assert_eq!(i32::zero(), 0);
        assert_eq!(i32::one(), 1);
    }

    #[test]
    fn test_float_constants() {
        let pi: f64 = Float::pi();
        assert!((pi - std::f64::consts::PI).abs() < f64::EPSILON);
        assert!(f64::nan().is_nan());
        assert!(!f64::infinity().is_finite());
    }

    #[test]
    fn test_float_ops() {
        let x: f64 = 4.0;
        assert_eq!(x.sqrt(), 2.0);
        assert_eq!(Float::abs(-3.0_f64), 3.0);
        assert_eq!(x.recip(), 0.25);
    }

    #[test]
    fn test_from_usize() {
        assert_eq!(f32::from_usize(42), 42.0_f32);
        assert_eq!(u8::from_usize(255), 255_u8);
    }

    #[test]
    fn test_integer_rem() {
        assert_eq!(Integer::rem(7_i32, 3), 1);
        assert_eq!(Integer::rem(10_u64, 4), 2);
    }
}
