//! Type promotion rules and casting utilities for tensor element types.
//!
//! Provides numpy-style type promotion via [`promote`], compile-time type
//! identification via [`DTypeOf`], and element-wise casting via [`CastFrom`].

use crate::dtype::Scalar;
use crate::tensor::Tensor;

// ---------------------------------------------------------------------------
// DType — runtime type tag
// ---------------------------------------------------------------------------

/// Runtime type tag for tensor element types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    U8,
    U16,
    U32,
    U64,
    I8,
    I16,
    I32,
    I64,
    F32,
    F64,
}

impl DType {
    /// Size in bytes of a single element of this type.
    #[inline]
    pub fn size_bytes(self) -> usize {
        match self {
            DType::U8 | DType::I8 => 1,
            DType::U16 | DType::I16 => 2,
            DType::U32 | DType::I32 | DType::F32 => 4,
            DType::U64 | DType::I64 | DType::F64 => 8,
        }
    }

    /// Returns `true` if this is a floating-point type.
    #[inline]
    pub fn is_float(self) -> bool {
        matches!(self, DType::F32 | DType::F64)
    }

    /// Returns `true` if this is a signed type (signed integers or floats).
    #[inline]
    pub fn is_signed(self) -> bool {
        matches!(
            self,
            DType::I8 | DType::I16 | DType::I32 | DType::I64 | DType::F32 | DType::F64
        )
    }

    /// Returns `true` if this is an integer type (signed or unsigned).
    #[inline]
    pub fn is_integer(self) -> bool {
        !self.is_float()
    }
}

// ---------------------------------------------------------------------------
// promote — numpy-style type promotion
// ---------------------------------------------------------------------------

/// Determine the result type when combining two dtypes (numpy-style promotion).
///
/// Rules:
/// - Same type -> same type
/// - Integer + Integer -> wider integer (preserving signedness if possible)
/// - Signed + Unsigned -> signed type wide enough for both
/// - Any Integer + Any Float -> the float type
/// - F32 + F64 -> F64
pub fn promote(a: DType, b: DType) -> DType {
    if a == b {
        return a;
    }

    // If either is float, the result is float.
    match (a.is_float(), b.is_float()) {
        (true, true) => {
            // F32 + F64 -> F64
            if a == DType::F64 || b == DType::F64 {
                return DType::F64;
            }
            return DType::F32;
        }
        (true, false) => return promote_int_float(b, a),
        (false, true) => return promote_int_float(a, b),
        (false, false) => {}
    }

    // Both are integers.
    let a_signed = a.is_signed();
    let b_signed = b.is_signed();
    let a_bytes = a.size_bytes();
    let b_bytes = b.size_bytes();

    match (a_signed, b_signed) {
        // Both same signedness: pick the wider one.
        (true, true) | (false, false) => {
            if a_bytes >= b_bytes {
                a
            } else {
                b
            }
        }
        // Mixed signedness: need a signed type wide enough for both.
        _ => {
            let (signed_dt, unsigned_dt) = if a_signed { (a, b) } else { (b, a) };
            let s_bytes = signed_dt.size_bytes();
            let u_bytes = unsigned_dt.size_bytes();
            // The signed type must be strictly wider than the unsigned type
            // to represent all values of both.
            if s_bytes > u_bytes {
                // Signed type is already wide enough.
                signed_dt
            } else {
                // Need to widen: pick the next signed type larger than the unsigned type.
                match u_bytes {
                    1 => DType::I16,
                    2 => DType::I32,
                    4 => DType::I64,
                    // u64 cannot be fully represented by i64; promote to f64.
                    _ => DType::F64,
                }
            }
        }
    }
}

/// Promote an integer dtype combined with a float dtype.
fn promote_int_float(_int_dt: DType, float_dt: DType) -> DType {
    // Any integer + any float -> the float type.
    // If the integer is very wide (64-bit) and float is F32, we still return
    // the float type (matching numpy behavior).
    float_dt
}

// ---------------------------------------------------------------------------
// DTypeOf — compile-time type -> DType mapping
// ---------------------------------------------------------------------------

/// Trait to get the [`DType`] tag for a [`Scalar`] type at compile time.
pub trait DTypeOf: Scalar {
    /// The runtime [`DType`] tag for this type.
    fn dtype() -> DType;
}

macro_rules! impl_dtype_of {
    ($($ty:ty => $variant:ident),* $(,)?) => {
        $(
            impl DTypeOf for $ty {
                #[inline]
                fn dtype() -> DType {
                    DType::$variant
                }
            }
        )*
    };
}

impl_dtype_of!(
    u8 => U8,
    u16 => U16,
    u32 => U32,
    u64 => U64,
    i8 => I8,
    i16 => I16,
    i32 => I32,
    i64 => I64,
    f32 => F32,
    f64 => F64,
);

// ---------------------------------------------------------------------------
// CastFrom — numeric type casting
// ---------------------------------------------------------------------------

/// Trait for numeric type casting between scalar types.
///
/// Implementations use Rust `as` casts, which means:
/// - Integer widening is lossless.
/// - Float-to-integer truncates toward zero.
/// - Integer-to-float may lose precision for large values.
/// - f64-to-f32 may lose precision or become infinity.
pub trait CastFrom<T> {
    /// Cast a value of type `T` into `Self`.
    fn cast_from(val: T) -> Self;
}

// Macro to generate CastFrom impls for all reasonable pairs.
macro_rules! impl_cast_from {
    ($src:ty => $($dst:ty),* $(,)?) => {
        $(
            impl CastFrom<$src> for $dst {
                #[inline]
                #[allow(clippy::cast_possible_truncation)]
                #[allow(clippy::cast_possible_wrap)]
                #[allow(clippy::cast_sign_loss)]
                #[allow(clippy::cast_lossless)]
                #[allow(clippy::cast_precision_loss)]
                fn cast_from(val: $src) -> Self {
                    val as Self
                }
            }
        )*
    };
}

// From every integer/float type to every integer/float type.
impl_cast_from!(u8  => u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);
impl_cast_from!(u16 => u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);
impl_cast_from!(u32 => u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);
impl_cast_from!(u64 => u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);
impl_cast_from!(i8  => u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);
impl_cast_from!(i16 => u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);
impl_cast_from!(i32 => u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);
impl_cast_from!(i64 => u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);
impl_cast_from!(f32 => u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);
impl_cast_from!(f64 => u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);

// Also support usize / isize as sources and destinations.
impl_cast_from!(usize => u8, u16, u32, u64, i8, i16, i32, i64, f32, f64, usize, isize);
impl_cast_from!(isize => u8, u16, u32, u64, i8, i16, i32, i64, f32, f64, usize, isize);
impl_cast_from!(u8    => usize, isize);
impl_cast_from!(u16   => usize, isize);
impl_cast_from!(u32   => usize, isize);
impl_cast_from!(u64   => usize, isize);
impl_cast_from!(i8    => usize, isize);
impl_cast_from!(i16   => usize, isize);
impl_cast_from!(i32   => usize, isize);
impl_cast_from!(i64   => usize, isize);
impl_cast_from!(f32   => usize, isize);
impl_cast_from!(f64   => usize, isize);

// ---------------------------------------------------------------------------
// Tensor::cast_to
// ---------------------------------------------------------------------------

impl<T: Scalar> Tensor<T> {
    /// Cast every element of this tensor to a different scalar type.
    ///
    /// This allocates a new tensor with the same shape and copies each element
    /// through [`CastFrom`].
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_core::Tensor;
    /// # use scivex_core::promote::CastFrom;
    /// let t = Tensor::from_vec(vec![1_u8, 2, 3, 4], vec![2, 2]).unwrap();
    /// let f: Tensor<f64> = t.cast_to();
    /// assert_eq!(f.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
    /// ```
    pub fn cast_to<U: Scalar + CastFrom<T>>(&self) -> Tensor<U> {
        let data: Vec<U> = self.as_slice().iter().map(|&v| U::cast_from(v)).collect();
        // Shape is unchanged so the product of dimensions equals data.len();
        // from_vec will never fail here.
        Tensor::from_vec(data, self.shape().to_vec())
            .expect("cast_to: shape unchanged, from_vec cannot fail")
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_promote_same_type() {
        assert_eq!(promote(DType::F32, DType::F32), DType::F32);
        assert_eq!(promote(DType::I32, DType::I32), DType::I32);
        assert_eq!(promote(DType::U8, DType::U8), DType::U8);
    }

    #[test]
    fn test_promote_int_float() {
        assert_eq!(promote(DType::I32, DType::F32), DType::F32);
        assert_eq!(promote(DType::U8, DType::F64), DType::F64);
        assert_eq!(promote(DType::U64, DType::F32), DType::F32);
        assert_eq!(promote(DType::I64, DType::F64), DType::F64);
    }

    #[test]
    fn test_promote_int_widening() {
        assert_eq!(promote(DType::I8, DType::I32), DType::I32);
        assert_eq!(promote(DType::U8, DType::U16), DType::U16);
        assert_eq!(promote(DType::U16, DType::U32), DType::U32);
        assert_eq!(promote(DType::I16, DType::I64), DType::I64);
    }

    #[test]
    fn test_promote_signed_unsigned() {
        // I8 (1 byte signed) + U8 (1 byte unsigned) -> I16 (need wider signed)
        assert_eq!(promote(DType::I8, DType::U8), DType::I16);
        // I16 (2 bytes signed) + U16 (2 bytes unsigned) -> I32
        assert_eq!(promote(DType::I16, DType::U16), DType::I32);
        // I32 + U32 -> I64
        assert_eq!(promote(DType::I32, DType::U32), DType::I64);
        // I64 + U64 -> F64 (no wider signed integer)
        assert_eq!(promote(DType::I64, DType::U64), DType::F64);
        // I32 (4 bytes) + U8 (1 byte) -> I32 (signed already wider)
        assert_eq!(promote(DType::I32, DType::U8), DType::I32);
    }

    #[test]
    fn test_dtype_of() {
        assert_eq!(<u8 as DTypeOf>::dtype(), DType::U8);
        assert_eq!(<u16 as DTypeOf>::dtype(), DType::U16);
        assert_eq!(<u32 as DTypeOf>::dtype(), DType::U32);
        assert_eq!(<u64 as DTypeOf>::dtype(), DType::U64);
        assert_eq!(<i8 as DTypeOf>::dtype(), DType::I8);
        assert_eq!(<i16 as DTypeOf>::dtype(), DType::I16);
        assert_eq!(<i32 as DTypeOf>::dtype(), DType::I32);
        assert_eq!(<i64 as DTypeOf>::dtype(), DType::I64);
        assert_eq!(<f32 as DTypeOf>::dtype(), DType::F32);
        assert_eq!(<f64 as DTypeOf>::dtype(), DType::F64);
    }

    #[test]
    fn test_dtype_properties() {
        assert_eq!(DType::U8.size_bytes(), 1);
        assert_eq!(DType::U16.size_bytes(), 2);
        assert_eq!(DType::U32.size_bytes(), 4);
        assert_eq!(DType::U64.size_bytes(), 8);
        assert_eq!(DType::I8.size_bytes(), 1);
        assert_eq!(DType::I16.size_bytes(), 2);
        assert_eq!(DType::I32.size_bytes(), 4);
        assert_eq!(DType::I64.size_bytes(), 8);
        assert_eq!(DType::F32.size_bytes(), 4);
        assert_eq!(DType::F64.size_bytes(), 8);

        assert!(DType::F32.is_float());
        assert!(DType::F64.is_float());
        assert!(!DType::I32.is_float());
        assert!(!DType::U8.is_float());

        assert!(DType::I8.is_signed());
        assert!(DType::F64.is_signed());
        assert!(!DType::U8.is_signed());
        assert!(!DType::U64.is_signed());

        assert!(DType::I32.is_integer());
        assert!(DType::U16.is_integer());
        assert!(!DType::F32.is_integer());
    }

    #[test]
    fn test_cast_from_u8_to_f64() {
        let t = Tensor::from_vec(vec![1_u8, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
        let f: Tensor<f64> = t.cast_to();
        assert_eq!(f.shape(), &[2, 3]);
        assert_eq!(f.as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_cast_from_f64_to_f32() {
        let t = Tensor::from_vec(vec![1.5_f64, 2.25, -3.0, 1e30], vec![4]).unwrap();
        let f: Tensor<f32> = t.cast_to();
        assert_eq!(f.shape(), &[4]);
        assert!((f.as_slice()[0] - 1.5_f32).abs() < f32::EPSILON);
        assert!((f.as_slice()[1] - 2.25_f32).abs() < f32::EPSILON);
        assert!((f.as_slice()[2] - (-3.0_f32)).abs() < f32::EPSILON);
    }
}
