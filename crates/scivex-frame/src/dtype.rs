//! Runtime type identification for type-erased columns.

/// Runtime representation of a column's element type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F64,
    F32,
    I64,
    I32,
    I16,
    I8,
    U64,
    U32,
    U16,
    U8,
    Bool,
    Str,
    Categorical,
}

impl core::fmt::Display for DType {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let s = match self {
            Self::F64 => "f64",
            Self::F32 => "f32",
            Self::I64 => "i64",
            Self::I32 => "i32",
            Self::I16 => "i16",
            Self::I8 => "i8",
            Self::U64 => "u64",
            Self::U32 => "u32",
            Self::U16 => "u16",
            Self::U8 => "u8",
            Self::Bool => "bool",
            Self::Str => "str",
            Self::Categorical => "categorical",
        };
        write!(f, "{s}")
    }
}

/// Compile-time mapping from a Rust type to its [`DType`] variant.
pub trait HasDType {
    /// The runtime dtype for this type.
    fn dtype() -> DType;
}

macro_rules! impl_has_dtype {
    ($($ty:ty => $variant:ident),+ $(,)?) => {
        $(
            impl HasDType for $ty {
                #[inline]
                fn dtype() -> DType {
                    DType::$variant
                }
            }
        )+
    };
}

impl_has_dtype! {
    f64 => F64,
    f32 => F32,
    i64 => I64,
    i32 => I32,
    i16 => I16,
    i8  => I8,
    u64 => U64,
    u32 => U32,
    u16 => U16,
    u8  => U8,
    bool => Bool,
    String => Str,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_display() {
        assert_eq!(DType::F64.to_string(), "f64");
        assert_eq!(DType::Str.to_string(), "str");
    }

    #[test]
    fn test_has_dtype() {
        assert_eq!(f64::dtype(), DType::F64);
        assert_eq!(i32::dtype(), DType::I32);
        assert_eq!(bool::dtype(), DType::Bool);
        assert_eq!(String::dtype(), DType::Str);
    }
}
