//! Element-wise tensor math functions (ufuncs).
//!
//! Provides both methods on `Tensor<T: Float>` and free functions mirroring
//! `NumPy`'s top-level ufuncs (`np.sin`, `np.exp`, etc.).

use crate::Float;
use crate::tensor::Tensor;

// ======================================================================
// Tensor methods
// ======================================================================

impl<T: Float> Tensor<T> {
    /// Element-wise absolute value.
    #[inline]
    pub fn abs(&self) -> Tensor<T> {
        self.map(Float::abs)
    }

    /// Element-wise square root.
    #[inline]
    pub fn sqrt(&self) -> Tensor<T> {
        self.map(Float::sqrt)
    }

    /// Element-wise sine.
    #[inline]
    pub fn sin(&self) -> Tensor<T> {
        self.map(Float::sin)
    }

    /// Element-wise cosine.
    #[inline]
    pub fn cos(&self) -> Tensor<T> {
        self.map(Float::cos)
    }

    /// Element-wise tangent.
    #[inline]
    pub fn tan(&self) -> Tensor<T> {
        self.map(Float::tan)
    }

    /// Element-wise natural exponential.
    #[inline]
    pub fn exp(&self) -> Tensor<T> {
        self.map(Float::exp)
    }

    /// Element-wise natural logarithm.
    #[inline]
    pub fn ln(&self) -> Tensor<T> {
        self.map(Float::ln)
    }

    /// Element-wise base-2 logarithm.
    #[inline]
    pub fn log2(&self) -> Tensor<T> {
        self.map(Float::log2)
    }

    /// Element-wise base-10 logarithm.
    #[inline]
    pub fn log10(&self) -> Tensor<T> {
        self.map(Float::log10)
    }

    /// Element-wise floor.
    #[inline]
    pub fn floor(&self) -> Tensor<T> {
        self.map(Float::floor)
    }

    /// Element-wise ceiling.
    #[inline]
    pub fn ceil(&self) -> Tensor<T> {
        self.map(Float::ceil)
    }

    /// Element-wise rounding to nearest integer.
    #[inline]
    pub fn round(&self) -> Tensor<T> {
        self.map(Float::round)
    }

    /// Element-wise reciprocal (`1/x`).
    #[inline]
    pub fn recip(&self) -> Tensor<T> {
        self.map(Float::recip)
    }

    /// Raise every element to a floating-point power.
    #[inline]
    pub fn powf(&self, exponent: T) -> Tensor<T> {
        self.map(|x| x.powf(exponent))
    }

    /// Raise every element to an integer power.
    #[inline]
    pub fn powi(&self, n: i32) -> Tensor<T> {
        self.map(|x| x.powi(n))
    }

    /// Clamp every element to `[min, max]`.
    #[inline]
    pub fn clamp(&self, min: T, max: T) -> Tensor<T> {
        self.map(|x| x.max(min).min(max))
    }
}

// ======================================================================
// Free functions
// ======================================================================

/// Element-wise absolute value.
pub fn abs<T: Float>(t: &Tensor<T>) -> Tensor<T> {
    t.abs()
}

/// Element-wise square root.
pub fn sqrt<T: Float>(t: &Tensor<T>) -> Tensor<T> {
    t.sqrt()
}

/// Element-wise sine.
pub fn sin<T: Float>(t: &Tensor<T>) -> Tensor<T> {
    t.sin()
}

/// Element-wise cosine.
pub fn cos<T: Float>(t: &Tensor<T>) -> Tensor<T> {
    t.cos()
}

/// Element-wise tangent.
pub fn tan<T: Float>(t: &Tensor<T>) -> Tensor<T> {
    t.tan()
}

/// Element-wise natural exponential.
pub fn exp<T: Float>(t: &Tensor<T>) -> Tensor<T> {
    t.exp()
}

/// Element-wise natural logarithm.
pub fn ln<T: Float>(t: &Tensor<T>) -> Tensor<T> {
    t.ln()
}

/// Element-wise base-2 logarithm.
pub fn log2<T: Float>(t: &Tensor<T>) -> Tensor<T> {
    t.log2()
}

/// Element-wise base-10 logarithm.
pub fn log10<T: Float>(t: &Tensor<T>) -> Tensor<T> {
    t.log10()
}

/// Element-wise floor.
pub fn floor<T: Float>(t: &Tensor<T>) -> Tensor<T> {
    t.floor()
}

/// Element-wise ceiling.
pub fn ceil<T: Float>(t: &Tensor<T>) -> Tensor<T> {
    t.ceil()
}

/// Element-wise rounding.
pub fn round<T: Float>(t: &Tensor<T>) -> Tensor<T> {
    t.round()
}

/// Element-wise reciprocal.
pub fn recip<T: Float>(t: &Tensor<T>) -> Tensor<T> {
    t.recip()
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn test_sin_cos_known_values() {
        let t = Tensor::from_vec(vec![0.0_f64, std::f64::consts::FRAC_PI_2], vec![2]).unwrap();
        let s = t.sin();
        assert!((s.as_slice()[0] - 0.0).abs() < 1e-15);
        assert!((s.as_slice()[1] - 1.0).abs() < 1e-15);

        let c = t.cos();
        assert!((c.as_slice()[0] - 1.0).abs() < 1e-15);
        assert!(c.as_slice()[1].abs() < 1e-15);
    }

    #[test]
    fn test_exp_ln() {
        let t = Tensor::from_vec(vec![0.0_f64, 1.0], vec![2]).unwrap();
        let e = t.exp();
        assert!((e.as_slice()[0] - 1.0).abs() < 1e-15);
        assert!((e.as_slice()[1] - std::f64::consts::E).abs() < 1e-14);

        let l = e.ln();
        assert!((l.as_slice()[0] - 0.0).abs() < 1e-15);
        assert!((l.as_slice()[1] - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_sqrt() {
        let t = Tensor::from_vec(vec![0.0_f64, 1.0, 4.0, 9.0, 16.0], vec![5]).unwrap();
        let s = t.sqrt();
        assert_eq!(s.as_slice(), &[0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_abs() {
        let t = Tensor::from_vec(vec![-3.0_f64, -1.0, 0.0, 2.0, 5.0], vec![5]).unwrap();
        let a = t.abs();
        assert_eq!(a.as_slice(), &[3.0, 1.0, 0.0, 2.0, 5.0]);
    }

    #[test]
    fn test_powf_powi() {
        let t = Tensor::from_vec(vec![2.0_f64, 3.0], vec![2]).unwrap();
        let p = t.powf(3.0);
        assert!((p.as_slice()[0] - 8.0).abs() < 1e-14);
        assert!((p.as_slice()[1] - 27.0).abs() < 1e-14);

        let p2 = t.powi(2);
        assert!((p2.as_slice()[0] - 4.0).abs() < 1e-14);
        assert!((p2.as_slice()[1] - 9.0).abs() < 1e-14);
    }

    #[test]
    fn test_floor_ceil_round() {
        let t = Tensor::from_vec(vec![1.3_f64, 2.7, -0.5], vec![3]).unwrap();
        assert_eq!(t.floor().as_slice(), &[1.0, 2.0, -1.0]);
        assert_eq!(t.ceil().as_slice(), &[2.0, 3.0, 0.0]);
        // Rust rounds half-to-even for some cases, so test with clear values
        let t2 = Tensor::from_vec(vec![1.3_f64, 2.7, 3.5], vec![3]).unwrap();
        let r = t2.round();
        assert_eq!(r.as_slice()[0], 1.0);
        assert_eq!(r.as_slice()[1], 3.0);
        assert_eq!(r.as_slice()[2], 4.0);
    }

    #[test]
    fn test_recip() {
        let t = Tensor::from_vec(vec![2.0_f64, 4.0, 5.0], vec![3]).unwrap();
        let r = t.recip();
        assert_eq!(r.as_slice(), &[0.5, 0.25, 0.2]);
    }

    #[test]
    fn test_clamp() {
        let t = Tensor::from_vec(vec![-5.0_f64, 0.5, 3.0, 10.0], vec![4]).unwrap();
        let c = t.clamp(0.0, 2.0);
        assert_eq!(c.as_slice(), &[0.0, 0.5, 2.0, 2.0]);
    }

    #[test]
    fn test_shape_preserved() {
        let t = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        assert_eq!(t.sin().shape(), &[2, 3]);
        assert_eq!(t.exp().shape(), &[2, 3]);
        assert_eq!(t.sqrt().shape(), &[2, 3]);
    }

    #[test]
    fn test_log2_log10() {
        let t = Tensor::from_vec(vec![1.0_f64, 2.0, 4.0, 8.0], vec![4]).unwrap();
        let l2 = t.log2();
        assert_eq!(l2.as_slice(), &[0.0, 1.0, 2.0, 3.0]);

        let t2 = Tensor::from_vec(vec![1.0_f64, 10.0, 100.0], vec![3]).unwrap();
        let l10 = t2.log10();
        assert!((l10.as_slice()[0] - 0.0).abs() < 1e-15);
        assert!((l10.as_slice()[1] - 1.0).abs() < 1e-15);
        assert!((l10.as_slice()[2] - 2.0).abs() < 1e-14);
    }

    #[test]
    fn test_free_functions() {
        let t = Tensor::from_vec(vec![0.0_f64, 1.0], vec![2]).unwrap();
        let s = sin(&t);
        assert!((s.as_slice()[0]).abs() < 1e-15);

        let e = exp(&t);
        assert!((e.as_slice()[0] - 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_f32_works() {
        let t = Tensor::from_vec(vec![0.0_f32, 1.0, 4.0], vec![3]).unwrap();
        let s = t.sqrt();
        assert_eq!(s.as_slice(), &[0.0_f32, 1.0, 2.0]);
    }
}
