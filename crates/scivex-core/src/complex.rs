//! Native complex number type for generic tensor operations.
//!
//! Provides [`Complex<T>`] which implements [`Scalar`] so it can be used
//! directly as a tensor element type: `Tensor<Complex<f64>>`.

use crate::dtype::{Float, Scalar};
use core::fmt;
use core::iter::Sum;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

// ---------------------------------------------------------------------------
// Complex<T> struct
// ---------------------------------------------------------------------------

/// A complex number with real and imaginary parts of type `T`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Complex<T> {
    /// Real part.
    pub re: T,
    /// Imaginary part.
    pub im: T,
}

// ---------------------------------------------------------------------------
// Constructors and methods
// ---------------------------------------------------------------------------

impl<T: Float> Complex<T> {
    /// Create a new complex number from real and imaginary parts.
    #[inline]
    pub fn new(re: T, im: T) -> Self {
        Self { re, im }
    }

    /// Create a complex number from a real value (imaginary part is zero).
    #[inline]
    pub fn from_real(re: T) -> Self {
        Self { re, im: T::zero() }
    }

    /// Create a complex number from polar form: `r * (cos(theta) + i*sin(theta))`.
    #[inline]
    pub fn from_polar(r: T, theta: T) -> Self {
        Self {
            re: r * theta.cos(),
            im: r * theta.sin(),
        }
    }

    /// Complex conjugate: flips the sign of the imaginary part.
    #[inline]
    pub fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    /// Squared modulus: `re² + im²`. Avoids the `sqrt` in [`norm`](Self::norm).
    #[inline]
    pub fn norm_sqr(self) -> T {
        self.re * self.re + self.im * self.im
    }

    /// Modulus (absolute value): `sqrt(re² + im²)`.
    #[inline]
    pub fn norm(self) -> T {
        self.norm_sqr().sqrt()
    }

    /// Phase angle (argument): `atan2(im, re)`.
    #[inline]
    pub fn arg(self) -> T {
        // atan2 is not on Float trait; compute via the identity:
        // atan2(y, x) = 2 * atan(y / (sqrt(x²+y²) + x))  for x > -|z|
        // Fallback: use to_f64 / from_f64 for atan2.
        let angle = self.im.to_f64().atan2(self.re.to_f64());
        T::from_f64(angle)
    }

    /// Complex exponential: `e^(a+bi) = e^a * (cos b + i sin b)`.
    #[inline]
    pub fn exp(self) -> Self {
        let ea = self.re.exp();
        Self {
            re: ea * self.im.cos(),
            im: ea * self.im.sin(),
        }
    }

    /// Complex natural logarithm: `ln|z| + i*arg(z)`.
    #[inline]
    pub fn ln(self) -> Self {
        Self {
            re: self.norm().ln(),
            im: self.arg(),
        }
    }

    /// Complex square root.
    ///
    /// Uses the principal branch: `sqrt(r) * (cos(theta/2) + i*sin(theta/2))`.
    #[inline]
    pub fn sqrt(self) -> Self {
        let r = self.norm().sqrt();
        let half_theta = self.arg() * T::from_f64(0.5);
        Self {
            re: r * half_theta.cos(),
            im: r * half_theta.sin(),
        }
    }

    /// Complex power: `z^n = e^(n * ln(z))`.
    #[inline]
    pub fn pow(self, n: T) -> Self {
        let ln_z = self.ln();
        let scaled = Self {
            re: ln_z.re * n,
            im: ln_z.im * n,
        };
        scaled.exp()
    }

    /// Returns `true` if both real and imaginary parts are finite.
    #[inline]
    pub fn is_finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }

    /// Returns `true` if either part is NaN.
    #[inline]
    pub fn is_nan(self) -> bool {
        self.re.is_nan() || self.im.is_nan()
    }
}

// ---------------------------------------------------------------------------
// The imaginary unit
// ---------------------------------------------------------------------------

impl<T: Float> Complex<T> {
    /// The imaginary unit `i`.
    #[inline]
    pub fn i() -> Self {
        Self {
            re: T::zero(),
            im: T::one(),
        }
    }
}

// ---------------------------------------------------------------------------
// Display
// ---------------------------------------------------------------------------

impl<T: Float + fmt::Display> fmt::Display for Complex<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // We need to determine the sign of the imaginary part.
        // Compare im >= zero using to_f64.
        let im_f64 = self.im.to_f64();
        if im_f64 < 0.0 {
            write!(f, "{}-{}i", self.re, Float::abs(self.im))
        } else {
            write!(f, "{}+{}i", self.re, self.im)
        }
    }
}

// ---------------------------------------------------------------------------
// PartialOrd — compare by norm (non-standard but required by Scalar)
// ---------------------------------------------------------------------------

impl<T: Float> PartialOrd for Complex<T> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.norm_sqr().partial_cmp(&other.norm_sqr())
    }
}

// ---------------------------------------------------------------------------
// Arithmetic: Complex + Complex
// ---------------------------------------------------------------------------

impl<T: Float> Add for Complex<T> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

impl<T: Float> Sub for Complex<T> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }
}

impl<T: Float> Mul for Complex<T> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

impl<T: Float> Div for Complex<T> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        // (a+bi)/(c+di) = ((ac+bd) + (bc-ad)i) / (c²+d²)
        let denom = rhs.norm_sqr();
        Self {
            re: (self.re * rhs.re + self.im * rhs.im) / denom,
            im: (self.im * rhs.re - self.re * rhs.im) / denom,
        }
    }
}

// ---------------------------------------------------------------------------
// Arithmetic: Complex + T scalar
// ---------------------------------------------------------------------------

impl<T: Float> Add<T> for Complex<T> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: T) -> Self {
        Self {
            re: self.re + rhs,
            im: self.im,
        }
    }
}

impl<T: Float> Sub<T> for Complex<T> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: T) -> Self {
        Self {
            re: self.re - rhs,
            im: self.im,
        }
    }
}

impl<T: Float> Mul<T> for Complex<T> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: T) -> Self {
        Self {
            re: self.re * rhs,
            im: self.im * rhs,
        }
    }
}

impl<T: Float> Div<T> for Complex<T> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: T) -> Self {
        Self {
            re: self.re / rhs,
            im: self.im / rhs,
        }
    }
}

// ---------------------------------------------------------------------------
// Assign ops
// ---------------------------------------------------------------------------

impl<T: Float> AddAssign for Complex<T> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.re += rhs.re;
        self.im += rhs.im;
    }
}

impl<T: Float> SubAssign for Complex<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.re -= rhs.re;
        self.im -= rhs.im;
    }
}

impl<T: Float> MulAssign for Complex<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<T: Float> DivAssign for Complex<T> {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

// ---------------------------------------------------------------------------
// Neg
// ---------------------------------------------------------------------------

impl<T: Float> Neg for Complex<T> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self {
            re: -self.re,
            im: -self.im,
        }
    }
}

// ---------------------------------------------------------------------------
// Sum
// ---------------------------------------------------------------------------

impl<T: Float> Sum for Complex<T> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Complex::new(T::zero(), T::zero()), |acc, x| acc + x)
    }
}

// ---------------------------------------------------------------------------
// Scalar impl
// ---------------------------------------------------------------------------

impl<T: Float> Scalar for Complex<T> {
    #[inline]
    fn zero() -> Self {
        Complex::new(T::zero(), T::zero())
    }

    #[inline]
    fn one() -> Self {
        Complex::new(T::one(), T::zero())
    }

    #[inline]
    fn from_usize(v: usize) -> Self {
        Complex::new(T::from_usize(v), T::zero())
    }
}

// ---------------------------------------------------------------------------
// Interleaved conversion utilities
// ---------------------------------------------------------------------------

/// Convert interleaved real data `[re0, im0, re1, im1, ...]` to a `Vec<Complex<T>>`.
///
/// The input slice length must be even. If it is odd the last element is ignored.
pub fn from_interleaved<T: Float>(data: &[T]) -> Vec<Complex<T>> {
    let n = data.len() / 2;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(Complex::new(data[i * 2], data[i * 2 + 1]));
    }
    out
}

/// Convert a slice of `Complex<T>` to interleaved real data `[re0, im0, re1, im1, ...]`.
pub fn to_interleaved<T: Float>(data: &[Complex<T>]) -> Vec<T> {
    let mut out = Vec::with_capacity(data.len() * 2);
    for c in data {
        out.push(c.re);
        out.push(c.im);
    }
    out
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-10;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < EPS
    }

    fn approx_c(a: Complex<f64>, b: Complex<f64>) -> bool {
        approx(a.re, b.re) && approx(a.im, b.im)
    }

    #[test]
    fn test_complex_arithmetic() {
        let a = Complex::new(1.0, 2.0);
        let b = Complex::new(3.0, 4.0);

        // add
        let s = a + b;
        assert!(approx(s.re, 4.0));
        assert!(approx(s.im, 6.0));

        // sub
        let d = a - b;
        assert!(approx(d.re, -2.0));
        assert!(approx(d.im, -2.0));

        // mul: (1+2i)(3+4i) = 3+4i+6i+8i² = 3+10i-8 = -5+10i
        let m = a * b;
        assert!(approx(m.re, -5.0));
        assert!(approx(m.im, 10.0));

        // div: (1+2i)/(3+4i) = (1+2i)(3-4i)/25 = (3-4i+6i-8i²)/25 = (11+2i)/25
        let q = a / b;
        assert!(approx(q.re, 11.0 / 25.0));
        assert!(approx(q.im, 2.0 / 25.0));
    }

    #[test]
    fn test_complex_conjugate() {
        let z = Complex::new(3.0, -7.0);
        let c = z.conj();
        assert!(approx(c.re, 3.0));
        assert!(approx(c.im, 7.0));
    }

    #[test]
    fn test_complex_norm() {
        let z = Complex::new(3.0, 4.0);
        assert!(approx(z.norm(), 5.0));
        assert!(approx(z.norm_sqr(), 25.0));
    }

    #[test]
    fn test_complex_arg() {
        let z = Complex::new(1.0, 1.0);
        let expected = std::f64::consts::FRAC_PI_4;
        assert!(approx(z.arg(), expected));
    }

    #[test]
    fn test_complex_exp() {
        // e^(i*pi) ≈ -1 + 0i
        let z = Complex::new(0.0, std::f64::consts::PI);
        let r = z.exp();
        assert!(approx(r.re, -1.0));
        assert!(approx(r.im, 0.0));
    }

    #[test]
    fn test_complex_from_polar() {
        let r = 5.0;
        let theta = std::f64::consts::FRAC_PI_4;
        let z = Complex::from_polar(r, theta);

        // Roundtrip: norm and arg should recover r and theta
        assert!(approx(z.norm(), r));
        assert!(approx(z.arg(), theta));
    }

    #[test]
    fn test_complex_scalar_mul() {
        let z = Complex::new(2.0, 3.0);
        let scaled = z * 4.0;
        assert!(approx(scaled.re, 8.0));
        assert!(approx(scaled.im, 12.0));
    }

    #[test]
    fn test_complex_sqrt() {
        // sqrt(-1) = i
        let z = Complex::new(-1.0, 0.0);
        let s = z.sqrt();
        assert!(approx_c(s, Complex::new(0.0, 1.0)));

        // sqrt(4) = 2
        let z2 = Complex::new(4.0, 0.0);
        let s2 = z2.sqrt();
        assert!(approx_c(s2, Complex::new(2.0, 0.0)));
    }

    #[test]
    fn test_complex_display() {
        let a = Complex::new(3.0_f64, 4.0_f64);
        let s = format!("{a}");
        assert!(s.contains('+'));
        assert!(s.contains('i'));

        let b = Complex::new(1.0_f64, -2.0_f64);
        let s2 = format!("{b}");
        assert!(s2.contains('-'));
        assert!(s2.contains('i'));
    }

    #[test]
    fn test_complex_tensor() {
        use crate::tensor::Tensor;

        let data = vec![
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 1.0),
            Complex::new(1.0, 1.0),
            Complex::new(2.0, -1.0),
        ];
        let t = Tensor::from_vec(data, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        // Element-wise add with itself
        let t2 = &t + &t;
        let elem = t2.get(&[0, 1]).unwrap();
        assert!(approx_c(*elem, Complex::new(0.0, 2.0)));
    }

    #[test]
    fn test_complex_interleaved_roundtrip() {
        let original = vec![
            Complex::new(1.0, 2.0),
            Complex::new(3.0, 4.0),
            Complex::new(5.0, 6.0),
        ];
        let interleaved = to_interleaved(&original);
        assert_eq!(interleaved, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let recovered = from_interleaved(&interleaved);
        assert_eq!(recovered, original);
    }

    #[test]
    fn test_complex_sum() {
        let vals = vec![
            Complex::new(1.0, 2.0),
            Complex::new(3.0, 4.0),
            Complex::new(5.0, 6.0),
        ];
        let total: Complex<f64> = vals.into_iter().sum();
        assert!(approx(total.re, 9.0));
        assert!(approx(total.im, 12.0));
    }

    #[test]
    fn test_complex_scalar_trait() {
        // Verify Scalar trait methods work
        let z: Complex<f64> = Scalar::zero();
        assert!(approx(z.re, 0.0));
        assert!(approx(z.im, 0.0));

        let o: Complex<f64> = Scalar::one();
        assert!(approx(o.re, 1.0));
        assert!(approx(o.im, 0.0));

        let from_5: Complex<f64> = Scalar::from_usize(5);
        assert!(approx(from_5.re, 5.0));
        assert!(approx(from_5.im, 0.0));
    }
}
