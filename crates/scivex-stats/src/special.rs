//! Special mathematical functions used internally by distributions.
//!
//! All functions are generic over `T: Float`.

use scivex_core::Float;

use crate::error::{Result, StatsError};

const MAX_ITER: usize = 200;

// ---------------------------------------------------------------------------
// Error function
// ---------------------------------------------------------------------------

/// Compute the error function erf(x) using the Abramowitz & Stegun 7.1.26
/// rational approximation.
///
/// # Examples
///
/// ```ignore
/// # use scivex_stats::special::erf;
/// assert!(erf(0.0_f64).abs() < 1e-10); // erf(0) = 0
/// assert!((erf(100.0_f64) - 1.0).abs() < 1e-10); // erf(∞) ≈ 1
/// ```
pub fn erf<T: Float>(x: T) -> T {
    let zero = T::from_f64(0.0);
    let one = T::from_f64(1.0);

    if x < zero {
        return -erf(-x);
    }

    let p = T::from_f64(0.327_591_1);
    let a1 = T::from_f64(0.254_829_592);
    let a2 = T::from_f64(-0.284_496_736);
    let a3 = T::from_f64(1.421_413_741);
    let a4 = T::from_f64(-1.453_152_027);
    let a5 = T::from_f64(1.061_405_429);

    let t = one / (one + p * x);
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t3 * t;
    let t5 = t4 * t;

    let poly = a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5;
    one - poly * (-x * x).exp()
}

/// Compute the complementary error function erfc(x) = 1 - erf(x).
///
/// # Examples
///
/// ```ignore
/// # use scivex_stats::special::erfc;
/// assert!((erfc(0.0_f64) - 1.0).abs() < 1e-10);
/// ```
#[allow(dead_code)]
pub fn erfc<T: Float>(x: T) -> T {
    T::from_f64(1.0) - erf(x)
}

// ---------------------------------------------------------------------------
// Gamma function (Lanczos approximation)
// ---------------------------------------------------------------------------

/// Lanczos coefficients (g=7, 9 terms).
const LANCZOS_G: f64 = 7.0;
const LANCZOS_COEFF: [f64; 9] = [
    0.999_999_999_999_809_9,
    676.520_368_121_885_1,
    -1_259.139_216_722_402_9,
    771.323_428_777_653_1,
    -176.615_029_162_140_6,
    12.507_343_278_686_905,
    -0.138_571_095_265_720_12,
    9.984_369_578_019_572e-6,
    1.505_632_735_149_311_6e-7,
];

/// Compute ln(Gamma(x)) for x > 0 using the Lanczos approximation.
///
/// # Examples
///
/// ```ignore
/// # use scivex_stats::special::ln_gamma;
/// // Gamma(5) = 4! = 24, so ln(Gamma(5)) = ln(24)
/// let val = ln_gamma(5.0_f64);
/// assert!((val - 24.0_f64.ln()).abs() < 1e-8);
/// ```
pub fn ln_gamma<T: Float>(x: T) -> T {
    let half = T::from_f64(0.5);
    let g = T::from_f64(LANCZOS_G);
    let ln_sqrt_2pi = T::from_f64(0.918_938_533_204_672_8); // ln(sqrt(2*pi))

    let xm1 = x - T::from_f64(1.0);
    let mut sum = T::from_f64(LANCZOS_COEFF[0]);
    for (i, &c) in LANCZOS_COEFF.iter().enumerate().skip(1) {
        sum += T::from_f64(c) / (xm1 + T::from_f64(i as f64));
    }

    let t = xm1 + g + half;
    ln_sqrt_2pi + (xm1 + half) * t.ln() - t + sum.ln()
}

/// Compute Gamma(x) = exp(ln_gamma(x)).
///
/// # Examples
///
/// ```ignore
/// # use scivex_stats::special::gamma;
/// assert!((gamma(5.0_f64) - 24.0).abs() < 1e-6);
/// ```
#[allow(dead_code)]
pub fn gamma<T: Float>(x: T) -> T {
    ln_gamma(x).exp()
}

// ---------------------------------------------------------------------------
// Beta function
// ---------------------------------------------------------------------------

/// Compute ln(Beta(a, b)) = ln(Gamma(a)) + ln(Gamma(b)) - ln(Gamma(a+b)).
///
/// # Examples
///
/// ```ignore
/// # use scivex_stats::special::ln_beta;
/// let val = ln_beta(2.0_f64, 3.0);
/// // Beta(2,3) = 1/12, so ln(Beta(2,3)) = ln(1/12)
/// assert!((val - (1.0_f64 / 12.0).ln()).abs() < 1e-8);
/// ```
pub fn ln_beta<T: Float>(a: T, b: T) -> T {
    ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
}

// ---------------------------------------------------------------------------
// Regularized incomplete gamma function P(a, x)
// ---------------------------------------------------------------------------

/// Lower regularized incomplete gamma function P(a, x) = gamma(a,x)/Gamma(a).
///
/// Uses series expansion when x < a+1, continued fraction otherwise.
///
/// # Examples
///
/// ```ignore
/// # use scivex_stats::special::regularized_gamma_p;
/// let p = regularized_gamma_p(1.0_f64, 1.0).unwrap();
/// // P(1, 1) = 1 - e^(-1) ≈ 0.6321
/// assert!((p - (1.0 - (-1.0_f64).exp())).abs() < 1e-8);
/// ```
pub fn regularized_gamma_p<T: Float>(a: T, x: T) -> Result<T> {
    let zero = T::from_f64(0.0);
    let one = T::from_f64(1.0);
    let eps = T::epsilon();

    if x < zero {
        return Ok(zero);
    }
    if x == zero {
        return Ok(zero);
    }

    if x < a + one {
        // Series expansion
        gamma_series(a, x, eps)
    } else {
        // Continued fraction, P = 1 - Q
        let q = gamma_cf(a, x, eps)?;
        Ok(one - q)
    }
}

/// Upper regularized incomplete gamma function Q(a, x) = 1 - P(a, x).
///
/// # Examples
///
/// ```ignore
/// # use scivex_stats::special::regularized_gamma_q;
/// let q = regularized_gamma_q(1.0_f64, 1.0).unwrap();
/// // Q(1, 1) = e^(-1) ≈ 0.3679
/// assert!((q - (-1.0_f64).exp()).abs() < 1e-8);
/// ```
pub fn regularized_gamma_q<T: Float>(a: T, x: T) -> Result<T> {
    let one = T::from_f64(1.0);
    Ok(one - regularized_gamma_p(a, x)?)
}

/// Series expansion for P(a, x).
fn gamma_series<T: Float>(a: T, x: T, eps: T) -> Result<T> {
    let one = T::from_f64(1.0);
    let lng = ln_gamma(a);

    let mut ap = a;
    let mut sum = one / a;
    let mut term = one / a;

    for i in 0..MAX_ITER {
        ap += one;
        term = term * x / ap;
        sum += term;
        if term.abs() < sum.abs() * eps {
            return Ok(sum * (-x + a * x.ln() - lng).exp());
        }
        if i == MAX_ITER - 1 {
            return Err(StatsError::ConvergenceFailure {
                iterations: MAX_ITER,
            });
        }
    }
    unreachable!()
}

/// Continued fraction for Q(a, x) using the modified Lentz algorithm.
fn gamma_cf<T: Float>(a: T, x: T, eps: T) -> Result<T> {
    let one = T::from_f64(1.0);
    let tiny = T::from_f64(1e-30);
    let lng = ln_gamma(a);

    let mut b = x + one - a;
    let mut c = one / tiny;
    let mut d = one / b;
    let mut h = d;

    for i in 1..=MAX_ITER {
        let an = -T::from_f64(i as f64) * (T::from_f64(i as f64) - a);
        b += T::from_f64(2.0);
        d = an * d + b;
        if d.abs() < tiny {
            d = tiny;
        }
        c = b + an / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = one / d;
        let del = d * c;
        h *= del;
        if (del - one).abs() < eps {
            return Ok(h * (-x + a * x.ln() - lng).exp());
        }
    }
    Err(StatsError::ConvergenceFailure {
        iterations: MAX_ITER,
    })
}

// ---------------------------------------------------------------------------
// Regularized incomplete beta function I_x(a, b)
// ---------------------------------------------------------------------------

/// Regularized incomplete beta function I_x(a, b).
///
/// Uses the Lentz continued fraction expansion.
///
/// # Examples
///
/// ```ignore
/// # use scivex_stats::special::regularized_beta;
/// let val = regularized_beta(0.5_f64, 1.0, 1.0).unwrap();
/// // I_{0.5}(1,1) = 0.5 (uniform distribution)
/// assert!((val - 0.5).abs() < 1e-8);
/// ```
pub fn regularized_beta<T: Float>(x: T, a: T, b: T) -> Result<T> {
    let zero = T::from_f64(0.0);
    let one = T::from_f64(1.0);

    if x <= zero {
        return Ok(zero);
    }
    if x >= one {
        return Ok(one);
    }

    // Use symmetry: I_x(a,b) = 1 - I_{1-x}(b,a)  when x > (a+1)/(a+b+2)
    let threshold = (a + one) / (a + b + T::from_f64(2.0));
    if x > threshold {
        let val = beta_cf(one - x, b, a)?;
        return Ok(one - val);
    }

    beta_cf(x, a, b)
}

/// Continued fraction for I_x(a,b) using the modified Lentz algorithm.
fn beta_cf<T: Float>(x: T, a: T, b: T) -> Result<T> {
    let one = T::from_f64(1.0);
    let two = T::from_f64(2.0);
    let tiny = T::from_f64(1e-30);
    let eps = T::epsilon();

    let lnb = ln_beta(a, b);
    let prefix = (a * x.ln() + b * (one - x).ln() - lnb).exp() / a;

    let mut c = one;
    let mut d = one / (one - (a + b) * x / (a + one)).max(tiny);
    let mut h = d;

    for m in 1..=MAX_ITER {
        let mf = T::from_f64(m as f64);

        // Even step: d_{2m}
        let num_even = mf * (b - mf) * x / ((a + two * mf - one) * (a + two * mf));
        d = one / (one + num_even * d).max(tiny);
        c = (one + num_even / c).max(tiny);
        h = h * d * c;

        // Odd step: d_{2m+1}
        let num_odd = -((a + mf) * (a + b + mf) * x) / ((a + two * mf) * (a + two * mf + one));
        d = one / (one + num_odd * d).max(tiny);
        c = (one + num_odd / c).max(tiny);
        let del = d * c;
        h *= del;

        if (del - one).abs() < eps {
            return Ok(prefix * h);
        }
    }
    Err(StatsError::ConvergenceFailure {
        iterations: MAX_ITER,
    })
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) {
        assert!(
            (a - b).abs() < tol,
            "expected {b}, got {a}, diff {}",
            (a - b).abs()
        );
    }

    #[test]
    fn test_erf_zero() {
        approx(erf(0.0_f64), 0.0, 1e-7);
    }

    #[test]
    fn test_erf_one() {
        approx(erf(1.0_f64), 0.842_700_792_949_715, 1e-6);
    }

    #[test]
    fn test_erf_negative() {
        approx(erf(-1.0_f64), -0.842_700_792_949_715, 1e-6);
    }

    #[test]
    fn test_ln_gamma_one() {
        approx(ln_gamma(1.0_f64), 0.0, 1e-10);
    }

    #[test]
    fn test_gamma_five() {
        // Gamma(5) = 4! = 24
        approx(gamma(5.0_f64), 24.0, 1e-8);
    }

    #[test]
    fn test_regularized_gamma_p_1_1() {
        // P(1, 1) = 1 - e^{-1} ≈ 0.6321
        let val = regularized_gamma_p(1.0_f64, 1.0).unwrap();
        approx(val, 0.632_120_558_828_558, 1e-6);
    }

    #[test]
    fn test_regularized_beta_half() {
        // I_{0.5}(1, 1) = 0.5 (uniform distribution)
        let val = regularized_beta(0.5_f64, 1.0, 1.0).unwrap();
        approx(val, 0.5, 1e-10);
    }

    #[test]
    fn test_regularized_beta_boundaries() {
        let val0 = regularized_beta(0.0_f64, 2.0, 3.0).unwrap();
        let val1 = regularized_beta(1.0_f64, 2.0, 3.0).unwrap();
        approx(val0, 0.0, 1e-10);
        approx(val1, 1.0, 1e-10);
    }
}
