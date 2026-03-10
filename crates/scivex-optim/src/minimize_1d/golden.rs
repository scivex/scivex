//! Golden-section search for 1-D minimization.

use scivex_core::Float;

use crate::error::Result;

use super::Minimize1dResult;

/// Find the minimum of a unimodal function `f` on `[a, b]` using
/// golden-section search.
///
/// Linear convergence, reducing the bracket by a factor of the golden
/// ratio (~0.618) each iteration.
pub fn golden_section<T: Float, F: Fn(T) -> T>(
    f: F,
    a: T,
    b: T,
    xtol: T,
    max_iter: usize,
) -> Result<Minimize1dResult<T>> {
    let gr = T::from_f64(0.618_033_988_749_895); // (sqrt(5) - 1) / 2
    let one_minus_gr = T::one() - gr;

    let mut a = a;
    let mut b = b;

    let mut x1 = a + one_minus_gr * (b - a);
    let mut x2 = a + gr * (b - a);
    let mut f1 = f(x1);
    let mut f2 = f(x2);

    for i in 0..max_iter {
        if (b - a).abs() < xtol {
            let (x_min, f_min) = if f1 < f2 { (x1, f1) } else { (x2, f2) };
            return Ok(Minimize1dResult {
                x_min,
                f_min,
                iterations: i + 1,
                converged: true,
            });
        }

        if f1 < f2 {
            b = x2;
            x2 = x1;
            f2 = f1;
            x1 = a + one_minus_gr * (b - a);
            f1 = f(x1);
        } else {
            a = x1;
            x1 = x2;
            f1 = f2;
            x2 = a + gr * (b - a);
            f2 = f(x2);
        }
    }

    let (x_min, f_min) = if f1 < f2 { (x1, f1) } else { (x2, f2) };
    Ok(Minimize1dResult {
        x_min,
        f_min,
        iterations: max_iter,
        converged: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_golden_section_quadratic() {
        // (x - 3)^2 on [0, 5] => minimum at x = 3
        let result = golden_section(|x: f64| (x - 3.0) * (x - 3.0), 0.0, 5.0, 1e-10, 200).unwrap();
        assert!(result.converged);
        assert!((result.x_min - 3.0).abs() < 1e-8);
        assert!(result.f_min < 1e-14);
    }

    #[test]
    fn test_golden_section_cubic_min() {
        // x^4 - 2x^2 has local minima at ±1. On [0, 3] the minimum is at 1.
        let result =
            golden_section(|x: f64| x.powi(4) - 2.0 * x * x, 0.0, 3.0, 1e-10, 200).unwrap();
        assert!(result.converged);
        assert!((result.x_min - 1.0).abs() < 1e-6);
    }
}
