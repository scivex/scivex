//! Adaptive Gauss-Kronrod quadrature (G7-K15).
//!
//! The 7-point Gauss rule and 15-point Kronrod extension share function
//! evaluations, giving both an integral estimate and an error estimate.

use scivex_core::Float;

use crate::error::Result;

use super::{QuadOptions, QuadResult};

// Gauss-Kronrod G7-K15 nodes and weights (positive half only, symmetric).
// Nodes are on [-1, 1].

/// Kronrod nodes (positive half + zero). 8 values for the 15-point rule.
const KRONROD_NODES: [f64; 8] = [
    0.0,
    0.207_784_955_007_898_47,
    0.405_845_151_377_397_2,
    0.586_087_235_467_691_1,
    0.741_531_185_599_394_4,
    0.864_864_423_359_769_1,
    0.949_107_912_342_758_5,
    0.991_455_371_120_812_6,
];

/// Kronrod weights corresponding to the nodes above.
const KRONROD_WEIGHTS: [f64; 8] = [
    0.209_482_141_084_727_83,
    0.204_432_940_075_298_9,
    0.190_350_578_064_785_41,
    0.169_004_726_639_267_9,
    0.140_653_259_715_525_9,
    0.104_790_010_322_250_18,
    0.063_092_092_629_978_56,
    0.022_935_322_010_529_225,
];

/// Gauss weights for the 7-point rule (nodes at indices 1, 3, 5, 7 of Kronrod).
/// Index 0 = center node (Kronrod index 0 is not a Gauss node for G7).
const GAUSS_WEIGHTS: [f64; 4] = [
    0.417_959_183_673_469_4,  // center (node index 0)
    0.381_830_050_505_118_9,  // node index 1
    0.279_705_391_489_276_67, // node index 3
    0.129_484_966_168_869_7,  // node index 5
];

/// Gauss node indices in the Kronrod array.
const GAUSS_INDICES: [usize; 4] = [0, 2, 4, 6];

/// Adaptive quadrature using the Gauss-Kronrod G7-K15 rule.
///
/// Approximates `∫_a^b f(x) dx` by recursively subdividing intervals
/// until the error estimate is within the requested tolerances.
pub fn quad<T: Float, F: Fn(T) -> T>(
    f: F,
    a: T,
    b: T,
    options: &QuadOptions<T>,
) -> Result<QuadResult<T>> {
    let mut total_evals = 0usize;

    let value = adaptive_gk(
        &f,
        a,
        b,
        options.abs_tol,
        options.rel_tol,
        options.max_subdivisions,
        0,
        &mut total_evals,
    );

    // For the final error estimate, re-evaluate with one pass to get the
    // Kronrod-Gauss difference on the full interval.
    let (k_full, g_full) = gk15_pair(&f, a, b, &mut 0);
    let error_estimate = (k_full - g_full).abs();

    Ok(QuadResult {
        value,
        error_estimate,
        n_evals: total_evals,
    })
}

/// Compute both the K15 and G7 estimates on `[a, b]`.
fn gk15_pair<T: Float, F: Fn(T) -> T>(f: &F, a: T, b: T, n_evals: &mut usize) -> (T, T) {
    let two = T::from_f64(2.0);
    let center = (a + b) / two;
    let half_len = (b - a) / two;

    let mut kronrod_sum = T::zero();
    let mut gauss_sum = T::zero();

    // Center node
    let f_center = f(center);
    *n_evals += 1;
    kronrod_sum += f_center * T::from_f64(KRONROD_WEIGHTS[0]);
    gauss_sum += f_center * T::from_f64(GAUSS_WEIGHTS[0]);

    for i in 1..8 {
        let node = T::from_f64(KRONROD_NODES[i]);
        let weight = T::from_f64(KRONROD_WEIGHTS[i]);

        let x_plus = center + half_len * node;
        let x_minus = center - half_len * node;
        let f_sum = f(x_plus) + f(x_minus);
        *n_evals += 2;

        kronrod_sum += f_sum * weight;

        // Check if this is also a Gauss node
        if let Some(gi) = GAUSS_INDICES.iter().position(|&idx| idx == i) {
            gauss_sum += f_sum * T::from_f64(GAUSS_WEIGHTS[gi]);
        }
    }

    (kronrod_sum * half_len, gauss_sum * half_len)
}

/// Recursive adaptive subdivision.
#[allow(clippy::too_many_arguments)]
fn adaptive_gk<T: Float, F: Fn(T) -> T>(
    f: &F,
    a: T,
    b: T,
    abs_tol: T,
    rel_tol: T,
    max_depth: usize,
    depth: usize,
    n_evals: &mut usize,
) -> T {
    let (kronrod, gauss) = gk15_pair(f, a, b, n_evals);
    let error = (kronrod - gauss).abs();
    let tolerance = abs_tol.max(rel_tol * kronrod.abs());

    if error < tolerance || depth >= max_depth {
        return kronrod;
    }

    // Subdivide
    let two = T::from_f64(2.0);
    let mid = (a + b) / two;
    let half_tol = abs_tol / T::from_f64(std::f64::consts::SQRT_2);

    let left = adaptive_gk(f, a, mid, half_tol, rel_tol, max_depth, depth + 1, n_evals);
    let right = adaptive_gk(f, mid, b, half_tol, rel_tol, max_depth, depth + 1, n_evals);

    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quad_sin() {
        // ∫_0^π sin(x) dx = 2
        let result = quad(
            |x: f64| x.sin(),
            0.0,
            std::f64::consts::PI,
            &QuadOptions::default(),
        )
        .unwrap();
        assert!((result.value - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_quad_gaussian() {
        // ∫_0^1 e^(-x^2) dx ≈ 0.746824132812427
        let result = quad(|x: f64| (-x * x).exp(), 0.0, 1.0, &QuadOptions::default()).unwrap();
        let expected = 0.746_824_132_812_427;
        assert!(
            (result.value - expected).abs() < 1e-10,
            "got {}, expected {}",
            result.value,
            expected
        );
    }

    #[test]
    fn test_quad_polynomial() {
        // ∫_0^1 x^4 dx = 0.2
        let result = quad(|x: f64| x.powi(4), 0.0, 1.0, &QuadOptions::default()).unwrap();
        assert!((result.value - 0.2).abs() < 1e-12);
    }
}
