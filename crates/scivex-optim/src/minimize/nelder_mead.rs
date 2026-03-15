//! Nelder-Mead simplex method for derivative-free unconstrained optimization.
//!
//! The downhill simplex method maintains a simplex of n+1 vertices in n-dimensional
//! space and iteratively improves the worst vertex through reflection, expansion,
//! contraction, and shrinkage operations.

use scivex_core::{Float, Tensor};

use crate::error::Result;

use super::{MinimizeOptions, MinimizeResult};

/// Minimize `f` using the Nelder-Mead simplex method.
///
/// No gradient is required. Convergence is checked using `ftol` (function value
/// spread) and `xtol` (simplex diameter). Standard coefficients are used:
/// α=1 (reflection), γ=2 (expansion), ρ=0.5 (contraction), σ=0.5 (shrinkage).
pub fn nelder_mead<T, F>(
    f: F,
    x0: &Tensor<T>,
    options: &MinimizeOptions<T>,
) -> Result<MinimizeResult<T>>
where
    T: Float,
    F: Fn(&Tensor<T>) -> T,
{
    let n = x0.numel();
    let mut f_evals = 0usize;

    let coeffs = SimplexCoeffs {
        alpha: T::one(),
        gamma: T::from_f64(2.0),
        rho: T::from_f64(0.5),
        sigma: T::from_f64(0.5),
    };

    let (mut simplex, mut f_vals) = init_simplex(&f, x0, n, &mut f_evals);

    for iter in 0..options.max_iter {
        sort_simplex(&mut simplex, &mut f_vals, n);

        if let Some(result) = check_convergence(&simplex, &f_vals, n, iter, f_evals, options) {
            return Ok(result);
        }

        let centroid = compute_centroid(&simplex, n);
        iterate_simplex(
            &f,
            &mut simplex,
            &mut f_vals,
            &centroid,
            n,
            &coeffs,
            &mut f_evals,
        );
    }

    sort_simplex(&mut simplex, &mut f_vals, n);
    let x = Tensor::from_vec(simplex[0].clone(), vec![n]).expect("best vertex length matches n");
    Ok(MinimizeResult {
        x,
        f_val: f_vals[0],
        grad: None,
        iterations: options.max_iter,
        f_evals,
        g_evals: 0,
        converged: false,
    })
}

struct SimplexCoeffs<T> {
    alpha: T,
    gamma: T,
    rho: T,
    sigma: T,
}

fn init_simplex<T: Float, F: Fn(&Tensor<T>) -> T>(
    f: &F,
    x0: &Tensor<T>,
    n: usize,
    f_evals: &mut usize,
) -> (Vec<Vec<T>>, Vec<T>) {
    let initial_step = T::from_f64(0.05);
    let mut simplex: Vec<Vec<T>> = Vec::with_capacity(n + 1);
    simplex.push(x0.as_slice().to_vec());

    for i in 0..n {
        let mut vertex = x0.as_slice().to_vec();
        let step = if vertex[i] == T::zero() {
            T::from_f64(0.00025)
        } else {
            vertex[i] * initial_step
        };
        vertex[i] += step;
        simplex.push(vertex);
    }

    let f_vals: Vec<T> = simplex
        .iter()
        .map(|v| {
            let t = Tensor::from_vec(v.clone(), vec![n])
                .expect("simplex vertex length matches dimension n");
            *f_evals += 1;
            f(&t)
        })
        .collect();

    (simplex, f_vals)
}

fn sort_simplex<T: Float>(simplex: &mut [Vec<T>], f_vals: &mut [T], n: usize) {
    // Selection-sort by f_vals (n+1 is small)
    for i in 0..n {
        let mut min_idx = i;
        for j in (i + 1)..=n {
            if f_vals[j] < f_vals[min_idx] {
                min_idx = j;
            }
        }
        if min_idx != i {
            f_vals.swap(i, min_idx);
            simplex.swap(i, min_idx);
        }
    }
}

fn check_convergence<T: Float>(
    simplex: &[Vec<T>],
    f_vals: &[T],
    n: usize,
    iter: usize,
    f_evals: usize,
    options: &MinimizeOptions<T>,
) -> Option<MinimizeResult<T>> {
    let f_spread = f_vals[n] - f_vals[0];
    if f_spread.abs() < options.ftol {
        let diam = simplex_diameter(simplex, n);
        if diam < options.xtol {
            let x = Tensor::from_vec(simplex[0].clone(), vec![n])
                .expect("best vertex length matches n");
            return Some(MinimizeResult {
                x,
                f_val: f_vals[0],
                grad: None,
                iterations: iter,
                f_evals,
                g_evals: 0,
                converged: true,
            });
        }
    }
    None
}

fn compute_centroid<T: Float>(simplex: &[Vec<T>], n: usize) -> Vec<T> {
    let mut centroid = vec![T::zero(); n];
    for vertex in simplex.iter().take(n) {
        for j in 0..n {
            centroid[j] += vertex[j];
        }
    }
    let n_t = T::from_usize(n);
    for c in &mut centroid {
        *c /= n_t;
    }
    centroid
}

fn eval_point<T: Float, F: Fn(&Tensor<T>) -> T>(
    f: &F,
    point: &[T],
    n: usize,
    f_evals: &mut usize,
) -> T {
    let t = Tensor::from_vec(point.to_vec(), vec![n]).expect("point length matches n");
    *f_evals += 1;
    f(&t)
}

fn vertex_op<T: Float>(centroid: &[T], other: &[T], coeff: T, n: usize) -> Vec<T> {
    let mut result = vec![T::zero(); n];
    for j in 0..n {
        result[j] = centroid[j] + coeff * (other[j] - centroid[j]);
    }
    result
}

#[allow(clippy::too_many_arguments)]
fn iterate_simplex<T: Float, F: Fn(&Tensor<T>) -> T>(
    f: &F,
    simplex: &mut [Vec<T>],
    f_vals: &mut [T],
    centroid: &[T],
    n: usize,
    coeffs: &SimplexCoeffs<T>,
    f_evals: &mut usize,
) {
    // Reflection
    let x_r = vertex_op(centroid, &simplex[n], -coeffs.alpha, n);
    let f_r = eval_point(f, &x_r, n, f_evals);

    if f_r >= f_vals[0] && f_r < f_vals[n - 1] {
        simplex[n] = x_r;
        f_vals[n] = f_r;
        return;
    }

    if f_r < f_vals[0] {
        // Expansion
        let x_e = vertex_op(centroid, &x_r, coeffs.gamma, n);
        let f_e = eval_point(f, &x_e, n, f_evals);
        if f_e < f_r {
            simplex[n] = x_e;
            f_vals[n] = f_e;
        } else {
            simplex[n] = x_r;
            f_vals[n] = f_r;
        }
        return;
    }

    // Contraction
    let x_c = vertex_op(centroid, &simplex[n], coeffs.rho, n);
    let f_c = eval_point(f, &x_c, n, f_evals);
    if f_c < f_vals[n] {
        simplex[n] = x_c;
        f_vals[n] = f_c;
        return;
    }

    // Shrinkage
    let best = simplex[0].clone();
    for i in 1..=n {
        for j in 0..n {
            simplex[i][j] = best[j] + coeffs.sigma * (simplex[i][j] - best[j]);
        }
        f_vals[i] = eval_point(f, &simplex[i], n, f_evals);
    }
}

fn simplex_diameter<T: Float>(simplex: &[Vec<T>], n: usize) -> T {
    let mut max_dist = T::zero();
    for (i, vi) in simplex.iter().enumerate() {
        for vj in &simplex[i + 1..] {
            let dist_sq: T = (0..n)
                .map(|k| {
                    let d = vi[k] - vj[k];
                    d * d
                })
                .sum();
            let dist = dist_sq.sqrt();
            if dist > max_dist {
                max_dist = dist;
            }
        }
    }
    max_dist
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nelder_mead_quadratic() {
        let f = |x: &Tensor<f64>| {
            let s = x.as_slice();
            s[0] * s[0] + s[1] * s[1]
        };

        let x0 = Tensor::from_vec(vec![5.0, 5.0], vec![2]).unwrap();
        let opts = MinimizeOptions {
            max_iter: 1000,
            ftol: 1e-12,
            xtol: 1e-12,
            ..MinimizeOptions::default()
        };
        let result = nelder_mead(f, &x0, &opts).unwrap();
        assert!(
            result.converged,
            "did not converge after {} iters",
            result.iterations
        );
        let s = result.x.as_slice();
        assert!(s[0].abs() < 1e-4, "x = {}", s[0]);
        assert!(s[1].abs() < 1e-4, "y = {}", s[1]);
    }

    #[test]
    fn test_nelder_mead_rosenbrock() {
        let f = |x: &Tensor<f64>| {
            let s = x.as_slice();
            let a = 1.0 - s[0];
            let b = s[1] - s[0] * s[0];
            a * a + 100.0 * b * b
        };

        let x0 = Tensor::from_vec(vec![-1.0, 1.0], vec![2]).unwrap();
        let opts = MinimizeOptions {
            max_iter: 10000,
            ftol: 1e-12,
            xtol: 1e-12,
            ..MinimizeOptions::default()
        };
        let result = nelder_mead(f, &x0, &opts).unwrap();
        assert!(
            result.converged,
            "did not converge after {} iters, f = {}",
            result.iterations, result.f_val
        );
        let s = result.x.as_slice();
        assert!((s[0] - 1.0).abs() < 1e-3, "x = {}, expected 1.0", s[0]);
        assert!((s[1] - 1.0).abs() < 1e-3, "y = {}, expected 1.0", s[1]);
    }

    #[test]
    fn test_nelder_mead_high_dim() {
        let f = |x: &Tensor<f64>| x.as_slice().iter().map(|&v| v * v).sum::<f64>();

        let x0 = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
        let opts = MinimizeOptions {
            max_iter: 10000,
            ftol: 1e-12,
            xtol: 1e-12,
            ..MinimizeOptions::default()
        };
        let result = nelder_mead(f, &x0, &opts).unwrap();
        assert!(result.converged, "did not converge");
        for (i, &v) in result.x.as_slice().iter().enumerate() {
            assert!(v.abs() < 1e-3, "x[{i}] = {v}");
        }
    }

    #[test]
    fn test_nelder_mead_no_convergence_low_iter() {
        let f = |x: &Tensor<f64>| {
            let s = x.as_slice();
            let a = 1.0 - s[0];
            let b = s[1] - s[0] * s[0];
            a * a + 100.0 * b * b
        };

        let x0 = Tensor::from_vec(vec![-1.0, 1.0], vec![2]).unwrap();
        let opts = MinimizeOptions {
            max_iter: 5,
            ..MinimizeOptions::default()
        };
        let result = nelder_mead(f, &x0, &opts).unwrap();
        assert!(!result.converged);
    }
}
