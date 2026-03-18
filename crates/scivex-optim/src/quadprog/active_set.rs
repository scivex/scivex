//! Active set method for convex quadratic programming.
//!
//! Implements the primal active set algorithm for problems with inequality
//! constraints. The Hessian `H` must be symmetric positive semi-definite.

use scivex_core::Float;

use crate::error::{OptimError, Result};

use super::QpResult;

/// Solve a convex quadratic program using the active set method.
///
/// Minimizes `0.5 x^T H x + c^T x` subject to `A_ub x <= b_ub` where `H` is
/// symmetric positive semi-definite.
///
/// # Arguments
///
/// * `h` - `n x n` symmetric PSD Hessian matrix (row-major `Vec<Vec<T>>`)
/// * `c` - `n`-vector linear cost term
/// * `a_ub` - `m x n` inequality constraint matrix
/// * `b_ub` - `m`-vector inequality right-hand side
/// * `max_iter` - maximum number of active set iterations
///
/// # Errors
///
/// Returns [`OptimError::InvalidParameter`] if dimensions are inconsistent.
/// Returns [`OptimError::ConvergenceFailure`] if `max_iter` is exhausted.
#[allow(clippy::too_many_lines)]
pub fn quadprog<T: Float>(
    h: &[Vec<T>],
    c: &[T],
    a_ub: &[Vec<T>],
    b_ub: &[T],
    max_iter: usize,
) -> Result<QpResult<T>> {
    let n = c.len();
    let m = a_ub.len();

    validate_inputs(h, c, a_ub, b_ub, n, m)?;

    // Find an initial feasible point.
    let mut x = find_feasible_point(h, c, a_ub, b_ub, n, m)?;

    // Build the initial working set: indices of constraints active at x.
    let mut working_set: Vec<usize> = Vec::new();
    let tol = T::from_f64(1e-10);
    for i in 0..m {
        let ai_x = dot_row(&a_ub[i], &x);
        if (ai_x - b_ub[i]).abs() < tol {
            working_set.push(i);
        }
    }

    let mut iterations = 0usize;

    while iterations < max_iter {
        iterations += 1;

        // Gradient at current point: g = H*x + c
        let g = hessian_times_vec(h, &x, n, c);

        // Solve the equality-constrained QP subproblem for step p and
        // Lagrange multipliers lambda via the KKT system.
        let w_size = working_set.len();
        let kkt_dim = n + w_size;

        // Build KKT matrix:
        // [ H        A_w^T ]   [ p ]   [ -g ]
        // [ A_w      0     ] * [ l ] = [  0  ]
        let mut kkt = vec![T::zero(); kkt_dim * kkt_dim];
        let mut rhs = vec![T::zero(); kkt_dim];

        // Fill H block
        for i in 0..n {
            for j in 0..n {
                kkt[i * kkt_dim + j] = h[i][j];
            }
        }

        // Fill A_w^T block (top-right) and A_w block (bottom-left)
        for (wi, &ci) in working_set.iter().enumerate() {
            for j in 0..n {
                kkt[j * kkt_dim + (n + wi)] = a_ub[ci][j];
                kkt[(n + wi) * kkt_dim + j] = a_ub[ci][j];
            }
        }

        // RHS: [ -g; 0 ]
        for i in 0..n {
            rhs[i] = -g[i];
        }

        // Solve KKT system
        let solution = solve_linear_system(&kkt, &rhs, kkt_dim);

        let (p, lambda) = match solution {
            Some(sol) => {
                let p_vec: Vec<T> = sol[..n].to_vec();
                let l_vec: Vec<T> = sol[n..].to_vec();
                (p_vec, l_vec)
            }
            None => {
                // Singular KKT — treat as zero step
                (vec![T::zero(); n], vec![T::zero(); w_size])
            }
        };

        let p_norm: T = p.iter().map(|&pi| pi * pi).sum();

        if p_norm < tol * tol {
            // p is essentially zero — check Lagrange multipliers.
            if working_set.is_empty() {
                // Unconstrained optimum, done.
                let fun = objective(h, c, &x, n);
                return Ok(QpResult {
                    x,
                    fun,
                    iterations,
                    converged: true,
                });
            }

            // Find most negative multiplier.
            let mut min_lambda = T::zero();
            let mut min_idx: Option<usize> = None;
            for (wi, &li) in lambda.iter().enumerate() {
                if li < min_lambda {
                    min_lambda = li;
                    min_idx = Some(wi);
                }
            }

            if let Some(idx) = min_idx {
                // Drop the constraint with the most negative multiplier.
                working_set.remove(idx);
            } else {
                // All multipliers >= 0, we are optimal.
                let fun = objective(h, c, &x, n);
                return Ok(QpResult {
                    x,
                    fun,
                    iterations,
                    converged: true,
                });
            }
        } else {
            // p != 0: compute maximum step size alpha staying feasible.
            let mut alpha = T::one();

            let mut blocking_constraint: Option<usize> = None;

            for i in 0..m {
                // Skip constraints already in the working set.
                if working_set.contains(&i) {
                    continue;
                }

                let a_p: T = dot_row(&a_ub[i], &p);
                if a_p <= tol {
                    // Constraint is not tightened by this step.
                    continue;
                }

                let a_x: T = dot_row(&a_ub[i], &x);
                let slack = b_ub[i] - a_x;
                let step = slack / a_p;

                if step < alpha {
                    alpha = step;
                    blocking_constraint = Some(i);
                }
            }

            // Clamp alpha to avoid negative due to numerical noise.
            if alpha < T::zero() {
                alpha = T::zero();
            }

            // Update x += alpha * p
            for j in 0..n {
                x[j] += alpha * p[j];
            }

            // If a constraint blocked, add it to the working set.
            if let Some(bc) = blocking_constraint.filter(|_| alpha < T::one()) {
                working_set.push(bc);
            }
        }
    }

    // Did not converge within max_iter.
    Err(OptimError::ConvergenceFailure {
        iterations: max_iter,
    })
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Validate that all dimensions are consistent.
fn validate_inputs<T: Float>(
    h: &[Vec<T>],
    _c: &[T],
    a_ub: &[Vec<T>],
    b_ub: &[T],
    n: usize,
    m: usize,
) -> Result<()> {
    if n == 0 {
        return Err(OptimError::InvalidParameter {
            name: "c",
            reason: "must be non-empty",
        });
    }

    if h.len() != n {
        return Err(OptimError::InvalidParameter {
            name: "h",
            reason: "must have n rows matching length of c",
        });
    }
    for row in h {
        if row.len() != n {
            return Err(OptimError::InvalidParameter {
                name: "h",
                reason: "all rows must have length n",
            });
        }
    }

    if b_ub.len() != m {
        return Err(OptimError::InvalidParameter {
            name: "b_ub",
            reason: "length must match number of rows in a_ub",
        });
    }

    for row in a_ub {
        if row.len() != n {
            return Err(OptimError::InvalidParameter {
                name: "a_ub",
                reason: "all rows must have length equal to c",
            });
        }
    }

    Ok(())
}

/// Compute `H * x + c`.
fn hessian_times_vec<T: Float>(h: &[Vec<T>], x: &[T], n: usize, c: &[T]) -> Vec<T> {
    let mut g = vec![T::zero(); n];
    for i in 0..n {
        let mut sum = T::zero();
        for j in 0..n {
            sum += h[i][j] * x[j];
        }
        g[i] = sum + c[i];
    }
    g
}

/// Dot product of a row vector with x.
fn dot_row<T: Float>(row: &[T], x: &[T]) -> T {
    let mut s = T::zero();
    for i in 0..row.len() {
        s += row[i] * x[i];
    }
    s
}

/// Compute objective 0.5 * x^T H x + c^T x.
fn objective<T: Float>(h: &[Vec<T>], c: &[T], x: &[T], n: usize) -> T {
    let half = T::from_f64(0.5);
    let mut quad = T::zero();
    for i in 0..n {
        for j in 0..n {
            quad += x[i] * h[i][j] * x[j];
        }
    }
    let mut lin = T::zero();
    for i in 0..n {
        lin += c[i] * x[i];
    }
    half * quad + lin
}

/// Find an initial feasible point satisfying all `A x <= b`.
///
/// Tries the origin first. If the origin is infeasible, projects towards
/// feasibility by iteratively adjusting the point along violated constraint
/// normals.
fn find_feasible_point<T: Float>(
    _h: &[Vec<T>],
    _c: &[T],
    a_ub: &[Vec<T>],
    b_ub: &[T],
    n: usize,
    m: usize,
) -> Result<Vec<T>> {
    if m == 0 {
        return Ok(vec![T::zero(); n]);
    }

    let mut x = vec![T::zero(); n];

    // Check if origin is feasible.
    let origin_feasible = (0..m).all(|i| dot_row(&a_ub[i], &x) <= b_ub[i]);
    if origin_feasible {
        return Ok(x);
    }

    // Iterative projection to find a feasible point.
    // For each violated constraint a_i^T x > b_i, push x to satisfy it.
    let tol = T::from_f64(1e-10);
    let max_proj_iter = 1000usize;

    for _ in 0..max_proj_iter {
        let mut feasible = true;
        for i in 0..m {
            let ai_x = dot_row(&a_ub[i], &x);
            let violation = ai_x - b_ub[i];
            if violation > tol {
                feasible = false;
                // Project x onto the half-space a_i^T x <= b_i:
                // x_new = x - ((a_i^T x - b_i) / ||a_i||^2) * a_i
                let norm_sq: T = a_ub[i].iter().map(|&v| v * v).sum();
                if norm_sq > tol {
                    let factor = violation / norm_sq;
                    for j in 0..n {
                        x[j] -= factor * a_ub[i][j];
                    }
                }
            }
        }
        if feasible {
            return Ok(x);
        }
    }

    Err(OptimError::InvalidParameter {
        name: "constraints",
        reason: "could not find a feasible starting point",
    })
}

/// Solve a dense linear system `A x = b` via Gaussian elimination with
/// partial pivoting. Returns `None` if the matrix is singular.
fn solve_linear_system<T: Float>(a: &[T], b: &[T], n: usize) -> Option<Vec<T>> {
    let mut aug = vec![T::zero(); n * (n + 1)];
    for i in 0..n {
        for j in 0..n {
            aug[i * (n + 1) + j] = a[i * n + j];
        }
        aug[i * (n + 1) + n] = b[i];
    }

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_val = aug[col * (n + 1) + col].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            let val = aug[row * (n + 1) + col].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        if max_val < T::from_f64(1e-14) {
            return None; // Singular
        }

        // Swap rows
        if max_row != col {
            for j in 0..=n {
                aug.swap(col * (n + 1) + j, max_row * (n + 1) + j);
            }
        }

        // Eliminate
        let pivot = aug[col * (n + 1) + col];
        for row in (col + 1)..n {
            let factor = aug[row * (n + 1) + col] / pivot;
            for j in col..=n {
                let val = aug[col * (n + 1) + j];
                aug[row * (n + 1) + j] -= factor * val;
            }
        }
    }

    // Back substitution
    let mut x = vec![T::zero(); n];
    for i in (0..n).rev() {
        let mut sum = aug[i * (n + 1) + n];
        for j in (i + 1)..n {
            sum -= aug[i * (n + 1) + j] * x[j];
        }
        x[i] = sum / aug[i * (n + 1) + i];
    }

    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn approx_vec(a: &[f64], b: &[f64], tol: f64) -> bool {
        a.len() == b.len()
            && a.iter()
                .zip(b.iter())
                .all(|(&ai, &bi)| approx_eq(ai, bi, tol))
    }

    #[test]
    fn test_qp_unconstrained() {
        // minimize 0.5*(x1^2 + x2^2) + (-2)*x1 + (-1)*x2
        // H = I, c = [-2, -1]
        // Optimal: x = [2, 1], fun = -2.5
        let h = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let c = vec![-2.0, -1.0];
        let a_ub: Vec<Vec<f64>> = vec![];
        let b_ub: Vec<f64> = vec![];

        let res = quadprog(&h, &c, &a_ub, &b_ub, 100).expect("should converge");
        assert!(res.converged);
        assert!(approx_vec(&res.x, &[2.0, 1.0], 1e-6));
        assert!(approx_eq(res.fun, -2.5, 1e-6));
    }

    #[test]
    fn test_qp_with_constraints() {
        // minimize 0.5*(x1^2 + x2^2) - 2*x1 - x2
        // subject to x1 + x2 <= 1
        // Unconstrained optimum is [2, 1] which violates x1+x2<=1.
        // Constrained optimum: x1 + x2 = 1, with x1 = 1.5 - 0.5 = ?
        // KKT: x1 - 2 + lambda = 0, x2 - 1 + lambda = 0, x1+x2=1
        // => x1 = 2 - lambda, x2 = 1 - lambda, (2-lambda)+(1-lambda)=1 => lambda=1
        // => x1 = 1, x2 = 0, fun = 0.5 - 2 = -1.5
        let h = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let c = vec![-2.0, -1.0];
        let a_ub = vec![vec![1.0, 1.0]];
        let b_ub = vec![1.0];

        let res = quadprog(&h, &c, &a_ub, &b_ub, 100).expect("should converge");
        assert!(res.converged);
        assert!(approx_vec(&res.x, &[1.0, 0.0], 1e-6));
        assert!(approx_eq(res.fun, -1.5, 1e-6));
    }

    #[test]
    fn test_qp_box_constraints() {
        // minimize 0.5*(x1^2 + x2^2) - 5*x1 - 5*x2
        // subject to x1 <= 2, x2 <= 3, -x1 <= 0, -x2 <= 0
        // Unconstrained: [5, 5]. With bounds: [2, 3].
        let h = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let c = vec![-5.0, -5.0];
        let a_ub = vec![
            vec![1.0, 0.0],  // x1 <= 2
            vec![0.0, 1.0],  // x2 <= 3
            vec![-1.0, 0.0], // -x1 <= 0 (i.e. x1 >= 0)
            vec![0.0, -1.0], // -x2 <= 0 (i.e. x2 >= 0)
        ];
        let b_ub = vec![2.0, 3.0, 0.0, 0.0];

        let res = quadprog(&h, &c, &a_ub, &b_ub, 200).expect("should converge");
        assert!(res.converged);
        assert!(approx_vec(&res.x, &[2.0, 3.0], 1e-6));
        // fun = 0.5*(4+9) - 10 - 15 = 6.5 - 25 = -18.5
        assert!(approx_eq(res.fun, -18.5, 1e-6));
    }

    #[test]
    fn test_qp_infeasible_start() {
        // minimize 0.5*(x1^2 + x2^2)
        // subject to x1 + x2 >= 2 (i.e. -x1 - x2 <= -2)
        //            x1 >= 0, x2 >= 0
        // Origin is NOT feasible. Optimal: x1=1, x2=1, fun=1.
        let h = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let c = vec![0.0, 0.0];
        let a_ub = vec![
            vec![-1.0, -1.0], // -x1-x2 <= -2
            vec![-1.0, 0.0],  // -x1 <= 0
            vec![0.0, -1.0],  // -x2 <= 0
        ];
        let b_ub = vec![-2.0, 0.0, 0.0];

        let res = quadprog(&h, &c, &a_ub, &b_ub, 200).expect("should converge");
        assert!(res.converged);
        assert!(approx_vec(&res.x, &[1.0, 1.0], 1e-6));
        assert!(approx_eq(res.fun, 1.0, 1e-6));
    }

    #[test]
    fn test_qp_convergence() {
        // Simple problem that must converge.
        // minimize 0.5 * x^2 - x, optimal at x=1, fun=-0.5
        let h = vec![vec![1.0]];
        let c = vec![-1.0];
        let a_ub: Vec<Vec<f64>> = vec![];
        let b_ub: Vec<f64> = vec![];

        let res = quadprog(&h, &c, &a_ub, &b_ub, 50).expect("should converge");
        assert!(res.converged);
        assert!(approx_eq(res.x[0], 1.0, 1e-8));
        assert!(approx_eq(res.fun, -0.5, 1e-8));
        assert!(res.iterations <= 50);
    }
}
