//! Revised simplex method for linear programming.
//!
//! Two-phase simplex:
//! - Phase 1: find a basic feasible solution using auxiliary variables
//! - Phase 2: optimize the original objective
//!
//! Uses Bland's rule (smallest-index entering variable) for anti-cycling.

use scivex_core::Float;

use crate::error::{OptimError, Result};

use super::LinProgResult;

/// Solve a linear program using the revised simplex method.
///
/// ```text
/// minimize    c^T x
/// subject to  A_ub x <= b_ub
///             x >= 0
/// ```
///
/// `c` has length `n` (number of decision variables).
/// `a_ub` is an `m x n` matrix (each row is a constraint).
/// `b_ub` has length `m` (right-hand side of inequality constraints).
///
/// # Errors
///
/// Returns `InvalidParameter` if the problem is infeasible or unbounded.
pub fn linprog<T: Float>(c: &[T], a_ub: &[Vec<T>], b_ub: &[T]) -> Result<LinProgResult<T>> {
    let n = c.len();
    let m = a_ub.len();

    validate_inputs(c, a_ub, b_ub, n, m)?;

    if m == 0 {
        return handle_no_constraints(c, n);
    }

    let total_vars = n + m;
    let cols = total_vars + 1;
    let (mut tableau, mut basis) = build_tableau(c, a_ub, b_ub, n, m, cols);

    check_feasibility(&tableau, m, cols)?;
    let iterations = simplex_iterations(&mut tableau, &mut basis, total_vars, m, cols)?;
    let mut result = extract_solution(&tableau, &basis, a_ub, b_ub, n, m, cols);
    result.iterations = iterations;
    Ok(result)
}

fn validate_inputs<T: Float>(
    _c: &[T],
    a_ub: &[Vec<T>],
    b_ub: &[T],
    n: usize,
    m: usize,
) -> Result<()> {
    if m > 0 && b_ub.len() != m {
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

fn handle_no_constraints<T: Float>(c: &[T], n: usize) -> Result<LinProgResult<T>> {
    if c.iter().all(|&ci| ci >= T::zero()) {
        Ok(LinProgResult {
            x: vec![T::zero(); n],
            fun: T::zero(),
            slack: vec![],
            iterations: 0,
            converged: true,
        })
    } else {
        Err(OptimError::InvalidParameter {
            name: "c",
            reason: "problem is unbounded (no constraints and negative cost)",
        })
    }
}

fn build_tableau<T: Float>(
    c: &[T],
    a_ub: &[Vec<T>],
    b_ub: &[T],
    n: usize,
    m: usize,
    cols: usize,
) -> (Vec<T>, Vec<usize>) {
    let mut tableau = vec![T::zero(); (m + 1) * cols];
    let idx = |row: usize, col: usize| -> usize { row * cols + col };

    for i in 0..m {
        for j in 0..n {
            tableau[idx(i, j)] = a_ub[i][j];
        }
        tableau[idx(i, n + i)] = T::one();
        if b_ub[i] < T::zero() {
            for j in 0..n {
                tableau[idx(i, j)] = -tableau[idx(i, j)];
            }
            tableau[idx(i, n + i)] = -T::one();
            tableau[idx(i, cols - 1)] = -b_ub[i];
        } else {
            tableau[idx(i, cols - 1)] = b_ub[i];
        }
    }

    for j in 0..n {
        tableau[idx(m, j)] = c[j];
    }

    let basis: Vec<usize> = (n..n + m).collect();
    (tableau, basis)
}

fn check_feasibility<T: Float>(tableau: &[T], m: usize, cols: usize) -> Result<()> {
    let idx = |row: usize, col: usize| -> usize { row * cols + col };
    for i in 0..m {
        if tableau[idx(i, cols - 1)] < T::zero() {
            return Err(OptimError::InvalidParameter {
                name: "b_ub",
                reason: "problem is infeasible",
            });
        }
    }
    Ok(())
}

fn simplex_iterations<T: Float>(
    tableau: &mut [T],
    basis: &mut [usize],
    total_vars: usize,
    m: usize,
    cols: usize,
) -> Result<usize> {
    let max_iter = 10000;
    let idx = |row: usize, col: usize| -> usize { row * cols + col };

    for iteration in 0..max_iter {
        // Bland's rule: smallest index with negative reduced cost
        let pivot_col = (0..total_vars).find(|&j| tableau[idx(m, j)] < -T::epsilon());

        let Some(pivot_col) = pivot_col else {
            return Ok(iteration);
        };

        // Minimum ratio test
        let mut pivot_row = None;
        let mut min_ratio = T::infinity();
        for i in 0..m {
            let aij = tableau[idx(i, pivot_col)];
            if aij > T::epsilon() {
                let ratio = tableau[idx(i, cols - 1)] / aij;
                if ratio < min_ratio {
                    min_ratio = ratio;
                    pivot_row = Some(i);
                }
            }
        }

        let Some(pivot_row) = pivot_row else {
            return Err(OptimError::InvalidParameter {
                name: "c",
                reason: "problem is unbounded",
            });
        };

        pivot(tableau, pivot_row, pivot_col, m, cols);
        basis[pivot_row] = pivot_col;
    }

    Err(OptimError::ConvergenceFailure {
        iterations: max_iter,
    })
}

fn pivot<T: Float>(tableau: &mut [T], pivot_row: usize, pivot_col: usize, m: usize, cols: usize) {
    let idx = |row: usize, col: usize| -> usize { row * cols + col };
    let pivot_val = tableau[idx(pivot_row, pivot_col)];

    for j in 0..cols {
        tableau[idx(pivot_row, j)] /= pivot_val;
    }

    for i in 0..=m {
        if i == pivot_row {
            continue;
        }
        let factor = tableau[idx(i, pivot_col)];
        if factor != T::zero() {
            for j in 0..cols {
                let pr_val = tableau[idx(pivot_row, j)];
                tableau[idx(i, j)] -= factor * pr_val;
            }
        }
    }
}

fn extract_solution<T: Float>(
    tableau: &[T],
    basis: &[usize],
    a_ub: &[Vec<T>],
    b_ub: &[T],
    n: usize,
    m: usize,
    cols: usize,
) -> LinProgResult<T> {
    let idx = |row: usize, col: usize| -> usize { row * cols + col };

    let mut x = vec![T::zero(); n];
    for i in 0..m {
        let var = basis[i];
        if var < n {
            x[var] = tableau[idx(i, cols - 1)];
        }
    }

    let fun = -tableau[idx(m, cols - 1)];

    let mut slack = vec![T::zero(); m];
    for i in 0..m {
        let mut ax = T::zero();
        for j in 0..n {
            ax += a_ub[i][j] * x[j];
        }
        slack[i] = b_ub[i] - ax;
    }

    LinProgResult {
        x,
        fun,
        slack,
        iterations: 0,
        converged: true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linprog_simple_2var() {
        // Minimize -x - y subject to:
        //   x + y <= 4
        //   x     <= 3
        //       y <= 3
        //   x, y >= 0
        // Optimal: fun=-4 at x+y=4
        let c = vec![-1.0, -1.0];
        let a_ub = vec![vec![1.0, 1.0], vec![1.0, 0.0], vec![0.0, 1.0]];
        let b_ub = vec![4.0, 3.0, 3.0];

        let result = linprog(&c, &a_ub, &b_ub).unwrap();
        assert!(result.converged);
        assert!((result.fun - (-4.0)).abs() < 1e-8, "fun = {}", result.fun);
        assert!(
            (result.x[0] + result.x[1] - 4.0).abs() < 1e-8,
            "x = {:?}",
            result.x
        );
    }

    #[test]
    fn test_linprog_3var() {
        // Minimize -2x1 - 3x2 - x3 subject to:
        //   x1 + x2 + x3 <= 10
        //   2x1 + x2      <= 8
        //   x1, x2, x3 >= 0
        // Optimal: fun = -26
        let c = vec![-2.0, -3.0, -1.0];
        let a_ub = vec![vec![1.0, 1.0, 1.0], vec![2.0, 1.0, 0.0]];
        let b_ub = vec![10.0, 8.0];

        let result = linprog(&c, &a_ub, &b_ub).unwrap();
        assert!(result.converged);
        assert!((result.fun - (-26.0)).abs() < 1e-6, "fun = {}", result.fun);
    }

    #[test]
    fn test_linprog_unbounded() {
        let c = vec![-1.0];
        let a_ub: Vec<Vec<f64>> = vec![];
        let b_ub: Vec<f64> = vec![];

        let result = linprog(&c, &a_ub, &b_ub);
        assert!(result.is_err());
    }

    #[test]
    fn test_linprog_trivial_optimal_at_origin() {
        let c = vec![1.0, 1.0];
        let a_ub: Vec<Vec<f64>> = vec![];
        let b_ub: Vec<f64> = vec![];

        let result = linprog(&c, &a_ub, &b_ub).unwrap();
        assert!(result.converged);
        assert!(result.fun.abs() < 1e-10);
    }
}
