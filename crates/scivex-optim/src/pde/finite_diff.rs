//! Finite difference PDE solvers for the heat equation, wave equation, and
//! Laplace equation.

use scivex_core::Float;

use crate::error::{OptimError, Result};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Boundary condition specification.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BoundaryCondition<T: Float> {
    /// Fixed value at boundary (Dirichlet).
    Dirichlet(T),
    /// Fixed derivative at boundary (Neumann).
    Neumann(T),
}

/// Result of a PDE solve.
#[derive(Debug, Clone)]
pub struct PdeResult<T: Float> {
    /// Solution values: for 1-D time-dependent problems the shape is
    /// `[n_time][n_space]`.  For 2-D steady-state problems the shape is
    /// `[ny][nx]`.
    pub u: Vec<Vec<T>>,
    /// Spatial grid points (x-axis).
    pub x: Vec<T>,
    /// Time points (for time-dependent problems) or y-axis grid (for 2-D
    /// steady-state).
    pub t_or_y: Vec<T>,
    /// Number of time / iteration steps taken.
    pub steps: usize,
    /// Whether the solution converged (meaningful for iterative methods such
    /// as Gauss-Seidel).
    pub converged: bool,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a uniform grid of `n` points spanning `[a, b]`.
fn linspace<T: Float>(a: T, b: T, n: usize) -> Vec<T> {
    if n < 2 {
        return vec![a];
    }
    let n_intervals = T::from_usize(n - 1);
    let dx = (b - a) / n_intervals;
    (0..n).map(|i| a + T::from_usize(i) * dx).collect()
}

/// Apply a boundary condition at one end of a 1-D solution row.
///
/// * `is_left`  — `true` for the left boundary (index 0), `false` for the
///   right boundary (last index).
/// * `row`      — mutable slice of the current solution row.
/// * `dx`       — spatial step size.
fn apply_bc_1d<T: Float>(bc: &BoundaryCondition<T>, is_left: bool, row: &mut [T], dx: T) {
    let n = row.len();
    match *bc {
        BoundaryCondition::Dirichlet(val) => {
            if is_left {
                row[0] = val;
            } else {
                row[n - 1] = val;
            }
        }
        BoundaryCondition::Neumann(deriv) => {
            // Ghost-node approach: u[-1] = u[1] - 2*dx*deriv  (left)
            //                      u[n]  = u[n-2] + 2*dx*deriv (right)
            if is_left {
                row[0] = row[1] - dx * deriv;
            } else {
                row[n - 1] = row[n - 2] + dx * deriv;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 1-D Heat equation  (FTCS explicit scheme)
// ---------------------------------------------------------------------------

/// Solve the 1-D heat equation
///
/// ```text
///   ∂u/∂t = α ∂²u/∂x²
/// ```
///
/// using the explicit forward-time, centred-space (FTCS) scheme.
///
/// # Parameters
///
/// * `x_range`  — spatial domain `[x0, x1]`.
/// * `n_x`      — number of spatial grid points (must be >= 3).
/// * `t_final`  — simulate until this time (must be > 0).
/// * `n_t`      — number of time steps (must be >= 1).
/// * `alpha`    — thermal diffusivity (must be > 0).
/// * `initial`  — initial condition `u(x, 0)`.
/// * `left_bc`  — boundary condition at `x = x0`.
/// * `right_bc` — boundary condition at `x = x1`.
///
/// # Errors
///
/// Returns [`OptimError::InvalidParameter`] when the grid is too small,
/// parameters are non-positive, or the CFL stability condition
/// `r = α dt / dx² <= 0.5` is violated.
#[allow(clippy::too_many_arguments)]
pub fn heat_equation_1d<T: Float>(
    x_range: (T, T),
    n_x: usize,
    t_final: T,
    n_t: usize,
    alpha: T,
    initial: &dyn Fn(T) -> T,
    left_bc: BoundaryCondition<T>,
    right_bc: BoundaryCondition<T>,
) -> Result<PdeResult<T>> {
    // --- Validate inputs ---------------------------------------------------
    if n_x < 3 {
        return Err(OptimError::InvalidParameter {
            name: "n_x",
            reason: "need at least 3 spatial points",
        });
    }
    if n_t < 1 {
        return Err(OptimError::InvalidParameter {
            name: "n_t",
            reason: "need at least 1 time step",
        });
    }
    let zero = T::zero();
    if t_final <= zero {
        return Err(OptimError::InvalidParameter {
            name: "t_final",
            reason: "must be positive",
        });
    }
    if alpha <= zero {
        return Err(OptimError::InvalidParameter {
            name: "alpha",
            reason: "must be positive",
        });
    }

    let x = linspace(x_range.0, x_range.1, n_x);
    let dx = x[1] - x[0];
    let dt = t_final / T::from_usize(n_t);
    let r = alpha * dt / (dx * dx);

    let half = T::from_f64(0.5);
    if r > half {
        return Err(OptimError::InvalidParameter {
            name: "n_t",
            reason: "stability condition violated: r = alpha*dt/dx^2 must be <= 0.5",
        });
    }

    // --- Initial condition --------------------------------------------------
    let mut u_prev: Vec<T> = x.iter().map(|&xi| initial(xi)).collect();
    apply_bc_1d(&left_bc, true, &mut u_prev, dx);
    apply_bc_1d(&right_bc, false, &mut u_prev, dx);

    let mut all_u: Vec<Vec<T>> = Vec::with_capacity(n_t + 1);
    all_u.push(u_prev.clone());

    let mut t_vals: Vec<T> = Vec::with_capacity(n_t + 1);
    t_vals.push(zero);

    // --- Time stepping (FTCS) ----------------------------------------------
    let two = T::from_f64(2.0);
    for step in 0..n_t {
        let mut u_next = u_prev.clone();
        for i in 1..(n_x - 1) {
            u_next[i] = u_prev[i] + r * (u_prev[i + 1] - two * u_prev[i] + u_prev[i - 1]);
        }
        apply_bc_1d(&left_bc, true, &mut u_next, dx);
        apply_bc_1d(&right_bc, false, &mut u_next, dx);

        t_vals.push(T::from_usize(step + 1) * dt);
        all_u.push(u_next.clone());
        u_prev = u_next;
    }

    Ok(PdeResult {
        u: all_u,
        x,
        t_or_y: t_vals,
        steps: n_t,
        converged: true,
    })
}

// ---------------------------------------------------------------------------
// 1-D Wave equation (explicit three-level scheme)
// ---------------------------------------------------------------------------

/// Solve the 1-D wave equation
///
/// ```text
///   ∂²u/∂t² = c² ∂²u/∂x²
/// ```
///
/// using the explicit three-level centred-difference scheme.
///
/// # Parameters
///
/// * `x_range`    — spatial domain `[x0, x1]`.
/// * `n_x`        — number of spatial grid points (must be >= 3).
/// * `t_final`    — simulate until this time (must be > 0).
/// * `n_t`        — number of time steps (must be >= 1).
/// * `c`          — wave speed (must be > 0).
/// * `initial_u`  — initial displacement `u(x, 0)`.
/// * `initial_ut` — initial velocity `∂u/∂t(x, 0)`.
/// * `left_bc`    — boundary condition at `x = x0`.
/// * `right_bc`   — boundary condition at `x = x1`.
///
/// # Errors
///
/// Returns [`OptimError::InvalidParameter`] when the grid is too small,
/// parameters are non-positive, or the CFL condition `c dt / dx <= 1` is
/// violated.
#[allow(clippy::too_many_arguments)]
pub fn wave_equation_1d<T: Float>(
    x_range: (T, T),
    n_x: usize,
    t_final: T,
    n_t: usize,
    c: T,
    initial_u: &dyn Fn(T) -> T,
    initial_ut: &dyn Fn(T) -> T,
    left_bc: BoundaryCondition<T>,
    right_bc: BoundaryCondition<T>,
) -> Result<PdeResult<T>> {
    // --- Validate inputs ---------------------------------------------------
    if n_x < 3 {
        return Err(OptimError::InvalidParameter {
            name: "n_x",
            reason: "need at least 3 spatial points",
        });
    }
    if n_t < 1 {
        return Err(OptimError::InvalidParameter {
            name: "n_t",
            reason: "need at least 1 time step",
        });
    }
    let zero = T::zero();
    if t_final <= zero {
        return Err(OptimError::InvalidParameter {
            name: "t_final",
            reason: "must be positive",
        });
    }
    if c <= zero {
        return Err(OptimError::InvalidParameter {
            name: "c",
            reason: "must be positive",
        });
    }

    let x = linspace(x_range.0, x_range.1, n_x);
    let dx = x[1] - x[0];
    let dt = t_final / T::from_usize(n_t);
    let r = c * dt / dx; // Courant number

    if r > T::one() {
        return Err(OptimError::InvalidParameter {
            name: "n_t",
            reason: "CFL condition violated: c*dt/dx must be <= 1",
        });
    }

    let r2 = r * r;
    let two = T::from_f64(2.0);

    // --- Level 0: u(x, 0) --------------------------------------------------
    let mut u_prev: Vec<T> = x.iter().map(|&xi| initial_u(xi)).collect();
    apply_bc_1d(&left_bc, true, &mut u_prev, dx);
    apply_bc_1d(&right_bc, false, &mut u_prev, dx);

    let mut all_u: Vec<Vec<T>> = Vec::with_capacity(n_t + 1);
    all_u.push(u_prev.clone());

    let mut t_vals: Vec<T> = Vec::with_capacity(n_t + 1);
    t_vals.push(zero);

    // --- Level 1: special first step using initial velocity -----------------
    // u^1_i = u^0_i + dt * ut(x_i) + 0.5*r²*(u^0_{i+1} - 2 u^0_i + u^0_{i-1})
    let half = T::from_f64(0.5);
    let mut u_curr: Vec<T> = vec![zero; n_x];
    for i in 1..(n_x - 1) {
        let laplacian = u_prev[i + 1] - two * u_prev[i] + u_prev[i - 1];
        u_curr[i] = u_prev[i] + dt * initial_ut(x[i]) + half * r2 * laplacian;
    }
    apply_bc_1d(&left_bc, true, &mut u_curr, dx);
    apply_bc_1d(&right_bc, false, &mut u_curr, dx);

    t_vals.push(dt);
    all_u.push(u_curr.clone());

    // --- Remaining steps (three-level scheme) -------------------------------
    for step in 1..n_t {
        let mut u_next = vec![zero; n_x];
        for i in 1..(n_x - 1) {
            let laplacian = u_curr[i + 1] - two * u_curr[i] + u_curr[i - 1];
            u_next[i] = two * u_curr[i] - u_prev[i] + r2 * laplacian;
        }
        apply_bc_1d(&left_bc, true, &mut u_next, dx);
        apply_bc_1d(&right_bc, false, &mut u_next, dx);

        t_vals.push(T::from_usize(step + 1) * dt);
        all_u.push(u_next.clone());
        u_prev = u_curr;
        u_curr = u_next;
    }

    Ok(PdeResult {
        u: all_u,
        x,
        t_or_y: t_vals,
        steps: n_t,
        converged: true,
    })
}

// ---------------------------------------------------------------------------
// 2-D Laplace equation (Gauss-Seidel iteration)
// ---------------------------------------------------------------------------

/// Solve the 2-D Laplace equation
///
/// ```text
///   ∂²u/∂x² + ∂²u/∂y² = 0
/// ```
///
/// on a rectangular domain using Gauss-Seidel relaxation.
///
/// The `boundary` closure receives `(x, y)` and must return `Some(value)` for
/// every point on the boundary of the domain (i.e.\ the first/last row/column
/// of the grid).  Interior points should return `None`.
///
/// # Parameters
///
/// * `x_range`  — `[x0, x1]`.
/// * `y_range`  — `[y0, y1]`.
/// * `n_x`      — number of grid points in x (must be >= 3).
/// * `n_y`      — number of grid points in y (must be >= 3).
/// * `boundary` — closure returning `Some(T)` for boundary points.
/// * `max_iter` — maximum number of Gauss-Seidel sweeps.
/// * `tol`      — convergence tolerance on the max absolute update.
///
/// # Errors
///
/// Returns [`OptimError::InvalidParameter`] for grids that are too small or
/// non-positive `max_iter` / `tol`.
pub fn laplace_2d<T: Float>(
    x_range: (T, T),
    y_range: (T, T),
    n_x: usize,
    n_y: usize,
    boundary: &dyn Fn(T, T) -> Option<T>,
    max_iter: usize,
    tol: T,
) -> Result<PdeResult<T>> {
    // --- Validate inputs ---------------------------------------------------
    if n_x < 3 {
        return Err(OptimError::InvalidParameter {
            name: "n_x",
            reason: "need at least 3 grid points in x",
        });
    }
    if n_y < 3 {
        return Err(OptimError::InvalidParameter {
            name: "n_y",
            reason: "need at least 3 grid points in y",
        });
    }
    if max_iter == 0 {
        return Err(OptimError::InvalidParameter {
            name: "max_iter",
            reason: "must be at least 1",
        });
    }
    if tol <= T::zero() {
        return Err(OptimError::InvalidParameter {
            name: "tol",
            reason: "must be positive",
        });
    }

    let x = linspace(x_range.0, x_range.1, n_x);
    let y = linspace(y_range.0, y_range.1, n_y);

    // --- Initialise grid with boundary values; interior = 0 ----------------
    let mut u: Vec<Vec<T>> = Vec::with_capacity(n_y);
    let mut is_boundary: Vec<Vec<bool>> = Vec::with_capacity(n_y);

    for yj in &y {
        let mut row = vec![T::zero(); n_x];
        let mut brow = vec![false; n_x];
        for (i, xi) in x.iter().enumerate() {
            if let Some(val) = boundary(*xi, *yj) {
                row[i] = val;
                brow[i] = true;
            }
        }
        u.push(row);
        is_boundary.push(brow);
    }

    // --- Gauss-Seidel iteration --------------------------------------------
    let quarter = T::from_f64(0.25);
    let mut converged = false;
    let mut steps: usize = 0;

    for _iter in 0..max_iter {
        let mut max_diff = T::zero();
        for j in 1..(n_y - 1) {
            for i in 1..(n_x - 1) {
                if is_boundary[j][i] {
                    continue;
                }
                let new_val = quarter * (u[j][i + 1] + u[j][i - 1] + u[j + 1][i] + u[j - 1][i]);
                let diff = (new_val - u[j][i]).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
                u[j][i] = new_val;
            }
        }
        steps += 1;
        if max_diff < tol {
            converged = true;
            break;
        }
    }

    Ok(PdeResult {
        u,
        x,
        t_or_y: y,
        steps,
        converged,
    })
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Heat equation with fixed BCs (0 on left, 1 on right) should converge
    /// toward a linear profile u(x) = x / L in steady state.
    #[test]
    fn test_heat_steady_state() {
        let n_x = 21;
        let n_t = 50_000;
        let result = heat_equation_1d(
            (0.0, 1.0),
            n_x,
            50.0, // long enough to approach steady state (L²/α = 10)
            n_t,
            0.1,       // alpha
            &|_x| 0.0, // initial = 0 everywhere
            BoundaryCondition::Dirichlet(0.0),
            BoundaryCondition::Dirichlet(1.0),
        )
        .unwrap();

        // Last row should be approximately linear: u(x) ≈ x
        let last = result.u.last().unwrap();
        for (i, &xi) in result.x.iter().enumerate() {
            let err = (last[i] - xi).abs();
            assert!(
                err < 0.05,
                "steady-state error too large at x={xi}: u={}, expected={xi}, err={err}",
                last[i],
            );
        }
    }

    /// A Gaussian pulse should diffuse (spread out) under the heat equation:
    /// its peak amplitude should decrease over time.
    #[test]
    fn test_heat_gaussian_diffusion() {
        let n_x = 101;
        let n_t = 5000;
        let result = heat_equation_1d(
            (0.0, 1.0),
            n_x,
            0.05,
            n_t,
            0.01,
            &|x: f64| (-(x - 0.5).powi(2) / 0.01).exp(),
            BoundaryCondition::Dirichlet(0.0),
            BoundaryCondition::Dirichlet(0.0),
        )
        .unwrap();

        // Peak of initial condition
        let initial_max = result.u[0]
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);

        // Peak at final time
        let final_max = result
            .u
            .last()
            .unwrap()
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);

        assert!(
            final_max < initial_max,
            "Gaussian peak should decrease: initial={initial_max}, final={final_max}",
        );
    }

    /// A sine standing-wave should oscillate: u(x,0) = sin(pi*x), ut=0.
    /// After half a period the solution should be approximately -sin(pi*x).
    #[test]
    fn test_wave_standing_wave() {
        let n_x = 101;
        let c = 1.0_f64;
        // Period = 2*L/c = 2.0 for L=1, c=1.  Half-period = 1.0.
        let t_final = 1.0;
        let n_t = 200;
        let result = wave_equation_1d(
            (0.0, 1.0),
            n_x,
            t_final,
            n_t,
            c,
            &|x: f64| (std::f64::consts::PI * x).sin(),
            &|_x: f64| 0.0,
            BoundaryCondition::Dirichlet(0.0),
            BoundaryCondition::Dirichlet(0.0),
        )
        .unwrap();

        // At t = half-period the displacement should be ≈ -sin(pi*x).
        let last = result.u.last().unwrap();
        let mid = n_x / 2; // x = 0.5
        // sin(pi*0.5) = 1.0, so we expect ≈ -1.0.
        assert!(
            last[mid] < -0.8,
            "standing wave mid-point should be near -1 at half-period, got {}",
            last[mid],
        );
    }

    /// Laplace equation with boundary u = x (linear) should yield the exact
    /// linear interior solution u(x, y) = x.
    #[test]
    fn test_laplace_linear_boundary() {
        let n_x = 21;
        let n_y = 21;
        let result = laplace_2d(
            (0.0, 1.0),
            (0.0, 1.0),
            n_x,
            n_y,
            &|x: f64, y: f64| {
                // Mark every edge point as boundary with value = x.
                if x < 1e-12 || (x - 1.0).abs() < 1e-12 || y < 1e-12 || (y - 1.0).abs() < 1e-12 {
                    Some(x)
                } else {
                    None
                }
            },
            10_000,
            1e-10,
        )
        .unwrap();

        assert!(result.converged, "Laplace solver should converge");

        // Interior should be ≈ x.
        for j in 1..(n_y - 1) {
            for i in 1..(n_x - 1) {
                let err = (result.u[j][i] - result.x[i]).abs();
                assert!(
                    err < 1e-6,
                    "Laplace linear solution error at ({}, {}): u={}, expected={}, err={err}",
                    result.x[i],
                    result.t_or_y[j],
                    result.u[j][i],
                    result.x[i],
                );
            }
        }
    }

    /// Verify the `converged` flag is actually set when the solver meets the
    /// tolerance.
    #[test]
    fn test_laplace_convergence() {
        let result = laplace_2d(
            (0.0, 1.0),
            (0.0, 1.0),
            11,
            11,
            &|x: f64, y: f64| {
                if x < 1e-12 || (x - 1.0).abs() < 1e-12 || y < 1e-12 || (y - 1.0).abs() < 1e-12 {
                    Some(x * y)
                } else {
                    None
                }
            },
            50_000,
            1e-8,
        )
        .unwrap();

        assert!(
            result.converged,
            "Laplace solver should converge within 50 000 iterations",
        );
        assert!(
            result.steps < 50_000,
            "should converge before hitting max_iter (took {} steps)",
            result.steps,
        );
    }
}
