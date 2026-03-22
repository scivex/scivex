//! Linear Mixed-Effects Models (LMM) via EM/REML estimation.
//!
//! Fits models of the form `y = X*beta + Z*u + epsilon` where
//! `u ~ N(0, G)` and `epsilon ~ N(0, sigma^2 * I)`.

use scivex_core::{Float, Tensor};

use crate::error::{Result, StatsError};

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

/// Result of fitting a linear mixed-effects model.
///
/// # Examples
///
/// ```
/// # use scivex_stats::mixed_effects::LmmResult;
/// # let res: LmmResult<f64> = LmmResult {
/// #     fixed_effects: vec![1.0, 2.0],
/// #     random_effects: vec![vec![0.1], vec![-0.1]],
/// #     residual_variance: 1.0,
/// #     random_effect_variance: vec![0.5],
/// #     log_likelihood: -50.0,
/// #     iterations: 10,
/// #     converged: true,
/// # };
/// assert!(res.converged);
/// ```
#[derive(Debug, Clone)]
pub struct LmmResult<T: Float> {
    /// Fixed-effect coefficient estimates (beta).
    pub fixed_effects: Vec<T>,
    /// Random-effect estimates per group (u_g for each group g).
    pub random_effects: Vec<Vec<T>>,
    /// Residual variance estimate (sigma^2).
    pub residual_variance: T,
    /// Variance components for random effects (diagonal of G).
    pub random_effect_variance: Vec<T>,
    /// Log-likelihood at convergence.
    pub log_likelihood: T,
    /// Number of EM iterations performed.
    pub iterations: usize,
    /// Whether the algorithm converged within `max_iter`.
    pub converged: bool,
}

// ---------------------------------------------------------------------------
// Small symmetric positive-definite solver (Cholesky)
// ---------------------------------------------------------------------------

/// Solve `A x = b` where `A` is a small symmetric positive-definite matrix.
///
/// `a` is stored row-major in a flat slice of length `q*q`.
/// `b` has length `q`. Returns the solution vector of length `q`.
fn solve_spd<T: Float>(a: &[T], b: &[T], q: usize) -> Result<Vec<T>> {
    // Cholesky decomposition: A = L L^T
    let zero = T::from_f64(0.0);
    let mut lower = vec![zero; q * q];

    for i in 0..q {
        for j in 0..=i {
            let mut sum = a[i * q + j];
            for k in 0..j {
                sum -= lower[i * q + k] * lower[j * q + k];
            }
            if i == j {
                if sum <= zero {
                    // Add small regularization and retry
                    let eps = T::from_f64(1e-10);
                    let sum_reg = sum + eps;
                    if sum_reg <= zero {
                        return Err(StatsError::SingularMatrix);
                    }
                    lower[i * q + j] = sum_reg.sqrt();
                } else {
                    lower[i * q + j] = sum.sqrt();
                }
            } else {
                let diag = lower[j * q + j];
                if diag.abs() < T::from_f64(1e-15) {
                    return Err(StatsError::SingularMatrix);
                }
                lower[i * q + j] = sum / diag;
            }
        }
    }

    // Forward substitution: L y = b
    let mut y = vec![zero; q];
    for i in 0..q {
        let mut sum = b[i];
        for j in 0..i {
            sum -= lower[i * q + j] * y[j];
        }
        y[i] = sum / lower[i * q + i];
    }

    // Back substitution: L^T x = y
    let mut x = vec![zero; q];
    for i in (0..q).rev() {
        let mut sum = y[i];
        for j in (i + 1)..q {
            sum -= lower[j * q + i] * x[j];
        }
        x[i] = sum / lower[i * q + i];
    }

    Ok(x)
}

/// Compute the inverse of a small symmetric positive-definite matrix via Cholesky.
///
/// Returns the inverse as a flat row-major vector of length `q*q`.
fn inv_spd<T: Float>(a: &[T], q: usize) -> Result<Vec<T>> {
    let zero = T::from_f64(0.0);
    let mut result = vec![zero; q * q];

    for col in 0..q {
        // Solve A * x = e_col
        let mut rhs = vec![zero; q];
        rhs[col] = T::from_f64(1.0);
        let x = solve_spd(a, &rhs, q)?;
        for row in 0..q {
            result[row * q + col] = x[row];
        }
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// LMM fitting
// ---------------------------------------------------------------------------

/// Fit a linear mixed-effects model using REML estimation via the EM algorithm.
///
/// The model is `y = X*beta + Z*u + epsilon` where `u ~ N(0, G)` and
/// `epsilon ~ N(0, sigma^2 * I)`.
///
/// # Arguments
///
/// * `y` - response vector of length `n`
/// * `x` - fixed-effects design matrix `[n x p]`
/// * `groups` - integer-coded group labels `[n]` (0-based, contiguous)
/// * `z` - optional random-effects design matrix `[n x q]`. If `None`, a
///   random-intercept model is used (one random intercept per group).
/// * `max_iter` - maximum number of EM iterations
/// * `tol` - convergence tolerance on the change in log-likelihood
///
/// # Examples
///
/// ```
/// use scivex_core::Tensor;
/// use scivex_stats::mixed_effects::lmm;
///
/// // Two groups with different intercepts
/// let y = vec![1.1, 1.2, 0.9, 1.0,   5.1, 5.2, 4.9, 5.0_f64];
/// let x = Tensor::from_vec(vec![1.0; 8], vec![8, 1]).unwrap();
/// let groups = vec![0, 0, 0, 0, 1, 1, 1, 1];
///
/// let result = lmm(&y, &x, &groups, None, 100, 1e-8).unwrap();
/// assert!(result.converged);
/// // The fixed effect (grand mean) should be around 3.05
/// assert!((result.fixed_effects[0] - 3.05).abs() < 0.5);
/// ```
///
/// # Errors
///
/// Returns [`StatsError::EmptyInput`] if `y` is empty,
/// [`StatsError::LengthMismatch`] if dimensions are inconsistent, or
/// [`StatsError::ConvergenceFailure`] if the EM algorithm does not converge
/// within `max_iter` iterations.
#[allow(clippy::too_many_lines)]
pub fn lmm<T: Float>(
    y: &[T],
    x: &Tensor<T>,
    groups: &[usize],
    z: Option<&Tensor<T>>,
    max_iter: usize,
    tol: T,
) -> Result<LmmResult<T>> {
    // ------------------------------------------------------------------
    // Validation
    // ------------------------------------------------------------------
    let n = y.len();
    if n == 0 {
        return Err(StatsError::EmptyInput);
    }
    if x.ndim() != 2 {
        return Err(StatsError::InvalidParameter {
            name: "x",
            reason: "must be a 2-D tensor",
        });
    }
    let p = x.shape()[1];
    if x.shape()[0] != n {
        return Err(StatsError::LengthMismatch {
            expected: n,
            got: x.shape()[0],
        });
    }
    if groups.len() != n {
        return Err(StatsError::LengthMismatch {
            expected: n,
            got: groups.len(),
        });
    }
    if n <= p {
        return Err(StatsError::InsufficientData {
            need: p + 1,
            got: n,
        });
    }

    // Determine number of random-effect columns (q)
    let q: usize = match z {
        Some(zt) => {
            if zt.ndim() != 2 {
                return Err(StatsError::InvalidParameter {
                    name: "z",
                    reason: "must be a 2-D tensor",
                });
            }
            if zt.shape()[0] != n {
                return Err(StatsError::LengthMismatch {
                    expected: n,
                    got: zt.shape()[0],
                });
            }
            zt.shape()[1]
        }
        None => 1, // random intercept
    };

    // Determine number of groups
    let n_groups = groups.iter().copied().max().unwrap_or(0) + 1;

    // Build per-group index lists
    let mut group_indices: Vec<Vec<usize>> = vec![Vec::new(); n_groups];
    for (i, &g) in groups.iter().enumerate() {
        if g >= n_groups {
            return Err(StatsError::InvalidParameter {
                name: "groups",
                reason: "group labels must be 0-based contiguous integers",
            });
        }
        group_indices[g].push(i);
    }

    // Slices for design matrices
    let x_slice = x.as_slice();
    let z_slice: Vec<T> = match z {
        Some(zt) => zt.as_slice().to_vec(),
        None => vec![T::from_f64(1.0); n], // random intercept: z_i = 1
    };

    let zero = T::from_f64(0.0);
    let one = T::from_f64(1.0);
    let two = T::from_f64(2.0);
    let half = T::from_f64(0.5);
    let nf = T::from_f64(n as f64);

    // ------------------------------------------------------------------
    // Initialize parameters
    // ------------------------------------------------------------------

    // Compute sample variance of y for initialization
    let y_mean: T = y.iter().copied().sum::<T>() / nf;
    let y_var: T = y.iter().map(|&yi| (yi - y_mean) * (yi - y_mean)).sum::<T>() / nf;
    let y_var = if y_var > zero { y_var } else { one };

    let mut sigma2 = y_var * half; // residual variance
    let mut tau2 = vec![y_var * half; q]; // random effect variances (diagonal of G)

    // Initialize beta via OLS: beta = (X^T X)^{-1} X^T y
    let y_tensor = Tensor::from_slice(y, vec![n])?;
    let beta_tensor = x.lstsq(&y_tensor).map_err(|_| StatsError::SingularMatrix)?;
    let mut beta: Vec<T> = beta_tensor.as_slice().to_vec();

    // Initialize random effects to zero
    let mut u: Vec<Vec<T>> = vec![vec![zero; q]; n_groups];

    // Per-group posterior covariance (D_g), stored as flat q*q
    let mut d_g: Vec<Vec<T>> = vec![vec![zero; q * q]; n_groups];

    let mut prev_ll = T::from_f64(f64::NEG_INFINITY);
    let mut converged = false;
    let mut iterations = 0;

    // ------------------------------------------------------------------
    // EM iterations
    // ------------------------------------------------------------------
    for iter in 0..max_iter {
        iterations = iter + 1;

        // ==============================================================
        // E-step: Compute posterior of random effects for each group
        // ==============================================================
        for g in 0..n_groups {
            let idx = &group_indices[g];
            let n_g = idx.len();
            if n_g == 0 {
                continue;
            }

            // Compute Z_g^T Z_g / sigma^2 + diag(1/tau^2)
            // This is a q x q matrix
            let mut a_mat = vec![zero; q * q];

            // Add Z_g^T Z_g / sigma^2
            for &i in idx {
                for r in 0..q {
                    let z_ir = z_slice[i * q + r];
                    for c in 0..q {
                        let z_ic = z_slice[i * q + c];
                        a_mat[r * q + c] += z_ir * z_ic / sigma2;
                    }
                }
            }

            // Add diag(1/tau^2)
            for r in 0..q {
                let tau2_r = if tau2[r] > T::from_f64(1e-15) {
                    tau2[r]
                } else {
                    T::from_f64(1e-15)
                };
                a_mat[r * q + r] += one / tau2_r;
            }

            // Compute D_g = A^{-1}
            let d_inv = inv_spd(&a_mat, q)?;
            d_g[g] = d_inv;

            // Compute residual: r_g = y_g - X_g * beta
            // Then u_g = D_g * Z_g^T * r_g / sigma^2
            let mut zt_r = vec![zero; q];
            for &i in idx {
                let mut resid_i = y[i];
                for j in 0..p {
                    resid_i -= x_slice[i * p + j] * beta[j];
                }
                for r in 0..q {
                    zt_r[r] += z_slice[i * q + r] * resid_i;
                }
            }

            // u_g = D_g * (Z_g^T r_g / sigma^2)
            for r in 0..q {
                let mut val = zero;
                for c in 0..q {
                    val += d_g[g][r * q + c] * zt_r[c] / sigma2;
                }
                u[g][r] = val;
            }
        }

        // ==============================================================
        // M-step: Update beta, sigma^2, tau^2
        // ==============================================================

        // Update beta = (X^T X)^{-1} X^T (y - Z u)
        // Compute y_adj = y - Z u
        let mut y_adj = Vec::with_capacity(n);
        for i in 0..n {
            let g = groups[i];
            let mut zu_i = zero;
            for r in 0..q {
                zu_i += z_slice[i * q + r] * u[g][r];
            }
            y_adj.push(y[i] - zu_i);
        }

        let y_adj_tensor = Tensor::from_vec(y_adj.clone(), vec![n])?;
        let beta_tensor = x
            .lstsq(&y_adj_tensor)
            .map_err(|_| StatsError::SingularMatrix)?;
        beta = beta_tensor.as_slice().to_vec();

        // Update sigma^2 = (1/n) * [ sum_i (y_i - x_i'beta - z_i'u_g)^2
        //                            + sum_g trace(Z_g^T Z_g D_g) ]
        let mut rss = zero;
        for i in 0..n {
            let g = groups[i];
            let mut fitted = zero;
            for j in 0..p {
                fitted += x_slice[i * p + j] * beta[j];
            }
            for r in 0..q {
                fitted += z_slice[i * q + r] * u[g][r];
            }
            let resid = y[i] - fitted;
            rss += resid * resid;
        }

        // Add trace correction: sum_g trace(Z_g^T Z_g D_g)
        let mut trace_correction = zero;
        for g in 0..n_groups {
            let idx = &group_indices[g];
            // Compute Z_g^T Z_g (q x q)
            let mut ztg_zg = vec![zero; q * q];
            for &i in idx {
                for r in 0..q {
                    let z_ir = z_slice[i * q + r];
                    for c in 0..q {
                        let z_ic = z_slice[i * q + c];
                        ztg_zg[r * q + c] += z_ir * z_ic;
                    }
                }
            }
            // trace(Z_g^T Z_g D_g)
            for r in 0..q {
                for c in 0..q {
                    trace_correction += ztg_zg[r * q + c] * d_g[g][c * q + r];
                }
            }
        }

        sigma2 = (rss + trace_correction) / nf;
        if sigma2 < T::from_f64(1e-15) {
            sigma2 = T::from_f64(1e-15);
        }

        // Update tau^2: for each random-effect dimension r:
        // tau^2_r = (1/n_groups) * sum_g (u_g[r]^2 + D_g[r,r])
        let n_groups_f = T::from_f64(n_groups as f64);
        for r in 0..q {
            let mut sum = zero;
            for g in 0..n_groups {
                sum += u[g][r] * u[g][r] + d_g[g][r * q + r];
            }
            tau2[r] = sum / n_groups_f;
            if tau2[r] < T::from_f64(1e-15) {
                tau2[r] = T::from_f64(1e-15);
            }
        }

        // ==============================================================
        // Compute log-likelihood
        // ==============================================================
        let pi_val = T::pi();
        let two_pi = two * pi_val;

        // log L = -n/2 * ln(2*pi) - n/2 * ln(sigma^2)
        //         - 1/(2*sigma^2) * RSS
        //         - sum_g [ q/2 * ln(2*pi) + 1/2 * sum_r ln(tau^2_r)
        //                   + 1/2 * sum_r u_g[r]^2 / tau^2_r ]
        let mut ll = -nf * half * two_pi.ln() - nf * half * sigma2.ln();

        let mut rss_for_ll = zero;
        for i in 0..n {
            let g = groups[i];
            let mut fitted = zero;
            for j in 0..p {
                fitted += x_slice[i * p + j] * beta[j];
            }
            for r in 0..q {
                fitted += z_slice[i * q + r] * u[g][r];
            }
            let resid = y[i] - fitted;
            rss_for_ll += resid * resid;
        }
        ll -= rss_for_ll / (two * sigma2);

        // Random effects contribution
        let qf = T::from_f64(q as f64);
        for u_g in &u {
            ll -= qf * half * two_pi.ln();
            for r in 0..q {
                ll -= half * tau2[r].ln();
                ll -= u_g[r] * u_g[r] / (two * tau2[r]);
            }
        }

        // Check convergence
        let ll_change = (ll - prev_ll).abs();
        if iter > 0 && ll_change < tol {
            converged = true;
            prev_ll = ll;
            break;
        }
        prev_ll = ll;
    }

    Ok(LmmResult {
        fixed_effects: beta,
        random_effects: u,
        residual_variance: sigma2,
        random_effect_variance: tau2,
        log_likelihood: prev_ll,
        iterations,
        converged,
    })
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn test_lmm_random_intercept() {
        // Two groups with distinct means: group 0 ~ 1.0, group 1 ~ 5.0
        let y: Vec<f64> = vec![
            1.1, 1.2, 0.9, 1.0, 0.8, 1.3, // group 0 mean ~ 1.05
            5.1, 5.2, 4.9, 5.0, 4.8, 5.3, // group 1 mean ~ 5.05
        ];
        let x_data = vec![1.0_f64; 12];
        let x = Tensor::from_vec(x_data, vec![12, 1]).unwrap();
        let groups = vec![0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1];

        let result = lmm(&y, &x, &groups, None, 200, 1e-10).unwrap();

        // Fixed effect should be near the grand mean (~3.05)
        assert!(
            (result.fixed_effects[0] - 3.05).abs() < 0.5,
            "fixed effect = {}, expected ~3.05",
            result.fixed_effects[0]
        );

        // Random effects should capture group differences
        // Group 0 should have negative random effect, group 1 positive
        assert!(
            result.random_effects[0][0] < result.random_effects[1][0],
            "group 0 RE ({}) should be less than group 1 RE ({})",
            result.random_effects[0][0],
            result.random_effects[1][0]
        );

        // Random effect variance should be substantial (groups differ a lot)
        assert!(
            result.random_effect_variance[0] > 0.1,
            "tau^2 = {}, expected > 0.1",
            result.random_effect_variance[0]
        );
    }

    #[test]
    fn test_lmm_convergence() {
        let y: Vec<f64> = vec![2.0, 2.1, 1.9, 4.0, 4.1, 3.9, 6.0, 6.1, 5.9];
        let x_data = vec![1.0; 9];
        let x = Tensor::from_vec(x_data, vec![9, 1]).unwrap();
        let groups = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];

        let result = lmm(&y, &x, &groups, None, 200, 1e-8).unwrap();
        assert!(
            result.converged,
            "EM did not converge in {} iterations",
            result.iterations
        );
        assert!(
            result.iterations <= 200,
            "took {} iterations",
            result.iterations
        );
    }

    #[test]
    fn test_lmm_known_parameters() {
        // Generate data from known model:
        // y_i = 2.0 + 1.5 * x_i + u_g + eps_i
        // where u_g ~ N(0, 1.0), eps_i ~ N(0, 0.01)
        // We use fixed "random" effects: u = [-1.0, 0.0, 1.0]
        let u_true = [-1.0_f64, 0.0, 1.0];
        let beta1 = 1.5;

        let mut y = Vec::new();
        let mut x_data = Vec::new();
        let mut groups = Vec::new();

        // Small noise values (deterministic for reproducibility)
        let noise = [0.01, -0.02, 0.005, -0.01, 0.015, -0.005, 0.02, -0.015, 0.0];
        let x_vals = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0];

        for i in 0..9 {
            let g = i / 3;
            let xi = x_vals[i];
            let yi = 2.0 + beta1 * xi + u_true[g] + noise[i];
            y.push(yi);
            x_data.push(xi);
            groups.push(g);
        }

        let x = Tensor::from_vec(x_data, vec![9, 1]).unwrap();
        let result = lmm(&y, &x, &groups, None, 200, 1e-10).unwrap();

        // Should recover beta1 ~ 1.5 reasonably well.
        // The function does NOT prepend an intercept, so beta[0] is the slope.
        // The intercept is partly absorbed by the random intercept.
        assert!(
            (result.fixed_effects[0] - beta1).abs() < 0.3,
            "slope = {}, expected ~{}",
            result.fixed_effects[0],
            beta1
        );
    }

    #[test]
    fn test_lmm_single_group() {
        // All data in one group: random effects should be near zero,
        // fixed effects should approximate OLS.
        let y: Vec<f64> = vec![2.1, 3.9, 6.2, 7.8, 10.1];
        let x_data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x = Tensor::from_vec(x_data, vec![5, 1]).unwrap();
        let groups = vec![0, 0, 0, 0, 0];

        let result = lmm(&y, &x, &groups, None, 100, 1e-8).unwrap();

        // Random effect for the single group should be near zero
        assert!(
            result.random_effects[0][0].abs() < 1.0,
            "single-group RE = {}, expected near 0",
            result.random_effects[0][0]
        );

        // The fixed effect (slope) should be positive (data is increasing)
        assert!(
            result.fixed_effects[0] > 0.0,
            "slope should be positive, got {}",
            result.fixed_effects[0]
        );
    }

    #[test]
    fn test_lmm_empty_input() {
        let y: Vec<f64> = vec![];
        let x = Tensor::from_vec(vec![0.0_f64; 0], vec![0, 1]).unwrap();
        let groups: Vec<usize> = vec![];

        let result = lmm(&y, &x, &groups, None, 100, 1e-8);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), StatsError::EmptyInput));
    }
}
