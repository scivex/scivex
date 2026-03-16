//! Generalized Linear Models (GLM) via Iteratively Reweighted Least Squares.

use scivex_core::Float;
use scivex_core::Tensor;

use crate::distributions::{Distribution, Normal};
use crate::error::{Result, StatsError};

// ---------------------------------------------------------------------------
// Family & Link
// ---------------------------------------------------------------------------

/// Exponential family distribution for the response.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Family {
    /// Normal/Gaussian errors.
    Gaussian,
    /// Binomial (logistic regression).
    Binomial,
    /// Poisson count data.
    Poisson,
    /// Gamma distribution.
    Gamma,
}

/// Link function mapping the mean to the linear predictor.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkFunction {
    /// mu = eta (identity).
    Identity,
    /// eta = logit(mu) = ln(mu / (1 - mu)).
    Logit,
    /// eta = ln(mu).
    Log,
    /// eta = 1 / mu.
    Inverse,
}

/// Result of a GLM fit.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct GlmResult<T: Float> {
    /// Estimated coefficients (intercept at index 0).
    pub coefficients: Vec<T>,
    /// Standard errors of the coefficients.
    pub std_errors: Vec<T>,
    /// z-scores (coefficient / std_error).
    pub z_scores: Vec<T>,
    /// Two-sided p-values from standard normal.
    pub p_values: Vec<T>,
    /// Deviance of the fitted model.
    pub deviance: T,
    /// Akaike Information Criterion.
    pub aic: T,
    /// Log-likelihood at convergence.
    pub log_likelihood: T,
    /// Number of IRLS iterations used.
    pub n_iter: usize,
}

// ---------------------------------------------------------------------------
// Link helpers
// ---------------------------------------------------------------------------

fn link_fn<T: Float>(link: LinkFunction, mu: T) -> T {
    let one = T::from_f64(1.0);
    match link {
        LinkFunction::Identity => mu,
        LinkFunction::Logit => (mu / (one - mu)).ln(),
        LinkFunction::Log => mu.ln(),
        LinkFunction::Inverse => one / mu,
    }
}

fn link_inv<T: Float>(link: LinkFunction, eta: T) -> T {
    let one = T::from_f64(1.0);
    match link {
        LinkFunction::Identity => eta,
        LinkFunction::Logit => one / (one + (-eta).exp()),
        LinkFunction::Log => eta.exp(),
        LinkFunction::Inverse => one / eta,
    }
}

fn link_inv_deriv<T: Float>(link: LinkFunction, eta: T) -> T {
    let one = T::from_f64(1.0);
    match link {
        LinkFunction::Identity => one,
        LinkFunction::Logit => {
            let mu = one / (one + (-eta).exp());
            mu * (one - mu)
        }
        LinkFunction::Log => eta.exp(),
        LinkFunction::Inverse => -one / (eta * eta),
    }
}

fn variance_fn<T: Float>(family: Family, mu: T) -> T {
    let one = T::from_f64(1.0);
    match family {
        Family::Gaussian => one,
        Family::Binomial => mu * (one - mu),
        Family::Poisson => mu,
        Family::Gamma => mu * mu,
    }
}

fn deviance_unit<T: Float>(family: Family, y: T, mu: T) -> T {
    let two = T::from_f64(2.0);
    let one = T::from_f64(1.0);
    let eps = T::from_f64(1e-15);
    match family {
        Family::Gaussian => (y - mu) * (y - mu),
        Family::Binomial => {
            let mut d = T::from_f64(0.0);
            if y > eps {
                d += y * (y / mu).ln();
            }
            if (one - y) > eps {
                d += (one - y) * ((one - y) / (one - mu)).ln();
            }
            two * d
        }
        Family::Poisson => {
            if y > eps {
                two * (y * (y / mu).ln() - (y - mu))
            } else {
                two * mu
            }
        }
        Family::Gamma => two * (-(y / mu).ln() + (y - mu) / mu),
    }
}

fn log_lik_unit<T: Float>(family: Family, y: T, mu: T) -> T {
    let eps = T::from_f64(1e-15);
    let one = T::from_f64(1.0);
    match family {
        Family::Gaussian => T::from_f64(-0.5) * (y - mu) * (y - mu),
        Family::Binomial => y * (mu + eps).ln() + (one - y) * (one - mu + eps).ln(),
        Family::Poisson => y * (mu + eps).ln() - mu,
        Family::Gamma => -(y / mu) - mu.ln(),
    }
}

// ---------------------------------------------------------------------------
// GLM fitting via IRLS
// ---------------------------------------------------------------------------

/// Fit a Generalized Linear Model using Iteratively Reweighted Least Squares.
///
/// `x` is a `[n x p]` tensor of predictors (an intercept column is prepended
/// automatically). `y` is a slice of `n` response values.
#[allow(clippy::too_many_lines)]
#[allow(clippy::similar_names)]
pub fn glm<T: Float>(
    x: &Tensor<T>,
    y: &[T],
    family: Family,
    link: LinkFunction,
) -> Result<GlmResult<T>> {
    if x.ndim() != 2 {
        return Err(StatsError::InvalidParameter {
            name: "x",
            reason: "must be a 2-D tensor",
        });
    }
    let n = x.shape()[0];
    let p_raw = x.shape()[1];
    if y.len() != n {
        return Err(StatsError::LengthMismatch {
            expected: n,
            got: y.len(),
        });
    }
    let k = p_raw + 1;
    if n <= k {
        return Err(StatsError::InsufficientData {
            need: k + 1,
            got: n,
        });
    }

    let one = T::from_f64(1.0);
    let zero = T::from_f64(0.0);
    let two = T::from_f64(2.0);
    let eps = T::from_f64(1e-10);

    let x_slice = x.as_slice();
    let mut design = Vec::with_capacity(n * k);
    for i in 0..n {
        design.push(one);
        for j in 0..p_raw {
            design.push(x_slice[i * p_raw + j]);
        }
    }

    let mut mu: Vec<T> = y
        .iter()
        .map(|&yi| match family {
            Family::Binomial => yi.max(T::from_f64(0.01)).min(T::from_f64(0.99)),
            Family::Poisson => yi.max(T::from_f64(0.1)),
            Family::Gamma => yi.max(T::from_f64(0.01)),
            Family::Gaussian => yi,
        })
        .collect();

    let mut eta: Vec<T> = mu.iter().map(|&m| link_fn(link, m)).collect();
    let mut beta = vec![zero; k];

    let max_iter = 100;
    let mut n_iter = 0;

    for iteration in 0..max_iter {
        n_iter = iteration + 1;

        let mut w_vec = Vec::with_capacity(n);
        let mut z_vec = Vec::with_capacity(n);

        for i in 0..n {
            let g_prime = link_inv_deriv(link, eta[i]);
            let v = variance_fn(family, mu[i]);
            let weight = if v.abs() > eps && g_prime.abs() > eps {
                g_prime * g_prime / v
            } else {
                eps
            };
            w_vec.push(weight);

            let working_resid = (y[i] - mu[i]) / g_prime.max(eps);
            z_vec.push(eta[i] + working_resid);
        }

        let mut xtx = vec![zero; k * k];
        let mut xtz = vec![zero; k];

        for i in 0..n {
            let wi = w_vec[i];
            let xi = &design[i * k..(i + 1) * k];
            for j in 0..k {
                xtz[j] += xi[j] * wi * z_vec[i];
                for l in j..k {
                    let val = xi[j] * wi * xi[l];
                    xtx[j * k + l] += val;
                    if l != j {
                        xtx[l * k + j] += val;
                    }
                }
            }
        }

        let xtx_t = Tensor::from_vec(xtx, vec![k, k])?;
        let xtz_t = Tensor::from_slice(&xtz, vec![k])?;
        let new_beta_t = xtx_t
            .lstsq(&xtz_t)
            .map_err(|_| StatsError::SingularMatrix)?;
        let new_beta = new_beta_t.as_slice().to_vec();

        let mut max_change = zero;
        for j in 0..k {
            let change = (new_beta[j] - beta[j]).abs();
            if change > max_change {
                max_change = change;
            }
        }
        beta = new_beta;

        for i in 0..n {
            let xi = &design[i * k..(i + 1) * k];
            eta[i] = xi.iter().zip(beta.iter()).map(|(&x_ij, &b)| x_ij * b).sum();
            mu[i] = link_inv(link, eta[i]);
            match family {
                Family::Binomial => {
                    mu[i] = mu[i].max(T::from_f64(1e-10)).min(one - T::from_f64(1e-10));
                }
                Family::Poisson | Family::Gamma => {
                    mu[i] = mu[i].max(T::from_f64(1e-10));
                }
                Family::Gaussian => {}
            }
        }

        if max_change < eps {
            break;
        }
    }

    let deviance: T = (0..n).map(|i| deviance_unit(family, y[i], mu[i])).sum();
    let log_likelihood: T = (0..n).map(|i| log_lik_unit(family, y[i], mu[i])).sum();
    let aic = -two * log_likelihood + two * T::from_f64(k as f64);

    // Standard errors from (X^T W X)^{-1}.
    let mut w_vec_final = Vec::with_capacity(n);
    for i in 0..n {
        let g_prime = link_inv_deriv(link, eta[i]);
        let v = variance_fn(family, mu[i]);
        let weight = if v.abs() > eps && g_prime.abs() > eps {
            g_prime * g_prime / v
        } else {
            eps
        };
        w_vec_final.push(weight);
    }

    let mut xtx_final = vec![zero; k * k];
    for i in 0..n {
        let wi = w_vec_final[i];
        let xi = &design[i * k..(i + 1) * k];
        for j in 0..k {
            for l in j..k {
                let val = xi[j] * wi * xi[l];
                xtx_final[j * k + l] += val;
                if l != j {
                    xtx_final[l * k + j] += val;
                }
            }
        }
    }

    let info_t = Tensor::from_vec(xtx_final, vec![k, k])?;
    let info_inv = info_t.inv().map_err(|_| StatsError::SingularMatrix)?;
    let info_inv_s = info_inv.as_slice();

    let normal = Normal::standard();
    let mut std_errors = Vec::with_capacity(k);
    let mut z_scores = Vec::with_capacity(k);
    let mut p_values = Vec::with_capacity(k);

    let dispersion = if family == Family::Gaussian {
        deviance / T::from_f64((n - k) as f64)
    } else {
        one
    };

    for j in 0..k {
        let var_j = info_inv_s[j * k + j] * dispersion;
        let se = if var_j > zero { var_j.sqrt() } else { zero };
        std_errors.push(se);
        let z = if se > zero { beta[j] / se } else { zero };
        z_scores.push(z);
        let pv = two * (one - normal.cdf(z.abs()));
        p_values.push(pv);
    }

    Ok(GlmResult {
        coefficients: beta,
        std_errors,
        z_scores,
        p_values,
        deviance,
        aic,
        log_likelihood,
        n_iter,
    })
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use crate::regression::ols;

    #[test]
    fn test_logistic_regression() {
        let x_data: Vec<f64> = vec![-3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0];
        let y: Vec<f64> = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let x = Tensor::from_vec(x_data, vec![8, 1]).unwrap();
        let result = glm(&x, &y, Family::Binomial, LinkFunction::Logit).unwrap();
        assert!(result.coefficients[1] > 0.0);
    }

    #[test]
    fn test_poisson_regression() {
        let x_data: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let y: Vec<f64> = vec![1.0, 2.0, 4.0, 7.0, 12.0, 20.0, 35.0, 55.0];
        let x = Tensor::from_vec(x_data, vec![8, 1]).unwrap();
        let result = glm(&x, &y, Family::Poisson, LinkFunction::Log).unwrap();
        assert!(result.coefficients[1] > 0.0);
    }

    #[test]
    fn test_gaussian_matches_ols() {
        let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y: Vec<f64> = vec![2.1, 3.9, 6.2, 7.8, 10.1, 12.0, 13.9, 16.1, 18.0, 20.2];
        let x = Tensor::from_vec(x_data, vec![10, 1]).unwrap();

        let glm_result = glm(&x, &y, Family::Gaussian, LinkFunction::Identity).unwrap();
        let ols_result = ols(&x, &y).unwrap();

        for j in 0..2 {
            assert!(
                (glm_result.coefficients[j] - ols_result.coefficients[j]).abs() < 0.1,
                "coef {j}: glm={}, ols={}",
                glm_result.coefficients[j],
                ols_result.coefficients[j],
            );
        }
    }

    #[test]
    fn test_convergence() {
        let x_data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y: Vec<f64> = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let x = Tensor::from_vec(x_data, vec![8, 1]).unwrap();
        let result = glm(&x, &y, Family::Binomial, LinkFunction::Logit).unwrap();
        assert!(result.n_iter <= 100);
    }

    #[test]
    fn test_invalid_parameters() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let y = vec![1.0, 2.0, 3.0];
        assert!(glm(&x, &y, Family::Gaussian, LinkFunction::Identity).is_err());
    }
}
