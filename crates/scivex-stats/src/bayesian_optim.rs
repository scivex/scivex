//! Bayesian optimization using a Gaussian process surrogate with acquisition functions.
//!
//! This module provides a black-box optimizer that builds a probabilistic model
//! (Gaussian process) of the objective function and uses acquisition functions
//! (Expected Improvement, Upper Confidence Bound, Probability of Improvement)
//! to decide where to evaluate next.

use scivex_core::Float;
use scivex_core::random::Rng;

use crate::error::{Result, StatsError};

// ─── Normal distribution helpers ─────────────────────────────────────────────

/// Standard normal PDF: φ(x) = exp(-x²/2) / √(2π).
fn standard_normal_pdf<T: Float>(x: T) -> T {
    let two = T::from_f64(2.0);
    let half = T::from_f64(0.5);
    (-(x * x) * half).exp() / (two * T::pi()).sqrt()
}

/// Standard normal CDF via Abramowitz & Stegun rational approximation (formula 26.2.17).
/// Maximum absolute error ≈ 7.5 × 10⁻⁸.
fn standard_normal_cdf<T: Float>(x: T) -> T {
    let zero = T::zero();
    let one = T::one();

    if x < T::from_f64(-8.0) {
        return zero;
    }
    if x > T::from_f64(8.0) {
        return one;
    }

    let is_negative = x < zero;
    let x_abs = x.abs();

    let b1 = T::from_f64(0.319_381_530);
    let b2 = T::from_f64(-0.356_563_782);
    let b3 = T::from_f64(1.781_477_937);
    let b4 = T::from_f64(-1.821_255_978);
    let b5 = T::from_f64(1.330_274_429);
    let p = T::from_f64(0.231_641_9);

    let t = one / (one + p * x_abs);
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t3 * t;
    let t5 = t4 * t;

    let poly = b1 * t + b2 * t2 + b3 * t3 + b4 * t4 + b5 * t5;
    let cdf_positive = one - standard_normal_pdf(x_abs) * poly;

    if is_negative {
        one - cdf_positive
    } else {
        cdf_positive
    }
}

// ─── Kernel ──────────────────────────────────────────────────────────────────

/// Kernel function for the Gaussian process.
#[derive(Debug, Clone)]
pub enum Kernel {
    /// Squared exponential (RBF) kernel: k(x, x') = σ² exp(−‖x − x'‖² / (2l²)).
    SquaredExponential {
        length_scale: f64,
        signal_variance: f64,
    },
    /// Matérn 5/2 kernel.
    Matern52 {
        length_scale: f64,
        signal_variance: f64,
    },
}

impl Kernel {
    /// Evaluate the kernel between two points.
    fn eval<T: Float>(&self, x: &[T], y: &[T]) -> T {
        match self {
            Self::SquaredExponential {
                length_scale,
                signal_variance,
            } => {
                let l = T::from_f64(*length_scale);
                let sv = T::from_f64(*signal_variance);
                let mut sq_dist = T::zero();
                for i in 0..x.len() {
                    let d = x[i] - y[i];
                    sq_dist += d * d;
                }
                sv * (sq_dist * T::from_f64(-0.5) / (l * l)).exp()
            }
            Self::Matern52 {
                length_scale,
                signal_variance,
            } => {
                let l = T::from_f64(*length_scale);
                let sv = T::from_f64(*signal_variance);
                let mut sq_dist = T::zero();
                for i in 0..x.len() {
                    let d = x[i] - y[i];
                    sq_dist += d * d;
                }
                let r = sq_dist.sqrt();
                let sqrt5 = T::from_f64(5.0_f64.sqrt());
                let scaled = sqrt5 * r / l;
                let three = T::from_f64(3.0);
                let five = T::from_f64(5.0);
                sv * (T::one() + scaled + five * sq_dist / (three * l * l)) * (-scaled).exp()
            }
        }
    }
}

// ─── Acquisition function ────────────────────────────────────────────────────

/// Acquisition function for selecting the next evaluation point.
#[derive(Debug, Clone)]
pub enum AcquisitionFunction {
    /// Expected Improvement.
    ExpectedImprovement,
    /// Upper Confidence Bound with exploration parameter κ.
    UpperConfidenceBound { kappa: f64 },
    /// Probability of Improvement.
    ProbabilityOfImprovement,
}

impl AcquisitionFunction {
    /// Evaluate the acquisition function given GP predictions and the current best.
    fn eval<T: Float>(&self, mean: T, std: T, best_y: T) -> T {
        let eps = T::from_f64(1e-12);
        if std < eps {
            return T::zero();
        }
        match self {
            Self::ExpectedImprovement => {
                let z = (mean - best_y) / std;
                let cdf_z = standard_normal_cdf(z);
                let pdf_z = standard_normal_pdf(z);
                (mean - best_y) * cdf_z + std * pdf_z
            }
            Self::UpperConfidenceBound { kappa } => mean + T::from_f64(*kappa) * std,
            Self::ProbabilityOfImprovement => {
                let z = (mean - best_y) / std;
                standard_normal_cdf(z)
            }
        }
    }
}

// ─── Configuration ───────────────────────────────────────────────────────────

/// Configuration for Bayesian optimization.
#[derive(Debug, Clone)]
pub struct BayesOptConfig {
    /// Kernel for the Gaussian process surrogate.
    pub kernel: Kernel,
    /// Acquisition function used to select the next evaluation point.
    pub acquisition: AcquisitionFunction,
    /// Observation noise variance σ_n².
    pub noise_variance: f64,
    /// Number of random initial points (default 5).
    pub n_initial: usize,
    /// Number of Bayesian optimization iterations (default 50).
    pub n_iterations: usize,
    /// Random seed.
    pub seed: u64,
}

impl Default for BayesOptConfig {
    fn default() -> Self {
        Self {
            kernel: Kernel::SquaredExponential {
                length_scale: 1.0,
                signal_variance: 1.0,
            },
            acquisition: AcquisitionFunction::ExpectedImprovement,
            noise_variance: 1e-6,
            n_initial: 5,
            n_iterations: 50,
            seed: 42,
        }
    }
}

// ─── Result ──────────────────────────────────────────────────────────────────

/// Result of Bayesian optimization.
#[derive(Debug, Clone)]
pub struct BayesOptResult<T: Float> {
    /// Best input found.
    pub best_x: Vec<T>,
    /// Best objective value found.
    pub best_y: T,
    /// History of all evaluated inputs.
    pub x_history: Vec<Vec<T>>,
    /// History of all evaluated objective values.
    pub y_history: Vec<T>,
    /// Total number of iterations performed.
    pub iterations: usize,
}

// ─── Cholesky decomposition (inline, small matrices) ─────────────────────────

/// Cholesky decomposition of a symmetric positive-definite matrix (row-major).
/// Returns the lower-triangular factor L such that A = L Lᵀ.
fn cholesky<T: Float>(a: &[T], n: usize) -> Result<Vec<T>> {
    let mut l = vec![T::zero(); n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = T::zero();
            for k in 0..j {
                s += l[i * n + k] * l[j * n + k];
            }
            if i == j {
                let diag = a[i * n + i] - s;
                if diag <= T::zero() {
                    return Err(StatsError::SingularMatrix);
                }
                l[i * n + j] = diag.sqrt();
            } else {
                l[i * n + j] = (a[i * n + j] - s) / l[j * n + j];
            }
        }
    }
    Ok(l)
}

/// Solve L y = b (forward substitution) where L is lower-triangular, row-major.
fn forward_sub<T: Float>(l: &[T], b: &[T], n: usize) -> Vec<T> {
    let mut y = vec![T::zero(); n];
    for i in 0..n {
        let mut s = T::zero();
        for j in 0..i {
            s += l[i * n + j] * y[j];
        }
        y[i] = (b[i] - s) / l[i * n + i];
    }
    y
}

/// Solve Lᵀ x = y (back substitution) where L is lower-triangular, row-major.
fn back_sub<T: Float>(l: &[T], y: &[T], n: usize) -> Vec<T> {
    let mut x = vec![T::zero(); n];
    for i in (0..n).rev() {
        let mut s = T::zero();
        for j in (i + 1)..n {
            s += l[j * n + i] * x[j]; // L' element [i][j] = L[j][i]
        }
        x[i] = (y[i] - s) / l[i * n + i];
    }
    x
}

// ─── Bayesian Optimizer ──────────────────────────────────────────────────────

/// Bayesian optimizer using a Gaussian process surrogate and acquisition functions.
///
/// # Examples
///
/// ```
/// use scivex_stats::bayesian_optim::{BayesianOptimizer, BayesOptConfig};
///
/// let bounds = vec![(0.0_f64, 6.0)];
/// let config = BayesOptConfig { n_initial: 5, n_iterations: 30, ..Default::default() };
/// let mut opt = BayesianOptimizer::new(bounds, config);
/// let result = opt.minimize(|x: &[f64]| (x[0] - 3.0) * (x[0] - 3.0)).unwrap();
/// assert!((result.best_x[0] - 3.0).abs() < 1.0);
/// ```
pub struct BayesianOptimizer<T: Float> {
    config: BayesOptConfig,
    x_observed: Vec<Vec<T>>,
    y_observed: Vec<T>,
    bounds: Vec<(T, T)>,
    rng: Rng,
}

impl<T: Float> BayesianOptimizer<T> {
    /// Create a new Bayesian optimizer.
    ///
    /// # Arguments
    ///
    /// * `bounds` — Per-dimension `(lower, upper)` bounds for the search space.
    /// * `config` — Optimization configuration.
    pub fn new(bounds: Vec<(T, T)>, config: BayesOptConfig) -> Self {
        let rng = Rng::new(config.seed);
        Self {
            config,
            x_observed: Vec::new(),
            y_observed: Vec::new(),
            bounds,
            rng,
        }
    }

    /// Generate a random point within the bounds.
    fn random_point(&mut self) -> Vec<T> {
        self.bounds
            .iter()
            .map(|&(lo, hi)| {
                let u = T::from_f64(self.rng.next_f64());
                lo + u * (hi - lo)
            })
            .collect()
    }

    /// Build the kernel matrix K(X, X) + σ_n² I.
    fn build_kernel_matrix(&self) -> Vec<T> {
        let n = self.x_observed.len();
        let noise = T::from_f64(self.config.noise_variance);
        let mut k = vec![T::zero(); n * n];
        for i in 0..n {
            for j in 0..=i {
                let val = self
                    .config
                    .kernel
                    .eval(&self.x_observed[i], &self.x_observed[j]);
                k[i * n + j] = val;
                k[j * n + i] = val;
            }
            k[i * n + i] += noise;
        }
        k
    }

    /// Compute the GP predictive mean and variance at a test point.
    fn gp_predict(&self, x_star: &[T], alpha: &[T], k_mat: &[T], l: &[T]) -> (T, T) {
        let n = self.x_observed.len();

        // k(X, x*) — kernel vector between training points and test point
        let k_star: Vec<T> = (0..n)
            .map(|i| self.config.kernel.eval(&self.x_observed[i], x_star))
            .collect();

        // Predictive mean: μ = k*' α
        let mut mean = T::zero();
        for i in 0..n {
            mean += k_star[i] * alpha[i];
        }

        // Predictive variance: σ² = k(x*,x*) - k*' K⁻¹ k* = k(x*,x*) - v' v
        // where L v = k*
        let v = forward_sub(l, &k_star, n);
        let mut v_dot = T::zero();
        for vi in &v {
            v_dot += *vi * *vi;
        }

        let k_self = self.config.kernel.eval(x_star, x_star);
        let var = (k_self - v_dot).abs(); // abs for numerical safety

        // Ignore `k_mat` usage — we pass L directly for efficiency
        let _ = k_mat;

        (mean, var)
    }

    /// Index of the best (maximum) observed y value.
    fn best_idx(&self) -> usize {
        let mut best = 0;
        for i in 1..self.y_observed.len() {
            if self.y_observed[i] > self.y_observed[best] {
                best = i;
            }
        }
        best
    }

    /// Run the full optimization loop, maximizing the objective.
    ///
    /// The `objective` closure is called at each candidate point to get the
    /// function value.  We **maximize** the objective.
    #[allow(clippy::too_many_lines)]
    pub fn maximize<F>(&mut self, objective: F) -> Result<BayesOptResult<T>>
    where
        F: Fn(&[T]) -> T,
    {
        if self.bounds.is_empty() {
            return Err(StatsError::InvalidParameter {
                name: "bounds",
                reason: "bounds must not be empty",
            });
        }

        let dim = self.bounds.len();
        let _ = dim; // used implicitly through bounds

        // Phase 1: random initial evaluations
        for _ in 0..self.config.n_initial {
            let x = self.random_point();
            let y = objective(&x);
            self.x_observed.push(x);
            self.y_observed.push(y);
        }

        let n_candidates: usize = 1000;

        // Phase 2: BO loop
        for _ in 0..self.config.n_iterations {
            let n = self.x_observed.len();

            // Build kernel matrix and Cholesky factor
            let k_mat = self.build_kernel_matrix();
            let l = cholesky(&k_mat, n)?;

            // Solve K α = y
            let y_fwd = forward_sub(&l, &self.y_observed, n);
            let alpha = back_sub(&l, &y_fwd, n);

            // Current best
            let best_y = self.y_observed[self.best_idx()];

            // Find next point by maximizing acquisition over random candidates
            let mut best_acq = T::from_f64(f64::NEG_INFINITY);
            let mut best_candidate: Option<Vec<T>> = None;

            for _ in 0..n_candidates {
                let cand = self.random_point();
                let (mu, var) = self.gp_predict(&cand, &alpha, &k_mat, &l);
                let std = var.sqrt();
                let acq = self.config.acquisition.eval(mu, std, best_y);

                if acq > best_acq {
                    best_acq = acq;
                    best_candidate = Some(cand);
                }
            }

            if let Some(x_next) = best_candidate {
                let y_next = objective(&x_next);
                self.x_observed.push(x_next);
                self.y_observed.push(y_next);
            }
        }

        // Build result
        let best = self.best_idx();
        Ok(BayesOptResult {
            best_x: self.x_observed[best].clone(),
            best_y: self.y_observed[best],
            x_history: self.x_observed.clone(),
            y_history: self.y_observed.clone(),
            iterations: self.config.n_iterations,
        })
    }

    /// Run optimization to **minimize** the objective.
    ///
    /// Internally negates the objective and calls [`maximize`](Self::maximize).
    pub fn minimize<F>(&mut self, objective: F) -> Result<BayesOptResult<T>>
    where
        F: Fn(&[T]) -> T,
    {
        let result = self.maximize(|x| {
            let val = objective(x);
            T::zero() - val
        })?;

        // Negate y values back
        Ok(BayesOptResult {
            best_x: result.best_x,
            best_y: T::zero() - result.best_y,
            x_history: result.x_history,
            y_history: result
                .y_history
                .into_iter()
                .map(|y| T::zero() - y)
                .collect(),
            iterations: result.iterations,
        })
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bayesopt_1d_quadratic() {
        let bounds = vec![(0.0_f64, 6.0)];
        let config = BayesOptConfig {
            n_initial: 5,
            n_iterations: 40,
            seed: 123,
            noise_variance: 1e-6,
            ..Default::default()
        };
        let mut opt = BayesianOptimizer::new(bounds, config);
        let result = opt
            .minimize(|x: &[f64]| (x[0] - 3.0) * (x[0] - 3.0))
            .unwrap();

        assert!(
            (result.best_x[0] - 3.0).abs() < 1.0,
            "Expected best_x ≈ 3.0, got {}",
            result.best_x[0]
        );
        assert!(
            result.best_y < 1.0,
            "Expected best_y near 0, got {}",
            result.best_y
        );
    }

    #[test]
    fn test_bayesopt_2d() {
        let bounds = vec![(-5.0_f64, 5.0), (-5.0, 5.0)];
        let config = BayesOptConfig {
            n_initial: 10,
            n_iterations: 50,
            seed: 7,
            noise_variance: 1e-6,
            ..Default::default()
        };
        let mut opt = BayesianOptimizer::new(bounds, config);
        let result = opt
            .minimize(|x: &[f64]| (x[0] - 1.0).powi(2) + (x[1] + 2.0).powi(2))
            .unwrap();

        assert!(
            (result.best_x[0] - 1.0).abs() < 2.0,
            "Expected x ≈ 1.0, got {}",
            result.best_x[0]
        );
        assert!(
            (result.best_x[1] + 2.0).abs() < 2.0,
            "Expected y ≈ -2.0, got {}",
            result.best_x[1]
        );
    }

    #[test]
    fn test_bayesopt_ucb() {
        let bounds = vec![(0.0_f64, 6.0)];
        let config = BayesOptConfig {
            kernel: Kernel::Matern52 {
                length_scale: 1.0,
                signal_variance: 1.0,
            },
            acquisition: AcquisitionFunction::UpperConfidenceBound { kappa: 2.0 },
            n_initial: 5,
            n_iterations: 40,
            seed: 42,
            noise_variance: 1e-6,
        };
        let mut opt = BayesianOptimizer::new(bounds, config);
        let result = opt
            .minimize(|x: &[f64]| (x[0] - 3.0) * (x[0] - 3.0))
            .unwrap();

        assert!(
            (result.best_x[0] - 3.0).abs() < 1.5,
            "UCB: Expected best_x ≈ 3.0, got {}",
            result.best_x[0]
        );
    }

    #[test]
    fn test_bayesopt_maximize() {
        let bounds = vec![(-5.0_f64, 5.0)];
        let config = BayesOptConfig {
            n_initial: 5,
            n_iterations: 40,
            seed: 99,
            noise_variance: 1e-6,
            ..Default::default()
        };
        let mut opt = BayesianOptimizer::new(bounds, config);
        let result = opt.maximize(|x: &[f64]| -(x[0] * x[0])).unwrap();

        assert!(
            result.best_x[0].abs() < 2.0,
            "Expected best_x ≈ 0.0, got {}",
            result.best_x[0]
        );
        assert!(
            result.best_y > -4.0,
            "Expected best_y near 0, got {}",
            result.best_y
        );
    }

    #[test]
    fn test_bayesopt_empty_bounds() {
        let bounds: Vec<(f64, f64)> = vec![];
        let config = BayesOptConfig::default();
        let mut opt = BayesianOptimizer::new(bounds, config);
        let result = opt.minimize(|_: &[f64]| 0.0);

        assert!(result.is_err());
        match result {
            Err(StatsError::InvalidParameter { name, .. }) => {
                assert_eq!(name, "bounds");
            }
            other => panic!("Expected InvalidParameter, got {other:?}"),
        }
    }
}
