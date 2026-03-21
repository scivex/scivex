//! L-BFGS-B: Limited-memory BFGS with box constraints.
//!
//! Extends L-BFGS to handle per-variable lower and upper bounds. Uses
//! projected gradient steps with a limited-memory (m=10) two-loop
//! recursion for the approximate inverse Hessian.

use scivex_core::{Float, Tensor};

use crate::error::Result;

use super::{MinimizeOptions, MinimizeResult};

/// Per-variable box constraint.
///
/// # Examples
///
/// ```
/// # use scivex_optim::minimize::Bounds;
/// let b = Bounds::both(0.0_f64, 1.0);
/// assert_eq!(b.lower, Some(0.0));
/// assert_eq!(b.upper, Some(1.0));
/// ```
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct Bounds<T> {
    /// Optional lower bound.
    pub lower: Option<T>,
    /// Optional upper bound.
    pub upper: Option<T>,
}

impl<T> Bounds<T> {
    /// No bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_optim::minimize::Bounds;
    /// let b = Bounds::<f64>::none();
    /// assert!(b.lower.is_none());
    /// assert!(b.upper.is_none());
    /// ```
    pub fn none() -> Self {
        Self {
            lower: None,
            upper: None,
        }
    }

    /// Lower bound only.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_optim::minimize::Bounds;
    /// let b = Bounds::lower(0.0_f64);
    /// assert_eq!(b.lower, Some(0.0));
    /// ```
    pub fn lower(lb: T) -> Self {
        Self {
            lower: Some(lb),
            upper: None,
        }
    }

    /// Upper bound only.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_optim::minimize::Bounds;
    /// let b = Bounds::upper(10.0_f64);
    /// assert_eq!(b.upper, Some(10.0));
    /// ```
    pub fn upper(ub: T) -> Self {
        Self {
            lower: None,
            upper: Some(ub),
        }
    }

    /// Both lower and upper bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_optim::minimize::Bounds;
    /// let b = Bounds::both(-1.0_f64, 1.0);
    /// assert_eq!(b.lower, Some(-1.0));
    /// assert_eq!(b.upper, Some(1.0));
    /// ```
    pub fn both(lb: T, ub: T) -> Self {
        Self {
            lower: Some(lb),
            upper: Some(ub),
        }
    }
}

/// Project `x` onto the feasible box defined by `bounds`.
fn project<T: Float>(x: &mut [T], bounds: &[Bounds<T>]) {
    for (xi, b) in x.iter_mut().zip(bounds.iter()) {
        #[allow(clippy::collapsible_if)]
        if let Some(lb) = b.lower {
            if *xi < lb {
                *xi = lb;
            }
        }
        #[allow(clippy::collapsible_if)]
        if let Some(ub) = b.upper {
            if *xi > ub {
                *xi = ub;
            }
        }
    }
}

/// Minimize `f` using L-BFGS-B (bounded limited-memory BFGS).
///
/// Uses a two-loop recursion with `m=10` stored correction pairs, combined
/// with projected gradient steps for per-variable box constraints.
///
/// # Examples
///
/// ```
/// # use scivex_core::Tensor;
/// # use scivex_optim::minimize::{lbfgsb, Bounds, MinimizeOptions};
/// // Minimize f(x) = (x-5)^2 with x in [0, 3]
/// let result = lbfgsb(
///     |x: &Tensor<f64>| { let v = x.as_slice()[0] - 5.0; v * v },
///     |x: &Tensor<f64>| { Tensor::from_vec(vec![2.0 * (x.as_slice()[0] - 5.0)], vec![1]).unwrap() },
///     &Tensor::from_vec(vec![1.0], vec![1]).unwrap(),
///     &[Bounds::both(0.0, 3.0)],
///     &MinimizeOptions::default(),
/// ).unwrap();
/// assert!((result.x.as_slice()[0] - 3.0).abs() < 1e-6); // clipped to bound
/// ```
pub fn lbfgsb<T, F, G>(
    f: F,
    grad: G,
    x0: &Tensor<T>,
    bounds: &[Bounds<T>],
    options: &MinimizeOptions<T>,
) -> Result<MinimizeResult<T>>
where
    T: Float,
    F: Fn(&Tensor<T>) -> T,
    G: Fn(&Tensor<T>) -> Tensor<T>,
{
    let n = x0.numel();
    let mut state = LbfgsbState::new(&f, &grad, x0, bounds, n);

    for iter in 0..options.max_iter {
        let pg_norm = projected_gradient_norm(&state.x, &state.g, bounds);
        if pg_norm < options.gtol {
            return Ok(state.into_result(iter, true));
        }

        let d = lbfgs_direction(&state.g, &state.s_hist, &state.y_hist, &state.rho_hist, n);
        let p: Vec<T> = d.iter().map(|&v| -v).collect();

        let (alpha, fx_new, ls_evals) = projected_line_search(
            &f,
            &LineSearchInput {
                x: &state.x,
                p: &p,
                fx: state.fx,
                g: &state.g,
                bounds,
                n,
            },
        );
        state.f_evals += ls_evals;

        let (step_norm, f_change) = state.step(&grad, &p, alpha, fx_new, bounds, n);

        if step_norm < options.xtol
            || (f_change < options.ftol && f_change < options.ftol * state.fx.abs().max(T::one()))
        {
            return Ok(state.into_result(iter + 1, true));
        }
    }

    Ok(state.into_result(options.max_iter, false))
}

struct LbfgsbState<T: Float> {
    x: Vec<T>,
    x_tensor: Tensor<T>,
    fx: T,
    g: Vec<T>,
    g_tensor: Tensor<T>,
    f_evals: usize,
    g_evals: usize,
    s_hist: Vec<Vec<T>>,
    y_hist: Vec<Vec<T>>,
    rho_hist: Vec<T>,
}

impl<T: Float> LbfgsbState<T> {
    fn new<F, G>(f: &F, grad: &G, x0: &Tensor<T>, bounds: &[Bounds<T>], n: usize) -> Self
    where
        F: Fn(&Tensor<T>) -> T,
        G: Fn(&Tensor<T>) -> Tensor<T>,
    {
        let mut x = x0.as_slice().to_vec();
        project(&mut x, bounds);
        let x_tensor = Tensor::from_vec(x.clone(), vec![n]).expect("projected x0 length matches n");
        let fx = f(&x_tensor);
        let g_tensor = grad(&x_tensor);
        let g = g_tensor.as_slice().to_vec();
        Self {
            x,
            x_tensor,
            fx,
            g,
            g_tensor,
            f_evals: 1,
            g_evals: 1,
            s_hist: Vec::with_capacity(10),
            y_hist: Vec::with_capacity(10),
            rho_hist: Vec::with_capacity(10),
        }
    }

    fn step<G: Fn(&Tensor<T>) -> Tensor<T>>(
        &mut self,
        grad: &G,
        p: &[T],
        alpha: T,
        fx_new: T,
        bounds: &[Bounds<T>],
        n: usize,
    ) -> (T, T) {
        let mut x_new = vec![T::zero(); n];
        for j in 0..n {
            x_new[j] = self.x[j] + alpha * p[j];
        }
        project(&mut x_new, bounds);

        let x_new_tensor =
            Tensor::from_vec(x_new.clone(), vec![n]).expect("new x length matches n");
        let g_new_tensor = grad(&x_new_tensor);
        self.g_evals += 1;
        let g_new = g_new_tensor.as_slice().to_vec();

        let mut sy = T::zero();
        let mut s_k = vec![T::zero(); n];
        let mut y_k = vec![T::zero(); n];
        for j in 0..n {
            s_k[j] = x_new[j] - self.x[j];
            y_k[j] = g_new[j] - self.g[j];
            sy += s_k[j] * y_k[j];
        }

        let step_norm: T = s_k.iter().map(|&v| v * v).sum::<T>().sqrt();
        let f_change = (self.fx - fx_new).abs();

        if sy > T::epsilon() {
            if self.s_hist.len() == 10 {
                self.s_hist.remove(0);
                self.y_hist.remove(0);
                self.rho_hist.remove(0);
            }
            self.s_hist.push(s_k);
            self.y_hist.push(y_k);
            self.rho_hist.push(T::one() / sy);
        }

        self.x = x_new;
        self.x_tensor = x_new_tensor;
        self.fx = fx_new;
        self.g = g_new;
        self.g_tensor = g_new_tensor;

        (step_norm, f_change)
    }

    fn into_result(self, iterations: usize, converged: bool) -> MinimizeResult<T> {
        MinimizeResult {
            x: self.x_tensor,
            f_val: self.fx,
            grad: Some(self.g_tensor),
            iterations,
            f_evals: self.f_evals,
            g_evals: self.g_evals,
            converged,
        }
    }
}

/// Compute the norm of the projected gradient.
///
/// For each component, the projected gradient is zero if:
/// - x_i is at the lower bound and g_i > 0
/// - x_i is at the upper bound and g_i < 0
fn projected_gradient_norm<T: Float>(x: &[T], g: &[T], bounds: &[Bounds<T>]) -> T {
    let mut norm_sq = T::zero();
    for i in 0..x.len() {
        let gi = g[i];
        let at_lower = bounds[i]
            .lower
            .is_some_and(|lb| x[i] <= lb && gi > T::zero());
        let at_upper = bounds[i]
            .upper
            .is_some_and(|ub| x[i] >= ub && gi < T::zero());
        if !at_lower && !at_upper {
            norm_sq += gi * gi;
        }
    }
    norm_sq.sqrt()
}

/// L-BFGS two-loop recursion to approximate H_inv * g.
fn lbfgs_direction<T: Float>(
    g: &[T],
    s_history: &[Vec<T>],
    y_history: &[Vec<T>],
    rho_history: &[T],
    n: usize,
) -> Vec<T> {
    let k = s_history.len();
    if k == 0 {
        // No history: use gradient directly (steepest descent)
        return g.to_vec();
    }

    let mut q = g.to_vec();
    let mut alphas = vec![T::zero(); k];

    // First loop: backward
    for i in (0..k).rev() {
        let mut si_q = T::zero();
        for j in 0..n {
            si_q += s_history[i][j] * q[j];
        }
        alphas[i] = rho_history[i] * si_q;
        for j in 0..n {
            q[j] -= alphas[i] * y_history[i][j];
        }
    }

    // Initial Hessian approximation: gamma * I
    // gamma = s_{k-1}^T y_{k-1} / (y_{k-1}^T y_{k-1})
    let last = k - 1;
    let mut sy = T::zero();
    let mut yy = T::zero();
    for j in 0..n {
        sy += s_history[last][j] * y_history[last][j];
        yy += y_history[last][j] * y_history[last][j];
    }
    let gamma = if yy > T::epsilon() { sy / yy } else { T::one() };

    let mut r: Vec<T> = q.iter().map(|&v| gamma * v).collect();

    // Second loop: forward
    for i in 0..k {
        let mut yi_r = T::zero();
        for j in 0..n {
            yi_r += y_history[i][j] * r[j];
        }
        let beta = rho_history[i] * yi_r;
        for j in 0..n {
            r[j] += (alphas[i] - beta) * s_history[i][j];
        }
    }

    r
}

struct LineSearchInput<'a, T> {
    x: &'a [T],
    p: &'a [T],
    fx: T,
    g: &'a [T],
    bounds: &'a [Bounds<T>],
    n: usize,
}

/// Projected backtracking line search with Armijo condition.
fn projected_line_search<T, F>(f: &F, input: &LineSearchInput<'_, T>) -> (T, T, usize)
where
    T: Float,
    F: Fn(&Tensor<T>) -> T,
{
    let x = input.x;
    let p = input.p;
    let fx = input.fx;
    let g = input.g;
    let bounds = input.bounds;
    let n = input.n;

    let c1 = T::from_f64(1e-4);
    let shrink = T::from_f64(0.5);

    let dg: T = g.iter().zip(p.iter()).map(|(&gi, &pi)| gi * pi).sum();

    let mut alpha = T::one();
    let mut evals = 0usize;
    let max_ls = 20usize;

    for _ in 0..max_ls {
        let mut x_trial = vec![T::zero(); n];
        for j in 0..n {
            x_trial[j] = x[j] + alpha * p[j];
        }
        project(&mut x_trial, bounds);

        let t = Tensor::from_vec(x_trial, vec![n]).expect("trial point length matches n");
        let f_trial = f(&t);
        evals += 1;

        if f_trial <= fx + c1 * alpha * dg {
            return (alpha, f_trial, evals);
        }
        alpha *= shrink;
    }

    // Return last attempt
    let mut x_trial = vec![T::zero(); n];
    for j in 0..n {
        x_trial[j] = x[j] + alpha * p[j];
    }
    project(&mut x_trial, bounds);
    let t = Tensor::from_vec(x_trial, vec![n]).expect("trial point length matches n");
    let f_trial = f(&t);
    evals += 1;
    (alpha, f_trial, evals)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lbfgsb_unconstrained_quadratic() {
        let f = |x: &Tensor<f64>| {
            let s = x.as_slice();
            s[0] * s[0] + s[1] * s[1]
        };
        let grad = |x: &Tensor<f64>| {
            let s = x.as_slice();
            Tensor::from_vec(vec![2.0 * s[0], 2.0 * s[1]], vec![2]).unwrap()
        };

        let x0 = Tensor::from_vec(vec![5.0, 5.0], vec![2]).unwrap();
        let bounds = vec![Bounds::none(), Bounds::none()];
        let result = lbfgsb(f, grad, &x0, &bounds, &MinimizeOptions::default()).unwrap();
        assert!(result.converged);
        let s = result.x.as_slice();
        assert!(s[0].abs() < 1e-6, "x = {}", s[0]);
        assert!(s[1].abs() < 1e-6, "y = {}", s[1]);
    }

    #[test]
    fn test_lbfgsb_active_bounds() {
        // f(x, y) = (x - 3)^2 + (y - 3)^2, bounds: x >= 1, y >= 2
        // Unconstrained min at (3, 3), which is feasible.
        let f = |x: &Tensor<f64>| {
            let s = x.as_slice();
            (s[0] - 3.0).powi(2) + (s[1] - 3.0).powi(2)
        };
        let grad = |x: &Tensor<f64>| {
            let s = x.as_slice();
            Tensor::from_vec(vec![2.0 * (s[0] - 3.0), 2.0 * (s[1] - 3.0)], vec![2]).unwrap()
        };

        let x0 = Tensor::from_vec(vec![1.0, 2.0], vec![2]).unwrap();
        let bounds = vec![Bounds::lower(1.0), Bounds::lower(2.0)];
        let result = lbfgsb(f, grad, &x0, &bounds, &MinimizeOptions::default()).unwrap();
        assert!(result.converged);
        let s = result.x.as_slice();
        assert!((s[0] - 3.0).abs() < 1e-4, "x = {}", s[0]);
        assert!((s[1] - 3.0).abs() < 1e-4, "y = {}", s[1]);
    }

    #[test]
    fn test_lbfgsb_active_bound_at_solution() {
        // f(x) = (x - (-1))^2, bound: x >= 0
        // Unconstrained min at -1, but bound forces solution at 0.
        let f = |x: &Tensor<f64>| {
            let s = x.as_slice();
            (s[0] + 1.0).powi(2)
        };
        let grad = |x: &Tensor<f64>| {
            let s = x.as_slice();
            Tensor::from_vec(vec![2.0 * (s[0] + 1.0)], vec![1]).unwrap()
        };

        let x0 = Tensor::from_vec(vec![5.0], vec![1]).unwrap();
        let bounds = vec![Bounds::lower(0.0)];
        let result = lbfgsb(f, grad, &x0, &bounds, &MinimizeOptions::default()).unwrap();
        assert!(result.converged);
        let s = result.x.as_slice();
        assert!(s[0].abs() < 1e-6, "x = {}, expected 0.0", s[0]);
    }

    #[test]
    fn test_lbfgsb_all_bounded() {
        // f(x, y) = x^2 + y^2, bounds: 1 <= x <= 2, 1 <= y <= 2
        // Min at (1, 1) since origin is outside bounds.
        let f = |x: &Tensor<f64>| {
            let s = x.as_slice();
            s[0] * s[0] + s[1] * s[1]
        };
        let grad = |x: &Tensor<f64>| {
            let s = x.as_slice();
            Tensor::from_vec(vec![2.0 * s[0], 2.0 * s[1]], vec![2]).unwrap()
        };

        let x0 = Tensor::from_vec(vec![1.5, 1.5], vec![2]).unwrap();
        let bounds = vec![Bounds::both(1.0, 2.0), Bounds::both(1.0, 2.0)];
        let result = lbfgsb(f, grad, &x0, &bounds, &MinimizeOptions::default()).unwrap();
        assert!(result.converged);
        let s = result.x.as_slice();
        assert!((s[0] - 1.0).abs() < 1e-4, "x = {}", s[0]);
        assert!((s[1] - 1.0).abs() < 1e-4, "y = {}", s[1]);
    }
}
