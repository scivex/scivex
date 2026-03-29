# Migrating from SciPy to Scivex

This guide maps SciPy's Python API to the equivalent Scivex Rust API.
Every function listed here exists in the Scivex source code. If a SciPy
feature is not listed, it has no Scivex equivalent yet.

---

## Quick Reference Table

### scipy.optimize

| SciPy | Scivex (`scivex_optim`) | Notes |
|---|---|---|
| `scipy.optimize.minimize(f, x0, method='BFGS')` | `bfgs(f, grad, &x0, &opts)` | Gradient required (use `numerical_gradient` if needed) |
| `scipy.optimize.minimize(f, x0, method='Nelder-Mead')` | `nelder_mead(f, &x0, &opts)` | Gradient-free |
| `scipy.optimize.minimize(f, x0, method='L-BFGS-B')` | `lbfgsb(f, grad, &x0, &bounds, &opts)` | Bounded optimization via `Bounds` |
| Gradient descent (no direct SciPy equivalent) | `gradient_descent(f, grad, &x0, &opts)` | Simple first-order method |
| `scipy.optimize.minimize_scalar(method='golden')` | `golden_section(f, a, b, &opts)` | 1-D golden section search |
| `scipy.optimize.minimize_scalar(method='brent')` | `brent_min(f, a, b, &opts)` | 1-D Brent minimization |
| `scipy.optimize.curve_fit(f, xdata, ydata, p0)` | `curve_fit(model, &x_data, &y_data, &p0)` | Levenberg-Marquardt under the hood |
| `scipy.optimize.least_squares` | `levenberg_marquardt(model, &x_data, &y_data, &p0, max_iter, tol)` | Full control over iterations and tolerance |
| `scipy.optimize.linprog(c, A_ub, b_ub)` | `linprog(&c, &a_ub, &b_ub)` | Revised simplex method, `x >= 0` implied |
| `scipy.optimize.brentq(f, a, b)` | `brent_root(f, a, b, &opts)` | Scalar root finding |
| `scipy.optimize.bisect(f, a, b)` | `bisection(f, a, b, &opts)` | Bracketing root finder |
| `scipy.optimize.newton(f, x0)` | `newton(f, df, x0, &opts)` | Newton-Raphson (derivative required) |
| (no direct equivalent) | `quadprog(&h, &c, &a_ub, &b_ub)` | Quadratic programming via active set |
| `scipy.optimize.numerical_gradient` | `numerical_gradient(&f, &x)` | Central finite differences |

### scipy.integrate

| SciPy | Scivex (`scivex_optim`) | Notes |
|---|---|---|
| `scipy.integrate.quad(f, a, b)` | `quad(f, a, b, &opts)` | Adaptive Gauss-Kronrod quadrature |
| `scipy.integrate.trapezoid(y, x)` | `trapezoid(f, a, b, n)` | Composite trapezoidal rule |
| `scipy.integrate.simpson(y, x)` | `simpson(f, a, b, n)` | Composite Simpson's rule |
| `scipy.integrate.solve_ivp(f, t_span, y0, method='RK45')` | `solve_ivp(f, t_span, &y0, OdeMethod::RK45, &opts)` | Dormand-Prince adaptive |
| `scipy.integrate.solve_ivp(..., method='BDF')` | `solve_ivp(f, t_span, &y0, OdeMethod::BDF2, &opts)` | Stiff system solver |
| (Euler, no SciPy equivalent) | `solve_ivp(f, t_span, &y0, OdeMethod::Euler, &opts)` | First-order fixed step |
| `solve_ivp(..., events=...)` | `OdeOptions { event_fn: Some(Box::new(\|t, y\| ...)), .. }` | Event detection via zero-crossing |

### scipy.interpolate

| SciPy | Scivex (`scivex_optim::interpolate`) | Notes |
|---|---|---|
| `scipy.interpolate.interp1d(x, y, kind='linear')` | `interp1d(&xs, &ys, x_query, Interp1dMethod::Linear, Extrapolate::Error)` | Also available: `Linear1d::new()` |
| `scipy.interpolate.CubicSpline(x, y)` | `CubicSpline::new(&xs, &ys, SplineBoundary::Natural, Extrapolate::Error)` | Supports Natural and Clamped BCs |
| `scipy.interpolate.BSpline` | `BSpline::new(&xs, &ys, Extrapolate::Error)` | Uniform degree-3 B-spline |
| `scipy.interpolate.interp2d` (bilinear) | `interp2d(&xs, &ys, &zs, x, y, Interp2dMethod::Bilinear, Extrapolate::Error)` | Rectilinear grid |
| `scipy.interpolate.interp2d` (bicubic) | `Bicubic2d::new(...)` or `interp2d(..., Interp2dMethod::Bicubic, ...)` | Rectilinear grid |

### scipy.stats (Distributions)

| SciPy | Scivex (`scivex_stats::distributions`) | Notes |
|---|---|---|
| `scipy.stats.norm` | `Normal::new(mu, sigma)` | `Normal::standard()` for N(0,1) |
| `scipy.stats.uniform` | `Uniform::new(low, high)` | |
| `scipy.stats.expon` | `Exponential::new(lambda)` | |
| `scipy.stats.gamma` | `Gamma::new(shape, scale)` | |
| `scipy.stats.beta` | `Beta::new(alpha, beta)` | |
| `scipy.stats.t` | `StudentT::new(df)` | |
| `scipy.stats.chi2` | `ChiSquared::new(df)` | |
| `scipy.stats.poisson` | `Poisson::new(lambda)` | |
| `scipy.stats.binom` | `Binomial::new(n, p)` | |
| `scipy.stats.bernoulli` | `Bernoulli::new(p)` | |
| `scipy.stats.lognorm` | `LogNormal::new(mu, sigma)` | |
| `scipy.stats.cauchy` | `Cauchy::new(x0, gamma)` | |
| `scipy.stats.laplace` | `Laplace::new(mu, b)` | |
| `scipy.stats.pareto` | `Pareto::new(x_m, alpha)` | |
| `scipy.stats.weibull_min` | `Weibull::new(shape, scale)` | |
| `scipy.stats.hypergeom` | `Hypergeometric::new(N, K, n)` | |
| `scipy.stats.nbinom` | `NegativeBinomial::new(r, p)` | |
| `dist.pdf(x)` | `dist.pdf(x)` | All distributions implement the `Distribution` trait |
| `dist.cdf(x)` | `dist.cdf(x)` | |
| `dist.ppf(p)` | `dist.ppf(p)` | Returns `Result<T>` |
| `dist.rvs(size=n)` | `dist.sample_n(&mut rng, n)` | Requires an explicit `Rng` |
| `dist.mean()` | `dist.mean()` | Theoretical mean |
| `dist.var()` | `dist.variance()` | Theoretical variance |

### scipy.stats (Hypothesis Tests)

| SciPy | Scivex (`scivex_stats::hypothesis`) | Notes |
|---|---|---|
| `scipy.stats.ttest_1samp(data, mu)` | `t_test_one_sample(&data, mu_0)` | Returns `TestResult { statistic, p_value, df }` |
| `scipy.stats.ttest_ind(x, y)` | `t_test_two_sample(&x, &y)` | Welch's t-test (unequal variances) |
| `scipy.stats.chisquare(observed, expected)` | `chi_square_test(&observed, &expected)` | Goodness-of-fit |
| `scipy.stats.ks_2samp(x, y)` | `ks_test_two_sample(&x, &y)` | Two-sample Kolmogorov-Smirnov |
| `scipy.stats.mannwhitneyu(x, y)` | `mann_whitney_u(&x, &y)` | Non-parametric, normal approximation |
| `scipy.stats.f_oneway(*groups)` | `anova_oneway(&[&g1, &g2, &g3])` | One-way ANOVA |

### scipy.stats (Descriptive & Correlation)

| SciPy | Scivex (`scivex_stats`) | Notes |
|---|---|---|
| `numpy.mean(data)` | `mean(&data)` | Returns `Result<T>` |
| `numpy.var(data)` | `variance(&data)` | Bessel-corrected (ddof=1) |
| `numpy.std(data)` | `std_dev(&data)` | |
| `numpy.median(data)` | `median(&data)` | |
| `numpy.quantile(data, q)` | `quantile(&data, q)` | |
| `scipy.stats.describe(data)` | `describe(&data)` | Returns `DescribeResult` |
| `scipy.stats.skew(data)` | `skewness(&data)` | |
| `scipy.stats.kurtosis(data)` | `kurtosis(&data)` | |
| `scipy.stats.pearsonr(x, y)` | `pearson(&x, &y)` | |
| `scipy.stats.spearmanr(x, y)` | `spearman(&x, &y)` | |
| `scipy.stats.kendalltau(x, y)` | `kendall(&x, &y)` | |
| Correlation matrix | `corr_matrix(&data, CorrelationMethod::Pearson)` | |

### scipy.signal

| SciPy | Scivex (`scivex_signal`) | Notes |
|---|---|---|
| `scipy.signal.lfilter(b, a, x)` | `filter::lfilter(&b, &a, &x)` | Direct Form II transposed |
| `scipy.signal.filtfilt(b, a, x)` | `filter::filtfilt(&b, &a, &x)` | Zero-phase forward-backward |
| `scipy.signal.firwin(num_taps, cutoff)` | `filter::FirFilter::low_pass(cutoff, num_taps)` | Hamming-windowed sinc |
| (high-pass FIR) | `filter::FirFilter::high_pass(cutoff, num_taps)` | Spectral inversion of LP |
| (band-pass FIR) | `filter::FirFilter::band_pass(low, high, num_taps)` | LP(high) - LP(low) |
| `scipy.signal.stft(x, nperseg, noverlap)` | `spectral::stft(&x, window_size, hop_size, None)` | Returns complex tensor `[frames, bins, 2]` |
| `scipy.signal.istft(Zxx)` | `spectral::istft(&stft, window_size, hop_size, None)` | Overlap-add reconstruction |
| `scipy.signal.spectrogram(x)` | `spectral::spectrogram(&x, window_size, hop_size)` | Power spectrogram `\|STFT\|^2` |
| `scipy.signal.periodogram(x)` | `spectral::periodogram(&x)` | Returns `(frequencies, psd)` |
| `scipy.signal.welch(x)` | `spectral::welch(&x, ...)` | Welch PSD estimate |
| `scipy.signal.find_peaks(x)` | `peak::find_peaks(&x, min_height, min_distance)` | Returns peak indices |
| `scipy.signal.resample(x, num)` | `resample::resample(&x, num_samples)` | FFT-based resampling |
| `scipy.signal.decimate(x, q)` | `resample::decimate(&x, factor)` | Downsample with anti-alias filter |
| `scipy.signal.hann(N)` | `window::hann(n)` | Returns 1-D `Tensor` |
| `scipy.signal.hamming(N)` | `window::hamming(n)` | |
| `scipy.signal.blackman(N)` | `window::blackman(n)` | |
| `pywt.dwt(x, 'haar')` | `wavelet::dwt(&x, Wavelet::Haar)` | Returns `(approx, detail)` |
| `pywt.idwt(cA, cD, 'haar')` | `wavelet::idwt(&approx, &detail, Wavelet::Haar)` | Inverse DWT |

---

## Code Examples

### Minimization (scipy.optimize.minimize)

**Python (SciPy)**

```python
from scipy.optimize import minimize

def rosenbrock(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

result = minimize(rosenbrock, x0=[0.0, 0.0], method='BFGS')
print(result.x)    # [1.0, 1.0]
print(result.fun)   # ~0.0
```

**Rust (Scivex)**

```rust
use scivex_core::Tensor;
use scivex_optim::{MinimizeOptions, bfgs, numerical_gradient};

let f = |x: &Tensor<f64>| {
    let s = x.as_slice();
    (1.0 - s[0]).powi(2) + 100.0 * (s[1] - s[0] * s[0]).powi(2)
};
let grad = |x: &Tensor<f64>| numerical_gradient(&f, x);
let x0 = Tensor::from_vec(vec![0.0, 0.0], vec![2]).unwrap();
let opts = MinimizeOptions::default();

let result = bfgs(&f, &grad, &x0, &opts).unwrap();
assert!(result.converged);
// result.x  -- the minimizer
// result.f_val -- function value at minimum
```

### Curve Fitting (scipy.optimize.curve_fit)

**Python (SciPy)**

```python
from scipy.optimize import curve_fit
import numpy as np

def model(x, a, b):
    return a * np.exp(b * x)

xdata = np.linspace(0, 4, 50)
ydata = 2.5 * np.exp(1.3 * xdata)
popt, _ = curve_fit(model, xdata, ydata, p0=[1.0, 1.0])
```

**Rust (Scivex)**

```rust
use scivex_optim::curve_fit;

let model = |x: f64, p: &[f64]| p[0] * (p[1] * x).exp();
let x_data: Vec<f64> = (0..50).map(|i| i as f64 * 0.08).collect();
let y_data: Vec<f64> = x_data.iter().map(|&x| 2.5 * (1.3 * x).exp()).collect();

let result = curve_fit(model, &x_data, &y_data, &[1.0, 1.0]).unwrap();
assert!(result.converged);
// result.params  -- [a, b]
// result.cost    -- sum of squared residuals
```

### Numerical Integration (scipy.integrate.quad)

**Python (SciPy)**

```python
from scipy.integrate import quad

result, error = quad(lambda x: x**2, 0, 1)
# result = 0.3333...
```

**Rust (Scivex)**

```rust
use scivex_optim::{QuadOptions, quad};

let result = quad(|x: f64| x * x, 0.0, 1.0, &QuadOptions::default()).unwrap();
// result.value          -- 0.3333...
// result.error_estimate -- estimated absolute error
// result.n_evals        -- number of function evaluations
```

### ODE Solving (scipy.integrate.solve_ivp)

**Python (SciPy)**

```python
from scipy.integrate import solve_ivp

# dy/dt = -y, y(0) = 1
sol = solve_ivp(lambda t, y: [-y[0]], [0, 5], [1.0], method='RK45')
print(sol.t[-1], sol.y[0, -1])
```

**Rust (Scivex)**

```rust
use scivex_optim::ode::{solve_ivp, OdeMethod, OdeOptions};

let result = solve_ivp(
    |_t: f64, y: &[f64]| vec![-y[0]],
    [0.0, 5.0],
    &[1.0],
    OdeMethod::RK45,
    &OdeOptions::default(),
).unwrap();

let t_final = *result.t.last().unwrap();
let y_final = result.y.last().unwrap()[0];
// y_final is approximately e^(-5)
```

### Linear Programming (scipy.optimize.linprog)

**Python (SciPy)**

```python
from scipy.optimize import linprog

# minimize c^T x subject to A_ub x <= b_ub, x >= 0
c = [-1, -2]
A_ub = [[1, 1], [2, 1]]
b_ub = [4, 6]
res = linprog(c, A_ub=A_ub, b_ub=b_ub)
```

**Rust (Scivex)**

```rust
use scivex_optim::linprog;

let c = vec![-1.0, -2.0];
let a_ub = vec![vec![1.0, 1.0], vec![2.0, 1.0]];
let b_ub = vec![4.0, 6.0];

let result = linprog(&c, &a_ub, &b_ub).unwrap();
// result.x    -- optimal decision variables
// result.fun  -- optimal objective value
```

### Distributions (scipy.stats.norm)

**Python (SciPy)**

```python
from scipy.stats import norm

d = norm(loc=0, scale=1)
d.pdf(0.0)        # 0.3989...
d.cdf(1.96)       # 0.975...
d.ppf(0.975)      # 1.96...
d.rvs(size=1000)  # 1000 random samples
```

**Rust (Scivex)**

```rust
use scivex_core::random::Rng;
use scivex_stats::distributions::{Normal, Distribution};

let d = Normal::standard();
let pdf_val = d.pdf(0.0);       // 0.3989...
let cdf_val = d.cdf(1.96);     // 0.975...
let x = d.ppf(0.975).unwrap(); // 1.96...

let mut rng = Rng::with_seed(42);
let samples = d.sample_n(&mut rng, 1000);
```

### Hypothesis Tests (scipy.stats.ttest_ind)

**Python (SciPy)**

```python
from scipy.stats import ttest_ind

x = [10.0, 10.5, 9.8, 10.2, 10.1]
y = [5.0, 5.2, 4.9, 5.1, 5.3]
stat, p_value = ttest_ind(x, y)
```

**Rust (Scivex)**

```rust
use scivex_stats::t_test_two_sample;

let x = [10.0, 10.5, 9.8, 10.2, 10.1];
let y = [5.0, 5.2, 4.9, 5.1, 5.3];

let result = t_test_two_sample(&x, &y).unwrap();
// result.statistic -- t-statistic
// result.p_value   -- two-tailed p-value
// result.df        -- Welch-Satterthwaite degrees of freedom
```

### Signal Filtering (scipy.signal.lfilter)

**Python (SciPy)**

```python
from scipy.signal import lfilter

b = [0.5, 0.5]  # 2-point moving average
a = [1.0]
x = [1.0, 3.0, 5.0, 7.0]
y = lfilter(b, a, x)
```

**Rust (Scivex)**

```rust
use scivex_core::Tensor;
use scivex_signal::filter::lfilter;

let b = Tensor::from_vec(vec![0.5, 0.5], vec![2]).unwrap();
let a = Tensor::from_vec(vec![1.0], vec![1]).unwrap();
let x = Tensor::from_vec(vec![1.0, 3.0, 5.0, 7.0], vec![4]).unwrap();

let y = lfilter(&b, &a, &x).unwrap();
```

### FIR Filter Design (scipy.signal.firwin)

**Python (SciPy)**

```python
from scipy.signal import firwin, lfilter

# Low-pass FIR, 31 taps, cutoff at 0.3 * Nyquist
b = firwin(31, 0.3)
y = lfilter(b, [1.0], x)
```

**Rust (Scivex)**

```rust
use scivex_core::Tensor;
use scivex_signal::filter::{FirFilter, lfilter};

let b = FirFilter::low_pass::<f64>(0.3, 31).unwrap();
let a = Tensor::from_vec(vec![1.0], vec![1]).unwrap();
// let y = lfilter(&b, &a, &x).unwrap();
```

### STFT / Spectrogram (scipy.signal.stft)

**Python (SciPy)**

```python
from scipy.signal import stft, spectrogram

f, t, Zxx = stft(x, nperseg=256, noverlap=128)
f, t, Sxx = spectrogram(x, nperseg=256, noverlap=128)
```

**Rust (Scivex)**

```rust
use scivex_signal::spectral::{stft, spectrogram};

let s = stft(&x, 256, 128, None).unwrap();
// s has shape [num_frames, freq_bins, 2]  (real, imag)

let power = spectrogram(&x, 256, 128).unwrap();
// power has shape [num_frames, freq_bins]
```

---

## Key Differences

### 1. Error Handling: Exceptions vs. Results

SciPy raises Python exceptions. Scivex returns `Result<T, Error>`.

```rust
// Every fallible call must be unwrapped or propagated
let result = bfgs(&f, &grad, &x0, &opts)?;  // propagate with ?
let result = bfgs(&f, &grad, &x0, &opts).unwrap();  // panic on error
```

Error types are crate-specific: `OptimError`, `StatsError`, `SignalError`. All
provide variants like `DimensionMismatch`, `NotConverged`, `InvalidParameter`,
and `EmptyInput`.

### 2. Closures vs. Callables

SciPy accepts any Python callable. Scivex uses Rust closures or function
pointers, which must satisfy trait bounds like `Fn(&Tensor<T>) -> T`.

```rust
// Closure captures are fine
let a = 2.0;
let f = |x: &Tensor<f64>| {
    let s = x.as_slice();
    a * s[0] * s[0]  // captures `a` from the environment
};
```

### 3. Explicit Randomness

SciPy's `dist.rvs()` uses a global random state. Scivex requires an
explicit `Rng` instance:

```rust
use scivex_core::random::Rng;
let mut rng = Rng::with_seed(42);
let samples = dist.sample_n(&mut rng, 100);
```

### 4. Tensors Instead of NumPy Arrays

Where SciPy uses NumPy arrays, Scivex uses `Tensor<T>` from `scivex-core`.
Filter coefficients, signal data, and optimization vectors are all `Tensor`
values. Conversion from slices and `Vec` is straightforward:

```rust
use scivex_core::Tensor;
let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
let s = t.as_slice();  // &[f64]
```

### 5. Generic Over Floating-Point Type

All Scivex functions are generic over `Float`. You choose `f32` or `f64`
at the call site:

```rust
let result_f32 = quad(|x: f32| x * x, 0.0_f32, 1.0_f32, &QuadOptions::default());
let result_f64 = quad(|x: f64| x * x, 0.0_f64, 1.0_f64, &QuadOptions::default());
```

### 6. Feature Flags

The umbrella `scivex` crate uses feature flags. To use only optimization
and statistics without pulling in neural networks or visualization:

```toml
[dependencies]
scivex = { version = "0.1", features = ["optim", "stats", "signal"] }
```

Or depend on sub-crates directly:

```toml
[dependencies]
scivex-optim = { version = "0.1" }
scivex-stats = { version = "0.1" }
scivex-signal = { version = "0.1" }
```

### 7. Struct Results vs. Tuples

SciPy functions often return plain tuples. Scivex returns named structs
with descriptive fields:

```rust
// scivex_optim::MinimizeResult
result.x          // Tensor<T> -- the minimizer
result.f_val      // T -- function value at minimum
result.grad       // Option<Tensor<T>>
result.iterations // usize
result.f_evals    // usize
result.converged  // bool

// scivex_stats::TestResult
result.statistic  // T
result.p_value    // T
result.df         // Option<T>
```

### 8. Options Structs with Defaults

Instead of keyword arguments, Scivex uses options structs with `Default`
implementations:

```rust
// Use all defaults
let opts = MinimizeOptions::default();

// Override specific fields
let opts = MinimizeOptions {
    max_iter: 5000,
    gtol: 1e-10,
    ..MinimizeOptions::default()
};
```

---

## Additional Scivex Capabilities Without SciPy Equivalents

These features are available in Scivex but have no direct SciPy counterpart
(or live in separate Python packages):

| Scivex | Description |
|---|---|
| `scivex_optim::quadprog(...)` | Quadratic programming (active set method) |
| `scivex_optim::pde::heat_equation_1d(...)` | 1-D heat equation solver (finite difference) |
| `scivex_optim::pde::wave_equation_1d(...)` | 1-D wave equation solver |
| `scivex_optim::pde::laplace_2d(...)` | 2-D Laplace equation solver |
| `scivex_stats::bayesian::MetropolisHastings` | MCMC sampling |
| `scivex_stats::bayesian::HamiltonianMC` | Hamiltonian Monte Carlo |
| `scivex_stats::survival::kaplan_meier(...)` | Kaplan-Meier estimator |
| `scivex_stats::survival::cox_ph(...)` | Cox proportional hazards |
| `scivex_stats::glm::glm(...)` | Generalized linear models |
| `scivex_stats::kalman::KalmanFilter` | Kalman filter |
| `scivex_stats::timeseries::Arima` | ARIMA models |
| `scivex_stats::garch::Garch` | GARCH volatility models |
| `scivex_stats::var::VarModel` | Vector autoregression |
| `scivex_stats::effect_size::cohens_d(...)` | Effect size measures |
| `scivex_signal::features::mfcc(...)` | MFCC audio features |
| `scivex_signal::features::mel_spectrogram(...)` | Mel spectrogram |
| `scivex_signal::features::chroma_stft(...)` | Chroma features |
| `scivex_signal::features::pitch_yin(...)` | Pitch detection (YIN) |
| `scivex_core::einsum::einsum(...)` | Einstein summation notation |
| `scivex_core::named_tensor::NamedTensor` | Named/labeled tensor dimensions (xarray-style) |
| `scivex_core::spatial::KdTree` | KD-tree for nearest neighbor queries |
| `scivex_core::numexpr::NumExpr` | NumExpr-style expression JIT |
| `scivex_stats::bayesian::nuts(...)` | No-U-Turn Sampler (NUTS) |
| `scivex_stats::prophet::Prophet` | Prophet-style time series forecasting |
| `scivex_stats::ts_features::extract_features(...)` | Automated time series feature extraction |
| `scivex_stats::anomaly::ZScoreDetector` | Time series anomaly detection |
| `scivex_stats::lmm::LinearMixedModel` | Linear mixed-effects models (REML) |
| `scivex_optim::linprog::linprog(...)` | Linear programming (simplex method) |
| `scivex_optim::minimize::nelder_mead(...)` | Nelder-Mead derivative-free optimization |
| `scivex_optim::minimize::lbfgsb(...)` | L-BFGS-B with box constraints |
