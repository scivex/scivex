# scivex-stats

Statistical computing for Scivex. Distributions, hypothesis testing,
correlation, regression, and time series analysis.

## Highlights

- **15+ distributions** — Normal, Uniform, Exponential, Poisson, Binomial, Chi-Squared, Student's t, F, Beta, Gamma, Geometric, Bernoulli, Hypergeometric, NegativeBinomial, Pareto
- **Hypothesis tests** — t-test (one-sample, two-sample, paired), chi-squared, ANOVA, Kolmogorov-Smirnov, Mann-Whitney U, Wilcoxon signed-rank
- **Correlation** — Pearson, Spearman, Kendall tau, point-biserial
- **Regression** — OLS, weighted, polynomial, multiple regression
- **Time series** — ARIMA, SARIMAX, exponential smoothing, GARCH, VAR, Kalman filter
- **Effect sizes** — Cohen's d, Cramer's V, Cohen's w
- **Feature extraction** — Statistical feature extraction from time series
- **Bayesian** — Bayesian linear regression

## Usage

```rust
use scivex_stats::prelude::*;

let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];

let m = mean(&data);
let s = std_dev(&data);

// Distributions
let normal = Normal::new(0.0, 1.0);
let p = normal.cdf(1.96);

// Hypothesis testing
let result = t_test_one_sample(&data, 4.0).unwrap();
```

## License

MIT
