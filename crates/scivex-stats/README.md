# scivex-stats

Statistical computing for Scivex. Probability distributions, hypothesis tests,
correlation, regression, and descriptive statistics.

## Highlights

- **10 distributions** — Normal, Uniform, Exponential, Beta, Gamma, Chi-Squared, Student-t, Bernoulli, Binomial, Poisson
- **Distribution trait** — `pdf()`, `cdf()`, `ppf()`, `sample()`, `mean()`, `variance()` for all
- **Hypothesis tests** — One/two-sample t-test, chi-square, KS, Mann-Whitney, one-way ANOVA
- **Correlation** — Pearson, Spearman, Kendall + correlation matrices
- **OLS regression** — Full summary: coefficients, std errors, t-stats, p-values, R², F-stat
- **Descriptive** — mean, median, std, variance, quantiles, skewness, kurtosis, describe()

## Usage

```rust
use scivex_stats::prelude::*;

// Descriptive statistics
let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
let summary = describe(&data);

// Distributions
let normal = Normal::new(0.0, 1.0);
let p = normal.cdf(1.96);       // ≈ 0.975
let x = normal.ppf(0.975);      // ≈ 1.96
let samples = normal.sample_n(&mut rng, 1000);

// Hypothesis testing
let result = t_test_two_sample(&group_a, &group_b).unwrap();

// Regression
let ols = ols(&features, &target).unwrap();
println!("R² = {}", ols.r_squared);
```

## License

MIT
