//! `scivex-stats` — Statistical distributions, descriptive statistics,
//! hypothesis tests, correlation, regression, and confidence intervals.
//!
//! Built on top of [`scivex_core`] with zero external dependencies for the
//! math itself. All functions are generic over [`Float`](scivex_core::Float).

/// Confidence interval construction (mean, proportion).
pub mod confidence;
/// Multiple comparison corrections (Bonferroni, Benjamini-Hochberg).
pub mod correction;
/// Correlation coefficients (Pearson, Spearman, Kendall).
pub mod correlation;
/// Descriptive statistics (mean, variance, median, quantiles, skewness, kurtosis).
pub mod descriptive;
/// Probability distributions (Normal, Uniform, Exponential, etc.).
pub mod distributions;
/// Statistics error types.
pub mod error;
/// Hypothesis tests (t-test, chi-square, ANOVA, Mann-Whitney, KS).
pub mod hypothesis;
/// Ordinary least-squares regression.
pub mod regression;
pub(crate) mod special;
/// Time series analysis (ACF, PACF, ARIMA, exponential smoothing, seasonal decomposition).
pub mod timeseries;

pub use confidence::{ConfidenceInterval, ci_mean, ci_mean_z, ci_proportion};
pub use correction::{benjamini_hochberg, bonferroni};
pub use correlation::{CorrelationMethod, corr_matrix, kendall, pearson, spearman};
pub use descriptive::{
    DescribeResult, describe, kurtosis, mean, median, quantile, skewness, std_dev,
    std_dev_with_ddof, variance, variance_with_ddof,
};
pub use distributions::Distribution;
pub use error::{Result, StatsError};
pub use hypothesis::{
    TestResult, anova_oneway, chi_square_test, ks_test_two_sample, mann_whitney_u,
    t_test_one_sample, t_test_two_sample,
};
pub use regression::{OlsResult, ols};
pub use timeseries::{
    AdfResult, Arima, DecomposeResult, ExponentialSmoothing, SmoothingMethod, acf, adf_test, pacf,
    seasonal_decompose,
};

/// Items intended for glob-import: `use scivex_stats::prelude::*;`
pub mod prelude {
    pub use crate::confidence::{ConfidenceInterval, ci_mean, ci_mean_z, ci_proportion};
    pub use crate::correction::{benjamini_hochberg, bonferroni};
    pub use crate::correlation::{CorrelationMethod, corr_matrix, kendall, pearson, spearman};
    pub use crate::descriptive::{
        DescribeResult, describe, kurtosis, mean, median, quantile, skewness, std_dev, variance,
    };
    pub use crate::distributions::{
        Bernoulli, Beta, Binomial, Cauchy, ChiSquared, Distribution, Exponential, Gamma,
        Hypergeometric, Laplace, LogNormal, NegativeBinomial, Normal, Pareto, Poisson, StudentT,
        Uniform, Weibull,
    };
    pub use crate::error::{Result, StatsError};
    pub use crate::hypothesis::{
        TestResult, anova_oneway, chi_square_test, ks_test_two_sample, mann_whitney_u,
        t_test_one_sample, t_test_two_sample,
    };
    pub use crate::regression::{OlsResult, ols};
}
