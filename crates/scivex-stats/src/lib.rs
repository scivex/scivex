//! `scivex-stats` — Statistical distributions, descriptive statistics,
//! hypothesis tests, correlation, and regression.
//!
//! Built on top of [`scivex_core`] with zero external dependencies for the
//! math itself. All functions are generic over [`Float`](scivex_core::Float).

pub mod correlation;
pub mod descriptive;
pub mod distributions;
pub mod error;
pub mod hypothesis;
pub mod regression;
pub(crate) mod special;

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

/// Items intended for glob-import: `use scivex_stats::prelude::*;`
pub mod prelude {
    pub use crate::correlation::{CorrelationMethod, corr_matrix, kendall, pearson, spearman};
    pub use crate::descriptive::{
        DescribeResult, describe, kurtosis, mean, median, quantile, skewness, std_dev, variance,
    };
    pub use crate::distributions::{
        Bernoulli, Beta, Binomial, ChiSquared, Distribution, Exponential, Gamma, Normal, Poisson,
        StudentT, Uniform,
    };
    pub use crate::error::{Result, StatsError};
    pub use crate::hypothesis::{
        TestResult, anova_oneway, chi_square_test, ks_test_two_sample, mann_whitney_u,
        t_test_one_sample, t_test_two_sample,
    };
    pub use crate::regression::{OlsResult, ols};
}
