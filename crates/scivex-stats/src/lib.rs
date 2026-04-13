#![allow(
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::doc_markdown,
    clippy::module_name_repetitions
)]
//! `scivex-stats` — Statistical distributions, descriptive statistics,
//! hypothesis tests, correlation, regression, and confidence intervals.
//!
//! Built on top of [`scivex_core`] with zero external dependencies for the
//! math itself. All functions are generic over [`Float`](scivex_core::Float).
//!
//! # Examples
//!
//! ```
//! use scivex_stats::{mean, variance, std_dev, median};
//!
//! let data = [2.0_f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
//! let m = mean(&data).unwrap();
//! let v = variance(&data).unwrap();
//! let s = std_dev(&data).unwrap();
//! let med = median(&data).unwrap();
//!
//! assert!((m - 5.0).abs() < 1e-10);
//! assert!((s - v.sqrt()).abs() < 1e-10);
//! assert!((med - 4.5).abs() < 1e-10);
//! ```

/// Bayesian inference: MCMC samplers and convergence diagnostics.
pub mod bayesian;
/// Bayesian optimization with Gaussian process surrogates.
pub mod bayesian_optim;
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
/// Effect size measures (Cohen's d, Hedges' g, η², ω², Cramér's V, etc.).
pub mod effect_size;
/// Statistics error types.
pub mod error;
/// GARCH volatility models.
pub mod garch;
/// Generalized Linear Models.
pub mod glm;
/// Hypothesis tests (t-test, chi-square, ANOVA, Mann-Whitney, KS).
pub mod hypothesis;
/// Kalman filter for linear state estimation.
pub mod kalman;
/// Linear Mixed-Effects Models (LMM).
pub mod mixed_effects;
/// Online (streaming) statistics using Welford's algorithm.
pub mod online;
/// Prophet-style additive decomposable time series forecasting.
pub mod prophet;
/// Ordinary least-squares regression.
pub mod regression;
pub(crate) mod special;
/// Survival analysis (Kaplan-Meier, log-rank test, Cox PH).
pub mod survival;
/// Time series analysis (ACF, PACF, ARIMA, exponential smoothing, seasonal decomposition).
pub mod timeseries;
/// Time series anomaly detection (z-score, seasonal, isolation forest, EWMA).
pub mod ts_anomaly;
/// Automated time series feature extraction for ML feature engineering.
pub mod ts_features;
/// Vector Autoregression (VAR) models for multivariate time series.
pub mod var;

pub use bayesian_optim::{
    AcquisitionFunction, BayesOptConfig, BayesOptResult, BayesianOptimizer, Kernel,
};
pub use confidence::{ConfidenceInterval, ci_mean, ci_mean_z, ci_proportion};
pub use correction::{benjamini_hochberg, bonferroni};
pub use correlation::{CorrelationMethod, corr_matrix, kendall, pearson, spearman};
pub use descriptive::{
    DescribeResult, describe, kurtosis, mean, median, quantile, skewness, std_dev,
    std_dev_with_ddof, variance, variance_with_ddof,
};
pub use distributions::Distribution;
pub use effect_size::{
    EffectSizeInterpretation, cohens_d, cohens_w, cramers_v, eta_squared, glass_delta, hedges_g,
    interpret_cohens_d, omega_squared, point_biserial,
};
pub use error::{Result, StatsError};
pub use garch::Garch;
pub use glm::{Family, GlmResult, LinkFunction, glm};
pub use hypothesis::{
    TestResult, anova_oneway, batch_t_test_one_sample, batch_t_test_two_sample, chi_square_test,
    ks_test_two_sample, mann_whitney_u, t_test_one_sample, t_test_two_sample,
};
pub use kalman::KalmanFilter;
pub use mixed_effects::{LmmResult, lmm};
pub use online::OnlineStats;
pub use prophet::{Prophet, ProphetConfig, ProphetForecast};
pub use regression::{OlsResult, ols};
pub use survival::{
    CoxPHResult, KaplanMeierEstimate, LogRankResult, SurvivalRecord, cox_ph, kaplan_meier,
    log_rank_test, median_survival_time,
};
pub use timeseries::{
    AdfResult, Arima, DecomposeResult, ExponentialSmoothing, Sarimax, SmoothingMethod, acf,
    adf_test, pacf, seasonal_decompose,
};
pub use ts_anomaly::{
    AnomalyResult, ewma_anomaly, isolation_forest_anomaly, seasonal_anomaly, zscore_anomaly,
};
pub use ts_features::{TsFeature, TsFeatureResult, extract_default_features, extract_features};
pub use var::{GrangerResult, VarModel};

/// Items intended for glob-import: `use scivex_stats::prelude::*;`
pub mod prelude {
    pub use crate::bayesian::{
        HamiltonianMC, McmcConfig, McmcResult, MetropolisHastings, Nuts, TraceSummary,
        effective_sample_size, rhat, trace_summary,
    };
    pub use crate::bayesian_optim::{
        AcquisitionFunction, BayesOptConfig, BayesOptResult, BayesianOptimizer, Kernel,
    };
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
    pub use crate::effect_size::{
        EffectSizeInterpretation, cohens_d, cohens_w, cramers_v, eta_squared, glass_delta,
        hedges_g, interpret_cohens_d, omega_squared, point_biserial,
    };
    pub use crate::error::{Result, StatsError};
    pub use crate::garch::Garch;
    pub use crate::glm::{Family, GlmResult, LinkFunction, glm};
    pub use crate::hypothesis::{
        TestResult, anova_oneway, batch_t_test_one_sample, batch_t_test_two_sample,
        chi_square_test, ks_test_two_sample, mann_whitney_u, t_test_one_sample, t_test_two_sample,
    };
    pub use crate::kalman::KalmanFilter;
    pub use crate::mixed_effects::{LmmResult, lmm};
    pub use crate::online::OnlineStats;
    pub use crate::prophet::{Prophet, ProphetConfig, ProphetForecast};
    pub use crate::regression::{OlsResult, ols};
    pub use crate::survival::{
        CoxPHResult, KaplanMeierEstimate, LogRankResult, SurvivalRecord, cox_ph, kaplan_meier,
        log_rank_test, median_survival_time,
    };
    pub use crate::ts_anomaly::{
        AnomalyResult, ewma_anomaly, isolation_forest_anomaly, seasonal_anomaly, zscore_anomaly,
    };
    pub use crate::ts_features::{
        TsFeature, TsFeatureResult, extract_default_features, extract_features,
    };
    pub use crate::var::{GrangerResult, VarModel};
}
