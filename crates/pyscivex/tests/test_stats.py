"""Tests for pyscivex statistics functions."""

import math
import pytest
import pyscivex as sv


# ---------------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------------

class TestDescriptiveStats:
    def test_mean(self):
        assert sv.mean([1.0, 2.0, 3.0, 4.0, 5.0]) == 3.0

    def test_variance(self):
        # sample variance (n-1 denominator)
        result = sv.variance([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        assert abs(result - 4.571428) < 0.01

    def test_std_dev(self):
        result = sv.std_dev([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        assert abs(result - 2.13809) < 0.01

    def test_median_odd(self):
        assert sv.median([1.0, 3.0, 5.0]) == 3.0

    def test_median_even(self):
        assert sv.median([1.0, 2.0, 3.0, 4.0]) == 2.5

    def test_quantile_median(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        q50 = sv.stats.quantile(data, 0.5)
        assert abs(q50 - 3.0) < 0.5  # median of 1..5

    def test_quantile_extremes(self):
        data = [10.0, 20.0, 30.0, 40.0, 50.0]
        q0 = sv.stats.quantile(data, 0.0)
        q1 = sv.stats.quantile(data, 1.0)
        assert q0 <= 10.0 + 1e-9
        assert q1 >= 50.0 - 1e-9

    def test_skewness_symmetric(self):
        # symmetric data should have skewness near 0
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        skew = sv.stats.skewness(data)
        assert abs(skew) < 0.1

    def test_kurtosis(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        k = sv.stats.kurtosis(data)
        # kurtosis should be a finite number
        assert math.isfinite(k)

    def test_describe_keys(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        d = sv.stats.describe(data)
        for key in ["count", "mean", "std", "min", "q25", "median", "q75", "max"]:
            assert key in d, f"Missing key: {key}"

    def test_describe_values(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        d = sv.stats.describe(data)
        assert d["count"] == 5
        assert abs(d["mean"] - 3.0) < 1e-9
        assert d["min"] <= d["q25"] <= d["median"] <= d["q75"] <= d["max"]

    def test_pearson_perfect(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]
        r = sv.pearson(x, y)
        assert abs(r - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------

class TestDistributions:
    # --- Normal ---
    def test_normal_creation(self):
        n = sv.stats.Normal(0.0, 1.0)
        assert n is not None

    def test_normal_mean(self):
        n = sv.stats.Normal(5.0, 2.0)
        assert abs(n.mean() - 5.0) < 1e-10

    def test_normal_variance(self):
        n = sv.stats.Normal(0.0, 3.0)
        assert abs(n.variance() - 9.0) < 1e-10

    def test_normal_pdf_at_mean(self):
        n = sv.stats.Normal(0.0, 1.0)
        # pdf(0) = 1/sqrt(2*pi) ~ 0.3989
        assert abs(n.pdf(0.0) - 0.3989422804) < 1e-4

    def test_normal_cdf_at_zero(self):
        n = sv.stats.Normal(0.0, 1.0)
        assert abs(n.cdf(0.0) - 0.5) < 1e-6

    def test_normal_ppf_roundtrip(self):
        n = sv.stats.Normal(0.0, 1.0)
        for p in [0.1, 0.25, 0.5, 0.75, 0.9]:
            x = n.ppf(p)
            p_back = n.cdf(x)
            assert abs(p_back - p) < 1e-4, f"ppf roundtrip failed at p={p}"

    def test_normal_sample(self):
        n = sv.stats.Normal(0.0, 1.0)
        s = n.sample(42)
        assert isinstance(s, float)

    def test_normal_sample_n(self):
        n = sv.stats.Normal(0.0, 1.0)
        samples = n.sample_n(100, 42)
        assert len(samples) == 100

    # --- Uniform ---
    def test_uniform_mean(self):
        u = sv.stats.Uniform(0.0, 10.0)
        assert abs(u.mean() - 5.0) < 1e-10

    def test_uniform_variance(self):
        u = sv.stats.Uniform(0.0, 12.0)
        # Var = (b-a)^2 / 12 = 144/12 = 12
        assert abs(u.variance() - 12.0) < 1e-10

    def test_uniform_cdf(self):
        u = sv.stats.Uniform(0.0, 1.0)
        assert abs(u.cdf(0.5) - 0.5) < 1e-10

    def test_uniform_pdf(self):
        u = sv.stats.Uniform(0.0, 4.0)
        # pdf inside [0,4] = 1/4 = 0.25
        assert abs(u.pdf(2.0) - 0.25) < 1e-10

    # --- Exponential ---
    def test_exponential_mean(self):
        e = sv.stats.Exponential(2.0)
        # mean = 1/lambda = 0.5
        assert abs(e.mean() - 0.5) < 1e-10

    def test_exponential_cdf(self):
        e = sv.stats.Exponential(1.0)
        # CDF(1) = 1 - e^{-1} ~ 0.6321
        assert abs(e.cdf(1.0) - 0.6321205588) < 1e-4

    def test_exponential_ppf_roundtrip(self):
        e = sv.stats.Exponential(1.0)
        x = e.ppf(0.5)
        p_back = e.cdf(x)
        assert abs(p_back - 0.5) < 1e-4

    # --- StudentT ---
    def test_studentt_mean(self):
        t = sv.stats.StudentT(5.0)
        assert abs(t.mean()) < 1e-10  # mean = 0 for df > 1

    def test_studentt_variance(self):
        t = sv.stats.StudentT(5.0)
        # Var = df/(df-2) = 5/3
        assert abs(t.variance() - 5.0 / 3.0) < 1e-6

    def test_studentt_cdf_symmetry(self):
        t = sv.stats.StudentT(10.0)
        assert abs(t.cdf(0.0) - 0.5) < 1e-6

    # --- Beta ---
    def test_beta_mean(self):
        b = sv.stats.Beta(2.0, 5.0)
        # mean = alpha/(alpha+beta) = 2/7
        assert abs(b.mean() - 2.0 / 7.0) < 1e-6

    def test_beta_variance(self):
        b = sv.stats.Beta(2.0, 5.0)
        # Var = ab / ((a+b)^2 (a+b+1)) = 10 / (49*8) = 10/392
        assert abs(b.variance() - 10.0 / 392.0) < 1e-6

    def test_beta_cdf_bounds(self):
        b = sv.stats.Beta(2.0, 5.0)
        assert abs(b.cdf(0.0)) < 1e-10
        assert abs(b.cdf(1.0) - 1.0) < 1e-10

    # --- Gamma ---
    def test_gamma_mean(self):
        g = sv.stats.Gamma(2.0, 1.0)
        # mean = shape/rate = 2.0
        assert abs(g.mean() - 2.0) < 1e-6

    def test_gamma_variance(self):
        g = sv.stats.Gamma(2.0, 1.0)
        # Var = shape/rate^2 = 2.0
        assert abs(g.variance() - 2.0) < 1e-6

    # --- Poisson ---
    def test_poisson_mean(self):
        p = sv.stats.Poisson(5.0)
        assert abs(p.mean() - 5.0) < 1e-10

    def test_poisson_variance(self):
        p = sv.stats.Poisson(5.0)
        assert abs(p.variance() - 5.0) < 1e-10

    # --- Binomial ---
    def test_binomial_mean(self):
        b = sv.stats.Binomial(10, 0.5)
        assert abs(b.mean() - 5.0) < 1e-10

    def test_binomial_variance(self):
        b = sv.stats.Binomial(10, 0.5)
        # Var = n*p*(1-p) = 2.5
        assert abs(b.variance() - 2.5) < 1e-10

    # --- ChiSquared ---
    def test_chisquared_mean(self):
        c = sv.stats.ChiSquared(3.0)
        assert abs(c.mean() - 3.0) < 1e-10

    def test_chisquared_variance(self):
        c = sv.stats.ChiSquared(3.0)
        # Var = 2*df = 6
        assert abs(c.variance() - 6.0) < 1e-10

    # --- Bernoulli ---
    def test_bernoulli_mean(self):
        b = sv.stats.Bernoulli(0.3)
        assert abs(b.mean() - 0.3) < 1e-10

    def test_bernoulli_variance(self):
        b = sv.stats.Bernoulli(0.3)
        # Var = p*(1-p) = 0.21
        assert abs(b.variance() - 0.21) < 1e-10

    # --- Cauchy ---
    def test_cauchy_pdf_at_location(self):
        c = sv.stats.Cauchy(0.0, 1.0)
        # pdf(0) = 1/(pi*scale) ~ 0.31831
        assert abs(c.pdf(0.0) - 1.0 / math.pi) < 1e-4

    def test_cauchy_cdf_at_location(self):
        c = sv.stats.Cauchy(0.0, 1.0)
        assert abs(c.cdf(0.0) - 0.5) < 1e-6

    # --- LogNormal ---
    def test_lognormal_mean(self):
        ln = sv.stats.LogNormal(0.0, 1.0)
        # mean = exp(mu + sigma^2/2) = exp(0.5) ~ 1.6487
        assert abs(ln.mean() - math.exp(0.5)) < 1e-4

    def test_lognormal_cdf(self):
        ln = sv.stats.LogNormal(0.0, 1.0)
        # CDF(1) = 0.5 (since median = exp(mu) = 1)
        assert abs(ln.cdf(1.0) - 0.5) < 1e-4

    # --- Weibull ---
    def test_weibull_sample_n(self):
        w = sv.stats.Weibull(1.5, 1.0)
        samples = w.sample_n(50, 99)
        assert len(samples) == 50

    # --- Laplace ---
    def test_laplace_mean(self):
        la = sv.stats.Laplace(3.0, 2.0)
        assert abs(la.mean() - 3.0) < 1e-10

    def test_laplace_variance(self):
        la = sv.stats.Laplace(0.0, 2.0)
        # Var = 2*scale^2 = 8
        assert abs(la.variance() - 8.0) < 1e-10

    def test_laplace_cdf_at_location(self):
        la = sv.stats.Laplace(0.0, 1.0)
        assert abs(la.cdf(0.0) - 0.5) < 1e-6


# ---------------------------------------------------------------------------
# Hypothesis tests
# ---------------------------------------------------------------------------

class TestHypothesisTests:
    def test_ttest_1samp_null_true(self):
        # data centred at 0 — should NOT reject
        data = [-1.0, -0.5, 0.0, 0.5, 1.0]
        result = sv.stats.ttest_1samp(data, 0.0)
        assert "statistic" in result and "p_value" in result
        assert result["p_value"] > 0.05

    def test_ttest_1samp_null_false(self):
        data = [10.0, 11.0, 12.0, 13.0, 14.0]
        result = sv.stats.ttest_1samp(data, 0.0)
        assert result["p_value"] < 0.05

    def test_ttest_ind_same(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [1.5, 2.5, 3.5, 4.5, 5.5]
        result = sv.stats.ttest_ind(x, y)
        assert result["p_value"] > 0.05  # similar distributions

    def test_ttest_ind_different(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [100.0, 101.0, 102.0, 103.0, 104.0]
        result = sv.stats.ttest_ind(x, y)
        assert result["p_value"] < 0.01

    def test_chi2_test_uniform(self):
        observed = [25.0, 25.0, 25.0, 25.0]
        expected = [25.0, 25.0, 25.0, 25.0]
        result = sv.stats.chi2_test(observed, expected)
        assert abs(result["statistic"]) < 1e-10
        assert result["p_value"] > 0.99

    def test_chi2_test_skewed(self):
        observed = [50.0, 10.0, 10.0, 10.0]
        expected = [20.0, 20.0, 20.0, 20.0]
        result = sv.stats.chi2_test(observed, expected)
        assert result["p_value"] < 0.05

    def test_ks_2samp_same(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        y = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5]
        result = sv.stats.ks_2samp(x, y)
        assert result["p_value"] > 0.05

    def test_anova_oneway_same(self):
        groups = [[1.0, 2.0, 3.0], [1.5, 2.5, 3.5], [1.0, 2.0, 3.0]]
        result = sv.stats.anova_oneway(groups)
        assert result["p_value"] > 0.05

    def test_anova_oneway_different(self):
        groups = [[1.0, 2.0, 3.0], [100.0, 101.0, 102.0], [200.0, 201.0, 202.0]]
        result = sv.stats.anova_oneway(groups)
        assert result["p_value"] < 0.01

    def test_mann_whitney_u(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [1.5, 2.5, 3.5, 4.5, 5.5]
        result = sv.stats.mann_whitney_u(x, y)
        assert "statistic" in result and "p_value" in result
        assert result["p_value"] > 0.05


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------

class TestCorrelation:
    def test_spearman_perfect_monotonic(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]
        r = sv.stats.spearman(x, y)
        assert abs(r - 1.0) < 1e-6

    def test_kendall_perfect_monotonic(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [10.0, 20.0, 30.0, 40.0, 50.0]
        tau = sv.stats.kendall(x, y)
        assert abs(tau - 1.0) < 1e-6

    def test_spearman_negative(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [50.0, 40.0, 30.0, 20.0, 10.0]
        r = sv.stats.spearman(x, y)
        assert abs(r - (-1.0)) < 1e-6

    def test_corr_matrix_shape(self):
        # 5 observations, 3 variables -> 3x3 correlation matrix
        data = sv.Tensor([
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
            10.0, 11.0, 12.0,
            13.0, 14.0, 15.0,
        ], [5, 3])
        mat = sv.stats.corr_matrix(data, "pearson")
        assert mat.shape() == [3, 3]

    def test_pearson_top_level(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [5.0, 4.0, 3.0, 2.0, 1.0]
        r = sv.pearson(x, y)
        assert abs(r - (-1.0)) < 1e-6


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------

class TestRegression:
    def test_ols_perfect_linear(self):
        # y = 2*x + 1 (perfect fit) — OLS auto-prepends intercept
        x = sv.Tensor([1.0, 2.0, 3.0, 4.0, 5.0], [5, 1])
        y = [3.0, 5.0, 7.0, 9.0, 11.0]
        result = sv.stats.ols(x, y)
        assert abs(result["r_squared"] - 1.0) < 1e-6

    def test_ols_coefficients(self):
        # y = 2*x + 1 — OLS auto-prepends intercept
        x = sv.Tensor([1.0, 2.0, 3.0, 4.0, 5.0], [5, 1])
        y = [3.0, 5.0, 7.0, 9.0, 11.0]
        result = sv.stats.ols(x, y)
        coeffs = result["coefficients"]
        # coeffs[0] = intercept ~ 1.0, coeffs[1] = slope ~ 2.0
        assert abs(coeffs[0] - 1.0) < 1e-4
        assert abs(coeffs[1] - 2.0) < 1e-4

    def test_ols_result_keys(self):
        x = sv.Tensor([1.0, 2.0, 3.0, 4.0, 5.0], [5, 1])
        y = [2.0, 4.0, 6.0, 8.0, 10.0]
        result = sv.stats.ols(x, y)
        expected_keys = [
            "coefficients", "std_errors", "t_statistics", "p_values",
            "r_squared", "r_squared_adj", "f_statistic", "f_p_value",
            "residuals", "n_obs", "df_resid",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_ols_n_obs(self):
        x = sv.Tensor([1.0, 2.0, 3.0, 4.0], [4, 1])
        y = [2.0, 3.0, 4.0, 5.0]
        result = sv.stats.ols(x, y)
        assert result["n_obs"] == 4

    def test_glm_gaussian(self):
        # Gaussian GLM should behave like OLS — OLS auto-prepends intercept
        x = sv.Tensor([1.0, 2.0, 3.0, 4.0, 5.0], [5, 1])
        y = [3.0, 5.0, 7.0, 9.0, 11.0]
        result = sv.stats.glm(x, y, "gaussian")
        coeffs = result["coefficients"]
        # intercept ~ 1.0, slope ~ 2.0
        assert abs(coeffs[0] - 1.0) < 0.5
        assert abs(coeffs[1] - 2.0) < 0.5


# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------

class TestConfidenceIntervals:
    def test_ci_mean_contains_true(self):
        data = [5.0, 5.1, 4.9, 5.0, 5.2, 4.8, 5.0, 5.1]
        ci = sv.stats.ci_mean(data, 0.95)
        assert ci["lower"] <= 5.0 <= ci["upper"]

    def test_ci_mean_keys(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        ci = sv.stats.ci_mean(data, 0.95)
        for key in ["lower", "upper", "estimate", "confidence"]:
            assert key in ci, f"Missing key: {key}"

    def test_ci_proportion(self):
        ci = sv.stats.ci_proportion(50, 100, 0.95)
        assert ci["lower"] < 0.5
        assert ci["upper"] > 0.5
        assert 0.0 <= ci["lower"] <= ci["upper"] <= 1.0


# ---------------------------------------------------------------------------
# Effect sizes
# ---------------------------------------------------------------------------

class TestEffectSizes:
    def test_cohens_d_identical(self):
        g1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        g2 = [1.0, 2.0, 3.0, 4.0, 5.0]
        d = sv.stats.cohens_d(g1, g2)
        assert abs(d) < 1e-10

    def test_hedges_g(self):
        g1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        g2 = [1.0, 2.0, 3.0, 4.0, 5.0]
        g = sv.stats.hedges_g(g1, g2)
        assert abs(g) < 1e-6

    def test_glass_delta(self):
        treatment = [5.0, 6.0, 7.0, 8.0, 9.0]
        control = [1.0, 2.0, 3.0, 4.0, 5.0]
        delta = sv.stats.glass_delta(treatment, control)
        # treatment mean = 7, control mean = 3, control std ~ 1.58
        # delta ~ (7-3)/1.58 ~ 2.53
        assert delta > 2.0

    def test_eta_squared(self):
        groups = [[1.0, 2.0, 3.0], [10.0, 11.0, 12.0], [20.0, 21.0, 22.0]]
        eta2 = sv.stats.eta_squared(groups)
        assert 0.0 <= eta2 <= 1.0
        # very different groups → high eta^2
        assert eta2 > 0.9

    def test_omega_squared(self):
        groups = [[1.0, 2.0, 3.0], [10.0, 11.0, 12.0], [20.0, 21.0, 22.0]]
        omega2 = sv.stats.omega_squared(groups)
        assert 0.0 <= omega2 <= 1.0


# ---------------------------------------------------------------------------
# Multiple-testing corrections
# ---------------------------------------------------------------------------

class TestCorrections:
    def test_bonferroni_all_ge_original(self):
        p_values = [0.01, 0.04, 0.03, 0.05]
        adjusted = sv.stats.bonferroni(p_values)
        assert len(adjusted) == len(p_values)
        for orig, adj in zip(p_values, adjusted):
            assert adj >= orig - 1e-15

    def test_bonferroni_values(self):
        p_values = [0.01, 0.02, 0.05]
        adjusted = sv.stats.bonferroni(p_values)
        # Bonferroni: multiply by k, cap at 1.0
        k = len(p_values)
        for orig, adj in zip(p_values, adjusted):
            expected = min(orig * k, 1.0)
            assert abs(adj - expected) < 1e-10

    def test_benjamini_hochberg_length(self):
        p_values = [0.005, 0.01, 0.03, 0.04, 0.5]
        adjusted = sv.stats.benjamini_hochberg(p_values)
        assert len(adjusted) == len(p_values)
        for orig, adj in zip(p_values, adjusted):
            assert adj >= orig - 1e-15
