"""Statistics — descriptive stats, hypothesis tests, distributions."""
import pyscivex as sv

# Descriptive statistics
data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
print(f"Mean:     {sv.mean(data)}")
print(f"Std Dev:  {sv.std_dev(data):.4f}")
print(f"Variance: {sv.variance(data):.4f}")
print(f"Median:   {sv.median(data)}")

# Hypothesis test: independent t-test
group_a = [23.0, 25.0, 28.0, 30.0, 32.0]
group_b = [18.0, 20.0, 22.0, 24.0, 26.0]
result = sv.stats.ttest_ind(group_a, group_b)
print(f"\nt-test: t={result['statistic']:.3f}, p={result['p_value']:.4f}")

# Effect size
d = sv.stats.cohens_d(group_a, group_b)
print(f"Cohen's d: {d:.3f}")

# Correlation
x = [1.0, 2.0, 3.0, 4.0, 5.0]
y = [2.1, 4.0, 5.9, 8.1, 9.8]
r = sv.pearson(x, y)
print(f"\nPearson r: {r:.4f}")

# Distribution
normal = sv.stats.Normal(0.0, 1.0)
print(f"\nN(0,1) pdf(0) = {normal.pdf(0.0):.4f}")
print(f"N(0,1) cdf(1.96) = {normal.cdf(1.96):.4f}")

# Time series
ts = [1.0, 2.0, 3.0, 2.5, 3.5, 4.0, 3.0, 4.5, 5.0, 4.0]
acf_vals = sv.stats.acf(ts, 5)
print(f"\nACF(lag 0-5): {[round(v, 3) for v in acf_vals]}")
