//! Descriptive statistics on slices of floating-point values.

use scivex_core::Float;

use crate::error::{Result, StatsError};

/// Compute the arithmetic mean of `data`.
///
/// # Examples
///
/// ```
/// # use scivex_stats::descriptive::mean;
/// let data = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
/// let m = mean(&data).unwrap();
/// assert!((m - 3.0).abs() < 1e-10);
/// ```
pub fn mean<T: Float>(data: &[T]) -> Result<T> {
    if data.is_empty() {
        return Err(StatsError::EmptyInput);
    }
    let sum: T = data.iter().copied().sum();
    Ok(sum / T::from_f64(data.len() as f64))
}

/// Compute the sample variance with `ddof` (delta degrees of freedom).
///
/// Uses `ddof = 1` (Bessel's correction) by default in [`variance`].
pub fn variance_with_ddof<T: Float>(data: &[T], ddof: usize) -> Result<T> {
    let n = data.len();
    if n <= ddof {
        return Err(StatsError::InsufficientData {
            need: ddof + 1,
            got: n,
        });
    }
    let m = mean(data)?;
    let ss: T = data.iter().map(|&x| (x - m) * (x - m)).sum();
    Ok(ss / T::from_f64((n - ddof) as f64))
}

/// Sample variance (ddof = 1).
///
/// # Examples
///
/// ```
/// # use scivex_stats::descriptive::variance;
/// let data = [2.0_f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
/// let v = variance(&data).unwrap();
/// assert!((v - 4.571_428_571_428_571).abs() < 1e-8);
/// ```
pub fn variance<T: Float>(data: &[T]) -> Result<T> {
    variance_with_ddof(data, 1)
}

/// Sample standard deviation with given `ddof`.
pub fn std_dev_with_ddof<T: Float>(data: &[T], ddof: usize) -> Result<T> {
    Ok(variance_with_ddof(data, ddof)?.sqrt())
}

/// Sample standard deviation (ddof = 1).
pub fn std_dev<T: Float>(data: &[T]) -> Result<T> {
    std_dev_with_ddof(data, 1)
}

/// Compute the median of `data`.
///
/// For even-length data, returns the average of the two middle values.
pub fn median<T: Float>(data: &[T]) -> Result<T> {
    if data.is_empty() {
        return Err(StatsError::EmptyInput);
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    let n = sorted.len();
    if n % 2 == 1 {
        Ok(sorted[n / 2])
    } else {
        Ok((sorted[n / 2 - 1] + sorted[n / 2]) / T::from_f64(2.0))
    }
}

/// Compute the `q`-th quantile (0 <= q <= 1) using linear interpolation.
pub fn quantile<T: Float>(data: &[T], q: T) -> Result<T> {
    if data.is_empty() {
        return Err(StatsError::EmptyInput);
    }
    let zero = T::from_f64(0.0);
    let one = T::from_f64(1.0);
    if q < zero || q > one {
        return Err(StatsError::InvalidParameter {
            name: "q",
            reason: "must be in [0, 1]",
        });
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    let n = sorted.len();
    if n == 1 {
        return Ok(sorted[0]);
    }

    let idx = q * T::from_f64((n - 1) as f64);
    let lo = idx.floor();
    let hi = idx.ceil();
    let frac = idx - lo;

    let lo_idx = lo.min(T::from_f64((n - 1) as f64));
    let hi_idx = hi.min(T::from_f64((n - 1) as f64));

    // Convert to usize via linear scan (generic T has no direct to-usize)
    let lo_i = float_to_usize(lo_idx, n);
    let hi_i = float_to_usize(hi_idx, n);

    Ok(sorted[lo_i] * (one - frac) + sorted[hi_i] * frac)
}

/// Convert a generic Float index to usize by linear scan (up to `max_n - 1`).
fn float_to_usize<T: Float>(val: T, max_n: usize) -> usize {
    let mut i = 0usize;
    while T::from_f64(i as f64) < val && i < max_n - 1 {
        i += 1;
    }
    i
}

/// Adjusted Fisher–Pearson skewness.
pub fn skewness<T: Float>(data: &[T]) -> Result<T> {
    let n = data.len();
    if n < 3 {
        return Err(StatsError::InsufficientData { need: 3, got: n });
    }
    let m = mean(data)?;
    let s = std_dev(data)?;
    if s == T::from_f64(0.0) {
        return Ok(T::from_f64(0.0));
    }
    let nf = T::from_f64(n as f64);
    let m3: T = data.iter().map(|&x| ((x - m) / s).powi(3)).sum();
    Ok(m3 / nf)
}

/// Excess kurtosis.
pub fn kurtosis<T: Float>(data: &[T]) -> Result<T> {
    let n = data.len();
    if n < 4 {
        return Err(StatsError::InsufficientData { need: 4, got: n });
    }
    let m = mean(data)?;
    let s = std_dev(data)?;
    if s == T::from_f64(0.0) {
        return Ok(T::from_f64(0.0));
    }
    let nf = T::from_f64(n as f64);
    let m4: T = data.iter().map(|&x| ((x - m) / s).powi(4)).sum();
    Ok(m4 / nf - T::from_f64(3.0))
}

/// Summary statistics returned by [`describe`].
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct DescribeResult<T: Float> {
    pub count: usize,
    pub mean: T,
    pub std_dev: T,
    pub min: T,
    pub q25: T,
    pub median: T,
    pub q75: T,
    pub max: T,
    pub skewness: T,
    pub kurtosis: T,
}

/// Compute summary statistics for `data`.
pub fn describe<T: Float>(data: &[T]) -> Result<DescribeResult<T>> {
    if data.is_empty() {
        return Err(StatsError::EmptyInput);
    }
    let n = data.len();
    let m = mean(data)?;
    let s = if n >= 2 {
        std_dev(data)?
    } else {
        T::from_f64(0.0)
    };

    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));

    let sk = if n >= 3 {
        skewness(data)?
    } else {
        T::from_f64(0.0)
    };
    let ku = if n >= 4 {
        kurtosis(data)?
    } else {
        T::from_f64(0.0)
    };

    Ok(DescribeResult {
        count: n,
        mean: m,
        std_dev: s,
        min: sorted[0],
        q25: quantile(data, T::from_f64(0.25))?,
        median: median(data)?,
        q75: quantile(data, T::from_f64(0.75))?,
        max: sorted[n - 1],
        skewness: sk,
        kurtosis: ku,
    })
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn test_mean() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((mean(&data).unwrap() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_mean_empty() {
        assert!(mean::<f64>(&[]).is_err());
    }

    #[test]
    fn test_variance() {
        let data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        // sample variance = 4.571428...
        let v = variance(&data).unwrap();
        assert!((v - 4.571_428_571_428_571).abs() < 1e-8);
    }

    #[test]
    fn test_median_odd() {
        let data = [1.0, 3.0, 2.0];
        assert!((median(&data).unwrap() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_median_even() {
        let data = [1.0, 2.0, 3.0, 4.0];
        assert!((median(&data).unwrap() - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_quantile() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((quantile(&data, 0.0).unwrap() - 1.0).abs() < 1e-10);
        assert!((quantile(&data, 1.0).unwrap() - 5.0).abs() < 1e-10);
        assert!((quantile(&data, 0.5).unwrap() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_skewness_symmetric() {
        // Symmetric data should have ~0 skewness
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(skewness(&data).unwrap().abs() < 1e-10);
    }

    #[test]
    fn test_kurtosis() {
        // Normal-like data should have excess kurtosis near 0
        let data = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0];
        let k = kurtosis(&data).unwrap();
        // Not exactly 0 for 9 points, but should be reasonable
        assert!(k.abs() < 2.0);
    }

    #[test]
    fn test_describe() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let d = describe(&data).unwrap();
        assert_eq!(d.count, 10);
        assert!((d.mean - 5.5).abs() < 1e-10);
        assert!((d.min - 1.0).abs() < 1e-10);
        assert!((d.max - 10.0).abs() < 1e-10);
        assert!((d.median - 5.5).abs() < 1e-10);
    }

    #[test]
    fn test_insufficient_data() {
        assert!(variance::<f64>(&[1.0]).is_err());
        assert!(skewness::<f64>(&[1.0, 2.0]).is_err());
        assert!(kurtosis::<f64>(&[1.0, 2.0, 3.0]).is_err());
    }
}
