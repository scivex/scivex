//! Correlation coefficients and correlation matrices.

use scivex_core::Float;
use scivex_core::Tensor;

use crate::descriptive::{mean, std_dev};
use crate::error::{Result, StatsError};

/// Which correlation method to use.
///
/// # Examples
///
/// ```
/// # use scivex_stats::correlation::CorrelationMethod;
/// let method = CorrelationMethod::Pearson;
/// assert_eq!(method, CorrelationMethod::Pearson);
/// ```
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorrelationMethod {
    Pearson,
    Spearman,
    Kendall,
}

/// Pearson product-moment correlation coefficient.
///
/// # Examples
///
/// ```
/// # use scivex_stats::correlation::pearson;
/// let x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
/// let y = vec![2.0_f64, 4.0, 6.0, 8.0, 10.0];
/// let r = pearson(&x, &y).unwrap();
/// assert!((r - 1.0).abs() < 1e-10); // perfect positive correlation
/// ```
pub fn pearson<T: Float>(x: &[T], y: &[T]) -> Result<T> {
    let n = x.len();
    if n != y.len() {
        return Err(StatsError::LengthMismatch {
            expected: n,
            got: y.len(),
        });
    }
    if n < 2 {
        return Err(StatsError::InsufficientData { need: 2, got: n });
    }
    let mx = mean(x)?;
    let my = mean(y)?;
    let sx = std_dev(x)?;
    let sy = std_dev(y)?;

    let zero = T::from_f64(0.0);
    if sx == zero || sy == zero {
        return Ok(zero);
    }

    let nf = T::from_f64(n as f64);
    let cov: T = x
        .iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| (xi - mx) * (yi - my))
        .sum();
    let cov = cov / (nf - T::from_f64(1.0));
    Ok(cov / (sx * sy))
}

/// Spearman rank correlation coefficient.
///
/// Ranks both arrays (with average tie-breaking), then computes Pearson
/// on the ranks.
///
/// # Examples
///
/// ```
/// # use scivex_stats::correlation::spearman;
/// let x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
/// let y = vec![5.0_f64, 4.0, 3.0, 2.0, 1.0];
/// let r = spearman(&x, &y).unwrap();
/// assert!((r - (-1.0)).abs() < 1e-10); // perfect negative rank correlation
/// ```
pub fn spearman<T: Float>(x: &[T], y: &[T]) -> Result<T> {
    if x.len() != y.len() {
        return Err(StatsError::LengthMismatch {
            expected: x.len(),
            got: y.len(),
        });
    }
    let rx = rank(x);
    let ry = rank(y);
    pearson(&rx, &ry)
}

/// Kendall's tau-b rank correlation coefficient.
///
/// # Examples
///
/// ```
/// # use scivex_stats::correlation::kendall;
/// let x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
/// let y = vec![2.0_f64, 4.0, 6.0, 8.0, 10.0];
/// let tau = kendall(&x, &y).unwrap();
/// assert!((tau - 1.0).abs() < 1e-10); // perfect concordance
/// ```
pub fn kendall<T: Float>(x: &[T], y: &[T]) -> Result<T> {
    let n = x.len();
    if n != y.len() {
        return Err(StatsError::LengthMismatch {
            expected: n,
            got: y.len(),
        });
    }
    if n < 2 {
        return Err(StatsError::InsufficientData { need: 2, got: n });
    }

    let mut concordant: i64 = 0;
    let mut discordant: i64 = 0;
    let mut tied_x: i64 = 0;
    let mut tied_y: i64 = 0;

    for i in 0..n {
        for j in (i + 1)..n {
            let dx = x[i].partial_cmp(&x[j]);
            let dy = y[i].partial_cmp(&y[j]);
            match (dx, dy) {
                (Some(core::cmp::Ordering::Equal), Some(core::cmp::Ordering::Equal)) => {
                    tied_x += 1;
                    tied_y += 1;
                }
                (Some(core::cmp::Ordering::Equal), _) => tied_x += 1,
                (_, Some(core::cmp::Ordering::Equal)) => tied_y += 1,
                (Some(a), Some(b)) => {
                    if a == b {
                        concordant += 1;
                    } else {
                        discordant += 1;
                    }
                }
                _ => {}
            }
        }
    }

    let n0 = (n * (n - 1) / 2) as f64;
    let denom = ((n0 - tied_x as f64) * (n0 - tied_y as f64)).sqrt();
    if denom < f64::EPSILON {
        return Ok(T::from_f64(0.0));
    }
    Ok(T::from_f64((concordant - discordant) as f64 / denom))
}

/// Compute a correlation matrix for columns of a 2-D tensor.
///
/// `data` should be `[n_obs x n_vars]`. Returns `[n_vars x n_vars]`.
pub fn corr_matrix<T: Float>(data: &Tensor<T>, method: CorrelationMethod) -> Result<Tensor<T>> {
    if data.ndim() != 2 {
        return Err(StatsError::InvalidParameter {
            name: "data",
            reason: "must be a 2-D tensor",
        });
    }
    let n_obs = data.shape()[0];
    let n_vars = data.shape()[1];
    let slice = data.as_slice();

    // Extract columns
    let cols: Vec<Vec<T>> = (0..n_vars)
        .map(|j| (0..n_obs).map(|i| slice[i * n_vars + j]).collect())
        .collect();

    let corr_fn = match method {
        CorrelationMethod::Pearson => pearson,
        CorrelationMethod::Spearman => spearman,
        CorrelationMethod::Kendall => kendall,
    };

    let mut result = vec![T::from_f64(0.0); n_vars * n_vars];
    for i in 0..n_vars {
        result[i * n_vars + i] = T::from_f64(1.0);
        for j in (i + 1)..n_vars {
            let r = corr_fn(&cols[i], &cols[j])?;
            result[i * n_vars + j] = r;
            result[j * n_vars + i] = r;
        }
    }

    Ok(Tensor::from_vec(result, vec![n_vars, n_vars])?)
}

// ---------------------------------------------------------------------------
// Ranking helper
// ---------------------------------------------------------------------------

/// Rank data with average tie-breaking. Returns Vec<T> of ranks (1-based).
fn rank<T: Float>(data: &[T]) -> Vec<T> {
    let n = data.len();
    let mut indexed: Vec<(usize, T)> = data.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal));

    let mut ranks = vec![T::from_f64(0.0); n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n - 1
            && indexed[j + 1]
                .1
                .partial_cmp(&indexed[j].1)
                .unwrap_or(core::cmp::Ordering::Equal)
                == core::cmp::Ordering::Equal
        {
            j += 1;
        }
        // Average rank for ties at positions i..=j
        let avg_rank = T::from_f64((i + j) as f64 / 2.0 + 1.0);
        for k in i..=j {
            ranks[indexed[k].0] = avg_rank;
        }
        i = j + 1;
    }
    ranks
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn test_pearson_perfect_positive() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [2.0, 4.0, 6.0, 8.0, 10.0];
        assert!((pearson(&x, &y).unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_pearson_perfect_negative() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [10.0, 8.0, 6.0, 4.0, 2.0];
        assert!((pearson(&x, &y).unwrap() + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_spearman_monotonic() {
        // Spearman = Pearson for monotonic transformations
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [1.0, 4.0, 9.0, 16.0, 25.0]; // perfect monotonic
        assert!((spearman(&x, &y).unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_kendall_known() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((kendall(&x, &y).unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_corr_matrix_diagonal() {
        let data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]).unwrap();
        let cm = corr_matrix(&data, CorrelationMethod::Pearson).unwrap();
        assert_eq!(cm.shape(), &[2, 2]);
        assert!((cm.as_slice()[0] - 1.0).abs() < 1e-10); // [0,0]
        assert!((cm.as_slice()[3] - 1.0).abs() < 1e-10); // [1,1]
    }

    #[test]
    fn test_length_mismatch() {
        assert!(pearson(&[1.0, 2.0], &[1.0]).is_err());
    }
}
