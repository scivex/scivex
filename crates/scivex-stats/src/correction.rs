//! Multiple comparison corrections.
//!
//! When performing many hypothesis tests simultaneously, the chance of
//! false positives increases. These corrections adjust p-values to control
//! the family-wise error rate (FWER) or false discovery rate (FDR).

use scivex_core::Float;

use crate::error::{Result, StatsError};

/// Bonferroni correction: multiply each p-value by the number of tests.
///
/// Controls the family-wise error rate (FWER). Conservative but simple.
///
/// Returns adjusted p-values clamped to `[0, 1]`.
pub fn bonferroni<T: Float>(p_values: &[T]) -> Result<Vec<T>> {
    if p_values.is_empty() {
        return Err(StatsError::EmptyInput);
    }
    let m = T::from_f64(p_values.len() as f64);
    Ok(p_values.iter().map(|&p| (p * m).min(T::one())).collect())
}

/// Benjamini-Hochberg correction: controls the false discovery rate (FDR).
///
/// Less conservative than Bonferroni; more powerful when many tests are performed.
///
/// Returns adjusted p-values. The adjusted p-values maintain the original order.
pub fn benjamini_hochberg<T: Float>(p_values: &[T]) -> Result<Vec<T>> {
    if p_values.is_empty() {
        return Err(StatsError::EmptyInput);
    }

    let m = p_values.len();
    let mf = T::from_f64(m as f64);

    // Create index-value pairs and sort by p-value
    let mut indexed: Vec<(usize, T)> = p_values.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal));

    // Compute adjusted p-values: p_adj[i] = p[i] * m / rank
    // Then enforce monotonicity from the largest rank down
    let mut adjusted = vec![T::zero(); m];
    let mut cummin = T::one();

    for i in (0..m).rev() {
        let rank = T::from_f64((i + 1) as f64);
        let (orig_idx, p_val) = indexed[i];
        let adj = (p_val * mf / rank).min(T::one());
        cummin = cummin.min(adj);
        adjusted[orig_idx] = cummin;
    }

    Ok(adjusted)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bonferroni_basic() {
        let p = vec![0.01_f64, 0.04, 0.03, 0.005];
        let adj = bonferroni(&p).unwrap();
        assert!((adj[0] - 0.04).abs() < 1e-10);
        assert!((adj[1] - 0.16).abs() < 1e-10);
        assert!((adj[2] - 0.12).abs() < 1e-10);
        assert!((adj[3] - 0.02).abs() < 1e-10);
    }

    #[test]
    fn test_bonferroni_clamp() {
        let p = vec![0.5_f64, 0.8];
        let adj = bonferroni(&p).unwrap();
        assert!((adj[0] - 1.0).abs() < 1e-10);
        assert!((adj[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_bonferroni_empty() {
        assert!(bonferroni::<f64>(&[]).is_err());
    }

    #[test]
    fn test_bh_basic() {
        let p = vec![0.01_f64, 0.04, 0.03, 0.005];
        let adj = benjamini_hochberg(&p).unwrap();
        // Sorted: 0.005 (rank 1), 0.01 (rank 2), 0.03 (rank 3), 0.04 (rank 4)
        // Raw adj: 0.005*4/1=0.02, 0.01*4/2=0.02, 0.03*4/3=0.04, 0.04*4/4=0.04
        // After cummin from right: 0.02, 0.02, 0.04, 0.04
        assert!((adj[3] - 0.02).abs() < 1e-10); // p=0.005 -> 0.02
        assert!((adj[0] - 0.02).abs() < 1e-10); // p=0.01 -> 0.02
        assert!((adj[2] - 0.04).abs() < 1e-10); // p=0.03 -> 0.04
        assert!((adj[1] - 0.04).abs() < 1e-10); // p=0.04 -> 0.04
    }

    #[test]
    fn test_bh_less_conservative_than_bonferroni() {
        let p = vec![0.01_f64, 0.02, 0.03, 0.04, 0.05];
        let bon = bonferroni(&p).unwrap();
        let bh = benjamini_hochberg(&p).unwrap();
        // BH should be <= Bonferroni for each p-value
        for i in 0..p.len() {
            assert!(
                bh[i] <= bon[i] + 1e-10,
                "BH[{i}]={} > Bonferroni[{i}]={}",
                bh[i],
                bon[i]
            );
        }
    }

    #[test]
    fn test_bh_single() {
        let adj = benjamini_hochberg(&[0.05_f64]).unwrap();
        assert!((adj[0] - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_bh_empty() {
        assert!(benjamini_hochberg::<f64>(&[]).is_err());
    }

    #[test]
    fn test_bh_preserves_order_significance() {
        // If p1 < p2, then adjusted p1 <= adjusted p2
        let p = vec![0.001_f64, 0.01, 0.05, 0.1, 0.5];
        let adj = benjamini_hochberg(&p).unwrap();
        for i in 1..adj.len() {
            assert!(adj[i - 1] <= adj[i] + 1e-10);
        }
    }
}
