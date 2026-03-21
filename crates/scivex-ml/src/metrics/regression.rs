use scivex_core::Float;

use crate::error::{MlError, Result};

/// Mean squared error.
///
/// # Examples
///
/// ```
/// # use scivex_ml::metrics::mse;
/// let y_true = [1.0_f64, 2.0, 3.0];
/// let y_pred = [1.5, 2.5, 3.5];
/// assert!((mse(&y_true, &y_pred).unwrap() - 0.25).abs() < 1e-12);
/// ```
pub fn mse<T: Float>(y_true: &[T], y_pred: &[T]) -> Result<T> {
    check_inputs(y_true, y_pred)?;
    let n = T::from_usize(y_true.len());
    let sum = y_true
        .iter()
        .zip(y_pred)
        .map(|(&t, &p)| {
            let d = t - p;
            d * d
        })
        .fold(T::zero(), |a, b| a + b);
    Ok(sum / n)
}

/// Root mean squared error.
///
/// # Examples
///
/// ```
/// # use scivex_ml::metrics::rmse;
/// let y_true = [1.0_f64, 2.0, 3.0];
/// let y_pred = [1.5, 2.5, 3.5];
/// assert!((rmse(&y_true, &y_pred).unwrap() - 0.5).abs() < 1e-12);
/// ```
pub fn rmse<T: Float>(y_true: &[T], y_pred: &[T]) -> Result<T> {
    Ok(mse(y_true, y_pred)?.sqrt())
}

/// Mean absolute error.
///
/// # Examples
///
/// ```
/// # use scivex_ml::metrics::mae;
/// let y_true = [1.0_f64, 2.0, 3.0];
/// let y_pred = [1.5, 2.5, 3.5];
/// assert!((mae(&y_true, &y_pred).unwrap() - 0.5).abs() < 1e-12);
/// ```
pub fn mae<T: Float>(y_true: &[T], y_pred: &[T]) -> Result<T> {
    check_inputs(y_true, y_pred)?;
    let n = T::from_usize(y_true.len());
    let sum = y_true
        .iter()
        .zip(y_pred)
        .map(|(&t, &p)| (t - p).abs())
        .fold(T::zero(), |a, b| a + b);
    Ok(sum / n)
}

/// R-squared (coefficient of determination).
///
/// # Examples
///
/// ```
/// # use scivex_ml::metrics::r2_score;
/// let y = [1.0_f64, 2.0, 3.0];
/// assert!((r2_score(&y, &y).unwrap() - 1.0).abs() < 1e-12);
/// ```
pub fn r2_score<T: Float>(y_true: &[T], y_pred: &[T]) -> Result<T> {
    check_inputs(y_true, y_pred)?;
    let n = T::from_usize(y_true.len());
    let mean = y_true.iter().copied().fold(T::zero(), |a, b| a + b) / n;
    let ss_res = y_true
        .iter()
        .zip(y_pred)
        .map(|(&t, &p)| {
            let d = t - p;
            d * d
        })
        .fold(T::zero(), |a, b| a + b);
    let ss_tot = y_true
        .iter()
        .map(|&t| {
            let d = t - mean;
            d * d
        })
        .fold(T::zero(), |a, b| a + b);
    if ss_tot < T::epsilon() {
        return Ok(T::zero());
    }
    Ok(T::one() - ss_res / ss_tot)
}

fn check_inputs<T: Float>(y_true: &[T], y_pred: &[T]) -> Result<()> {
    if y_true.is_empty() {
        return Err(MlError::EmptyInput);
    }
    if y_true.len() != y_pred.len() {
        return Err(MlError::DimensionMismatch {
            expected: y_true.len(),
            got: y_pred.len(),
        });
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn test_mse_perfect() {
        let y = [1.0_f64, 2.0, 3.0];
        assert!((mse(&y, &y).unwrap()).abs() < 1e-12);
    }

    #[test]
    fn test_mse_known() {
        let y_true = [1.0_f64, 2.0, 3.0];
        let y_pred = [1.5, 2.5, 3.5];
        // MSE = (0.25 + 0.25 + 0.25) / 3 = 0.25
        assert!((mse(&y_true, &y_pred).unwrap() - 0.25).abs() < 1e-12);
    }

    #[test]
    fn test_rmse() {
        let y_true = [1.0_f64, 2.0, 3.0];
        let y_pred = [1.5, 2.5, 3.5];
        assert!((rmse(&y_true, &y_pred).unwrap() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_mae() {
        let y_true = [1.0_f64, 2.0, 3.0];
        let y_pred = [1.5, 2.5, 3.5];
        assert!((mae(&y_true, &y_pred).unwrap() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_r2_perfect() {
        let y = [1.0_f64, 2.0, 3.0];
        assert!((r2_score(&y, &y).unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_r2_known() {
        let y_true = [3.0_f64, -0.5, 2.0, 7.0];
        let y_pred = [2.5, 0.0, 2.0, 8.0];
        // ss_res = 0.25 + 0.25 + 0 + 1 = 1.5
        // mean = 11.5/4 = 2.875
        // ss_tot = (3-2.875)^2 + (-0.5-2.875)^2 + (2-2.875)^2 + (7-2.875)^2
        //        = 0.015625 + 11.390625 + 0.765625 + 17.015625 = 29.1875
        // R2 = 1 - 1.5/29.1875 ≈ 0.9486
        let r2 = r2_score(&y_true, &y_pred).unwrap();
        assert!((r2 - 0.948_608_137_045_f64).abs() < 1e-6);
    }
}
