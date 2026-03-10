use scivex_core::{Float, Tensor};

use crate::error::{MlError, Result};
use crate::traits::Transformer;

/// Standardises features by removing the mean and scaling to unit variance.
#[derive(Debug, Clone)]
pub struct StandardScaler<T: Float> {
    mean: Option<Vec<T>>,
    std: Option<Vec<T>>,
}

impl<T: Float> Default for StandardScaler<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float> StandardScaler<T> {
    /// Create a new, unfitted standard scaler.
    pub fn new() -> Self {
        Self {
            mean: None,
            std: None,
        }
    }

    /// Inverse-transform: reconstruct original values from standardised ones.
    pub fn inverse_transform(&self, x: &Tensor<T>) -> Result<Tensor<T>> {
        let mean = self.mean.as_ref().ok_or(MlError::NotFitted)?;
        let std = self.std.as_ref().ok_or(MlError::NotFitted)?;
        let (n, p) = matrix_shape(x)?;
        if p != mean.len() {
            return Err(MlError::DimensionMismatch {
                expected: mean.len(),
                got: p,
            });
        }
        let src = x.as_slice();
        let mut out = vec![T::zero(); n * p];
        for i in 0..n {
            for j in 0..p {
                out[i * p + j] = src[i * p + j] * std[j] + mean[j];
            }
        }
        Ok(Tensor::from_vec(out, vec![n, p])?)
    }
}

impl<T: Float> Transformer<T> for StandardScaler<T> {
    fn fit(&mut self, x: &Tensor<T>) -> Result<()> {
        let (n, p) = matrix_shape(x)?;
        let data = x.as_slice();
        let nf = T::from_usize(n);

        let mut mean = vec![T::zero(); p];
        for i in 0..n {
            for j in 0..p {
                mean[j] += data[i * p + j];
            }
        }
        for m in &mut mean {
            *m /= nf;
        }

        let mut variance = vec![T::zero(); p];
        for i in 0..n {
            for j in 0..p {
                let d = data[i * p + j] - mean[j];
                variance[j] += d * d;
            }
        }
        let std: Vec<T> = variance
            .iter()
            .map(|&v| {
                let s = (v / nf).sqrt();
                if s < T::epsilon() { T::one() } else { s }
            })
            .collect();

        self.mean = Some(mean);
        self.std = Some(std);
        Ok(())
    }

    fn transform(&self, x: &Tensor<T>) -> Result<Tensor<T>> {
        let mean = self.mean.as_ref().ok_or(MlError::NotFitted)?;
        let std = self.std.as_ref().ok_or(MlError::NotFitted)?;
        let (n, p) = matrix_shape(x)?;
        if p != mean.len() {
            return Err(MlError::DimensionMismatch {
                expected: mean.len(),
                got: p,
            });
        }
        let src = x.as_slice();
        let mut out = vec![T::zero(); n * p];
        for i in 0..n {
            for j in 0..p {
                out[i * p + j] = (src[i * p + j] - mean[j]) / std[j];
            }
        }
        Ok(Tensor::from_vec(out, vec![n, p])?)
    }
}

/// Scales features to the range `[0, 1]`.
#[derive(Debug, Clone)]
pub struct MinMaxScaler<T: Float> {
    min: Option<Vec<T>>,
    range: Option<Vec<T>>,
}

impl<T: Float> Default for MinMaxScaler<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float> MinMaxScaler<T> {
    /// Create a new, unfitted min-max scaler.
    pub fn new() -> Self {
        Self {
            min: None,
            range: None,
        }
    }
}

impl<T: Float> Transformer<T> for MinMaxScaler<T> {
    fn fit(&mut self, x: &Tensor<T>) -> Result<()> {
        let (n, p) = matrix_shape(x)?;
        let data = x.as_slice();
        let mut min_v = vec![T::infinity(); p];
        let mut max_v = vec![T::neg_infinity(); p];
        for i in 0..n {
            for j in 0..p {
                let v = data[i * p + j];
                min_v[j] = min_v[j].min(v);
                max_v[j] = max_v[j].max(v);
            }
        }
        let range: Vec<T> = min_v
            .iter()
            .zip(&max_v)
            .map(|(&lo, &hi)| {
                let r = hi - lo;
                if r < T::epsilon() { T::one() } else { r }
            })
            .collect();
        self.min = Some(min_v);
        self.range = Some(range);
        Ok(())
    }

    fn transform(&self, x: &Tensor<T>) -> Result<Tensor<T>> {
        let min = self.min.as_ref().ok_or(MlError::NotFitted)?;
        let range = self.range.as_ref().ok_or(MlError::NotFitted)?;
        let (n, p) = matrix_shape(x)?;
        if p != min.len() {
            return Err(MlError::DimensionMismatch {
                expected: min.len(),
                got: p,
            });
        }
        let src = x.as_slice();
        let mut out = vec![T::zero(); n * p];
        for i in 0..n {
            for j in 0..p {
                out[i * p + j] = (src[i * p + j] - min[j]) / range[j];
            }
        }
        Ok(Tensor::from_vec(out, vec![n, p])?)
    }
}

/// Extract `(n_samples, n_features)` from a 2-D tensor.
fn matrix_shape<T: Float>(x: &Tensor<T>) -> Result<(usize, usize)> {
    let s = x.shape();
    if s.len() != 2 {
        return Err(MlError::DimensionMismatch {
            expected: 2,
            got: s.len(),
        });
    }
    if s[0] == 0 {
        return Err(MlError::EmptyInput);
    }
    Ok((s[0], s[1]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_scaler_roundtrip() {
        let x = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]).unwrap();
        let mut scaler = StandardScaler::new();
        let transformed = scaler.fit_transform(&x).unwrap();
        let recovered = scaler.inverse_transform(&transformed).unwrap();
        for (a, b) in x.as_slice().iter().zip(recovered.as_slice()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn test_standard_scaler_zero_mean() {
        let x = Tensor::from_vec(vec![1.0_f64, 10.0, 2.0, 20.0, 3.0, 30.0], vec![3, 2]).unwrap();
        let mut scaler = StandardScaler::new();
        let t = scaler.fit_transform(&x).unwrap();
        // Column means should be ~0
        let d = t.as_slice();
        let m0 = (d[0] + d[2] + d[4]) / 3.0;
        let m1 = (d[1] + d[3] + d[5]) / 3.0;
        assert!(m0.abs() < 1e-12);
        assert!(m1.abs() < 1e-12);
    }

    #[test]
    fn test_minmax_scaler_range() {
        let x = Tensor::from_vec(vec![1.0_f64, 10.0, 3.0, 30.0, 5.0, 50.0], vec![3, 2]).unwrap();
        let mut scaler = MinMaxScaler::new();
        let t = scaler.fit_transform(&x).unwrap();
        let d = t.as_slice();
        // Min should be 0, max should be 1 for each column
        assert!(d[0].abs() < 1e-12); // min of col 0
        assert!((d[4] - 1.0).abs() < 1e-12); // max of col 0
        assert!(d[1].abs() < 1e-12); // min of col 1
        assert!((d[5] - 1.0).abs() < 1e-12); // max of col 1
    }

    #[test]
    fn test_not_fitted() {
        let scaler = StandardScaler::<f64>::new();
        let x = Tensor::from_vec(vec![1.0, 2.0], vec![1, 2]).unwrap();
        assert!(scaler.transform(&x).is_err());
    }
}
