use scivex_core::{Float, Tensor};

use crate::error::{MlError, Result};

/// Target encoder: replaces categorical features with smoothed target means.
///
/// This is a supervised encoder — it requires target values (`y`) during
/// fitting. For each category in each column, the encoder computes a smoothed
/// mean using Bayesian smoothing:
///
/// ```text
/// encoded = (count * category_mean + smoothing * global_mean) / (count + smoothing)
/// ```
///
/// where `count` is the number of samples with that category, `category_mean`
/// is the mean target for that category, `global_mean` is the overall target
/// mean, and `smoothing` controls the strength of regularization toward the
/// global mean.
///
/// Unknown categories encountered during `transform` are mapped to the global
/// mean.
///
/// # Examples
///
/// ```
/// # use scivex_ml::prelude::*;
/// # use scivex_core::prelude::*;
/// let x = Tensor::from_vec(vec![0.0_f64, 0.0, 1.0, 1.0], vec![4, 1]).unwrap();
/// let y = Tensor::from_vec(vec![10.0, 20.0, 30.0, 40.0], vec![4]).unwrap();
/// let mut enc = TargetEncoder::new(1.0);
/// enc.fit_supervised(&x, &y).unwrap();
/// let out = enc.transform(&x).unwrap();
/// assert_eq!(out.shape(), &[4, 1]);
/// ```
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct TargetEncoder<T: Float> {
    /// Smoothing parameter (higher = more regularization toward global mean).
    smoothing: T,
    /// Per-column category-to-encoded-value mappings.
    /// Each entry maps (sorted categories, encoded values).
    encodings: Option<Vec<(Vec<T>, Vec<T>)>>,
    /// Global target mean.
    global_mean: Option<T>,
}

impl<T: Float> TargetEncoder<T> {
    /// Create a new target encoder with the given smoothing parameter.
    ///
    /// # Arguments
    ///
    /// * `smoothing` — regularization strength. A value of 0 uses pure
    ///   category means; higher values shrink toward the global mean.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_ml::prelude::*;
    /// let enc = TargetEncoder::<f64>::new(1.0);
    /// ```
    pub fn new(smoothing: T) -> Self {
        Self {
            smoothing,
            encodings: None,
            global_mean: None,
        }
    }

    /// Fit the encoder using features `x` and targets `y`.
    ///
    /// `x` must be a 2-D tensor of shape `[n_samples, n_features]`.
    /// `y` must be a 1-D tensor of shape `[n_samples]`.
    pub fn fit_supervised(&mut self, x: &Tensor<T>, y: &Tensor<T>) -> Result<()> {
        let (n, p) = matrix_shape(x)?;
        let y_shape = y.shape();
        let y_len = if y_shape.len() == 1 || (y_shape.len() == 2 && y_shape[1] == 1) {
            y_shape[0]
        } else {
            return Err(MlError::DimensionMismatch {
                expected: 1,
                got: y_shape.len(),
            });
        };
        if y_len != n {
            return Err(MlError::DimensionMismatch {
                expected: n,
                got: y_len,
            });
        }

        let x_data = x.as_slice();
        let y_data = y.as_slice();

        // Compute global mean
        let mut sum = T::zero();
        for &yv in &y_data[..n] {
            sum += yv;
        }
        let global_mean = sum / T::from_f64(n as f64);
        self.global_mean = Some(global_mean);

        let mut encodings = Vec::with_capacity(p);

        for col in 0..p {
            // Collect unique categories
            let mut unique: Vec<T> = Vec::new();
            for row in 0..n {
                let val = round_val(x_data[row * p + col]);
                if !unique.iter().any(|&u| (u - val).abs() < T::from_f64(1e-9)) {
                    unique.push(val);
                }
            }
            unique.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            // Compute smoothed means
            let mut encoded = Vec::with_capacity(unique.len());
            for &cat in &unique {
                let mut cat_sum = T::zero();
                let mut cat_count = T::zero();
                for row in 0..n {
                    let val = round_val(x_data[row * p + col]);
                    if (val - cat).abs() < T::from_f64(1e-9) {
                        cat_sum += y_data[row];
                        cat_count += T::one();
                    }
                }
                let cat_mean = cat_sum / cat_count;
                // Bayesian smoothing
                let smoothed = (cat_count * cat_mean + self.smoothing * global_mean)
                    / (cat_count + self.smoothing);
                encoded.push(smoothed);
            }

            encodings.push((unique, encoded));
        }

        self.encodings = Some(encodings);
        Ok(())
    }

    /// Transform features using the learned target-based encodings.
    ///
    /// Unknown categories are mapped to the global target mean.
    pub fn transform(&self, x: &Tensor<T>) -> Result<Tensor<T>> {
        let encodings = self.encodings.as_ref().ok_or(MlError::NotFitted)?;
        let global_mean = self.global_mean.ok_or(MlError::NotFitted)?;
        let (n, p) = matrix_shape(x)?;
        if p != encodings.len() {
            return Err(MlError::DimensionMismatch {
                expected: encodings.len(),
                got: p,
            });
        }

        let data = x.as_slice();
        let mut out = vec![T::zero(); n * p];

        for row in 0..n {
            for col in 0..p {
                let val = round_val(data[row * p + col]);
                let (cats, encoded) = &encodings[col];
                let idx = cats
                    .iter()
                    .position(|&c| (c - val).abs() < T::from_f64(1e-9));
                out[row * p + col] = match idx {
                    Some(i) => encoded[i],
                    None => global_mean,
                };
            }
        }

        Tensor::from_vec(out, vec![n, p]).map_err(MlError::from)
    }

    /// Return the global target mean, or `None` if not fitted.
    pub fn global_mean(&self) -> Option<T> {
        self.global_mean
    }

    /// Return the smoothing parameter.
    pub fn smoothing(&self) -> T {
        self.smoothing
    }
}

/// Round to nearest integer (using floor(x + 0.5)).
fn round_val<T: Float>(v: T) -> T {
    (v + T::from_f64(0.5)).floor()
}

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
    fn test_target_encoder_basic() {
        // 1 feature, 2 categories: cat0 has targets {10,20}→mean=15, cat1 has {30,40}→mean=35
        // global mean = 25, smoothing = 1
        // cat0: (2*15 + 1*25)/(2+1) = 55/3 ≈ 18.333...
        // cat1: (2*35 + 1*25)/(2+1) = 95/3 ≈ 31.666...
        let x = Tensor::from_vec(vec![0.0_f64, 0.0, 1.0, 1.0], vec![4, 1]).unwrap();
        let y = Tensor::from_vec(vec![10.0, 20.0, 30.0, 40.0], vec![4]).unwrap();
        let mut enc = TargetEncoder::new(1.0);
        enc.fit_supervised(&x, &y).unwrap();
        let out = enc.transform(&x).unwrap();

        assert_eq!(out.shape(), &[4, 1]);
        let d = out.as_slice();
        let expected_cat0 = 55.0 / 3.0;
        let expected_cat1 = 95.0 / 3.0;
        assert!((d[0] - expected_cat0).abs() < 1e-9);
        assert!((d[1] - expected_cat0).abs() < 1e-9);
        assert!((d[2] - expected_cat1).abs() < 1e-9);
        assert!((d[3] - expected_cat1).abs() < 1e-9);
    }

    #[test]
    fn test_target_encoder_unknown() {
        let x = Tensor::from_vec(vec![0.0_f64, 1.0], vec![2, 1]).unwrap();
        let y = Tensor::from_vec(vec![10.0, 30.0], vec![2]).unwrap();
        let mut enc = TargetEncoder::new(1.0);
        enc.fit_supervised(&x, &y).unwrap();

        let x_test = Tensor::from_vec(vec![99.0], vec![1, 1]).unwrap();
        let out = enc.transform(&x_test).unwrap();
        // Unknown → global mean = 20.0
        assert!((out.as_slice()[0] - 20.0).abs() < 1e-9);
    }

    #[test]
    fn test_target_encoder_zero_smoothing() {
        // With smoothing=0, should get pure category means
        let x = Tensor::from_vec(vec![0.0_f64, 0.0, 1.0], vec![3, 1]).unwrap();
        let y = Tensor::from_vec(vec![10.0, 20.0, 50.0], vec![3]).unwrap();
        let mut enc = TargetEncoder::new(0.0);
        enc.fit_supervised(&x, &y).unwrap();
        let out = enc.transform(&x).unwrap();

        let d = out.as_slice();
        // cat0 mean = 15.0, cat1 mean = 50.0
        assert!((d[0] - 15.0).abs() < 1e-9);
        assert!((d[1] - 15.0).abs() < 1e-9);
        assert!((d[2] - 50.0).abs() < 1e-9);
    }
}
