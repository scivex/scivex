//! Automated feature engineering: polynomial features and nonlinear transforms.

use scivex_core::{Float, Tensor};

use crate::error::{MlError, Result};
use crate::traits::Transformer;

/// Helper: validate that `x` is a 2-D tensor and return `(n_samples, n_features)`.
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

// ---------------------------------------------------------------------------
// FeatureTransform enum
// ---------------------------------------------------------------------------

/// A primitive nonlinear transform that can be applied column-wise.
#[derive(Debug, Clone, PartialEq)]
pub enum FeatureTransform {
    /// x squared.
    Square,
    /// Square root of the absolute value.
    Sqrt,
    /// Natural logarithm of `|x| + 1`.
    Log,
    /// `1 / (|x| + epsilon)`.
    Reciprocal,
    /// Pairwise products of all feature pairs (i < j).
    Interaction,
}

// ---------------------------------------------------------------------------
// PolynomialFeatures
// ---------------------------------------------------------------------------

/// Generates polynomial (and interaction) features up to a given degree.
///
/// For `n` input features and `degree = d` the output contains all monomials
/// of degree ≤ `d`.  When `interaction_only` is `true`, only cross-product
/// terms with distinct features are kept (each feature appears at most once).
///
/// # Examples
///
/// ```
/// # use scivex_ml::feature_engineering::PolynomialFeatures;
/// # use scivex_ml::traits::Transformer;
/// # use scivex_core::Tensor;
/// let x = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
/// let mut poly = PolynomialFeatures::<f64>::new(2, true, false).unwrap();
/// let out = poly.fit_transform(&x).unwrap();
/// // bias(1) + x1 + x2 + x1^2 + x1*x2 + x2^2 = 6 features
/// assert_eq!(out.shape()[1], 6);
/// ```
#[derive(Debug, Clone)]
pub struct PolynomialFeatures<T: Float> {
    degree: usize,
    include_bias: bool,
    interaction_only: bool,
    n_input_features: Option<usize>,
    n_output_features: Option<usize>,
    /// The exponent tuples; each inner vec has length `n_input_features`.
    powers: Option<Vec<Vec<usize>>>,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Float> PolynomialFeatures<T> {
    /// Create a new `PolynomialFeatures` generator.
    ///
    /// # Errors
    ///
    /// Returns [`MlError::InvalidParameter`] if `degree` is zero.
    pub fn new(degree: usize, include_bias: bool, interaction_only: bool) -> Result<Self> {
        if degree == 0 {
            return Err(MlError::InvalidParameter {
                name: "degree",
                reason: "must be >= 1",
            });
        }
        Ok(Self {
            degree,
            include_bias,
            interaction_only,
            n_input_features: None,
            n_output_features: None,
            powers: None,
            _marker: std::marker::PhantomData,
        })
    }

    /// Return the number of output features (available after `fit`).
    pub fn n_output_features(&self) -> Option<usize> {
        self.n_output_features
    }

    /// Enumerate all power tuples up to `self.degree` for `n_feat` features.
    fn enumerate_powers(&self, n_feat: usize) -> Vec<Vec<usize>> {
        let mut result: Vec<Vec<usize>> = Vec::new();

        // Start with degree 0 (bias) if requested.
        if self.include_bias {
            result.push(vec![0; n_feat]);
        }

        // Degree 1 terms.
        for i in 0..n_feat {
            let mut p = vec![0; n_feat];
            p[i] = 1;
            result.push(p);
        }

        // Higher degree terms.
        for d in 2..=self.degree {
            let combos = Self::combinations(n_feat, d);
            for combo in combos {
                if self.interaction_only {
                    // Each feature appears at most once.
                    let max_exp = *combo.iter().max().unwrap_or(&0);
                    if max_exp > 1 {
                        continue;
                    }
                }
                result.push(combo);
            }
        }

        result
    }

    /// Generate all exponent tuples of length `n_feat` that sum to exactly `degree`.
    fn combinations(n_feat: usize, degree: usize) -> Vec<Vec<usize>> {
        let mut results = Vec::new();
        let mut current = vec![0usize; n_feat];
        Self::combinations_recurse(n_feat, degree, 0, &mut current, &mut results);
        results
    }

    fn combinations_recurse(
        n_feat: usize,
        remaining: usize,
        start: usize,
        current: &mut Vec<usize>,
        results: &mut Vec<Vec<usize>>,
    ) {
        if start == n_feat {
            if remaining == 0 {
                results.push(current.clone());
            }
            return;
        }
        if start == n_feat - 1 {
            current[start] = remaining;
            results.push(current.clone());
            current[start] = 0;
            return;
        }
        let max_val = remaining;
        for v in 0..=max_val {
            current[start] = v;
            Self::combinations_recurse(n_feat, remaining - v, start + 1, current, results);
        }
        current[start] = 0;
    }
}

impl<T: Float> Transformer<T> for PolynomialFeatures<T> {
    fn fit(&mut self, x: &Tensor<T>) -> Result<()> {
        let (_n, n_feat) = matrix_shape(x)?;
        let powers = self.enumerate_powers(n_feat);
        self.n_input_features = Some(n_feat);
        self.n_output_features = Some(powers.len());
        self.powers = Some(powers);
        Ok(())
    }

    fn transform(&self, x: &Tensor<T>) -> Result<Tensor<T>> {
        let powers = self.powers.as_ref().ok_or(MlError::NotFitted)?;
        let n_in = self.n_input_features.ok_or(MlError::NotFitted)?;
        let (n_samples, n_feat) = matrix_shape(x)?;
        if n_feat != n_in {
            return Err(MlError::DimensionMismatch {
                expected: n_in,
                got: n_feat,
            });
        }

        let data = x.as_slice();
        let n_out = powers.len();
        let mut out = Vec::with_capacity(n_samples * n_out);

        for row in 0..n_samples {
            let row_start = row * n_feat;
            for power in powers {
                let mut val = T::one();
                for (j, &exp) in power.iter().enumerate() {
                    let xj = data[row_start + j];
                    for _ in 0..exp {
                        val *= xj;
                    }
                }
                out.push(val);
            }
        }

        Tensor::from_vec(out, vec![n_samples, n_out]).map_err(MlError::from)
    }
}

// ---------------------------------------------------------------------------
// FeatureGenerator
// ---------------------------------------------------------------------------

/// Applies a set of nonlinear transforms column-wise, concatenating the
/// generated features alongside the originals.
///
/// # Examples
///
/// ```
/// # use scivex_ml::feature_engineering::{FeatureGenerator, FeatureTransform};
/// # use scivex_ml::traits::Transformer;
/// # use scivex_core::Tensor;
/// let x = Tensor::from_vec(vec![1.0_f64, 4.0, 9.0, 16.0], vec![2, 2]).unwrap();
/// let mut fg = FeatureGenerator::<f64>::new(vec![FeatureTransform::Square]);
/// let out = fg.fit_transform(&x).unwrap();
/// // original 2 cols + 2 squared cols = 4
/// assert_eq!(out.shape()[1], 4);
/// ```
#[derive(Debug, Clone)]
pub struct FeatureGenerator<T: Float> {
    transforms: Vec<FeatureTransform>,
    n_input_features: Option<usize>,
    n_output_features: Option<usize>,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Float> FeatureGenerator<T> {
    /// Create a new generator with the given set of transforms.
    pub fn new(transforms: Vec<FeatureTransform>) -> Self {
        Self {
            transforms,
            n_input_features: None,
            n_output_features: None,
            _marker: std::marker::PhantomData,
        }
    }

    /// Number of output features (available after `fit`).
    pub fn n_output_features(&self) -> Option<usize> {
        self.n_output_features
    }

    /// Compute the number of extra features for a single transform.
    fn extra_features(t: &FeatureTransform, n_feat: usize) -> usize {
        match t {
            FeatureTransform::Square
            | FeatureTransform::Sqrt
            | FeatureTransform::Log
            | FeatureTransform::Reciprocal => n_feat,
            FeatureTransform::Interaction => n_feat * (n_feat - 1) / 2,
        }
    }
}

impl<T: Float> Transformer<T> for FeatureGenerator<T> {
    fn fit(&mut self, x: &Tensor<T>) -> Result<()> {
        let (_n, n_feat) = matrix_shape(x)?;
        let extra: usize = self
            .transforms
            .iter()
            .map(|t| Self::extra_features(t, n_feat))
            .sum();
        self.n_input_features = Some(n_feat);
        self.n_output_features = Some(n_feat + extra);
        Ok(())
    }

    fn transform(&self, x: &Tensor<T>) -> Result<Tensor<T>> {
        let n_in = self.n_input_features.ok_or(MlError::NotFitted)?;
        let n_out = self.n_output_features.ok_or(MlError::NotFitted)?;
        let (n_samples, n_feat) = matrix_shape(x)?;
        if n_feat != n_in {
            return Err(MlError::DimensionMismatch {
                expected: n_in,
                got: n_feat,
            });
        }

        let data = x.as_slice();
        let eps = T::from_f64(1e-12);
        let mut out = Vec::with_capacity(n_samples * n_out);

        for row in 0..n_samples {
            let row_start = row * n_feat;
            let row_slice = &data[row_start..row_start + n_feat];

            // Original features.
            out.extend_from_slice(row_slice);

            // Generated features.
            for t in &self.transforms {
                match t {
                    FeatureTransform::Square => {
                        for &v in row_slice {
                            out.push(v * v);
                        }
                    }
                    FeatureTransform::Sqrt => {
                        for &v in row_slice {
                            out.push(v.abs().sqrt());
                        }
                    }
                    FeatureTransform::Log => {
                        for &v in row_slice {
                            out.push((v.abs() + T::one()).ln());
                        }
                    }
                    FeatureTransform::Reciprocal => {
                        for &v in row_slice {
                            out.push(T::one() / (v.abs() + eps));
                        }
                    }
                    FeatureTransform::Interaction => {
                        for i in 0..n_feat {
                            for j in (i + 1)..n_feat {
                                out.push(row_slice[i] * row_slice[j]);
                            }
                        }
                    }
                }
            }
        }

        Tensor::from_vec(out, vec![n_samples, n_out]).map_err(MlError::from)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn test_polynomial_degree2() {
        // 2 features, degree 2, no bias, not interaction only
        // Terms: x1, x2, x1^2, x1*x2, x2^2 = 5
        let x = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let mut poly = PolynomialFeatures::<f64>::new(2, false, false).unwrap();
        let out = poly.fit_transform(&x).unwrap();
        assert_eq!(out.shape()[1], 5);
        assert_eq!(out.shape()[0], 2);

        let d = out.as_slice();
        // Row 0: x1=1, x2=2
        // Degree-1 terms: x1, x2
        assert!(approx_eq(d[0], 1.0)); // x1
        assert!(approx_eq(d[1], 2.0)); // x2
        // Degree-2 terms ordered by combinations: x2^2, x1*x2, x1^2
        assert!(approx_eq(d[2], 4.0)); // x2^2
        assert!(approx_eq(d[3], 2.0)); // x1*x2
        assert!(approx_eq(d[4], 1.0)); // x1^2
    }

    #[test]
    fn test_polynomial_interaction_only() {
        // 3 features, degree 2, no bias, interaction only
        // Terms: x1, x2, x3, x1*x2, x1*x3, x2*x3 = 6
        let x = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let mut poly = PolynomialFeatures::<f64>::new(2, false, true).unwrap();
        let out = poly.fit_transform(&x).unwrap();
        assert_eq!(out.shape()[1], 6);

        let d = out.as_slice();
        // Row 0: x1=1, x2=2, x3=3
        // Interaction terms ordered by combinations: x2*x3, x1*x3, x1*x2
        assert!(approx_eq(d[3], 6.0)); // x2*x3
        assert!(approx_eq(d[4], 3.0)); // x1*x3
        assert!(approx_eq(d[5], 2.0)); // x1*x2
    }

    #[test]
    fn test_polynomial_with_bias() {
        // 2 features, degree 2, with bias
        // Terms: 1(bias), x1, x2, x1^2, x1*x2, x2^2 = 6
        let x = Tensor::from_vec(vec![2.0_f64, 3.0, 4.0, 5.0], vec![2, 2]).unwrap();
        let mut poly = PolynomialFeatures::<f64>::new(2, true, false).unwrap();
        let out = poly.fit_transform(&x).unwrap();
        assert_eq!(out.shape()[1], 6);

        let d = out.as_slice();
        // Row 0: bias=1, x1=2, x2=3, then degree-2: x2^2=9, x1*x2=6, x1^2=4
        assert!(approx_eq(d[0], 1.0));
        assert!(approx_eq(d[1], 2.0));
        assert!(approx_eq(d[2], 3.0));
        assert!(approx_eq(d[3], 9.0));
        assert!(approx_eq(d[4], 6.0));
        assert!(approx_eq(d[5], 4.0));
    }

    #[test]
    fn test_feature_generator_square_log() {
        let x = Tensor::from_vec(vec![2.0_f64, 3.0, 4.0, 5.0], vec![2, 2]).unwrap();
        let mut fg =
            FeatureGenerator::<f64>::new(vec![FeatureTransform::Square, FeatureTransform::Log]);
        let out = fg.fit_transform(&x).unwrap();
        // 2 original + 2 square + 2 log = 6
        assert_eq!(out.shape()[1], 6);

        let d = out.as_slice();
        // Row 0: orig=[2, 3], square=[4, 9], log=[ln(3), ln(4)]
        assert!(approx_eq(d[0], 2.0));
        assert!(approx_eq(d[1], 3.0));
        assert!(approx_eq(d[2], 4.0));
        assert!(approx_eq(d[3], 9.0));
        assert!(approx_eq(d[4], 3.0_f64.ln()));
        assert!(approx_eq(d[5], 4.0_f64.ln()));
    }

    #[test]
    fn test_feature_generator_interaction() {
        let x = Tensor::from_vec(vec![2.0_f64, 3.0, 5.0, 4.0, 6.0, 7.0], vec![2, 3]).unwrap();
        let mut fg_inter = FeatureGenerator::<f64>::new(vec![FeatureTransform::Interaction]);
        let out = fg_inter.fit_transform(&x).unwrap();
        // 3 original + 3 interaction pairs = 6
        assert_eq!(out.shape()[1], 6);

        let d = out.as_slice();
        // Row 0: orig=[2, 3, 5], interactions: 2*3=6, 2*5=10, 3*5=15
        assert!(approx_eq(d[0], 2.0));
        assert!(approx_eq(d[1], 3.0));
        assert!(approx_eq(d[2], 5.0));
        assert!(approx_eq(d[3], 6.0));
        assert!(approx_eq(d[4], 10.0));
        assert!(approx_eq(d[5], 15.0));
    }
}
