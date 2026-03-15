use scivex_core::{Float, Tensor};

use crate::error::{MlError, Result};
use crate::traits::{Classifier, Predictor};

/// Gaussian Naive Bayes classifier.
///
/// Assumes features are normally distributed within each class.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct GaussianNB<T: Float> {
    /// Per-class prior probabilities, shape `[n_classes]`.
    class_priors: Option<Vec<T>>,
    /// Per-class, per-feature means, shape `[n_classes][n_features]`.
    class_means: Option<Vec<Vec<T>>>,
    /// Per-class, per-feature variances, shape `[n_classes][n_features]`.
    class_vars: Option<Vec<Vec<T>>>,
    /// Sorted unique classes.
    classes: Option<Vec<T>>,
    n_features: usize,
}

impl<T: Float> Default for GaussianNB<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float> GaussianNB<T> {
    /// Create a new, unfitted Gaussian Naive Bayes classifier.
    pub fn new() -> Self {
        Self {
            class_priors: None,
            class_means: None,
            class_vars: None,
            classes: None,
            n_features: 0,
        }
    }

    /// Returns the sorted unique class labels learned during fitting.
    pub fn classes(&self) -> Option<&[T]> {
        self.classes.as_deref()
    }
}

impl<T: Float> Predictor<T> for GaussianNB<T> {
    fn fit(&mut self, x: &Tensor<T>, y: &Tensor<T>) -> Result<()> {
        let (n, p) = matrix_shape(x)?;
        check_y(y, n)?;
        let x_data = x.as_slice();
        let y_data = y.as_slice();

        // Find unique classes
        let mut classes: Vec<T> = Vec::new();
        for &v in y_data {
            if !classes.iter().any(|&c| (c - v).abs() < T::epsilon()) {
                classes.push(v);
            }
        }
        classes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n_classes = classes.len();

        let mut priors = vec![T::zero(); n_classes];
        let mut means = vec![vec![T::zero(); p]; n_classes];
        let mut vars = vec![vec![T::zero(); p]; n_classes];
        let mut counts = vec![0usize; n_classes];

        // Accumulate sums
        for i in 0..n {
            let ci = classes
                .iter()
                .position(|&c| (c - y_data[i]).abs() < T::epsilon())
                .unwrap_or(0);
            counts[ci] += 1;
            for j in 0..p {
                means[ci][j] += x_data[i * p + j];
            }
        }

        // Compute means
        for c in 0..n_classes {
            let nc = T::from_usize(counts[c]);
            for m in &mut means[c] {
                *m /= nc;
            }
        }

        // Compute variances
        for i in 0..n {
            let ci = classes
                .iter()
                .position(|&c| (c - y_data[i]).abs() < T::epsilon())
                .unwrap_or(0);
            for j in 0..p {
                let d = x_data[i * p + j] - means[ci][j];
                vars[ci][j] += d * d;
            }
        }
        for c in 0..n_classes {
            let nc = T::from_usize(counts[c]);
            for v in &mut vars[c] {
                *v /= nc;
                // Add a small epsilon to avoid division by zero
                if *v < T::epsilon() {
                    *v = T::epsilon();
                }
            }
        }

        // Priors
        let nf = T::from_usize(n);
        for c in 0..n_classes {
            priors[c] = T::from_usize(counts[c]) / nf;
        }

        self.class_priors = Some(priors);
        self.class_means = Some(means);
        self.class_vars = Some(vars);
        self.classes = Some(classes);
        self.n_features = p;
        Ok(())
    }

    fn predict(&self, x: &Tensor<T>) -> Result<Tensor<T>> {
        let proba = self.predict_proba(x)?;
        let (n, n_classes) = {
            let s = proba.shape();
            (s[0], s[1])
        };
        let proba_data = proba.as_slice();
        let classes = self.classes.as_ref().ok_or(MlError::NotFitted)?;
        let mut out = vec![T::zero(); n];
        for i in 0..n {
            let mut best_c = 0;
            let mut best_p = T::neg_infinity();
            for c in 0..n_classes {
                let p = proba_data[i * n_classes + c];
                if p > best_p {
                    best_p = p;
                    best_c = c;
                }
            }
            out[i] = classes[best_c];
        }
        Ok(Tensor::from_vec(out, vec![n])?)
    }
}

impl<T: Float> Classifier<T> for GaussianNB<T> {
    fn predict_proba(&self, x: &Tensor<T>) -> Result<Tensor<T>> {
        let priors = self.class_priors.as_ref().ok_or(MlError::NotFitted)?;
        let means = self.class_means.as_ref().ok_or(MlError::NotFitted)?;
        let vars = self.class_vars.as_ref().ok_or(MlError::NotFitted)?;
        let (n, p) = matrix_shape(x)?;
        if p != self.n_features {
            return Err(MlError::DimensionMismatch {
                expected: self.n_features,
                got: p,
            });
        }
        let n_classes = priors.len();
        let data = x.as_slice();
        let two = T::from_usize(2);
        let two_pi = two * T::pi();

        let mut out = vec![T::zero(); n * n_classes];
        for i in 0..n {
            // Compute log-likelihood for each class
            let mut log_probs = vec![T::zero(); n_classes];
            for c in 0..n_classes {
                let mut log_p = priors[c].ln();
                for j in 0..p {
                    let xij = data[i * p + j];
                    let mu = means[c][j];
                    let var = vars[c][j];
                    // log of Gaussian PDF
                    let d = xij - mu;
                    log_p -= (two_pi * var).ln() / two + d * d / (two * var);
                }
                log_probs[c] = log_p;
            }
            // Convert to probabilities via log-sum-exp
            let max_lp = log_probs.iter().copied().fold(T::neg_infinity(), T::max);
            let sum_exp: T = log_probs
                .iter()
                .map(|&lp| (lp - max_lp).exp())
                .fold(T::zero(), |a, b| a + b);
            let log_sum = max_lp + sum_exp.ln();
            for c in 0..n_classes {
                out[i * n_classes + c] = (log_probs[c] - log_sum).exp();
            }
        }
        Ok(Tensor::from_vec(out, vec![n, n_classes])?)
    }
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

fn check_y<T: Float>(y: &Tensor<T>, n: usize) -> Result<()> {
    if y.ndim() != 1 || y.shape()[0] != n {
        return Err(MlError::DimensionMismatch {
            expected: n,
            got: y.shape()[0],
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_nb_simple() {
        // Two classes with clearly separated Gaussian distributions
        let x = Tensor::from_vec(
            vec![
                -1.0_f64, -1.0, -0.5, -0.5, -1.5, -1.5, 1.0, 1.0, 0.5, 0.5, 1.5, 1.5,
            ],
            vec![6, 2],
        )
        .unwrap();
        let y = Tensor::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0], vec![6]).unwrap();

        let mut nb = GaussianNB::new();
        nb.fit(&x, &y).unwrap();

        let preds = nb.predict(&x).unwrap();
        assert_eq!(preds.as_slice(), y.as_slice());
    }

    #[test]
    fn test_predict_proba_sums_to_one() {
        let x = Tensor::from_vec(vec![0.0_f64, 1.0, 2.0, 3.0], vec![2, 2]).unwrap();
        let y = Tensor::from_vec(vec![0.0, 1.0], vec![2]).unwrap();

        let mut nb = GaussianNB::new();
        nb.fit(&x, &y).unwrap();

        let proba = nb.predict_proba(&x).unwrap();
        let p = proba.as_slice();
        // Each row should sum to ~1
        assert!((p[0] + p[1] - 1.0).abs() < 1e-10);
        assert!((p[2] + p[3] - 1.0).abs() < 1e-10);
    }
}
