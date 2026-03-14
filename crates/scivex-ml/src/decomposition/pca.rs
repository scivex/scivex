use scivex_core::linalg::decomp::SvdDecomposition;
use scivex_core::{Float, Tensor};

use crate::error::{MlError, Result};
use crate::traits::Transformer;

/// Principal Component Analysis (PCA).
///
/// Projects data onto the top `n_components` directions of maximum variance.
/// Internally computes the SVD of the mean-centred data matrix.
///
/// # Algorithm
/// 1. Centre the data: `X_c = X - mean(X)`
/// 2. Compute SVD: `X_c = U S V^T`
/// 3. The principal components are the first `k` columns of `V`
/// 4. Transform: `X_c * V_k`
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct PCA<T: Float> {
    pub(crate) n_components: usize,
    /// Per-feature mean (length p).
    pub(crate) mean: Option<Vec<T>>,
    /// Principal component axes, shape `(n_components, n_features)` (row-major).
    pub(crate) components: Option<Vec<T>>,
    /// Variance explained by each component.
    pub(crate) explained_variance: Option<Vec<T>>,
    /// Total variance in the data.
    pub(crate) total_variance: Option<T>,
    pub(crate) n_features: usize,
}

impl<T: Float> PCA<T> {
    /// Create a new PCA that retains `n_components` dimensions.
    pub fn new(n_components: usize) -> Result<Self> {
        if n_components == 0 {
            return Err(MlError::InvalidParameter {
                name: "n_components",
                reason: "must be at least 1",
            });
        }
        Ok(Self {
            n_components,
            mean: None,
            components: None,
            explained_variance: None,
            total_variance: None,
            n_features: 0,
        })
    }

    /// Explained variance for each retained component.
    pub fn explained_variance(&self) -> Option<&[T]> {
        self.explained_variance.as_deref()
    }

    /// Ratio of variance explained by each component (sums to ≤ 1).
    pub fn explained_variance_ratio(&self) -> Option<Vec<T>> {
        let ev = self.explained_variance.as_ref()?;
        let total = (*self.total_variance.as_ref()?).max(T::from_f64(1e-15));
        Some(ev.iter().map(|&v| v / total).collect())
    }

    /// The principal axes (rows are components), shape `(n_components, n_features)`.
    pub fn components(&self) -> Option<&[T]> {
        self.components.as_deref()
    }

    /// Inverse transform: reconstruct approximate data from reduced representation.
    pub fn inverse_transform(&self, x_reduced: &Tensor<T>) -> Result<Tensor<T>> {
        let comps = self.components.as_ref().ok_or(MlError::NotFitted)?;
        let mean = self.mean.as_ref().ok_or(MlError::NotFitted)?;
        let s = x_reduced.shape();
        if s.len() != 2 || s[1] != self.n_components {
            return Err(MlError::DimensionMismatch {
                expected: self.n_components,
                got: if s.len() == 2 { s[1] } else { 0 },
            });
        }
        let n = s[0];
        let k = self.n_components;
        let p = self.n_features;
        let rd = x_reduced.as_slice();

        // X_approx = X_reduced * components + mean
        let mut out = vec![T::zero(); n * p];
        for i in 0..n {
            for j in 0..p {
                let mut sum = mean[j];
                for c in 0..k {
                    sum += rd[i * k + c] * comps[c * p + j];
                }
                out[i * p + j] = sum;
            }
        }
        Tensor::from_vec(out, vec![n, p]).map_err(MlError::from)
    }
}

impl<T: Float> Transformer<T> for PCA<T> {
    fn fit(&mut self, x: &Tensor<T>) -> Result<()> {
        let (n, p) = matrix_shape(x)?;
        let k = self.n_components.min(p).min(n);
        self.n_features = p;

        let data = x.as_slice();

        // Compute mean
        let mut mean = vec![T::zero(); p];
        for i in 0..n {
            for j in 0..p {
                mean[j] += data[i * p + j];
            }
        }
        let n_t = T::from_usize(n);
        for m in &mut mean {
            *m /= n_t;
        }

        // Centre the data
        let mut centred = vec![T::zero(); n * p];
        for i in 0..n {
            for j in 0..p {
                centred[i * p + j] = data[i * p + j] - mean[j];
            }
        }

        // SVD of centred data (n x p)
        let centred_tensor = Tensor::from_vec(centred, vec![n, p])?;
        let svd = SvdDecomposition::decompose(&centred_tensor)?;
        let singular_values = svd.singular_values();

        // Components = first k rows of V^T (each row is a principal axis)
        let vt = svd.vt();
        let vt_data = vt.as_slice();
        // vt is (p x p), row-major. First k rows are the top-k right singular vectors.
        let mut components = vec![T::zero(); k * p];
        components.copy_from_slice(&vt_data[..k * p]);

        // Explained variance = s_i^2 / (n - 1)
        let denom = T::from_usize(n.saturating_sub(1).max(1));
        let mut explained = Vec::with_capacity(k);
        let mut total = T::zero();
        for (i, &sv) in singular_values.iter().enumerate() {
            let var = sv * sv / denom;
            if i < k {
                explained.push(var);
            }
            total += var;
        }

        self.mean = Some(mean);
        self.components = Some(components);
        self.explained_variance = Some(explained);
        self.total_variance = Some(total);
        self.n_components = k;
        Ok(())
    }

    fn transform(&self, x: &Tensor<T>) -> Result<Tensor<T>> {
        let comps = self.components.as_ref().ok_or(MlError::NotFitted)?;
        let mean = self.mean.as_ref().ok_or(MlError::NotFitted)?;
        let (n, p) = matrix_shape(x)?;
        if p != self.n_features {
            return Err(MlError::DimensionMismatch {
                expected: self.n_features,
                got: p,
            });
        }
        let data = x.as_slice();
        let k = self.n_components;

        // Project: (X - mean) * V_k^T  =  centred @ components^T
        // components is (k x p), so we compute (n x p) @ (p x k) = (n x k)
        let mut out = vec![T::zero(); n * k];
        for i in 0..n {
            for c in 0..k {
                let mut sum = T::zero();
                for j in 0..p {
                    sum += (data[i * p + j] - mean[j]) * comps[c * p + j];
                }
                out[i * k + c] = sum;
            }
        }
        Tensor::from_vec(out, vec![n, k]).map_err(MlError::from)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pca_reduces_dimensions() {
        // 5 samples, 3 features → reduce to 2
        let x = Tensor::from_vec(
            vec![
                1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
            ],
            vec![5, 3],
        )
        .unwrap();

        let mut pca = PCA::new(2).unwrap();
        let out = pca.fit_transform(&x).unwrap();
        assert_eq!(out.shape(), &[5, 2]);
    }

    #[test]
    fn test_pca_explained_variance() {
        let x = Tensor::from_vec(
            vec![1.0_f64, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0],
            vec![5, 2],
        )
        .unwrap();

        let mut pca = PCA::new(2).unwrap();
        pca.fit(&x).unwrap();

        let ratio = pca.explained_variance_ratio().unwrap();
        // First component should capture ~100% variance (second feature is constant)
        assert!(ratio[0] > 0.99, "first component ratio {}", ratio[0]);
    }

    #[test]
    fn test_pca_inverse_transform() {
        let x = Tensor::from_vec(
            vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            vec![3, 3],
        )
        .unwrap();

        let mut pca = PCA::new(3).unwrap();
        let reduced = pca.fit_transform(&x).unwrap();
        let reconstructed = pca.inverse_transform(&reduced).unwrap();

        // With all components, reconstruction should be near-perfect
        for (&orig, &rec) in x.as_slice().iter().zip(reconstructed.as_slice()) {
            assert!(
                (orig - rec).abs() < 1e-8,
                "orig={orig}, reconstructed={rec}"
            );
        }
    }

    #[test]
    fn test_pca_clamps_components() {
        // Request 5 components from 3-feature data → should clamp to 3
        let x = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();

        let mut pca = PCA::new(5).unwrap();
        let out = pca.fit_transform(&x).unwrap();
        assert_eq!(out.shape()[1], 2); // min(n=2, p=3, k=5) = 2
    }

    #[test]
    fn test_pca_invalid_components() {
        assert!(PCA::<f64>::new(0).is_err());
    }

    #[test]
    fn test_pca_not_fitted() {
        let pca = PCA::<f64>::new(2).unwrap();
        let x = Tensor::from_vec(vec![1.0, 2.0], vec![1, 2]).unwrap();
        assert!(pca.transform(&x).is_err());
    }
}
