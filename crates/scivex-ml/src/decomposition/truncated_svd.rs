use scivex_core::linalg::decomp::SvdDecomposition;
use scivex_core::{Float, Tensor};

use crate::error::{MlError, Result};
use crate::traits::Transformer;

/// Truncated SVD (aka LSA — Latent Semantic Analysis).
///
/// Unlike [`PCA`](super::PCA), this does **not** centre the data before
/// decomposition, making it suitable for sparse or non-negative matrices
/// (e.g. TF-IDF).
///
/// Computes `X ≈ U_k S_k V_k^T` keeping only the top `k` singular values.
/// The transform projects new data onto the `k` right singular vectors.
#[derive(Debug, Clone)]
pub struct TruncatedSVD<T: Float> {
    n_components: usize,
    /// Top-k right singular vectors, shape `(k, p)` row-major.
    components: Option<Vec<T>>,
    /// Top-k singular values.
    singular_values: Option<Vec<T>>,
    /// Explained variance for each component.
    explained_variance: Option<Vec<T>>,
    /// Total variance (sum of all squared singular values / (n-1)).
    total_variance: Option<T>,
    n_features: usize,
}

impl<T: Float> TruncatedSVD<T> {
    /// Create a new TruncatedSVD retaining `n_components` dimensions.
    pub fn new(n_components: usize) -> Result<Self> {
        if n_components == 0 {
            return Err(MlError::InvalidParameter {
                name: "n_components",
                reason: "must be at least 1",
            });
        }
        Ok(Self {
            n_components,
            components: None,
            singular_values: None,
            explained_variance: None,
            total_variance: None,
            n_features: 0,
        })
    }

    /// The top-k singular values.
    pub fn singular_values(&self) -> Option<&[T]> {
        self.singular_values.as_deref()
    }

    /// Explained variance for each component.
    pub fn explained_variance(&self) -> Option<&[T]> {
        self.explained_variance.as_deref()
    }

    /// Ratio of variance explained per component.
    pub fn explained_variance_ratio(&self) -> Option<Vec<T>> {
        let ev = self.explained_variance.as_ref()?;
        let total = (*self.total_variance.as_ref()?).max(T::from_f64(1e-15));
        Some(ev.iter().map(|&v| v / total).collect())
    }

    /// The principal axes (rows), shape `(n_components, n_features)`.
    pub fn components(&self) -> Option<&[T]> {
        self.components.as_deref()
    }

    /// Inverse transform: approximate reconstruction from reduced data.
    pub fn inverse_transform(&self, x_reduced: &Tensor<T>) -> Result<Tensor<T>> {
        let comps = self.components.as_ref().ok_or(MlError::NotFitted)?;
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

        // X_approx = X_reduced * components
        let mut out = vec![T::zero(); n * p];
        for i in 0..n {
            for j in 0..p {
                let mut sum = T::zero();
                for c in 0..k {
                    sum += rd[i * k + c] * comps[c * p + j];
                }
                out[i * p + j] = sum;
            }
        }
        Tensor::from_vec(out, vec![n, p]).map_err(MlError::from)
    }
}

impl<T: Float> Transformer<T> for TruncatedSVD<T> {
    fn fit(&mut self, x: &Tensor<T>) -> Result<()> {
        let (n, p) = matrix_shape(x)?;
        let k = self.n_components.min(p).min(n);
        self.n_features = p;

        let svd = SvdDecomposition::decompose(x)?;
        let sv = svd.singular_values();
        let vt = svd.vt();
        let vt_data = vt.as_slice();

        let mut components = vec![T::zero(); k * p];
        components.copy_from_slice(&vt_data[..k * p]);

        let denom = T::from_usize(n.saturating_sub(1).max(1));
        let mut explained = Vec::with_capacity(k);
        let mut total = T::zero();
        for (i, &s) in sv.iter().enumerate() {
            let var = s * s / denom;
            if i < k {
                explained.push(var);
            }
            total += var;
        }

        self.components = Some(components);
        self.singular_values = Some(sv[..k].to_vec());
        self.explained_variance = Some(explained);
        self.total_variance = Some(total);
        self.n_components = k;
        Ok(())
    }

    fn transform(&self, x: &Tensor<T>) -> Result<Tensor<T>> {
        let comps = self.components.as_ref().ok_or(MlError::NotFitted)?;
        let (n, p) = matrix_shape(x)?;
        if p != self.n_features {
            return Err(MlError::DimensionMismatch {
                expected: self.n_features,
                got: p,
            });
        }
        let data = x.as_slice();
        let k = self.n_components;

        // Project: X * V_k^T → (n x p) @ (p x k) = (n x k)
        let mut out = vec![T::zero(); n * k];
        for i in 0..n {
            for c in 0..k {
                let mut sum = T::zero();
                for j in 0..p {
                    sum += data[i * p + j] * comps[c * p + j];
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
    fn test_truncated_svd_reduces_dims() {
        let x = Tensor::from_vec(
            vec![
                1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            vec![4, 3],
        )
        .unwrap();

        let mut svd = TruncatedSVD::new(2).unwrap();
        let out = svd.fit_transform(&x).unwrap();
        assert_eq!(out.shape(), &[4, 2]);
    }

    #[test]
    fn test_truncated_svd_singular_values() {
        let x = Tensor::from_vec(vec![3.0_f64, 0.0, 0.0, 4.0], vec![2, 2]).unwrap();

        let mut svd = TruncatedSVD::new(2).unwrap();
        svd.fit(&x).unwrap();
        let sv = svd.singular_values().unwrap();
        // diag(3,4) → singular values 4, 3
        assert!((sv[0] - 4.0).abs() < 1e-8);
        assert!((sv[1] - 3.0).abs() < 1e-8);
    }

    #[test]
    fn test_truncated_svd_inverse() {
        let x = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();

        let mut svd = TruncatedSVD::new(2).unwrap();
        let reduced = svd.fit_transform(&x).unwrap();
        let recon = svd.inverse_transform(&reduced).unwrap();

        for (&o, &r) in x.as_slice().iter().zip(recon.as_slice()) {
            assert!((o - r).abs() < 1e-6, "orig={o}, recon={r}");
        }
    }

    #[test]
    fn test_truncated_svd_not_fitted() {
        let svd = TruncatedSVD::<f64>::new(2).unwrap();
        let x = Tensor::from_vec(vec![1.0, 2.0], vec![1, 2]).unwrap();
        assert!(svd.transform(&x).is_err());
    }

    #[test]
    fn test_truncated_svd_no_centering() {
        // TruncatedSVD should NOT center data (unlike PCA)
        // Transform of data with large offset should differ from PCA
        let x = Tensor::from_vec(
            vec![100.0_f64, 200.0, 101.0, 201.0, 102.0, 202.0],
            vec![3, 2],
        )
        .unwrap();

        let mut svd = TruncatedSVD::new(1).unwrap();
        let out = svd.fit_transform(&x).unwrap();
        // Values should be large (not centered around 0)
        for &v in out.as_slice() {
            assert!(v.abs() > 10.0, "expected large values, got {v}");
        }
    }
}
