//! DBSCAN (Density-Based Spatial Clustering of Applications with Noise).

use scivex_core::{Float, Tensor};

use crate::error::{MlError, Result};

/// DBSCAN clustering — finds core samples of high density and expands clusters
/// from them. Points that lie in sparse regions are labelled as noise (-1).
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct DBSCAN<T: Float> {
    eps: T,
    min_samples: usize,
    labels: Option<Vec<i64>>,
    n_clusters: Option<usize>,
}

impl<T: Float> DBSCAN<T> {
    /// Create a new DBSCAN instance.
    ///
    /// - `eps`: maximum distance between two samples in the same neighbourhood
    /// - `min_samples`: minimum number of points to form a dense region (core point)
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_core::Tensor;
    /// # use scivex_ml::cluster::DBSCAN;
    /// let x = Tensor::from_vec(
    ///     vec![0.0_f64, 0.0, 0.1, 0.1, 10.0, 10.0, 10.1, 10.1],
    ///     vec![4, 2],
    /// ).unwrap();
    /// let mut db = DBSCAN::new(1.0, 2).unwrap();
    /// db.fit(&x).unwrap();
    /// assert_eq!(db.n_clusters(), Some(2));
    /// ```
    pub fn new(eps: T, min_samples: usize) -> Result<Self> {
        if eps <= T::zero() {
            return Err(MlError::InvalidParameter {
                name: "eps",
                reason: "must be positive",
            });
        }
        if min_samples == 0 {
            return Err(MlError::InvalidParameter {
                name: "min_samples",
                reason: "must be at least 1",
            });
        }
        Ok(Self {
            eps,
            min_samples,
            labels: None,
            n_clusters: None,
        })
    }

    /// Fit the DBSCAN model on data `x` (shape `[n_samples, n_features]`).
    pub fn fit(&mut self, x: &Tensor<T>) -> Result<()> {
        let shape = x.shape();
        if shape.len() != 2 {
            return Err(MlError::DimensionMismatch {
                expected: 2,
                got: shape.len(),
            });
        }
        let n = shape[0];
        let p = shape[1];
        if n == 0 {
            return Err(MlError::EmptyInput);
        }
        let data = x.as_slice();

        // Pre-compute pairwise distance matrix and neighbourhood lists.
        let eps_sq = self.eps * self.eps;
        let mut neighbours: Vec<Vec<usize>> = Vec::with_capacity(n);
        for i in 0..n {
            let mut nbrs = Vec::new();
            for j in 0..n {
                let dist_sq =
                    euclidean_dist_sq(&data[i * p..(i + 1) * p], &data[j * p..(j + 1) * p]);
                if dist_sq <= eps_sq {
                    nbrs.push(j);
                }
            }
            neighbours.push(nbrs);
        }

        // Classify points: core if |neighbours| >= min_samples
        let is_core: Vec<bool> = neighbours
            .iter()
            .map(|nbrs| nbrs.len() >= self.min_samples)
            .collect();

        // DBSCAN expansion
        let mut labels = vec![-1_i64; n];
        let mut cluster_id: i64 = 0;

        for i in 0..n {
            if labels[i] != -1 || !is_core[i] {
                continue;
            }
            // Start a new cluster from core point i
            let mut stack = vec![i];
            labels[i] = cluster_id;

            while let Some(q) = stack.pop() {
                for &nb in &neighbours[q] {
                    if labels[nb] == -1 {
                        labels[nb] = cluster_id;
                        if is_core[nb] {
                            stack.push(nb);
                        }
                    }
                }
            }
            cluster_id += 1;
        }

        self.labels = Some(labels);
        self.n_clusters = Some(cluster_id as usize);
        Ok(())
    }

    /// Return the cluster labels assigned by `fit`. Noise points have label -1.
    pub fn labels(&self) -> Option<&[i64]> {
        self.labels.as_deref()
    }

    /// Return the number of clusters found (excluding noise).
    pub fn n_clusters(&self) -> Option<usize> {
        self.n_clusters
    }

    /// Predict is not directly supported by DBSCAN — use `labels()` after `fit()`.
    /// This method runs `fit` and returns labels as `Tensor<T>`.
    pub fn fit_predict(&mut self, x: &Tensor<T>) -> Result<Tensor<T>> {
        self.fit(x)?;
        let labels = self.labels.as_ref().expect("fit succeeded");
        let float_labels: Vec<T> = labels.iter().map(|&l| T::from_f64(l as f64)).collect();
        Ok(Tensor::from_vec(float_labels, vec![labels.len()])?)
    }

    /// Return core sample indices (points with at least `min_samples` neighbours).
    pub fn core_sample_indices(&self, x: &Tensor<T>) -> Result<Vec<usize>> {
        let shape = x.shape();
        if shape.len() != 2 {
            return Err(MlError::DimensionMismatch {
                expected: 2,
                got: shape.len(),
            });
        }
        let n = shape[0];
        let p = shape[1];
        let data = x.as_slice();
        let eps_sq = self.eps * self.eps;

        let mut cores = Vec::new();
        for i in 0..n {
            let mut count = 0;
            for j in 0..n {
                let dist_sq =
                    euclidean_dist_sq(&data[i * p..(i + 1) * p], &data[j * p..(j + 1) * p]);
                if dist_sq <= eps_sq {
                    count += 1;
                }
            }
            if count >= self.min_samples {
                cores.push(i);
            }
        }
        Ok(cores)
    }
}

/// Squared Euclidean distance between two points.
fn euclidean_dist_sq<T: Float>(a: &[T], b: &[T]) -> T {
    a.iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| {
            let d = ai - bi;
            d * d
        })
        .fold(T::zero(), |acc, x| acc + x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dbscan_two_clusters() {
        #[rustfmt::skip]
        let x = Tensor::from_vec(
            vec![
                0.0_f64, 0.0,
                0.1, 0.1,
                0.0, 0.1,
                10.0, 10.0,
                10.1, 10.1,
                10.0, 10.1,
            ],
            vec![6, 2],
        ).unwrap();

        let mut db = DBSCAN::new(1.0, 2).unwrap();
        db.fit(&x).unwrap();
        assert_eq!(db.n_clusters(), Some(2));

        let labels = db.labels().unwrap();
        // First three points share a label
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
        // Last three share a different label
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[4], labels[5]);
        // Different clusters
        assert_ne!(labels[0], labels[3]);
    }

    #[test]
    fn test_dbscan_noise_points() {
        #[rustfmt::skip]
        let x = Tensor::from_vec(
            vec![
                0.0_f64, 0.0,
                0.1, 0.0,
                0.0, 0.1,
                100.0, 100.0, // noise point — far away
            ],
            vec![4, 2],
        ).unwrap();

        let mut db = DBSCAN::new(0.5, 2).unwrap();
        db.fit(&x).unwrap();
        let labels = db.labels().unwrap();
        // First three form a cluster
        assert!(labels[0] >= 0);
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
        // Last point is noise
        assert_eq!(labels[3], -1);
    }

    #[test]
    fn test_dbscan_all_noise() {
        // Points too far apart with min_samples=3
        let x = Tensor::from_vec(vec![0.0_f64, 0.0, 10.0, 10.0, 20.0, 20.0], vec![3, 2]).unwrap();

        let mut db = DBSCAN::new(1.0, 3).unwrap();
        db.fit(&x).unwrap();
        assert_eq!(db.n_clusters(), Some(0));
        let labels = db.labels().unwrap();
        assert!(labels.iter().all(|&l| l == -1));
    }

    #[test]
    fn test_dbscan_fit_predict() {
        let x = Tensor::from_vec(
            vec![0.0_f64, 0.0, 0.1, 0.1, 10.0, 10.0, 10.1, 10.1],
            vec![4, 2],
        )
        .unwrap();

        let mut db = DBSCAN::new(1.0, 2).unwrap();
        let labels = db.fit_predict(&x).unwrap();
        assert_eq!(labels.shape(), &[4]);
    }

    #[test]
    fn test_dbscan_invalid_params() {
        assert!(DBSCAN::<f64>::new(0.0, 2).is_err());
        assert!(DBSCAN::<f64>::new(-1.0, 2).is_err());
        assert!(DBSCAN::<f64>::new(1.0, 0).is_err());
    }

    #[test]
    fn test_dbscan_core_samples() {
        let x = Tensor::from_vec(
            vec![0.0_f64, 0.0, 0.1, 0.0, 0.0, 0.1, 100.0, 100.0],
            vec![4, 2],
        )
        .unwrap();

        let mut db = DBSCAN::new(0.5, 2).unwrap();
        db.fit(&x).unwrap();
        let cores = db.core_sample_indices(&x).unwrap();
        // First three points are cores; point at 100,100 is not
        assert_eq!(cores.len(), 3);
        assert!(!cores.contains(&3));
    }
}
