//! Online (mini-batch) K-Means clustering.

use scivex_core::{Float, Tensor, random::Rng};

use crate::error::{MlError, Result};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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

fn nearest_centroid<T: Float>(point: &[T], centroids: &[T], k: usize, p: usize) -> usize {
    let mut best = 0;
    let mut best_dist = T::infinity();
    for c in 0..k {
        let mut dist = T::zero();
        for j in 0..p {
            let d = point[j] - centroids[c * p + j];
            dist += d * d;
        }
        if dist < best_dist {
            best_dist = dist;
            best = c;
        }
    }
    best
}

// ---------------------------------------------------------------------------
// OnlineKMeans
// ---------------------------------------------------------------------------

/// Online (mini-batch) K-Means clustering.
///
/// Centroids are initialised from the first batch and then refined
/// incrementally with each subsequent call to [`partial_fit`](Self::partial_fit).
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct OnlineKMeans<T: Float> {
    n_clusters: usize,
    seed: u64,
    centroids: Option<Vec<T>>,
    counts: Vec<usize>,
    n_features: usize,
    n_samples_seen: usize,
}

impl<T: Float> OnlineKMeans<T> {
    /// Create a new `OnlineKMeans` with the given number of clusters.
    ///
    /// # Errors
    ///
    /// Returns [`MlError::InvalidParameter`] if `n_clusters` is zero.
    pub fn new(n_clusters: usize, seed: u64) -> Result<Self> {
        if n_clusters == 0 {
            return Err(MlError::InvalidParameter {
                name: "n_clusters",
                reason: "must be at least 1",
            });
        }
        Ok(Self {
            n_clusters,
            seed,
            centroids: None,
            counts: vec![0; n_clusters],
            n_features: 0,
            n_samples_seen: 0,
        })
    }

    /// Update centroids with a new batch of data `x` (shape `[n_samples, n_features]`).
    ///
    /// On the first call, centroids are initialised by randomly selecting
    /// `n_clusters` points from the batch.
    pub fn partial_fit(&mut self, x: &Tensor<T>) -> Result<()> {
        let (n, p) = matrix_shape(x)?;
        let data = x.as_slice();

        // Initialise centroids from first batch if needed.
        if self.centroids.is_none() {
            if n < self.n_clusters {
                return Err(MlError::InvalidParameter {
                    name: "n_clusters",
                    reason: "first batch must have at least n_clusters samples",
                });
            }
            self.n_features = p;
            let mut rng = Rng::new(self.seed);
            let mut centroids = vec![T::zero(); self.n_clusters * p];
            let mut chosen: Vec<usize> = Vec::with_capacity(self.n_clusters);
            while chosen.len() < self.n_clusters {
                let idx = (rng.next_f64() * n as f64) as usize % n;
                if !chosen.contains(&idx) {
                    chosen.push(idx);
                }
            }
            for (k, &idx) in chosen.iter().enumerate() {
                centroids[k * p..(k + 1) * p].copy_from_slice(&data[idx * p..(idx + 1) * p]);
            }
            self.centroids = Some(centroids);
            self.counts = vec![0; self.n_clusters];
        }

        if p != self.n_features {
            return Err(MlError::DimensionMismatch {
                expected: self.n_features,
                got: p,
            });
        }

        let centroids = self.centroids.as_mut().expect("centroids initialised above");

        // For each sample, find nearest centroid and update with running mean.
        for i in 0..n {
            let row = &data[i * p..(i + 1) * p];
            let c = nearest_centroid(row, centroids, self.n_clusters, p);
            self.counts[c] += 1;
            let lr = T::one() / T::from_usize(self.counts[c]);
            for j in 0..p {
                let diff = lr * (row[j] - centroids[c * p + j]);
                centroids[c * p + j] += diff;
            }
        }

        self.n_samples_seen += n;
        Ok(())
    }

    /// Assign each sample in `x` to the nearest centroid.
    pub fn predict(&self, x: &Tensor<T>) -> Result<Tensor<T>> {
        let centroids = self.centroids.as_ref().ok_or(MlError::NotFitted)?;
        let (n, p) = matrix_shape(x)?;
        if p != self.n_features {
            return Err(MlError::DimensionMismatch {
                expected: self.n_features,
                got: p,
            });
        }
        let data = x.as_slice();
        let mut labels = vec![T::zero(); n];
        for i in 0..n {
            labels[i] = T::from_usize(nearest_centroid(
                &data[i * p..(i + 1) * p],
                centroids,
                self.n_clusters,
                p,
            ));
        }
        Ok(Tensor::from_vec(labels, vec![n])?)
    }

    /// The current centroid coordinates (flattened, `[n_clusters * n_features]`).
    pub fn centroids(&self) -> Option<&[T]> {
        self.centroids.as_deref()
    }

    /// Total number of training samples seen across all batches.
    pub fn n_samples_seen(&self) -> usize {
        self.n_samples_seen
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_cluster_data() -> Tensor<f64> {
        Tensor::from_vec(
            vec![
                0.0, 0.0, 0.1, 0.1, -0.1, 0.0, 0.0, 0.1, 10.0, 10.0, 10.1, 10.1, 9.9, 10.0,
                10.0, 9.9,
            ],
            vec![8, 2],
        )
        .unwrap()
    }

    #[test]
    fn test_online_kmeans_basic() {
        let x = two_cluster_data();
        let mut km = OnlineKMeans::<f64>::new(2, 42).unwrap();
        km.partial_fit(&x).unwrap();
        let labels = km.predict(&x).unwrap();
        let l = labels.as_slice();
        // First four should share a label, last four another.
        for i in 1..4 {
            assert!(
                (l[0] - l[i]).abs() < 0.5,
                "points in cluster 0 should share label"
            );
        }
        for i in 5..8 {
            assert!(
                (l[4] - l[i]).abs() < 0.5,
                "points in cluster 1 should share label"
            );
        }
        assert!(
            (l[0] - l[4]).abs() > 0.5,
            "clusters should have different labels"
        );
    }

    #[test]
    fn test_online_kmeans_multiple_batches() {
        let mut km = OnlineKMeans::<f64>::new(2, 42).unwrap();

        let batch1 = Tensor::from_vec(
            vec![0.0, 0.0, 0.1, 0.1, 10.0, 10.0, 10.1, 10.1],
            vec![4, 2],
        )
        .unwrap();
        let batch2 = Tensor::from_vec(
            vec![-0.1, 0.0, 0.0, -0.1, 9.9, 10.0, 10.0, 9.9],
            vec![4, 2],
        )
        .unwrap();
        let batch3 = Tensor::from_vec(
            vec![0.05, 0.05, -0.05, -0.05, 10.05, 10.05, 9.95, 9.95],
            vec![4, 2],
        )
        .unwrap();

        km.partial_fit(&batch1).unwrap();
        km.partial_fit(&batch2).unwrap();
        km.partial_fit(&batch3).unwrap();

        assert_eq!(km.n_samples_seen(), 12);

        let x_test = Tensor::from_vec(vec![0.0, 0.0, 10.0, 10.0], vec![2, 2]).unwrap();
        let labels = km.predict(&x_test).unwrap();
        let l = labels.as_slice();
        assert!(
            (l[0] - l[1]).abs() > 0.5,
            "clusters should have different labels"
        );
    }

    #[test]
    fn test_online_kmeans_not_fitted() {
        let km = OnlineKMeans::<f64>::new(2, 42).unwrap();
        let x = Tensor::from_vec(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let result = km.predict(&x);
        assert!(result.is_err());
    }

    #[test]
    fn test_online_kmeans_centroids() {
        let mut km = OnlineKMeans::<f64>::new(2, 42).unwrap();
        // Feed lots of data so centroids converge near true centres (0,0) and (10,10).
        for _ in 0..20 {
            let x = two_cluster_data();
            km.partial_fit(&x).unwrap();
        }
        let c = km.centroids().unwrap();
        // Two centroids, each 2-D, flattened.
        assert_eq!(c.len(), 4);

        // Sort centroids by first coordinate to identify them.
        let (c0, c1) = if c[0] < c[2] {
            (&c[0..2], &c[2..4])
        } else {
            (&c[2..4], &c[0..2])
        };

        // Centroid near (0, 0)
        assert!(c0[0].abs() < 0.5, "centroid0[0] = {}", c0[0]);
        assert!(c0[1].abs() < 0.5, "centroid0[1] = {}", c0[1]);
        // Centroid near (10, 10)
        assert!((c1[0] - 10.0).abs() < 0.5, "centroid1[0] = {}", c1[0]);
        assert!((c1[1] - 10.0).abs() < 0.5, "centroid1[1] = {}", c1[1]);
    }
}
