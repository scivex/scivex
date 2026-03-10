use scivex_core::{Float, Tensor, random::Rng};

use crate::error::{MlError, Result};

/// K-Means clustering using Lloyd's algorithm.
#[derive(Debug, Clone)]
pub struct KMeans<T: Float> {
    n_clusters: usize,
    max_iter: usize,
    tol: T,
    n_init: usize,
    seed: u64,
    centroids: Option<Vec<T>>,
    n_features: usize,
    inertia: Option<T>,
}

impl<T: Float> KMeans<T> {
    /// Create a new `KMeans` with the given number of clusters and hyper-parameters.
    pub fn new(
        n_clusters: usize,
        max_iter: usize,
        tol: T,
        n_init: usize,
        seed: u64,
    ) -> Result<Self> {
        if n_clusters == 0 {
            return Err(MlError::InvalidParameter {
                name: "n_clusters",
                reason: "must be at least 1",
            });
        }
        Ok(Self {
            n_clusters,
            max_iter,
            tol,
            n_init,
            seed,
            centroids: None,
            n_features: 0,
            inertia: None,
        })
    }

    /// The sum of squared distances of samples to their closest cluster centre.
    pub fn inertia(&self) -> Option<T> {
        self.inertia
    }

    /// Fitted cluster centroids, shape `[n_clusters, n_features]` (flattened).
    pub fn centroids(&self) -> Option<&[T]> {
        self.centroids.as_deref()
    }

    /// Fit the model to data `x` (shape `[n_samples, n_features]`).
    pub fn fit(&mut self, x: &Tensor<T>) -> Result<()> {
        let (n, p) = matrix_shape(x)?;
        if n < self.n_clusters {
            return Err(MlError::InvalidParameter {
                name: "n_clusters",
                reason: "more clusters than samples",
            });
        }
        let data = x.as_slice();
        let mut rng = Rng::new(self.seed);

        let mut best_centroids: Option<Vec<T>> = None;
        let mut best_inertia = T::infinity();

        for _ in 0..self.n_init {
            let (centroids, inertia) = self.run_once(data, n, p, &mut rng);
            if inertia < best_inertia {
                best_inertia = inertia;
                best_centroids = Some(centroids);
            }
        }

        self.centroids = best_centroids;
        self.n_features = p;
        self.inertia = Some(best_inertia);
        Ok(())
    }

    /// Predict cluster labels for `x`.
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

    fn run_once(&self, data: &[T], n: usize, p: usize, rng: &mut Rng) -> (Vec<T>, T) {
        // Random initialisation: pick n_clusters random points
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

        let mut labels = vec![0usize; n];

        for _iter in 0..self.max_iter {
            // Assign
            for i in 0..n {
                labels[i] =
                    nearest_centroid(&data[i * p..(i + 1) * p], &centroids, self.n_clusters, p);
            }

            // Update centroids
            let mut new_centroids = vec![T::zero(); self.n_clusters * p];
            let mut counts = vec![0usize; self.n_clusters];
            for i in 0..n {
                let k = labels[i];
                counts[k] += 1;
                for j in 0..p {
                    new_centroids[k * p + j] += data[i * p + j];
                }
            }
            for k in 0..self.n_clusters {
                if counts[k] > 0 {
                    let nk = T::from_usize(counts[k]);
                    for j in 0..p {
                        new_centroids[k * p + j] /= nk;
                    }
                } else {
                    // Empty cluster: keep old centroid
                    new_centroids[k * p..(k + 1) * p]
                        .copy_from_slice(&centroids[k * p..(k + 1) * p]);
                }
            }

            // Check convergence
            let shift: T = centroids
                .iter()
                .zip(&new_centroids)
                .map(|(&a, &b)| {
                    let d = a - b;
                    d * d
                })
                .fold(T::zero(), |a, b| a + b)
                .sqrt();

            centroids = new_centroids;
            if shift < self.tol {
                break;
            }
        }

        // Compute inertia
        let mut inertia = T::zero();
        for i in 0..n {
            let k = labels[i];
            for j in 0..p {
                let d = data[i * p + j] - centroids[k * p + j];
                inertia += d * d;
            }
        }

        (centroids, inertia)
    }
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
    fn test_kmeans_two_clusters() {
        // Two well-separated clusters
        let x = Tensor::from_vec(
            vec![
                0.0_f64, 0.0, 0.1, 0.1, -0.1, 0.0, 10.0, 10.0, 10.1, 10.1, 9.9, 10.0,
            ],
            vec![6, 2],
        )
        .unwrap();

        let mut km = KMeans::new(2, 100, 1e-6, 5, 42).unwrap();
        km.fit(&x).unwrap();

        let labels = km.predict(&x).unwrap();
        let l = labels.as_slice();
        // First three should be one cluster, last three another
        assert!(
            ((l[0] - l[1]).abs() < 0.5)
                && ((l[1] - l[2]).abs() < 0.5)
                && ((l[3] - l[4]).abs() < 0.5)
                && ((l[4] - l[5]).abs() < 0.5),
            "points in same cluster should have same label"
        );
        assert!(
            (l[0] - l[3]).abs() > 0.5,
            "different clusters should have different labels"
        );
    }

    #[test]
    fn test_inertia_decreases() {
        let x =
            Tensor::from_vec(vec![0.0_f64, 0.0, 1.0, 1.0, 5.0, 5.0, 6.0, 6.0], vec![4, 2]).unwrap();

        let mut km1 = KMeans::new(1, 100, 1e-6, 1, 42).unwrap();
        km1.fit(&x).unwrap();
        let inertia1 = km1.inertia().unwrap();

        let mut km2 = KMeans::new(2, 100, 1e-6, 3, 42).unwrap();
        km2.fit(&x).unwrap();
        let inertia2 = km2.inertia().unwrap();

        assert!(
            inertia2 < inertia1,
            "2 clusters should have lower inertia than 1"
        );
    }
}
