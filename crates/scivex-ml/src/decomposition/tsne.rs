use scivex_core::random::Rng;
use scivex_core::{Float, Tensor};

use crate::error::{MlError, Result};

/// t-Distributed Stochastic Neighbor Embedding (t-SNE).
///
/// Non-linear dimensionality reduction that preserves local structure,
/// commonly used for visualising high-dimensional data in 2-D or 3-D.
///
/// This is the **exact** (non-Barnes-Hut) O(n²) algorithm, suitable for
/// datasets up to a few thousand points.
///
/// # Algorithm
/// 1. Compute pairwise affinities `p_{ij}` from Gaussian kernels (symmetric SNE)
/// 2. Initialise low-dimensional embedding randomly
/// 3. Gradient descent to minimise the KL divergence between the high-dimensional
///    affinities `p_{ij}` and low-dimensional Student-t affinities `q_{ij}`
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct TSNE<T: Float> {
    n_components: usize,
    perplexity: f64,
    learning_rate: f64,
    n_iter: usize,
    seed: u64,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Float> TSNE<T> {
    /// Create a new t-SNE with default parameters.
    ///
    /// - `n_components`: output dimensionality (typically 2)
    /// - `perplexity`: effective number of neighbours (typically 5–50)
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_ml::decomposition::TSNE;
    /// let tsne = TSNE::<f64>::new(2, 30.0).unwrap();
    /// ```
    pub fn new(n_components: usize, perplexity: f64) -> Result<Self> {
        if n_components == 0 {
            return Err(MlError::InvalidParameter {
                name: "n_components",
                reason: "must be at least 1",
            });
        }
        if perplexity <= 0.0 {
            return Err(MlError::InvalidParameter {
                name: "perplexity",
                reason: "must be positive",
            });
        }
        Ok(Self {
            n_components,
            perplexity,
            learning_rate: 200.0,
            n_iter: 1000,
            seed: 42,
            _marker: std::marker::PhantomData,
        })
    }

    /// Set the learning rate (default 200).
    pub fn set_learning_rate(&mut self, lr: f64) -> &mut Self {
        self.learning_rate = lr.max(1.0);
        self
    }

    /// Set the number of iterations (default 1000).
    pub fn set_n_iter(&mut self, n_iter: usize) -> &mut Self {
        self.n_iter = n_iter.max(1);
        self
    }

    /// Set the random seed.
    pub fn set_seed(&mut self, seed: u64) -> &mut Self {
        self.seed = seed;
        self
    }

    /// Fit and transform: compute the low-dimensional embedding.
    ///
    /// Input `x` has shape `[n_samples, n_features]`.
    /// Returns a tensor of shape `[n_samples, n_components]`.
    pub fn fit_transform(&self, x: &Tensor<T>) -> Result<Tensor<T>> {
        let (n, p) = matrix_shape(x)?;
        if n < 2 {
            return Err(MlError::InvalidParameter {
                name: "n_samples",
                reason: "need at least 2 samples",
            });
        }
        let data = x.as_slice();
        let d = self.n_components;
        let lr = self.learning_rate;

        // 1. Pairwise squared distances (high-dimensional)
        let dist_sq = pairwise_sq_distances(data, n, p);

        // 2. Compute joint probabilities P (symmetric)
        let p_joint = compute_joint_probabilities(&dist_sq, n, self.perplexity);

        // 3. Initialise embedding with small random values
        let mut rng = Rng::new(self.seed);
        let mut y = vec![0.0_f64; n * d];
        for v in &mut y {
            *v = rng.next_normal_f64() * 1e-4;
        }
        let mut y_prev = y.clone();
        let momentum_init = 0.5;
        let momentum_final = 0.8;
        let momentum_switch = 250;

        // 4. Gradient descent with momentum
        for iter in 0..self.n_iter {
            let momentum = if iter < momentum_switch {
                momentum_init
            } else {
                momentum_final
            };

            // Early exaggeration: multiply P by 4 for first 100 iterations
            let exag = if iter < 100 { 4.0 } else { 1.0 };

            // Compute Q and gradients
            let grad = compute_gradient(&y, &p_joint, n, d, exag);

            // Update with momentum
            for i in 0..y.len() {
                let new_val = y[i] - lr * grad[i] + momentum * (y[i] - y_prev[i]);
                y_prev[i] = y[i];
                y[i] = new_val;
            }

            // Re-centre embedding
            let mut means = vec![0.0; d];
            for i in 0..n {
                for j in 0..d {
                    means[j] += y[i * d + j];
                }
            }
            let n_f = n as f64;
            for m in &mut means {
                *m /= n_f;
            }
            for i in 0..n {
                for j in 0..d {
                    y[i * d + j] -= means[j];
                }
            }
        }

        // Convert to Tensor<T>
        let out: Vec<T> = y.iter().map(|&v| T::from_f64(v)).collect();
        Tensor::from_vec(out, vec![n, d]).map_err(MlError::from)
    }
}

// ── Internal helpers (all f64 for numerical stability) ────────────────────

/// Squared Euclidean distances between all pairs.
fn pairwise_sq_distances(data: &[impl Float], n: usize, p: usize) -> Vec<f64> {
    let mut dist = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let mut d = 0.0;
            for k in 0..p {
                let diff = to_f64(data[i * p + k]) - to_f64(data[j * p + k]);
                d += diff * diff;
            }
            dist[i * n + j] = d;
            dist[j * n + i] = d;
        }
    }
    dist
}

/// Compute symmetric joint probabilities using binary search for per-point sigma.
fn compute_joint_probabilities(dist_sq: &[f64], n: usize, perplexity: f64) -> Vec<f64> {
    let log_perp = perplexity.ln();
    let mut p = vec![0.0; n * n];

    // For each point, binary search for sigma that gives desired perplexity
    for i in 0..n {
        let mut lo = 1e-10_f64;
        let mut hi = 1e10_f64;
        let mut beta = 1.0; // beta = 1 / (2 * sigma^2)

        for _ in 0..50 {
            // Compute conditional probabilities p(j|i)
            let mut sum = 0.0;
            for j in 0..n {
                if i == j {
                    continue;
                }
                let v = (-beta * dist_sq[i * n + j]).exp();
                p[i * n + j] = v;
                sum += v;
            }
            if sum < 1e-15 {
                sum = 1e-15;
            }

            // Normalise and compute entropy
            let mut entropy = 0.0;
            for j in 0..n {
                if i == j {
                    p[i * n + j] = 0.0;
                    continue;
                }
                p[i * n + j] /= sum;
                if p[i * n + j] > 1e-15 {
                    entropy -= p[i * n + j] * p[i * n + j].ln();
                }
            }

            let diff = entropy - log_perp;
            if diff.abs() < 1e-5 {
                break;
            }
            if diff > 0.0 {
                lo = beta;
            } else {
                hi = beta;
            }
            beta = f64::midpoint(lo, hi);
        }
    }

    // Symmetrise: P_ij = (p(j|i) + p(i|j)) / (2n)
    let denom = 2.0 * n as f64;
    let mut p_sym = vec![0.0; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let v = (p[i * n + j] + p[j * n + i]) / denom;
            let v = v.max(1e-12); // floor for numerical stability
            p_sym[i * n + j] = v;
            p_sym[j * n + i] = v;
        }
    }
    p_sym
}

/// Compute the t-SNE gradient.
fn compute_gradient(y: &[f64], p: &[f64], n: usize, d: usize, exag: f64) -> Vec<f64> {
    // Pairwise distances in low-D
    let mut q_num = vec![0.0; n * n]; // (1 + ||yi - yj||^2)^(-1)
    let mut q_denom = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            let mut dist_sq = 0.0;
            for k in 0..d {
                let diff = y[i * d + k] - y[j * d + k];
                dist_sq += diff * diff;
            }
            let v = 1.0 / (1.0 + dist_sq);
            q_num[i * n + j] = v;
            q_num[j * n + i] = v;
            q_denom += 2.0 * v;
        }
    }
    if q_denom < 1e-15 {
        q_denom = 1e-15;
    }

    // Gradient: dC/dy_i = 4 * sum_j (p_ij - q_ij) * q_num_ij * (y_i - y_j)
    let mut grad = vec![0.0; n * d];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let q_ij = q_num[i * n + j] / q_denom;
            let mult = 4.0 * (exag * p[i * n + j] - q_ij) * q_num[i * n + j];
            for k in 0..d {
                grad[i * d + k] += mult * (y[i * d + k] - y[j * d + k]);
            }
        }
    }
    grad
}

fn to_f64<T: Float>(v: T) -> f64 {
    // Same approach as model_selection — format and parse.
    format!("{v}").parse::<f64>().unwrap_or(0.0)
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
    fn test_tsne_output_shape() {
        // 6 points in 5-D → embed in 2-D
        let x = Tensor::from_vec((0..30).map(f64::from).collect(), vec![6, 5]).unwrap();

        let tsne = TSNE::<f64>::new(2, 3.0).unwrap();
        let out = tsne.fit_transform(&x).unwrap();
        assert_eq!(out.shape(), &[6, 2]);
    }

    #[test]
    fn test_tsne_preserves_clusters() {
        // Two tight clusters far apart
        let mut data = Vec::new();
        // Cluster A around (0, 0)
        for i in 0..5 {
            data.push(f64::from(i) * 0.1);
            data.push(f64::from(i) * 0.1);
        }
        // Cluster B around (100, 100)
        for i in 0..5 {
            data.push(100.0 + f64::from(i) * 0.1);
            data.push(100.0 + f64::from(i) * 0.1);
        }
        let x = Tensor::from_vec(data, vec![10, 2]).unwrap();

        let tsne = TSNE::<f64>::new(2, 3.0).unwrap();
        let out = tsne.fit_transform(&x).unwrap();
        let d = out.as_slice();

        // Compute mean position of each cluster in embedding
        let (mut cx_a, mut cy_a) = (0.0, 0.0);
        let (mut cx_b, mut cy_b) = (0.0, 0.0);
        for i in 0..5 {
            cx_a += d[i * 2];
            cy_a += d[i * 2 + 1];
            cx_b += d[(i + 5) * 2];
            cy_b += d[(i + 5) * 2 + 1];
        }
        cx_a /= 5.0;
        cy_a /= 5.0;
        cx_b /= 5.0;
        cy_b /= 5.0;

        let inter_dist = ((cx_a - cx_b).powi(2) + (cy_a - cy_b).powi(2)).sqrt();
        assert!(
            inter_dist > 1.0,
            "clusters should be separated, dist={inter_dist}"
        );
    }

    #[test]
    fn test_tsne_deterministic() {
        let x = Tensor::from_vec((0..20).map(f64::from).collect(), vec![4, 5]).unwrap();

        let tsne = TSNE::<f64>::new(2, 2.0).unwrap();
        let out1 = tsne.fit_transform(&x).unwrap();
        let out2 = tsne.fit_transform(&x).unwrap();

        for (&a, &b) in out1.as_slice().iter().zip(out2.as_slice()) {
            assert!((a - b).abs() < 1e-10, "not deterministic");
        }
    }

    #[test]
    fn test_tsne_invalid_params() {
        assert!(TSNE::<f64>::new(0, 5.0).is_err());
        assert!(TSNE::<f64>::new(2, 0.0).is_err());
        assert!(TSNE::<f64>::new(2, -1.0).is_err());
    }

    #[test]
    fn test_tsne_too_few_samples() {
        let x = Tensor::from_vec(vec![1.0_f64, 2.0], vec![1, 2]).unwrap();
        let tsne = TSNE::<f64>::new(2, 5.0).unwrap();
        assert!(tsne.fit_transform(&x).is_err());
    }

    #[test]
    fn test_tsne_3d_output() {
        let x = Tensor::from_vec((0..50).map(f64::from).collect(), vec![5, 10]).unwrap();

        let tsne = TSNE::<f64>::new(3, 2.0).unwrap();
        let out = tsne.fit_transform(&x).unwrap();
        assert_eq!(out.shape(), &[5, 3]);
    }
}
