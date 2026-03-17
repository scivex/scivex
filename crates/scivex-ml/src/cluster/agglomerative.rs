//! Agglomerative (bottom-up) hierarchical clustering.

use scivex_core::{Float, Tensor};

use crate::error::{MlError, Result};

/// Linkage criterion for agglomerative clustering.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Linkage {
    /// Minimum distance between any pair of points in the two clusters.
    Single,
    /// Maximum distance between any pair of points in the two clusters.
    Complete,
    /// Weighted average of pairwise distances.
    Average,
    /// Minimises the total within-cluster variance (Ward's method).
    Ward,
}

/// A single merge step in the dendrogram.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct DendrogramNode<T> {
    /// Index of the left child (original point index or `n + merge_index`).
    pub left: usize,
    /// Index of the right child.
    pub right: usize,
    /// Distance at which this merge occurred.
    pub distance: T,
    /// Number of original points contained in this cluster.
    pub size: usize,
}

/// Agglomerative (bottom-up) hierarchical clustering.
///
/// Builds a full dendrogram using the naive O(n^3) algorithm with
/// Lance-Williams distance updates, then cuts the tree to produce
/// `n_clusters` flat clusters.
///
/// # Examples
///
/// ```
/// # use scivex_core::Tensor;
/// # use scivex_ml::cluster::{AgglomerativeClustering, Linkage};
/// let x = Tensor::from_vec(
///     vec![0.0_f64, 0.0, 0.1, 0.1, 10.0, 10.0, 10.1, 10.1],
///     vec![4, 2],
/// ).unwrap();
/// let mut agg = AgglomerativeClustering::new(2, Linkage::Single).unwrap();
/// agg.fit(&x).unwrap();
/// let labels = agg.labels().unwrap();
/// assert_eq!(labels[0], labels[1]);
/// assert_ne!(labels[0], labels[2]);
/// ```
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct AgglomerativeClustering<T: Float> {
    pub(crate) n_clusters: usize,
    pub(crate) linkage: Linkage,
    pub(crate) dendrogram: Option<Vec<DendrogramNode<T>>>,
    pub(crate) labels: Option<Vec<usize>>,
}

impl<T: Float> AgglomerativeClustering<T> {
    /// Create a new agglomerative clustering model.
    ///
    /// # Errors
    ///
    /// Returns [`MlError::InvalidParameter`] if `n_clusters` is zero.
    pub fn new(n_clusters: usize, linkage: Linkage) -> Result<Self> {
        if n_clusters == 0 {
            return Err(MlError::InvalidParameter {
                name: "n_clusters",
                reason: "must be at least 1",
            });
        }
        Ok(Self {
            n_clusters,
            linkage,
            dendrogram: None,
            labels: None,
        })
    }

    /// Fit the model to data `x` (shape `[n_samples, n_features]`).
    ///
    /// Builds a full dendrogram and cuts it to produce `n_clusters` labels.
    ///
    /// # Errors
    ///
    /// Returns an error if `x` is not 2-D, is empty, or has fewer samples
    /// than requested clusters.
    #[allow(clippy::too_many_lines, clippy::needless_range_loop)]
    pub fn fit(&mut self, x: &Tensor<T>) -> Result<()> {
        let (n, p) = matrix_shape(x)?;
        if n < self.n_clusters {
            return Err(MlError::InvalidParameter {
                name: "n_clusters",
                reason: "more clusters than samples",
            });
        }

        let data = x.as_slice();

        // --- Compute condensed pairwise squared-Euclidean distance matrix ---
        // Index into condensed form: for i < j, index = i*n - i*(i+1)/2 + j - i - 1
        let condensed_len = n * (n - 1) / 2;
        let mut dist: Vec<T> = Vec::with_capacity(condensed_len);

        for i in 0..n {
            for j in (i + 1)..n {
                let mut d = T::zero();
                for k in 0..p {
                    let diff = data[i * p + k] - data[j * p + k];
                    d += diff * diff;
                }
                // For Ward we keep squared distances; for others take sqrt.
                if self.linkage == Linkage::Ward {
                    dist.push(d);
                } else {
                    dist.push(d.sqrt());
                }
            }
        }

        // --- Cluster sizes ---
        let mut sizes: Vec<usize> = vec![1; n];

        // active[i] == true means cluster i has not yet been merged into another.
        let mut active: Vec<bool> = vec![true; n];

        // Dendrogram: n-1 merges
        let mut dendrogram: Vec<DendrogramNode<T>> = Vec::with_capacity(n - 1);

        for merge_idx in 0..(n - 1) {
            // Find the minimum distance pair among active clusters.
            let mut min_dist = T::infinity();
            let mut min_i = 0;
            let mut min_j = 0;

            for i in 0..n {
                if !active[i] {
                    continue;
                }
                for j in (i + 1)..n {
                    if !active[j] {
                        continue;
                    }
                    let idx = condensed_index(n, i, j);
                    if dist[idx] < min_dist {
                        min_dist = dist[idx];
                        min_i = i;
                        min_j = j;
                    }
                }
            }

            let new_size = sizes[min_i] + sizes[min_j];

            // Record the merge distance (take sqrt for Ward since we stored squared).
            let merge_distance = if self.linkage == Linkage::Ward {
                min_dist.sqrt()
            } else {
                min_dist
            };

            dendrogram.push(DendrogramNode {
                left: min_i,
                right: min_j,
                distance: merge_distance,
                size: new_size,
            });

            // Update distances: the merged cluster replaces min_i; deactivate min_j.
            active[min_j] = false;

            let size_i = T::from_usize(sizes[min_i]);
            let size_j = T::from_usize(sizes[min_j]);

            for k in 0..n {
                if !active[k] || k == min_i {
                    continue;
                }

                let d_ik = dist[condensed_index(n, k.min(min_i), k.max(min_i))];
                let d_jk = dist[condensed_index(n, k.min(min_j), k.max(min_j))];
                let d_ij = dist[condensed_index(n, min_i, min_j)];

                let new_d = match self.linkage {
                    Linkage::Single => {
                        if d_ik < d_jk {
                            d_ik
                        } else {
                            d_jk
                        }
                    }
                    Linkage::Complete => {
                        if d_ik > d_jk {
                            d_ik
                        } else {
                            d_jk
                        }
                    }
                    Linkage::Average => (size_i * d_ik + size_j * d_jk) / (size_i + size_j),
                    Linkage::Ward => {
                        let size_k = T::from_usize(sizes[k]);
                        let total = size_i + size_j + size_k;
                        let numerator =
                            (size_i + size_k) * d_ik + (size_j + size_k) * d_jk - size_k * d_ij;
                        numerator / total
                    }
                };

                let idx = condensed_index(n, k.min(min_i), k.max(min_i));
                dist[idx] = new_d;
            }

            sizes[min_i] = new_size;

            // Remap indices in dendrogram: original points keep their index,
            // merged clusters get index n + merge_idx. We need to update
            // the current node's left/right to use the merged-cluster index
            // for any cluster that was itself produced by a prior merge.
            //
            // We store a mapping from the "active slot" index to the
            // dendrogram node index.
            // (This is handled by post-processing below.)
            let _ = merge_idx; // used implicitly via dendrogram.len()
        }

        // --- Post-process dendrogram indices ---
        // During merging we stored raw slot indices (min_i, min_j) as left/right.
        // We need to remap so that each merged cluster gets id `n + merge_step`.
        // `slot_map[slot]` tracks the current dendrogram-node id occupying that slot.
        let mut slot_map: Vec<usize> = (0..n).collect();

        for (step, node) in dendrogram.iter_mut().enumerate() {
            let left_slot = node.left;
            let right_slot = node.right;
            node.left = slot_map[left_slot];
            node.right = slot_map[right_slot];
            // The merged cluster now lives in the left slot.
            slot_map[left_slot] = n + step;
        }

        // --- Cut dendrogram to get flat labels ---
        let labels = cut_dendrogram(&dendrogram, n, self.n_clusters);

        self.dendrogram = Some(dendrogram);
        self.labels = Some(labels);
        Ok(())
    }

    /// Return the cluster labels assigned after fitting.
    ///
    /// # Errors
    ///
    /// Returns [`MlError::NotFitted`] if the model has not been fitted.
    pub fn labels(&self) -> Result<&[usize]> {
        self.labels.as_deref().ok_or(MlError::NotFitted)
    }

    /// Return the dendrogram (merge history) after fitting.
    ///
    /// # Errors
    ///
    /// Returns [`MlError::NotFitted`] if the model has not been fitted.
    pub fn dendrogram(&self) -> Result<&[DendrogramNode<T>]> {
        self.dendrogram.as_deref().ok_or(MlError::NotFitted)
    }

    /// Fit the model and return the cluster labels.
    ///
    /// # Errors
    ///
    /// Forwards any error from [`fit`](Self::fit).
    pub fn fit_predict(&mut self, x: &Tensor<T>) -> Result<Vec<usize>> {
        self.fit(x)?;
        Ok(self.labels.clone().unwrap_or_default())
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Index into a condensed distance matrix for pair (i, j) with i < j.
#[inline]
fn condensed_index(n: usize, i: usize, j: usize) -> usize {
    debug_assert!(i < j);
    i * n - i * (i + 1) / 2 + j - i - 1
}

/// Extract shape `(n_samples, n_features)` from a 2-D tensor.
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

/// Cut a dendrogram to produce `n_clusters` flat cluster labels.
///
/// Strategy: the last `n_clusters - 1` merges are "undone", leaving
/// `n_clusters` sub-trees. Each sub-tree is assigned a unique label.
fn cut_dendrogram<T>(
    dendrogram: &[DendrogramNode<T>],
    n_samples: usize,
    n_clusters: usize,
) -> Vec<usize> {
    let n_merges = dendrogram.len(); // == n_samples - 1

    // The root is merge index n_merges - 1, node id = n_samples + n_merges - 1.
    // We keep the last (n_clusters - 1) merges "uncut" — wait, we *undo* them.
    // Roots of sub-trees are the node-ids that appear at the top n_clusters - 1
    // merge steps when those merges are removed.

    // Collect the set of "root" node ids: start from the overall root and
    // un-merge the top merges until we have n_clusters roots.
    let mut roots: Vec<usize> = vec![n_samples + n_merges - 1];

    // We undo merges from the last one backwards.
    let merges_to_undo = n_clusters.saturating_sub(1);
    let mut undone = 0;
    // Process merges from the end (highest distance) to produce more roots.
    for step in (0..n_merges).rev() {
        if undone >= merges_to_undo {
            break;
        }
        let node_id = n_samples + step;
        if let Some(pos) = roots.iter().position(|&r| r == node_id) {
            roots.remove(pos);
            roots.push(dendrogram[step].left);
            roots.push(dendrogram[step].right);
            undone += 1;
        }
    }

    // Assign labels: BFS/DFS from each root to find original point indices.
    let mut labels = vec![0_usize; n_samples];
    for (cluster_id, &root) in roots.iter().enumerate() {
        let mut stack = vec![root];
        while let Some(node) = stack.pop() {
            if node < n_samples {
                labels[node] = cluster_id;
            } else {
                let merge_step = node - n_samples;
                stack.push(dendrogram[merge_step].left);
                stack.push(dendrogram[merge_step].right);
            }
        }
    }

    labels
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two well-separated 2-D clusters with single linkage.
    #[test]
    fn test_agglomerative_single_linkage() {
        let x = Tensor::from_vec(
            vec![
                0.0_f64, 0.0, 0.1, 0.1, -0.1, 0.0, 10.0, 10.0, 10.1, 10.1, 9.9, 10.0,
            ],
            vec![6, 2],
        )
        .unwrap();

        let mut agg = AgglomerativeClustering::new(2, Linkage::Single).unwrap();
        agg.fit(&x).unwrap();
        let labels = agg.labels().unwrap();

        // First three points in one cluster, last three in another.
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[4], labels[5]);
        assert_ne!(labels[0], labels[3]);
    }

    /// Same data with complete linkage — should still separate well.
    #[test]
    fn test_agglomerative_complete_linkage() {
        let x = Tensor::from_vec(
            vec![
                0.0_f64, 0.0, 0.1, 0.1, -0.1, 0.0, 10.0, 10.0, 10.1, 10.1, 9.9, 10.0,
            ],
            vec![6, 2],
        )
        .unwrap();

        let mut agg = AgglomerativeClustering::new(2, Linkage::Complete).unwrap();
        agg.fit(&x).unwrap();
        let labels = agg.labels().unwrap();

        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[4], labels[5]);
        assert_ne!(labels[0], labels[3]);
    }

    /// Ward linkage produces valid labels.
    #[test]
    fn test_agglomerative_ward() {
        let x = Tensor::from_vec(
            vec![
                0.0_f64, 0.0, 1.0, 0.0, 0.0, 1.0, 10.0, 10.0, 11.0, 10.0, 10.0, 11.0,
            ],
            vec![6, 2],
        )
        .unwrap();

        let mut agg = AgglomerativeClustering::new(2, Linkage::Ward).unwrap();
        let labels = agg.fit_predict(&x).unwrap();

        assert_eq!(labels.len(), 6);
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[4], labels[5]);
        assert_ne!(labels[0], labels[3]);
    }

    /// Dendrogram has n-1 nodes with non-decreasing distances.
    #[test]
    fn test_dendrogram_structure() {
        let x =
            Tensor::from_vec(vec![0.0_f64, 0.0, 1.0, 0.0, 5.0, 0.0, 6.0, 0.0], vec![4, 2]).unwrap();

        let mut agg = AgglomerativeClustering::new(1, Linkage::Single).unwrap();
        agg.fit(&x).unwrap();
        let dend = agg.dendrogram().unwrap();

        // n-1 merges
        assert_eq!(dend.len(), 3);

        // Distances should be non-decreasing for single linkage.
        for i in 1..dend.len() {
            assert!(
                dend[i].distance >= dend[i - 1].distance,
                "dendrogram distances must be non-decreasing, got {} < {}",
                dend[i].distance,
                dend[i - 1].distance,
            );
        }

        // Last node should contain all points.
        assert_eq!(dend.last().unwrap().size, 4);
    }

    /// Requesting a single cluster puts everything in cluster 0.
    #[test]
    fn test_agglomerative_n_clusters_1() {
        let x = Tensor::from_vec(vec![0.0_f64, 0.0, 5.0, 5.0, 10.0, 10.0], vec![3, 2]).unwrap();

        let mut agg = AgglomerativeClustering::new(1, Linkage::Average).unwrap();
        agg.fit(&x).unwrap();
        let labels = agg.labels().unwrap();

        assert_eq!(labels, &[0, 0, 0]);
    }
}
