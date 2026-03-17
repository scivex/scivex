//! Feature matching for binary descriptors.

use std::collections::HashMap;

#[allow(unused_imports)]
use crate::error::{ImageError, Result};

/// A single descriptor match between a query and a training set.
#[derive(Debug, Clone)]
pub struct FeatureMatch {
    /// Index in the query descriptor array.
    pub query_idx: usize,
    /// Index in the train descriptor array.
    pub train_idx: usize,
    /// Hamming distance between the two descriptors (0–256).
    pub distance: u32,
}

/// Compute Hamming distance between two 256-bit binary descriptors.
pub fn hamming_distance(a: &[u8; 32], b: &[u8; 32]) -> u32 {
    let mut dist = 0u32;
    for i in 0..32 {
        dist += (a[i] ^ b[i]).count_ones();
    }
    dist
}

// ---------------------------------------------------------------------------
// Brute-force matcher
// ---------------------------------------------------------------------------

/// Exhaustive brute-force matcher for binary descriptors.
pub struct BruteForceMatcher;

impl BruteForceMatcher {
    /// Create a new brute-force matcher.
    pub fn new() -> Self {
        Self
    }

    /// Match each query descriptor to the best (lowest Hamming distance) train
    /// descriptor.
    pub fn match_descriptors(&self, query: &[[u8; 32]], train: &[[u8; 32]]) -> Vec<FeatureMatch> {
        let mut matches = Vec::with_capacity(query.len());
        for (qi, qd) in query.iter().enumerate() {
            let mut best_dist = u32::MAX;
            let mut best_idx = 0;
            for (ti, td) in train.iter().enumerate() {
                let d = hamming_distance(qd, td);
                if d < best_dist {
                    best_dist = d;
                    best_idx = ti;
                }
            }
            if !train.is_empty() {
                matches.push(FeatureMatch {
                    query_idx: qi,
                    train_idx: best_idx,
                    distance: best_dist,
                });
            }
        }
        matches
    }

    /// Match with Lowe's ratio test.
    ///
    /// For each query descriptor the two nearest train descriptors are found.
    /// A match is kept only when `best_distance / second_best_distance < ratio`.
    /// This filters out ambiguous matches.
    pub fn match_with_ratio_test(
        &self,
        query: &[[u8; 32]],
        train: &[[u8; 32]],
        ratio: f64,
    ) -> Vec<FeatureMatch> {
        if train.len() < 2 {
            // Cannot apply ratio test with fewer than two train descriptors.
            return Vec::new();
        }

        let mut matches = Vec::new();
        for (qi, qd) in query.iter().enumerate() {
            let mut best_dist = u32::MAX;
            let mut second_best_dist = u32::MAX;
            let mut best_idx = 0;

            for (ti, td) in train.iter().enumerate() {
                let d = hamming_distance(qd, td);
                if d < best_dist {
                    second_best_dist = best_dist;
                    best_dist = d;
                    best_idx = ti;
                } else if d < second_best_dist {
                    second_best_dist = d;
                }
            }

            if second_best_dist > 0 {
                let r = f64::from(best_dist) / f64::from(second_best_dist);
                if r < ratio {
                    matches.push(FeatureMatch {
                        query_idx: qi,
                        train_idx: best_idx,
                        distance: best_dist,
                    });
                }
            }
        }
        matches
    }

    /// Return the `k` nearest matches for each query descriptor.
    pub fn knn_match(
        &self,
        query: &[[u8; 32]],
        train: &[[u8; 32]],
        k: usize,
    ) -> Vec<Vec<FeatureMatch>> {
        let k = k.min(train.len());
        let mut result = Vec::with_capacity(query.len());

        for (qi, qd) in query.iter().enumerate() {
            // Collect all distances then partially sort for k-smallest.
            let mut dists: Vec<(usize, u32)> = train
                .iter()
                .enumerate()
                .map(|(ti, td)| (ti, hamming_distance(qd, td)))
                .collect();

            // Partial sort: bring the k smallest to the front.
            if k < dists.len() {
                dists.select_nth_unstable_by_key(k - 1, |&(_, d)| d);
                dists.truncate(k);
            }
            dists.sort_unstable_by_key(|&(_, d)| d);

            let knn: Vec<FeatureMatch> = dists
                .into_iter()
                .map(|(ti, d)| FeatureMatch {
                    query_idx: qi,
                    train_idx: ti,
                    distance: d,
                })
                .collect();

            result.push(knn);
        }
        result
    }
}

impl Default for BruteForceMatcher {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// FLANN-style LSH matcher
// ---------------------------------------------------------------------------

/// A simplified FLANN-style matcher using Locality-Sensitive Hashing (LSH)
/// for fast approximate matching of binary descriptors.
pub struct FlannMatcher {
    /// Number of hash tables.
    table_count: usize,
    /// Number of bits per hash key.
    key_size: usize,
    /// The bit positions sampled for each hash table.
    /// `bit_positions[t]` has `key_size` entries, each in `0..256`.
    bit_positions: Vec<Vec<usize>>,
    /// Hash tables: `tables[t]` maps a hash key to a list of descriptor indices.
    tables: Vec<HashMap<u64, Vec<usize>>>,
    /// Stored training descriptors.
    train: Vec<[u8; 32]>,
}

impl FlannMatcher {
    /// Create a new FLANN matcher with default parameters (6 tables, 12-bit keys).
    pub fn new() -> Self {
        Self {
            table_count: 6,
            key_size: 12,
            bit_positions: Vec::new(),
            tables: Vec::new(),
            train: Vec::new(),
        }
    }

    /// Set the number of hash tables.
    pub fn with_table_count(mut self, n: usize) -> Self {
        self.table_count = n;
        self
    }

    /// Set the number of bits per hash key.
    pub fn with_key_size(mut self, k: usize) -> Self {
        self.key_size = k;
        self
    }

    /// Build the LSH index from training descriptors.
    ///
    /// Bit positions for table `t` are computed deterministically:
    /// `(t * key_size + i) * 37 % 256` for `i` in `0..key_size`.
    pub fn build_index(&mut self, train: &[[u8; 32]]) {
        self.train = train.to_vec();

        // Compute bit positions for each table.
        self.bit_positions = (0..self.table_count)
            .map(|t| {
                (0..self.key_size)
                    .map(|i| ((t * self.key_size + i) * 37) % 256)
                    .collect()
            })
            .collect();

        // Build hash tables.
        self.tables = Vec::with_capacity(self.table_count);
        for t in 0..self.table_count {
            let mut table: HashMap<u64, Vec<usize>> = HashMap::new();
            for (idx, desc) in train.iter().enumerate() {
                let key = self.hash_descriptor(desc, t);
                table.entry(key).or_default().push(idx);
            }
            self.tables.push(table);
        }
    }

    /// Match query descriptors against the built index.
    ///
    /// For each query descriptor, candidate matches are collected from all hash
    /// tables, then exact Hamming distances are computed only for those
    /// candidates. The best candidate is returned.
    pub fn match_descriptors(&self, query: &[[u8; 32]]) -> Vec<FeatureMatch> {
        let mut matches = Vec::with_capacity(query.len());

        for (qi, qd) in query.iter().enumerate() {
            // Gather candidate indices from all tables.
            let mut candidates: Vec<bool> = vec![false; self.train.len()];
            for t in 0..self.table_count {
                let key = self.hash_descriptor(qd, t);
                if let Some(indices) = self.tables.get(t).and_then(|tbl| tbl.get(&key)) {
                    for &idx in indices {
                        candidates[idx] = true;
                    }
                }
            }

            // Find best match among candidates.
            let mut best_dist = u32::MAX;
            let mut best_idx = 0;
            for (ti, &is_candidate) in candidates.iter().enumerate() {
                if is_candidate {
                    let d = hamming_distance(qd, &self.train[ti]);
                    if d < best_dist {
                        best_dist = d;
                        best_idx = ti;
                    }
                }
            }

            if best_dist < u32::MAX {
                matches.push(FeatureMatch {
                    query_idx: qi,
                    train_idx: best_idx,
                    distance: best_dist,
                });
            }
        }
        matches
    }

    /// Compute the hash key for a descriptor in table `t`.
    fn hash_descriptor(&self, desc: &[u8; 32], t: usize) -> u64 {
        let positions = &self.bit_positions[t];
        let mut key: u64 = 0;
        for (i, &bit_pos) in positions.iter().enumerate() {
            let byte_idx = bit_pos / 8;
            let bit_idx = bit_pos % 8;
            if (desc[byte_idx] >> bit_idx) & 1 == 1 {
                key |= 1u64 << i;
            }
        }
        key
    }
}

impl Default for FlannMatcher {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hamming_identical_is_zero() {
        let a = [0xABu8; 32];
        assert_eq!(hamming_distance(&a, &a), 0);
    }

    #[test]
    fn hamming_all_different_is_256() {
        let a = [0x00u8; 32];
        let b = [0xFFu8; 32];
        assert_eq!(hamming_distance(&a, &b), 256);
    }

    #[test]
    fn brute_force_matches_identical_sets() {
        let descriptors: Vec<[u8; 32]> = (0..10)
            .map(|i| {
                let mut d = [0u8; 32];
                d[0] = i as u8;
                d[1] = (i * 7) as u8;
                d
            })
            .collect();

        let matcher = BruteForceMatcher::new();
        let matches = matcher.match_descriptors(&descriptors, &descriptors);

        assert_eq!(matches.len(), descriptors.len());
        for m in &matches {
            assert_eq!(m.query_idx, m.train_idx, "identical sets should self-match");
            assert_eq!(m.distance, 0);
        }
    }

    #[test]
    fn ratio_test_filters_ambiguous() {
        // Create two distinct train descriptors and one that is equidistant.
        let train: Vec<[u8; 32]> = vec![[0x00; 32], [0xFF; 32]];

        // Query that is very close to train[0] — should pass ratio test.
        let mut good_query = [0x00u8; 32];
        good_query[0] = 0x01; // distance 1 to train[0], ~255 to train[1]

        // Query roughly equidistant — should fail ratio test.
        let mut ambiguous_query = [0x00u8; 32];
        // Set half the bytes to 0xFF so distance to both is ~128.
        for byte in ambiguous_query.iter_mut().take(16) {
            *byte = 0xFF;
        }

        let matcher = BruteForceMatcher::new();
        let matches = matcher.match_with_ratio_test(&[good_query, ambiguous_query], &train, 0.7);

        // The good query should survive; the ambiguous one should be filtered.
        assert!(
            matches.iter().any(|m| m.query_idx == 0),
            "good match should pass ratio test"
        );
        assert!(
            !matches.iter().any(|m| m.query_idx == 1),
            "ambiguous match should be filtered"
        );
    }

    #[test]
    fn flann_finds_correct_matches() {
        let descriptors: Vec<[u8; 32]> = (0..20)
            .map(|i| {
                let mut d = [0u8; 32];
                // Spread values across multiple bytes for varied hashes.
                d[0] = i as u8;
                d[1] = (i * 13) as u8;
                d[2] = (i * 53) as u8;
                d[3] = (i * 97) as u8;
                d
            })
            .collect();

        let mut matcher = FlannMatcher::new();
        matcher.build_index(&descriptors);
        let matches = matcher.match_descriptors(&descriptors);

        // FLANN is approximate, but on identical query/train with well-separated
        // descriptors it should find most correct matches.
        let correct = matches
            .iter()
            .filter(|m| m.query_idx == m.train_idx)
            .count();
        assert!(
            correct >= descriptors.len() / 2,
            "FLANN should find at least half the correct matches, got {correct}/{}",
            descriptors.len()
        );
    }
}
