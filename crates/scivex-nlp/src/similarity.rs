//! Text and vector similarity measures.

use std::collections::HashSet;

use scivex_core::{Float, Tensor};

use crate::error::{NlpError, Result};
use crate::text::levenshtein;

/// Cosine similarity between two 1-D tensors: `dot(a,b) / (||a|| * ||b||)`.
///
/// Returns `1.0` for identical directions, `0.0` for orthogonal, `-1.0` for
/// opposite directions.
///
/// # Examples
///
/// ```
/// # use scivex_core::Tensor;
/// # use scivex_nlp::similarity::cosine_similarity;
/// let a = Tensor::from_vec(vec![1.0_f64, 0.0], vec![2]).unwrap();
/// let b = Tensor::from_vec(vec![0.0_f64, 1.0], vec![2]).unwrap();
/// let sim = cosine_similarity(&a, &b).unwrap();
/// assert!(sim.abs() < 1e-10); // orthogonal vectors
/// ```
pub fn cosine_similarity<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> Result<T> {
    if a.ndim() != 1 || b.ndim() != 1 {
        return Err(NlpError::InvalidParameter {
            name: "vectors",
            reason: "expected 1-D tensors",
        });
    }
    if a.shape()[0] != b.shape()[0] {
        return Err(NlpError::InvalidParameter {
            name: "vectors",
            reason: "vectors must have the same length",
        });
    }

    let dot: T = a.dot(b)?;
    let norm_a: T = a.norm()?;
    let norm_b: T = b.norm()?;

    let denom = norm_a * norm_b;
    if denom == T::zero() {
        return Ok(T::zero());
    }

    Ok(dot / denom)
}

/// Jaccard similarity: `|A ∩ B| / |A ∪ B|`.
///
/// Treats each slice as a set of tokens.
///
/// # Examples
///
/// ```
/// # use scivex_nlp::similarity::jaccard_similarity;
/// let sim = jaccard_similarity(&["a", "b", "c"], &["b", "c", "d"]);
/// assert!((sim - 0.5).abs() < 1e-10); // intersection=2, union=4
/// ```
#[must_use]
pub fn jaccard_similarity(a: &[&str], b: &[&str]) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let set_a: HashSet<&str> = a.iter().copied().collect();
    let set_b: HashSet<&str> = b.iter().copied().collect();
    let intersection = set_a.intersection(&set_b).count();
    let union = set_a.union(&set_b).count();
    if union == 0 {
        return 0.0;
    }
    intersection as f64 / union as f64
}

/// Normalized edit distance: `1.0 - levenshtein(a, b) / max(len(a), len(b))`.
///
/// Returns `1.0` for identical strings, `0.0` for completely different strings.
///
/// # Examples
///
/// ```
/// # use scivex_nlp::similarity::edit_distance_normalized;
/// let sim = edit_distance_normalized("kitten", "sitting");
/// assert!(sim > 0.0 && sim < 1.0);
/// assert!((edit_distance_normalized("hello", "hello") - 1.0).abs() < 1e-10);
/// ```
#[must_use]
pub fn edit_distance_normalized(a: &str, b: &str) -> f64 {
    let max_len = a.chars().count().max(b.chars().count());
    if max_len == 0 {
        return 1.0;
    }
    let dist = levenshtein(a, b);
    1.0 - (dist as f64 / max_len as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_identical() {
        let a = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0], vec![3]).unwrap();
        let sim = cosine_similarity(&a, &a).unwrap();
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn cosine_orthogonal() {
        let a = Tensor::from_vec(vec![1.0_f64, 0.0], vec![2]).unwrap();
        let b = Tensor::from_vec(vec![0.0_f64, 1.0], vec![2]).unwrap();
        let sim = cosine_similarity(&a, &b).unwrap();
        assert!(sim.abs() < 1e-10);
    }

    #[test]
    fn cosine_zero_vector() {
        let a = Tensor::from_vec(vec![0.0_f64, 0.0], vec![2]).unwrap();
        let b = Tensor::from_vec(vec![1.0_f64, 1.0], vec![2]).unwrap();
        let sim = cosine_similarity(&a, &b).unwrap();
        assert!(sim.abs() < 1e-10);
    }

    #[test]
    fn jaccard_same() {
        assert!((jaccard_similarity(&["a", "b", "c"], &["a", "b", "c"]) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn jaccard_disjoint() {
        assert!(jaccard_similarity(&["a", "b"], &["c", "d"]).abs() < 1e-10);
    }

    #[test]
    fn jaccard_partial() {
        let sim = jaccard_similarity(&["a", "b", "c"], &["b", "c", "d"]);
        // intersection = {b,c} = 2, union = {a,b,c,d} = 4 → 0.5
        assert!((sim - 0.5).abs() < 1e-10);
    }

    #[test]
    fn edit_distance_normalized_identical() {
        assert!((edit_distance_normalized("hello", "hello") - 1.0).abs() < 1e-10);
    }

    #[test]
    fn edit_distance_normalized_different() {
        let sim = edit_distance_normalized("abc", "xyz");
        assert!(sim < 1.0);
        assert!(sim >= 0.0);
    }

    #[test]
    fn edit_distance_normalized_empty() {
        assert!((edit_distance_normalized("", "") - 1.0).abs() < 1e-10);
    }
}
