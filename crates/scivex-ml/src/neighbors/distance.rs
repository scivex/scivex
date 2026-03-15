use scivex_core::Float;

/// Distance metrics for vector search.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetric {
    /// Euclidean (L2) distance.
    L2,
    /// Cosine distance: `1 - cos(a, b)`.
    Cosine,
    /// Negative dot product (so smaller = more similar).
    DotProduct,
    /// Hamming distance (count of differing positions).
    Hamming,
}

/// Compute the distance between two vectors using the given metric.
///
/// Both slices must have the same length. This function does **not** check
/// lengths for performance; the caller is responsible for ensuring they match.
pub fn compute_distance<T: Float>(a: &[T], b: &[T], metric: DistanceMetric) -> T {
    match metric {
        DistanceMetric::L2 => l2_distance(a, b),
        DistanceMetric::Cosine => cosine_distance(a, b),
        DistanceMetric::DotProduct => dot_product_distance(a, b),
        DistanceMetric::Hamming => hamming_distance(a, b),
    }
}

fn l2_distance<T: Float>(a: &[T], b: &[T]) -> T {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| {
            let d = x - y;
            d * d
        })
        .fold(T::zero(), |acc, v| acc + v)
        .sqrt()
}

fn cosine_distance<T: Float>(a: &[T], b: &[T]) -> T {
    let mut dot = T::zero();
    let mut norm_a = T::zero();
    let mut norm_b = T::zero();
    for (&x, &y) in a.iter().zip(b) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < T::epsilon() {
        return T::one();
    }
    T::one() - dot / denom
}

fn dot_product_distance<T: Float>(a: &[T], b: &[T]) -> T {
    let dot: T = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| x * y)
        .fold(T::zero(), |acc, v| acc + v);
    T::zero() - dot
}

fn hamming_distance<T: Float>(a: &[T], b: &[T]) -> T {
    let threshold = T::from_f64(0.5);
    let count = a
        .iter()
        .zip(b)
        .filter(|&(x, y)| (*x - *y).abs() > threshold)
        .count();
    T::from_usize(count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l2_distance() {
        let a = [0.0_f64, 0.0, 0.0];
        let b = [1.0, 2.0, 2.0];
        let d = compute_distance(&a, &b, DistanceMetric::L2);
        assert!((d - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_distance() {
        // Orthogonal vectors → cosine distance = 1.0
        let a = [1.0_f64, 0.0];
        let b = [0.0, 1.0];
        let d = compute_distance(&a, &b, DistanceMetric::Cosine);
        assert!((d - 1.0).abs() < 1e-10);

        // Same direction → cosine distance = 0.0
        let c = [3.0_f64, 4.0];
        let e = [6.0, 8.0];
        let d2 = compute_distance(&c, &e, DistanceMetric::Cosine);
        assert!(d2.abs() < 1e-10);
    }

    #[test]
    fn test_dot_product_distance() {
        let a = [1.0_f64, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        // dot = 4 + 10 + 18 = 32, distance = -32
        let d = compute_distance(&a, &b, DistanceMetric::DotProduct);
        assert!((d - (-32.0)).abs() < 1e-10);
    }

    #[test]
    fn test_hamming_distance() {
        let a = [1.0_f64, 0.0, 1.0, 0.0];
        let b = [1.0, 1.0, 0.0, 0.0];
        // positions 1 and 2 differ
        let d = compute_distance(&a, &b, DistanceMetric::Hamming);
        assert!((d - 2.0).abs() < 1e-10);
    }
}
