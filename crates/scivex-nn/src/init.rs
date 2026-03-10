//! Weight initialization strategies.

use scivex_core::random::Rng;
use scivex_core::{Float, Tensor};

/// Xavier (Glorot) uniform initialization.
///
/// Samples from `Uniform[-a, a]` where `a = sqrt(6 / (fan_in + fan_out))`.
/// `shape` should be `[fan_out, fan_in]` (weight matrix convention).
pub fn xavier_uniform<T: Float>(shape: &[usize], rng: &mut Rng) -> Tensor<T> {
    let (fan_in, fan_out) = compute_fans(shape);
    let a = T::from_f64((6.0 / (fan_in + fan_out) as f64).sqrt());
    let t = scivex_core::random::uniform::<T>(rng, shape.to_vec());
    // uniform gives [0,1), scale to [-a, a]
    let two = T::from_f64(2.0);
    t.map(|v| v * two * a - a)
}

/// Xavier (Glorot) normal initialization.
///
/// Samples from `Normal(0, std)` where `std = sqrt(2 / (fan_in + fan_out))`.
pub fn xavier_normal<T: Float>(shape: &[usize], rng: &mut Rng) -> Tensor<T> {
    let (fan_in, fan_out) = compute_fans(shape);
    let std = (2.0 / (fan_in + fan_out) as f64).sqrt();
    scivex_core::random::normal::<T>(rng, shape.to_vec(), T::zero(), T::from_f64(std))
}

/// Kaiming (He) uniform initialization.
///
/// Samples from `Uniform[-a, a]` where `a = sqrt(6 / fan_in)`.
pub fn kaiming_uniform<T: Float>(shape: &[usize], rng: &mut Rng) -> Tensor<T> {
    let (fan_in, _) = compute_fans(shape);
    let a = T::from_f64((6.0 / fan_in as f64).sqrt());
    let t = scivex_core::random::uniform::<T>(rng, shape.to_vec());
    let two = T::from_f64(2.0);
    t.map(|v| v * two * a - a)
}

/// Kaiming (He) normal initialization.
///
/// Samples from `Normal(0, std)` where `std = sqrt(2 / fan_in)`.
pub fn kaiming_normal<T: Float>(shape: &[usize], rng: &mut Rng) -> Tensor<T> {
    let (fan_in, _) = compute_fans(shape);
    let std = (2.0 / fan_in as f64).sqrt();
    scivex_core::random::normal::<T>(rng, shape.to_vec(), T::zero(), T::from_f64(std))
}

/// Compute `(fan_in, fan_out)` from a weight shape.
///
/// - 1-D `[n]` → `(n, n)`
/// - 2-D `[out, in]` → `(in, out)`
fn compute_fans(shape: &[usize]) -> (usize, usize) {
    match shape.len() {
        1 => (shape[0], shape[0]),
        2 => (shape[1], shape[0]),
        _ => {
            // For higher-dimensional shapes, treat as [out, in, *kernel].
            let fan_out = shape[0];
            let fan_in = shape[1..].iter().product();
            (fan_in, fan_out)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xavier_uniform_shape() {
        let mut rng = Rng::new(42);
        let w: Tensor<f64> = xavier_uniform(&[10, 5], &mut rng);
        assert_eq!(w.shape(), &[10, 5]);
    }

    #[test]
    fn test_xavier_uniform_bounds() {
        let mut rng = Rng::new(42);
        let w: Tensor<f64> = xavier_uniform(&[100, 50], &mut rng);
        let a = (6.0 / 150.0_f64).sqrt();
        for &v in w.as_slice() {
            assert!(v >= -a - 1e-10 && v <= a + 1e-10);
        }
    }

    #[test]
    fn test_kaiming_normal_shape() {
        let mut rng = Rng::new(42);
        let w: Tensor<f64> = kaiming_normal(&[20, 10], &mut rng);
        assert_eq!(w.shape(), &[20, 10]);
    }

    #[test]
    fn test_xavier_normal_mean_near_zero() {
        let mut rng = Rng::new(42);
        let w: Tensor<f64> = xavier_normal(&[200, 100], &mut rng);
        let mean = w.mean();
        assert!(mean.abs() < 0.1, "mean was {mean}");
    }

    #[test]
    fn test_kaiming_uniform_shape() {
        let mut rng = Rng::new(42);
        let w: Tensor<f64> = kaiming_uniform(&[10, 5], &mut rng);
        assert_eq!(w.shape(), &[10, 5]);
    }
}
