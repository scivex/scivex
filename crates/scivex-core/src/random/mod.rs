//! Pseudo-random number generation and random tensor creation.
//!
//! Provides a fast, high-quality PRNG (`Rng`) based on the xoshiro256\*\*
//! algorithm and convenience functions for creating tensors filled with random
//! values drawn from common distributions.
//!
//! # Design
//!
//! - **Zero external dependencies** — the PRNG is implemented from scratch.
//! - **Explicit state** — all functions take `&mut Rng`; there is no hidden
//!   global or thread-local state.
//! - Seeding uses `SplitMix64` to expand a single `u64` into the 4-word
//!   xoshiro256\*\* state (avoids the zero-state trap).

use crate::error::{CoreError, Result};
use crate::tensor::Tensor;
use crate::{Float, Integer, Scalar};

// ---------------------------------------------------------------------------
// SplitMix64 — used only for seeding
// ---------------------------------------------------------------------------

/// Advance a `SplitMix64` state by one step and return the mixed output.
#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

// ---------------------------------------------------------------------------
// Rng — xoshiro256**
// ---------------------------------------------------------------------------

/// A fast, high-quality pseudo-random number generator.
///
/// Uses the xoshiro256\*\* algorithm (Blackman & Vigna), which has a period of
/// 2^256 − 1 and passes all `BigCrush` tests.
///
/// # Examples
///
/// ```
/// use scivex_core::random::Rng;
///
/// let mut rng = Rng::new(42);
/// let value = rng.next_f64(); // uniform in [0, 1)
/// assert!((0.0..1.0).contains(&value));
/// ```
pub struct Rng {
    s: [u64; 4],
    /// Cached spare normal value from Box-Muller (None when empty).
    spare_normal: Option<f64>,
}

impl Rng {
    /// Create a new PRNG seeded from a single `u64`.
    ///
    /// The seed is expanded into the 4-word internal state via `SplitMix64`.
    pub fn new(seed: u64) -> Self {
        let mut sm = seed;
        let s = [
            splitmix64(&mut sm),
            splitmix64(&mut sm),
            splitmix64(&mut sm),
            splitmix64(&mut sm),
        ];
        Self {
            s,
            spare_normal: None,
        }
    }

    /// Re-seed the generator, discarding all previous state.
    ///
    /// # Examples
    ///
    /// ```
    /// use scivex_core::random::Rng;
    /// let mut rng = Rng::new(1);
    /// let first = rng.next_u64();
    /// rng.seed(1);
    /// assert_eq!(rng.next_u64(), first);
    /// ```
    pub fn seed(&mut self, seed: u64) {
        *self = Self::new(seed);
    }

    /// Create `n` independent child RNGs by advancing the state.
    ///
    /// Each child receives a unique seed derived from the parent's state.
    /// This is useful for parallel workloads where each thread needs its
    /// own RNG to avoid contention.
    ///
    /// # Examples
    ///
    /// ```
    /// use scivex_core::random::Rng;
    /// let mut rng = Rng::new(42);
    /// let children = rng.fork(4);
    /// assert_eq!(children.len(), 4);
    /// ```
    pub fn fork(&mut self, n: usize) -> Vec<Self> {
        (0..n).map(|_| Self::new(self.next_u64())).collect()
    }

    /// Generate the next random `u64`.
    ///
    /// # Examples
    ///
    /// ```
    /// use scivex_core::random::Rng;
    /// let mut rng = Rng::new(1);
    /// let v = rng.next_u64(); // some pseudo-random u64
    /// let _ = v; // value is deterministic but not checked here
    /// ```
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let result = (self.s[1].wrapping_mul(5)).rotate_left(7).wrapping_mul(9);

        let t = self.s[1] << 17;

        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];

        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);

        result
    }

    /// Generate a random `f64` uniformly distributed in [0, 1).
    ///
    /// Uses the upper 53 bits of `next_u64` divided by 2^53.
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Generate a standard normal (N(0,1)) `f64` via the Ziggurat algorithm.
    ///
    /// ~97% of samples require only a multiply and comparison (no
    /// transcendentals), making this much faster than Box-Muller.
    ///
    /// # Examples
    ///
    /// ```
    /// use scivex_core::random::Rng;
    /// let mut rng = Rng::new(0);
    /// // Draw 1000 samples and verify mean is near 0
    /// let mean: f64 = (0..1000).map(|_| rng.next_normal_f64()).sum::<f64>() / 1000.0;
    /// assert!(mean.abs() < 0.2);
    /// ```
    pub fn next_normal_f64(&mut self) -> f64 {
        ziggurat_normal(self)
    }
}

// ---------------------------------------------------------------------------
// Ziggurat algorithm for normal distribution
// ---------------------------------------------------------------------------

/// Number of rectangles in the Ziggurat decomposition.
const ZIG_N: usize = 128;
/// Right-most x of the base rectangle.
const ZIG_R: f64 = 3.442619855899;
/// Area of each rectangle (= area of the tail).
const ZIG_V: f64 = 9.91256303526217e-3;

/// Precomputed Ziggurat table: x-coordinates of rectangle right edges.
/// `YTAB[i] = f(XTAB[i])` where `f(x) = exp(-x*x/2)`.
fn zig_tables() -> ([f64; ZIG_N + 1], [f64; ZIG_N + 1]) {
    let mut xtab = [0.0f64; ZIG_N + 1];
    let mut ytab = [0.0f64; ZIG_N + 1];

    let f = |x: f64| (-0.5 * x * x).exp();

    xtab[ZIG_N] = ZIG_V / f(ZIG_R);
    xtab[ZIG_N - 1] = ZIG_R;
    ytab[ZIG_N] = 0.0;
    ytab[ZIG_N - 1] = f(xtab[ZIG_N - 1]);

    let mut i = ZIG_N - 2;
    loop {
        xtab[i] = (-2.0 * (ZIG_V / xtab[i + 1] + f(xtab[i + 1])).ln()).sqrt();
        ytab[i] = f(xtab[i]);
        if i == 0 {
            break;
        }
        i -= 1;
    }
    // xtab[0] is the top (smallest x); ytab[0] = f(xtab[0]) ≈ 1.
    (xtab, ytab)
}

/// Sample from the tail of the normal distribution (|x| > R).
fn zig_tail(rng: &mut Rng, positive: bool) -> f64 {
    loop {
        let x = -rng.next_f64().ln() / ZIG_R; // exponential with rate R
        let y = -rng.next_f64().ln();
        if 2.0 * y >= x * x {
            return if positive { ZIG_R + x } else { -(ZIG_R + x) };
        }
    }
}

/// Ziggurat normal: O(1) expected time, ~97% fast-path.
fn ziggurat_normal(rng: &mut Rng) -> f64 {
    // We compute tables once per call. In a tight loop the compiler will
    // usually const-fold or cache these, but for absolute best perf we use
    // a static lazy init.
    use std::sync::OnceLock;
    static TABLES: OnceLock<([f64; ZIG_N + 1], [f64; ZIG_N + 1])> = OnceLock::new();
    let (xtab, ytab) = TABLES.get_or_init(zig_tables);

    loop {
        let u = rng.next_u64();
        let i = (u & 0x7F) as usize; // bottom 7 bits → layer index [0, 127]
        let sign = if u & 0x80 != 0 { 1.0 } else { -1.0 };
        // Use remaining bits for a uniform float in [0, 1).
        let u_float = (u >> 8) as f64 / ((1u64 << 56) as f64);
        let x = u_float * xtab[i];

        // Fast accept: x falls strictly inside rectangle i.
        if x < xtab[i + 1] {
            return sign * x;
        }

        // Bottom layer includes the tail.
        if i == 0 {
            return zig_tail(rng, sign > 0.0);
        }

        // Slow accept: sample within the wedge between rectangles.
        let y = ytab[i + 1] + (ytab[i] - ytab[i + 1]) * rng.next_f64();
        if y < (-0.5 * x * x).exp() {
            return sign * x;
        }
    }
}

// ---------------------------------------------------------------------------
// Free functions — random tensor generation
// ---------------------------------------------------------------------------

/// Create a tensor filled with values uniformly distributed in [0, 1).
///
/// # Examples
///
/// ```
/// use scivex_core::random::{Rng, uniform};
///
/// let mut rng = Rng::new(0);
/// let t = uniform::<f64>(&mut rng, vec![2, 3]);
/// assert_eq!(t.shape(), &[2, 3]);
/// assert!(t.iter().all(|&x| (0.0..1.0).contains(&x)));
/// ```
pub fn uniform<T: Float>(rng: &mut Rng, shape: Vec<usize>) -> Tensor<T> {
    let numel: usize = shape.iter().product();
    let data: Vec<T> = (0..numel).map(|_| T::from_f64(rng.next_f64())).collect();
    Tensor::from_vec(data, shape).expect("shape product matches data length")
}

/// Create a tensor filled with values uniformly distributed in [`low`, `high`).
///
/// Returns an error if `low >= high`.
///
/// # Examples
///
/// ```
/// use scivex_core::random::{Rng, uniform_range};
/// let mut rng = Rng::new(0);
/// let t = uniform_range::<f64>(&mut rng, vec![5], 2.0_f64, 5.0_f64).unwrap();
/// assert!(t.iter().all(|&x| (2.0_f64..5.0_f64).contains(&x)));
/// ```
pub fn uniform_range<T: Float>(
    rng: &mut Rng,
    shape: Vec<usize>,
    low: T,
    high: T,
) -> Result<Tensor<T>> {
    if low >= high {
        return Err(CoreError::InvalidArgument {
            reason: "uniform_range requires low < high",
        });
    }
    let range = high - low;
    let numel: usize = shape.iter().product();
    let data: Vec<T> = (0..numel)
        .map(|_| low + T::from_f64(rng.next_f64()) * range)
        .collect();
    Ok(Tensor::from_vec(data, shape).expect("shape product matches data length"))
}

/// Create a tensor of samples from a Gaussian distribution.
///
/// Uses the Box-Muller transform internally.
///
/// # Examples
///
/// ```
/// use scivex_core::random::{Rng, normal};
/// let mut rng = Rng::new(1);
/// let t = normal::<f64>(&mut rng, vec![3], 0.0_f64, 1.0_f64);
/// assert_eq!(t.shape(), &[3]);
/// ```
pub fn normal<T: Float>(rng: &mut Rng, shape: Vec<usize>, mean: T, std_dev: T) -> Tensor<T> {
    let numel: usize = shape.iter().product();
    let data: Vec<T> = (0..numel)
        .map(|_| mean + std_dev * T::from_f64(rng.next_normal_f64()))
        .collect();
    Tensor::from_vec(data, shape).expect("shape product matches data length")
}

/// Create a tensor of samples from the standard normal distribution N(0, 1).
///
/// # Examples
///
/// ```
/// use scivex_core::random::{Rng, standard_normal};
/// let mut rng = Rng::new(7);
/// let t = standard_normal::<f64>(&mut rng, vec![4]);
/// assert_eq!(t.shape(), &[4]);
/// ```
pub fn standard_normal<T: Float>(rng: &mut Rng, shape: Vec<usize>) -> Tensor<T> {
    normal(rng, shape, T::zero(), T::one())
}

/// Create a tensor of random integers in [`low`, `high`).
///
/// Returns an error if `low >= high`.
///
/// # Examples
///
/// ```
/// use scivex_core::random::{Rng, randint};
/// let mut rng = Rng::new(0);
/// let t = randint::<i32>(&mut rng, vec![10], 0, 5).unwrap();
/// assert!(t.iter().all(|&x| (0..5).contains(&x)));
/// ```
pub fn randint<T: Integer>(rng: &mut Rng, shape: Vec<usize>, low: T, high: T) -> Result<Tensor<T>> {
    if low >= high {
        return Err(CoreError::InvalidArgument {
            reason: "randint requires low < high",
        });
    }
    // Find the range as a usize via binary search on from_usize.
    let range = int_range_as_usize(low, high);
    let numel: usize = shape.iter().product();
    let data: Vec<T> = (0..numel)
        .map(|_| {
            let idx = (rng.next_f64() * range as f64) as usize;
            low + T::from_usize(idx.min(range - 1))
        })
        .collect();
    Ok(Tensor::from_vec(data, shape).expect("shape product matches data length"))
}

/// Compute the number of integers in [low, high) as a `usize`.
/// used for random integer generation where ranges are typically small, the
/// linear scan is acceptable. For large ranges a binary search on `from_usize`
/// is unsafe because signed types wrap on overflow.
fn int_range_as_usize<T: Integer>(low: T, high: T) -> usize {
    (high - low).to_usize()
}

/// Create a tensor of Bernoulli random variables (0 or 1) with probability `p`.
///
/// Returns an error if `p` is not in [0, 1].
///
/// # Examples
///
/// ```
/// use scivex_core::random::{Rng, bernoulli};
/// let mut rng = Rng::new(0);
/// let t = bernoulli::<f64>(&mut rng, vec![10], 0.5).unwrap();
/// assert!(t.iter().all(|&x| x == 0.0 || x == 1.0));
/// ```
pub fn bernoulli<T: Scalar>(rng: &mut Rng, shape: Vec<usize>, p: f64) -> Result<Tensor<T>> {
    if !(0.0..=1.0).contains(&p) {
        return Err(CoreError::InvalidArgument {
            reason: "bernoulli requires p in [0, 1]",
        });
    }
    let numel: usize = shape.iter().product();
    let data: Vec<T> = (0..numel)
        .map(|_| {
            if rng.next_f64() < p {
                T::one()
            } else {
                T::zero()
            }
        })
        .collect();
    Ok(Tensor::from_vec(data, shape).expect("shape product matches data length"))
}

/// Shuffle the elements of a tensor in-place using the Fisher-Yates algorithm.
///
/// Operates on the flat (storage-order) data regardless of shape.
///
/// # Examples
///
/// ```
/// use scivex_core::random::{Rng, shuffle};
/// use scivex_core::Tensor;
/// let mut rng = Rng::new(0);
/// let mut t = Tensor::from_vec(vec![1, 2, 3, 4, 5], vec![5]).unwrap();
/// shuffle(&mut rng, &mut t);
/// let mut sorted = t.as_slice().to_vec();
/// sorted.sort_unstable();
/// assert_eq!(sorted, vec![1, 2, 3, 4, 5]);
/// ```
pub fn shuffle<T: Scalar>(rng: &mut Rng, tensor: &mut Tensor<T>) {
    let n = tensor.numel();
    if n <= 1 {
        return;
    }
    let data = tensor.as_mut_slice();
    for i in (1..n).rev() {
        // Generate j in [0, i] using rejection-free modular reduction.
        let j = (rng.next_f64() * (i + 1) as f64) as usize;
        // Clamp to valid range (floating-point edge case).
        let j = j.min(i);
        data.swap(i, j);
    }
}

/// Sample `n` elements from a 1-D tensor.
///
/// - `replace = true`: sampling with replacement (may contain duplicates).
/// - `replace = false`: sampling without replacement. Returns an error if
///   `n > tensor.numel()`.
///
/// Returns an error if the tensor is not 1-D.
///
/// # Examples
///
/// ```
/// use scivex_core::random::{Rng, choice};
/// use scivex_core::Tensor;
/// let mut rng = Rng::new(0);
/// let t = Tensor::from_vec(vec![10, 20, 30, 40, 50], vec![5]).unwrap();
/// let sample = choice(&mut rng, &t, 3, false).unwrap();
/// assert_eq!(sample.shape(), &[3]);
/// ```
pub fn choice<T: Scalar>(
    rng: &mut Rng,
    tensor: &Tensor<T>,
    n: usize,
    replace: bool,
) -> Result<Tensor<T>> {
    if tensor.ndim() != 1 {
        return Err(CoreError::InvalidArgument {
            reason: "choice requires a 1-D tensor",
        });
    }
    let len = tensor.numel();

    if !replace && n > len {
        return Err(CoreError::InvalidArgument {
            reason: "choice without replacement: n > tensor length",
        });
    }

    let src = tensor.as_slice();

    if replace {
        let data: Vec<T> = (0..n)
            .map(|_| {
                let idx = (rng.next_f64() * len as f64) as usize;
                src[idx.min(len - 1)]
            })
            .collect();
        Tensor::from_vec(data, vec![n])
    } else {
        // Fisher-Yates partial shuffle on index array
        let mut indices: Vec<usize> = (0..len).collect();
        for i in 0..n {
            let j = i + (rng.next_f64() * (len - i) as f64) as usize;
            let j = j.min(len - 1);
            indices.swap(i, j);
        }
        let data: Vec<T> = indices[..n].iter().map(|&i| src[i]).collect();
        Tensor::from_vec(data, vec![n])
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn test_rng_reproducibility() {
        let mut rng1 = Rng::new(12345);
        let mut rng2 = Rng::new(12345);
        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn test_rng_different_seeds() {
        let mut rng1 = Rng::new(1);
        let mut rng2 = Rng::new(2);
        // Extremely unlikely to be equal
        let seq1: Vec<u64> = (0..10).map(|_| rng1.next_u64()).collect();
        let seq2: Vec<u64> = (0..10).map(|_| rng2.next_u64()).collect();
        assert_ne!(seq1, seq2);
    }

    #[test]
    fn test_next_f64_range() {
        let mut rng = Rng::new(42);
        for _ in 0..10_000 {
            let v = rng.next_f64();
            assert!((0.0..1.0).contains(&v), "next_f64 out of range: {v}");
        }
    }

    #[test]
    fn test_reseed() {
        let mut rng = Rng::new(99);
        let first = rng.next_u64();
        rng.seed(99);
        let second = rng.next_u64();
        assert_eq!(first, second);
    }

    #[test]
    fn test_fork() {
        let mut rng = Rng::new(42);
        let children = rng.fork(4);
        assert_eq!(children.len(), 4);
        // All children should produce different sequences.
        let vals: Vec<u64> = children.into_iter().map(|mut r| r.next_u64()).collect();
        for i in 0..vals.len() {
            for j in (i + 1)..vals.len() {
                assert_ne!(vals[i], vals[j], "child RNGs should be independent");
            }
        }
    }

    #[test]
    fn test_fork_reproducible() {
        let mut rng1 = Rng::new(42);
        let children1 = rng1.fork(3);
        let mut rng2 = Rng::new(42);
        let children2 = rng2.fork(3);
        // Same seed → same children.
        for (mut c1, mut c2) in children1.into_iter().zip(children2) {
            assert_eq!(c1.next_u64(), c2.next_u64());
        }
    }

    #[test]
    fn test_uniform_shape() {
        let mut rng = Rng::new(0);
        let t = uniform::<f64>(&mut rng, vec![3, 4, 5]);
        assert_eq!(t.shape(), &[3, 4, 5]);
        assert_eq!(t.numel(), 60);
    }

    #[test]
    fn test_uniform_range_values() {
        let mut rng = Rng::new(0);
        let t = uniform::<f64>(&mut rng, vec![1000]);
        for &v in t.as_slice() {
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn test_uniform_range_bounds() {
        let mut rng = Rng::new(7);
        let t = uniform_range::<f64>(&mut rng, vec![5000], 2.0, 5.0).unwrap();
        for &v in t.as_slice() {
            assert!((2.0..5.0).contains(&v), "value {v} out of [2, 5)");
        }
    }

    #[test]
    fn test_uniform_range_invalid() {
        let mut rng = Rng::new(0);
        assert!(uniform_range::<f64>(&mut rng, vec![10], 5.0, 2.0).is_err());
        assert!(uniform_range::<f64>(&mut rng, vec![10], 3.0, 3.0).is_err());
    }

    #[test]
    fn test_standard_normal_stats() {
        let mut rng = Rng::new(42);
        let t = standard_normal::<f64>(&mut rng, vec![100_000]);
        let data = t.as_slice();

        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let var = data.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / data.len() as f64;
        let std = var.sqrt();

        assert!(
            mean.abs() < 0.02,
            "standard normal mean too far from 0: {mean}"
        );
        assert!(
            (std - 1.0).abs() < 0.02,
            "standard normal std too far from 1: {std}"
        );
    }

    #[test]
    fn test_normal_custom() {
        let mut rng = Rng::new(42);
        let t = normal::<f64>(&mut rng, vec![50_000], 10.0, 2.0);
        let data = t.as_slice();

        let mean = data.iter().sum::<f64>() / data.len() as f64;
        assert!(
            (mean - 10.0).abs() < 0.1,
            "normal(10, 2) mean too far from 10: {mean}"
        );
    }

    #[test]
    fn test_randint_range() {
        let mut rng = Rng::new(0);
        let t = randint::<i32>(&mut rng, vec![10_000], 5, 10).unwrap();
        for &v in t.as_slice() {
            assert!((5..10).contains(&v), "randint value {v} not in [5, 10)");
        }
    }

    #[test]
    fn test_randint_invalid() {
        let mut rng = Rng::new(0);
        assert!(randint::<i32>(&mut rng, vec![10], 10, 5).is_err());
        assert!(randint::<i32>(&mut rng, vec![10], 5, 5).is_err());
    }

    #[test]
    fn test_bernoulli_values() {
        let mut rng = Rng::new(0);
        let t = bernoulli::<f64>(&mut rng, vec![10_000], 0.3).unwrap();
        for &v in t.as_slice() {
            assert!(v == 0.0 || v == 1.0, "bernoulli value {v} not 0 or 1");
        }
        // Check frequency is approximately p
        let ones = t.as_slice().iter().filter(|&&x| x == 1.0).count();
        let freq = ones as f64 / 10_000.0;
        assert!(
            (freq - 0.3).abs() < 0.03,
            "bernoulli frequency {freq} too far from 0.3"
        );
    }

    #[test]
    fn test_bernoulli_invalid() {
        let mut rng = Rng::new(0);
        assert!(bernoulli::<f64>(&mut rng, vec![10], -0.1).is_err());
        assert!(bernoulli::<f64>(&mut rng, vec![10], 1.1).is_err());
    }

    #[test]
    fn test_shuffle_preserves_elements() {
        let mut rng = Rng::new(42);
        let mut t = Tensor::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10], vec![10]).unwrap();
        shuffle(&mut rng, &mut t);

        let mut sorted = t.as_slice().to_vec();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    }

    #[test]
    fn test_shuffle_modifies_order() {
        let mut rng = Rng::new(42);
        let original = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let mut t = Tensor::from_vec(original.clone(), vec![10]).unwrap();
        shuffle(&mut rng, &mut t);
        // Very unlikely to remain in original order with 10 elements
        assert_ne!(t.as_slice(), &original[..]);
    }

    #[test]
    fn test_choice_with_replacement() {
        let mut rng = Rng::new(0);
        let t = Tensor::from_vec(vec![10.0, 20.0, 30.0, 40.0, 50.0], vec![5]).unwrap();
        let sample = choice(&mut rng, &t, 100, true).unwrap();
        assert_eq!(sample.shape(), &[100]);
        // All values should be from the original tensor
        let valid = [10.0, 20.0, 30.0, 40.0, 50.0];
        for &v in sample.as_slice() {
            assert!(valid.contains(&v), "unexpected value {v}");
        }
    }

    #[test]
    fn test_choice_without_replacement() {
        let mut rng = Rng::new(0);
        let t = Tensor::from_vec(vec![10, 20, 30, 40, 50], vec![5]).unwrap();
        let sample = choice(&mut rng, &t, 3, false).unwrap();
        assert_eq!(sample.shape(), &[3]);

        // No duplicates
        let data = sample.as_slice();
        let mut dedup = data.to_vec();
        dedup.sort_unstable();
        dedup.dedup();
        assert_eq!(dedup.len(), 3);
    }

    #[test]
    fn test_choice_without_replacement_too_many() {
        let mut rng = Rng::new(0);
        let t = Tensor::from_vec(vec![1, 2, 3], vec![3]).unwrap();
        assert!(choice(&mut rng, &t, 5, false).is_err());
    }

    #[test]
    fn test_choice_not_1d() {
        let mut rng = Rng::new(0);
        let t = Tensor::from_vec(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
        assert!(choice(&mut rng, &t, 2, true).is_err());
    }

    #[test]
    fn test_uniform_f32() {
        let mut rng = Rng::new(42);
        let t = uniform::<f32>(&mut rng, vec![100]);
        assert_eq!(t.shape(), &[100]);
        for &v in t.as_slice() {
            assert!((0.0..1.0).contains(&v));
        }
    }
}
