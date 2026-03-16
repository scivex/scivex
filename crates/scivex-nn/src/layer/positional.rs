//! Positional encoding layers for Transformer architectures.

use scivex_core::{Float, Tensor};

use crate::error::{NnError, Result};
use crate::variable::Variable;

use super::Layer;

// ── Sinusoidal Positional Encoding ──────────────────────────────────────

/// Sinusoidal positional encoding (Vaswani et al., 2017).
///
/// Precomputes a `[max_len, d_model]` encoding table using sine/cosine functions
/// at different frequencies. During forward, the encoding is added to the input.
///
/// Input: `[batch, seq_len * d_model]`
/// Output: `[batch, seq_len * d_model]` (with positional encoding added)
pub struct SinusoidalPositionalEncoding<T: Float> {
    d_model: usize,
    max_len: usize,
    encoding: Variable<T>, // precomputed [max_len, d_model]
}

impl<T: Float> SinusoidalPositionalEncoding<T> {
    /// Create a new sinusoidal positional encoding.
    ///
    /// - `d_model`: dimensionality of the model
    /// - `max_len`: maximum sequence length supported
    pub fn new(d_model: usize, max_len: usize) -> Self {
        let mut data = vec![T::zero(); max_len * d_model];

        for pos in 0..max_len {
            for i in 0..d_model {
                let angle = T::from_f64(pos as f64)
                    / T::from_f64((10000.0_f64).powf(2.0 * (i / 2) as f64 / d_model as f64));
                if i % 2 == 0 {
                    data[pos * d_model + i] = angle.sin();
                } else {
                    data[pos * d_model + i] = angle.cos();
                }
            }
        }

        let tensor = Tensor::from_vec(data, vec![max_len, d_model]).expect("valid shape");
        let encoding = Variable::new(tensor, false);

        Self {
            d_model,
            max_len,
            encoding,
        }
    }

    /// Return the precomputed encoding variable.
    pub fn encoding(&self) -> &Variable<T> {
        &self.encoding
    }

    /// Return the maximum sequence length.
    pub fn max_len(&self) -> usize {
        self.max_len
    }

    /// Return the model dimension.
    pub fn d_model(&self) -> usize {
        self.d_model
    }
}

impl<T: Float> Layer<T> for SinusoidalPositionalEncoding<T> {
    fn forward(&self, x: &Variable<T>) -> Result<Variable<T>> {
        let shape = x.shape();
        if shape.len() != 2 {
            return Err(NnError::ShapeMismatch {
                expected: vec![0, 0],
                got: shape,
            });
        }
        let batch = shape[0];
        let total = shape[1];
        #[allow(clippy::manual_is_multiple_of)]
        if total % self.d_model != 0 {
            return Err(NnError::ShapeMismatch {
                expected: vec![batch, self.d_model],
                got: shape,
            });
        }
        let seq_len = total / self.d_model;
        if seq_len > self.max_len {
            return Err(NnError::InvalidParameter {
                name: "seq_len",
                reason: "sequence length exceeds max_len",
            });
        }

        let xd = x.data();
        let xs = xd.as_slice();
        let enc_d = self.encoding.data();
        let enc_s = enc_d.as_slice();
        let dm = self.d_model;

        let mut out = vec![T::zero(); batch * total];
        for b in 0..batch {
            for s in 0..seq_len {
                for d in 0..dm {
                    let idx = b * total + s * dm + d;
                    out[idx] = xs[idx] + enc_s[s * dm + d];
                }
            }
        }

        let out_tensor = Tensor::from_vec(out, vec![batch, total]).expect("valid shape");

        let grad_fn = Box::new(move |g: &Tensor<T>| vec![g.clone()]);

        Ok(Variable::from_op(out_tensor, vec![x.clone()], grad_fn))
    }

    fn parameters(&self) -> Vec<Variable<T>> {
        // Positional encoding is not learnable
        Vec::new()
    }

    fn train(&mut self) {}
    fn eval(&mut self) {}
}

// ── Rotary Positional Encoding (RoPE) ───────────────────────────────────

/// Rotary Position Embedding (RoPE) (Su et al., 2021).
///
/// Applies rotation to pairs of dimensions in the input. Unlike sinusoidal
/// encoding, RoPE is applied multiplicatively (rotation) rather than additively.
///
/// `d_model` must be even because rotations are applied to pairs.
pub struct RotaryPositionalEncoding<T: Float> {
    d_model: usize,
    base: T,
}

impl<T: Float> RotaryPositionalEncoding<T> {
    /// Create a new RoPE encoder.
    ///
    /// - `d_model`: dimensionality (must be even)
    /// - `base`: base for frequency computation (typically 10000.0)
    pub fn new(d_model: usize, base: T) -> Result<Self> {
        #[allow(clippy::manual_is_multiple_of)]
        if d_model % 2 != 0 {
            return Err(NnError::InvalidParameter {
                name: "d_model",
                reason: "d_model must be even for RoPE",
            });
        }
        Ok(Self { d_model, base })
    }

    /// Apply rotary positional encoding to input.
    ///
    /// Input: `[batch, seq_len * d_model]`
    /// Output: `[batch, seq_len * d_model]`
    pub fn apply(&self, x: &Variable<T>, seq_len: usize) -> Result<Variable<T>> {
        let shape = x.shape();
        let expected_cols = seq_len * self.d_model;
        if shape.len() != 2 || shape[1] != expected_cols {
            return Err(NnError::ShapeMismatch {
                expected: vec![0, expected_cols],
                got: shape,
            });
        }
        let batch = shape[0];
        let dm = self.d_model;
        let half = dm / 2;

        let xd = x.data();
        let xs = xd.as_slice();

        // Precompute frequencies: theta_i = 1 / (base^(2i/d_model))
        let mut freqs = vec![T::zero(); half];
        for (i, freq) in freqs.iter_mut().enumerate().take(half) {
            let exp = T::from_f64(2.0 * i as f64 / dm as f64);
            *freq = T::one() / self.base.powf(exp);
        }

        let mut out = vec![T::zero(); batch * expected_cols];

        for b in 0..batch {
            for s in 0..seq_len {
                let pos = T::from_f64(s as f64);
                for i in 0..half {
                    let angle = pos * freqs[i];
                    let cos_a = angle.cos();
                    let sin_a = angle.sin();

                    let base_idx = b * expected_cols + s * dm;
                    let x0 = xs[base_idx + 2 * i];
                    let x1 = xs[base_idx + 2 * i + 1];

                    // Rotation: [cos -sin; sin cos] @ [x0; x1]
                    out[base_idx + 2 * i] = x0 * cos_a - x1 * sin_a;
                    out[base_idx + 2 * i + 1] = x0 * sin_a + x1 * cos_a;
                }
            }
        }

        let out_tensor = Tensor::from_vec(out, vec![batch, expected_cols]).expect("valid shape");

        let grad_fn = Box::new(move |g: &Tensor<T>| {
            // Inverse rotation for gradient (transpose of rotation matrix)
            let gs = g.as_slice();
            let mut gx = vec![T::zero(); batch * expected_cols];
            for b2 in 0..batch {
                for s2 in 0..seq_len {
                    let pos = T::from_f64(s2 as f64);
                    for i in 0..half {
                        let angle = pos * freqs[i];
                        let cos_a = angle.cos();
                        let sin_a = angle.sin();

                        let base_idx = b2 * expected_cols + s2 * dm;
                        let g0 = gs[base_idx + 2 * i];
                        let g1 = gs[base_idx + 2 * i + 1];

                        // Inverse rotation: [cos sin; -sin cos] @ [g0; g1]
                        gx[base_idx + 2 * i] = g0 * cos_a + g1 * sin_a;
                        gx[base_idx + 2 * i + 1] = -g0 * sin_a + g1 * cos_a;
                    }
                }
            }
            vec![Tensor::from_vec(gx, vec![batch, expected_cols]).expect("valid shape")]
        });

        Ok(Variable::from_op(out_tensor, vec![x.clone()], grad_fn))
    }

    /// Return the model dimension.
    pub fn d_model(&self) -> usize {
        self.d_model
    }
}

// ── Causal Mask ─────────────────────────────────────────────────────────

/// Generate a causal (lower-triangular) attention mask.
///
/// Returns a `[seq_len, seq_len]` variable where position `(i, j)` is `1.0`
/// if `j <= i` (allowed) and `0.0` otherwise (masked). This can be used with
/// masked attention to prevent attending to future positions.
pub fn causal_mask<T: Float>(seq_len: usize) -> Variable<T> {
    let mut data = vec![T::zero(); seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..=i {
            data[i * seq_len + j] = T::one();
        }
    }
    let tensor = Tensor::from_vec(data, vec![seq_len, seq_len]).expect("valid shape");
    Variable::new(tensor, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sinusoidal_output_shape() {
        let pe = SinusoidalPositionalEncoding::<f64>::new(8, 100);
        // batch=2, seq=4, d_model=8 -> [2, 32]
        let x = Variable::new(Tensor::zeros(vec![2, 32]), true);
        let y = pe.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![2, 32]);
    }

    #[test]
    fn test_sinusoidal_values_bounded() {
        let pe = SinusoidalPositionalEncoding::<f64>::new(8, 100);
        let enc = pe.encoding().data();
        for &v in enc.as_slice() {
            assert!(
                (-1.0..=1.0).contains(&v),
                "encoding value {v} out of [-1, 1]"
            );
        }
    }

    #[test]
    fn test_rope_output_shape() {
        let rope = RotaryPositionalEncoding::<f64>::new(8, 10000.0).unwrap();
        let x = Variable::new(Tensor::ones(vec![2, 32]), true);
        let y = rope.apply(&x, 4).unwrap();
        assert_eq!(y.shape(), vec![2, 32]);
    }

    #[test]
    fn test_rope_odd_d_model_error() {
        let result = RotaryPositionalEncoding::<f64>::new(7, 10000.0);
        assert!(result.is_err());
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_causal_mask_structure() {
        let mask = causal_mask::<f64>(4);
        let md = mask.data();
        let ms = md.as_slice();
        // Row 0: [1, 0, 0, 0]
        assert_eq!(ms[0], 1.0);
        assert_eq!(ms[1], 0.0);
        assert_eq!(ms[2], 0.0);
        assert_eq!(ms[3], 0.0);
        // Row 1: [1, 1, 0, 0]
        assert_eq!(ms[4], 1.0);
        assert_eq!(ms[5], 1.0);
        assert_eq!(ms[6], 0.0);
        assert_eq!(ms[7], 0.0);
        // Row 3: [1, 1, 1, 1]
        assert_eq!(ms[12], 1.0);
        assert_eq!(ms[13], 1.0);
        assert_eq!(ms[14], 1.0);
        assert_eq!(ms[15], 1.0);
    }
}
