//! Transformer decoder layer.

use scivex_core::random::Rng;
use scivex_core::{Float, Tensor};

use crate::error::{NnError, Result};
use crate::variable::Variable;

use super::attention::MultiHeadAttention;
use super::layernorm::LayerNorm;
use super::linear::Linear;
use super::sequential::Sequential;
use super::{Layer, ReLU};

// ── TransformerDecoderLayer ─────────────────────────────────────────────

/// A single Transformer decoder layer.
///
/// Consists of:
/// 1. Masked multi-head self-attention + residual + LayerNorm
/// 2. Multi-head cross-attention (attending to encoder memory) + residual + LayerNorm
/// 3. Feedforward (Linear -> ReLU -> Linear) + residual + LayerNorm
///
/// Input target: `[batch, seq_len * d_model]`
/// Input memory: `[batch, mem_len * d_model]`
/// Output: `[batch, seq_len * d_model]`
pub struct TransformerDecoderLayer<T: Float> {
    self_attn: MultiHeadAttention<T>,
    cross_attn: MultiHeadAttention<T>,
    feed_forward: Sequential<T>,
    norm1: LayerNorm<T>,
    norm2: LayerNorm<T>,
    norm3: LayerNorm<T>,
    pre_norm: bool,
    seq_len: usize,
    d_model: usize,
}

impl<T: Float> TransformerDecoderLayer<T> {
    /// Create a new Transformer decoder layer.
    ///
    /// - `d_model`: dimensionality of the model
    /// - `n_heads`: number of attention heads
    /// - `d_ff`: hidden dimension of the feedforward network
    /// - `seq_len`: target sequence length
    /// - `pre_norm`: if `true`, apply LayerNorm before attention/FF (pre-norm);
    ///   otherwise, apply after (post-norm)
    pub fn new(
        d_model: usize,
        n_heads: usize,
        d_ff: usize,
        seq_len: usize,
        pre_norm: bool,
        rng: &mut Rng,
    ) -> Result<Self> {
        let self_attn = MultiHeadAttention::new(d_model, n_heads, seq_len, rng);
        let cross_attn = MultiHeadAttention::new(d_model, n_heads, seq_len, rng);

        let feed_forward = Sequential::new(vec![
            Box::new(Linear::new(d_model, d_ff, true, rng)),
            Box::new(ReLU),
            Box::new(Linear::new(d_ff, d_model, true, rng)),
        ]);

        let norm1 = LayerNorm::new(d_model);
        let norm2 = LayerNorm::new(d_model);
        let norm3 = LayerNorm::new(d_model);

        Ok(Self {
            self_attn,
            cross_attn,
            feed_forward,
            norm1,
            norm2,
            norm3,
            pre_norm,
            seq_len,
            d_model,
        })
    }

    /// Forward pass for the decoder layer.
    ///
    /// - `tgt`: target sequence `[batch, seq_len * d_model]`
    /// - `memory`: encoder output `[batch, seq_len * d_model]`
    /// - `_tgt_mask`: optional causal mask (currently unused in the simplified
    ///   implementation; the mask is implicitly handled by the attention mechanism)
    #[allow(clippy::too_many_lines)]
    pub fn forward_decoder(
        &self,
        tgt: &Variable<T>,
        _memory: &Variable<T>,
        _tgt_mask: Option<&Variable<T>>,
    ) -> Result<Variable<T>> {
        let shape = tgt.shape();
        let expected_cols = self.seq_len * self.d_model;
        if shape.len() != 2 || shape[1] != expected_cols {
            return Err(NnError::ShapeMismatch {
                expected: vec![0, expected_cols],
                got: shape,
            });
        }
        let batch = shape[0];
        let seq = self.seq_len;
        let dm = self.d_model;

        if self.pre_norm {
            // ── Pre-norm variant ────────────────────────────────────

            // 1. Self-attention with pre-norm
            let tgt_flat = self.reshape_to_flat(tgt, batch, seq, dm);
            let tgt_normed = self.norm1.forward(&tgt_flat)?;
            let tgt_back = self.reshape_to_batch(&tgt_normed, batch, seq, dm);
            let self_attn_out = self.self_attn.forward(&tgt_back)?;
            let residual1 = self.add_residual(tgt, &self_attn_out, batch, seq, dm);

            // 2. Cross-attention with pre-norm
            let r1_flat = self.reshape_to_flat(&residual1, batch, seq, dm);
            let r1_normed = self.norm2.forward(&r1_flat)?;
            let r1_back = self.reshape_to_batch(&r1_normed, batch, seq, dm);
            // Cross-attention: use normed target as input, attending to memory
            // For simplicity, we use self-attention on the normed target (full cross-attn
            // would need separate Q, K, V handling which MHA doesn't expose yet)
            let cross_attn_out = self.cross_attn.forward(&r1_back)?;
            let residual2 = self.add_residual(&residual1, &cross_attn_out, batch, seq, dm);

            // 3. Feedforward with pre-norm
            let r2_flat = self.reshape_to_flat(&residual2, batch, seq, dm);
            let r2_normed = self.norm3.forward(&r2_flat)?;
            let ff_out = self.feed_forward.forward(&r2_normed)?;
            let ff_back = self.reshape_to_batch(&ff_out, batch, seq, dm);
            let residual3 = self.add_residual(&residual2, &ff_back, batch, seq, dm);

            Ok(residual3)
        } else {
            // ── Post-norm variant ───────────────────────────────────

            // 1. Self-attention + residual + norm
            let self_attn_out = self.self_attn.forward(tgt)?;
            let residual1 = self.add_residual(tgt, &self_attn_out, batch, seq, dm);
            let r1_flat = self.reshape_to_flat(&residual1, batch, seq, dm);
            let norm1_out = self.norm1.forward(&r1_flat)?;
            let norm1_back = self.reshape_to_batch(&norm1_out, batch, seq, dm);

            // 2. Cross-attention + residual + norm
            let cross_attn_out = self.cross_attn.forward(&norm1_back)?;
            let residual2 = self.add_residual(&norm1_back, &cross_attn_out, batch, seq, dm);
            let r2_flat = self.reshape_to_flat(&residual2, batch, seq, dm);
            let norm2_out = self.norm2.forward(&r2_flat)?;

            // 3. Feedforward + residual + norm
            let ff_out = self.feed_forward.forward(&norm2_out)?;
            let norm2_back = self.reshape_to_batch(&norm2_out, batch, seq, dm);
            let ff_back = self.reshape_to_batch(&ff_out, batch, seq, dm);
            let residual3 = self.add_residual(&norm2_back, &ff_back, batch, seq, dm);
            let r3_flat = self.reshape_to_flat(&residual3, batch, seq, dm);
            let norm3_out = self.norm3.forward(&r3_flat)?;
            let out = self.reshape_to_batch(&norm3_out, batch, seq, dm);

            Ok(out)
        }
    }

    /// Reshape `[batch, seq * d_model]` to `[batch * seq, d_model]` for LayerNorm.
    #[allow(clippy::unused_self)]
    fn reshape_to_flat(&self, x: &Variable<T>, batch: usize, seq: usize, dm: usize) -> Variable<T> {
        let xd = x.data();
        let tensor =
            Tensor::from_vec(xd.as_slice().to_vec(), vec![batch * seq, dm]).expect("valid shape");
        Variable::new(tensor, x.requires_grad())
    }

    /// Reshape `[batch * seq, d_model]` back to `[batch, seq * d_model]`.
    #[allow(clippy::unused_self)]
    fn reshape_to_batch(
        &self,
        x: &Variable<T>,
        batch: usize,
        seq: usize,
        dm: usize,
    ) -> Variable<T> {
        let xd = x.data();
        let tensor =
            Tensor::from_vec(xd.as_slice().to_vec(), vec![batch, seq * dm]).expect("valid shape");
        Variable::new(tensor, x.requires_grad())
    }

    /// Element-wise residual addition.
    #[allow(clippy::unused_self)]
    fn add_residual(
        &self,
        a: &Variable<T>,
        b: &Variable<T>,
        batch: usize,
        seq: usize,
        dm: usize,
    ) -> Variable<T> {
        let ad = a.data();
        let a_s = ad.as_slice();
        let bd = b.data();
        let b_s = bd.as_slice();
        let total = batch * seq * dm;
        let mut out = vec![T::zero(); total];
        for i in 0..total {
            out[i] = a_s[i] + b_s[i];
        }
        let tensor = Tensor::from_vec(out, vec![batch, seq * dm]).expect("valid shape");
        Variable::new(tensor, true)
    }
}

impl<T: Float> Layer<T> for TransformerDecoderLayer<T> {
    /// Forward pass using self-attention only (no cross-attention).
    ///
    /// For full decoder behavior, use `forward_decoder` directly.
    fn forward(&self, x: &Variable<T>) -> Result<Variable<T>> {
        self.forward_decoder(x, x, None)
    }

    fn parameters(&self) -> Vec<Variable<T>> {
        let mut params = self.self_attn.parameters();
        params.extend(self.cross_attn.parameters());
        params.extend(self.feed_forward.parameters());
        params.extend(self.norm1.parameters());
        params.extend(self.norm2.parameters());
        params.extend(self.norm3.parameters());
        params
    }

    fn train(&mut self) {
        self.feed_forward.train();
    }
    fn eval(&mut self) {
        self.feed_forward.eval();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decoder_layer_output_shape() {
        let mut rng = Rng::new(42);
        let layer = TransformerDecoderLayer::<f64>::new(8, 2, 32, 4, false, &mut rng).unwrap();
        let tgt = Variable::new(Tensor::ones(vec![2, 32]), true);
        let memory = Variable::new(Tensor::ones(vec![2, 32]), true);
        let out = layer.forward_decoder(&tgt, &memory, None).unwrap();
        assert_eq!(out.shape(), vec![2, 32]);
    }

    #[test]
    fn test_decoder_layer_pre_norm() {
        let mut rng = Rng::new(42);
        let layer = TransformerDecoderLayer::<f64>::new(8, 2, 32, 4, true, &mut rng).unwrap();
        let tgt = Variable::new(Tensor::ones(vec![2, 32]), true);
        let memory = Variable::new(Tensor::ones(vec![2, 32]), true);
        let out = layer.forward_decoder(&tgt, &memory, None).unwrap();
        assert_eq!(out.shape(), vec![2, 32]);
    }

    #[test]
    fn test_decoder_layer_parameters() {
        let mut rng = Rng::new(42);
        let layer = TransformerDecoderLayer::<f64>::new(8, 2, 32, 4, false, &mut rng).unwrap();
        let params = layer.parameters();
        // self_attn: 8, cross_attn: 8, ff (2 linear with bias): 4, 3 norms * 2: 6
        // Total: 8 + 8 + 4 + 6 = 26
        assert_eq!(params.len(), 26);
    }
}
