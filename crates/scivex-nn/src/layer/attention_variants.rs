//! Attention mechanism variants: Multi-Query, Grouped-Query, and Flash Attention.

use scivex_core::random::Rng;
use scivex_core::{Float, Tensor};

use crate::error::{NnError, Result};
use crate::variable::Variable;

use super::Layer;
use super::linear::Linear;

// Re-use the row_softmax helper from the attention module pattern.
fn row_softmax<T: Float>(data: &mut [T], rows: usize, cols: usize) {
    for i in 0..rows {
        let row = &mut data[i * cols..(i + 1) * cols];
        let max = row.iter().copied().fold(T::neg_infinity(), T::max);
        let mut sum = T::zero();
        for v in row.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        if sum > T::zero() {
            for v in row.iter_mut() {
                *v /= sum;
            }
        }
    }
}

// ── MultiQueryAttention ──────────────────────────────────────────────────

/// Multi-Query Attention (MQA).
///
/// All query heads have their own projections, but K and V share a **single**
/// head.  This dramatically reduces KV-cache memory compared to standard MHA.
///
/// Input:  `[batch, seq_len * d_model]`
/// Output: `[batch, seq_len * d_model]`
pub struct MultiQueryAttention<T: Float> {
    w_q: Linear<T>,
    w_k: Linear<T>,
    w_v: Linear<T>,
    w_o: Linear<T>,
    num_heads: usize,
    d_model: usize,
    d_k: usize,
    seq_len: usize,
}

impl<T: Float> MultiQueryAttention<T> {
    /// Create a new Multi-Query Attention layer.
    ///
    /// `d_model` must be divisible by `num_heads`.
    /// K and V are projected to a single head of size `d_k = d_model / num_heads`.
    #[allow(clippy::manual_is_multiple_of)]
    pub fn new(d_model: usize, num_heads: usize, seq_len: usize, rng: &mut Rng) -> Result<Self> {
        if d_model % num_heads != 0 {
            return Err(NnError::InvalidParameter {
                name: "d_model",
                reason: "d_model must be divisible by num_heads",
            });
        }
        let d_k = d_model / num_heads;
        Ok(Self {
            w_q: Linear::new(d_model, d_model, true, rng), // [d_model -> d_model]
            w_k: Linear::new(d_model, d_k, true, rng),     // [d_model -> d_k] single head
            w_v: Linear::new(d_model, d_k, true, rng),     // [d_model -> d_k] single head
            w_o: Linear::new(d_model, d_model, true, rng),
            num_heads,
            d_model,
            d_k,
            seq_len,
        })
    }
}

impl<T: Float> Layer<T> for MultiQueryAttention<T> {
    #[allow(clippy::too_many_lines)]
    fn forward(&self, x: &Variable<T>) -> Result<Variable<T>> {
        let shape = x.shape();
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
        let nh = self.num_heads;
        let dk = self.d_k;

        // Reshape to [batch * seq, d_model]
        let xd = x.data();
        let xs = xd.as_slice();
        let flat_tensor =
            Tensor::from_vec(xs.to_vec(), vec![batch * seq, dm]).expect("valid shape");
        let flat_var = Variable::new(flat_tensor, x.requires_grad());

        // Q: [batch * seq, d_model],  K: [batch * seq, d_k],  V: [batch * seq, d_k]
        let q_flat = self.w_q.forward(&flat_var)?;
        let k_flat = self.w_k.forward(&flat_var)?;
        let v_flat = self.w_v.forward(&flat_var)?;

        let qd = q_flat.data();
        let qs = qd.as_slice();
        let kd = k_flat.data();
        let ks = kd.as_slice();
        let vd = v_flat.data();
        let vs = vd.as_slice();

        let scale = T::from_f64(1.0 / (dk as f64).sqrt());

        let mut attn_out = vec![T::zero(); batch * seq * dm];

        for b in 0..batch {
            // Extract single K, V head for this batch: [seq, d_k]
            let mut k_shared = vec![T::zero(); seq * dk];
            let mut v_shared = vec![T::zero(); seq * dk];
            for s in 0..seq {
                let base = (b * seq + s) * dk;
                for d in 0..dk {
                    k_shared[s * dk + d] = ks[base + d];
                    v_shared[s * dk + d] = vs[base + d];
                }
            }

            for h in 0..nh {
                // Extract Q_h for this head: [seq, d_k]
                let mut q_h = vec![T::zero(); seq * dk];
                for s in 0..seq {
                    let base = (b * seq + s) * dm + h * dk;
                    for d in 0..dk {
                        q_h[s * dk + d] = qs[base + d];
                    }
                }

                // scores = Q_h @ K^T * scale -> [seq, seq]
                let mut scores = vec![T::zero(); seq * seq];
                for i in 0..seq {
                    for j in 0..seq {
                        let mut sum = T::zero();
                        for d in 0..dk {
                            sum += q_h[i * dk + d] * k_shared[j * dk + d];
                        }
                        scores[i * seq + j] = sum * scale;
                    }
                }

                row_softmax(&mut scores, seq, seq);

                // out_h = attn @ V -> [seq, d_k]
                for i in 0..seq {
                    for d in 0..dk {
                        let mut sum = T::zero();
                        for j in 0..seq {
                            sum += scores[i * seq + j] * v_shared[j * dk + d];
                        }
                        attn_out[b * seq * dm + i * dm + h * dk + d] = sum;
                    }
                }
            }
        }

        // Output projection
        let concat_tensor = Tensor::from_vec(attn_out, vec![batch * seq, dm]).expect("valid shape");
        let concat_var = Variable::new(concat_tensor, true);
        let projected = self.w_o.forward(&concat_var)?;

        let pd = projected.data();
        let ps = pd.as_slice();
        let out_tensor = Tensor::from_vec(ps.to_vec(), vec![batch, seq * dm]).expect("valid shape");

        let grad_fn = Box::new(move |g: &Tensor<T>| {
            let gs = g.as_slice();
            let gx = gs.to_vec();
            vec![
                Tensor::from_vec(gx, vec![batch, seq * dm]).expect("valid"),
                Tensor::zeros(vec![batch * seq, dm]),
                Tensor::zeros(vec![batch * seq, dk]),
                Tensor::zeros(vec![batch * seq, dk]),
                Tensor::from_vec(gs.to_vec(), vec![batch * seq, dm]).expect("valid"),
            ]
        });

        Ok(Variable::from_op(
            out_tensor,
            vec![x.clone(), q_flat, k_flat, v_flat, projected],
            grad_fn,
        ))
    }

    fn parameters(&self) -> Vec<Variable<T>> {
        let mut params = self.w_q.parameters();
        params.extend(self.w_k.parameters());
        params.extend(self.w_v.parameters());
        params.extend(self.w_o.parameters());
        params
    }

    fn train(&mut self) {}
    fn eval(&mut self) {}
}

// ── GroupedQueryAttention ─────────────────────────────────────────────────

/// Grouped-Query Attention (GQA).
///
/// Queries are divided into groups; each group shares K/V heads (Llama 2 style).
/// When `num_kv_heads == 1` this is equivalent to MQA.
/// When `num_kv_heads == num_heads` this is equivalent to standard MHA.
///
/// Input:  `[batch, seq_len * d_model]`
/// Output: `[batch, seq_len * d_model]`
pub struct GroupedQueryAttention<T: Float> {
    w_q: Linear<T>,
    w_k: Linear<T>,
    w_v: Linear<T>,
    w_o: Linear<T>,
    num_heads: usize,
    num_kv_heads: usize,
    d_model: usize,
    d_k: usize,
    seq_len: usize,
}

impl<T: Float> GroupedQueryAttention<T> {
    /// Create a new Grouped-Query Attention layer.
    ///
    /// `d_model` must be divisible by `num_heads`, and `num_heads` must be
    /// divisible by `num_kv_heads`.
    #[allow(clippy::manual_is_multiple_of)]
    pub fn new(
        d_model: usize,
        num_heads: usize,
        num_kv_heads: usize,
        seq_len: usize,
        rng: &mut Rng,
    ) -> Result<Self> {
        if d_model % num_heads != 0 {
            return Err(NnError::InvalidParameter {
                name: "d_model",
                reason: "d_model must be divisible by num_heads",
            });
        }
        if num_heads % num_kv_heads != 0 {
            return Err(NnError::InvalidParameter {
                name: "num_heads",
                reason: "num_heads must be divisible by num_kv_heads",
            });
        }
        let d_k = d_model / num_heads;
        let kv_dim = num_kv_heads * d_k;
        Ok(Self {
            w_q: Linear::new(d_model, d_model, true, rng), // [d_model -> d_model]
            w_k: Linear::new(d_model, kv_dim, true, rng),  // [d_model -> kv_dim]
            w_v: Linear::new(d_model, kv_dim, true, rng),  // [d_model -> kv_dim]
            w_o: Linear::new(d_model, d_model, true, rng),
            num_heads,
            num_kv_heads,
            d_model,
            d_k,
            seq_len,
        })
    }
}

impl<T: Float> Layer<T> for GroupedQueryAttention<T> {
    #[allow(clippy::too_many_lines)]
    fn forward(&self, x: &Variable<T>) -> Result<Variable<T>> {
        let shape = x.shape();
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
        let nh = self.num_heads;
        let nkv = self.num_kv_heads;
        let dk = self.d_k;
        let kv_dim = nkv * dk;
        let heads_per_group = nh / nkv;

        // Reshape to [batch * seq, d_model]
        let xd = x.data();
        let xs = xd.as_slice();
        let flat_tensor =
            Tensor::from_vec(xs.to_vec(), vec![batch * seq, dm]).expect("valid shape");
        let flat_var = Variable::new(flat_tensor, x.requires_grad());

        // Q: [batch * seq, d_model],  K: [batch * seq, kv_dim],  V: [batch * seq, kv_dim]
        let q_flat = self.w_q.forward(&flat_var)?;
        let k_flat = self.w_k.forward(&flat_var)?;
        let v_flat = self.w_v.forward(&flat_var)?;

        let qd = q_flat.data();
        let qs = qd.as_slice();
        let kd = k_flat.data();
        let ks = kd.as_slice();
        let vd = v_flat.data();
        let vs = vd.as_slice();

        let scale = T::from_f64(1.0 / (dk as f64).sqrt());

        let mut attn_out = vec![T::zero(); batch * seq * dm];

        for b in 0..batch {
            for h in 0..nh {
                let kv_head = h / heads_per_group; // which KV head this query head uses

                // Extract Q_h: [seq, d_k]
                let mut q_h = vec![T::zero(); seq * dk];
                for s in 0..seq {
                    let base_q = (b * seq + s) * dm + h * dk;
                    for d in 0..dk {
                        q_h[s * dk + d] = qs[base_q + d];
                    }
                }

                // Extract K_kv, V_kv for the shared kv_head: [seq, d_k]
                let mut k_h = vec![T::zero(); seq * dk];
                let mut v_h = vec![T::zero(); seq * dk];
                for s in 0..seq {
                    let base_kv = (b * seq + s) * kv_dim + kv_head * dk;
                    for d in 0..dk {
                        k_h[s * dk + d] = ks[base_kv + d];
                        v_h[s * dk + d] = vs[base_kv + d];
                    }
                }

                // scores = Q_h @ K_h^T * scale -> [seq, seq]
                let mut scores = vec![T::zero(); seq * seq];
                for i in 0..seq {
                    for j in 0..seq {
                        let mut sum = T::zero();
                        for d in 0..dk {
                            sum += q_h[i * dk + d] * k_h[j * dk + d];
                        }
                        scores[i * seq + j] = sum * scale;
                    }
                }

                row_softmax(&mut scores, seq, seq);

                // out_h = attn @ V_h -> [seq, d_k]
                for i in 0..seq {
                    for d in 0..dk {
                        let mut sum = T::zero();
                        for j in 0..seq {
                            sum += scores[i * seq + j] * v_h[j * dk + d];
                        }
                        attn_out[b * seq * dm + i * dm + h * dk + d] = sum;
                    }
                }
            }
        }

        // Output projection
        let concat_tensor = Tensor::from_vec(attn_out, vec![batch * seq, dm]).expect("valid shape");
        let concat_var = Variable::new(concat_tensor, true);
        let projected = self.w_o.forward(&concat_var)?;

        let pd = projected.data();
        let ps = pd.as_slice();
        let out_tensor = Tensor::from_vec(ps.to_vec(), vec![batch, seq * dm]).expect("valid shape");

        let grad_fn = Box::new(move |g: &Tensor<T>| {
            let gs = g.as_slice();
            let gx = gs.to_vec();
            vec![
                Tensor::from_vec(gx, vec![batch, seq * dm]).expect("valid"),
                Tensor::zeros(vec![batch * seq, dm]),
                Tensor::zeros(vec![batch * seq, kv_dim]),
                Tensor::zeros(vec![batch * seq, kv_dim]),
                Tensor::from_vec(gs.to_vec(), vec![batch * seq, dm]).expect("valid"),
            ]
        });

        Ok(Variable::from_op(
            out_tensor,
            vec![x.clone(), q_flat, k_flat, v_flat, projected],
            grad_fn,
        ))
    }

    fn parameters(&self) -> Vec<Variable<T>> {
        let mut params = self.w_q.parameters();
        params.extend(self.w_k.parameters());
        params.extend(self.w_v.parameters());
        params.extend(self.w_o.parameters());
        params
    }

    fn train(&mut self) {}
    fn eval(&mut self) {}
}

// ── FlashAttention ───────────────────────────────────────────────────────

/// Memory-efficient "Flash" Attention.
///
/// Computes scaled dot-product attention in tiles without materializing the
/// full `[seq, seq]` attention matrix.  Uses the online softmax algorithm
/// (running max + running sum) to accumulate results block-by-block.
///
/// Input:  `[batch, seq_len * d_model]`
/// Output: `[batch, seq_len * d_model]`
pub struct FlashAttention<T: Float> {
    w_q: Linear<T>,
    w_k: Linear<T>,
    w_v: Linear<T>,
    w_o: Linear<T>,
    num_heads: usize,
    d_model: usize,
    d_k: usize,
    seq_len: usize,
    block_size: usize,
}

impl<T: Float> FlashAttention<T> {
    /// Create a new Flash Attention layer.
    ///
    /// `d_model` must be divisible by `num_heads`.
    /// `block_size` controls the tile size for the block-wise computation.
    #[allow(clippy::manual_is_multiple_of)]
    pub fn new(
        d_model: usize,
        num_heads: usize,
        seq_len: usize,
        block_size: usize,
        rng: &mut Rng,
    ) -> Result<Self> {
        if d_model % num_heads != 0 {
            return Err(NnError::InvalidParameter {
                name: "d_model",
                reason: "d_model must be divisible by num_heads",
            });
        }
        if block_size == 0 {
            return Err(NnError::InvalidParameter {
                name: "block_size",
                reason: "block_size must be > 0",
            });
        }
        let d_k = d_model / num_heads;
        Ok(Self {
            w_q: Linear::new(d_model, d_model, true, rng),
            w_k: Linear::new(d_model, d_model, true, rng),
            w_v: Linear::new(d_model, d_model, true, rng),
            w_o: Linear::new(d_model, d_model, true, rng),
            num_heads,
            d_model,
            d_k,
            seq_len,
            block_size,
        })
    }
}

impl<T: Float> Layer<T> for FlashAttention<T> {
    #[allow(clippy::too_many_lines)]
    fn forward(&self, x: &Variable<T>) -> Result<Variable<T>> {
        let shape = x.shape();
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
        let nh = self.num_heads;
        let dk = self.d_k;
        let bs = self.block_size;

        // Reshape to [batch * seq, d_model]
        let xd = x.data();
        let xs = xd.as_slice();
        let flat_tensor =
            Tensor::from_vec(xs.to_vec(), vec![batch * seq, dm]).expect("valid shape");
        let flat_var = Variable::new(flat_tensor, x.requires_grad());

        let q_flat = self.w_q.forward(&flat_var)?;
        let k_flat = self.w_k.forward(&flat_var)?;
        let v_flat = self.w_v.forward(&flat_var)?;

        let qd = q_flat.data();
        let qs = qd.as_slice();
        let kd = k_flat.data();
        let ks = kd.as_slice();
        let vd = v_flat.data();
        let vs = vd.as_slice();

        let scale = T::from_f64(1.0 / (dk as f64).sqrt());

        let mut attn_out = vec![T::zero(); batch * seq * dm];

        for b in 0..batch {
            for h in 0..nh {
                // Extract Q_h, K_h, V_h: [seq, d_k]
                let mut q_h = vec![T::zero(); seq * dk];
                let mut k_h = vec![T::zero(); seq * dk];
                let mut v_h = vec![T::zero(); seq * dk];

                for s in 0..seq {
                    let base = (b * seq + s) * dm + h * dk;
                    for d in 0..dk {
                        q_h[s * dk + d] = qs[base + d];
                        k_h[s * dk + d] = ks[base + d];
                        v_h[s * dk + d] = vs[base + d];
                    }
                }

                // Block-wise (tiled) attention with online softmax.
                // For each block of query rows, accumulate attention over K/V blocks.
                //
                // Per query row i we maintain:
                //   m_i : running max of scores
                //   l_i : running sum of exp(score - m_i)
                //   o_i : running weighted sum (unnormalised)
                let mut m_vec = vec![T::neg_infinity(); seq]; // running max per query row
                let mut l_vec = vec![T::zero(); seq]; // running sum per query row
                let mut o_buf = vec![T::zero(); seq * dk]; // running output per query row

                // Iterate over key blocks
                let num_k_blocks = seq.div_ceil(bs);
                for kb in 0..num_k_blocks {
                    let k_start = kb * bs;
                    let k_end = (k_start + bs).min(seq);
                    let k_len = k_end - k_start;

                    // Iterate over query blocks
                    let num_q_blocks = seq.div_ceil(bs);
                    for qb in 0..num_q_blocks {
                        let q_start = qb * bs;
                        let q_end = (q_start + bs).min(seq);

                        for qi in q_start..q_end {
                            // Compute scores for this query row against the key block
                            // and update online softmax accumulators.
                            let old_m = m_vec[qi];

                            // Find new local max
                            let mut block_max = T::neg_infinity();
                            for kj in 0..k_len {
                                let mut dot = T::zero();
                                for d in 0..dk {
                                    dot += q_h[qi * dk + d] * k_h[(k_start + kj) * dk + d];
                                }
                                let s = dot * scale;
                                if s > block_max {
                                    block_max = s;
                                }
                            }

                            let new_m = if block_max > old_m { block_max } else { old_m };

                            // Correction factor for old accumulators
                            let correction = (old_m - new_m).exp();

                            // Scale old accumulator
                            l_vec[qi] *= correction;
                            for d in 0..dk {
                                o_buf[qi * dk + d] *= correction;
                            }

                            // Accumulate new block
                            for kj in 0..k_len {
                                let mut dot = T::zero();
                                for d in 0..dk {
                                    dot += q_h[qi * dk + d] * k_h[(k_start + kj) * dk + d];
                                }
                                let s = dot * scale;
                                let w = (s - new_m).exp();
                                l_vec[qi] += w;
                                for d in 0..dk {
                                    o_buf[qi * dk + d] += w * v_h[(k_start + kj) * dk + d];
                                }
                            }

                            m_vec[qi] = new_m;
                        }
                    }
                }

                // Normalise: o_i / l_i
                for i in 0..seq {
                    let li = l_vec[i];
                    if li > T::zero() {
                        for d in 0..dk {
                            let val = o_buf[i * dk + d] / li;
                            attn_out[b * seq * dm + i * dm + h * dk + d] = val;
                        }
                    }
                }
            }
        }

        // Output projection
        let concat_tensor = Tensor::from_vec(attn_out, vec![batch * seq, dm]).expect("valid shape");
        let concat_var = Variable::new(concat_tensor, true);
        let projected = self.w_o.forward(&concat_var)?;

        let pd = projected.data();
        let ps = pd.as_slice();
        let out_tensor = Tensor::from_vec(ps.to_vec(), vec![batch, seq * dm]).expect("valid shape");

        let grad_fn = Box::new(move |g: &Tensor<T>| {
            let gs = g.as_slice();
            let gx = gs.to_vec();
            vec![
                Tensor::from_vec(gx, vec![batch, seq * dm]).expect("valid"),
                Tensor::zeros(vec![batch * seq, dm]),
                Tensor::zeros(vec![batch * seq, dm]),
                Tensor::zeros(vec![batch * seq, dm]),
                Tensor::from_vec(gs.to_vec(), vec![batch * seq, dm]).expect("valid"),
            ]
        });

        Ok(Variable::from_op(
            out_tensor,
            vec![x.clone(), q_flat, k_flat, v_flat, projected],
            grad_fn,
        ))
    }

    fn parameters(&self) -> Vec<Variable<T>> {
        let mut params = self.w_q.parameters();
        params.extend(self.w_k.parameters());
        params.extend(self.w_v.parameters());
        params.extend(self.w_o.parameters());
        params
    }

    fn train(&mut self) {}
    fn eval(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use scivex_core::Tensor;

    #[test]
    fn test_mqa_output_shape() {
        let mut rng = Rng::new(42);
        let mqa = MultiQueryAttention::<f64>::new(8, 2, 4, &mut rng).unwrap();
        // batch=2, seq=4, d_model=8 -> input [2, 32]
        let x = Variable::new(Tensor::ones(vec![2, 32]), true);
        let y = mqa.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![2, 32]);
    }

    #[test]
    fn test_mqa_fewer_kv_params() {
        use crate::layer::attention::MultiHeadAttention;

        let mut rng = Rng::new(42);
        let mha = MultiHeadAttention::<f64>::new(8, 2, 4, &mut rng);
        let mha_params: usize = mha
            .parameters()
            .iter()
            .map(|p| p.data().as_slice().len())
            .sum();

        let mut rng2 = Rng::new(42);
        let mqa = MultiQueryAttention::<f64>::new(8, 2, 4, &mut rng2).unwrap();
        let mqa_params: usize = mqa
            .parameters()
            .iter()
            .map(|p| p.data().as_slice().len())
            .sum();

        // MQA should have fewer total parameter elements because K, V project
        // to d_k instead of d_model.
        assert!(
            mqa_params < mha_params,
            "MQA params ({mqa_params}) should be fewer than MHA params ({mha_params})"
        );
    }

    #[test]
    fn test_gqa_output_shape() {
        let mut rng = Rng::new(42);
        let gqa = GroupedQueryAttention::<f64>::new(8, 4, 2, 4, &mut rng).unwrap();
        // batch=1, seq=4, d_model=8 -> input [1, 32]
        let x = Variable::new(Tensor::ones(vec![1, 32]), true);
        let y = gqa.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![1, 32]);
    }

    #[test]
    fn test_gqa_num_heads_divisibility() {
        let mut rng = Rng::new(42);
        // num_heads=4, num_kv_heads=3 -> 4 % 3 != 0, should fail
        let result = GroupedQueryAttention::<f64>::new(8, 4, 3, 4, &mut rng);
        assert!(result.is_err());
    }

    #[test]
    fn test_flash_attention_output_shape() {
        let mut rng = Rng::new(42);
        let flash = FlashAttention::<f64>::new(8, 2, 4, 2, &mut rng).unwrap();
        // batch=2, seq=4, d_model=8 -> input [2, 32]
        let x = Variable::new(Tensor::ones(vec![2, 32]), true);
        let y = flash.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![2, 32]);
    }
}
