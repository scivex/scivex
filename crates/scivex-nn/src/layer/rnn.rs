//! Recurrent layers — RNN, LSTM, GRU.
//!
//! All recurrent layers expect input shape `[batch, seq_len, input_size]`
//! (flattened row-major) and return `[batch, seq_len, hidden_size]`.
//!
//! Hidden state shape: `[batch, hidden_size]`.

use scivex_core::random::Rng;
use scivex_core::{Float, Tensor};

use crate::error::{NnError, Result};
use crate::init;
use crate::variable::Variable;

use super::Layer;

// ── Helpers ──────────────────────────────────────────────────────────────

/// Compute tanh element-wise via (exp(2x) - 1) / (exp(2x) + 1).
fn tanh_elem<T: Float>(v: T) -> T {
    let e2x = (v + v).exp();
    (e2x - T::one()) / (e2x + T::one())
}

/// Sigmoid element-wise.
fn sigmoid_elem<T: Float>(v: T) -> T {
    T::one() / (T::one() + (-v).exp())
}

/// Manual matmul: a [m, k] @ b [k, n] → out [m, n], accumulated into `out`.
fn matmul_acc<T: Float>(out: &mut [T], a: &[T], b: &[T], m: usize, k: usize, n: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = T::zero();
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            out[i * n + j] += sum;
        }
    }
}

// ── RNN ──────────────────────────────────────────────────────────────────

/// Simple (Elman) RNN layer.
///
/// `h_t = tanh(x_t @ W_ih^T + h_{t-1} @ W_hh^T + b_ih + b_hh)`
///
/// Input: `[batch, seq_len * input_size]`
/// Output: `[batch, seq_len * hidden_size]`
pub struct SimpleRNN<T: Float> {
    w_ih: Variable<T>, // [hidden, input]
    w_hh: Variable<T>, // [hidden, hidden]
    b_ih: Variable<T>, // [hidden]
    b_hh: Variable<T>, // [hidden]
    input_size: usize,
    hidden_size: usize,
    seq_len: usize,
}

impl<T: Float> SimpleRNN<T> {
    /// Create a new RNN layer.
    pub fn new(input_size: usize, hidden_size: usize, seq_len: usize, rng: &mut Rng) -> Self {
        let w_ih = Variable::new(
            init::xavier_uniform::<T>(&[hidden_size, input_size], rng),
            true,
        );
        let w_hh = Variable::new(
            init::xavier_uniform::<T>(&[hidden_size, hidden_size], rng),
            true,
        );
        let b_ih = Variable::new(Tensor::zeros(vec![hidden_size]), true);
        let b_hh = Variable::new(Tensor::zeros(vec![hidden_size]), true);
        Self {
            w_ih,
            w_hh,
            b_ih,
            b_hh,
            input_size,
            hidden_size,
            seq_len,
        }
    }
}

impl<T: Float> Layer<T> for SimpleRNN<T> {
    #[allow(clippy::too_many_lines)]
    fn forward(&self, x: &Variable<T>) -> Result<Variable<T>> {
        let shape = x.shape();
        let expected_cols = self.seq_len * self.input_size;
        if shape.len() != 2 || shape[1] != expected_cols {
            return Err(NnError::ShapeMismatch {
                expected: vec![0, expected_cols],
                got: shape,
            });
        }
        let batch = shape[0];
        let inp = self.input_size;
        let hid = self.hidden_size;
        let seq = self.seq_len;

        let xd = x.data();
        let xs = xd.as_slice();
        let wih = self.w_ih.data();
        let wihs = wih.as_slice();
        let whh = self.w_hh.data();
        let whhs = whh.as_slice();
        let bih = self.b_ih.data();
        let bihs = bih.as_slice();
        let bhh = self.b_hh.data();
        let bhhs = bhh.as_slice();

        // Transpose weights for matmul: W_ih^T [inp, hid], W_hh^T [hid, hid]
        let mut wiht = vec![T::zero(); inp * hid];
        for i in 0..hid {
            for j in 0..inp {
                wiht[j * hid + i] = wihs[i * inp + j];
            }
        }
        let mut whht = vec![T::zero(); hid * hid];
        for i in 0..hid {
            for j in 0..hid {
                whht[j * hid + i] = whhs[i * hid + j];
            }
        }

        let mut h = vec![T::zero(); batch * hid]; // hidden state
        let mut all_h = vec![T::zero(); batch * seq * hid]; // all hidden states
        // Store pre-activation for backward
        let mut all_pre_tanh = vec![T::zero(); batch * seq * hid];

        for t in 0..seq {
            // Extract x_t: [batch, inp]
            let mut xt = vec![T::zero(); batch * inp];
            for b in 0..batch {
                for j in 0..inp {
                    xt[b * inp + j] = xs[b * (seq * inp) + t * inp + j];
                }
            }

            // pre = x_t @ W_ih^T + h @ W_hh^T + b_ih + b_hh
            let mut pre = vec![T::zero(); batch * hid];
            matmul_acc(&mut pre, &xt, &wiht, batch, inp, hid);
            matmul_acc(&mut pre, &h, &whht, batch, hid, hid);
            for b in 0..batch {
                for j in 0..hid {
                    pre[b * hid + j] += bihs[j] + bhhs[j];
                }
            }

            // h = tanh(pre)
            for i in 0..batch * hid {
                all_pre_tanh[t * batch * hid + i] = pre[i];
                h[i] = tanh_elem(pre[i]);
            }

            // Store h_t
            for b in 0..batch {
                for j in 0..hid {
                    all_h[b * seq * hid + t * hid + j] = h[b * hid + j];
                }
            }
        }

        let out_tensor = Tensor::from_vec(all_h, vec![batch, seq * hid]).expect("valid shape");

        // Capture for backward
        let xs_cap = xs.to_vec();
        let wihs_cap = wihs.to_vec();
        let whhs_cap = whhs.to_vec();

        let grad_fn = Box::new(move |g: &Tensor<T>| {
            let gd = g.as_slice();

            let mut gx = vec![T::zero(); batch * seq * inp];
            let mut gwih = vec![T::zero(); hid * inp];
            let mut gwhh = vec![T::zero(); hid * hid];
            let mut gbih = vec![T::zero(); hid];
            let mut gbhh = vec![T::zero(); hid];

            let mut dh_next = vec![T::zero(); batch * hid];

            // BPTT: iterate backwards through time
            for t in (0..seq).rev() {
                // dh = grad_output_t + dh_next
                let mut dh = vec![T::zero(); batch * hid];
                for b in 0..batch {
                    for j in 0..hid {
                        dh[b * hid + j] = gd[b * seq * hid + t * hid + j] + dh_next[b * hid + j];
                    }
                }

                // d_pre = dh * (1 - tanh^2(pre))
                let mut d_pre = vec![T::zero(); batch * hid];
                for i in 0..batch * hid {
                    let th = tanh_elem(all_pre_tanh[t * batch * hid + i]);
                    d_pre[i] = dh[i] * (T::one() - th * th);
                }

                // Reconstruct h_{t-1}
                let mut h_prev = vec![T::zero(); batch * hid];
                if t > 0 {
                    for b in 0..batch {
                        for j in 0..hid {
                            // Recompute from all_pre_tanh at t-1
                            h_prev[b * hid + j] =
                                tanh_elem(all_pre_tanh[(t - 1) * batch * hid + b * hid + j]);
                        }
                    }
                }

                // Extract x_t
                let mut xt = vec![T::zero(); batch * inp];
                for b in 0..batch {
                    for j in 0..inp {
                        xt[b * inp + j] = xs_cap[b * (seq * inp) + t * inp + j];
                    }
                }

                // gwih += d_pre^T @ x_t → [hid, inp]
                for b in 0..batch {
                    for i in 0..hid {
                        for j in 0..inp {
                            gwih[i * inp + j] += d_pre[b * hid + i] * xt[b * inp + j];
                        }
                    }
                }

                // gwhh += d_pre^T @ h_prev → [hid, hid]
                for b in 0..batch {
                    for i in 0..hid {
                        for j in 0..hid {
                            gwhh[i * hid + j] += d_pre[b * hid + i] * h_prev[b * hid + j];
                        }
                    }
                }

                // bias gradients
                for b in 0..batch {
                    for j in 0..hid {
                        gbih[j] += d_pre[b * hid + j];
                        gbhh[j] += d_pre[b * hid + j];
                    }
                }

                // gx_t = d_pre @ W_ih → [batch, inp]
                for b in 0..batch {
                    for j in 0..inp {
                        let mut sum = T::zero();
                        for i in 0..hid {
                            sum += d_pre[b * hid + i] * wihs_cap[i * inp + j];
                        }
                        gx[b * (seq * inp) + t * inp + j] += sum;
                    }
                }

                // dh_next = d_pre @ W_hh → [batch, hid]
                dh_next = vec![T::zero(); batch * hid];
                for b in 0..batch {
                    for j in 0..hid {
                        let mut sum = T::zero();
                        for i in 0..hid {
                            sum += d_pre[b * hid + i] * whhs_cap[i * hid + j];
                        }
                        dh_next[b * hid + j] = sum;
                    }
                }
            }

            vec![
                Tensor::from_vec(gx, vec![batch, seq * inp]).expect("valid"),
                Tensor::from_vec(gwih, vec![hid, inp]).expect("valid"),
                Tensor::from_vec(gwhh, vec![hid, hid]).expect("valid"),
                Tensor::from_vec(gbih, vec![hid]).expect("valid"),
                Tensor::from_vec(gbhh, vec![hid]).expect("valid"),
            ]
        });

        Ok(Variable::from_op(
            out_tensor,
            vec![
                x.clone(),
                self.w_ih.clone(),
                self.w_hh.clone(),
                self.b_ih.clone(),
                self.b_hh.clone(),
            ],
            grad_fn,
        ))
    }

    fn parameters(&self) -> Vec<Variable<T>> {
        vec![
            self.w_ih.clone(),
            self.w_hh.clone(),
            self.b_ih.clone(),
            self.b_hh.clone(),
        ]
    }

    fn train(&mut self) {}
    fn eval(&mut self) {}
}

// ── LSTM ─────────────────────────────────────────────────────────────────

/// Long Short-Term Memory layer.
///
/// Gates: input (i), forget (f), cell (g), output (o).
///
/// Input: `[batch, seq_len * input_size]`
/// Output: `[batch, seq_len * hidden_size]`
pub struct LSTM<T: Float> {
    // Combined weight matrices for all 4 gates [4*hidden, input] and [4*hidden, hidden]
    w_ih: Variable<T>, // [4*hidden, input]
    w_hh: Variable<T>, // [4*hidden, hidden]
    b_ih: Variable<T>, // [4*hidden]
    b_hh: Variable<T>, // [4*hidden]
    input_size: usize,
    hidden_size: usize,
    seq_len: usize,
}

impl<T: Float> LSTM<T> {
    /// Create a new LSTM layer.
    pub fn new(input_size: usize, hidden_size: usize, seq_len: usize, rng: &mut Rng) -> Self {
        let gate_size = 4 * hidden_size;
        let w_ih = Variable::new(
            init::xavier_uniform::<T>(&[gate_size, input_size], rng),
            true,
        );
        let w_hh = Variable::new(
            init::xavier_uniform::<T>(&[gate_size, hidden_size], rng),
            true,
        );
        let b_ih = Variable::new(Tensor::zeros(vec![gate_size]), true);
        let b_hh = Variable::new(Tensor::zeros(vec![gate_size]), true);
        Self {
            w_ih,
            w_hh,
            b_ih,
            b_hh,
            input_size,
            hidden_size,
            seq_len,
        }
    }
}

impl<T: Float> Layer<T> for LSTM<T> {
    #[allow(clippy::too_many_lines)]
    fn forward(&self, x: &Variable<T>) -> Result<Variable<T>> {
        let shape = x.shape();
        let expected_cols = self.seq_len * self.input_size;
        if shape.len() != 2 || shape[1] != expected_cols {
            return Err(NnError::ShapeMismatch {
                expected: vec![0, expected_cols],
                got: shape,
            });
        }
        let batch = shape[0];
        let inp = self.input_size;
        let hid = self.hidden_size;
        let seq = self.seq_len;
        let gate4 = 4 * hid;

        let xd = x.data();
        let xs = xd.as_slice();
        let wih = self.w_ih.data();
        let wihs = wih.as_slice();
        let whh = self.w_hh.data();
        let whhs = whh.as_slice();
        let bih = self.b_ih.data();
        let bihs = bih.as_slice();
        let bhh = self.b_hh.data();
        let bhhs = bhh.as_slice();

        // Transpose weights
        let mut wiht = vec![T::zero(); inp * gate4];
        for i in 0..gate4 {
            for j in 0..inp {
                wiht[j * gate4 + i] = wihs[i * inp + j];
            }
        }
        let mut whht = vec![T::zero(); hid * gate4];
        for i in 0..gate4 {
            for j in 0..hid {
                whht[j * gate4 + i] = whhs[i * hid + j];
            }
        }

        let mut h = vec![T::zero(); batch * hid];
        let mut c = vec![T::zero(); batch * hid];
        let mut all_h = vec![T::zero(); batch * seq * hid];
        // Store gate values for backward
        let mut all_gates = vec![T::zero(); seq * batch * gate4];
        let mut all_c = vec![T::zero(); (seq + 1) * batch * hid];

        for t in 0..seq {
            let mut xt = vec![T::zero(); batch * inp];
            for b in 0..batch {
                for j in 0..inp {
                    xt[b * inp + j] = xs[b * (seq * inp) + t * inp + j];
                }
            }

            // gates = x_t @ W_ih^T + h @ W_hh^T + b_ih + b_hh
            let mut gates = vec![T::zero(); batch * gate4];
            matmul_acc(&mut gates, &xt, &wiht, batch, inp, gate4);
            matmul_acc(&mut gates, &h, &whht, batch, hid, gate4);
            for b in 0..batch {
                for j in 0..gate4 {
                    gates[b * gate4 + j] += bihs[j] + bhhs[j];
                }
            }

            // Apply activations: i=sigmoid, f=sigmoid, g=tanh, o=sigmoid
            for b in 0..batch {
                let base = b * gate4;
                for j in 0..hid {
                    gates[base + j] = sigmoid_elem(gates[base + j]); // i
                    gates[base + hid + j] = sigmoid_elem(gates[base + hid + j]); // f
                    gates[base + 2 * hid + j] = tanh_elem(gates[base + 2 * hid + j]); // g
                    gates[base + 3 * hid + j] = sigmoid_elem(gates[base + 3 * hid + j]); // o
                }
            }

            // Store gates
            all_gates[t * batch * gate4..(t + 1) * batch * gate4].copy_from_slice(&gates);
            // Store c_{t-1}
            all_c[t * batch * hid..(t + 1) * batch * hid].copy_from_slice(&c);

            // c = f * c + i * g
            // h = o * tanh(c)
            for b in 0..batch {
                let gb = b * gate4;
                for j in 0..hid {
                    let i_g = gates[gb + j];
                    let f_g = gates[gb + hid + j];
                    let g_g = gates[gb + 2 * hid + j];
                    let o_g = gates[gb + 3 * hid + j];
                    c[b * hid + j] = f_g * c[b * hid + j] + i_g * g_g;
                    h[b * hid + j] = o_g * tanh_elem(c[b * hid + j]);
                }
            }

            // Store final c and h
            all_c[(t + 1) * batch * hid..(t + 2) * batch * hid].copy_from_slice(&c);
            for b in 0..batch {
                for j in 0..hid {
                    all_h[b * seq * hid + t * hid + j] = h[b * hid + j];
                }
            }
        }

        let out_tensor = Tensor::from_vec(all_h, vec![batch, seq * hid]).expect("valid shape");

        let xs_cap = xs.to_vec();
        let wihs_cap = wihs.to_vec();
        let whhs_cap = whhs.to_vec();

        let grad_fn = Box::new(move |g: &Tensor<T>| {
            let gd = g.as_slice();
            let mut gx = vec![T::zero(); batch * seq * inp];
            let mut gwih = vec![T::zero(); gate4 * inp];
            let mut gwhh = vec![T::zero(); gate4 * hid];
            let mut gbih = vec![T::zero(); gate4];
            let mut gbhh = vec![T::zero(); gate4];

            let mut dh_next = vec![T::zero(); batch * hid];
            let mut dc_next = vec![T::zero(); batch * hid];

            for t in (0..seq).rev() {
                let mut dh = vec![T::zero(); batch * hid];
                for b in 0..batch {
                    for j in 0..hid {
                        dh[b * hid + j] = gd[b * seq * hid + t * hid + j] + dh_next[b * hid + j];
                    }
                }

                let gates = &all_gates[t * batch * gate4..(t + 1) * batch * gate4];
                let c_cur = &all_c[(t + 1) * batch * hid..(t + 2) * batch * hid];

                let mut d_gates = vec![T::zero(); batch * gate4];
                let mut dc = vec![T::zero(); batch * hid];

                for b in 0..batch {
                    let gb = b * gate4;
                    for j in 0..hid {
                        let i_g = gates[gb + j];
                        let f_g = gates[gb + hid + j];
                        let g_g = gates[gb + 2 * hid + j];
                        let o_g = gates[gb + 3 * hid + j];
                        let tc = tanh_elem(c_cur[b * hid + j]);

                        // do = dh * tanh(c), dc += dh * o * (1 - tanh^2(c))
                        let d_o = dh[b * hid + j] * tc;
                        dc[b * hid + j] =
                            dh[b * hid + j] * o_g * (T::one() - tc * tc) + dc_next[b * hid + j];

                        let c_prev = all_c[t * batch * hid + b * hid + j];
                        let d_f = dc[b * hid + j] * c_prev;
                        let d_i = dc[b * hid + j] * g_g;
                        let d_g = dc[b * hid + j] * i_g;

                        // Gate derivatives (sigmoid: s*(1-s), tanh: 1-t^2)
                        d_gates[gb + j] = d_i * i_g * (T::one() - i_g);
                        d_gates[gb + hid + j] = d_f * f_g * (T::one() - f_g);
                        d_gates[gb + 2 * hid + j] = d_g * (T::one() - g_g * g_g);
                        d_gates[gb + 3 * hid + j] = d_o * o_g * (T::one() - o_g);

                        dc_next[b * hid + j] = dc[b * hid + j] * f_g;
                    }
                }

                // Reconstruct h_{t-1} and x_t
                let mut h_prev = vec![T::zero(); batch * hid];
                if t > 0 {
                    for b in 0..batch {
                        for j in 0..hid {
                            h_prev[b * hid + j] =
                                all_gates[(t - 1) * batch * gate4 + b * gate4 + 3 * hid + j]; // o
                            let c_prev = all_c[t * batch * hid + b * hid + j];
                            h_prev[b * hid + j] *= tanh_elem(c_prev);
                        }
                    }
                    // Actually get h from output
                    for b in 0..batch {
                        for j in 0..hid {
                            let prev_gates = &all_gates[(t - 1) * batch * gate4..t * batch * gate4];
                            let o_prev = prev_gates[b * gate4 + 3 * hid + j];
                            let c_prev = all_c[t * batch * hid + b * hid + j];
                            h_prev[b * hid + j] = o_prev * tanh_elem(c_prev);
                        }
                    }
                }

                let mut xt = vec![T::zero(); batch * inp];
                for b in 0..batch {
                    for j in 0..inp {
                        xt[b * inp + j] = xs_cap[b * (seq * inp) + t * inp + j];
                    }
                }

                // Accumulate weight gradients
                for b in 0..batch {
                    for i in 0..gate4 {
                        for j in 0..inp {
                            gwih[i * inp + j] += d_gates[b * gate4 + i] * xt[b * inp + j];
                        }
                        for j in 0..hid {
                            gwhh[i * hid + j] += d_gates[b * gate4 + i] * h_prev[b * hid + j];
                        }
                        gbih[i] += d_gates[b * gate4 + i];
                        gbhh[i] += d_gates[b * gate4 + i];
                    }
                }

                // Input gradient
                for b in 0..batch {
                    for j in 0..inp {
                        let mut sum = T::zero();
                        for i in 0..gate4 {
                            sum += d_gates[b * gate4 + i] * wihs_cap[i * inp + j];
                        }
                        gx[b * (seq * inp) + t * inp + j] += sum;
                    }
                }

                // dh_next = d_gates @ W_hh
                dh_next = vec![T::zero(); batch * hid];
                for b in 0..batch {
                    for j in 0..hid {
                        let mut sum = T::zero();
                        for i in 0..gate4 {
                            sum += d_gates[b * gate4 + i] * whhs_cap[i * hid + j];
                        }
                        dh_next[b * hid + j] = sum;
                    }
                }
            }

            vec![
                Tensor::from_vec(gx, vec![batch, seq * inp]).expect("valid"),
                Tensor::from_vec(gwih, vec![gate4, inp]).expect("valid"),
                Tensor::from_vec(gwhh, vec![gate4, hid]).expect("valid"),
                Tensor::from_vec(gbih, vec![gate4]).expect("valid"),
                Tensor::from_vec(gbhh, vec![gate4]).expect("valid"),
            ]
        });

        Ok(Variable::from_op(
            out_tensor,
            vec![
                x.clone(),
                self.w_ih.clone(),
                self.w_hh.clone(),
                self.b_ih.clone(),
                self.b_hh.clone(),
            ],
            grad_fn,
        ))
    }

    fn parameters(&self) -> Vec<Variable<T>> {
        vec![
            self.w_ih.clone(),
            self.w_hh.clone(),
            self.b_ih.clone(),
            self.b_hh.clone(),
        ]
    }

    fn train(&mut self) {}
    fn eval(&mut self) {}
}

// ── GRU ──────────────────────────────────────────────────────────────────

/// Gated Recurrent Unit layer.
///
/// Gates: reset (r), update (z), new (n).
///
/// Input: `[batch, seq_len * input_size]`
/// Output: `[batch, seq_len * hidden_size]`
pub struct GRU<T: Float> {
    w_ih: Variable<T>, // [3*hidden, input]
    w_hh: Variable<T>, // [3*hidden, hidden]
    b_ih: Variable<T>, // [3*hidden]
    b_hh: Variable<T>, // [3*hidden]
    input_size: usize,
    hidden_size: usize,
    seq_len: usize,
}

impl<T: Float> GRU<T> {
    /// Create a new GRU layer.
    pub fn new(input_size: usize, hidden_size: usize, seq_len: usize, rng: &mut Rng) -> Self {
        let gate_size = 3 * hidden_size;
        let w_ih = Variable::new(
            init::xavier_uniform::<T>(&[gate_size, input_size], rng),
            true,
        );
        let w_hh = Variable::new(
            init::xavier_uniform::<T>(&[gate_size, hidden_size], rng),
            true,
        );
        let b_ih = Variable::new(Tensor::zeros(vec![gate_size]), true);
        let b_hh = Variable::new(Tensor::zeros(vec![gate_size]), true);
        Self {
            w_ih,
            w_hh,
            b_ih,
            b_hh,
            input_size,
            hidden_size,
            seq_len,
        }
    }
}

impl<T: Float> Layer<T> for GRU<T> {
    #[allow(clippy::too_many_lines)]
    fn forward(&self, x: &Variable<T>) -> Result<Variable<T>> {
        let shape = x.shape();
        let expected_cols = self.seq_len * self.input_size;
        if shape.len() != 2 || shape[1] != expected_cols {
            return Err(NnError::ShapeMismatch {
                expected: vec![0, expected_cols],
                got: shape,
            });
        }
        let batch = shape[0];
        let inp = self.input_size;
        let hid = self.hidden_size;
        let seq = self.seq_len;

        let xd = x.data();
        let xs = xd.as_slice();
        let wih = self.w_ih.data();
        let wihs = wih.as_slice();
        let whh = self.w_hh.data();
        let whhs = whh.as_slice();
        let bih = self.b_ih.data();
        let bihs = bih.as_slice();
        let bhh = self.b_hh.data();
        let bhhs = bhh.as_slice();

        let mut h = vec![T::zero(); batch * hid];
        let mut all_h = vec![T::zero(); batch * seq * hid];
        // Store intermediate values for backward
        let mut all_r = vec![T::zero(); seq * batch * hid];
        let mut all_z = vec![T::zero(); seq * batch * hid];
        let mut all_n = vec![T::zero(); seq * batch * hid];
        let mut all_h_prev = vec![T::zero(); seq * batch * hid];

        for t in 0..seq {
            let mut xt = vec![T::zero(); batch * inp];
            for b in 0..batch {
                for j in 0..inp {
                    xt[b * inp + j] = xs[b * (seq * inp) + t * inp + j];
                }
            }

            // Store h_prev
            all_h_prev[t * batch * hid..(t + 1) * batch * hid].copy_from_slice(&h);

            // Compute input projection: x_t @ W_ih^T + b_ih → [batch, 3*hid]
            let gate3 = 3 * hid;
            let mut x_proj = vec![T::zero(); batch * gate3];
            for b in 0..batch {
                for i in 0..gate3 {
                    let mut sum = bihs[i];
                    for k in 0..inp {
                        sum += xt[b * inp + k] * wihs[i * inp + k];
                    }
                    x_proj[b * gate3 + i] = sum;
                }
            }

            // Compute hidden projection: h @ W_hh^T + b_hh → [batch, 3*hid]
            let mut h_proj = vec![T::zero(); batch * gate3];
            for b in 0..batch {
                for i in 0..gate3 {
                    let mut sum = bhhs[i];
                    for k in 0..hid {
                        sum += h[b * hid + k] * whhs[i * hid + k];
                    }
                    h_proj[b * gate3 + i] = sum;
                }
            }

            // r = sigmoid(x_proj[0..hid] + h_proj[0..hid])
            // z = sigmoid(x_proj[hid..2*hid] + h_proj[hid..2*hid])
            // n = tanh(x_proj[2*hid..3*hid] + r * h_proj[2*hid..3*hid])
            for b in 0..batch {
                for j in 0..hid {
                    let r = sigmoid_elem(x_proj[b * gate3 + j] + h_proj[b * gate3 + j]);
                    let z = sigmoid_elem(x_proj[b * gate3 + hid + j] + h_proj[b * gate3 + hid + j]);
                    let n = tanh_elem(
                        x_proj[b * gate3 + 2 * hid + j] + r * h_proj[b * gate3 + 2 * hid + j],
                    );

                    all_r[t * batch * hid + b * hid + j] = r;
                    all_z[t * batch * hid + b * hid + j] = z;
                    all_n[t * batch * hid + b * hid + j] = n;

                    // h = (1 - z) * n + z * h_prev
                    h[b * hid + j] = (T::one() - z) * n + z * h[b * hid + j];
                }
            }

            for b in 0..batch {
                for j in 0..hid {
                    all_h[b * seq * hid + t * hid + j] = h[b * hid + j];
                }
            }
        }

        let out_tensor = Tensor::from_vec(all_h, vec![batch, seq * hid]).expect("valid shape");

        let xs_cap = xs.to_vec();
        let wihs_cap = wihs.to_vec();
        let whhs_cap = whhs.to_vec();
        let bhhs_cap = bhhs.to_vec();

        let grad_fn = Box::new(move |g: &Tensor<T>| {
            let gd = g.as_slice();
            let gate3 = 3 * hid;
            let mut gx = vec![T::zero(); batch * seq * inp];
            let mut gwih = vec![T::zero(); gate3 * inp];
            let mut gwhh = vec![T::zero(); gate3 * hid];
            let mut gbih = vec![T::zero(); gate3];
            let mut gbhh = vec![T::zero(); gate3];

            let mut dh_next = vec![T::zero(); batch * hid];

            for t in (0..seq).rev() {
                let mut dh = vec![T::zero(); batch * hid];
                for b in 0..batch {
                    for j in 0..hid {
                        dh[b * hid + j] = gd[b * seq * hid + t * hid + j] + dh_next[b * hid + j];
                    }
                }

                let r = &all_r[t * batch * hid..(t + 1) * batch * hid];
                let z = &all_z[t * batch * hid..(t + 1) * batch * hid];
                let n = &all_n[t * batch * hid..(t + 1) * batch * hid];
                let h_prev = &all_h_prev[t * batch * hid..(t + 1) * batch * hid];

                let mut xt = vec![T::zero(); batch * inp];
                for b in 0..batch {
                    for j in 0..inp {
                        xt[b * inp + j] = xs_cap[b * (seq * inp) + t * inp + j];
                    }
                }

                // Recompute h_proj_n for the n gate (the part multiplied by r)
                let mut h_proj_n = vec![T::zero(); batch * hid];
                for b in 0..batch {
                    for j in 0..hid {
                        let mut sum = bhhs_cap[2 * hid + j];
                        for k in 0..hid {
                            sum += h_prev[b * hid + k] * whhs_cap[(2 * hid + j) * hid + k];
                        }
                        h_proj_n[b * hid + j] = sum;
                    }
                }

                let mut d_gates_ih = vec![T::zero(); batch * gate3];
                let mut d_gates_hh = vec![T::zero(); batch * gate3];

                dh_next = vec![T::zero(); batch * hid];

                for b in 0..batch {
                    for j in 0..hid {
                        let rj = r[b * hid + j];
                        let zj = z[b * hid + j];
                        let nj = n[b * hid + j];
                        let hp = h_prev[b * hid + j];

                        // dh/dz = h_prev - n, dh/dn = 1 - z
                        let d_n = dh[b * hid + j] * (T::one() - zj) * (T::one() - nj * nj);
                        let d_z = dh[b * hid + j] * (hp - nj) * zj * (T::one() - zj);
                        let d_r = d_n * h_proj_n[b * hid + j] * rj * (T::one() - rj);

                        // For W_ih: d_r, d_z, d_n use same input projection
                        d_gates_ih[b * gate3 + j] = d_r;
                        d_gates_ih[b * gate3 + hid + j] = d_z;
                        d_gates_ih[b * gate3 + 2 * hid + j] = d_n;

                        // For W_hh: r/z use h_prev, n uses r*h_prev
                        d_gates_hh[b * gate3 + j] = d_r;
                        d_gates_hh[b * gate3 + hid + j] = d_z;
                        d_gates_hh[b * gate3 + 2 * hid + j] = d_n * rj;

                        // dh_next contributions
                        dh_next[b * hid + j] += dh[b * hid + j] * zj;
                    }
                }

                // Accumulate gradients
                for b in 0..batch {
                    for i in 0..gate3 {
                        for j in 0..inp {
                            gwih[i * inp + j] += d_gates_ih[b * gate3 + i] * xt[b * inp + j];
                        }
                        gbih[i] += d_gates_ih[b * gate3 + i];
                    }
                    for i in 0..gate3 {
                        for j in 0..hid {
                            gwhh[i * hid + j] += d_gates_hh[b * gate3 + i] * h_prev[b * hid + j];
                        }
                        gbhh[i] += d_gates_hh[b * gate3 + i];
                    }
                }

                // gx
                for b in 0..batch {
                    for j in 0..inp {
                        let mut sum = T::zero();
                        for i in 0..gate3 {
                            sum += d_gates_ih[b * gate3 + i] * wihs_cap[i * inp + j];
                        }
                        gx[b * (seq * inp) + t * inp + j] += sum;
                    }
                }

                // dh_next += d_gates_hh @ W_hh
                for b in 0..batch {
                    for j in 0..hid {
                        let mut sum = T::zero();
                        for i in 0..gate3 {
                            sum += d_gates_hh[b * gate3 + i] * whhs_cap[i * hid + j];
                        }
                        dh_next[b * hid + j] += sum;
                    }
                }
            }

            vec![
                Tensor::from_vec(gx, vec![batch, seq * inp]).expect("valid"),
                Tensor::from_vec(gwih, vec![gate3, inp]).expect("valid"),
                Tensor::from_vec(gwhh, vec![gate3, hid]).expect("valid"),
                Tensor::from_vec(gbih, vec![gate3]).expect("valid"),
                Tensor::from_vec(gbhh, vec![gate3]).expect("valid"),
            ]
        });

        Ok(Variable::from_op(
            out_tensor,
            vec![
                x.clone(),
                self.w_ih.clone(),
                self.w_hh.clone(),
                self.b_ih.clone(),
                self.b_hh.clone(),
            ],
            grad_fn,
        ))
    }

    fn parameters(&self) -> Vec<Variable<T>> {
        vec![
            self.w_ih.clone(),
            self.w_hh.clone(),
            self.b_ih.clone(),
            self.b_hh.clone(),
        ]
    }

    fn train(&mut self) {}
    fn eval(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rnn_output_shape() {
        let mut rng = Rng::new(42);
        let rnn = SimpleRNN::<f64>::new(4, 8, 3, &mut rng);
        // batch=2, seq=3, inp=4 → input [2, 12]
        let x = Variable::new(Tensor::ones(vec![2, 12]), true);
        let y = rnn.forward(&x).unwrap();
        // output [2, 24] = batch * seq * hidden
        assert_eq!(y.shape(), vec![2, 24]);
    }

    #[test]
    fn test_rnn_backward() {
        let mut rng = Rng::new(42);
        let rnn = SimpleRNN::<f64>::new(3, 4, 2, &mut rng);
        let x = Variable::new(Tensor::ones(vec![1, 6]), true);
        let y = rnn.forward(&x).unwrap();
        let loss = crate::ops::sum(&y);
        loss.backward();
        let gx = x.grad().unwrap();
        assert_eq!(gx.shape(), &[1, 6]);
    }

    #[test]
    fn test_rnn_parameters() {
        let mut rng = Rng::new(42);
        let rnn = SimpleRNN::<f64>::new(4, 8, 3, &mut rng);
        assert_eq!(rnn.parameters().len(), 4);
    }

    #[test]
    fn test_lstm_output_shape() {
        let mut rng = Rng::new(42);
        let lstm = LSTM::<f64>::new(4, 8, 3, &mut rng);
        let x = Variable::new(Tensor::ones(vec![2, 12]), true);
        let y = lstm.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![2, 24]);
    }

    #[test]
    fn test_lstm_backward() {
        let mut rng = Rng::new(42);
        let lstm = LSTM::<f64>::new(3, 4, 2, &mut rng);
        let x = Variable::new(Tensor::ones(vec![1, 6]), true);
        let y = lstm.forward(&x).unwrap();
        let loss = crate::ops::sum(&y);
        loss.backward();
        let gx = x.grad().unwrap();
        assert_eq!(gx.shape(), &[1, 6]);
    }

    #[test]
    fn test_lstm_parameters() {
        let mut rng = Rng::new(42);
        let lstm = LSTM::<f64>::new(4, 8, 3, &mut rng);
        assert_eq!(lstm.parameters().len(), 4);
    }

    #[test]
    fn test_gru_output_shape() {
        let mut rng = Rng::new(42);
        let gru = GRU::<f64>::new(4, 8, 3, &mut rng);
        let x = Variable::new(Tensor::ones(vec![2, 12]), true);
        let y = gru.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![2, 24]);
    }

    #[test]
    fn test_gru_backward() {
        let mut rng = Rng::new(42);
        let gru = GRU::<f64>::new(3, 4, 2, &mut rng);
        let x = Variable::new(Tensor::ones(vec![1, 6]), true);
        let y = gru.forward(&x).unwrap();
        let loss = crate::ops::sum(&y);
        loss.backward();
        let gx = x.grad().unwrap();
        assert_eq!(gx.shape(), &[1, 6]);
    }

    #[test]
    fn test_gru_parameters() {
        let mut rng = Rng::new(42);
        let gru = GRU::<f64>::new(4, 8, 3, &mut rng);
        assert_eq!(gru.parameters().len(), 4);
    }

    #[test]
    fn test_rnn_wrong_shape() {
        let mut rng = Rng::new(42);
        let rnn = SimpleRNN::<f64>::new(4, 8, 3, &mut rng);
        let x = Variable::new(Tensor::ones(vec![2, 10]), true); // wrong: should be 12
        assert!(rnn.forward(&x).is_err());
    }
}
