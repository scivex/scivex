//! Graph Neural Network layers.
//!
//! These layers operate on graph-structured data, taking both node features
//! and an adjacency matrix as input. Because the standard [`Layer`](super::Layer)
//! trait accepts only a single `Variable`, GNN layers define their own
//! `forward` methods with signature `(x: &Variable<T>, adj: &Tensor<T>)`.

use scivex_core::random::Rng;
use scivex_core::{Float, Tensor};

use crate::error::{NnError, Result};
use crate::init;
use crate::ops;
use crate::variable::Variable;

// ── GCNConv ──────────────────────────────────────────────────────────────

/// Graph Convolutional Network layer (Kipf & Welling, 2017).
///
/// Computes `out = D^{-1/2} A_hat D^{-1/2} X W + b` where `A_hat = A + I`.
///
/// # Examples
///
/// ```
/// # use scivex_core::{Tensor, random::Rng};
/// # use scivex_nn::variable::Variable;
/// # use scivex_nn::layer::gnn::GCNConv;
/// let mut rng = Rng::new(42);
/// let gcn = GCNConv::<f64>::new(4, 2, true, &mut rng);
/// let x = Variable::new(Tensor::ones(vec![3, 4]), true);
/// let adj = Tensor::eye(3);
/// let y = gcn.forward(&x, &adj).unwrap();
/// assert_eq!(y.shape(), vec![3, 2]);
/// ```
pub struct GCNConv<T: Float> {
    weight: Variable<T>,
    bias: Option<Variable<T>>,
}

impl<T: Float> GCNConv<T> {
    /// Create a new GCN convolution layer.
    ///
    /// - `in_features`: number of input features per node
    /// - `out_features`: number of output features per node
    /// - `use_bias`: whether to include a bias term
    /// - `rng`: random number generator for weight initialization
    pub fn new(in_features: usize, out_features: usize, use_bias: bool, rng: &mut Rng) -> Self {
        let w_data = init::kaiming_uniform::<T>(&[in_features, out_features], rng);
        let weight = Variable::new(w_data, true);

        let bias = if use_bias {
            Some(Variable::new(Tensor::zeros(vec![out_features]), true))
        } else {
            None
        };

        Self { weight, bias }
    }

    /// Forward pass: `out = adj_norm @ x @ weight + bias`.
    ///
    /// - `x`: node features `[n_nodes, in_features]`
    /// - `adj`: adjacency matrix `[n_nodes, n_nodes]`
    pub fn forward(&self, x: &Variable<T>, adj: &Tensor<T>) -> Result<Variable<T>> {
        let x_shape = x.shape();
        if x_shape.len() != 2 {
            return Err(NnError::ShapeMismatch {
                expected: vec![0, 0],
                got: x_shape,
            });
        }
        let n = x_shape[0];
        let adj_shape = adj.shape();
        if adj_shape != [n, n] {
            return Err(NnError::ShapeMismatch {
                expected: vec![n, n],
                got: adj_shape.to_vec(),
            });
        }

        // Compute normalized adjacency: A_hat = A + I, D^{-1/2} A_hat D^{-1/2}
        let adj_norm = normalize_adjacency(adj)?;
        let adj_var = Variable::new(adj_norm, false);

        // adj_norm @ x
        let ax = ops::matmul(&adj_var, x);
        // (adj_norm @ x) @ weight
        let y = ops::matmul(&ax, &self.weight);

        match &self.bias {
            Some(b) => Ok(ops::add_bias(&y, b)),
            None => Ok(y),
        }
    }

    /// Return all learnable parameters.
    pub fn parameters(&self) -> Vec<Variable<T>> {
        let mut params = vec![self.weight.clone()];
        if let Some(b) = &self.bias {
            params.push(b.clone());
        }
        params
    }
}

/// Compute the symmetric normalized adjacency: `D^{-1/2} (A + I) D^{-1/2}`.
fn normalize_adjacency<T: Float>(adj: &Tensor<T>) -> Result<Tensor<T>> {
    let n = adj.shape()[0];
    let adj_slice = adj.as_slice();

    // A_hat = A + I (add self-loops)
    let mut a_hat = adj_slice.to_vec();
    for i in 0..n {
        a_hat[i * n + i] += T::one();
    }

    // Compute degree vector: D_ii = sum_j A_hat_ij
    let mut deg = vec![T::zero(); n];
    for i in 0..n {
        for j in 0..n {
            deg[i] += a_hat[i * n + j];
        }
    }

    // D^{-1/2}
    let deg_inv_sqrt: Vec<T> = deg
        .iter()
        .map(|&d| {
            if d > T::zero() {
                T::one() / d.sqrt()
            } else {
                T::zero()
            }
        })
        .collect();

    // D^{-1/2} A_hat D^{-1/2}
    let mut normed = vec![T::zero(); n * n];
    for i in 0..n {
        for j in 0..n {
            normed[i * n + j] = deg_inv_sqrt[i] * a_hat[i * n + j] * deg_inv_sqrt[j];
        }
    }

    Tensor::from_vec(normed, vec![n, n]).map_err(NnError::CoreError)
}

// ── GATConv ──────────────────────────────────────────────────────────────

/// Graph Attention Network layer (Velickovic et al., 2018).
///
/// Single-head attention: transforms features, computes attention coefficients
/// using LeakyReLU, applies masked softmax over neighbors, and aggregates.
///
/// # Examples
///
/// ```
/// # use scivex_core::{Tensor, random::Rng};
/// # use scivex_nn::variable::Variable;
/// # use scivex_nn::layer::gnn::GATConv;
/// let mut rng = Rng::new(42);
/// let gat = GATConv::<f64>::new(4, 2, &mut rng);
/// let x = Variable::new(Tensor::ones(vec![3, 4]), true);
/// let adj = Tensor::eye(3);
/// let y = gat.forward(&x, &adj).unwrap();
/// assert_eq!(y.shape(), vec![3, 2]);
/// ```
pub struct GATConv<T: Float> {
    weight: Variable<T>,
    attn_left: Variable<T>,
    attn_right: Variable<T>,
    #[allow(dead_code)]
    num_heads: usize,
    out_features: usize,
}

impl<T: Float> GATConv<T> {
    /// Create a new GAT convolution layer (single-head).
    ///
    /// - `in_features`: number of input features per node
    /// - `out_features`: number of output features per node
    /// - `rng`: random number generator for weight initialization
    pub fn new(in_features: usize, out_features: usize, rng: &mut Rng) -> Self {
        let w_data = init::kaiming_uniform::<T>(&[in_features, out_features], rng);
        let weight = Variable::new(w_data, true);

        let attn_l_data = init::kaiming_uniform::<T>(&[out_features, 1], rng);
        let attn_left = Variable::new(
            Tensor::from_vec(attn_l_data.as_slice().to_vec(), vec![out_features])
                .expect("reshape attn_left"),
            true,
        );

        let attn_r_data = init::kaiming_uniform::<T>(&[out_features, 1], rng);
        let attn_right = Variable::new(
            Tensor::from_vec(attn_r_data.as_slice().to_vec(), vec![out_features])
                .expect("reshape attn_right"),
            true,
        );

        Self {
            weight,
            attn_left,
            attn_right,
            num_heads: 1,
            out_features,
        }
    }

    /// Forward pass with graph attention.
    ///
    /// - `x`: node features `[n_nodes, in_features]`
    /// - `adj`: adjacency matrix `[n_nodes, n_nodes]` (1 where edge exists)
    pub fn forward(&self, x: &Variable<T>, adj: &Tensor<T>) -> Result<Variable<T>> {
        let x_shape = x.shape();
        if x_shape.len() != 2 {
            return Err(NnError::ShapeMismatch {
                expected: vec![0, 0],
                got: x_shape,
            });
        }
        let n = x_shape[0];
        let adj_shape = adj.shape();
        if adj_shape != [n, n] {
            return Err(NnError::ShapeMismatch {
                expected: vec![n, n],
                got: adj_shape.to_vec(),
            });
        }

        // h = X @ W  -> [n, out_features]
        let h = ops::matmul(x, &self.weight);
        let h_data = h.data();
        let h_slice = h_data.as_slice();
        let out_f = self.out_features;

        // Compute attention scores using the learned attention vectors.
        // e_i = h_i . attn_left (dot product per node)
        // e_j = h_j . attn_right
        // e_ij = LeakyReLU(e_i + e_j)
        let al = self.attn_left.data();
        let ar = self.attn_right.data();
        let al_slice = al.as_slice();
        let ar_slice = ar.as_slice();

        // Compute left scores: [n] and right scores: [n]
        let mut left_scores = vec![T::zero(); n];
        let mut right_scores = vec![T::zero(); n];
        for i in 0..n {
            let mut sl = T::zero();
            let mut sr = T::zero();
            for f in 0..out_f {
                sl += h_slice[i * out_f + f] * al_slice[f];
                sr += h_slice[i * out_f + f] * ar_slice[f];
            }
            left_scores[i] = sl;
            right_scores[i] = sr;
        }

        // e_ij = LeakyReLU(left_scores[i] + right_scores[j])
        // Masked by adjacency (including self-loops: A + I)
        let adj_slice = adj.as_slice();
        let neg_slope = T::from_f64(0.2);
        let neg_inf = T::from_f64(-1e9);

        let mut attn_scores = vec![T::zero(); n * n];
        for i in 0..n {
            for j in 0..n {
                let connected = adj_slice[i * n + j] > T::zero() || i == j; // self-loops
                if connected {
                    let e = left_scores[i] + right_scores[j];
                    // LeakyReLU
                    attn_scores[i * n + j] = if e > T::zero() { e } else { neg_slope * e };
                } else {
                    attn_scores[i * n + j] = neg_inf;
                }
            }
        }

        // Softmax over neighbors (row-wise)
        for i in 0..n {
            let row = &mut attn_scores[i * n..(i + 1) * n];
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

        // Aggregate: out_i = sum_j alpha_ij * h_j
        // Build alpha as a Variable for gradient flow through h
        let alpha_tensor = Tensor::from_vec(attn_scores, vec![n, n]).map_err(NnError::CoreError)?;
        let alpha_var = Variable::new(alpha_tensor, false);

        // out = alpha @ h  -> [n, out_features]
        let out = ops::matmul(&alpha_var, &h);
        Ok(out)
    }

    /// Return all learnable parameters.
    pub fn parameters(&self) -> Vec<Variable<T>> {
        vec![
            self.weight.clone(),
            self.attn_left.clone(),
            self.attn_right.clone(),
        ]
    }
}

// ── SAGEConv ─────────────────────────────────────────────────────────────

/// GraphSAGE mean-aggregator layer (Hamilton et al., 2017).
///
/// Aggregates neighbor features via mean pooling, concatenates with self
/// features, and applies a linear transformation.
///
/// # Examples
///
/// ```
/// # use scivex_core::{Tensor, random::Rng};
/// # use scivex_nn::variable::Variable;
/// # use scivex_nn::layer::gnn::SAGEConv;
/// let mut rng = Rng::new(42);
/// let sage = SAGEConv::<f64>::new(4, 2, true, &mut rng);
/// let x = Variable::new(Tensor::ones(vec![3, 4]), true);
/// let adj = Tensor::eye(3);
/// let y = sage.forward(&x, &adj).unwrap();
/// assert_eq!(y.shape(), vec![3, 2]);
/// ```
pub struct SAGEConv<T: Float> {
    weight: Variable<T>,
    bias: Option<Variable<T>>,
    in_features: usize,
}

impl<T: Float> SAGEConv<T> {
    /// Create a new GraphSAGE convolution layer.
    ///
    /// - `in_features`: number of input features per node
    /// - `out_features`: number of output features per node
    /// - `use_bias`: whether to include a bias term
    /// - `rng`: random number generator for weight initialization
    pub fn new(in_features: usize, out_features: usize, use_bias: bool, rng: &mut Rng) -> Self {
        // Weight transforms the concatenation [self_features, neighbor_mean]
        let w_data = init::kaiming_uniform::<T>(&[2 * in_features, out_features], rng);
        let weight = Variable::new(w_data, true);

        let bias = if use_bias {
            Some(Variable::new(Tensor::zeros(vec![out_features]), true))
        } else {
            None
        };

        Self {
            weight,
            bias,
            in_features,
        }
    }

    /// Forward pass with mean aggregation.
    ///
    /// - `x`: node features `[n_nodes, in_features]`
    /// - `adj`: adjacency matrix `[n_nodes, n_nodes]`
    pub fn forward(&self, x: &Variable<T>, adj: &Tensor<T>) -> Result<Variable<T>> {
        let x_shape = x.shape();
        if x_shape.len() != 2 {
            return Err(NnError::ShapeMismatch {
                expected: vec![0, 0],
                got: x_shape,
            });
        }
        let n = x_shape[0];
        let in_f = x_shape[1];
        let adj_shape = adj.shape();
        if adj_shape != [n, n] {
            return Err(NnError::ShapeMismatch {
                expected: vec![n, n],
                got: adj_shape.to_vec(),
            });
        }

        // Row-normalize adjacency for mean aggregation
        let adj_norm = row_normalize(adj)?;
        let adj_var = Variable::new(adj_norm, false);

        // neigh = adj_norm @ x  -> [n, in_features]
        let neigh = ops::matmul(&adj_var, x);

        // Concatenate [x, neigh] along feature axis -> [n, 2*in_features]
        let x_data = x.data();
        let neigh_data = neigh.data();
        let x_slice = x_data.as_slice();
        let neigh_slice = neigh_data.as_slice();

        let mut concat_data = Vec::with_capacity(n * 2 * in_f);
        for i in 0..n {
            for f in 0..in_f {
                concat_data.push(x_slice[i * in_f + f]);
            }
            for f in 0..in_f {
                concat_data.push(neigh_slice[i * in_f + f]);
            }
        }
        let concat_tensor =
            Tensor::from_vec(concat_data, vec![n, 2 * in_f]).map_err(NnError::CoreError)?;

        // Build a Variable that connects gradients back to x and neigh
        let in_features = self.in_features;
        let concat_var = Variable::from_op(
            concat_tensor,
            vec![x.clone(), neigh],
            Box::new(move |g: &Tensor<T>| {
                // g: [n, 2*in_features]
                // Split gradient back to x part and neigh part
                let g_slice = g.as_slice();
                let rows = g.shape()[0];
                let mut gx = Vec::with_capacity(rows * in_features);
                let mut gn = Vec::with_capacity(rows * in_features);
                for i in 0..rows {
                    for f in 0..in_features {
                        gx.push(g_slice[i * 2 * in_features + f]);
                    }
                    for f in 0..in_features {
                        gn.push(g_slice[i * 2 * in_features + in_features + f]);
                    }
                }
                let grad_x =
                    Tensor::from_vec(gx, vec![rows, in_features]).expect("grad shape matches");
                let grad_neigh =
                    Tensor::from_vec(gn, vec![rows, in_features]).expect("grad shape matches");
                vec![grad_x, grad_neigh]
            }),
        );

        // out = concat @ weight + bias
        let y = ops::matmul(&concat_var, &self.weight);

        match &self.bias {
            Some(b) => Ok(ops::add_bias(&y, b)),
            None => Ok(y),
        }
    }

    /// Return all learnable parameters.
    pub fn parameters(&self) -> Vec<Variable<T>> {
        let mut params = vec![self.weight.clone()];
        if let Some(b) = &self.bias {
            params.push(b.clone());
        }
        params
    }
}

/// Row-normalize an adjacency matrix: each row sums to 1 (mean aggregation).
fn row_normalize<T: Float>(adj: &Tensor<T>) -> Result<Tensor<T>> {
    let n = adj.shape()[0];
    let adj_slice = adj.as_slice();
    let mut normed = adj_slice.to_vec();

    for i in 0..n {
        let mut row_sum = T::zero();
        for j in 0..n {
            row_sum += normed[i * n + j];
        }
        if row_sum > T::zero() {
            for j in 0..n {
                normed[i * n + j] /= row_sum;
            }
        }
    }

    Tensor::from_vec(normed, vec![n, n]).map_err(NnError::CoreError)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scivex_core::Tensor;

    /// Helper: create a simple 3-node graph adjacency.
    fn simple_adj() -> Tensor<f64> {
        // Triangle graph: 0-1, 1-2, 0-2 (undirected)
        Tensor::from_vec(
            vec![
                0.0, 1.0, 1.0, //
                1.0, 0.0, 1.0, //
                1.0, 1.0, 0.0, //
            ],
            vec![3, 3],
        )
        .unwrap()
    }

    #[test]
    fn test_gcn_forward_shape() {
        let mut rng = Rng::new(42);
        let gcn = GCNConv::<f64>::new(4, 2, true, &mut rng);
        let x = Variable::new(Tensor::ones(vec![3, 4]), true);
        let adj = simple_adj();
        let y = gcn.forward(&x, &adj).unwrap();
        assert_eq!(y.shape(), vec![3, 2]);
    }

    #[test]
    fn test_gcn_parameters() {
        let mut rng = Rng::new(42);
        let gcn_bias = GCNConv::<f64>::new(4, 2, true, &mut rng);
        assert_eq!(gcn_bias.parameters().len(), 2); // weight + bias

        let gcn_no_bias = GCNConv::<f64>::new(4, 2, false, &mut rng);
        assert_eq!(gcn_no_bias.parameters().len(), 1); // weight only
    }

    #[test]
    fn test_gat_forward_shape() {
        let mut rng = Rng::new(42);
        let gat = GATConv::<f64>::new(4, 2, &mut rng);
        let x = Variable::new(Tensor::ones(vec![3, 4]), true);
        let adj = simple_adj();
        let y = gat.forward(&x, &adj).unwrap();
        assert_eq!(y.shape(), vec![3, 2]);
    }

    #[test]
    fn test_sage_forward_shape() {
        let mut rng = Rng::new(42);
        let sage = SAGEConv::<f64>::new(4, 2, true, &mut rng);
        let x = Variable::new(Tensor::ones(vec![3, 4]), true);
        let adj = simple_adj();
        let y = sage.forward(&x, &adj).unwrap();
        assert_eq!(y.shape(), vec![3, 2]);
    }

    #[test]
    fn test_gcn_self_loops() {
        // With identity adjacency (only self-loops after normalization),
        // output should be x @ weight + bias, producing meaningful values.
        let mut rng = Rng::new(42);
        let gcn = GCNConv::<f64>::new(4, 2, true, &mut rng);
        let x = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 4]).unwrap(),
            true,
        );
        // Identity adjacency: each node is its own neighbor only (after +I => 2*I, normalized => I)
        let adj = Tensor::eye(2);
        let y = gcn.forward(&x, &adj).unwrap();
        assert_eq!(y.shape(), vec![2, 2]);

        // Output should not be all zeros (weights are initialized randomly)
        let y_slice = y.data();
        let y_data = y_slice.as_slice();
        let all_zero = y_data.iter().all(|&v| v.abs() < 1e-15);
        assert!(
            !all_zero,
            "GCN output should not be all zeros with identity adjacency"
        );
    }
}
