//! Embedding layer — lookup table for integer indices.

use scivex_core::random::Rng;
use scivex_core::{Float, Tensor};

use crate::error::{NnError, Result};
use crate::init;
use crate::variable::Variable;

use super::Layer;

/// Embedding lookup table.
///
/// Maps integer indices to dense vectors. Input values are rounded to the
/// nearest integer and used as row indices into a weight matrix.
///
/// Input: `[batch, seq_len]` (values treated as integer indices)
/// Output: `[batch, seq_len * dim]` (flattened for compatibility)
///
/// The weight matrix has shape `[num_embeddings, dim]`.
pub struct Embedding<T: Float> {
    weight: Variable<T>, // [num_embeddings, dim]
    num_embeddings: usize,
    dim: usize,
}

impl<T: Float> Embedding<T> {
    /// Create a new Embedding layer.
    pub fn new(num_embeddings: usize, dim: usize, rng: &mut Rng) -> Self {
        let w_data = init::xavier_uniform::<T>(&[num_embeddings, dim], rng);
        let weight = Variable::new(w_data, true);
        Self {
            weight,
            num_embeddings,
            dim,
        }
    }

    /// Access the weight matrix.
    pub fn weight(&self) -> &Variable<T> {
        &self.weight
    }

    /// Number of embeddings (vocabulary size).
    pub fn num_embeddings(&self) -> usize {
        self.num_embeddings
    }

    /// Embedding vector dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }
}

impl<T: Float> Layer<T> for Embedding<T> {
    fn forward(&self, x: &Variable<T>) -> Result<Variable<T>> {
        let shape = x.shape();
        if shape.len() != 2 {
            return Err(NnError::ShapeMismatch {
                expected: vec![0, 0],
                got: shape,
            });
        }
        let batch = shape[0];
        let seq_len = shape[1];
        let dim = self.dim;
        let num_emb = self.num_embeddings;

        let xd = x.data();
        let xs = xd.as_slice();
        let wd = self.weight.data();
        let ws = wd.as_slice();

        // Convert float indices to integer indices
        let mut indices = Vec::with_capacity(batch * seq_len);
        for &v in xs {
            let idx = v.round();
            // Convert to usize via f64
            let idx_f64 = format!("{idx:.0}");
            let idx_usize: usize = idx_f64.parse().map_err(|_| NnError::IndexOutOfBounds {
                index: 0,
                len: num_emb,
            })?;
            if idx_usize >= num_emb {
                return Err(NnError::IndexOutOfBounds {
                    index: idx_usize,
                    len: num_emb,
                });
            }
            indices.push(idx_usize);
        }

        // Gather embeddings
        let mut out = vec![T::zero(); batch * seq_len * dim];
        for (i, &idx) in indices.iter().enumerate() {
            let src = &ws[idx * dim..(idx + 1) * dim];
            out[i * dim..(i + 1) * dim].copy_from_slice(src);
        }

        let out_tensor = Tensor::from_vec(out, vec![batch, seq_len * dim]).expect("valid shape");

        let grad_fn = Box::new(move |g: &Tensor<T>| {
            let gd = g.as_slice();
            // Scatter gradients back to weight matrix
            let mut gw = vec![T::zero(); num_emb * dim];
            for (i, &idx) in indices.iter().enumerate() {
                for j in 0..dim {
                    gw[idx * dim + j] += gd[i * dim + j];
                }
            }

            // No gradient for input (indices are discrete)
            vec![
                Tensor::zeros(vec![batch, seq_len]),
                Tensor::from_vec(gw, vec![num_emb, dim]).expect("valid shape"),
            ]
        });

        Ok(Variable::from_op(
            out_tensor,
            vec![x.clone(), self.weight.clone()],
            grad_fn,
        ))
    }

    fn parameters(&self) -> Vec<Variable<T>> {
        vec![self.weight.clone()]
    }

    fn train(&mut self) {}
    fn eval(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_output_shape() {
        let mut rng = Rng::new(42);
        let emb = Embedding::<f64>::new(10, 8, &mut rng);
        // batch=2, seq_len=3
        let x = Variable::new(
            Tensor::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], vec![2, 3]).unwrap(),
            false,
        );
        let y = emb.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![2, 24]); // 3 * 8 = 24
    }

    #[test]
    fn test_embedding_values() {
        let mut rng = Rng::new(42);
        let emb = Embedding::<f64>::new(5, 3, &mut rng);
        let x = Variable::new(Tensor::from_vec(vec![0.0, 2.0], vec![1, 2]).unwrap(), false);
        let y = emb.forward(&x).unwrap();
        let yd = y.data();
        let ys = yd.as_slice();
        let wd = emb.weight().data();
        let ws = wd.as_slice();
        // First 3 values should match row 0 of weight
        assert_eq!(&ys[0..3], &ws[0..3]);
        // Next 3 values should match row 2
        assert_eq!(&ys[3..6], &ws[6..9]);
    }

    #[test]
    fn test_embedding_out_of_bounds() {
        let mut rng = Rng::new(42);
        let emb = Embedding::<f64>::new(5, 3, &mut rng);
        let x = Variable::new(Tensor::from_vec(vec![10.0], vec![1, 1]).unwrap(), false);
        assert!(emb.forward(&x).is_err());
    }

    #[test]
    fn test_embedding_backward() {
        let mut rng = Rng::new(42);
        let emb = Embedding::<f64>::new(5, 4, &mut rng);
        let x = Variable::new(Tensor::from_vec(vec![1.0, 3.0], vec![1, 2]).unwrap(), false);
        let y = emb.forward(&x).unwrap();
        let loss = crate::ops::sum(&y);
        loss.backward();
        let gw = emb.weight().grad().unwrap();
        assert_eq!(gw.shape(), &[5, 4]);
        // Row 1 and row 3 should have gradient 1.0, others 0.0
        let gs = gw.as_slice();
        for j in 0..4 {
            assert!((gs[4 + j] - 1.0).abs() < f64::EPSILON);
            assert!((gs[3 * 4 + j] - 1.0).abs() < f64::EPSILON);
            assert!(gs[j].abs() < f64::EPSILON);
        }
    }

    #[test]
    fn test_embedding_parameters() {
        let mut rng = Rng::new(42);
        let emb = Embedding::<f64>::new(10, 8, &mut rng);
        assert_eq!(emb.parameters().len(), 1);
    }
}
