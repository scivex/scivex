//! Dataset and data loading utilities.

use scivex_core::random::Rng;
use scivex_core::{Float, Tensor};

use crate::error::{NnError, Result};

/// A dataset of input-target pairs.
pub trait Dataset<T: Float> {
    /// Number of samples.
    fn len(&self) -> usize;

    /// Whether the dataset is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the i-th sample as `(input, target)`.
    fn get(&self, index: usize) -> Result<(Tensor<T>, Tensor<T>)>;
}

/// A dataset wrapping two tensors (inputs and targets).
///
/// The first dimension is the sample dimension.
pub struct TensorDataset<T: Float> {
    x: Tensor<T>,
    y: Tensor<T>,
    len: usize,
}

impl<T: Float> TensorDataset<T> {
    /// Create a new tensor dataset.
    ///
    /// `x` and `y` must have the same first dimension (number of samples).
    pub fn new(x: Tensor<T>, y: Tensor<T>) -> Result<Self> {
        let n = x.shape()[0];
        if y.shape()[0] != n {
            return Err(NnError::ShapeMismatch {
                expected: vec![n],
                got: vec![y.shape()[0]],
            });
        }
        Ok(Self { x, y, len: n })
    }
}

impl<T: Float> Dataset<T> for TensorDataset<T> {
    fn len(&self) -> usize {
        self.len
    }

    fn get(&self, index: usize) -> Result<(Tensor<T>, Tensor<T>)> {
        if index >= self.len {
            return Err(NnError::IndexOutOfBounds {
                index,
                len: self.len,
            });
        }
        let xi = self.x.select(0, index)?;
        let yi = self.y.select(0, index)?;
        Ok((xi, yi))
    }
}

/// An iterator over batches from a dataset.
pub struct DataLoader<'a, T: Float, D: Dataset<T>> {
    dataset: &'a D,
    batch_size: usize,
    shuffle: bool,
    indices: Vec<usize>,
    pos: usize,
    _marker: std::marker::PhantomData<T>,
}

impl<'a, T: Float, D: Dataset<T>> DataLoader<'a, T, D> {
    /// Create a new data loader.
    ///
    /// - `dataset`: the dataset to iterate over
    /// - `batch_size`: number of samples per batch
    /// - `shuffle`: whether to shuffle indices before each epoch
    /// - `rng`: optional RNG for shuffling
    pub fn new(dataset: &'a D, batch_size: usize, shuffle: bool, rng: Option<&mut Rng>) -> Self {
        let n = dataset.len();
        let mut indices: Vec<usize> = (0..n).collect();
        if shuffle && let Some(rng) = rng {
            fisher_yates_shuffle(&mut indices, rng);
        }
        Self {
            dataset,
            batch_size,
            shuffle,
            indices,
            pos: 0,
            _marker: std::marker::PhantomData,
        }
    }

    /// Re-shuffle and reset the iterator for a new epoch.
    pub fn reset(&mut self, rng: Option<&mut Rng>) {
        if self.shuffle
            && let Some(rng) = rng
        {
            fisher_yates_shuffle(&mut self.indices, rng);
        }
        self.pos = 0;
    }

    /// Number of batches in one full pass.
    pub fn num_batches(&self) -> usize {
        let n = self.dataset.len();
        n.div_ceil(self.batch_size)
    }
}

impl<T: Float, D: Dataset<T>> Iterator for DataLoader<'_, T, D> {
    type Item = Result<(Tensor<T>, Tensor<T>)>;

    fn next(&mut self) -> Option<Self::Item> {
        let n = self.dataset.len();
        if self.pos >= n {
            return None;
        }

        let end = (self.pos + self.batch_size).min(n);
        let batch_indices = &self.indices[self.pos..end];
        self.pos = end;

        Some(self.collect_batch(batch_indices))
    }
}

impl<T: Float, D: Dataset<T>> DataLoader<'_, T, D> {
    /// Collect a batch from the given indices.
    fn collect_batch(&self, batch_indices: &[usize]) -> Result<(Tensor<T>, Tensor<T>)> {
        let mut x_vecs = Vec::new();
        let mut y_vecs = Vec::new();
        let mut x_sample_shape = Vec::new();
        let mut y_sample_shape = Vec::new();

        for (i, &idx) in batch_indices.iter().enumerate() {
            let (xi, yi) = self.dataset.get(idx)?;
            if i == 0 {
                x_sample_shape = xi.shape().to_vec();
                y_sample_shape = yi.shape().to_vec();
            }
            x_vecs.extend_from_slice(xi.as_slice());
            y_vecs.extend_from_slice(yi.as_slice());
        }

        let batch_len = batch_indices.len();
        let mut x_shape = vec![batch_len];
        x_shape.extend_from_slice(&x_sample_shape);
        let mut y_shape = vec![batch_len];
        y_shape.extend_from_slice(&y_sample_shape);

        let x_batch = Tensor::from_vec(x_vecs, x_shape)?;
        let y_batch = Tensor::from_vec(y_vecs, y_shape)?;

        Ok((x_batch, y_batch))
    }
}

/// Fisher-Yates shuffle.
fn fisher_yates_shuffle(indices: &mut [usize], rng: &mut Rng) {
    let n = indices.len();
    for i in (1..n).rev() {
        let j = (rng.next_u64() as usize) % (i + 1);
        indices.swap(i, j);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scivex_core::Tensor;

    #[test]
    fn test_tensor_dataset() {
        let x = Tensor::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]).unwrap();
        let y = Tensor::from_vec(vec![0.0, 1.0, 2.0], vec![3]).unwrap();
        let ds = TensorDataset::new(x, y).unwrap();
        assert_eq!(ds.len(), 3);

        let (x0, y0) = ds.get(0).unwrap();
        assert_eq!(x0.as_slice(), &[1.0, 2.0]);
        assert_eq!(y0.as_slice(), &[0.0]);
    }

    #[test]
    fn test_dataloader_batches() {
        let x = Tensor::<f64>::from_vec((0..20).map(f64::from).collect(), vec![10, 2]).unwrap();
        let y = Tensor::from_vec((0..10).map(f64::from).collect(), vec![10]).unwrap();
        let ds = TensorDataset::new(x, y).unwrap();
        let loader = DataLoader::new(&ds, 3, false, None);

        assert_eq!(loader.num_batches(), 4); // ceil(10/3)

        let batches: Vec<_> = loader.map(|r| r.unwrap()).collect();
        assert_eq!(batches.len(), 4);
        assert_eq!(batches[0].0.shape(), &[3, 2]);
        assert_eq!(batches[3].0.shape(), &[1, 2]); // last batch is smaller
    }

    #[test]
    fn test_dataloader_shuffle() {
        let x = Tensor::<f64>::from_vec((0..10).map(f64::from).collect(), vec![10, 1]).unwrap();
        let y = Tensor::from_vec((0..10).map(f64::from).collect(), vec![10]).unwrap();
        let ds = TensorDataset::new(x, y).unwrap();
        let mut rng = Rng::new(42);
        let loader = DataLoader::new(&ds, 10, true, Some(&mut rng));

        let batches: Vec<_> = loader.map(|r| r.unwrap()).collect();
        // With shuffle, the order should differ from [0,1,...,9]
        // (with very high probability given seed 42)
        let vals: Vec<f64> = batches[0].1.as_slice().to_vec();
        let sorted: Vec<f64> = (0..10).map(f64::from).collect();
        assert_ne!(vals, sorted, "shuffle did not change order");
    }
}
