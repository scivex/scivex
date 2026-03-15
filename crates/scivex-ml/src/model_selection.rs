use scivex_core::{Float, Tensor, random::Rng};

use crate::error::{MlError, Result};
use crate::traits::Predictor;

/// Train/test split result: `(x_train, x_test, y_train, y_test)`.
pub type TrainTestSplit<T> = (Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>);

/// Split data into training and test sets.
///
/// Returns `(x_train, x_test, y_train, y_test)`.
pub fn train_test_split<T: Float>(
    x: &Tensor<T>,
    y: &Tensor<T>,
    test_ratio: T,
    rng: &mut Rng,
) -> Result<TrainTestSplit<T>> {
    let (n, p) = matrix_shape(x)?;
    check_y(y, n)?;

    if test_ratio <= T::zero() || test_ratio >= T::one() {
        return Err(MlError::InvalidParameter {
            name: "test_ratio",
            reason: "must be in (0, 1)",
        });
    }

    // Create shuffled indices
    let mut indices: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = (rng.next_f64() * (i + 1) as f64) as usize;
        let j = j.min(i);
        indices.swap(i, j);
    }

    // Compute split point — use f64 for the multiplication to avoid trait issues
    let test_n = {
        let ratio_f64 = to_f64(test_ratio);
        (n as f64 * ratio_f64).round() as usize
    };
    let test_n = test_n.max(1).min(n - 1);
    let train_n = n - test_n;

    let x_data = x.as_slice();
    let y_data = y.as_slice();

    let mut x_train = vec![T::zero(); train_n * p];
    let mut y_train = vec![T::zero(); train_n];
    let mut x_test = vec![T::zero(); test_n * p];
    let mut y_test = vec![T::zero(); test_n];

    for (out_i, &idx) in indices[..train_n].iter().enumerate() {
        x_train[out_i * p..(out_i + 1) * p].copy_from_slice(&x_data[idx * p..(idx + 1) * p]);
        y_train[out_i] = y_data[idx];
    }
    for (out_i, &idx) in indices[train_n..].iter().enumerate() {
        x_test[out_i * p..(out_i + 1) * p].copy_from_slice(&x_data[idx * p..(idx + 1) * p]);
        y_test[out_i] = y_data[idx];
    }

    Ok((
        Tensor::from_vec(x_train, vec![train_n, p])?,
        Tensor::from_vec(x_test, vec![test_n, p])?,
        Tensor::from_vec(y_train, vec![train_n])?,
        Tensor::from_vec(y_test, vec![test_n])?,
    ))
}

/// K-Fold cross-validation index generator.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct KFold {
    n_folds: usize,
    n_samples: usize,
    indices: Vec<usize>,
}

impl KFold {
    /// Create a new `KFold` with shuffled indices.
    pub fn new(n_folds: usize, n_samples: usize, rng: &mut Rng) -> Result<Self> {
        if n_folds < 2 {
            return Err(MlError::InvalidParameter {
                name: "n_folds",
                reason: "must be at least 2",
            });
        }
        if n_folds > n_samples {
            return Err(MlError::InvalidParameter {
                name: "n_folds",
                reason: "cannot exceed number of samples",
            });
        }
        let mut indices: Vec<usize> = (0..n_samples).collect();
        for i in (1..n_samples).rev() {
            let j = (rng.next_f64() * (i + 1) as f64) as usize;
            let j = j.min(i);
            indices.swap(i, j);
        }
        Ok(Self {
            n_folds,
            n_samples,
            indices,
        })
    }

    /// Iterate over folds, yielding `(train_indices, test_indices)`.
    pub fn iter(&self) -> KFoldIter<'_> {
        KFoldIter {
            kfold: self,
            current: 0,
        }
    }
}

impl<'a> IntoIterator for &'a KFold {
    type Item = (Vec<usize>, Vec<usize>);
    type IntoIter = KFoldIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Iterator over K-Fold cross-validation splits.
pub struct KFoldIter<'a> {
    kfold: &'a KFold,
    current: usize,
}

impl Iterator for KFoldIter<'_> {
    type Item = (Vec<usize>, Vec<usize>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.kfold.n_folds {
            return None;
        }
        let fold_size = self.kfold.n_samples / self.kfold.n_folds;
        let start = self.current * fold_size;
        let end = if self.current == self.kfold.n_folds - 1 {
            self.kfold.n_samples
        } else {
            start + fold_size
        };

        let test_indices: Vec<usize> = self.kfold.indices[start..end].to_vec();
        let train_indices: Vec<usize> = self.kfold.indices[..start]
            .iter()
            .chain(self.kfold.indices[end..].iter())
            .copied()
            .collect();

        self.current += 1;
        Some((train_indices, test_indices))
    }
}

/// Run k-fold cross-validation, returning per-fold scores.
///
/// `metric_fn` takes `(y_true, y_pred)` slices and returns a score.
pub fn cross_val_score<T, M, F>(
    model: &M,
    x: &Tensor<T>,
    y: &Tensor<T>,
    n_folds: usize,
    metric_fn: F,
    rng: &mut Rng,
) -> Result<Vec<T>>
where
    T: Float,
    M: Predictor<T> + Clone,
    F: Fn(&[T], &[T]) -> Result<T>,
{
    let (n, p) = matrix_shape(x)?;
    check_y(y, n)?;

    let kfold = KFold::new(n_folds, n, rng)?;
    let x_data = x.as_slice();
    let y_data = y.as_slice();
    let mut scores = Vec::with_capacity(n_folds);

    for (train_idx, test_idx) in &kfold {
        let train_n = train_idx.len();
        let test_n = test_idx.len();

        let mut x_train = vec![T::zero(); train_n * p];
        let mut y_train = vec![T::zero(); train_n];
        for (out_i, &idx) in train_idx.iter().enumerate() {
            x_train[out_i * p..(out_i + 1) * p].copy_from_slice(&x_data[idx * p..(idx + 1) * p]);
            y_train[out_i] = y_data[idx];
        }

        let mut x_test = vec![T::zero(); test_n * p];
        let mut y_test = vec![T::zero(); test_n];
        for (out_i, &idx) in test_idx.iter().enumerate() {
            x_test[out_i * p..(out_i + 1) * p].copy_from_slice(&x_data[idx * p..(idx + 1) * p]);
            y_test[out_i] = y_data[idx];
        }

        let x_train_t = Tensor::from_vec(x_train, vec![train_n, p])?;
        let y_train_t = Tensor::from_vec(y_train, vec![train_n])?;
        let x_test_t = Tensor::from_vec(x_test, vec![test_n, p])?;

        let mut fold_model = model.clone();
        fold_model.fit(&x_train_t, &y_train_t)?;
        let preds = fold_model.predict(&x_test_t)?;
        let score = metric_fn(&y_test, preds.as_slice())?;
        scores.push(score);
    }

    Ok(scores)
}

// ── helpers ──

fn to_f64<T: Float>(v: T) -> f64 {
    // We know T is f32 or f64. Use a safe approximation:
    // from_usize(1) gives 1.0, we can iterate or use a simpler approach.
    // Since Float doesn't expose `to_f64`, we use the fact that
    // for small values we can compare with from_usize.
    // Actually the simplest: format as string and parse.
    // But that's slow. Instead use a direct bit approach:
    // We can just use 0.0..1.0 stepping.
    // Simplest correct approach: use the Display impl.
    format!("{v}").parse::<f64>().unwrap_or(0.5)
}

fn matrix_shape<T: Float>(x: &Tensor<T>) -> Result<(usize, usize)> {
    let s = x.shape();
    if s.len() != 2 {
        return Err(MlError::DimensionMismatch {
            expected: 2,
            got: s.len(),
        });
    }
    if s[0] == 0 {
        return Err(MlError::EmptyInput);
    }
    Ok((s[0], s[1]))
}

fn check_y<T: Float>(y: &Tensor<T>, n: usize) -> Result<()> {
    if y.ndim() != 1 || y.shape()[0] != n {
        return Err(MlError::DimensionMismatch {
            expected: n,
            got: y.shape()[0],
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_train_test_split_sizes() {
        let x = Tensor::from_vec(
            vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            vec![5, 2],
        )
        .unwrap();
        let y = Tensor::from_vec(vec![0.0, 1.0, 0.0, 1.0, 0.0], vec![5]).unwrap();
        let mut rng = Rng::new(42);
        let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.4, &mut rng).unwrap();
        // 40% of 5 = 2 test, 3 train
        assert_eq!(x_train.shape()[0], 3);
        assert_eq!(x_test.shape()[0], 2);
        assert_eq!(y_train.shape()[0], 3);
        assert_eq!(y_test.shape()[0], 2);
    }

    #[test]
    fn test_kfold_yields_correct_folds() {
        let mut rng = Rng::new(42);
        let kfold = KFold::new(3, 9, &mut rng).unwrap();
        let folds: Vec<_> = kfold.iter().collect();
        assert_eq!(folds.len(), 3);
        for (train, test) in &folds {
            assert_eq!(train.len() + test.len(), 9);
        }
    }

    #[test]
    fn test_cross_val_score() {
        use crate::linear::LinearRegression;
        use crate::metrics::regression::r2_score;

        // y = 2x + 1
        let x = Tensor::from_vec(
            vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            vec![10, 1],
        )
        .unwrap();
        let y = Tensor::from_vec(
            vec![3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0],
            vec![10],
        )
        .unwrap();

        let model = LinearRegression::<f64>::new();
        let mut rng = Rng::new(42);
        let scores = cross_val_score(&model, &x, &y, 3, r2_score, &mut rng).unwrap();
        assert_eq!(scores.len(), 3);
        // R2 should be high for a perfect linear relationship
        for &s in &scores {
            assert!(s > 0.8, "R2 score {s} should be > 0.8");
        }
    }
}
