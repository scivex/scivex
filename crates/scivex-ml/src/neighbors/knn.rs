use scivex_core::{Float, Tensor};

use crate::error::{MlError, Result};
use crate::traits::Predictor;

/// K-nearest neighbours classifier (brute-force Euclidean distance).
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct KNNClassifier<T: Float> {
    k: usize,
    x_train: Option<Vec<T>>,
    y_train: Option<Vec<T>>,
    n_train: usize,
    n_features: usize,
}

impl<T: Float> KNNClassifier<T> {
    /// Create a new KNN classifier with `k` neighbours.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_core::Tensor;
    /// # use scivex_ml::neighbors::KNNClassifier;
    /// # use scivex_ml::traits::Predictor;
    /// let x = Tensor::from_vec(vec![0.0_f64, 0.0, 10.0, 10.0], vec![2, 2]).unwrap();
    /// let y = Tensor::from_vec(vec![0.0, 1.0], vec![2]).unwrap();
    /// let mut knn = KNNClassifier::new(1).unwrap();
    /// knn.fit(&x, &y).unwrap();
    /// let preds = knn.predict(&x).unwrap();
    /// assert_eq!(preds.as_slice(), &[0.0, 1.0]);
    /// ```
    pub fn new(k: usize) -> Result<Self> {
        if k == 0 {
            return Err(MlError::InvalidParameter {
                name: "k",
                reason: "must be at least 1",
            });
        }
        Ok(Self {
            k,
            x_train: None,
            y_train: None,
            n_train: 0,
            n_features: 0,
        })
    }
}

impl<T: Float> Predictor<T> for KNNClassifier<T> {
    fn fit(&mut self, x: &Tensor<T>, y: &Tensor<T>) -> Result<()> {
        let (n, p) = matrix_shape(x)?;
        check_y(y, n)?;
        self.x_train = Some(x.as_slice().to_vec());
        self.y_train = Some(y.as_slice().to_vec());
        self.n_train = n;
        self.n_features = p;
        Ok(())
    }

    fn predict(&self, x: &Tensor<T>) -> Result<Tensor<T>> {
        let x_tr = self.x_train.as_ref().ok_or(MlError::NotFitted)?;
        let y_tr = self.y_train.as_ref().ok_or(MlError::NotFitted)?;
        let (n, p) = matrix_shape(x)?;
        if p != self.n_features {
            return Err(MlError::DimensionMismatch {
                expected: self.n_features,
                got: p,
            });
        }
        let data = x.as_slice();
        let k = self.k.min(self.n_train);
        let mut out = vec![T::zero(); n];

        for i in 0..n {
            let query = &data[i * p..(i + 1) * p];
            let mut dists: Vec<(T, T)> = (0..self.n_train)
                .map(|j| {
                    let d = euclidean_dist(query, &x_tr[j * p..(j + 1) * p]);
                    (d, y_tr[j])
                })
                .collect();
            dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            // Majority vote among k nearest
            out[i] = majority_vote_knn(&dists[..k]);
        }
        Ok(Tensor::from_vec(out, vec![n])?)
    }
}

/// K-nearest neighbours regressor (brute-force Euclidean distance).
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct KNNRegressor<T: Float> {
    k: usize,
    x_train: Option<Vec<T>>,
    y_train: Option<Vec<T>>,
    n_train: usize,
    n_features: usize,
}

impl<T: Float> KNNRegressor<T> {
    /// Create a new KNN regressor with `k` neighbours.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_core::Tensor;
    /// # use scivex_ml::neighbors::KNNRegressor;
    /// # use scivex_ml::traits::Predictor;
    /// let x = Tensor::from_vec(vec![1.0_f64, 3.0, 5.0], vec![3, 1]).unwrap();
    /// let y = Tensor::from_vec(vec![2.0, 6.0, 10.0], vec![3]).unwrap();
    /// let mut knn = KNNRegressor::new(2).unwrap();
    /// knn.fit(&x, &y).unwrap();
    /// ```
    pub fn new(k: usize) -> Result<Self> {
        if k == 0 {
            return Err(MlError::InvalidParameter {
                name: "k",
                reason: "must be at least 1",
            });
        }
        Ok(Self {
            k,
            x_train: None,
            y_train: None,
            n_train: 0,
            n_features: 0,
        })
    }
}

impl<T: Float> Predictor<T> for KNNRegressor<T> {
    fn fit(&mut self, x: &Tensor<T>, y: &Tensor<T>) -> Result<()> {
        let (n, p) = matrix_shape(x)?;
        check_y(y, n)?;
        self.x_train = Some(x.as_slice().to_vec());
        self.y_train = Some(y.as_slice().to_vec());
        self.n_train = n;
        self.n_features = p;
        Ok(())
    }

    fn predict(&self, x: &Tensor<T>) -> Result<Tensor<T>> {
        let x_tr = self.x_train.as_ref().ok_or(MlError::NotFitted)?;
        let y_tr = self.y_train.as_ref().ok_or(MlError::NotFitted)?;
        let (n, p) = matrix_shape(x)?;
        if p != self.n_features {
            return Err(MlError::DimensionMismatch {
                expected: self.n_features,
                got: p,
            });
        }
        let data = x.as_slice();
        let k = self.k.min(self.n_train);
        let mut out = vec![T::zero(); n];

        for i in 0..n {
            let query = &data[i * p..(i + 1) * p];
            let mut dists: Vec<(T, T)> = (0..self.n_train)
                .map(|j| {
                    let d = euclidean_dist(query, &x_tr[j * p..(j + 1) * p]);
                    (d, y_tr[j])
                })
                .collect();
            dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            // Average of k nearest
            let sum = dists[..k]
                .iter()
                .map(|&(_, y)| y)
                .fold(T::zero(), |a, b| a + b);
            out[i] = sum / T::from_usize(k);
        }
        Ok(Tensor::from_vec(out, vec![n])?)
    }
}

fn euclidean_dist<T: Float>(a: &[T], b: &[T]) -> T {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| {
            let d = x - y;
            d * d
        })
        .fold(T::zero(), |acc, v| acc + v)
        .sqrt()
}

fn majority_vote_knn<T: Float>(nearest: &[(T, T)]) -> T {
    let mut counts: Vec<(T, usize)> = Vec::new();
    for &(_, label) in nearest {
        if let Some(entry) = counts
            .iter_mut()
            .find(|(c, _)| (*c - label).abs() < T::epsilon())
        {
            entry.1 += 1;
        } else {
            counts.push((label, 1));
        }
    }
    counts
        .into_iter()
        .max_by_key(|&(_, c)| c)
        .map_or(T::zero(), |(v, _)| v)
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
    fn test_knn_k1_memorises() {
        let x = Tensor::from_vec(vec![1.0_f64, 1.0, 5.0, 5.0, 9.0, 9.0], vec![3, 2]).unwrap();
        let y = Tensor::from_vec(vec![0.0, 1.0, 2.0], vec![3]).unwrap();

        let mut knn = KNNClassifier::new(1).unwrap();
        knn.fit(&x, &y).unwrap();
        let preds = knn.predict(&x).unwrap();
        assert_eq!(preds.as_slice(), y.as_slice());
    }

    #[test]
    fn test_knn_classifier_simple() {
        let x = Tensor::from_vec(
            vec![0.0_f64, 0.0, 0.1, 0.1, 10.0, 10.0, 10.1, 10.1],
            vec![4, 2],
        )
        .unwrap();
        let y = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0], vec![4]).unwrap();

        let mut knn = KNNClassifier::new(1).unwrap();
        knn.fit(&x, &y).unwrap();

        let test_x = Tensor::from_vec(vec![0.05_f64, 0.05, 10.05, 10.05], vec![2, 2]).unwrap();
        let preds = knn.predict(&test_x).unwrap();
        assert_eq!(preds.as_slice(), &[0.0, 1.0]);
    }

    #[test]
    fn test_knn_regressor() {
        let x = Tensor::from_vec(vec![1.0_f64, 3.0, 5.0], vec![3, 1]).unwrap();
        let y = Tensor::from_vec(vec![2.0, 6.0, 10.0], vec![3]).unwrap();

        let mut knn = KNNRegressor::new(2).unwrap();
        knn.fit(&x, &y).unwrap();

        // Query at x=2 should average the 2 nearest: y=2 and y=6 → 4
        let test_x = Tensor::from_vec(vec![2.0_f64], vec![1, 1]).unwrap();
        let preds = knn.predict(&test_x).unwrap();
        assert!((preds.as_slice()[0] - 4.0).abs() < 1e-10);
    }
}
