use scivex_core::{Float, Tensor, random::Rng};

use crate::error::{MlError, Result};
use crate::model_selection::cross_val_score;
use crate::traits::Predictor;

/// Result of a hyperparameter search (grid or random).
///
/// # Examples
///
/// ```
/// # use scivex_ml::search::SearchResult;
/// let result = SearchResult::<f64> {
///     best_index: 1,
///     best_score: 0.95_f64,
///     mean_scores: vec![0.90, 0.95, 0.92],
///     all_scores: vec![vec![0.90], vec![0.95], vec![0.92]],
/// };
/// assert_eq!(result.best_index, 1);
/// ```
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct SearchResult<T: Float> {
    /// Index of the best model in the candidate list.
    pub best_index: usize,
    /// Mean cross-validation score of the best model.
    pub best_score: T,
    /// Mean cross-validation scores for every candidate.
    pub mean_scores: Vec<T>,
    /// Per-fold scores for every candidate (outer = candidate, inner = fold).
    pub all_scores: Vec<Vec<T>>,
}

/// Exhaustive search over a list of pre-configured model candidates.
///
/// Each candidate is evaluated via k-fold cross-validation using the given
/// `scorer` (higher is better). Returns a [`SearchResult`] with the best
/// candidate index and all scores.
///
/// # Example
///
/// ```ignore
/// use scivex_ml::search::grid_search_cv;
/// use scivex_ml::linear::Ridge;
///
/// let candidates = vec![
///     Ridge::<f64>::new(0.01).unwrap(),
///     Ridge::<f64>::new(0.1).unwrap(),
///     Ridge::<f64>::new(1.0).unwrap(),
/// ];
/// let result = grid_search_cv(&candidates, &x, &y, 5, r2_score, &mut rng)?;
/// println!("Best alpha index: {}", result.best_index);
/// ```
pub fn grid_search_cv<T, M, S>(
    candidates: &[M],
    x: &Tensor<T>,
    y: &Tensor<T>,
    n_folds: usize,
    scorer: S,
    rng: &mut Rng,
) -> Result<SearchResult<T>>
where
    T: Float,
    M: Predictor<T> + Clone,
    S: Fn(&[T], &[T]) -> Result<T>,
{
    if candidates.is_empty() {
        return Err(MlError::InvalidParameter {
            name: "candidates",
            reason: "must provide at least one candidate model",
        });
    }

    let mut all_scores = Vec::with_capacity(candidates.len());
    let mut mean_scores = Vec::with_capacity(candidates.len());

    for candidate in candidates {
        let fold_scores = cross_val_score(candidate, x, y, n_folds, &scorer, rng)?;
        let mean = fold_scores.iter().copied().fold(T::zero(), |a, b| a + b)
            / T::from_usize(fold_scores.len());
        mean_scores.push(mean);
        all_scores.push(fold_scores);
    }

    let best_index = mean_scores
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(i, _)| i);

    Ok(SearchResult {
        best_index,
        best_score: mean_scores[best_index],
        mean_scores,
        all_scores,
    })
}

/// Random search: evaluate `n_iter` randomly-built model candidates.
///
/// The `builder` closure receives an RNG and should return a fresh model
/// with randomly sampled hyperparameters. Each candidate is evaluated via
/// k-fold cross-validation.
pub fn random_search_cv<T, M, B, S>(
    builder: B,
    n_iter: usize,
    x: &Tensor<T>,
    y: &Tensor<T>,
    n_folds: usize,
    scorer: S,
    rng: &mut Rng,
) -> Result<SearchResult<T>>
where
    T: Float,
    M: Predictor<T> + Clone,
    B: Fn(&mut Rng) -> Result<M>,
    S: Fn(&[T], &[T]) -> Result<T>,
{
    if n_iter == 0 {
        return Err(MlError::InvalidParameter {
            name: "n_iter",
            reason: "must be at least 1",
        });
    }

    let mut all_scores = Vec::with_capacity(n_iter);
    let mut mean_scores = Vec::with_capacity(n_iter);

    for _ in 0..n_iter {
        let candidate = builder(rng)?;
        let fold_scores = cross_val_score(&candidate, x, y, n_folds, &scorer, rng)?;
        let mean = fold_scores.iter().copied().fold(T::zero(), |a, b| a + b)
            / T::from_usize(fold_scores.len());
        mean_scores.push(mean);
        all_scores.push(fold_scores);
    }

    let best_index = mean_scores
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(i, _)| i);

    Ok(SearchResult {
        best_index,
        best_score: mean_scores[best_index],
        mean_scores,
        all_scores,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linear::Ridge;
    use crate::metrics::regression::r2_score;

    fn make_linear_data() -> (Tensor<f64>, Tensor<f64>) {
        let x = Tensor::from_vec(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0, 17.0, 18.0, 19.0, 20.0,
            ],
            vec![20, 1],
        )
        .unwrap();
        let y = Tensor::from_vec(
            vec![
                3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0, 23.0, 25.0, 27.0, 29.0,
                31.0, 33.0, 35.0, 37.0, 39.0, 41.0,
            ],
            vec![20],
        )
        .unwrap();
        (x, y)
    }

    #[test]
    fn test_grid_search_cv() {
        let (x, y) = make_linear_data();
        let candidates = vec![
            Ridge::<f64>::new(0.001).unwrap(),
            Ridge::<f64>::new(0.1).unwrap(),
            Ridge::<f64>::new(10.0).unwrap(),
            Ridge::<f64>::new(1000.0).unwrap(),
        ];
        let mut rng = Rng::new(42);
        let result = grid_search_cv(&candidates, &x, &y, 3, r2_score, &mut rng).unwrap();

        assert_eq!(result.mean_scores.len(), 4);
        assert_eq!(result.all_scores.len(), 4);
        assert!(result.best_score > 0.5, "best R2 should be reasonable");
        // Low regularisation should win for a perfect linear relationship
        assert!(
            result.best_index < 3,
            "heavy regularisation should not be best"
        );
    }

    #[test]
    fn test_grid_search_empty_candidates() {
        let (x, y) = make_linear_data();
        let candidates: Vec<Ridge<f64>> = vec![];
        let mut rng = Rng::new(42);
        assert!(grid_search_cv(&candidates, &x, &y, 3, r2_score, &mut rng).is_err());
    }

    #[test]
    fn test_random_search_cv() {
        let (x, y) = make_linear_data();
        let mut rng = Rng::new(42);

        let result = random_search_cv(
            |r| {
                let alpha = r.next_f64() * 10.0 + 0.001; // random alpha in [0.001, 10]
                Ridge::<f64>::new(alpha)
            },
            5,
            &x,
            &y,
            3,
            r2_score,
            &mut rng,
        )
        .unwrap();

        assert_eq!(result.mean_scores.len(), 5);
        assert!(result.best_score > 0.5);
    }

    #[test]
    fn test_random_search_zero_iter() {
        let (x, y) = make_linear_data();
        let mut rng = Rng::new(42);
        let result = random_search_cv(|_| Ridge::<f64>::new(1.0), 0, &x, &y, 3, r2_score, &mut rng);
        assert!(result.is_err());
    }
}
