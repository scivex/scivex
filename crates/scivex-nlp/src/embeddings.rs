//! Word embeddings storage and lookup.

use std::collections::HashMap;

use scivex_core::{Float, Tensor};

use crate::error::{NlpError, Result};
use crate::similarity::cosine_similarity;

/// Pre-computed word embedding vectors with lookup and similarity search.
///
/// # Examples
///
/// ```
/// # use scivex_nlp::WordEmbeddings;
/// let emb = WordEmbeddings::from_pairs(&[
///     ("cat".into(), vec![1.0_f64, 0.0]),
///     ("dog".into(), vec![0.9, 0.1]),
/// ]).unwrap();
/// assert_eq!(emb.vocab_size(), 2);
/// ```
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct WordEmbeddings<T: Float> {
    /// Matrix of shape (vocab_size, embedding_dim).
    vectors: Tensor<T>,
    word_to_index: HashMap<String, usize>,
    index_to_word: Vec<String>,
}

impl<T: Float> WordEmbeddings<T> {
    /// Create embeddings from a word list and a matrix of shape `(vocab_size, dim)`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_nlp::embeddings::WordEmbeddings;
    /// # use scivex_core::Tensor;
    /// let words = vec!["cat".into(), "dog".into()];
    /// let vecs = Tensor::from_vec(vec![1.0_f64, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
    /// let emb = WordEmbeddings::new(words, vecs).unwrap();
    /// assert_eq!(emb.vocab_size(), 2);
    /// ```
    pub fn new(words: Vec<String>, vectors: Tensor<T>) -> Result<Self> {
        if words.is_empty() {
            return Err(NlpError::EmptyVocabulary);
        }
        if vectors.ndim() != 2 {
            return Err(NlpError::InvalidParameter {
                name: "vectors",
                reason: "expected a 2-D tensor (vocab_size × embedding_dim)",
            });
        }
        if vectors.shape()[0] != words.len() {
            return Err(NlpError::InvalidParameter {
                name: "vectors",
                reason: "first dimension must equal number of words",
            });
        }

        let mut word_to_index = HashMap::with_capacity(words.len());
        for (i, w) in words.iter().enumerate() {
            word_to_index.insert(w.clone(), i);
        }

        Ok(Self {
            vectors,
            word_to_index,
            index_to_word: words,
        })
    }

    /// Create embeddings from (word, vector) pairs.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_nlp::embeddings::WordEmbeddings;
    /// let pairs = vec![
    ///     ("hello".into(), vec![1.0_f64, 0.0]),
    ///     ("world".into(), vec![0.0, 1.0]),
    /// ];
    /// let emb = WordEmbeddings::from_pairs(&pairs).unwrap();
    /// assert_eq!(emb.embedding_dim(), 2);
    /// ```
    pub fn from_pairs(pairs: &[(String, Vec<T>)]) -> Result<Self> {
        if pairs.is_empty() {
            return Err(NlpError::EmptyVocabulary);
        }
        let dim = pairs[0].1.len();
        if dim == 0 {
            return Err(NlpError::InvalidParameter {
                name: "pairs",
                reason: "embedding dimension must be > 0",
            });
        }

        let mut words = Vec::with_capacity(pairs.len());
        let mut data = Vec::with_capacity(pairs.len() * dim);

        for (word, vec) in pairs {
            if vec.len() != dim {
                return Err(NlpError::InvalidParameter {
                    name: "pairs",
                    reason: "all vectors must have the same dimension",
                });
            }
            words.push(word.clone());
            data.extend_from_slice(vec);
        }

        let vectors = Tensor::from_vec(data, vec![words.len(), dim])?;
        Self::new(words, vectors)
    }

    /// Look up the embedding vector for a word.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_nlp::embeddings::WordEmbeddings;
    /// let pairs = vec![("cat".into(), vec![1.0_f64, 2.0, 3.0])];
    /// let emb = WordEmbeddings::from_pairs(&pairs).unwrap();
    /// let v = emb.get("cat").unwrap();
    /// assert_eq!(v.as_slice(), &[1.0, 2.0, 3.0]);
    /// assert!(emb.get("dog").is_none());
    /// ```
    #[must_use]
    pub fn get(&self, word: &str) -> Option<Tensor<T>> {
        let &idx = self.word_to_index.get(word)?;
        let dim = self.embedding_dim();
        let start = idx * dim;
        let slice = &self.vectors.as_slice()[start..start + dim];
        Tensor::from_vec(slice.to_vec(), vec![dim]).ok()
    }

    /// Find the `top_k` most similar words by cosine similarity.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_nlp::embeddings::WordEmbeddings;
    /// let pairs = vec![
    ///     ("king".into(), vec![1.0_f64, 0.0, 1.0]),
    ///     ("queen".into(), vec![1.0, 1.0, 0.0]),
    ///     ("man".into(), vec![0.0, 0.0, 1.0]),
    /// ];
    /// let emb = WordEmbeddings::from_pairs(&pairs).unwrap();
    /// let similar = emb.most_similar("king", 2).unwrap();
    /// assert_eq!(similar.len(), 2);
    /// ```
    pub fn most_similar(&self, word: &str, top_k: usize) -> Result<Vec<(String, T)>> {
        let query = self.get(word).ok_or_else(|| NlpError::UnknownToken {
            token: word.to_string(),
        })?;
        let query_idx = self.word_to_index[word];

        let mut scores: Vec<(usize, T)> = Vec::with_capacity(self.vocab_size());
        let dim = self.embedding_dim();

        for i in 0..self.vocab_size() {
            if i == query_idx {
                continue;
            }
            let start = i * dim;
            let vec_i = Tensor::from_vec(
                self.vectors.as_slice()[start..start + dim].to_vec(),
                vec![dim],
            )?;
            let sim = cosine_similarity(&query, &vec_i)?;
            scores.push((i, sim));
        }

        // Sort descending by similarity.
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(top_k);

        Ok(scores
            .into_iter()
            .map(|(i, s)| (self.index_to_word[i].clone(), s))
            .collect())
    }

    /// Solve analogies: "a is to b as c is to ?"
    ///
    /// Computes `b - a + c` and finds the nearest vectors (excluding a, b, c).
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_nlp::embeddings::WordEmbeddings;
    /// let pairs = vec![
    ///     ("king".into(), vec![1.0_f64, 0.0, 1.0]),
    ///     ("queen".into(), vec![1.0, 1.0, 0.0]),
    ///     ("man".into(), vec![0.0, 0.0, 1.0]),
    ///     ("woman".into(), vec![0.0, 1.0, 0.0]),
    /// ];
    /// let emb = WordEmbeddings::from_pairs(&pairs).unwrap();
    /// let result = emb.analogy("man", "king", "woman", 1).unwrap();
    /// assert_eq!(result[0].0, "queen");
    /// ```
    pub fn analogy(&self, a: &str, b: &str, c: &str, top_k: usize) -> Result<Vec<(String, T)>> {
        let va = self.get(a).ok_or_else(|| NlpError::UnknownToken {
            token: a.to_string(),
        })?;
        let vb = self.get(b).ok_or_else(|| NlpError::UnknownToken {
            token: b.to_string(),
        })?;
        let vc = self.get(c).ok_or_else(|| NlpError::UnknownToken {
            token: c.to_string(),
        })?;

        // target = b - a + c
        let dim = self.embedding_dim();
        let mut target_data = vec![T::zero(); dim];
        let va_s = va.as_slice();
        let vb_s = vb.as_slice();
        let vc_s = vc.as_slice();
        for i in 0..dim {
            target_data[i] = vb_s[i] - va_s[i] + vc_s[i];
        }
        let target = Tensor::from_vec(target_data, vec![dim])?;

        let exclude: std::collections::HashSet<usize> = [
            self.word_to_index[a],
            self.word_to_index[b],
            self.word_to_index[c],
        ]
        .into_iter()
        .collect();

        let mut scores: Vec<(usize, T)> = Vec::with_capacity(self.vocab_size());
        for i in 0..self.vocab_size() {
            if exclude.contains(&i) {
                continue;
            }
            let start = i * dim;
            let vec_i = Tensor::from_vec(
                self.vectors.as_slice()[start..start + dim].to_vec(),
                vec![dim],
            )?;
            let sim = cosine_similarity(&target, &vec_i)?;
            scores.push((i, sim));
        }

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(top_k);

        Ok(scores
            .into_iter()
            .map(|(i, s)| (self.index_to_word[i].clone(), s))
            .collect())
    }

    /// Number of words in the vocabulary.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_nlp::embeddings::WordEmbeddings;
    /// let pairs = vec![("a".into(), vec![1.0_f64]), ("b".into(), vec![2.0])];
    /// let emb = WordEmbeddings::from_pairs(&pairs).unwrap();
    /// assert_eq!(emb.vocab_size(), 2);
    /// ```
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.index_to_word.len()
    }

    /// Dimensionality of each embedding vector.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_nlp::embeddings::WordEmbeddings;
    /// let pairs = vec![("x".into(), vec![1.0_f64, 2.0, 3.0])];
    /// let emb = WordEmbeddings::from_pairs(&pairs).unwrap();
    /// assert_eq!(emb.embedding_dim(), 3);
    /// ```
    #[must_use]
    pub fn embedding_dim(&self) -> usize {
        if self.vectors.ndim() == 2 {
            self.vectors.shape()[1]
        } else {
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_embeddings() -> WordEmbeddings<f64> {
        let pairs = vec![
            ("king".to_string(), vec![1.0, 0.0, 1.0]),
            ("queen".to_string(), vec![1.0, 1.0, 0.0]),
            ("man".to_string(), vec![0.0, 0.0, 1.0]),
            ("woman".to_string(), vec![0.0, 1.0, 0.0]),
        ];
        WordEmbeddings::from_pairs(&pairs).unwrap()
    }

    #[test]
    fn get_returns_correct_vector() {
        let emb = make_embeddings();
        let v = emb.get("king").unwrap();
        assert_eq!(v.as_slice(), &[1.0, 0.0, 1.0]);
    }

    #[test]
    fn get_unknown_returns_none() {
        let emb = make_embeddings();
        assert!(emb.get("emperor").is_none());
    }

    #[test]
    fn most_similar_sorted() {
        let emb = make_embeddings();
        let result = emb.most_similar("king", 3).unwrap();
        assert_eq!(result.len(), 3);
        // Scores should be descending.
        for w in result.windows(2) {
            assert!(w[0].1 >= w[1].1);
        }
    }

    #[test]
    fn analogy_works() {
        let emb = make_embeddings();
        // king - man + woman ≈ queen
        let result = emb.analogy("man", "king", "woman", 1).unwrap();
        assert_eq!(result[0].0, "queen");
    }

    #[test]
    fn vocab_size_and_dim() {
        let emb = make_embeddings();
        assert_eq!(emb.vocab_size(), 4);
        assert_eq!(emb.embedding_dim(), 3);
    }
}
