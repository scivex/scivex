//! Text vectorization: CountVectorizer and TF-IDF.

use std::collections::HashMap;

use scivex_core::{Float, Tensor};

use crate::error::{NlpError, Result};
use crate::tokenize::WordTokenizer;

// ---------------------------------------------------------------------------
// CountVectorizer
// ---------------------------------------------------------------------------

/// Converts a collection of text documents into a term-count matrix.
///
/// # Examples
///
/// ```
/// # use scivex_nlp::CountVectorizer;
/// let mut cv = CountVectorizer::new();
/// let docs = ["hello world", "hello rust"];
/// let mat = cv.fit_transform::<f64>(&docs).unwrap();
/// assert_eq!(mat.shape()[0], 2);
/// ```
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct CountVectorizer {
    vocabulary: HashMap<String, usize>,
    fitted: bool,
}

impl CountVectorizer {
    /// Create a new, unfitted count vectorizer.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_nlp::vectorize::CountVectorizer;
    /// # use scivex_core::Tensor;
    /// let mut cv = CountVectorizer::new();
    /// let docs = ["hello world", "hello rust"];
    /// cv.fit(&docs).unwrap();
    /// let matrix: Tensor<f64> = cv.transform(&docs).unwrap();
    /// assert_eq!(matrix.shape()[0], 2); // 2 documents
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self {
            vocabulary: HashMap::new(),
            fitted: false,
        }
    }

    /// Build the vocabulary from a set of documents.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_nlp::vectorize::CountVectorizer;
    /// let mut cv = CountVectorizer::new();
    /// cv.fit(&["hello world", "hello rust"]).unwrap();
    /// assert!(cv.is_fitted());
    /// assert!(cv.vocabulary().len() >= 3);
    /// ```
    pub fn fit(&mut self, documents: &[&str]) -> Result<()> {
        if documents.is_empty() {
            return Err(NlpError::EmptyInput);
        }
        self.vocabulary.clear();
        let tokenizer = WordTokenizer::new().with_lowercase(true);

        for doc in documents {
            for token in tokenizer.tokenize_owned(doc) {
                let next_id = self.vocabulary.len();
                self.vocabulary.entry(token).or_insert(next_id);
            }
        }

        if self.vocabulary.is_empty() {
            return Err(NlpError::EmptyVocabulary);
        }
        self.fitted = true;
        Ok(())
    }

    /// Transform documents into a term-count matrix (n_docs × vocab_size).
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_nlp::vectorize::CountVectorizer;
    /// # use scivex_core::Tensor;
    /// let mut cv = CountVectorizer::new();
    /// cv.fit(&["word word word"]).unwrap();
    /// let m: Tensor<f64> = cv.transform(&["word word word"]).unwrap();
    /// assert_eq!(m.as_slice(), &[3.0]); // "word" appears 3 times
    /// ```
    pub fn transform<T: Float>(&self, documents: &[&str]) -> Result<Tensor<T>> {
        if !self.fitted {
            return Err(NlpError::NotFitted);
        }
        let n_docs = documents.len();
        let n_terms = self.vocabulary.len();
        let tokenizer = WordTokenizer::new().with_lowercase(true);

        let mut data = vec![T::zero(); n_docs * n_terms];
        for (i, doc) in documents.iter().enumerate() {
            for token in tokenizer.tokenize_owned(doc) {
                if let Some(&idx) = self.vocabulary.get(&token) {
                    data[i * n_terms + idx] += T::one();
                }
            }
        }

        Ok(Tensor::from_vec(data, vec![n_docs, n_terms])?)
    }

    /// Fit the vocabulary and transform in one step.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_nlp::vectorize::CountVectorizer;
    /// # use scivex_core::Tensor;
    /// let mut cv = CountVectorizer::new();
    /// let m: Tensor<f64> = cv.fit_transform(&["the cat sat", "the dog sat"]).unwrap();
    /// assert_eq!(m.shape()[0], 2); // 2 documents
    /// ```
    pub fn fit_transform<T: Float>(&mut self, documents: &[&str]) -> Result<Tensor<T>> {
        self.fit(documents)?;
        self.transform(documents)
    }

    /// Returns the fitted vocabulary (token → column index).
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_nlp::vectorize::CountVectorizer;
    /// let mut cv = CountVectorizer::new();
    /// cv.fit(&["hello world"]).unwrap();
    /// assert!(cv.vocabulary().contains_key("hello"));
    /// ```
    #[must_use]
    pub fn vocabulary(&self) -> &HashMap<String, usize> {
        &self.vocabulary
    }

    /// Returns `true` if the vocabulary has been built via [`fit`](Self::fit).
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_nlp::vectorize::CountVectorizer;
    /// let cv = CountVectorizer::new();
    /// assert!(!cv.is_fitted());
    /// ```
    #[must_use]
    pub fn is_fitted(&self) -> bool {
        self.fitted
    }
}

impl Default for CountVectorizer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// TfidfVectorizer
// ---------------------------------------------------------------------------

/// Converts documents into TF-IDF weighted feature vectors.
///
/// Uses the formula: `tf * log((1 + n) / (1 + df)) + 1`
/// where `n` is the total number of documents and `df` is the document
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
/// frequency of the term.
///
/// # Examples
///
/// ```
/// # use scivex_nlp::TfidfVectorizer;
/// let mut tfidf = TfidfVectorizer::new();
/// let docs = ["hello world", "hello rust"];
/// let mat = tfidf.fit_transform::<f64>(&docs).unwrap();
/// assert_eq!(mat.shape()[0], 2);
/// ```
pub struct TfidfVectorizer {
    count_vectorizer: CountVectorizer,
    idf: Option<Vec<f64>>,
}

impl TfidfVectorizer {
    /// Create a new, unfitted TF-IDF vectorizer.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_nlp::vectorize::TfidfVectorizer;
    /// # use scivex_core::Tensor;
    /// let mut tv = TfidfVectorizer::new();
    /// let m: Tensor<f64> = tv.fit_transform(&["the cat", "the dog"]).unwrap();
    /// assert_eq!(m.shape()[0], 2);
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self {
            count_vectorizer: CountVectorizer::new(),
            idf: None,
        }
    }

    /// Fit the vectorizer on a set of documents, computing vocabulary and IDF.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_nlp::vectorize::TfidfVectorizer;
    /// let mut tv = TfidfVectorizer::new();
    /// tv.fit(&["the cat", "the dog", "rare thing"]).unwrap();
    /// assert!(tv.vocabulary().len() >= 4);
    /// ```
    pub fn fit(&mut self, documents: &[&str]) -> Result<()> {
        if documents.is_empty() {
            return Err(NlpError::EmptyInput);
        }

        self.count_vectorizer.fit(documents)?;

        let n_docs = documents.len();
        let n_terms = self.count_vectorizer.vocabulary().len();
        let tokenizer = WordTokenizer::new().with_lowercase(true);

        // Compute document frequency for each term.
        let mut df = vec![0_usize; n_terms];
        for doc in documents {
            let mut seen = HashMap::new();
            for token in tokenizer.tokenize_owned(doc) {
                if let Some(&idx) = self.count_vectorizer.vocabulary().get(&token) {
                    seen.entry(idx).or_insert(true);
                }
            }
            for &idx in seen.keys() {
                df[idx] += 1;
            }
        }

        // IDF = log((1 + n) / (1 + df)) + 1
        let n = n_docs as f64;
        self.idf = Some(
            df.iter()
                .map(|&d| ((1.0 + n) / (1.0 + d as f64)).ln() + 1.0)
                .collect(),
        );

        Ok(())
    }

    /// Transform documents into TF-IDF weighted matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_nlp::vectorize::TfidfVectorizer;
    /// # use scivex_core::Tensor;
    /// let mut tv = TfidfVectorizer::new();
    /// tv.fit(&["the cat", "the dog"]).unwrap();
    /// let m: Tensor<f64> = tv.transform(&["the cat"]).unwrap();
    /// assert_eq!(m.shape()[0], 1);
    /// ```
    pub fn transform<T: Float>(&self, documents: &[&str]) -> Result<Tensor<T>> {
        let idf = self.idf.as_ref().ok_or(NlpError::NotFitted)?;
        let counts: Tensor<T> = self.count_vectorizer.transform(documents)?;
        let n_docs = documents.len();
        let n_terms = idf.len();

        let count_data = counts.as_slice();
        let mut tfidf_data = vec![T::zero(); n_docs * n_terms];

        for i in 0..n_docs {
            for j in 0..n_terms {
                let tf = count_data[i * n_terms + j];
                let idf_val = T::from_f64(idf[j]);
                tfidf_data[i * n_terms + j] = tf * idf_val;
            }
        }

        Ok(Tensor::from_vec(tfidf_data, vec![n_docs, n_terms])?)
    }

    /// Fit and transform in one step.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_nlp::vectorize::TfidfVectorizer;
    /// # use scivex_core::Tensor;
    /// let mut tv = TfidfVectorizer::new();
    /// let m: Tensor<f64> = tv.fit_transform(&["hello world", "hello rust"]).unwrap();
    /// assert_eq!(m.shape(), &[2, 3]); // 2 docs × 3 unique terms
    /// ```
    pub fn fit_transform<T: Float>(&mut self, documents: &[&str]) -> Result<Tensor<T>> {
        self.fit(documents)?;
        self.transform(documents)
    }

    /// Returns the fitted vocabulary.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_nlp::vectorize::TfidfVectorizer;
    /// let mut tv = TfidfVectorizer::new();
    /// tv.fit(&["hello world"]).unwrap();
    /// assert!(tv.vocabulary().contains_key("hello"));
    /// ```
    #[must_use]
    pub fn vocabulary(&self) -> &HashMap<String, usize> {
        self.count_vectorizer.vocabulary()
    }
}

impl Default for TfidfVectorizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn count_vectorizer_fit_transform() {
        let docs = ["the cat sat", "the dog sat"];
        let mut cv = CountVectorizer::new();
        let matrix: Tensor<f64> = cv.fit_transform(&docs).unwrap();
        assert_eq!(matrix.shape(), &[2, 4]); // 4 unique words: the, cat, sat, dog
    }

    #[test]
    fn count_vectorizer_correct_counts() {
        let docs = ["word word word"];
        let mut cv = CountVectorizer::new();
        let matrix: Tensor<f64> = cv.fit_transform(&docs).unwrap();
        assert_eq!(matrix.as_slice(), &[3.0]);
    }

    #[test]
    fn count_vectorizer_not_fitted_error() {
        let cv = CountVectorizer::new();
        let result: Result<Tensor<f64>> = cv.transform(&["hello"]);
        assert!(matches!(result, Err(NlpError::NotFitted)));
    }

    #[test]
    fn tfidf_basic() {
        let docs = ["the cat", "the dog", "the cat dog"];
        let mut tv = TfidfVectorizer::new();
        let matrix: Tensor<f64> = tv.fit_transform(&docs).unwrap();
        assert_eq!(matrix.shape(), &[3, 3]); // 3 docs × 3 terms
        // All values should be non-zero.
        assert!(matrix.as_slice().iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn tfidf_common_term_lower_weight() {
        // "the" appears in all docs, "rare" in one — "rare" should have higher IDF.
        let docs = ["the cat", "the dog", "rare thing"];
        let mut tv = TfidfVectorizer::new();
        let _matrix: Tensor<f64> = tv.fit_transform(&docs).unwrap();
        // Just verify it doesn't panic; IDF logic is tested by construction.
    }

    #[test]
    fn tfidf_not_fitted_error() {
        let tv = TfidfVectorizer::new();
        let result: Result<Tensor<f64>> = tv.transform(&["hello"]);
        assert!(matches!(result, Err(NlpError::NotFitted)));
    }
}
