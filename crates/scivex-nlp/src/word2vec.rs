//! Word2Vec training — Skip-gram and CBOW with negative sampling.
//!
//! Implements the Word2Vec algorithm from scratch, supporting both the Skip-gram
//! and CBOW (Continuous Bag of Words) architectures. Training uses negative
//! sampling with a smoothed unigram distribution (frequency^0.75).

use std::collections::HashMap;

use scivex_core::Tensor;
use scivex_core::random::Rng;

use crate::embeddings::WordEmbeddings;
use crate::error::{NlpError, Result};

// ---------------------------------------------------------------------------
// Word2VecModel
// ---------------------------------------------------------------------------

/// The Word2Vec training architecture.
///
/// # Examples
///
/// ```
/// # use scivex_nlp::Word2VecModel;
/// let model = Word2VecModel::SkipGram;
/// assert_eq!(model, Word2VecModel::default());
/// ```
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum Word2VecModel {
    /// Predict context words from the center word.
    #[default]
    SkipGram,
    /// Predict the center word from the average of context vectors.
    CBOW,
}

// ---------------------------------------------------------------------------
// Word2VecConfig
// ---------------------------------------------------------------------------

/// Configuration for Word2Vec training (builder pattern).
///
/// # Examples
///
/// ```
/// # use scivex_nlp::Word2VecConfig;
/// let cfg = Word2VecConfig::new().with_dim(50).with_epochs(5);
/// assert_eq!(cfg.embedding_dim, 50);
/// ```
#[derive(Debug, Clone)]
pub struct Word2VecConfig {
    /// Training architecture (Skip-gram or CBOW).
    pub model: Word2VecModel,
    /// Dimensionality of embedding vectors.
    pub embedding_dim: usize,
    /// Context window radius (words to the left and right).
    pub window_size: usize,
    /// Minimum word frequency to include in vocabulary.
    pub min_count: usize,
    /// Initial learning rate for SGD.
    pub learning_rate: f64,
    /// Number of training epochs over the corpus.
    pub epochs: usize,
    /// Number of negative samples per positive pair.
    pub negative_samples: usize,
    /// RNG seed for reproducibility.
    pub seed: u64,
}

impl Default for Word2VecConfig {
    fn default() -> Self {
        Self {
            model: Word2VecModel::SkipGram,
            embedding_dim: 100,
            window_size: 5,
            min_count: 1,
            learning_rate: 0.025,
            epochs: 5,
            negative_samples: 5,
            seed: 42,
        }
    }
}

impl Word2VecConfig {
    /// Create a new configuration with default values.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_nlp::word2vec::{Word2VecConfig, Word2VecModel};
    /// let config = Word2VecConfig::new()
    ///     .with_model(Word2VecModel::SkipGram)
    ///     .with_dim(32)
    ///     .with_epochs(5);
    /// assert_eq!(config.embedding_dim, 32);
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the training model (Skip-gram or CBOW).
    #[must_use]
    pub fn with_model(mut self, model: Word2VecModel) -> Self {
        self.model = model;
        self
    }

    /// Set the embedding dimensionality.
    #[must_use]
    pub fn with_dim(mut self, dim: usize) -> Self {
        self.embedding_dim = dim;
        self
    }

    /// Set the context window radius.
    #[must_use]
    pub fn with_window(mut self, window: usize) -> Self {
        self.window_size = window;
        self
    }

    /// Set the minimum word frequency threshold.
    #[must_use]
    pub fn with_min_count(mut self, min_count: usize) -> Self {
        self.min_count = min_count;
        self
    }

    /// Set the initial learning rate.
    #[must_use]
    pub fn with_lr(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set the number of training epochs.
    #[must_use]
    pub fn with_epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    /// Set the number of negative samples per positive pair.
    #[must_use]
    pub fn with_negative_samples(mut self, n: usize) -> Self {
        self.negative_samples = n;
        self
    }

    /// Set the RNG seed.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

// ---------------------------------------------------------------------------
// Word2VecTrainer
// ---------------------------------------------------------------------------

/// Size of the noise distribution table for fast negative sampling.
const NOISE_TABLE_SIZE: usize = 1_000_000;

/// Sigmoid function clamped to avoid extreme values.
#[inline]
fn sigmoid(x: f64) -> f64 {
    if x > 6.0 {
        1.0
    } else if x < -6.0 {
        0.0
    } else {
        1.0 / (1.0 + (-x).exp())
    }
}

/// Word2Vec trainer holding configuration, vocabulary, and weight matrices.
///
/// # Examples
///
/// ```
/// # use scivex_nlp::{Word2VecConfig, Word2VecTrainer};
/// let trainer = Word2VecTrainer::new(Word2VecConfig::new().with_dim(10).with_epochs(1));
/// ```
pub struct Word2VecTrainer {
    config: Word2VecConfig,
    /// Word to vocabulary index.
    word_to_index: HashMap<String, usize>,
    /// Vocabulary index to word.
    index_to_word: Vec<String>,
    /// Input weight matrix: vocab_size rows x embedding_dim columns (row-major).
    w_in: Vec<f64>,
    /// Output weight matrix: vocab_size rows x embedding_dim columns (row-major).
    w_out: Vec<f64>,
    /// Vocabulary size (after min_count filtering).
    vocab_size: usize,
    /// Noise distribution table for fast negative sampling.
    noise_table: Vec<usize>,
    /// RNG state.
    rng: Rng,
}

impl Word2VecTrainer {
    /// Create a new trainer with the given configuration.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_nlp::word2vec::{Word2VecConfig, Word2VecTrainer};
    /// let trainer = Word2VecTrainer::new(Word2VecConfig::new().with_dim(16));
    /// ```
    #[must_use]
    pub fn new(config: Word2VecConfig) -> Self {
        Self {
            rng: Rng::new(config.seed),
            config,
            word_to_index: HashMap::new(),
            index_to_word: Vec::new(),
            w_in: Vec::new(),
            w_out: Vec::new(),
            vocab_size: 0,
            noise_table: Vec::new(),
        }
    }

    /// Build the vocabulary from the corpus, filtering by `min_count`.
    fn build_vocab(&mut self, corpus: &[&[&str]]) {
        // Count word frequencies.
        let mut freq: HashMap<String, usize> = HashMap::new();
        for sentence in corpus {
            for &word in *sentence {
                *freq.entry(word.to_string()).or_insert(0) += 1;
            }
        }

        // Filter by min_count and assign indices.
        self.word_to_index.clear();
        self.index_to_word.clear();

        // Sort by word for deterministic ordering.
        let mut words: Vec<(String, usize)> = freq
            .into_iter()
            .filter(|(_, count)| *count >= self.config.min_count)
            .collect();
        words.sort_by(|a, b| a.0.cmp(&b.0));

        for (i, (word, _)) in words.iter().enumerate() {
            self.word_to_index.insert(word.clone(), i);
            self.index_to_word.push(word.clone());
        }
        self.vocab_size = self.index_to_word.len();

        // Build noise distribution table (unigram^0.75).
        let freqs: Vec<f64> = words.iter().map(|(_, c)| (*c as f64).powf(0.75)).collect();
        let total: f64 = freqs.iter().sum();

        self.noise_table = Vec::with_capacity(NOISE_TABLE_SIZE);
        let mut cumulative = 0.0;
        let mut word_idx = 0;

        for i in 0..NOISE_TABLE_SIZE {
            let threshold = (i as f64 + 0.5) / NOISE_TABLE_SIZE as f64;
            while word_idx + 1 < self.vocab_size && cumulative + freqs[word_idx] / total < threshold
            {
                cumulative += freqs[word_idx] / total;
                word_idx += 1;
            }
            self.noise_table.push(word_idx);
        }
    }

    /// Initialize weight matrices with small random values.
    fn init_weights(&mut self) {
        let dim = self.config.embedding_dim;
        let n = self.vocab_size * dim;
        let scale = 0.5 / dim as f64;

        self.w_in = (0..n)
            .map(|_| (self.rng.next_f64() - 0.5) * scale)
            .collect();
        self.w_out = vec![0.0; n];
    }

    /// Sample a negative word index from the noise distribution.
    #[inline]
    fn sample_negative(&mut self) -> usize {
        let idx = (self.rng.next_f64() * NOISE_TABLE_SIZE as f64) as usize;
        self.noise_table[idx.min(NOISE_TABLE_SIZE - 1)]
    }

    /// Compute the dot product of a w_in row and a w_out row.
    #[inline]
    fn dot_in_out(&self, in_row: usize, out_row: usize) -> f64 {
        let dim = self.config.embedding_dim;
        let in_start = in_row * dim;
        let out_start = out_row * dim;
        self.w_in[in_start..in_start + dim]
            .iter()
            .zip(&self.w_out[out_start..out_start + dim])
            .map(|(a, b)| a * b)
            .sum()
    }

    /// Train a single Skip-gram (center, context) pair with negative sampling.
    #[allow(clippy::needless_range_loop)]
    fn train_skipgram_pair(&mut self, center: usize, context: usize) {
        let dim = self.config.embedding_dim;
        let lr = self.config.learning_rate;
        let n_neg = self.config.negative_samples;

        // Accumulate gradient for w_in[center].
        let mut grad_in = vec![0.0; dim];
        let in_offset = center * dim;

        // Positive sample: target = 1.
        let score = self.dot_in_out(center, context);
        let grad = sigmoid(score) - 1.0;
        let out_offset = context * dim;
        for d in 0..dim {
            grad_in[d] += grad * self.w_out[out_offset + d];
            self.w_out[out_offset + d] -= lr * grad * self.w_in[in_offset + d];
        }

        // Negative samples: target = 0.
        for _ in 0..n_neg {
            let neg = self.sample_negative();
            if neg == context {
                continue;
            }
            let score = self.dot_in_out(center, neg);
            let grad = sigmoid(score);
            let neg_offset = neg * dim;
            for d in 0..dim {
                grad_in[d] += grad * self.w_out[neg_offset + d];
                self.w_out[neg_offset + d] -= lr * grad * self.w_in[in_offset + d];
            }
        }

        // Update w_in[center].
        for (d, g) in grad_in.iter().enumerate() {
            self.w_in[in_offset + d] -= lr * g;
        }
    }

    /// Train a single CBOW example: predict center from context words.
    #[allow(clippy::needless_range_loop)]
    fn train_cbow(&mut self, center: usize, context_words: &[usize]) {
        if context_words.is_empty() {
            return;
        }

        let dim = self.config.embedding_dim;
        let lr = self.config.learning_rate;
        let n_neg = self.config.negative_samples;
        let n_ctx = context_words.len() as f64;

        // Compute average context vector (from w_in).
        let mut avg = vec![0.0; dim];
        for &ctx in context_words {
            let offset = ctx * dim;
            for d in 0..dim {
                avg[d] += self.w_in[offset + d];
            }
        }
        for val in &mut avg {
            *val /= n_ctx;
        }

        // Accumulate gradient for the average context vector.
        let mut grad_avg = vec![0.0; dim];

        // Positive sample: target = 1.
        let center_offset = center * dim;
        let score: f64 = avg
            .iter()
            .zip(&self.w_out[center_offset..center_offset + dim])
            .map(|(a, b)| a * b)
            .sum();
        let grad = sigmoid(score) - 1.0;
        for d in 0..dim {
            grad_avg[d] += grad * self.w_out[center_offset + d];
            self.w_out[center_offset + d] -= lr * grad * avg[d];
        }

        // Negative samples: target = 0.
        for _ in 0..n_neg {
            let neg = self.sample_negative();
            if neg == center {
                continue;
            }
            let neg_offset = neg * dim;
            let neg_score: f64 = avg
                .iter()
                .zip(&self.w_out[neg_offset..neg_offset + dim])
                .map(|(a, b)| a * b)
                .sum();
            let neg_grad = sigmoid(neg_score);
            for d in 0..dim {
                grad_avg[d] += neg_grad * self.w_out[neg_offset + d];
                self.w_out[neg_offset + d] -= lr * neg_grad * avg[d];
            }
        }

        // Distribute gradient back to each context word's w_in.
        for &ctx in context_words {
            let offset = ctx * dim;
            for (d, g) in grad_avg.iter().enumerate() {
                self.w_in[offset + d] -= lr * g / n_ctx;
            }
        }
    }

    /// Train Word2Vec on the given corpus and return the learned embeddings.
    ///
    /// The corpus is a slice of sentences, where each sentence is a slice of
    /// word tokens.
    ///
    /// # Errors
    ///
    /// Returns `NlpError::EmptyInput` if the corpus is empty, or
    /// `NlpError::EmptyVocabulary` if no words survive the `min_count` filter.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_nlp::word2vec::{Word2VecConfig, Word2VecTrainer};
    /// let corpus = vec![
    ///     vec!["the", "cat", "sat"],
    ///     vec!["the", "dog", "sat"],
    /// ];
    /// let borrowed: Vec<&[&str]> = corpus.iter().map(Vec::as_slice).collect();
    /// let config = Word2VecConfig::new().with_dim(8).with_epochs(2);
    /// let mut trainer = Word2VecTrainer::new(config);
    /// let emb = trainer.train(&borrowed).unwrap();
    /// assert!(emb.get("cat").is_some());
    /// ```
    pub fn train(&mut self, corpus: &[&[&str]]) -> Result<WordEmbeddings<f64>> {
        if corpus.is_empty() {
            return Err(NlpError::EmptyInput);
        }

        self.build_vocab(corpus);
        if self.vocab_size == 0 {
            return Err(NlpError::EmptyVocabulary);
        }
        self.init_weights();

        // Convert corpus to index sequences (skip unknown words).
        let sentences: Vec<Vec<usize>> = corpus
            .iter()
            .map(|s| {
                s.iter()
                    .filter_map(|w| self.word_to_index.get(*w).copied())
                    .collect()
            })
            .collect();

        let window = self.config.window_size;

        for _epoch in 0..self.config.epochs {
            for sentence in &sentences {
                let len = sentence.len();
                if len == 0 {
                    continue;
                }

                for (i, &center) in sentence.iter().enumerate() {
                    let start = i.saturating_sub(window);
                    let end = (i + window + 1).min(len);

                    match self.config.model {
                        Word2VecModel::SkipGram => {
                            for (j, &context) in sentence[start..end].iter().enumerate() {
                                if start + j == i {
                                    continue;
                                }
                                self.train_skipgram_pair(center, context);
                            }
                        }
                        Word2VecModel::CBOW => {
                            let context_words: Vec<usize> = sentence[start..end]
                                .iter()
                                .enumerate()
                                .filter(|&(j, _)| start + j != i)
                                .map(|(_, &w)| w)
                                .collect();
                            self.train_cbow(center, &context_words);
                        }
                    }
                }
            }
        }

        self.to_embeddings()
    }

    /// Convert the current input weight matrix to `WordEmbeddings`.
    ///
    /// # Errors
    ///
    /// Returns `NlpError::EmptyVocabulary` if no vocabulary has been built, or
    /// propagates tensor creation errors from `scivex-core`.
    pub fn to_embeddings(&self) -> Result<WordEmbeddings<f64>> {
        if self.vocab_size == 0 {
            return Err(NlpError::EmptyVocabulary);
        }

        let dim = self.config.embedding_dim;
        let data: Vec<f64> = self.w_in[..self.vocab_size * dim].to_vec();
        let vectors = Tensor::from_vec(data, vec![self.vocab_size, dim])?;
        WordEmbeddings::new(self.index_to_word.clone(), vectors)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// A small corpus for testing.
    fn small_corpus() -> Vec<Vec<&'static str>> {
        vec![
            vec!["the", "cat", "sat", "on", "the", "mat"],
            vec!["the", "dog", "sat", "on", "the", "log"],
            vec!["the", "cat", "chased", "the", "dog"],
            vec!["the", "dog", "chased", "the", "cat"],
            vec!["cat", "and", "dog", "sat", "on", "the", "mat"],
            vec!["the", "cat", "sat", "on", "the", "log"],
            vec!["the", "dog", "sat", "on", "the", "mat"],
            vec!["cat", "and", "dog", "are", "friends"],
            vec!["the", "cat", "is", "on", "the", "mat"],
            vec!["the", "dog", "is", "on", "the", "log"],
        ]
    }

    /// Helper to convert owned corpus to the borrowed form expected by train.
    fn to_borrowed<'a>(corpus: &'a [Vec<&'a str>]) -> Vec<&'a [&'a str]> {
        corpus.iter().map(Vec::as_slice).collect()
    }

    #[test]
    fn train_produces_correct_dimensions() {
        let corpus = small_corpus();
        let borrowed = to_borrowed(&corpus);
        let config = Word2VecConfig::new().with_dim(16).with_epochs(3);
        let mut trainer = Word2VecTrainer::new(config);
        let emb = trainer.train(&borrowed).unwrap();

        // Count unique words in corpus.
        let mut unique = std::collections::HashSet::new();
        for sentence in &corpus {
            for &word in sentence {
                unique.insert(word);
            }
        }

        assert_eq!(emb.vocab_size(), unique.len());
        assert_eq!(emb.embedding_dim(), 16);
    }

    #[test]
    fn skipgram_training_works() {
        let corpus = small_corpus();
        let borrowed = to_borrowed(&corpus);
        let config = Word2VecConfig::new()
            .with_model(Word2VecModel::SkipGram)
            .with_dim(32)
            .with_epochs(10)
            .with_window(3)
            .with_lr(0.05)
            .with_seed(123);
        let mut trainer = Word2VecTrainer::new(config);
        let emb = trainer.train(&borrowed).unwrap();

        // Basic sanity: every word in the corpus should be retrievable.
        assert!(emb.get("cat").is_some());
        assert!(emb.get("dog").is_some());
        assert!(emb.get("the").is_some());
        // Unknown word should be absent.
        assert!(emb.get("elephant").is_none());
    }

    #[test]
    fn cbow_training_works() {
        let corpus = small_corpus();
        let borrowed = to_borrowed(&corpus);
        let config = Word2VecConfig::new()
            .with_model(Word2VecModel::CBOW)
            .with_dim(32)
            .with_epochs(10)
            .with_window(3)
            .with_lr(0.05)
            .with_seed(456);
        let mut trainer = Word2VecTrainer::new(config);
        let emb = trainer.train(&borrowed).unwrap();

        assert!(emb.get("cat").is_some());
        assert!(emb.get("dog").is_some());
        assert!(emb.get("sat").is_some());
    }

    #[test]
    fn similar_words_have_higher_similarity() {
        // Build a corpus where "cat" and "dog" appear in very similar contexts
        // and "mat" / "log" also share contexts but are different from animals.
        let corpus = small_corpus();
        let borrowed = to_borrowed(&corpus);
        let config = Word2VecConfig::new()
            .with_model(Word2VecModel::SkipGram)
            .with_dim(32)
            .with_epochs(50)
            .with_window(3)
            .with_lr(0.05)
            .with_negative_samples(5)
            .with_seed(99);
        let mut trainer = Word2VecTrainer::new(config);
        let emb = trainer.train(&borrowed).unwrap();

        // "cat" and "dog" should be more similar to each other than to "on"
        // because they appear in the same syntactic positions.
        let cat = emb.get("cat").unwrap();
        let dog = emb.get("dog").unwrap();
        let on = emb.get("on").unwrap();

        let sim_cat_dog = crate::similarity::cosine_similarity(&cat, &dog).unwrap();
        let sim_cat_on = crate::similarity::cosine_similarity(&cat, &on).unwrap();

        assert!(
            sim_cat_dog > sim_cat_on,
            "Expected cat-dog similarity ({sim_cat_dog}) > cat-on similarity ({sim_cat_on})"
        );
    }
}
