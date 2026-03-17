//! Latent Dirichlet Allocation (LDA) topic modeling via collapsed Gibbs sampling.

use std::collections::HashMap;

use crate::error::{NlpError, Result};

// ---------------------------------------------------------------------------
// XorShift64 RNG
// ---------------------------------------------------------------------------

/// Minimal xorshift64 pseudo-random number generator.
struct XorShift {
    state: u64,
}

impl XorShift {
    fn new(seed: u64) -> Self {
        // Avoid a zero state which would produce only zeros.
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }

    /// Sample an index from an unnormalized categorical distribution.
    fn sample_categorical(&mut self, probs: &[f64]) -> usize {
        let total: f64 = probs.iter().sum();
        let mut r = self.next_f64() * total;
        for (i, &p) in probs.iter().enumerate() {
            r -= p;
            if r <= 0.0 {
                return i;
            }
        }
        probs.len() - 1
    }
}

// ---------------------------------------------------------------------------
// LdaConfig
// ---------------------------------------------------------------------------

/// Configuration for Latent Dirichlet Allocation.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct LdaConfig {
    /// Number of topics.
    pub n_topics: usize,
    /// Document-topic Dirichlet prior.
    pub alpha: f64,
    /// Topic-word Dirichlet prior.
    pub beta: f64,
    /// Number of Gibbs sampling iterations.
    pub n_iterations: usize,
    /// RNG seed.
    pub seed: u64,
}

impl Default for LdaConfig {
    fn default() -> Self {
        Self {
            n_topics: 10,
            alpha: 0.1,
            beta: 0.01,
            n_iterations: 100,
            seed: 42,
        }
    }
}

impl LdaConfig {
    /// Create a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of topics.
    #[must_use]
    pub fn with_n_topics(mut self, n_topics: usize) -> Self {
        self.n_topics = n_topics;
        self
    }

    /// Set the document-topic prior (alpha).
    #[must_use]
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set the topic-word prior (beta).
    #[must_use]
    pub fn with_beta(mut self, beta: f64) -> Self {
        self.beta = beta;
        self
    }

    /// Set the number of Gibbs sampling iterations.
    #[must_use]
    pub fn with_iterations(mut self, n_iterations: usize) -> Self {
        self.n_iterations = n_iterations;
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
// LdaModel
// ---------------------------------------------------------------------------

/// Latent Dirichlet Allocation model trained via collapsed Gibbs sampling.
pub struct LdaModel {
    /// Model configuration.
    pub config: LdaConfig,
    /// Number of topics (copied from config for convenience).
    pub n_topics: usize,
    /// Vocabulary (index → word).
    pub vocab: Vec<String>,
    /// Word-to-index mapping.
    pub word_to_id: HashMap<String, usize>,
    /// Topic-word counts: `topic_word_counts[k][w]` = number of times word `w`
    /// is assigned to topic `k`.
    pub topic_word_counts: Vec<Vec<usize>>,
    /// Total words assigned to each topic.
    pub topic_counts: Vec<usize>,
    /// Document-topic counts: `doc_topic_counts[d][k]` = number of words in
    /// document `d` assigned to topic `k`.
    pub doc_topic_counts: Vec<Vec<usize>>,
    /// Topic assignment for every word in every document.
    pub topic_assignments: Vec<Vec<usize>>,
}

impl LdaModel {
    /// Create a new, unfitted LDA model with the given configuration.
    #[must_use]
    pub fn new(config: LdaConfig) -> Self {
        let n_topics = config.n_topics;
        Self {
            config,
            n_topics,
            vocab: Vec::new(),
            word_to_id: HashMap::new(),
            topic_word_counts: Vec::new(),
            topic_counts: Vec::new(),
            doc_topic_counts: Vec::new(),
            topic_assignments: Vec::new(),
        }
    }

    /// Return the number of topics.
    #[must_use]
    pub fn n_topics(&self) -> usize {
        self.n_topics
    }

    /// Fit the model on a corpus of tokenised documents.
    ///
    /// Each document is a slice of word tokens (e.g., `&["the", "cat", "sat"]`).
    pub fn fit(&mut self, documents: &[&[&str]]) -> Result<()> {
        if documents.is_empty() {
            return Err(NlpError::EmptyInput);
        }

        if self.n_topics == 0 {
            return Err(NlpError::InvalidParameter {
                name: "n_topics",
                reason: "must be at least 1",
            });
        }

        // --- Build vocabulary ---
        self.vocab.clear();
        self.word_to_id.clear();

        for doc in documents {
            for &word in *doc {
                if !self.word_to_id.contains_key(word) {
                    let id = self.vocab.len();
                    self.word_to_id.insert(word.to_string(), id);
                    self.vocab.push(word.to_string());
                }
            }
        }

        if self.vocab.is_empty() {
            return Err(NlpError::EmptyVocabulary);
        }

        let k = self.n_topics;
        let v = self.vocab.len();
        let n_docs = documents.len();

        // --- Initialise counts ---
        self.topic_word_counts = vec![vec![0usize; v]; k];
        self.topic_counts = vec![0usize; k];
        self.doc_topic_counts = vec![vec![0usize; k]; n_docs];
        self.topic_assignments = Vec::with_capacity(n_docs);

        let mut rng = XorShift::new(self.config.seed);

        // Random initial assignments.
        for (d, doc) in documents.iter().enumerate() {
            let mut assignments = Vec::with_capacity(doc.len());
            for &word in *doc {
                let w = self.word_to_id[word];
                let topic = (rng.next_u64() as usize) % k;
                assignments.push(topic);
                self.topic_word_counts[topic][w] += 1;
                self.topic_counts[topic] += 1;
                self.doc_topic_counts[d][topic] += 1;
            }
            self.topic_assignments.push(assignments);
        }

        // --- Collapsed Gibbs sampling ---
        let alpha = self.config.alpha;
        let beta = self.config.beta;
        let v_beta = (v as f64) * beta;
        let mut probs = vec![0.0f64; k];

        for _iter in 0..self.config.n_iterations {
            for (d, doc) in documents.iter().enumerate() {
                for (i, &word) in doc.iter().enumerate() {
                    let w = self.word_to_id[word];
                    let old_topic = self.topic_assignments[d][i];

                    // 1. Remove current assignment.
                    self.topic_word_counts[old_topic][w] -= 1;
                    self.topic_counts[old_topic] -= 1;
                    self.doc_topic_counts[d][old_topic] -= 1;

                    // 2. Compute conditional distribution.
                    for (t, prob) in probs.iter_mut().enumerate().take(k) {
                        let n_dk = self.doc_topic_counts[d][t] as f64;
                        let n_kw = self.topic_word_counts[t][w] as f64;
                        let n_k = self.topic_counts[t] as f64;
                        *prob = (n_dk + alpha) * (n_kw + beta) / (n_k + v_beta);
                    }

                    // 3. Sample new topic.
                    let new_topic = rng.sample_categorical(&probs);

                    // 4. Increment counts with new assignment.
                    self.topic_assignments[d][i] = new_topic;
                    self.topic_word_counts[new_topic][w] += 1;
                    self.topic_counts[new_topic] += 1;
                    self.doc_topic_counts[d][new_topic] += 1;
                }
            }
        }

        Ok(())
    }

    /// Return the word distribution for a topic, sorted by descending
    /// probability.
    ///
    /// Each entry is `(word, probability)`.
    pub fn topic_word_distribution(&self, topic: usize) -> Result<Vec<(String, f64)>> {
        if topic >= self.n_topics {
            return Err(NlpError::InvalidParameter {
                name: "topic",
                reason: "topic index out of range",
            });
        }
        if self.vocab.is_empty() {
            return Err(NlpError::NotFitted);
        }

        let v = self.vocab.len();
        let beta = self.config.beta;
        let denom = self.topic_counts[topic] as f64 + (v as f64) * beta;

        let mut dist: Vec<(String, f64)> = self
            .vocab
            .iter()
            .enumerate()
            .map(|(w, word)| {
                let prob = (self.topic_word_counts[topic][w] as f64 + beta) / denom;
                (word.clone(), prob)
            })
            .collect();

        dist.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(dist)
    }

    /// Return the topic distribution for a document.
    ///
    /// Returns a vector of length `n_topics` where entry `k` is the
    /// probability of topic `k` in the given document.
    pub fn document_topic_distribution(&self, doc_idx: usize) -> Result<Vec<f64>> {
        if doc_idx >= self.doc_topic_counts.len() {
            return Err(NlpError::InvalidParameter {
                name: "doc_idx",
                reason: "document index out of range",
            });
        }

        let alpha = self.config.alpha;
        let k = self.n_topics;
        let n_d: usize = self.doc_topic_counts[doc_idx].iter().sum();
        let denom = n_d as f64 + (k as f64) * alpha;

        let dist: Vec<f64> = (0..k)
            .map(|t| (self.doc_topic_counts[doc_idx][t] as f64 + alpha) / denom)
            .collect();

        Ok(dist)
    }

    /// Return the top `n` words for a given topic, sorted by descending
    /// probability.
    pub fn top_words(&self, topic: usize, n: usize) -> Result<Vec<(String, f64)>> {
        let mut dist = self.topic_word_distribution(topic)?;
        dist.truncate(n);
        Ok(dist)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn small_corpus() -> Vec<Vec<&'static str>> {
        vec![
            vec!["cat", "dog", "cat", "fish"],
            vec!["dog", "dog", "cat", "bird"],
            vec!["fish", "fish", "bird", "cat"],
            vec!["bird", "bird", "dog", "fish"],
        ]
    }

    fn corpus_slices<'a>(corpus: &'a [Vec<&'a str>]) -> Vec<&'a [&'a str]> {
        corpus.iter().map(Vec::as_slice).collect()
    }

    #[test]
    fn fit_produces_correct_n_topics() {
        let corpus = small_corpus();
        let docs = corpus_slices(&corpus);
        let config = LdaConfig::new().with_n_topics(3).with_iterations(20);
        let mut model = LdaModel::new(config);
        model.fit(&docs).unwrap();
        assert_eq!(model.n_topics(), 3);
    }

    #[test]
    fn topic_word_distributions_sum_to_one() {
        let corpus = small_corpus();
        let docs = corpus_slices(&corpus);
        let config = LdaConfig::new().with_n_topics(2).with_iterations(30);
        let mut model = LdaModel::new(config);
        model.fit(&docs).unwrap();

        for k in 0..model.n_topics() {
            let dist = model.topic_word_distribution(k).unwrap();
            let total: f64 = dist.iter().map(|(_, p)| p).sum();
            assert!(
                (total - 1.0).abs() < 1e-9,
                "topic {k} word dist sums to {total}"
            );
        }
    }

    #[test]
    fn document_topic_distributions_sum_to_one() {
        let corpus = small_corpus();
        let docs = corpus_slices(&corpus);
        let config = LdaConfig::new().with_n_topics(2).with_iterations(30);
        let mut model = LdaModel::new(config);
        model.fit(&docs).unwrap();

        for d in 0..docs.len() {
            let dist = model.document_topic_distribution(d).unwrap();
            let total: f64 = dist.iter().sum();
            assert!(
                (total - 1.0).abs() < 1e-9,
                "doc {d} topic dist sums to {total}"
            );
        }
    }

    #[test]
    fn top_words_returns_correct_count() {
        let corpus = small_corpus();
        let docs = corpus_slices(&corpus);
        let config = LdaConfig::new().with_n_topics(2).with_iterations(20);
        let mut model = LdaModel::new(config);
        model.fit(&docs).unwrap();

        let top = model.top_words(0, 2).unwrap();
        assert_eq!(top.len(), 2);

        // Asking for more than vocab size should return vocab size.
        let top_all = model.top_words(0, 100).unwrap();
        assert_eq!(top_all.len(), model.vocab.len());
    }
}
