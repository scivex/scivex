//! Python bindings for scivex-nlp — natural language processing.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use scivex_nlp::prelude::*;
use std::collections::HashMap;

use crate::tensor::PyTensor;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn py_err(e: impl std::fmt::Display) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}

// ---------------------------------------------------------------------------
// Tokenizers
// ---------------------------------------------------------------------------

/// Tokenize text into lowercase words, splitting on whitespace and punctuation.
///
/// Returns a list of word tokens.
#[pyfunction]
fn word_tokenize(text: &str) -> Vec<String> {
    let tok = WordTokenizer::new().with_lowercase(true);
    tok.tokenize_owned(text)
}

/// Tokenize text into individual characters.
///
/// Returns a list of single-character strings.
#[pyfunction]
fn char_tokenize(text: &str) -> Vec<String> {
    let tok = CharTokenizer;
    tok.tokenize(text)
        .into_iter()
        .map(|s| s.to_string())
        .collect()
}

/// Tokenize text into character-level n-grams of the given size.
///
/// Parameters: `text` — input string, `n` — n-gram size.
/// Returns a list of n-gram strings.
#[pyfunction]
fn ngram_tokenize(text: &str, n: usize) -> Vec<String> {
    let tok = NGramTokenizer::new(n);
    tok.tokenize(text)
        .into_iter()
        .map(|s| s.to_string())
        .collect()
}

/// Tokenize text by splitting on whitespace only.
///
/// Returns a list of whitespace-delimited tokens.
#[pyfunction]
fn whitespace_tokenize(text: &str) -> Vec<String> {
    let tok = WhitespaceTokenizer;
    tok.tokenize(text)
        .into_iter()
        .map(|s| s.to_string())
        .collect()
}

// ---------------------------------------------------------------------------
// WordPiece tokenizer
// ---------------------------------------------------------------------------

#[pyclass(name = "WordPieceTokenizer")]
pub struct PyWordPieceTokenizer {
    inner: WordPieceTokenizer,
}

#[pymethods]
impl PyWordPieceTokenizer {
    /// Create a new WordPieceTokenizer from a vocabulary list.
    #[new]
    fn new(vocab: Vec<String>) -> PyResult<Self> {
        let refs: Vec<&str> = vocab.iter().map(|s| s.as_str()).collect();
        let inner = WordPieceTokenizer::from_vocab_list(&refs).map_err(py_err)?;
        Ok(Self { inner })
    }

    /// Tokenize text into WordPiece sub-word tokens.
    fn tokenize(&self, text: &str) -> Vec<String> {
        self.inner.tokenize_to_pieces(text)
    }

    /// Encode text into a list of vocabulary token IDs.
    fn encode(&self, text: &str) -> PyResult<Vec<usize>> {
        self.inner.encode(text).map_err(py_err)
    }

    /// Decode a list of token IDs back into a string.
    fn decode(&self, ids: Vec<usize>) -> String {
        self.inner.decode(&ids)
    }

    fn __repr__(&self) -> String {
        format!("WordPieceTokenizer(vocab_size={})", self.inner.vocab.len())
    }
}

// ---------------------------------------------------------------------------
// Unigram (SentencePiece) tokenizer
// ---------------------------------------------------------------------------

/// SentencePiece-style unigram tokenizer using Viterbi decoding.
#[pyclass(name = "UnigramTokenizer")]
pub struct PyUnigramTokenizer {
    inner: UnigramTokenizer,
}

#[pymethods]
impl PyUnigramTokenizer {
    /// Create a new UnigramTokenizer from a list of (piece, log_probability) tuples.
    #[new]
    fn new(pieces: Vec<(String, f64)>, unk_id: usize) -> PyResult<Self> {
        let inner = UnigramTokenizer::new(pieces, unk_id).map_err(py_err)?;
        Ok(Self { inner })
    }

    /// Tokenize text into sub-word string pieces.
    fn tokenize(&self, text: &str) -> Vec<String> {
        self.inner.tokenize_str(text)
    }

    /// Encode text into a list of piece IDs.
    fn encode(&self, text: &str) -> Vec<usize> {
        self.inner.encode(text)
    }

    /// Decode a list of piece IDs back into a string.
    fn decode(&self, ids: Vec<usize>) -> PyResult<String> {
        self.inner.decode(&ids).map_err(py_err)
    }

    /// Return the vocabulary size.
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    fn __repr__(&self) -> String {
        format!("UnigramTokenizer(vocab_size={})", self.inner.vocab_size())
    }
}

// ---------------------------------------------------------------------------
// Stemming
// ---------------------------------------------------------------------------

#[pyclass(name = "PorterStemmer")]
pub struct PyPorterStemmer {
    inner: PorterStemmer,
}

#[pymethods]
impl PyPorterStemmer {
    /// Create a new Porter stemmer instance.
    #[new]
    fn new() -> Self {
        Self {
            inner: PorterStemmer,
        }
    }

    /// Stem a single word using the Porter stemming algorithm.
    fn stem(&self, word: &str) -> String {
        self.inner.stem(word)
    }

    /// Stem a list of words, returning their stemmed forms.
    fn stem_many(&self, words: Vec<String>) -> Vec<String> {
        words.iter().map(|w| self.inner.stem(w)).collect()
    }

    fn __repr__(&self) -> String {
        "PorterStemmer()".to_string()
    }
}

// ---------------------------------------------------------------------------
// Stopwords
// ---------------------------------------------------------------------------

/// Return the default list of English stopwords.
#[pyfunction]
fn stopwords() -> Vec<String> {
    scivex_nlp::text::stopwords()
        .iter()
        .map(|s| s.to_string())
        .collect()
}

/// Check whether a word is in the default English stopword list.
#[pyfunction]
fn is_stopword(word: &str) -> bool {
    scivex_nlp::text::is_stopword(word)
}

/// Remove English stopwords from a list of tokens.
#[pyfunction]
fn remove_stopwords(tokens: Vec<String>) -> Vec<String> {
    let refs: Vec<&str> = tokens.iter().map(|s| s.as_str()).collect();
    scivex_nlp::text::remove_stopwords(&refs)
        .into_iter()
        .map(|s| s.to_string())
        .collect()
}

// ---------------------------------------------------------------------------
// Text utilities
// ---------------------------------------------------------------------------

/// Generate word-level n-grams from a list of tokens.
///
/// Parameters: `tokens` — input token list, `n` — n-gram size.
/// Returns a list of n-gram groups (each a list of strings).
#[pyfunction]
fn ngrams(tokens: Vec<String>, n: usize) -> Vec<Vec<String>> {
    let refs: Vec<&str> = tokens.iter().map(|s| s.as_str()).collect();
    scivex_nlp::text::ngrams(&refs, n)
        .into_iter()
        .map(|g| g.into_iter().map(|s| s.to_string()).collect())
        .collect()
}

/// Compute the Levenshtein edit distance between two strings.
#[pyfunction]
fn edit_distance(a: &str, b: &str) -> usize {
    scivex_nlp::text::levenshtein(a, b)
}

/// Normalize text by lowercasing, stripping extra whitespace, and removing punctuation.
#[pyfunction]
fn normalize(text: &str) -> String {
    scivex_nlp::text::normalize(text)
}

/// Pad or truncate integer sequences to a uniform length.
///
/// Parameters: `sequences` — list of token-ID sequences, `max_len` — target length,
/// `pad_value` — value used for padding shorter sequences.
#[pyfunction]
fn pad_sequences(sequences: Vec<Vec<usize>>, max_len: usize, pad_value: usize) -> Vec<Vec<usize>> {
    scivex_nlp::text::pad_sequences(&sequences, max_len, pad_value)
}

// ---------------------------------------------------------------------------
// Similarity
// ---------------------------------------------------------------------------

/// Compute cosine similarity between two f64 vectors. Returns a value in [-1, 1].
#[pyfunction]
fn cosine_sim(a: Vec<f64>, b: Vec<f64>) -> PyResult<f64> {
    let n = a.len();
    let ta = scivex_core::Tensor::from_vec(a, vec![n]).map_err(py_err)?;
    let tb = scivex_core::Tensor::from_vec(b, vec![n]).map_err(py_err)?;
    cosine_similarity(&ta, &tb).map_err(py_err)
}

/// Compute Jaccard similarity between two sets of strings. Returns a value in [0, 1].
#[pyfunction]
fn jaccard(a: Vec<String>, b: Vec<String>) -> f64 {
    let a_refs: Vec<&str> = a.iter().map(|s| s.as_str()).collect();
    let b_refs: Vec<&str> = b.iter().map(|s| s.as_str()).collect();
    jaccard_similarity(&a_refs, &b_refs)
}

/// Compute normalized edit distance between two strings. Returns a value in [0, 1].
#[pyfunction]
fn edit_distance_norm(a: &str, b: &str) -> f64 {
    edit_distance_normalized(a, b)
}

// ---------------------------------------------------------------------------
// CountVectorizer
// ---------------------------------------------------------------------------

#[pyclass(name = "CountVectorizer")]
pub struct PyCountVectorizer {
    inner: CountVectorizer,
}

#[pymethods]
impl PyCountVectorizer {
    /// Create a new, unfitted CountVectorizer.
    #[new]
    fn new() -> Self {
        Self {
            inner: CountVectorizer::new(),
        }
    }

    /// Build the vocabulary from a list of documents.
    fn fit(&mut self, documents: Vec<String>) -> PyResult<()> {
        let refs: Vec<&str> = documents.iter().map(|s| s.as_str()).collect();
        self.inner.fit(&refs).map_err(py_err)
    }

    /// Transform documents into a term-count matrix using the fitted vocabulary.
    /// Returns a Tensor of shape (n_documents, vocab_size).
    fn transform(&self, documents: Vec<String>) -> PyResult<PyTensor> {
        let refs: Vec<&str> = documents.iter().map(|s| s.as_str()).collect();
        let t: scivex_core::Tensor<f64> = self.inner.transform(&refs).map_err(py_err)?;
        Ok(PyTensor::from_f64(t))
    }

    /// Fit the vocabulary and transform documents in a single step.
    /// Returns a Tensor of shape (n_documents, vocab_size).
    fn fit_transform(&mut self, documents: Vec<String>) -> PyResult<PyTensor> {
        let refs: Vec<&str> = documents.iter().map(|s| s.as_str()).collect();
        let t: scivex_core::Tensor<f64> = self.inner.fit_transform(&refs).map_err(py_err)?;
        Ok(PyTensor::from_f64(t))
    }

    /// Return the vocabulary mapping (word -> index).
    fn vocabulary(&self) -> HashMap<String, usize> {
        self.inner.vocabulary().clone()
    }

    /// Return whether the vectorizer has been fitted.
    fn is_fitted(&self) -> bool {
        self.inner.is_fitted()
    }

    fn __repr__(&self) -> String {
        format!(
            "CountVectorizer(fitted={}, vocab_size={})",
            self.inner.is_fitted(),
            self.inner.vocabulary().len()
        )
    }
}

// ---------------------------------------------------------------------------
// TfidfVectorizer
// ---------------------------------------------------------------------------

#[pyclass(name = "TfidfVectorizer")]
pub struct PyTfidfVectorizer {
    inner: TfidfVectorizer,
}

#[pymethods]
impl PyTfidfVectorizer {
    /// Create a new, unfitted TfidfVectorizer.
    #[new]
    fn new() -> Self {
        Self {
            inner: TfidfVectorizer::new(),
        }
    }

    /// Build the vocabulary and IDF weights from a list of documents.
    fn fit(&mut self, documents: Vec<String>) -> PyResult<()> {
        let refs: Vec<&str> = documents.iter().map(|s| s.as_str()).collect();
        self.inner.fit(&refs).map_err(py_err)
    }

    /// Transform documents into a TF-IDF weighted matrix.
    /// Returns a Tensor of shape (n_documents, vocab_size).
    fn transform(&self, documents: Vec<String>) -> PyResult<PyTensor> {
        let refs: Vec<&str> = documents.iter().map(|s| s.as_str()).collect();
        let t: scivex_core::Tensor<f64> = self.inner.transform(&refs).map_err(py_err)?;
        Ok(PyTensor::from_f64(t))
    }

    /// Fit the vocabulary/IDF and transform documents in a single step.
    /// Returns a Tensor of shape (n_documents, vocab_size).
    fn fit_transform(&mut self, documents: Vec<String>) -> PyResult<PyTensor> {
        let refs: Vec<&str> = documents.iter().map(|s| s.as_str()).collect();
        let t: scivex_core::Tensor<f64> = self.inner.fit_transform(&refs).map_err(py_err)?;
        Ok(PyTensor::from_f64(t))
    }

    /// Return the vocabulary mapping (word -> index).
    fn vocabulary(&self) -> HashMap<String, usize> {
        self.inner.vocabulary().clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "TfidfVectorizer(vocab_size={})",
            self.inner.vocabulary().len()
        )
    }
}

// ---------------------------------------------------------------------------
// Word2Vec
// ---------------------------------------------------------------------------

#[pyclass(name = "Word2Vec", unsendable)]
pub struct PyWord2Vec {
    embeddings: Option<WordEmbeddings<f64>>,
}

#[pymethods]
impl PyWord2Vec {
    /// Create a new Word2Vec instance with the given hyperparameters.
    /// Call `Word2Vec.train(...)` to actually train on a corpus.
    #[new]
    #[pyo3(signature = (embedding_dim = 50, window_size = 5, min_count = 1, learning_rate = 0.025, epochs = 5, negative_samples = 5, seed = 42, model = "skipgram"))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        embedding_dim: usize,
        window_size: usize,
        min_count: usize,
        learning_rate: f64,
        epochs: usize,
        negative_samples: usize,
        seed: u64,
        model: &str,
    ) -> PyResult<Self> {
        let m = match model.to_lowercase().as_str() {
            "skipgram" | "skip_gram" | "skip-gram" => Word2VecModel::SkipGram,
            "cbow" => Word2VecModel::CBOW,
            _ => {
                return Err(py_err(format!(
                    "unknown model: {model} (use skipgram/cbow)"
                )));
            }
        };
        // Store config but don't train yet
        let _config = Word2VecConfig::new()
            .with_model(m)
            .with_dim(embedding_dim)
            .with_window(window_size)
            .with_min_count(min_count)
            .with_lr(learning_rate)
            .with_epochs(epochs)
            .with_negative_samples(negative_samples)
            .with_seed(seed);
        Ok(Self { embeddings: None })
    }

    /// Train on a corpus of sentences (list of lists of words).
    #[pyo3(signature = (corpus, embedding_dim = 50, window_size = 5, min_count = 1, learning_rate = 0.025, epochs = 5, negative_samples = 5, seed = 42, model = "skipgram"))]
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    fn train(
        corpus: Vec<Vec<String>>,
        embedding_dim: usize,
        window_size: usize,
        min_count: usize,
        learning_rate: f64,
        epochs: usize,
        negative_samples: usize,
        seed: u64,
        model: &str,
    ) -> PyResult<Self> {
        let m = match model.to_lowercase().as_str() {
            "skipgram" | "skip_gram" | "skip-gram" => Word2VecModel::SkipGram,
            "cbow" => Word2VecModel::CBOW,
            _ => return Err(py_err(format!("unknown model: {model}"))),
        };
        let config = Word2VecConfig::new()
            .with_model(m)
            .with_dim(embedding_dim)
            .with_window(window_size)
            .with_min_count(min_count)
            .with_lr(learning_rate)
            .with_epochs(epochs)
            .with_negative_samples(negative_samples)
            .with_seed(seed);

        // Convert corpus to &[&[&str]]
        let corpus_refs: Vec<Vec<&str>> = corpus
            .iter()
            .map(|sent| sent.iter().map(|w| w.as_str()).collect())
            .collect();
        let corpus_slices: Vec<&[&str]> = corpus_refs.iter().map(|s| s.as_slice()).collect();

        let mut trainer = Word2VecTrainer::new(config);
        let embeddings = trainer.train(&corpus_slices).map_err(py_err)?;
        Ok(Self {
            embeddings: Some(embeddings),
        })
    }

    /// Get the embedding vector for a word, or None if not in vocabulary.
    fn get(&self, word: &str) -> PyResult<Option<Vec<f64>>> {
        let emb = self
            .embeddings
            .as_ref()
            .ok_or_else(|| py_err("model not trained"))?;
        Ok(emb.get(word).map(|t| t.as_slice().to_vec()))
    }

    /// Find the `top_k` most similar words by cosine similarity.
    fn most_similar(&self, word: &str, top_k: usize) -> PyResult<Vec<(String, f64)>> {
        let emb = self
            .embeddings
            .as_ref()
            .ok_or_else(|| py_err("model not trained"))?;
        emb.most_similar(word, top_k).map_err(py_err)
    }

    /// Solve word analogy "a is to b as c is to ?". Returns `top_k` results.
    fn analogy(&self, a: &str, b: &str, c: &str, top_k: usize) -> PyResult<Vec<(String, f64)>> {
        let emb = self
            .embeddings
            .as_ref()
            .ok_or_else(|| py_err("model not trained"))?;
        emb.analogy(a, b, c, top_k).map_err(py_err)
    }

    /// Return the number of words in the vocabulary.
    fn vocab_size(&self) -> PyResult<usize> {
        let emb = self
            .embeddings
            .as_ref()
            .ok_or_else(|| py_err("model not trained"))?;
        Ok(emb.vocab_size())
    }

    /// Return the dimensionality of the embedding vectors.
    fn embedding_dim(&self) -> PyResult<usize> {
        let emb = self
            .embeddings
            .as_ref()
            .ok_or_else(|| py_err("model not trained"))?;
        Ok(emb.embedding_dim())
    }

    fn __repr__(&self) -> String {
        match &self.embeddings {
            Some(e) => format!(
                "Word2Vec(vocab_size={}, dim={})",
                e.vocab_size(),
                e.embedding_dim()
            ),
            None => "Word2Vec(not trained)".to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// WordEmbeddings (from pairs)
// ---------------------------------------------------------------------------

#[pyclass(name = "WordEmbeddings", unsendable)]
pub struct PyWordEmbeddings {
    inner: WordEmbeddings<f64>,
}

#[pymethods]
impl PyWordEmbeddings {
    /// Create embeddings from a list of (word, vector) pairs.
    #[new]
    fn new(pairs: Vec<(String, Vec<f64>)>) -> PyResult<Self> {
        let inner = WordEmbeddings::from_pairs(&pairs).map_err(py_err)?;
        Ok(Self { inner })
    }

    /// Get the embedding vector for a word, or None if not in vocabulary.
    fn get(&self, word: &str) -> Option<Vec<f64>> {
        self.inner.get(word).map(|t| t.as_slice().to_vec())
    }

    /// Find the `top_k` most similar words by cosine similarity.
    fn most_similar(&self, word: &str, top_k: usize) -> PyResult<Vec<(String, f64)>> {
        self.inner.most_similar(word, top_k).map_err(py_err)
    }

    /// Solve word analogy "a is to b as c is to ?". Returns `top_k` results.
    fn analogy(&self, a: &str, b: &str, c: &str, top_k: usize) -> PyResult<Vec<(String, f64)>> {
        self.inner.analogy(a, b, c, top_k).map_err(py_err)
    }

    /// Compute cosine similarity between two words in the embedding space.
    fn similarity(&self, a: &str, b: &str) -> PyResult<f64> {
        let va = self
            .inner
            .get(a)
            .ok_or_else(|| py_err(format!("word not found: {a}")))?;
        let vb = self
            .inner
            .get(b)
            .ok_or_else(|| py_err(format!("word not found: {b}")))?;
        cosine_similarity(&va, &vb).map_err(py_err)
    }

    /// Return the number of words in the vocabulary.
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    /// Return the dimensionality of the embedding vectors.
    fn embedding_dim(&self) -> usize {
        self.inner.embedding_dim()
    }

    fn __repr__(&self) -> String {
        format!(
            "WordEmbeddings(vocab_size={}, dim={})",
            self.inner.vocab_size(),
            self.inner.embedding_dim()
        )
    }
}

// ---------------------------------------------------------------------------
// Sentiment
// ---------------------------------------------------------------------------

#[pyclass(name = "SentimentAnalyzer")]
pub struct PySentimentAnalyzer {
    inner: SentimentAnalyzer,
}

#[pymethods]
impl PySentimentAnalyzer {
    /// Create a new lexicon-based sentiment analyzer.
    #[new]
    fn new() -> Self {
        Self {
            inner: SentimentAnalyzer::new(),
        }
    }

    /// Analyze the sentiment of the given text.
    /// Returns a dict with keys "polarity", "positive", and "negative".
    fn analyze(&self, text: &str, py: Python<'_>) -> PyResult<PyObject> {
        let result = self.inner.analyze(text);
        let dict = PyDict::new(py);
        dict.set_item("polarity", result.polarity)?;
        dict.set_item("positive", result.positive)?;
        dict.set_item("negative", result.negative)?;
        Ok(dict.into_any().unbind())
    }

    fn __repr__(&self) -> String {
        "SentimentAnalyzer()".to_string()
    }
}

// ---------------------------------------------------------------------------
// POS Tagging
// ---------------------------------------------------------------------------

#[pyclass(name = "HmmPosTagger")]
pub struct PyHmmPosTagger {
    inner: HmmPosTagger,
}

#[pymethods]
impl PyHmmPosTagger {
    /// Create a new HMM-based part-of-speech tagger with default parameters.
    #[new]
    fn new() -> Self {
        Self {
            inner: HmmPosTagger::new(),
        }
    }

    /// Tag a raw text string. Returns a list of (word, tag) pairs.
    fn tag(&self, text: &str) -> Vec<(String, String)> {
        self.inner
            .tag_str(text)
            .into_iter()
            .map(|(w, t)| (w, t.to_string()))
            .collect()
    }

    /// Tag pre-tokenized words. Returns a list of (word, tag) pairs.
    fn tag_tokens(&self, tokens: Vec<String>) -> Vec<(String, String)> {
        let refs: Vec<&str> = tokens.iter().map(|s| s.as_str()).collect();
        self.inner
            .tag(&refs)
            .into_iter()
            .map(|(w, t)| (w, t.to_string()))
            .collect()
    }

    fn __repr__(&self) -> String {
        "HmmPosTagger()".to_string()
    }
}

// ---------------------------------------------------------------------------
// NER
// ---------------------------------------------------------------------------

#[pyclass(name = "RuleBasedNer")]
pub struct PyRuleBasedNer {
    inner: RuleBasedNer,
}

#[pymethods]
impl PyRuleBasedNer {
    /// Create a new rule-based named entity recognizer with an empty entity list.
    #[new]
    fn new() -> Self {
        Self {
            inner: RuleBasedNer::new(),
        }
    }

    /// Register a named entity. `entity_type` is one of: person, organization, location, date, number, other.
    fn add_entity(&mut self, entity_type: &str, name: &str) -> PyResult<()> {
        let et = parse_entity_type(entity_type)?;
        self.inner.add_entity(et, name);
        Ok(())
    }

    /// Recognize named entities in a list of tokens.
    /// Returns a list of dicts with keys: text, type, start, end.
    fn recognize(&self, tokens: Vec<String>) -> PyResult<Vec<PyObject>> {
        let refs: Vec<&str> = tokens.iter().map(|s| s.as_str()).collect();
        let entities = self.inner.recognize(&refs);
        Python::with_gil(|py| {
            let mut result = Vec::with_capacity(entities.len());
            for e in &entities {
                let dict = PyDict::new(py);
                dict.set_item("text", &e.text)?;
                dict.set_item("type", e.entity_type.to_string())?;
                dict.set_item("start", e.start)?;
                dict.set_item("end", e.end)?;
                result.push(dict.into_any().unbind());
            }
            Ok(result)
        })
    }

    /// Recognize named entities in raw text (whitespace-tokenized internally).
    fn recognize_text(&self, text: &str) -> PyResult<Vec<PyObject>> {
        let tokens: Vec<String> = text.split_whitespace().map(|s| s.to_string()).collect();
        self.recognize(tokens)
    }

    fn __repr__(&self) -> String {
        "RuleBasedNer()".to_string()
    }
}

fn parse_entity_type(s: &str) -> PyResult<EntityType> {
    match s.to_lowercase().as_str() {
        "person" => Ok(EntityType::Person),
        "organization" | "org" => Ok(EntityType::Organization),
        "location" | "loc" => Ok(EntityType::Location),
        "date" => Ok(EntityType::Date),
        "number" => Ok(EntityType::Number),
        "other" => Ok(EntityType::Other),
        _ => Err(py_err(format!("unknown entity type: {s}"))),
    }
}

// ---------------------------------------------------------------------------
// LDA
// ---------------------------------------------------------------------------

#[pyclass(name = "LDA", unsendable)]
pub struct PyLda {
    inner: LdaModel,
}

#[pymethods]
impl PyLda {
    /// Create a new LDA model with the given hyperparameters.
    #[new]
    #[pyo3(signature = (n_topics = 10, alpha = 0.1, beta = 0.01, n_iterations = 100, seed = 42))]
    fn new(n_topics: usize, alpha: f64, beta: f64, n_iterations: usize, seed: u64) -> Self {
        let config = LdaConfig::new()
            .with_n_topics(n_topics)
            .with_alpha(alpha)
            .with_beta(beta)
            .with_iterations(n_iterations)
            .with_seed(seed);
        Self {
            inner: LdaModel::new(config),
        }
    }

    /// Fit the LDA model on a corpus of tokenized documents.
    fn fit(&mut self, documents: Vec<Vec<String>>) -> PyResult<()> {
        let docs_refs: Vec<Vec<&str>> = documents
            .iter()
            .map(|d| d.iter().map(|w| w.as_str()).collect())
            .collect();
        let docs_slices: Vec<&[&str]> = docs_refs.iter().map(|d| d.as_slice()).collect();
        self.inner.fit(&docs_slices).map_err(py_err)
    }

    /// Return the top `n` words for a given topic, with their weights.
    fn top_words(&self, topic: usize, n: usize) -> PyResult<Vec<(String, f64)>> {
        self.inner.top_words(topic, n).map_err(py_err)
    }

    /// Return the full word probability distribution for a topic.
    fn topic_word_distribution(&self, topic: usize) -> PyResult<Vec<(String, f64)>> {
        self.inner.topic_word_distribution(topic).map_err(py_err)
    }

    /// Return the topic distribution for a document by its index.
    fn document_topic_distribution(&self, doc_idx: usize) -> PyResult<Vec<f64>> {
        self.inner
            .document_topic_distribution(doc_idx)
            .map_err(py_err)
    }

    /// Return the number of topics in the model.
    fn n_topics(&self) -> usize {
        self.inner.n_topics()
    }

    fn __repr__(&self) -> String {
        format!("LDA(n_topics={})", self.inner.n_topics())
    }
}

// ---------------------------------------------------------------------------
// Register submodule
// ---------------------------------------------------------------------------

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "nlp")?;

    // Tokenizers
    m.add_function(wrap_pyfunction!(word_tokenize, &m)?)?;
    m.add_function(wrap_pyfunction!(char_tokenize, &m)?)?;
    m.add_function(wrap_pyfunction!(ngram_tokenize, &m)?)?;
    m.add_function(wrap_pyfunction!(whitespace_tokenize, &m)?)?;
    m.add_class::<PyWordPieceTokenizer>()?;
    m.add_class::<PyUnigramTokenizer>()?;

    // Stemming
    m.add_class::<PyPorterStemmer>()?;

    // Stopwords
    m.add_function(wrap_pyfunction!(stopwords, &m)?)?;
    m.add_function(wrap_pyfunction!(is_stopword, &m)?)?;
    m.add_function(wrap_pyfunction!(remove_stopwords, &m)?)?;

    // Text utilities
    m.add_function(wrap_pyfunction!(ngrams, &m)?)?;
    m.add_function(wrap_pyfunction!(edit_distance, &m)?)?;
    m.add_function(wrap_pyfunction!(normalize, &m)?)?;
    m.add_function(wrap_pyfunction!(pad_sequences, &m)?)?;

    // Similarity
    m.add_function(wrap_pyfunction!(cosine_sim, &m)?)?;
    m.add_function(wrap_pyfunction!(jaccard, &m)?)?;
    m.add_function(wrap_pyfunction!(edit_distance_norm, &m)?)?;

    // Vectorizers
    m.add_class::<PyCountVectorizer>()?;
    m.add_class::<PyTfidfVectorizer>()?;

    // Word2Vec & Embeddings
    m.add_class::<PyWord2Vec>()?;
    m.add_class::<PyWordEmbeddings>()?;

    // Sentiment
    m.add_class::<PySentimentAnalyzer>()?;

    // POS tagging
    m.add_class::<PyHmmPosTagger>()?;

    // NER
    m.add_class::<PyRuleBasedNer>()?;

    // LDA
    m.add_class::<PyLda>()?;

    parent.add_submodule(&m)?;
    Ok(())
}
