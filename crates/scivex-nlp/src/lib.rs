//! `scivex-nlp` — Tokenization, embeddings, and text processing.
//!
//! Provides from-scratch implementations of:
//! - Tokenizers: whitespace, word, character, n-gram
//! - Text utilities: stopwords, n-grams, edit distance, normalization
//! - Porter stemmer
//! - Count and TF-IDF vectorization
//! - Word embeddings with similarity search and analogy solving
//! - Similarity measures: cosine, Jaccard, normalized edit distance
//! - Lexicon-based sentiment analysis

/// Word embeddings with similarity search and analogy solving.
pub mod embeddings;
/// NLP error types.
pub mod error;
/// Lexicon-based sentiment analysis.
pub mod sentiment;
/// Similarity measures (cosine, Jaccard, normalized edit distance).
pub mod similarity;
/// Porter stemmer.
pub mod stem;
/// Text utilities: stopwords, n-grams, edit distance, normalization.
pub mod text;
/// Tokenizers (whitespace, word, character, n-gram).
pub mod tokenize;
/// Count and TF-IDF vectorization.
pub mod vectorize;

pub use embeddings::WordEmbeddings;
pub use error::{NlpError, Result};
pub use sentiment::{SentimentAnalyzer, SentimentResult};
pub use similarity::{cosine_similarity, edit_distance_normalized, jaccard_similarity};
pub use stem::PorterStemmer;
pub use tokenize::{CharTokenizer, NGramTokenizer, Tokenizer, WhitespaceTokenizer, WordTokenizer};
pub use vectorize::{CountVectorizer, TfidfVectorizer};

/// Items intended for glob-import: `use scivex_nlp::prelude::*;`
pub mod prelude {
    pub use crate::embeddings::WordEmbeddings;
    pub use crate::error::{NlpError, Result};
    pub use crate::sentiment::{SentimentAnalyzer, SentimentResult};
    pub use crate::similarity::{cosine_similarity, edit_distance_normalized, jaccard_similarity};
    pub use crate::stem::PorterStemmer;
    pub use crate::tokenize::{
        CharTokenizer, NGramTokenizer, Tokenizer, WhitespaceTokenizer, WordTokenizer,
    };
    pub use crate::vectorize::{CountVectorizer, TfidfVectorizer};
}
