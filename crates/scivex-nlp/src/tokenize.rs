//! Tokenizers for splitting text into tokens.

/// Trait for all tokenizer implementations.
pub trait Tokenizer {
    /// Split `text` into a sequence of token strings.
    fn tokenize<'a>(&self, text: &'a str) -> Vec<&'a str>;
}

// ---------------------------------------------------------------------------
// WhitespaceTokenizer
// ---------------------------------------------------------------------------

/// Splits text on whitespace boundaries.
///
/// # Examples
///
/// ```
/// # use scivex_nlp::tokenize::{Tokenizer, WhitespaceTokenizer};
/// let tok = WhitespaceTokenizer;
/// let tokens = tok.tokenize("hello world foo");
/// assert_eq!(tokens, vec!["hello", "world", "foo"]);
/// ```
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct WhitespaceTokenizer;

impl Tokenizer for WhitespaceTokenizer {
    fn tokenize<'a>(&self, text: &'a str) -> Vec<&'a str> {
        text.split_whitespace().collect()
    }
}

// ---------------------------------------------------------------------------
// WordTokenizer
// ---------------------------------------------------------------------------

#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
/// Splits text on non-alphanumeric boundaries, producing word tokens.
///
/// # Examples
///
/// ```
/// # use scivex_nlp::tokenize::{Tokenizer, WordTokenizer};
/// let tok = WordTokenizer::new();
/// let tokens = tok.tokenize("hello world");
/// assert_eq!(tokens, vec!["hello", "world"]);
/// ```
pub struct WordTokenizer {
    pub to_lowercase: bool,
}

impl WordTokenizer {
    /// Create a new word tokenizer (case-preserving by default).
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_nlp::tokenize::WordTokenizer;
    /// let tok = WordTokenizer::new().with_lowercase(true);
    /// let tokens = tok.tokenize_owned("Hello, World!");
    /// assert_eq!(tokens, vec!["hello", "world"]);
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self {
            to_lowercase: false,
        }
    }

    /// Enable or disable lowercase conversion for owned tokenization.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_nlp::tokenize::WordTokenizer;
    /// let tok = WordTokenizer::new().with_lowercase(true);
    /// let tokens = tok.tokenize_owned("Hello World");
    /// assert_eq!(tokens, vec!["hello", "world"]);
    /// ```
    #[must_use]
    pub fn with_lowercase(mut self, yes: bool) -> Self {
        self.to_lowercase = yes;
        self
    }
}

impl Default for WordTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl WordTokenizer {
    /// Tokenize text into word tokens. When `to_lowercase` is true, returns
    /// owned `String`s (lowered). Otherwise returns borrowed slices.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_nlp::tokenize::WordTokenizer;
    /// let tok = WordTokenizer::new();
    /// let tokens = tok.tokenize_owned("Hello, World!");
    /// assert_eq!(tokens, vec!["Hello", "World"]);
    /// ```
    pub fn tokenize_owned(&self, text: &str) -> Vec<String> {
        let tokens = Self::tokenize_borrowed(text);
        if self.to_lowercase {
            tokens.into_iter().map(str::to_lowercase).collect()
        } else {
            tokens.into_iter().map(String::from).collect()
        }
    }

    fn tokenize_borrowed(text: &str) -> Vec<&str> {
        let mut tokens = Vec::new();
        let mut start = None;
        for (i, c) in text.char_indices() {
            if c.is_alphanumeric() {
                if start.is_none() {
                    start = Some(i);
                }
            } else if let Some(s) = start {
                tokens.push(&text[s..i]);
                start = None;
            }
        }
        if let Some(s) = start {
            tokens.push(&text[s..]);
        }
        tokens
    }
}

impl Tokenizer for WordTokenizer {
    fn tokenize<'a>(&self, text: &'a str) -> Vec<&'a str> {
        Self::tokenize_borrowed(text)
    }
}

// ---------------------------------------------------------------------------
// CharTokenizer
// ---------------------------------------------------------------------------

/// Splits text into individual character tokens.
///
/// # Examples
///
/// ```
/// # use scivex_nlp::tokenize::{Tokenizer, CharTokenizer};
/// let tok = CharTokenizer;
/// let tokens = tok.tokenize("abc");
/// assert_eq!(tokens, vec!["a", "b", "c"]);
/// ```
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct CharTokenizer;

impl Tokenizer for CharTokenizer {
    fn tokenize<'a>(&self, text: &'a str) -> Vec<&'a str> {
        text.char_indices()
            .map(|(i, c)| &text[i..i + c.len_utf8()])
            .collect()
    }
}

// ---------------------------------------------------------------------------
// NGramTokenizer
// ---------------------------------------------------------------------------

/// Produces character-level n-grams from text.
///
/// # Examples
///
/// ```
/// # use scivex_nlp::tokenize::{Tokenizer, NGramTokenizer};
/// let tok = NGramTokenizer::new(2);
/// let grams = tok.tokenize("abc");
/// assert_eq!(grams, vec!["ab", "bc"]);
/// ```
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct NGramTokenizer {
    pub n: usize,
}

impl NGramTokenizer {
    /// Create an n-gram tokenizer with the given gram size.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_nlp::tokenize::NGramTokenizer;
    /// let tok = NGramTokenizer::new(3);
    /// assert_eq!(tok.n, 3);
    /// ```
    #[must_use]
    pub fn new(n: usize) -> Self {
        Self { n }
    }
}

impl Tokenizer for NGramTokenizer {
    fn tokenize<'a>(&self, text: &'a str) -> Vec<&'a str> {
        if self.n == 0 || text.is_empty() {
            return Vec::new();
        }
        let chars: Vec<(usize, char)> = text.char_indices().collect();
        if chars.len() < self.n {
            return Vec::new();
        }
        (0..=chars.len() - self.n)
            .map(|i| {
                let start = chars[i].0;
                let end = if i + self.n < chars.len() {
                    chars[i + self.n].0
                } else {
                    text.len()
                };
                &text[start..end]
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn whitespace_basic() {
        let tok = WhitespaceTokenizer;
        assert_eq!(tok.tokenize("hello world"), vec!["hello", "world"]);
    }

    #[test]
    fn whitespace_multiple_spaces() {
        let tok = WhitespaceTokenizer;
        assert_eq!(tok.tokenize("  a  b  c  "), vec!["a", "b", "c"]);
    }

    #[test]
    fn word_tokenizer_punctuation() {
        let tok = WordTokenizer::new();
        assert_eq!(
            tok.tokenize("Hello, world! How are you?"),
            vec!["Hello", "world", "How", "are", "you"]
        );
    }

    #[test]
    fn word_tokenizer_lowercase() {
        let tok = WordTokenizer::new().with_lowercase(true);
        let result = tok.tokenize_owned("Hello World");
        assert_eq!(result, vec!["hello", "world"]);
    }

    #[test]
    fn char_tokenizer() {
        let tok = CharTokenizer;
        assert_eq!(tok.tokenize("abc"), vec!["a", "b", "c"]);
    }

    #[test]
    fn ngram_tokenizer() {
        let tok = NGramTokenizer::new(2);
        assert_eq!(tok.tokenize("abcd"), vec!["ab", "bc", "cd"]);
    }

    #[test]
    fn ngram_too_short() {
        let tok = NGramTokenizer::new(5);
        assert!(tok.tokenize("abc").is_empty());
    }

    #[test]
    fn empty_input() {
        let ws = WhitespaceTokenizer;
        assert!(ws.tokenize("").is_empty());
        let wt = WordTokenizer::new();
        assert!(wt.tokenize("").is_empty());
    }
}
