//! BERT-style WordPiece subword tokenizer.
//!
//! Implements greedy longest-match-first subword segmentation. Words that
//! cannot be segmented into known vocabulary pieces are replaced with the
//! unknown token (`[UNK]`).

use std::collections::HashMap;

use crate::error::{NlpError, Result};
use crate::tokenize::Tokenizer;

// ---------------------------------------------------------------------------
// Special token constants
// ---------------------------------------------------------------------------

/// Classification token.
pub const CLS: &str = "[CLS]";
/// Separator token.
pub const SEP: &str = "[SEP]";
/// Padding token.
pub const PAD: &str = "[PAD]";
/// Unknown token.
pub const UNK: &str = "[UNK]";

// ---------------------------------------------------------------------------
// WordPieceTokenizer
// ---------------------------------------------------------------------------

/// A BERT-style WordPiece subword tokenizer.
///
/// Splits text into subword tokens using a greedy longest-match-first
/// algorithm. Words that exceed `max_word_len` characters or cannot be
/// decomposed into known vocabulary pieces are emitted as `[UNK]`.
///
/// # Examples
///
/// ```
/// # use scivex_nlp::WordPieceTokenizer;
/// let tok = WordPieceTokenizer::from_vocab_list(&["[UNK]", "hello", "world"]).unwrap();
/// let pieces = tok.tokenize_to_pieces("hello world");
/// assert_eq!(pieces, vec!["hello", "world"]);
/// ```
pub struct WordPieceTokenizer {
    /// Maps subword tokens to their integer IDs.
    pub vocab: HashMap<String, usize>,
    /// Reverse mapping from IDs back to subword tokens.
    id_to_token: HashMap<usize, String>,
    /// The token emitted for unknown words (default `"[UNK]"`).
    pub unk_token: String,
    /// Maximum character length of a word before it is treated as unknown
    /// (default 100).
    pub max_word_len: usize,
    /// Prefix prepended to continuing subword pieces (default `"##"`).
    pub continuing_prefix: String,
}

impl WordPieceTokenizer {
    // -- constructors -------------------------------------------------------

    /// Create a `WordPieceTokenizer` from an existing vocabulary map.
    ///
    /// # Errors
    ///
    /// Returns [`NlpError::EmptyVocabulary`] if `vocab` is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_nlp::wordpiece::WordPieceTokenizer;
    /// # use std::collections::HashMap;
    /// let mut vocab = HashMap::new();
    /// vocab.insert("[UNK]".to_string(), 0);
    /// vocab.insert("hello".to_string(), 1);
    /// let tok = WordPieceTokenizer::from_vocab(vocab).unwrap();
    /// let pieces = tok.tokenize_to_pieces("hello");
    /// assert_eq!(pieces, vec!["hello"]);
    /// ```
    pub fn from_vocab(vocab: HashMap<String, usize>) -> Result<Self> {
        if vocab.is_empty() {
            return Err(NlpError::EmptyVocabulary);
        }
        let id_to_token: HashMap<usize, String> =
            vocab.iter().map(|(tok, &id)| (id, tok.clone())).collect();
        Ok(Self {
            vocab,
            id_to_token,
            unk_token: UNK.to_owned(),
            max_word_len: 100,
            continuing_prefix: "##".to_owned(),
        })
    }

    /// Build a `WordPieceTokenizer` from a slice of token strings.
    ///
    /// IDs are assigned sequentially starting from 0.
    ///
    /// # Errors
    ///
    /// Returns [`NlpError::EmptyVocabulary`] if `words` is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_nlp::wordpiece::WordPieceTokenizer;
    /// let tok = WordPieceTokenizer::from_vocab_list(&["[UNK]", "play", "##ing"]).unwrap();
    /// let pieces = tok.tokenize_to_pieces("playing");
    /// assert_eq!(pieces, vec!["play", "##ing"]);
    /// ```
    pub fn from_vocab_list(words: &[&str]) -> Result<Self> {
        if words.is_empty() {
            return Err(NlpError::EmptyVocabulary);
        }
        let vocab: HashMap<String, usize> = words
            .iter()
            .enumerate()
            .map(|(i, &w)| (w.to_owned(), i))
            .collect();
        Self::from_vocab(vocab)
    }

    // -- tokenization -------------------------------------------------------

    /// Tokenize `text` into WordPiece subword strings.
    ///
    /// 1. Pre-tokenize on whitespace and lowercase the input.
    /// 2. For each word, apply greedy longest-match-first segmentation:
    ///    - Try the longest prefix present in the vocabulary.
    ///    - If found, consume it and continue with `"##" + remainder`.
    ///    - If no prefix matches at any point, emit `[UNK]` for the whole word.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_nlp::wordpiece::WordPieceTokenizer;
    /// let tok = WordPieceTokenizer::from_vocab_list(
    ///     &["[UNK]", "un", "##known", "hello"]
    /// ).unwrap();
    /// let pieces = tok.tokenize_to_pieces("unknown hello");
    /// assert_eq!(pieces, vec!["un", "##known", "hello"]);
    /// ```
    #[must_use]
    pub fn tokenize_to_pieces(&self, text: &str) -> Vec<String> {
        let lowered = text.to_lowercase();
        let words: Vec<&str> = lowered.split_whitespace().collect();
        let mut output = Vec::new();

        for word in words {
            if word.chars().count() > self.max_word_len {
                output.push(self.unk_token.clone());
                continue;
            }
            if !self.segment_word(word, &mut output) {
                // Could not segment — roll back any partial pieces we added
                // and emit [UNK] instead.
                output.push(self.unk_token.clone());
            }
        }

        output
    }

    /// Tokenize `text` into vocabulary IDs.
    ///
    /// # Errors
    ///
    /// Returns [`NlpError::UnknownToken`] if the unknown token itself is not
    /// present in the vocabulary (so we cannot represent unknown words).
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_nlp::wordpiece::WordPieceTokenizer;
    /// let tok = WordPieceTokenizer::from_vocab_list(&["[UNK]", "hello"]).unwrap();
    /// let ids = tok.encode("hello").unwrap();
    /// assert_eq!(ids, vec![1]);
    /// ```
    pub fn encode(&self, text: &str) -> Result<Vec<usize>> {
        let pieces = self.tokenize_to_pieces(text);
        let mut ids = Vec::with_capacity(pieces.len());
        for piece in &pieces {
            if let Some(&id) = self.vocab.get(piece) {
                ids.push(id);
            } else {
                // The piece is [UNK] (or theoretically something else not in
                // the vocab). Look up the unk token ID.
                let unk_id =
                    self.vocab
                        .get(&self.unk_token)
                        .ok_or_else(|| NlpError::UnknownToken {
                            token: piece.clone(),
                        })?;
                ids.push(*unk_id);
            }
        }
        Ok(ids)
    }

    /// Decode a sequence of IDs back into a string.
    ///
    /// Continuing subword pieces (those starting with `##`) are concatenated
    /// directly to the previous token. Unknown IDs are rendered as `[UNK]`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_nlp::wordpiece::WordPieceTokenizer;
    /// let tok = WordPieceTokenizer::from_vocab_list(
    ///     &["[UNK]", "play", "##ing"]
    /// ).unwrap();
    /// let decoded = tok.decode(&[1, 2]);
    /// assert_eq!(decoded, "playing");
    /// ```
    #[must_use]
    pub fn decode(&self, ids: &[usize]) -> String {
        let mut parts: Vec<String> = Vec::with_capacity(ids.len());
        for &id in ids {
            let token = self
                .id_to_token
                .get(&id)
                .cloned()
                .unwrap_or_else(|| self.unk_token.clone());

            if let Some(stripped) = token.strip_prefix(self.continuing_prefix.as_str()) {
                // Append directly to the last part (no space).
                if let Some(last) = parts.last_mut() {
                    last.push_str(stripped);
                } else {
                    parts.push(stripped.to_owned());
                }
            } else {
                parts.push(token);
            }
        }
        parts.join(" ")
    }

    // -- helpers ------------------------------------------------------------

    /// Try to segment `word` into vocab pieces using greedy longest-match.
    ///
    /// Pushes pieces into `output`. Returns `true` on success, `false` if the
    /// word cannot be fully segmented (in which case nothing is pushed).
    fn segment_word(&self, word: &str, output: &mut Vec<String>) -> bool {
        let start_len = output.len();
        let mut remaining = word;
        let mut is_first = true;

        while !remaining.is_empty() {
            let mut matched = false;
            let char_count = remaining.chars().count();

            // Try progressively shorter prefixes.
            for end in (1..=char_count).rev() {
                let prefix: String = remaining.chars().take(end).collect();
                let candidate = if is_first {
                    prefix.clone()
                } else {
                    format!("{}{}", self.continuing_prefix, prefix)
                };

                if self.vocab.contains_key(&candidate) {
                    output.push(candidate);
                    // Advance past the matched characters.
                    let byte_len: usize = remaining.chars().take(end).map(char::len_utf8).sum();
                    remaining = &remaining[byte_len..];
                    is_first = false;
                    matched = true;
                    break;
                }
            }

            if !matched {
                // Roll back any pieces we already pushed for this word.
                output.truncate(start_len);
                return false;
            }
        }

        true
    }
}

// ---------------------------------------------------------------------------
// Tokenizer trait implementation
// ---------------------------------------------------------------------------

impl Tokenizer for WordPieceTokenizer {
    /// Pre-tokenize `text` by splitting on whitespace.
    ///
    /// This returns borrowed slices from the original input. For full
    /// WordPiece subword segmentation use [`WordPieceTokenizer::tokenize_to_pieces`].
    fn tokenize<'a>(&self, text: &'a str) -> Vec<&'a str> {
        text.split_whitespace().collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a tokenizer with a small test vocabulary.
    fn test_tokenizer() -> WordPieceTokenizer {
        let words: &[&str] = &[
            "[UNK]", "[CLS]", "[SEP]", "[PAD]", "hello", "world", "un", "##known", "##able",
            "play", "##ing", "##s",
        ];
        WordPieceTokenizer::from_vocab_list(words).unwrap()
    }

    #[test]
    fn basic_word_splitting() {
        let tok = test_tokenizer();
        let pieces = tok.tokenize_to_pieces("Hello World");
        assert_eq!(pieces, vec!["hello", "world"]);
    }

    #[test]
    fn unknown_word() {
        let tok = test_tokenizer();
        let pieces = tok.tokenize_to_pieces("xyzzy");
        assert_eq!(pieces, vec!["[UNK]"]);
    }

    #[test]
    fn continuing_subwords() {
        let tok = test_tokenizer();
        // "unknown" → "un" + "##known"
        let pieces = tok.tokenize_to_pieces("unknown");
        assert_eq!(pieces, vec!["un", "##known"]);

        // "playing" → "play" + "##ing"
        let pieces = tok.tokenize_to_pieces("playing");
        assert_eq!(pieces, vec!["play", "##ing"]);
    }

    #[test]
    fn encode_decode_roundtrip() {
        let tok = test_tokenizer();
        let ids = tok.encode("playing unknown").unwrap();
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, "playing unknown");
    }

    #[test]
    fn empty_input() {
        let tok = test_tokenizer();
        let pieces = tok.tokenize_to_pieces("");
        assert!(pieces.is_empty());

        let ids = tok.encode("").unwrap();
        assert!(ids.is_empty());

        let decoded = tok.decode(&[]);
        assert_eq!(decoded, "");
    }
}
