//! Unigram language model tokenizer (Kudo, 2018).
//!
//! Implements sub-word tokenization using a unigram language model.
//! Segmentation is performed via Viterbi decoding to find the
//! maximum-likelihood tokenization under the learned piece probabilities.

use std::collections::HashMap;

use crate::error::{NlpError, Result};

// ---------------------------------------------------------------------------
// UnigramTokenizer
// ---------------------------------------------------------------------------

/// A SentencePiece-style unigram tokenizer.
///
/// The vocabulary is a set of string pieces, each associated with a
/// log-probability. Tokenization finds the segmentation that maximises the
/// total log-probability using the Viterbi algorithm.
///
/// # Examples
///
/// ```
/// # use scivex_nlp::UnigramTokenizer;
/// let pieces = vec![("<unk>".into(), -10.0), ("h".into(), -1.0), ("he".into(), -0.5)];
/// let tok = UnigramTokenizer::new(pieces, 0).unwrap();
/// assert_eq!(tok.vocab_size(), 3);
/// ```
#[derive(Debug, Clone)]
pub struct UnigramTokenizer {
    /// Vocabulary of (piece, log_probability) pairs, indexed by piece id.
    pieces: Vec<(String, f64)>,
    /// Reverse lookup: piece string -> piece id.
    piece_to_id: HashMap<String, usize>,
    /// Index of the unknown / fallback token.
    unk_id: usize,
}

impl UnigramTokenizer {
    /// Create a tokenizer from an explicit piece list.
    ///
    /// # Errors
    ///
    /// Returns [`NlpError::EmptyVocabulary`] if `pieces` is empty.
    /// Returns [`NlpError::InvalidParameter`] if `unk_id` is out of range.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_nlp::unigram::UnigramTokenizer;
    /// let pieces = vec![("<unk>".into(), -10.0), ("a".into(), -1.0), ("ab".into(), -0.5)];
    /// let tok = UnigramTokenizer::new(pieces, 0).unwrap();
    /// assert_eq!(tok.vocab_size(), 3);
    /// ```
    pub fn new(pieces: Vec<(String, f64)>, unk_id: usize) -> Result<Self> {
        if pieces.is_empty() {
            return Err(NlpError::EmptyVocabulary);
        }
        if unk_id >= pieces.len() {
            return Err(NlpError::InvalidParameter {
                name: "unk_id",
                reason: "unk_id is out of range of the piece list",
            });
        }
        let piece_to_id: HashMap<String, usize> = pieces
            .iter()
            .enumerate()
            .map(|(i, (s, _))| (s.clone(), i))
            .collect();
        Ok(Self {
            pieces,
            piece_to_id,
            unk_id,
        })
    }

    /// Convenience constructor from parallel slices of words and scores.
    ///
    /// # Errors
    ///
    /// Returns an error if `words` and `scores` differ in length, either is
    /// empty, or `unk_id` is out of range.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_nlp::unigram::UnigramTokenizer;
    /// let tok = UnigramTokenizer::from_vocab(
    ///     &["<unk>", "a", "b", "ab"],
    ///     &[-10.0, -1.0, -1.0, -0.5],
    ///     0,
    /// ).unwrap();
    /// let tokens = tok.tokenize_str("ab");
    /// assert_eq!(tokens, vec!["ab"]); // Viterbi picks highest-score segmentation
    /// ```
    pub fn from_vocab(words: &[&str], scores: &[f64], unk_id: usize) -> Result<Self> {
        if words.len() != scores.len() {
            return Err(NlpError::InvalidParameter {
                name: "scores",
                reason: "words and scores must have the same length",
            });
        }
        let pieces: Vec<(String, f64)> = words
            .iter()
            .zip(scores.iter())
            .map(|(w, &s)| ((*w).to_string(), s))
            .collect();
        Self::new(pieces, unk_id)
    }

    /// Return the vocabulary size (number of pieces including `<unk>`).
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.pieces.len()
    }

    // -----------------------------------------------------------------------
    // Viterbi segmentation
    // -----------------------------------------------------------------------

    /// Tokenize `text` into sub-word pieces using Viterbi decoding.
    ///
    /// Returns the segmentation that maximises the sum of log-probabilities.
    /// Characters that cannot be covered by any piece are replaced with the
    /// unknown token.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_nlp::unigram::UnigramTokenizer;
    /// let tok = UnigramTokenizer::from_vocab(
    ///     &["<unk>", "a", "b", "ab", "c", "abc"],
    ///     &[-10.0, -1.0, -1.0, -0.5, -1.0, -0.3],
    ///     0,
    /// ).unwrap();
    /// let tokens = tok.tokenize_str("abc");
    /// assert_eq!(tokens, vec!["abc"]); // best single-piece segmentation
    /// ```
    #[must_use]
    pub fn tokenize_str(&self, text: &str) -> Vec<String> {
        if text.is_empty() {
            return Vec::new();
        }

        // We work on byte positions that are aligned to UTF-8 char boundaries.
        let char_boundaries: Vec<usize> = text
            .char_indices()
            .map(|(i, _)| i)
            .chain(std::iter::once(text.len()))
            .collect();
        let n = char_boundaries.len(); // number of boundary points (chars + 1)

        // best_score[i] = best log-prob for text[..char_boundaries[i]]
        // back[i] = (start boundary index, piece_id) for the best piece ending at i
        let mut best_score: Vec<f64> = vec![f64::NEG_INFINITY; n];
        let mut back: Vec<(usize, usize)> = vec![(0, 0); n];
        best_score[0] = 0.0;

        for j in 1..n {
            let end_byte = char_boundaries[j];
            for i in 0..j {
                let start_byte = char_boundaries[i];
                let substr = &text[start_byte..end_byte];

                if let Some(&pid) = self.piece_to_id.get(substr) {
                    let score = best_score[i] + self.pieces[pid].1;
                    if score > best_score[j] {
                        best_score[j] = score;
                        back[j] = (i, pid);
                    }
                }
            }

            // If no piece could reach position j from any earlier position,
            // treat the single character ending at j as unknown.
            if best_score[j] == f64::NEG_INFINITY {
                // Advance by one character (from j-1 to j).
                let prev_score = best_score[j - 1];
                let fallback = if prev_score == f64::NEG_INFINITY {
                    0.0
                } else {
                    prev_score
                };
                best_score[j] = fallback + self.pieces[self.unk_id].1;
                back[j] = (j - 1, self.unk_id);
            }
        }

        // Back-track to recover the best segmentation.
        let mut segments: Vec<String> = Vec::new();
        let mut pos = n - 1;
        while pos > 0 {
            let (start, pid) = back[pos];
            if pid == self.unk_id {
                segments.push(self.pieces[self.unk_id].0.clone());
            } else {
                segments.push(self.pieces[pid].0.clone());
            }
            pos = start;
        }
        segments.reverse();
        segments
    }

    /// Tokenize `text` and return piece IDs.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_nlp::unigram::UnigramTokenizer;
    /// let tok = UnigramTokenizer::from_vocab(
    ///     &["<unk>", "a", "b", "ab"],
    ///     &[-10.0, -1.0, -1.0, -0.5],
    ///     0,
    /// ).unwrap();
    /// let ids = tok.encode("ab");
    /// assert_eq!(ids, vec![3]); // "ab" is piece 3
    /// ```
    #[must_use]
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let tokens = self.tokenize_str(text);
        tokens
            .iter()
            .map(|t| {
                self.piece_to_id
                    .get(t.as_str())
                    .copied()
                    .unwrap_or(self.unk_id)
            })
            .collect()
    }

    /// Decode a sequence of piece IDs back to text.
    ///
    /// # Errors
    ///
    /// Returns [`NlpError::InvalidParameter`] if any id is out of range.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_nlp::unigram::UnigramTokenizer;
    /// let tok = UnigramTokenizer::from_vocab(
    ///     &["<unk>", "a", "b"],
    ///     &[-10.0, -1.0, -1.0],
    ///     0,
    /// ).unwrap();
    /// let text = tok.decode(&[1, 2]).unwrap();
    /// assert_eq!(text, "ab");
    /// ```
    pub fn decode(&self, ids: &[usize]) -> Result<String> {
        let mut out = String::new();
        for &id in ids {
            if id >= self.pieces.len() {
                return Err(NlpError::InvalidParameter {
                    name: "ids",
                    reason: "piece id out of range",
                });
            }
            out.push_str(&self.pieces[id].0);
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_tokenizer() -> UnigramTokenizer {
        // Vocabulary:  0=<unk>, 1=a, 2=b, 3=ab, 4=c, 5=abc
        let words = &["<unk>", "a", "b", "ab", "c", "abc"];
        let scores = &[-10.0, -1.0, -1.0, -0.5, -1.0, -0.3];
        UnigramTokenizer::from_vocab(words, scores, 0).unwrap()
    }

    #[test]
    fn basic_segmentation() {
        let tok = sample_tokenizer();
        // "abc" should be tokenised as the single piece "abc" (score -0.3)
        // rather than "a"+"b"+"c" (score -3.0) or "ab"+"c" (score -1.5).
        let tokens = tok.tokenize_str("abc");
        assert_eq!(tokens, vec!["abc"]);
    }

    #[test]
    fn viterbi_picks_best_segmentation() {
        let tok = sample_tokenizer();
        // "ab" can be segmented as "ab" (score -0.5) or "a"+"b" (-2.0).
        // Viterbi should choose "ab".
        let tokens = tok.tokenize_str("ab");
        assert_eq!(tokens, vec!["ab"]);

        // "abab" -> "ab" + "ab" (score -1.0) beats "a"+"b"+"a"+"b" (-4.0).
        let tokens = tok.tokenize_str("abab");
        assert_eq!(tokens, vec!["ab", "ab"]);
    }

    #[test]
    fn unknown_character_handling() {
        let tok = sample_tokenizer();
        // 'z' is not in the vocabulary, so it should produce the unk token.
        let tokens = tok.tokenize_str("z");
        assert_eq!(tokens, vec!["<unk>"]);

        // "azb" -> "a" + <unk> + "b"
        let tokens = tok.tokenize_str("azb");
        assert_eq!(tokens, vec!["a", "<unk>", "b"]);
    }

    #[test]
    fn encode_decode_roundtrip() {
        let tok = sample_tokenizer();
        let text = "abc";
        let ids = tok.encode(text);
        assert_eq!(ids, vec![5]); // "abc" is piece 5
        let decoded = tok.decode(&ids).unwrap();
        assert_eq!(decoded, "abc");

        // Multi-piece roundtrip.
        let ids2 = tok.encode("abab");
        assert_eq!(ids2, vec![3, 3]); // "ab" is piece 3
        let decoded2 = tok.decode(&ids2).unwrap();
        assert_eq!(decoded2, "abab");
    }
}
