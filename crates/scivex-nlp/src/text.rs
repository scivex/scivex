//! Text utilities: stopwords, n-grams, edit distance, normalization.

/// Returns a slice of common English stopwords.
#[must_use]
pub fn stopwords() -> &'static [&'static str] {
    &STOPWORDS
}

/// Returns `true` if `word` (lowercased) is an English stopword.
///
/// # Examples
///
/// ```
/// # use scivex_nlp::text::is_stopword;
/// assert!(is_stopword("the"));
/// assert!(!is_stopword("science"));
/// ```
#[must_use]
pub fn is_stopword(word: &str) -> bool {
    let lower = word.to_lowercase();
    STOPWORDS.contains(&lower.as_str())
}

/// Filter out stopwords from a token list.
#[must_use]
pub fn remove_stopwords<'a>(tokens: &[&'a str]) -> Vec<&'a str> {
    tokens
        .iter()
        .copied()
        .filter(|t| {
            let lower = t.to_lowercase();
            !STOPWORDS.contains(&lower.as_str())
        })
        .collect()
}

/// Produce word-level n-grams from a token list.
///
/// Returns a vector of n-gram groups, each group being a `Vec` of `n` tokens.
///
/// # Examples
///
/// ```
/// # use scivex_nlp::text::ngrams;
/// let tokens = vec!["the", "cat", "sat"];
/// let bigrams = ngrams(&tokens, 2);
/// assert_eq!(bigrams, vec![vec!["the", "cat"], vec!["cat", "sat"]]);
/// ```
#[must_use]
pub fn ngrams<'a>(tokens: &[&'a str], n: usize) -> Vec<Vec<&'a str>> {
    if n == 0 || tokens.len() < n {
        return Vec::new();
    }
    (0..=tokens.len() - n)
        .map(|i| tokens[i..i + n].to_vec())
        .collect()
}

/// Compute the Levenshtein edit distance between two strings
/// using the Wagner-Fischer algorithm.
///
/// # Examples
///
/// ```
/// # use scivex_nlp::text::levenshtein;
/// assert_eq!(levenshtein("kitten", "sitting"), 3);
/// assert_eq!(levenshtein("", "abc"), 3);
/// ```
#[must_use]
pub fn levenshtein(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let m = a_chars.len();
    let n = b_chars.len();

    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    // Use two rows instead of full matrix.
    let mut prev = (0..=n).collect::<Vec<_>>();
    let mut curr = vec![0; n + 1];

    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = usize::from(a_chars[i - 1] != b_chars[j - 1]);
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[n]
}

/// Lowercase and strip non-alphanumeric characters (keep spaces).
///
/// # Examples
///
/// ```
/// # use scivex_nlp::text::normalize;
/// assert_eq!(normalize("Hello, World!"), "hello world");
/// ```
#[must_use]
pub fn normalize(text: &str) -> String {
    text.chars()
        .filter_map(|c| {
            if c.is_alphanumeric() {
                Some(c.to_lowercase().next().unwrap_or(c))
            } else if c.is_whitespace() {
                Some(' ')
            } else {
                None
            }
        })
        .collect()
}

/// Pad or truncate sequences to a uniform length.
///
/// Sequences shorter than `max_len` are right-padded with `pad_value`.
/// Sequences longer than `max_len` are truncated.
///
/// # Examples
///
/// ```
/// # use scivex_nlp::text::pad_sequences;
/// let seqs = vec![vec![1, 2], vec![3, 4, 5, 6]];
/// let padded = pad_sequences(&seqs, 3, 0);
/// assert_eq!(padded[0], vec![1, 2, 0]);
/// assert_eq!(padded[1], vec![3, 4, 5]);
/// ```
#[must_use]
pub fn pad_sequences(
    sequences: &[Vec<usize>],
    max_len: usize,
    pad_value: usize,
) -> Vec<Vec<usize>> {
    sequences
        .iter()
        .map(|seq| {
            if seq.len() >= max_len {
                seq[..max_len].to_vec()
            } else {
                let mut padded = seq.clone();
                padded.resize(max_len, pad_value);
                padded
            }
        })
        .collect()
}

// ~175 common English stopwords
#[rustfmt::skip]
static STOPWORDS: [&str; 183] = [
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "aren't", "as", "at", "be", "because", "been",
    "before", "being", "below", "between", "both", "but", "by", "can",
    "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does",
    "doesn't", "doing", "don't", "down", "during", "each", "few", "for",
    "from", "further", "get", "got", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "her", "here", "hers", "herself",
    "him", "himself", "his", "how", "i", "if", "in", "into", "is", "isn't",
    "it", "it's", "its", "itself", "just", "let's", "me", "might", "more",
    "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off",
    "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves",
    "out", "over", "own", "same", "shan't", "she", "should", "shouldn't",
    "so", "some", "such", "than", "that", "the", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "wasn't",
    "we", "were", "weren't", "what", "when", "where", "which", "while",
    "who", "whom", "why", "will", "with", "won't", "would", "wouldn't",
    "you", "your", "yours", "yourself", "yourselves",
    "also", "back", "even", "first", "go", "going", "good", "great",
    "know", "like", "look", "make", "much", "new", "now", "old", "one",
    "people", "really", "right", "say", "see", "still", "take", "tell",
    "think", "time", "two", "us", "use", "want", "way", "well", "work",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stopwords_non_empty() {
        assert!(!stopwords().is_empty());
    }

    #[test]
    fn is_stopword_works() {
        assert!(is_stopword("the"));
        assert!(is_stopword("The")); // case-insensitive
        assert!(!is_stopword("quantum"));
    }

    #[test]
    fn remove_stopwords_filters() {
        let tokens = vec!["the", "cat", "is", "on", "the", "mat"];
        let result = remove_stopwords(&tokens);
        assert_eq!(result, vec!["cat", "mat"]);
    }

    #[test]
    fn ngrams_basic() {
        let tokens = vec!["a", "b", "c"];
        let bi = ngrams(&tokens, 2);
        assert_eq!(bi, vec![vec!["a", "b"], vec!["b", "c"]]);
    }

    #[test]
    fn ngrams_too_short() {
        let tokens = vec!["a"];
        assert!(ngrams(&tokens, 3).is_empty());
    }

    #[test]
    fn levenshtein_known() {
        assert_eq!(levenshtein("kitten", "sitting"), 3);
        assert_eq!(levenshtein("", "abc"), 3);
        assert_eq!(levenshtein("abc", "abc"), 0);
    }

    #[test]
    fn normalize_strips_punctuation() {
        assert_eq!(normalize("Hello, World!"), "hello world");
    }

    #[test]
    fn pad_sequences_works() {
        let seqs = vec![vec![1, 2], vec![3, 4, 5, 6]];
        let padded = pad_sequences(&seqs, 3, 0);
        assert_eq!(padded, vec![vec![1, 2, 0], vec![3, 4, 5]]);
    }
}
