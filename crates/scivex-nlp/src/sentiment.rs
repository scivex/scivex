//! Simple lexicon-based sentiment analysis.

use std::collections::HashSet;

use crate::tokenize::WordTokenizer;

/// Result of sentiment analysis on a text.
///
/// # Examples
///
/// ```
/// # use scivex_nlp::SentimentAnalyzer;
/// let analyzer = SentimentAnalyzer::new();
/// let result = analyzer.analyze("great wonderful");
/// assert!(result.polarity > 0.0);
/// ```
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone, PartialEq)]
pub struct SentimentResult {
    /// Polarity score from -1.0 (most negative) to 1.0 (most positive).
    pub polarity: f64,
    /// Count of positive words found.
    pub positive: usize,
    /// Count of negative words found.
    pub negative: usize,
}

#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
/// Bag-of-words sentiment analyzer with a built-in English lexicon.
///
/// # Examples
///
/// ```
/// # use scivex_nlp::SentimentAnalyzer;
/// let analyzer = SentimentAnalyzer::new();
/// let result = analyzer.analyze("terrible awful");
/// assert!(result.polarity < 0.0);
/// ```
pub struct SentimentAnalyzer {
    positive_words: HashSet<String>,
    negative_words: HashSet<String>,
}

impl SentimentAnalyzer {
    /// Create with the built-in positive/negative word lists.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_nlp::sentiment::SentimentAnalyzer;
    /// let sa = SentimentAnalyzer::new();
    /// let result = sa.analyze("This is a great and wonderful day");
    /// assert!(result.polarity > 0.0);
    /// assert!(result.positive >= 2);
    /// ```
    #[must_use]
    pub fn new() -> Self {
        let positive_words: HashSet<String> =
            POSITIVE_WORDS.iter().map(|&s| s.to_string()).collect();
        let negative_words: HashSet<String> =
            NEGATIVE_WORDS.iter().map(|&s| s.to_string()).collect();
        Self {
            positive_words,
            negative_words,
        }
    }

    /// Analyze the sentiment of a text string.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_nlp::sentiment::SentimentAnalyzer;
    /// let sa = SentimentAnalyzer::new();
    /// let neg = sa.analyze("This is terrible and awful");
    /// assert!(neg.polarity < 0.0);
    /// assert!(neg.negative >= 2);
    /// ```
    #[must_use]
    pub fn analyze(&self, text: &str) -> SentimentResult {
        let tokenizer = WordTokenizer::new().with_lowercase(true);
        let tokens = tokenizer.tokenize_owned(text);

        let mut positive = 0_usize;
        let mut negative = 0_usize;

        for token in &tokens {
            if self.positive_words.contains(token) {
                positive += 1;
            } else if self.negative_words.contains(token) {
                negative += 1;
            }
        }

        let total = positive + negative;
        let polarity = if total == 0 {
            0.0
        } else {
            (positive as f64 - negative as f64) / total as f64
        };

        SentimentResult {
            polarity,
            positive,
            negative,
        }
    }
}

impl Default for SentimentAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[rustfmt::skip]
static POSITIVE_WORDS: [&str; 50] = [
    "good", "great", "excellent", "amazing", "wonderful", "fantastic",
    "awesome", "love", "happy", "joy", "beautiful", "brilliant", "best",
    "perfect", "superb", "outstanding", "magnificent", "delightful",
    "impressive", "marvelous", "terrific", "splendid", "fabulous",
    "pleasant", "nice", "fine", "enjoy", "glad", "pleased", "thankful",
    "grateful", "exciting", "positive", "success", "win", "winning",
    "triumph", "celebrate", "cheerful", "bright", "warm", "kind",
    "generous", "helpful", "clever", "strong", "powerful", "calm",
    "peaceful", "comfortable",
];

#[rustfmt::skip]
static NEGATIVE_WORDS: [&str; 50] = [
    "bad", "terrible", "horrible", "awful", "worst", "hate", "ugly",
    "disgusting", "dreadful", "poor", "sad", "angry", "miserable",
    "painful", "unhappy", "disappointing", "failure", "fail", "wrong",
    "broken", "weak", "useless", "annoying", "boring", "stupid",
    "ridiculous", "pathetic", "lousy", "nasty", "rude", "cruel",
    "harsh", "toxic", "negative", "disaster", "catastrophe", "tragic",
    "depressing", "hopeless", "fearful", "worried", "anxious", "stress",
    "frustrating", "confused", "lost", "lonely", "empty", "dark", "cold",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positive_text() {
        let sa = SentimentAnalyzer::new();
        let result = sa.analyze("This is a great and wonderful day");
        assert!(result.polarity > 0.0);
        assert!(result.positive > 0);
    }

    #[test]
    fn negative_text() {
        let sa = SentimentAnalyzer::new();
        let result = sa.analyze("This is terrible and awful");
        assert!(result.polarity < 0.0);
        assert!(result.negative > 0);
    }

    #[test]
    fn neutral_text() {
        let sa = SentimentAnalyzer::new();
        let result = sa.analyze("The cat sat on the mat");
        assert!((result.polarity - 0.0).abs() < 1e-10);
        assert_eq!(result.positive, 0);
        assert_eq!(result.negative, 0);
    }

    #[test]
    fn mixed_text() {
        let sa = SentimentAnalyzer::new();
        let result = sa.analyze("good but terrible");
        assert_eq!(result.positive, 1);
        assert_eq!(result.negative, 1);
        assert!((result.polarity - 0.0).abs() < 1e-10);
    }
}
