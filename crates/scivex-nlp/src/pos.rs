use std::collections::HashMap;
use std::fmt;

/// Part-of-speech tag variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PosTag {
    Noun,
    Verb,
    Adjective,
    Adverb,
    Pronoun,
    Preposition,
    Conjunction,
    Determiner,
    Interjection,
    Punctuation,
    Other,
}

impl fmt::Display for PosTag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            PosTag::Noun => "Noun",
            PosTag::Verb => "Verb",
            PosTag::Adjective => "Adjective",
            PosTag::Adverb => "Adverb",
            PosTag::Pronoun => "Pronoun",
            PosTag::Preposition => "Preposition",
            PosTag::Conjunction => "Conjunction",
            PosTag::Determiner => "Determiner",
            PosTag::Interjection => "Interjection",
            PosTag::Punctuation => "Punctuation",
            PosTag::Other => "Other",
        };
        write!(f, "{s}")
    }
}

/// HMM-based part-of-speech tagger using the Viterbi algorithm.
pub struct HmmPosTagger {
    /// Transition probabilities: P(tag_i | tag_{i-1}).
    transition: HashMap<(PosTag, PosTag), f64>,
    /// Emission probabilities: P(word | tag).
    emission: HashMap<(PosTag, String), f64>,
    /// Initial state probabilities: P(tag at start).
    initial: HashMap<PosTag, f64>,
    /// All known tags.
    tags: Vec<PosTag>,
}

impl Default for HmmPosTagger {
    fn default() -> Self {
        Self::new()
    }
}

impl HmmPosTagger {
    /// Create a new HMM POS tagger with built-in English transition/emission probabilities.
    pub fn new() -> Self {
        let tags = vec![
            PosTag::Noun,
            PosTag::Verb,
            PosTag::Adjective,
            PosTag::Adverb,
            PosTag::Pronoun,
            PosTag::Preposition,
            PosTag::Conjunction,
            PosTag::Determiner,
            PosTag::Interjection,
            PosTag::Punctuation,
            PosTag::Other,
        ];

        let initial = Self::build_initial_probs();
        let transition = Self::build_transition_probs();
        let emission = Self::build_emission_probs();

        HmmPosTagger {
            transition,
            emission,
            initial,
            tags,
        }
    }

    /// Build initial state probabilities.
    fn build_initial_probs() -> HashMap<PosTag, f64> {
        let mut initial = HashMap::new();
        initial.insert(PosTag::Determiner, 0.25);
        initial.insert(PosTag::Pronoun, 0.20);
        initial.insert(PosTag::Noun, 0.15);
        initial.insert(PosTag::Adjective, 0.08);
        initial.insert(PosTag::Adverb, 0.06);
        initial.insert(PosTag::Verb, 0.10);
        initial.insert(PosTag::Preposition, 0.05);
        initial.insert(PosTag::Conjunction, 0.03);
        initial.insert(PosTag::Interjection, 0.04);
        initial.insert(PosTag::Punctuation, 0.02);
        initial.insert(PosTag::Other, 0.02);
        initial
    }

    /// Build transition probabilities between tags.
    fn build_transition_probs() -> HashMap<(PosTag, PosTag), f64> {
        let mut transition = HashMap::new();

        // Determiner transitions: Det -> Noun (high), Det -> Adj (medium)
        transition.insert((PosTag::Determiner, PosTag::Noun), 0.50);
        transition.insert((PosTag::Determiner, PosTag::Adjective), 0.35);
        transition.insert((PosTag::Determiner, PosTag::Adverb), 0.05);
        transition.insert((PosTag::Determiner, PosTag::Verb), 0.02);
        transition.insert((PosTag::Determiner, PosTag::Other), 0.08);

        // Noun transitions: Noun -> Verb (high), Noun -> Prep
        transition.insert((PosTag::Noun, PosTag::Verb), 0.40);
        transition.insert((PosTag::Noun, PosTag::Preposition), 0.20);
        transition.insert((PosTag::Noun, PosTag::Conjunction), 0.10);
        transition.insert((PosTag::Noun, PosTag::Noun), 0.10);
        transition.insert((PosTag::Noun, PosTag::Punctuation), 0.08);
        transition.insert((PosTag::Noun, PosTag::Adverb), 0.05);
        transition.insert((PosTag::Noun, PosTag::Determiner), 0.04);
        transition.insert((PosTag::Noun, PosTag::Other), 0.03);

        // Verb transitions: Verb -> Noun, Verb -> Det, Verb -> Adj, Verb -> Adv
        transition.insert((PosTag::Verb, PosTag::Noun), 0.15);
        transition.insert((PosTag::Verb, PosTag::Determiner), 0.25);
        transition.insert((PosTag::Verb, PosTag::Adjective), 0.18);
        transition.insert((PosTag::Verb, PosTag::Adverb), 0.12);
        transition.insert((PosTag::Verb, PosTag::Pronoun), 0.08);
        transition.insert((PosTag::Verb, PosTag::Preposition), 0.10);
        transition.insert((PosTag::Verb, PosTag::Verb), 0.05);
        transition.insert((PosTag::Verb, PosTag::Other), 0.07);

        // Adjective transitions: Adj -> Noun (high)
        transition.insert((PosTag::Adjective, PosTag::Noun), 0.60);
        transition.insert((PosTag::Adjective, PosTag::Adjective), 0.10);
        transition.insert((PosTag::Adjective, PosTag::Conjunction), 0.08);
        transition.insert((PosTag::Adjective, PosTag::Verb), 0.05);
        transition.insert((PosTag::Adjective, PosTag::Preposition), 0.05);
        transition.insert((PosTag::Adjective, PosTag::Adverb), 0.05);
        transition.insert((PosTag::Adjective, PosTag::Other), 0.07);

        // Adverb transitions
        transition.insert((PosTag::Adverb, PosTag::Verb), 0.40);
        transition.insert((PosTag::Adverb, PosTag::Adjective), 0.30);
        transition.insert((PosTag::Adverb, PosTag::Adverb), 0.10);
        transition.insert((PosTag::Adverb, PosTag::Noun), 0.05);
        transition.insert((PosTag::Adverb, PosTag::Other), 0.15);

        // Pronoun transitions: Pronoun -> Verb (high)
        transition.insert((PosTag::Pronoun, PosTag::Verb), 0.60);
        transition.insert((PosTag::Pronoun, PosTag::Adverb), 0.10);
        transition.insert((PosTag::Pronoun, PosTag::Noun), 0.05);
        transition.insert((PosTag::Pronoun, PosTag::Adjective), 0.05);
        transition.insert((PosTag::Pronoun, PosTag::Other), 0.20);

        // Preposition transitions: Prep -> Det, Prep -> Noun
        transition.insert((PosTag::Preposition, PosTag::Determiner), 0.40);
        transition.insert((PosTag::Preposition, PosTag::Noun), 0.30);
        transition.insert((PosTag::Preposition, PosTag::Pronoun), 0.10);
        transition.insert((PosTag::Preposition, PosTag::Adjective), 0.08);
        transition.insert((PosTag::Preposition, PosTag::Other), 0.12);

        // Conjunction transitions
        transition.insert((PosTag::Conjunction, PosTag::Noun), 0.25);
        transition.insert((PosTag::Conjunction, PosTag::Determiner), 0.20);
        transition.insert((PosTag::Conjunction, PosTag::Pronoun), 0.15);
        transition.insert((PosTag::Conjunction, PosTag::Verb), 0.15);
        transition.insert((PosTag::Conjunction, PosTag::Adjective), 0.10);
        transition.insert((PosTag::Conjunction, PosTag::Adverb), 0.05);
        transition.insert((PosTag::Conjunction, PosTag::Other), 0.10);

        // Interjection transitions
        transition.insert((PosTag::Interjection, PosTag::Punctuation), 0.40);
        transition.insert((PosTag::Interjection, PosTag::Pronoun), 0.20);
        transition.insert((PosTag::Interjection, PosTag::Determiner), 0.15);
        transition.insert((PosTag::Interjection, PosTag::Other), 0.25);

        // Punctuation transitions
        transition.insert((PosTag::Punctuation, PosTag::Determiner), 0.25);
        transition.insert((PosTag::Punctuation, PosTag::Pronoun), 0.20);
        transition.insert((PosTag::Punctuation, PosTag::Noun), 0.15);
        transition.insert((PosTag::Punctuation, PosTag::Conjunction), 0.10);
        transition.insert((PosTag::Punctuation, PosTag::Other), 0.30);

        // Other transitions
        transition.insert((PosTag::Other, PosTag::Noun), 0.20);
        transition.insert((PosTag::Other, PosTag::Verb), 0.20);
        transition.insert((PosTag::Other, PosTag::Determiner), 0.15);
        transition.insert((PosTag::Other, PosTag::Preposition), 0.10);
        transition.insert((PosTag::Other, PosTag::Other), 0.35);

        transition
    }

    /// Build emission probabilities for known words.
    fn build_emission_probs() -> HashMap<(PosTag, String), f64> {
        let mut emission = HashMap::new();

        // Determiners
        for word in &["the", "a", "an", "this", "that", "these", "those"] {
            emission.insert((PosTag::Determiner, (*word).to_string()), 0.90);
            // Small probability for other tags
            emission.insert((PosTag::Pronoun, (*word).to_string()), 0.05);
        }

        // Verbs
        for word in &[
            "is", "are", "was", "were", "has", "have", "do", "does", "did", "can", "will", "would",
            "could", "should", "run", "runs", "go", "goes", "make", "makes", "see", "take", "come",
            "know", "get", "give", "say", "said",
        ] {
            emission.insert((PosTag::Verb, (*word).to_string()), 0.85);
            emission.insert((PosTag::Noun, (*word).to_string()), 0.05);
        }

        // Pronouns
        for word in &[
            "he", "she", "it", "they", "we", "you", "i", "me", "him", "her", "us", "them",
        ] {
            emission.insert((PosTag::Pronoun, (*word).to_string()), 0.92);
        }

        // Prepositions
        for word in &[
            "in", "on", "at", "to", "for", "with", "from", "by", "of", "about", "into",
        ] {
            emission.insert((PosTag::Preposition, (*word).to_string()), 0.88);
        }

        // Conjunctions
        for word in &["and", "or", "but", "nor", "yet", "so"] {
            emission.insert((PosTag::Conjunction, (*word).to_string()), 0.90);
        }

        // Adverbs
        for word in &[
            "quickly", "slowly", "very", "really", "often", "never", "always", "just", "also",
        ] {
            emission.insert((PosTag::Adverb, (*word).to_string()), 0.85);
        }

        // Adjectives
        for word in &[
            "good",
            "bad",
            "big",
            "small",
            "new",
            "old",
            "great",
            "long",
            "little",
            "high",
            "large",
            "young",
            "important",
            "few",
            "right",
        ] {
            emission.insert((PosTag::Adjective, (*word).to_string()), 0.82);
        }

        // Common nouns
        for word in &[
            "dog", "cat", "man", "woman", "house", "car", "time", "day", "world", "life", "people",
            "work", "water", "food", "city",
        ] {
            emission.insert((PosTag::Noun, (*word).to_string()), 0.88);
        }

        // Punctuation
        for word in &[".", ",", "!", "?", ";", ":", "-", "(", ")", "\""] {
            emission.insert((PosTag::Punctuation, (*word).to_string()), 0.95);
        }

        // Interjections
        for word in &["oh", "wow", "hey", "ouch", "hello", "yes", "no"] {
            emission.insert((PosTag::Interjection, (*word).to_string()), 0.80);
        }

        emission
    }

    /// Get the emission probability for a word given a tag.
    /// Unknown words default to a small probability with Noun bias.
    fn emission_prob(&self, tag: PosTag, word: &str) -> f64 {
        let lower = word.to_lowercase();
        if let Some(&prob) = self.emission.get(&(tag, lower)) {
            prob
        } else {
            // Unknown word handling: bias toward Noun
            match tag {
                PosTag::Noun => 0.10,
                PosTag::Verb | PosTag::Adjective => 0.03,
                PosTag::Adverb => 0.01,
                PosTag::Pronoun
                | PosTag::Preposition
                | PosTag::Conjunction
                | PosTag::Determiner
                | PosTag::Interjection => 0.005,
                PosTag::Punctuation => 0.001,
                PosTag::Other => 0.02,
            }
        }
    }

    /// Get the transition probability P(to_tag | from_tag).
    fn transition_prob(&self, from: PosTag, to: PosTag) -> f64 {
        *self.transition.get(&(from, to)).unwrap_or(&0.001)
    }

    /// Get the initial probability for a tag.
    fn initial_prob(&self, tag: PosTag) -> f64 {
        *self.initial.get(&tag).unwrap_or(&0.01)
    }

    /// Tag a sequence of tokens using the Viterbi algorithm.
    ///
    /// Returns a vector of (word, tag) pairs.
    pub fn tag(&self, tokens: &[&str]) -> Vec<(String, PosTag)> {
        if tokens.is_empty() {
            return Vec::new();
        }

        let n = tokens.len();
        let num_tags = self.tags.len();

        // viterbi[t][i] = best log-probability of reaching tag i at position t
        let mut viterbi = vec![vec![f64::NEG_INFINITY; num_tags]; n];
        // backpointer[t][i] = index of best previous tag
        let mut backpointer = vec![vec![0usize; num_tags]; n];

        // Initialization step
        for (i, &tag) in self.tags.iter().enumerate() {
            let init_p = self.initial_prob(tag);
            let emit_p = self.emission_prob(tag, tokens[0]);
            viterbi[0][i] = init_p.ln() + emit_p.ln();
        }

        // Recursion step
        for t in 1..n {
            for (j, &tag_j) in self.tags.iter().enumerate() {
                let emit_p = self.emission_prob(tag_j, tokens[t]).ln();
                let mut best_score = f64::NEG_INFINITY;
                let mut best_prev = 0;

                for (i, &tag_i) in self.tags.iter().enumerate() {
                    let score = viterbi[t - 1][i] + self.transition_prob(tag_i, tag_j).ln();
                    if score > best_score {
                        best_score = score;
                        best_prev = i;
                    }
                }

                viterbi[t][j] = best_score + emit_p;
                backpointer[t][j] = best_prev;
            }
        }

        // Termination: find best final tag
        let mut best_final = 0;
        let mut best_score = f64::NEG_INFINITY;
        for (i, row) in viterbi[n - 1].iter().enumerate().take(num_tags) {
            if *row > best_score {
                best_score = *row;
                best_final = i;
            }
        }

        // Backtrace
        let mut tag_indices = vec![0usize; n];
        tag_indices[n - 1] = best_final;
        for t in (0..n - 1).rev() {
            tag_indices[t] = backpointer[t + 1][tag_indices[t + 1]];
        }

        tokens
            .iter()
            .enumerate()
            .map(|(t, &word)| (word.to_string(), self.tags[tag_indices[t]]))
            .collect()
    }

    /// Tag text by first splitting on whitespace, then applying the Viterbi algorithm.
    pub fn tag_str(&self, text: &str) -> Vec<(String, PosTag)> {
        let tokens: Vec<&str> = text.split_whitespace().collect();
        self.tag(&tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_sentence() {
        let tagger = HmmPosTagger::new();
        let result = tagger.tag(&["the", "dog", "runs"]);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].1, PosTag::Determiner);
        assert_eq!(result[1].1, PosTag::Noun);
        assert_eq!(result[2].1, PosTag::Verb);
    }

    #[test]
    fn test_pronoun_sentence() {
        let tagger = HmmPosTagger::new();
        let result = tagger.tag(&["he", "is", "good"]);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].1, PosTag::Pronoun);
        assert_eq!(result[1].1, PosTag::Verb);
        assert_eq!(result[2].1, PosTag::Adjective);
    }

    #[test]
    fn test_unknown_words() {
        let tagger = HmmPosTagger::new();
        let result = tagger.tag(&["the", "flurbogriff", "zantiplied"]);
        assert_eq!(result.len(), 3);
        // First word is known
        assert_eq!(result[0].1, PosTag::Determiner);
        // Unknown words should still get tagged (likely as Noun or Adjective after Det)
        assert!(result[1].1 == PosTag::Noun || result[1].1 == PosTag::Adjective);
        // Third position after Noun, unknown word
        assert!(result[2].1 == PosTag::Verb || result[2].1 == PosTag::Noun);
    }

    #[test]
    fn test_empty_input() {
        let tagger = HmmPosTagger::new();
        let result = tagger.tag(&[]);
        assert!(result.is_empty());
    }
}
