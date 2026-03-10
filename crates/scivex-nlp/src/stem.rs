//! Porter stemmer implementation.

/// The Porter stemming algorithm for English words.
///
/// Implements the classic 5-step Porter stemmer that reduces words to their
/// stems by stripping suffixes.
pub struct PorterStemmer;

impl PorterStemmer {
    /// Stem a single word, returning the stemmed form.
    #[must_use]
    pub fn stem(&self, word: &str) -> String {
        let w = word.to_lowercase();
        if w.len() <= 2 {
            return w;
        }
        let w = step1a(&w);
        let w = step1b(&w);
        let w = step1c(&w);
        let w = step2(&w);
        let w = step3(&w);
        let w = step4(&w);
        step5(&w)
    }
}

// Measure: the number of VC sequences in the stem before a suffix.
fn measure(s: &str) -> usize {
    let chars: Vec<char> = s.chars().collect();
    let mut m = 0;
    let mut i = 0;
    let n = chars.len();

    // Skip initial consonants.
    while i < n && !is_vowel_at(&chars, i) {
        i += 1;
    }

    loop {
        // Skip vowels.
        while i < n && is_vowel_at(&chars, i) {
            i += 1;
        }
        if i >= n {
            break;
        }
        // Skip consonants.
        while i < n && !is_vowel_at(&chars, i) {
            i += 1;
        }
        m += 1;
    }

    m
}

fn is_vowel_at(chars: &[char], i: usize) -> bool {
    matches!(chars[i], 'a' | 'e' | 'i' | 'o' | 'u')
        || (chars[i] == 'y' && i > 0 && !matches!(chars[i - 1], 'a' | 'e' | 'i' | 'o' | 'u'))
}

fn has_vowel(s: &str) -> bool {
    let chars: Vec<char> = s.chars().collect();
    (0..chars.len()).any(|i| is_vowel_at(&chars, i))
}

fn ends_double_consonant(s: &str) -> bool {
    let b = s.as_bytes();
    if b.len() < 2 {
        return false;
    }
    let last = b[b.len() - 1];
    let prev = b[b.len() - 2];
    if last != prev {
        return false;
    }
    let chars: Vec<char> = s.chars().collect();
    !is_vowel_at(&chars, chars.len() - 1)
}

fn ends_cvc(s: &str) -> bool {
    let chars: Vec<char> = s.chars().collect();
    let n = chars.len();
    if n < 3 {
        return false;
    }
    !is_vowel_at(&chars, n - 1)
        && is_vowel_at(&chars, n - 2)
        && !is_vowel_at(&chars, n - 3)
        && !matches!(chars[n - 1], 'w' | 'x' | 'y')
}

// Step 1a: plurals
fn step1a(word: &str) -> String {
    if let Some(stem) = word.strip_suffix("sses") {
        return format!("{stem}ss");
    }
    if let Some(stem) = word.strip_suffix("ies") {
        return format!("{stem}i");
    }
    if word.ends_with("ss") {
        return word.to_string();
    }
    if let Some(stem) = word.strip_suffix('s')
        && !stem.is_empty()
    {
        return stem.to_string();
    }
    word.to_string()
}

// Step 1b: past tenses / progressive
fn step1b(word: &str) -> String {
    if let Some(stem) = word.strip_suffix("eed") {
        if measure(stem) > 0 {
            return format!("{stem}ee");
        }
        return word.to_string();
    }

    let (trimmed, found) = if let Some(stem) = word.strip_suffix("ed") {
        (stem, has_vowel(stem))
    } else if let Some(stem) = word.strip_suffix("ing") {
        (stem, has_vowel(stem))
    } else {
        return word.to_string();
    };

    if !found {
        return word.to_string();
    }

    let w = trimmed.to_string();

    if w.ends_with("at") || w.ends_with("bl") || w.ends_with("iz") {
        return format!("{w}e");
    }

    if ends_double_consonant(&w) {
        let last = w.as_bytes()[w.len() - 1];
        if !matches!(last, b'l' | b's' | b'z') {
            return w[..w.len() - 1].to_string();
        }
    }

    if measure(&w) == 1 && ends_cvc(&w) {
        return format!("{w}e");
    }

    w
}

// Step 1c: y → i
fn step1c(word: &str) -> String {
    if let Some(stem) = word.strip_suffix('y')
        && has_vowel(stem)
        && !stem.is_empty()
    {
        return format!("{stem}i");
    }
    word.to_string()
}

// Step 2: map double suffixes
fn step2(word: &str) -> String {
    let mappings: &[(&str, &str)] = &[
        ("ational", "ate"),
        ("tional", "tion"),
        ("enci", "ence"),
        ("anci", "ance"),
        ("izer", "ize"),
        ("abli", "able"),
        ("alli", "al"),
        ("entli", "ent"),
        ("eli", "e"),
        ("ousli", "ous"),
        ("ization", "ize"),
        ("ation", "ate"),
        ("ator", "ate"),
        ("alism", "al"),
        ("iveness", "ive"),
        ("fulness", "ful"),
        ("ousness", "ous"),
        ("aliti", "al"),
        ("iviti", "ive"),
        ("biliti", "ble"),
    ];

    for &(suffix, replacement) in mappings {
        if let Some(stem) = word.strip_suffix(suffix) {
            if measure(stem) > 0 {
                return format!("{stem}{replacement}");
            }
            return word.to_string();
        }
    }
    word.to_string()
}

// Step 3: map single suffixes
fn step3(word: &str) -> String {
    let mappings: &[(&str, &str)] = &[
        ("icate", "ic"),
        ("ative", ""),
        ("alize", "al"),
        ("iciti", "ic"),
        ("ical", "ic"),
        ("ful", ""),
        ("ness", ""),
    ];

    for &(suffix, replacement) in mappings {
        if let Some(stem) = word.strip_suffix(suffix) {
            if measure(stem) > 0 {
                return format!("{stem}{replacement}");
            }
            return word.to_string();
        }
    }
    word.to_string()
}

// Step 4: remove suffixes
fn step4(word: &str) -> String {
    let suffixes: &[&str] = &[
        "al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement", "ment", "ent", "ion",
        "ou", "ism", "ate", "iti", "ous", "ive", "ize",
    ];

    for &suffix in suffixes {
        if let Some(stem) = word.strip_suffix(suffix) {
            if measure(stem) > 1 {
                if suffix == "ion" {
                    if stem.ends_with('s') || stem.ends_with('t') {
                        return stem.to_string();
                    }
                } else {
                    return stem.to_string();
                }
            }
            return word.to_string();
        }
    }
    word.to_string()
}

// Step 5: tidy up
fn step5(word: &str) -> String {
    let mut w = word.to_string();

    // 5a: remove trailing e
    if w.ends_with('e') {
        let stem = &w[..w.len() - 1];
        let m = measure(stem);
        if m > 1 || (m == 1 && !ends_cvc(stem)) {
            w = stem.to_string();
        }
    }

    // 5b: ll -> l if m > 1
    if w.ends_with("ll") && measure(&w[..w.len() - 1]) > 1 {
        w.pop();
    }

    w
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stem_plurals() {
        let s = PorterStemmer;
        assert_eq!(s.stem("caresses"), "caress");
        assert_eq!(s.stem("cats"), "cat");
        assert_eq!(s.stem("ponies"), "poni");
    }

    #[test]
    fn stem_past_tense() {
        let s = PorterStemmer;
        assert_eq!(s.stem("agreed"), "agre");
        assert_eq!(s.stem("plastered"), "plaster");
    }

    #[test]
    fn stem_progressive() {
        let s = PorterStemmer;
        assert_eq!(s.stem("motoring"), "motor");
        assert_eq!(s.stem("sing"), "sing");
    }

    #[test]
    fn stem_short_words() {
        let s = PorterStemmer;
        assert_eq!(s.stem("a"), "a");
        assert_eq!(s.stem("is"), "is");
    }

    #[test]
    fn stem_y_to_i() {
        let s = PorterStemmer;
        assert_eq!(s.stem("happy"), "happi");
    }
}
