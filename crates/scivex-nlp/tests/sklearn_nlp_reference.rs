//! Reference tests comparing scivex-nlp against known sklearn/analytical values.

use scivex_core::Tensor;
use scivex_nlp::similarity::cosine_similarity;
use scivex_nlp::tokenize::{Tokenizer, WhitespaceTokenizer};
use scivex_nlp::vectorize::CountVectorizer;

const TOL: f64 = 1e-10;

// ─── Tokenization ────────────────────────────────────────────────────

#[test]
fn whitespace_tokenizer_basic() {
    let tokens = WhitespaceTokenizer.tokenize("hello world foo bar");
    assert_eq!(tokens, vec!["hello", "world", "foo", "bar"]);
}

#[test]
fn whitespace_tokenizer_multiple_spaces() {
    let tokens = WhitespaceTokenizer.tokenize("  hello   world  ");
    assert_eq!(tokens, vec!["hello", "world"]);
}

// ─── CountVectorizer ─────────────────────────────────────────────────

#[test]
fn count_vectorizer_basic() {
    let docs = vec!["cat dog", "dog bird", "cat bird cat"];
    let mut cv = CountVectorizer::new();
    let matrix = cv.fit_transform::<f64>(&docs).unwrap();

    // Vocabulary size
    let vocab = cv.vocabulary();
    assert_eq!(vocab.len(), 3); // cat, dog, bird

    // Matrix shape: 3 docs × 3 terms
    assert_eq!(matrix.shape(), &[3, 3]);

    // "cat" appears 2 times in doc 2
    let cat_idx = vocab["cat"];
    let cat_count_doc2: f64 = matrix.as_slice()[2 * 3 + cat_idx];
    assert!(
        (cat_count_doc2 - 2.0).abs() < TOL,
        "cat in doc2 = {cat_count_doc2}"
    );
}

// ─── Cosine similarity ──────────────────────────────────────────────

#[test]
fn cosine_similarity_parallel() {
    // Identical vectors have cosine similarity = 1
    let a = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0], vec![3]).unwrap();
    let b = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0], vec![3]).unwrap();
    let sim = cosine_similarity(&a, &b).unwrap();
    assert!((sim - 1.0).abs() < TOL, "cos_sim = {sim}");
}

#[test]
fn cosine_similarity_orthogonal() {
    // Orthogonal vectors have cosine similarity = 0
    let a = Tensor::from_vec(vec![1.0_f64, 0.0, 0.0], vec![3]).unwrap();
    let b = Tensor::from_vec(vec![0.0_f64, 1.0, 0.0], vec![3]).unwrap();
    let sim = cosine_similarity(&a, &b).unwrap();
    assert!(sim.abs() < TOL, "cos_sim = {sim}");
}

#[test]
fn cosine_similarity_opposite() {
    // Opposite vectors have cosine similarity = -1
    let a = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0], vec![3]).unwrap();
    let b = Tensor::from_vec(vec![-1.0_f64, -2.0, -3.0], vec![3]).unwrap();
    let sim = cosine_similarity(&a, &b).unwrap();
    assert!((sim - (-1.0)).abs() < TOL, "cos_sim = {sim}");
}

#[test]
fn cosine_similarity_known_value() {
    // cos([1,0], [1,1]) = 1/sqrt(2) ≈ 0.7071
    let a = Tensor::from_vec(vec![1.0_f64, 0.0], vec![2]).unwrap();
    let b = Tensor::from_vec(vec![1.0_f64, 1.0], vec![2]).unwrap();
    let sim = cosine_similarity(&a, &b).unwrap();
    let expected = 1.0 / 2.0_f64.sqrt();
    assert!(
        (sim - expected).abs() < TOL,
        "cos_sim = {sim}, expected = {expected}"
    );
}
