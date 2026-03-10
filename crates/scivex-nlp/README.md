# scivex-nlp

Natural language processing for Scivex. Tokenization, stemming, vectorization,
word embeddings, and sentiment analysis.

## Highlights

- **Tokenizers** — Whitespace, word, character, n-gram (trait-based, extensible)
- **Porter stemmer** — Full 5-step Porter stemming algorithm
- **Text utilities** — Stopwords, n-grams, edit distance, normalization
- **Vectorization** — CountVectorizer, TfidfVectorizer (returns Tensor<T>)
- **Word embeddings** — Vector storage, similarity search, analogy solving
- **Sentiment** — Lexicon-based sentiment analysis with modifier support
- **Similarity** — Cosine, Jaccard, normalized edit distance

## Usage

```rust
use scivex_nlp::prelude::*;

// Tokenize and vectorize
let tokenizer = WordTokenizer::new(true); // lowercase
let tokens = tokenizer.tokenize("Hello world");

let mut tfidf = TfidfVectorizer::new(tokenizer);
let matrix = tfidf.fit_transform(&documents).unwrap();

// Sentiment analysis
let analyzer = SentimentAnalyzer::new();
let result = analyzer.analyze("This library is amazing!");
println!("{:?}", result.sentiment()); // Positive

// Word embeddings
let embeddings = WordEmbeddings::from_pairs(word_vec_pairs);
let similar = embeddings.most_similar("king", 5);
let analogy = embeddings.analogy("king", "queen", "man"); // → "woman"
```

## License

MIT
