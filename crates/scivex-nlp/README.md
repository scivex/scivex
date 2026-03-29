# scivex-nlp

Natural language processing for Scivex. Tokenization, text processing,
embeddings, and text analysis tools.

## Highlights

- **Tokenizers** — BPE, WordPiece, Unigram, whitespace, regex-based
- **Stemming** — Porter stemmer for English
- **TF-IDF** — Term frequency-inverse document frequency vectorization
- **Word embeddings** — Word2Vec (Skip-gram, CBOW) training and lookup
- **Sentiment** — Lexicon-based sentiment analysis (VADER-style)
- **Text preprocessing** — Lowercasing, stopword removal, n-grams
- **Vocabulary** — Vocabulary building with frequency thresholds
- **Similarity** — Cosine similarity, Jaccard index for text comparison

## Usage

```rust
use scivex_nlp::prelude::*;

// Tokenization
let tokenizer = BpeTokenizer::train(&corpus, 8000);
let tokens = tokenizer.encode("Hello, world!");

// TF-IDF
let tfidf = TfIdf::fit(&documents);
let vector = tfidf.transform(&document);

// Word embeddings
let w2v = Word2Vec::train(&sentences, 100, 5);
let embedding = w2v.get_vector("science");
```

## License

MIT
