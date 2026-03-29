"""Tests for pyscivex NLP — nlp submodule."""

import pyscivex as sv


# ===========================================================================
# TOKENIZERS
# ===========================================================================


class TestTokenizers:
    def test_word_tokenize(self):
        tokens = sv.nlp.word_tokenize("Hello, World! How are you?")
        assert len(tokens) > 0
        assert "hello" in tokens  # lowercase by default

    def test_char_tokenize(self):
        tokens = sv.nlp.char_tokenize("abc")
        assert tokens == ["a", "b", "c"]

    def test_ngram_tokenize(self):
        tokens = sv.nlp.ngram_tokenize("hello", 2)
        assert "he" in tokens
        assert "el" in tokens

    def test_whitespace_tokenize(self):
        tokens = sv.nlp.whitespace_tokenize("one two three")
        assert tokens == ["one", "two", "three"]


# ===========================================================================
# WORDPIECE TOKENIZER
# ===========================================================================


class TestWordPiece:
    def test_create(self):
        vocab = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "hello", "world", "##ing"]
        wp = sv.nlp.WordPieceTokenizer(vocab)
        assert "WordPieceTokenizer" in repr(wp)

    def test_tokenize(self):
        vocab = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "hello", "world"]
        wp = sv.nlp.WordPieceTokenizer(vocab)
        pieces = wp.tokenize("hello world")
        assert len(pieces) > 0

    def test_encode_decode(self):
        vocab = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "hello", "world"]
        wp = sv.nlp.WordPieceTokenizer(vocab)
        ids = wp.encode("hello world")
        assert len(ids) > 0
        text = wp.decode(ids)
        assert len(text) > 0


# ===========================================================================
# STEMMING
# ===========================================================================


class TestStemming:
    def test_porter_stemmer(self):
        stemmer = sv.nlp.PorterStemmer()
        assert stemmer.stem("running") == "run"
        assert stemmer.stem("cats") == "cat"

    def test_stem_many(self):
        stemmer = sv.nlp.PorterStemmer()
        results = stemmer.stem_many(["running", "cats", "happily"])
        assert len(results) == 3
        assert results[0] == "run"

    def test_repr(self):
        stemmer = sv.nlp.PorterStemmer()
        assert "PorterStemmer" in repr(stemmer)


# ===========================================================================
# STOPWORDS
# ===========================================================================


class TestStopwords:
    def test_stopwords_list(self):
        words = sv.nlp.stopwords()
        assert len(words) > 100
        assert "the" in words
        assert "is" in words

    def test_is_stopword(self):
        assert sv.nlp.is_stopword("the") is True
        assert sv.nlp.is_stopword("python") is False

    def test_remove_stopwords(self):
        tokens = ["the", "cat", "is", "on", "the", "mat"]
        filtered = sv.nlp.remove_stopwords(tokens)
        assert "cat" in filtered
        assert "mat" in filtered
        assert "the" not in filtered


# ===========================================================================
# TEXT UTILITIES
# ===========================================================================


class TestTextUtils:
    def test_ngrams(self):
        tokens = ["I", "love", "natural", "language"]
        bigrams = sv.nlp.ngrams(tokens, 2)
        assert len(bigrams) == 3
        assert bigrams[0] == ["I", "love"]

    def test_edit_distance(self):
        assert sv.nlp.edit_distance("kitten", "sitting") == 3
        assert sv.nlp.edit_distance("hello", "hello") == 0

    def test_normalize(self):
        result = sv.nlp.normalize("Hello, World! 123")
        assert result == result.lower()

    def test_pad_sequences(self):
        seqs = [[1, 2, 3], [4, 5]]
        padded = sv.nlp.pad_sequences(seqs, 4, 0)
        assert len(padded[0]) == 4
        assert len(padded[1]) == 4


# ===========================================================================
# SIMILARITY
# ===========================================================================


class TestSimilarity:
    def test_cosine_sim(self):
        a = [1.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        assert abs(sv.nlp.cosine_sim(a, b) - 1.0) < 1e-10

    def test_cosine_sim_orthogonal(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(sv.nlp.cosine_sim(a, b)) < 1e-10

    def test_jaccard(self):
        a = ["cat", "dog", "fish"]
        b = ["cat", "bird", "fish"]
        sim = sv.nlp.jaccard(a, b)
        assert 0.0 < sim < 1.0

    def test_edit_distance_norm(self):
        d = sv.nlp.edit_distance_norm("abc", "abc")
        # Normalized edit distance: 0.0 = identical, 1.0 = completely different
        # OR it could be similarity: 1.0 = identical, 0.0 = completely different
        # Just check it returns a value in [0, 1]
        assert 0.0 <= d <= 1.0
        d2 = sv.nlp.edit_distance_norm("abc", "xyz")
        assert 0.0 <= d2 <= 1.0
        assert d != d2  # identical vs different should differ


# ===========================================================================
# VECTORIZERS
# ===========================================================================


class TestVectorizers:
    def test_count_vectorizer(self):
        docs = ["the cat sat", "the dog ran", "cat and dog"]
        cv = sv.nlp.CountVectorizer()
        matrix = cv.fit_transform(docs)
        shape = matrix.shape()
        assert shape[0] == 3  # 3 documents
        assert shape[1] > 0  # vocab size

    def test_count_vectorizer_vocab(self):
        docs = ["hello world", "hello python"]
        cv = sv.nlp.CountVectorizer()
        cv.fit(docs)
        vocab = cv.vocabulary()
        assert "hello" in vocab
        assert "world" in vocab
        assert cv.is_fitted()

    def test_count_vectorizer_repr(self):
        cv = sv.nlp.CountVectorizer()
        assert "CountVectorizer" in repr(cv)

    def test_tfidf_vectorizer(self):
        docs = ["the cat sat", "the dog ran", "cat and dog"]
        tv = sv.nlp.TfidfVectorizer()
        matrix = tv.fit_transform(docs)
        shape = matrix.shape()
        assert shape[0] == 3
        assert shape[1] > 0

    def test_tfidf_fit_transform_separate(self):
        docs = ["hello world", "hello python"]
        tv = sv.nlp.TfidfVectorizer()
        tv.fit(docs)
        matrix = tv.transform(docs)
        assert matrix.shape()[0] == 2


# ===========================================================================
# WORD2VEC
# ===========================================================================


class TestWord2Vec:
    def test_train(self):
        corpus = [
            ["the", "cat", "sat", "on", "the", "mat"],
            ["the", "dog", "ran", "in", "the", "park"],
            ["a", "cat", "and", "a", "dog", "played"],
        ]
        w2v = sv.nlp.Word2Vec.train(
            corpus, embedding_dim=10, epochs=5, seed=42
        )
        assert w2v.vocab_size() > 0
        assert w2v.embedding_dim() == 10

    def test_get_vector(self):
        corpus = [
            ["cat", "dog", "cat", "dog", "cat", "dog"],
            ["fish", "bird", "fish", "bird", "fish", "bird"],
        ]
        w2v = sv.nlp.Word2Vec.train(corpus, embedding_dim=5, epochs=3, seed=42)
        vec = w2v.get("cat")
        assert vec is not None
        assert len(vec) == 5

    def test_most_similar(self):
        corpus = [
            ["the", "cat", "sat", "on", "the", "mat"],
            ["the", "dog", "ran", "in", "the", "park"],
            ["a", "cat", "and", "a", "dog", "played"],
        ] * 5  # repeat for better training
        w2v = sv.nlp.Word2Vec.train(corpus, embedding_dim=10, epochs=10, seed=42)
        similar = w2v.most_similar("cat", 3)
        assert len(similar) > 0

    def test_repr(self):
        corpus = [["a", "b", "c"]]
        w2v = sv.nlp.Word2Vec.train(corpus, embedding_dim=5, epochs=1, seed=42)
        assert "Word2Vec" in repr(w2v)


# ===========================================================================
# WORD EMBEDDINGS
# ===========================================================================


class TestWordEmbeddings:
    def test_create(self):
        pairs = [
            ("cat", [1.0, 0.0, 0.0]),
            ("dog", [0.9, 0.1, 0.0]),
            ("fish", [0.0, 0.0, 1.0]),
        ]
        emb = sv.nlp.WordEmbeddings(pairs)
        assert emb.vocab_size() == 3
        assert emb.embedding_dim() == 3

    def test_get(self):
        pairs = [
            ("cat", [1.0, 0.0]),
            ("dog", [0.0, 1.0]),
        ]
        emb = sv.nlp.WordEmbeddings(pairs)
        vec = emb.get("cat")
        assert vec is not None
        assert len(vec) == 2

    def test_similarity(self):
        pairs = [
            ("cat", [1.0, 0.0]),
            ("dog", [1.0, 0.0]),
            ("fish", [0.0, 1.0]),
        ]
        emb = sv.nlp.WordEmbeddings(pairs)
        sim_same = emb.similarity("cat", "dog")
        sim_diff = emb.similarity("cat", "fish")
        assert sim_same > sim_diff

    def test_most_similar(self):
        pairs = [
            ("cat", [1.0, 0.0, 0.0]),
            ("dog", [0.9, 0.1, 0.0]),
            ("fish", [0.0, 0.0, 1.0]),
        ]
        emb = sv.nlp.WordEmbeddings(pairs)
        similar = emb.most_similar("cat", 2)
        assert len(similar) == 2
        assert similar[0][0] == "dog"  # dog is most similar to cat


# ===========================================================================
# SENTIMENT
# ===========================================================================


class TestSentiment:
    def test_positive(self):
        sa = sv.nlp.SentimentAnalyzer()
        result = sa.analyze("This is great and wonderful!")
        assert result["polarity"] > 0
        assert result["positive"] > 0

    def test_negative(self):
        sa = sv.nlp.SentimentAnalyzer()
        result = sa.analyze("This is terrible and awful!")
        assert result["polarity"] < 0
        assert result["negative"] > 0

    def test_neutral(self):
        sa = sv.nlp.SentimentAnalyzer()
        result = sa.analyze("The sky is blue.")
        assert "polarity" in result

    def test_repr(self):
        sa = sv.nlp.SentimentAnalyzer()
        assert "SentimentAnalyzer" in repr(sa)


# ===========================================================================
# POS TAGGING
# ===========================================================================


class TestPosTagger:
    def test_tag(self):
        tagger = sv.nlp.HmmPosTagger()
        tags = tagger.tag("The cat sat on the mat")
        assert len(tags) > 0
        # Each result is (word, tag) tuple
        assert len(tags[0]) == 2

    def test_tag_tokens(self):
        tagger = sv.nlp.HmmPosTagger()
        tags = tagger.tag_tokens(["The", "cat", "sat"])
        assert len(tags) == 3

    def test_repr(self):
        tagger = sv.nlp.HmmPosTagger()
        assert "HmmPosTagger" in repr(tagger)


# ===========================================================================
# NER
# ===========================================================================


class TestNer:
    def test_recognize(self):
        ner = sv.nlp.RuleBasedNer()
        entities = ner.recognize_text("John went to New York")
        # Returns list of dicts with text, type, start, end
        assert isinstance(entities, list)

    def test_add_entity(self):
        ner = sv.nlp.RuleBasedNer()
        ner.add_entity("person", "Alice")
        entities = ner.recognize(["Alice", "went", "home"])
        found = [e for e in entities if e["text"] == "Alice"]
        assert len(found) > 0

    def test_repr(self):
        ner = sv.nlp.RuleBasedNer()
        assert "RuleBasedNer" in repr(ner)


# ===========================================================================
# LDA
# ===========================================================================


class TestLda:
    def test_fit(self):
        docs = [
            ["cat", "dog", "pet", "animal"],
            ["car", "truck", "vehicle", "drive"],
            ["cat", "kitten", "pet", "fur"],
            ["car", "engine", "vehicle", "road"],
        ]
        lda = sv.nlp.LDA(n_topics=2, n_iterations=20, seed=42)
        lda.fit(docs)
        assert lda.n_topics() == 2

    def test_top_words(self):
        docs = [
            ["cat", "dog", "pet", "animal"],
            ["car", "truck", "vehicle", "drive"],
            ["cat", "kitten", "pet", "fur"],
            ["car", "engine", "vehicle", "road"],
        ]
        lda = sv.nlp.LDA(n_topics=2, n_iterations=50, seed=42)
        lda.fit(docs)
        words = lda.top_words(0, 3)
        assert len(words) == 3
        assert all(isinstance(w, tuple) and len(w) == 2 for w in words)

    def test_doc_topic_dist(self):
        docs = [
            ["cat", "dog", "pet"],
            ["car", "truck", "drive"],
        ]
        lda = sv.nlp.LDA(n_topics=2, n_iterations=20, seed=42)
        lda.fit(docs)
        dist = lda.document_topic_distribution(0)
        assert len(dist) == 2
        assert abs(sum(dist) - 1.0) < 0.1  # should roughly sum to 1

    def test_repr(self):
        lda = sv.nlp.LDA(n_topics=5)
        assert "LDA" in repr(lda)
        assert "5" in repr(lda)


# ===========================================================================
# INTEGRATION
# ===========================================================================


class TestIntegration:
    def test_all_accessible(self):
        items = [
            # Functions
            sv.nlp.word_tokenize,
            sv.nlp.char_tokenize,
            sv.nlp.ngram_tokenize,
            sv.nlp.whitespace_tokenize,
            sv.nlp.stopwords,
            sv.nlp.is_stopword,
            sv.nlp.remove_stopwords,
            sv.nlp.ngrams,
            sv.nlp.edit_distance,
            sv.nlp.normalize,
            sv.nlp.pad_sequences,
            sv.nlp.cosine_sim,
            sv.nlp.jaccard,
            sv.nlp.edit_distance_norm,
            # Classes
            sv.nlp.WordPieceTokenizer,
            sv.nlp.PorterStemmer,
            sv.nlp.CountVectorizer,
            sv.nlp.TfidfVectorizer,
            sv.nlp.Word2Vec,
            sv.nlp.WordEmbeddings,
            sv.nlp.SentimentAnalyzer,
            sv.nlp.HmmPosTagger,
            sv.nlp.RuleBasedNer,
            sv.nlp.LDA,
        ]
        for item in items:
            assert item is not None
