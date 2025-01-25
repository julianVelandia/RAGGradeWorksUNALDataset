from rag.vectorizer import TextVectorizer


def test_vectorizer_fit_transform():
    docs = ["Hello world", "Vectorization test"]
    vectorizer = TextVectorizer()
    embeddings = vectorizer.fit_transform(docs)
    assert embeddings.shape[0] == 2, "It should generate embeddings for all documents"
    assert embeddings.shape[1] > 0, "The vectorizer should generate a valid representation"


def test_vectorizer_transform():
    docs = ["Hello world", "Vectorization test"]
    vectorizer = TextVectorizer()
    vectorizer.fit_transform(docs)
    query_vector = vectorizer.transform("Hello")
    assert query_vector.shape[0] == 1, "It should generate a vector for the query"
    assert query_vector.shape[1] > 0, "The generated vector should have a valid dimension"
