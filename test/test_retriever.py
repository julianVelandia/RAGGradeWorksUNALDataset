from rag.retriever import Retriever
from rag.vectorizer import TextVectorizer


def test_retriever_find_similar_sections():
    docs = [
        "Hello world! This is a simple introductory text to warm things up.",
        "The package has a test case included for ensuring compatibility and performance.",
        "More data to test various edge cases in a simulated environment for better results.",
        "Testing is a crucial step in software development and ensures better code reliability.",
        "This document contains examples of tests, data preparation, and expected outputs."
    ]

    vectorizer = TextVectorizer()
    embeddings = vectorizer.fit_transform(docs)
    retriever = Retriever(vectorizer)

    query = "How important is a test in software development?"
    results = retriever.find_similar_sections(query, embeddings, docs)

    assert len(results) > 0, "It should find at least one similar section"
    assert "test" in results[0].lower(), "The most similar result should contain the word 'test'"
    assert "software development" in results[0].lower() or len(
        results) > 1, "The response should match key aspects of the query"
