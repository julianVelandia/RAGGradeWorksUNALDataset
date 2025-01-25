import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class Retriever:
    def __init__(self, vectorizer):
        self.vectorizer = vectorizer

    def find_similar_sections(self, query, embeddings, documents, max_sections=5, threshold=0.4):
        if not isinstance(query, str):
            raise ValueError("Query must be a string.")

        query_vector = self.vectorizer.transform(query)
        similarities = cosine_similarity(query_vector, embeddings).flatten()
        ranked_indices = np.argsort(-similarities)

        results = []
        for idx in ranked_indices[:max_sections]:
            if similarities[idx] < threshold:
                break
            results.append(documents[idx])

        return results
