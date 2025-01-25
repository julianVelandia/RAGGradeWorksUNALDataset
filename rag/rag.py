import os

from .constants import DEFAULT_DATASET
from .dataset_loader import load_remote_dataset, delete_original_dataset
from .retriever import Retriever
from .vectorizer import TextVectorizer


class Rag:
    def __init__(self, token=None, hf_dataset=DEFAULT_DATASET):
        self.token = token
        self.hf_dataset = hf_dataset
        self.dataset = load_remote_dataset(hf_dataset)

        self.documents = [" ".join(str(value) for value in item.values()) for item in self.dataset]

        self.vectorizer = TextVectorizer()
        self.embeddings = self.vectorizer.fit_transform(self.documents)
        self.retriever = Retriever(self.vectorizer)

        delete_original_dataset(os.path.join("data", hf_dataset))

    def retrieval_augmented_generation(self, query, max_sections=5, threshold=0.4, max_words=1000):
        """
        Generate a response enriched with context retrieved from the dataset.

        Parameters:
        - query (str): The input question or statement to be processed.
        - max_sections (int): Maximum number of context sections to retrieve (range: 1 to 10).
        Higher values provide more context but may dilute relevance.
        - threshold (float): Minimum similarity score for a section to be included (range: 0.0 to 1.0).
        Higher values ensure stricter relevance.
        - max_words (int, optional): Maximum number of words in the combined context (default: 1000).
        Longer limits provide more detail but may reduce conciseness.

        Returns:
        - str: The combined query and relevant context, or just the query if no context is found.
        """
        similar_sections = self.retriever.find_similar_sections(query, self.embeddings, self.documents, max_sections,
                                                                threshold)

        if similar_sections:
            combined_context = f"{query}\n\nKeep in mind this context:\n" + "\n".join(similar_sections)
            combined_context = " ".join(combined_context.split()[:max_words])
        else:
            combined_context = query

        return combined_context