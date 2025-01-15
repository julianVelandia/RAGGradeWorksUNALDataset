from .dataset_loader import load_remote_dataset
from .vectorizer import TextVectorizer
from .retriever import Retriever
from .constants import DEFAULT_DATASET

class Rag:
    def __init__(self, token=None, hf_dataset=DEFAULT_DATASET):
        self.token = token
        self.hf_dataset = hf_dataset
        self.dataset = load_remote_dataset(hf_dataset)
        self.documents = [item['text'] for item in self.dataset]
        self.vectorizer = TextVectorizer()
        self.embeddings = self.vectorizer.fit_transform(self.documents)
        self.retriever = Retriever(self.vectorizer)

    def retrieval_augmented_generation(self, query, max_sections=5, threshold=0.5):
        similar_sections = self.retriever.find_similar_sections(query, self.embeddings, self.documents, max_sections, threshold)
        combined_context = f"{query}\n\nTen en cuenta este contexto:\n" + "\n".join(similar_sections)
        return combined_context