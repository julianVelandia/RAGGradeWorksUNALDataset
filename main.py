from constants import PATH_RAW_DATASET_HUGGINGFACE
from src.dataset_loader import load_remote_dataset
from src.vectorizer import TextVectorizer
from src.retriever import Retriever

dataset = load_remote_dataset(PATH_RAW_DATASET_HUGGINGFACE)
documents = [item['text'] for item in dataset]

vectorizer = TextVectorizer()
embeddings = vectorizer.fit_transform(documents)
vectorizer.save_embeddings(embeddings, "embeddings")


def retrieval_augmented_generation(query):

    retriever = Retriever(vectorizer)
    similar_sections = retriever.find_similar_sections(query, embeddings, documents, max_sections=5, threshold=0.5)

    combined_context = f"{query}\n\nTen en cuenta este contexto:\n" + "\n".join(similar_sections)
    print(combined_context)


if __name__ == "__main__":
    pass
