
# Simple RAG HuggingFace

## Description
Designed to implement retrieval-augmented generation systems. It uses datasets from Hugging Face, vectorizes them, and allows fast queries based on cosine similarity.
![image](https://github.com/user-attachments/assets/ea271b48-376e-4496-a554-48ae915cecd4)

## Installation

```bash
pip install SimpleRAGHuggingFace
```

## Usage

### Initial Setup
During the first execution, the dataset is loaded, vectorized, and embeddings are stored:

```python
from rag import Rag

rag = Rag() # Default Dataset
query = "What is the lighting design, control, and beautification of the field at Alfonso López Stadium?"
response = rag.retrieval_augmented_generation(query)
print(response)
```

Once run for the first time, the dataset can be queried for cosine similarity with the following parameters

```
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
```

This process generates:
- **Original Database**: Stored in memory as a list of documents.
- **Vectorized Database**: Saved as a `.npy` file in the `embeddings/` folder.

### Query and Retrieval
Once the setup is complete, you can perform queries:

```python
query = "What is the lighting design, control, and beautification of the field at Alfonso López Stadium?"
response = rag.retrieval_augmented_generation(query)
print(response)
```

The result will be the initial `prompt` combined with the most relevant sections of context:

```
What is the lighting design, control, and beautification of the field at Alfonso López Stadium?

Keep in mind this context:
Lighting design ... Alfonso López Stadium, as well as the results obtained, understanding that a soccer team ...
...
```

## Workflow

1. **Setup (Preprocessing)**:
   - Load the dataset from Hugging Face.
   - Vectorize the documents using TF-IDF.
   - Save the embeddings in `.npy` format.

   ```plaintext
   HF Dataset -> Load -> Vectorization -> Embeddings (.npy)
   ```

2. **Querying**:
   - Vectorize the prompt.
   - Calculate cosine similarity between the prompt and the vectorized documents.
   - Retrieve the most relevant sections.
   - Combine the prompt with the retrieved context.

   ```plaintext
   Prompt -> Vectorization -> Cosine Similarity -> Retrieval -> Combined Context
   ```
