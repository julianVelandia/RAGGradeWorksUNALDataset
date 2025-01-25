from rag import Rag


def test_rag_pipeline():
    rag = Rag()
    query = "¿Cuál es el Diseño de iluminación, control y embellecimiento de la cancha del Estadio Alfonso López?"
    response = rag.retrieval_augmented_generation(query)

    assert query in response, "The initial query should be in the response"
    assert "Keep in mind this context" in response, "The response should include the retrieved context"

    context_start = response.find("Keep in mind this context") + len("Keep in mind this context")

    assert len(response[context_start:].strip()) > 0, "The context should contain additional information"
