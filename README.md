# RAG - Retrieval-Augmented Generation

## Descripción
**RAG** es una librería diseñada para implementar sistemas de generación aumentada con recuperación de información. Utiliza datasets de Hugging Face, los vectoriza y permite consultas rápidas basadas en similitud de coseno.
![image](https://github.com/user-attachments/assets/ea271b48-376e-4496-a554-48ae915cecd4)

## Instalación

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/usuario/rag.git
   cd rag
   ```

2. Instalar la librería:
   ```bash
   pip install .
   ```

## Uso

### Configuración Inicial
En la primera ejecución, se realiza la carga del dataset, su vectorización, y el almacenamiento de los embeddings:

```python
from rag import Rag

# Configurar el sistema con un dataset de Hugging Face
rag = Rag(hf_dataset="JulianVelandia/unal-repository-dataset")
```

Esto genera:
- **Base de datos original**: Almacenada en memoria como lista de documentos.
- **Base de datos vectorizada**: Archivo `.npy` en la carpeta `embeddings/`.

### Consulta y Recuperación
Una vez configurado, puedes realizar consultas:

```python
query = "¿Qué es la energía renovable?"
response = rag.retrieval_augmented_generation(query)
print(response)
```

El resultado será el `prompt` inicial combinado con las secciones más relevantes del contexto:

```
¿Qué es la energía renovable?

Ten en cuenta este contexto:
[Sección similar 1]
[Sección similar 2]
...
```

## Flujo de Trabajo

1. **Setup (Preprocesamiento)**:
   - Carga el dataset desde Hugging Face.
   - Vectoriza los documentos usando TF-IDF.
   - Guarda los embeddings en formato `.npy`.

   ```plaintext
   Dataset HF -> Carga -> Vectorización -> Embeddings (.npy)
   ```

2. **Querying (Consulta)**:
   - Vectoriza el prompt.
   - Calcula similitudes coseno entre el prompt y los documentos vectorizados.
   - Recupera las secciones más relevantes.
   - Combina el prompt con el contexto recuperado.

   ```plaintext
   Prompt -> Vectorización -> Similitud coseno -> Recuperación -> Contexto combinado
   ```
