"""Local text embeddings via fastembed (ONNX, no API key).

Local embeddings mean retrieval text never leaves the machine; only the final,
redacted prompt is sent to the LLM.
"""

from fastembed import TextEmbedding

MODEL_NAME = "BAAI/bge-small-en-v1.5"
DIMENSION = 384


class Embedder:
    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self._model = TextEmbedding(model_name=model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(value) for value in vector] for vector in self._model.embed(texts)]

    @staticmethod
    def to_pgvector(vector: list[float]) -> str:
        return "[" + ",".join(f"{value:.6f}" for value in vector) + "]"
