"""
Embedding generation + vector storage, backed by sentence-transformers + chromadb.

TODO (implementation):
- Load a sentence-transformers model once (e.g. "all-MiniLM-L6-v2")
- Initialize a persistent Chroma client pointed at ./chroma_db
- add_chunks(): embed a list of text chunks + metadata and upsert them into the collection
- query(question, top_k=4): embed the question, run similarity search, return the
  top_k chunks with their source metadata and similarity scores
- Keep this module's public interface stable — retrieve.py and evaluate.py both depend on it
"""

from pathlib import Path

CHROMA_DIR = Path(__file__).resolve().parent.parent / "chroma_db"
COLLECTION_NAME = "portfolio_docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def add_chunks(chunks: list[str], metadatas: list[dict]) -> None:
    """Embed and persist chunks with their metadata into the vector store."""
    raise NotImplementedError


def query(question: str, top_k: int = 4) -> list[dict]:
    """Return top_k chunks as [{"text": str, "source": str, "score": float}, ...]."""
    raise NotImplementedError
