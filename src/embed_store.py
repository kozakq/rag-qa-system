"""
Embedding generation + vector storage, backed by sentence-transformers + chromadb.

- Load a sentence-transformers model once (e.g. "all-MiniLM-L6-v2")
- Initialize a persistent Chroma client pointed at ./chroma_db
- add_chunks(): embed a list of text chunks + metadata and upsert them into the collection
- query(question, top_k=4): embed the question, run similarity search, return the
  top_k chunks with their source metadata and similarity scores
- Keep this module's public interface stable — retrieve.py and evaluate.py both depend on it
"""

from pathlib import Path

import chromadb

CHROMA_DIR = Path(__file__).resolve().parent.parent / "chroma_db"
COLLECTION_NAME = "portfolio_docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

_model = None
_collection = None


def _get_model():
    """Lazily load the sentence-transformers model (kept out of module import for testability)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _get_collection():
    """Lazily open the persistent Chroma collection."""
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_or_create_collection(COLLECTION_NAME)
    return _collection


def add_chunks(chunks: list[str], metadatas: list[dict]) -> None:
    """Embed and persist chunks with their metadata into the vector store."""
    if not chunks:
        return

    model = _get_model()
    collection = _get_collection()

    embeddings = model.encode(chunks).tolist()
    ids = [f"{m['source']}::{m['chunk_index']}" for m in metadatas]

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )


def count() -> int:
    """Return how many chunks are currently stored in the collection."""
    return _get_collection().count()


def query(question: str, top_k: int = 4) -> list[dict]:
    """Return top_k chunks as [{"text": str, "source": str, "score": float}, ...]."""
    model = _get_model()
    collection = _get_collection()

    if collection.count() == 0:
        return []

    question_embedding = model.encode([question]).tolist()
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=min(top_k, collection.count()),
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    output = []
    for text, metadata, distance in zip(documents, metadatas, distances):
        output.append(
            {
                "text": text,
                "source": metadata["source"],
                "chunk_index": metadata["chunk_index"],
                "score": 1 - distance,
            }
        )
    return output
