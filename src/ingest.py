"""
Load documents from data/documents/, split them into overlapping chunks, and hand them
off to embed_store for embedding + storage.

TODO (implementation):
- Load .pdf files with pypdf and .txt files directly
- Split each document's text into chunks (~300-500 tokens, ~15% overlap) — a simple
  recursive character splitter is fine; no need for anything fancy
- Track source metadata per chunk: {"source": filename, "chunk_index": i}
- Call embed_store.add_chunks(chunks, metadatas) to embed + persist them
- Should be runnable as: python -m src.ingest
"""

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "documents"


def load_documents(data_dir: Path = DATA_DIR) -> list[dict]:
    """Return a list of {"source": str, "text": str} for every file in data_dir."""
    raise NotImplementedError


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 60) -> list[str]:
    """Split text into overlapping chunks of roughly chunk_size tokens/words."""
    raise NotImplementedError


def main():
    raise NotImplementedError


if __name__ == "__main__":
    main()
