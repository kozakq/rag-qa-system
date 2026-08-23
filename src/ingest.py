"""
Load documents from data/documents/, split them into overlapping chunks, and hand them
off to embed_store for embedding + storage.

- Load .pdf files with pypdf and .txt files directly
- Split each document's text into chunks (~300-500 tokens, ~15% overlap) — a simple
  recursive character splitter is fine; no need for anything fancy
- Track source metadata per chunk: {"source": filename, "chunk_index": i}
- Call embed_store.add_chunks(chunks, metadatas) to embed + persist them
- Should be runnable as: python -m src.ingest
"""

from pathlib import Path

from pypdf import PdfReader

from src import embed_store

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "documents"


def load_documents(data_dir: Path = DATA_DIR) -> list[dict]:
    """Return a list of {"source": str, "text": str} for every file in data_dir."""
    documents = []
    for path in sorted(data_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() == ".pdf":
            reader = PdfReader(str(path))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        elif path.suffix.lower() == ".txt":
            text = path.read_text(encoding="utf-8", errors="ignore")
        else:
            continue
        text = text.strip()
        if text:
            documents.append({"source": path.name, "text": text})
    return documents


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 60) -> list[str]:
    """Split text into overlapping chunks of roughly chunk_size tokens/words."""
    words = text.split()
    if not words:
        return []

    step = max(chunk_size - overlap, 1)
    chunks = []
    for start in range(0, len(words), step):
        chunk_words = words[start : start + chunk_size]
        if not chunk_words:
            continue
        chunks.append(" ".join(chunk_words))
        if start + chunk_size >= len(words):
            break
    return chunks


def main():
    documents = load_documents()
    if not documents:
        print(f"No documents found in {DATA_DIR}. Add .pdf or .txt files and try again.")
        return

    all_chunks: list[str] = []
    all_metadatas: list[dict] = []
    for doc in documents:
        doc_chunks = chunk_text(doc["text"])
        for i, chunk in enumerate(doc_chunks):
            all_chunks.append(chunk)
            all_metadatas.append({"source": doc["source"], "chunk_index": i})

    embed_store.add_chunks(all_chunks, all_metadatas)
    print(f"Ingested {len(documents)} document(s) into {len(all_chunks)} chunk(s):")
    for doc in documents:
        count = sum(1 for m in all_metadatas if m["source"] == doc["source"])
        print(f"  - {doc['source']}: {count} chunk(s)")


if __name__ == "__main__":
    main()
