"""
Thin retrieval layer on top of embed_store — kept separate so retrieval logic
(re-ranking, filtering, dedup) can grow independently of the storage layer.

- retrieve(question, top_k=4): call embed_store.query, optionally re-rank or dedupe
  chunks from the same source, and return the final context list
- format_context(chunks): join retrieved chunks into a single context string with
  source citations (e.g. "[resume.pdf, chunk 2]") ready to drop into a prompt
"""

from src import embed_store


def retrieve(question: str, top_k: int = 4) -> list[dict]:
    # Over-fetch so dedup still leaves us with top_k distinct chunks.
    candidates = embed_store.query(question, top_k=top_k * 2)

    seen = set()
    deduped = []
    for chunk in candidates:
        key = (chunk["source"], chunk["chunk_index"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(chunk)

    deduped.sort(key=lambda c: c["score"], reverse=True)
    return deduped[:top_k]


def format_context(chunks: list[dict]) -> str:
    blocks = []
    for chunk in chunks:
        citation = f"[{chunk['source']}, chunk {chunk['chunk_index']}]"
        blocks.append(f"{citation}: {chunk['text']}")
    return "\n\n".join(blocks)
