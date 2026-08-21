"""
Thin retrieval layer on top of embed_store — kept separate so retrieval logic
(re-ranking, filtering, dedup) can grow independently of the storage layer.

TODO (implementation):
- retrieve(question, top_k=4): call embed_store.query, optionally re-rank or dedupe
  chunks from the same source, and return the final context list
- format_context(chunks): join retrieved chunks into a single context string with
  source citations (e.g. "[resume.pdf, chunk 2]") ready to drop into a prompt
"""


def retrieve(question: str, top_k: int = 4) -> list[dict]:
    raise NotImplementedError


def format_context(chunks: list[dict]) -> str:
    raise NotImplementedError
