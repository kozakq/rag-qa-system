"""
Tests for chunking (src.ingest) and retrieval (src.embed_store / src.retrieve).

Kept fast and offline: the embedding model is replaced with a tiny deterministic fake
so no network call or real model download happens during the test run.
"""

import hashlib

import numpy as np
import pytest

from src import embed_store, ingest, retrieve


class _FakeModel:
    """Deterministic bag-of-words style embedding, good enough to rank similar text higher."""

    def encode(self, texts):
        return np.array([self._embed(text) for text in texts])

    @staticmethod
    def _embed(text: str):
        vector = [0.0] * 16
        for word in text.lower().split():
            index = int(hashlib.md5(word.encode()).hexdigest(), 16) % len(vector)
            vector[index] += 1.0
        return vector


@pytest.fixture
def fake_embed_store(monkeypatch, tmp_path):
    """Point embed_store at a fresh temp Chroma dir and a fake embedding model."""
    import chromadb

    monkeypatch.setattr(embed_store, "_model", _FakeModel())
    monkeypatch.setattr(embed_store, "_get_model", lambda: embed_store._model)

    client = chromadb.PersistentClient(path=str(tmp_path / "chroma_db"))
    collection = client.get_or_create_collection("test_collection")
    monkeypatch.setattr(embed_store, "_collection", collection)
    monkeypatch.setattr(embed_store, "_get_collection", lambda: embed_store._collection)

    return embed_store


def test_chunk_text_produces_overlapping_chunks():
    text = " ".join(f"word{i}" for i in range(1000))
    chunks = ingest.chunk_text(text, chunk_size=400, overlap=60)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.split()) <= 400

    first_words = chunks[0].split()
    second_words = chunks[1].split()
    overlap_words = set(first_words[-60:]) & set(second_words[:60])
    assert overlap_words


def test_chunk_text_handles_empty_string():
    assert ingest.chunk_text("") == []


def test_query_returns_top_k_results(fake_embed_store):
    chunks = ["python programming language", "cats and dogs are pets", "python is a snake too"]
    metadatas = [
        {"source": "doc_a.txt", "chunk_index": 0},
        {"source": "doc_b.txt", "chunk_index": 0},
        {"source": "doc_a.txt", "chunk_index": 1},
    ]
    fake_embed_store.add_chunks(chunks, metadatas)

    results = fake_embed_store.query("python programming", top_k=2)

    assert len(results) == 2
    assert results[0]["source"] in {"doc_a.txt"}
    assert all("score" in r and "text" in r and "source" in r for r in results)


def test_retrieve_dedupes_and_formats_context(fake_embed_store, monkeypatch):
    chunks = ["alpha beta gamma", "delta epsilon zeta"]
    metadatas = [
        {"source": "doc_a.txt", "chunk_index": 0},
        {"source": "doc_b.txt", "chunk_index": 0},
    ]
    fake_embed_store.add_chunks(chunks, metadatas)

    results = retrieve.retrieve("alpha beta", top_k=2)
    assert len(results) <= 2

    context = retrieve.format_context(results)
    for chunk in results:
        assert f"[{chunk['source']}, chunk {chunk['chunk_index']}]" in context
        assert chunk["text"] in context


def test_count_reflects_added_chunks(fake_embed_store):
    assert fake_embed_store.count() == 0

    fake_embed_store.add_chunks(["one", "two"], [
        {"source": "doc_a.txt", "chunk_index": 0},
        {"source": "doc_a.txt", "chunk_index": 1},
    ])

    assert fake_embed_store.count() == 2
