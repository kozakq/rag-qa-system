"""
Basic tests for the ingestion + retrieval pipeline.

TODO (implementation):
- test_chunk_text_produces_overlapping_chunks(): sanity-check src.ingest.chunk_text
- test_query_returns_top_k_results(): after adding a couple of known chunks via
  embed_store.add_chunks, confirm embed_store.query returns them ranked sensibly
- Keep these fast and offline (no network calls) — mock the embedding model if needed
"""

import pytest


def test_placeholder():
    """Replace with real tests once ingest.py and embed_store.py are implemented."""
    assert True
