"""Tests for src.ingest.load_documents (chunk_text is covered in test_retrieval.py)."""

from src.ingest import load_documents


def test_load_documents_reads_txt_files(tmp_path):
    (tmp_path / "notes.txt").write_text("Hello from a test document.", encoding="utf-8")
    (tmp_path / "ignored.md").write_text("This extension is not supported.", encoding="utf-8")

    documents = load_documents(tmp_path)

    assert len(documents) == 1
    assert documents[0]["source"] == "notes.txt"
    assert documents[0]["text"] == "Hello from a test document."


def test_load_documents_skips_empty_files(tmp_path):
    (tmp_path / "empty.txt").write_text("   ", encoding="utf-8")

    documents = load_documents(tmp_path)

    assert documents == []
