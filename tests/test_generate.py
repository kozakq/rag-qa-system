"""Tests for src.generate prompt construction and backend dispatch (no real network calls)."""

from src import generate


def test_build_prompt_includes_question_and_context():
    prompt = generate.build_prompt("What is RAG?", "[doc.txt, chunk 0]: RAG stands for ...")

    assert "What is RAG?" in prompt
    assert "RAG stands for" in prompt
    assert "I don't know" in prompt


def test_generate_answer_unknown_backend(monkeypatch):
    monkeypatch.setenv("GENERATION_BACKEND", "not-a-real-backend")

    answer = generate.generate_answer("question", "context")

    assert "unknown GENERATION_BACKEND" in answer


def test_generate_answer_groq_without_key(monkeypatch):
    monkeypatch.setenv("GENERATION_BACKEND", "groq")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    answer = generate.generate_answer("question", "context")

    assert "GROQ_API_KEY" in answer


def test_generate_answer_ollama_connection_error(monkeypatch):
    monkeypatch.setenv("GENERATION_BACKEND", "ollama")

    def _raise_connection_error(*args, **kwargs):
        import requests

        raise requests.exceptions.ConnectionError("no server")

    monkeypatch.setattr(generate.requests, "post", _raise_connection_error)

    answer = generate.generate_answer("question", "context")

    assert "could not reach Ollama" in answer
