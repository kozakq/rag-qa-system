# Prompt for Claude Code

Copy everything below the line into a fresh Claude Code session, opened in this project's folder.

---

I'm a Computer Science student building a portfolio project to publicly demonstrate RAG
(Retrieval-Augmented Generation) skills for job applications — this needs to be real, working,
well-tested code, not a toy tutorial. I already have a scaffold in place: folder structure,
`requirements.txt`, a README describing the architecture, and stub modules with docstrings
describing exactly what each one should do. Please implement the whole thing end to end.

## What to build

A RAG Q&A system that answers questions over a folder of my own documents (PDFs/text files in
`data/documents/`). Read `README.md` first for the full architecture diagram and rationale.

Implement, in this order:

1. **`src/ingest.py`** — load PDFs (pypdf) and .txt files from `data/documents/`, split into
   overlapping chunks (~300-500 tokens, ~15% overlap), track source filename + chunk index as
   metadata. Runnable as `python -m src.ingest`.
2. **`src/embed_store.py`** — embed chunks with `sentence-transformers` (`all-MiniLM-L6-v2`) and
   persist them in a local `chromadb` collection. Implement `add_chunks()` and
   `query(question, top_k)`.
3. **`src/retrieve.py`** — a thin layer over `embed_store.query` that dedupes/re-ranks if useful,
   and formats retrieved chunks into a context string with source citations.
4. **`src/generate.py`** — build a prompt that instructs the model to answer ONLY from the
   provided context and say "I don't know" if the answer isn't there (don't let it hallucinate
   from training data — that's the whole point of RAG). Support two backends behind one
   interface, selected via `GENERATION_BACKEND` in `.env`:
   - `ollama`: local model via `http://localhost:11434/api/generate` (assume the user has Ollama
     installed with a small model pulled, e.g. `llama3.2:3b`)
   - `groq` (or `hf` for Hugging Face Inference): a free-tier hosted API using a key from `.env`
   Everything must be runnable at zero cost.
5. **`src/evaluate.py`** + **`scripts/run_eval.py`** — load `eval/questions.json`, for each
   question retrieve top-k chunks and check whether `expected_source` appears among them
   (retrieval accuracy), generate an answer and check whether `expected_keywords` appear in it
   (rough answer-quality proxy). Print a clear report: retrieval accuracy %, answer-keyword-hit %,
   and per-question detail. This evaluation step matters most — it's what proves understanding of
   the system rather than just wiring an API call.
6. **`app.py`** — a Streamlit chat UI: text input, display the answer plus which source chunks
   were used. Keep it simple and clean.
7. **`tests/test_retrieval.py`** — real tests for chunking and retrieval (mock the embedding
   model or use a tiny fixture so tests run fast and offline). Add a couple more test files if it
   makes sense (e.g. for `generate.py`'s prompt construction).

## Also do this

- Populate `eval/questions.json` with 5-8 real test questions once I've added real documents to
  `data/documents/` (ask me to add a couple of PDFs/text files first if that folder is still
  empty, or use placeholder text files you generate for testing that I can swap out).
- Update the README's "Status" section once implemented, and add a "Results" section with the
  actual evaluation numbers from a real run.
- Set up a minimal GitHub Actions workflow (`.github/workflows/test.yml`) that runs `pytest` on
  push, so the repo shows a passing CI badge.
- Write clear, incremental commits as you go (not one giant commit at the end) — this repo's
  commit history is part of what a reviewer will look at.
- Double check `.gitignore` covers `.venv/`, `chroma_db/`, `.env`, and real documents in
  `data/documents/` (only `.gitkeep` should be committed there — don't commit my personal PDFs).

## Constraints

- Everything must run with free/local tools only — no paid API keys required for the core path
  (Ollama local generation should be the default and always work).
- Keep dependencies to what's already in `requirements.txt` unless something is genuinely
  missing — add it there if so.
- Target Python 3.11+.

## When you're done

Give me the exact commands to run the eval script and the Streamlit app locally, and a short
checklist for deploying `app.py` to Streamlit Community Cloud or Hugging Face Spaces for a live
demo link.
