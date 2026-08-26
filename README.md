# Portfolio RAG Q&A

A small, fully local-friendly Retrieval-Augmented Generation (RAG) system that answers questions
over a folder of your own documents (resume, papers, notes, whatever you point it at).

Built to demonstrate, with real public code, the same skills used in a RAG system built during
an internship: document ingestion, chunking, embedding generation, vector search, prompt
construction, generation, and — importantly — retrieval evaluation.

## Why this project exists

Most of the AI/ML work in my resume (RAG, LLMs, computer vision, agentic workflows) was built at
an internship in a private company repo. This project is a small, from-scratch, public rebuild of
the RAG half of that work, so the skill claim has visible code behind it.

## Architecture

```
Documents (PDF/txt)
      |
      v
  [ingest.py]  ---->  chunks of text
      |
      v
[embed_store.py] ---->  embeddings + vector store (Chroma)
      |
      v
  [retrieve.py]  ---->  top-k relevant chunks for a question
      |
      v
  [generate.py]  ---->  final answer (local LLM via Ollama, or a free-tier hosted API)
      |
      v
    [app.py]     ---->  Streamlit chat UI
```

`evaluate.py` runs a fixed set of test questions (see `eval/questions.json`) against the pipeline
and reports retrieval precision — i.e., "how often did the system's top chunk actually contain the
answer" — plus a rough answer-quality check. This is the piece that turns this from a tutorial
clone into something that shows understanding: an unevaluated RAG demo just shows you can call an
API; this one shows you can reason about whether the system is actually working.

## Tech choices (all free to run)

- **Chunking**: simple recursive character/token splitter (no paid service)
- **Embeddings**: `sentence-transformers` (local, free, e.g. `all-MiniLM-L6-v2`)
- **Vector store**: `chromadb` (local, file-based, no server to run)
- **Generation**: pluggable — supports a local model via [Ollama](https://ollama.com) (e.g.
  `llama3.2:3b`) OR a free-tier hosted API (Groq or Hugging Face Inference) via an environment
  variable, so it runs with zero cost either way
- **UI**: Streamlit (free to deploy on Streamlit Community Cloud)

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# put a few PDFs or .txt files in data/documents/

python -m src.ingest          # chunk + embed + store
python -m scripts.run_eval    # check retrieval quality against eval/questions.json
streamlit run app.py          # launch the chat UI
```

## Status

Implemented end to end: ingestion/chunking, embedding + Chroma storage, retrieval with dedup,
prompt construction + pluggable generation (Ollama/Groq/HF), evaluation, and the Streamlit UI.
Unit tests (chunking, retrieval, prompt construction) run fully offline via a mocked embedding
model — see `tests/`. CI runs the test suite on every push (`.github/workflows/test.yml`).

`data/documents/` is currently empty — add your own PDFs/text files, then run:

```bash
python -m src.ingest
python -m scripts.run_eval
```

to populate the vector store and see real evaluation numbers. Once real documents and matching
`eval/questions.json` entries are in place, this section will be updated with a "Results" section
showing actual retrieval accuracy and answer-quality numbers from a real run.

## Deployment

Once working locally, deploy the Streamlit app for free on Streamlit Community Cloud or Hugging
Face Spaces so the README can link to a live demo, not just code.
