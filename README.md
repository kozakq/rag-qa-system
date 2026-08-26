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

# data/documents/ already ships with a small sample corpus (Skyrim lore/mechanics articles).
# Swap in your own PDFs/.txt files any time — just re-run ingest afterward.

python -m src.ingest          # chunk + embed + store
python -m scripts.run_eval    # check retrieval quality against eval/questions.json
streamlit run app.py          # launch the chat UI
```

## Status

Implemented end to end: ingestion/chunking, embedding + Chroma storage, retrieval with dedup,
prompt construction + pluggable generation (Ollama/Groq/HF), evaluation, and the Streamlit UI.
Unit tests (chunking, retrieval, prompt construction) run fully offline via a mocked embedding
model — see `tests/`. CI runs the test suite on every push (`.github/workflows/test.yml`).

## Results

The sample corpus in `data/documents/` is four short articles about *The Elder Scrolls V: Skyrim*
(overview, lore/history, factions & characters, gameplay mechanics), with 8 hand-written questions
in `eval/questions.json` targeting specific facts from specific documents. Generation backend:
local Ollama, `llama3.2:latest`.

At the default `top_k=4`:

| Metric | Result |
|---|---|
| Retrieval accuracy | 100.0% (8/8) |
| Answer keyword-hit rate | 100.0% (8/8) |

That 100% is a little too easy to be meaningful, though: this corpus only has 4 chunks total (each
document is short enough to fit in a single ~400-word chunk), so at `top_k=4` every query trivially
retrieves *every* chunk regardless of ranking quality — the metric can't fail. Re-running with a
stricter `python -c "from src.evaluate import evaluate; evaluate(top_k=1)"` — keeping only the
single best-ranked chunk — gives a real signal:

| Metric (top_k=1) | Result |
|---|---|
| Retrieval accuracy | 62.5% (5/8) |
| Answer keyword-hit rate | 62.5% (5/8) |

The three misses at `top_k=1` (questions about the Dwemer, Paarthurnax, and the Thu'um/Word Walls)
were all pulled toward `skyrim_lore_history.txt` instead of their actual source — because these
four documents are topically close (all Skyrim lore/mechanics) and each is embedded as one single
whole-document vector, fine-grained distinctions get blurred together at the embedding level. With
a larger or more granular corpus (more chunks per document, more separation between topics) this
would sharpen up; on a 4-chunk corpus it's expected. This is the actual point of having an
evaluation harness: it surfaces a real, explainable limitation instead of hiding behind a vanity
metric.

To reproduce:

```bash
python -m src.ingest          # chunk + embed + store the sample Skyrim documents
python -m scripts.run_eval    # top_k=4 (default) report
```

## Deployment

Once working locally, deploy the Streamlit app for free on Streamlit Community Cloud or Hugging
Face Spaces so the README can link to a live demo, not just code.
