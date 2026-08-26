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

The sample corpus is deliberately self-referential: it's the six papers that this project's own
pipeline is built from and adjacent to (transformers, RAG itself, sentence embeddings, dense
retrieval, vision transformers, and agentic prompting). You can ask this RAG system to explain
retrieval-augmented generation and watch it correctly retrieve and cite the RAG paper it
ingested — a more direct proof of understanding than a generic demo corpus would be.

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

python -m scripts.fetch_papers   # downloads the sample paper corpus from arXiv into data/documents/
python -m src.ingest             # chunk + embed + store
python -m scripts.run_eval       # check retrieval quality against eval/questions.json
streamlit run app.py             # launch the chat UI
```

Swap in your own PDFs/.txt files in `data/documents/` any time — just re-run `src.ingest`
afterward. Anything you add there beyond the fetched papers stays out of git (see `.gitignore`).

## Status

Implemented end to end: ingestion/chunking, embedding + Chroma storage, retrieval with dedup,
prompt construction + pluggable generation (Ollama/Groq/HF), evaluation, and the Streamlit UI.
Unit tests (chunking, retrieval, prompt construction) run fully offline via a mocked embedding
model — see `tests/`. CI runs the test suite on every push (`.github/workflows/test.yml`).

## Results

The sample corpus is six arXiv papers, fetched by `scripts/fetch_papers.py`: *Attention Is All You
Need*, *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, *Sentence-BERT*, *Dense
Passage Retrieval*, *An Image is Worth 16x16 Words* (Vision Transformer), and *ReAct*. Ingesting
them produces 174 chunks — enough for retrieval ranking to actually matter, unlike a handful of
short documents where every query would trivially return the whole corpus. `eval/questions.json`
has 8 hand-written questions, each targeting a specific fact in a specific paper. Generation
backend: local Ollama, `llama3.2:latest`.

At the default `top_k=4`:

| Metric | Result |
|---|---|
| Retrieval accuracy | 100.0% (8/8) |
| Answer keyword-hit rate | 87.5% (7/8) |

Retrieval found the correct source paper for every question. The one answer miss is the
interesting result: asked for the RAG paper's authors' institutional affiliations, the model
replied "I don't know the authors' institutional affiliations from the provided context" — and it
was right to. The byline chunk (`Facebook AI Research; University College London; ...`) is short
and mostly proper nouns, so it embeds poorly against a semantically-phrased query and didn't make
the top 4, even though other chunks from the same paper did (which is why the coarser
document-level "retrieval accuracy" metric still shows a hit here — it only checks whether the
right *paper* appeared, not whether the right *chunk* did). Faced with a genuine gap in its
context, the model declined to guess rather than hallucinating a plausible-sounding affiliation —
exactly the behavior `build_prompt()` is designed to produce. That's a more useful outcome from an
evaluation harness than a clean 100%: it surfaces a real, explainable retrieval-granularity
limitation and confirms the anti-hallucination instruction actually holds under a case it wasn't
specifically tuned for.

To reproduce:

```bash
python -m scripts.fetch_papers   # download the sample papers from arXiv
python -m src.ingest             # chunk + embed + store
python -m scripts.run_eval       # top_k=4 (default) report
```

## Deployment

Once working locally, deploy the Streamlit app for free on Streamlit Community Cloud or Hugging
Face Spaces so the README can link to a live demo, not just code.
