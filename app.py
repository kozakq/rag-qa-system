"""
Streamlit chat UI for the RAG system.

- Chat interface: text input (or one of a few suggested starter questions) for a
  question, showing the generated answer plus which source chunks were actually used
  and a rough confidence read on the top retrieval match (builds trust / shows the
  retrieval step isn't hidden).
- Sidebar surfaces what's actually loaded (corpus + backend + chunk count) so a
  visitor isn't looking at a black box.
- Should be runnable as: streamlit run app.py
"""

import os
from pathlib import Path

import streamlit as st

from src import embed_store, generate, ingest, retrieve

REPO_URL = "https://github.com/kozakq/rag-qa-system"

# Friendlier display names for the sample corpus; anything else just gets title-cased.
KNOWN_PAPER_TITLES = {
    "vaswani_2017_attention_is_all_you_need.pdf": "Vaswani et al., 2017 -- Attention Is All You Need",
    "lewis_2020_retrieval_augmented_generation.pdf": "Lewis et al., 2020 -- Retrieval-Augmented Generation",
    "reimers_2019_sentence_bert.pdf": "Reimers & Gurevych, 2019 -- Sentence-BERT",
    "karpukhin_2020_dense_passage_retrieval.pdf": "Karpukhin et al., 2020 -- Dense Passage Retrieval",
    "dosovitskiy_2020_vision_transformer.pdf": "Dosovitskiy et al., 2020 -- Vision Transformer (ViT)",
    "yao_2022_react.pdf": "Yao et al., 2022 -- ReAct",
}

EXAMPLE_QUESTIONS = [
    "What is retrieval-augmented generation and how does it work?",
    "How does multi-head attention work in a Transformer?",
    "What's the core idea behind the Vision Transformer (ViT)?",
    "How does ReAct combine reasoning and acting?",
]


def _pretty_doc_name(filename: str) -> str:
    if filename in KNOWN_PAPER_TITLES:
        return KNOWN_PAPER_TITLES[filename]
    return Path(filename).stem.replace("_", " ").replace("-", " ").title()


def _confidence(score: float) -> tuple[str, str]:
    # Rough calibration, not a validated threshold: observed top-hit scores on this
    # corpus (all-MiniLM-L6-v2, score = 1 - Chroma L2 distance) cluster around
    # 0.15-0.25 for genuinely relevant chunks and drop negative for noise/boilerplate.
    if score >= 0.15:
        return "High", "🟢"
    if score >= 0.0:
        return "Medium", "🟡"
    return "Low", "🔴"


def ask(question: str) -> None:
    with st.spinner("Retrieving context and generating an answer..."):
        chunks = retrieve.retrieve(question)
        context = retrieve.format_context(chunks)
        answer = generate.generate_answer(question, context)
    st.session_state.history.append({"question": question, "answer": answer, "chunks": chunks})


def render_turn(turn: dict) -> None:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        chunks = turn["chunks"]
        if chunks:
            label, icon = _confidence(chunks[0]["score"])
            st.caption(f"{icon} Retrieval confidence: **{label}** (top match score {chunks[0]['score']:.2f})")
            with st.expander(f"Sources it actually used ({len(chunks)} chunk{'s' if len(chunks) != 1 else ''})"):
                for chunk in chunks:
                    st.markdown(
                        f"**[{_pretty_doc_name(chunk['source'])}, chunk {chunk['chunk_index']}]** "
                        f"(score: {chunk['score']:.3f})"
                    )
                    st.text(chunk["text"])
        else:
            st.caption("No relevant chunks were found in the document store.")


st.set_page_config(page_title="RAG Paper Explorer", page_icon="📚")

st.title("RAG Paper Explorer")
st.caption(
    "A RAG pipeline I built end-to-end for chunking, embeddings, vector search, generation, "
    "and an actual evaluation harness that is pointed at the papers its own design is built from. "
    "Ask it how RAG works and it'll answer from the RAG paper it ingested, not from memory!"
)

with st.sidebar:
    st.header("About")
    st.markdown(f"Full source + eval results: [{REPO_URL.split('//')[1]}]({REPO_URL})")

    st.divider()
    st.subheader("Corpus")
    doc_files = sorted(
        p.name for p in ingest.DATA_DIR.glob("*") if p.suffix.lower() in {".pdf", ".txt"}
    )
    if doc_files:
        for name in doc_files:
            st.markdown(f"- {_pretty_doc_name(name)}")
    else:
        st.caption("No documents loaded yet. Run:")
        st.code("python -m scripts.fetch_papers\npython -m src.ingest", language="bash")

    st.divider()
    st.subheader("Backend")
    backend = os.getenv("GENERATION_BACKEND", "ollama")
    model = {
        "ollama": os.getenv("OLLAMA_MODEL", "llama3.2:latest"),
        "groq": os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        "hf": os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.3"),
    }.get(backend, "unknown")
    st.caption(f"Generation: **{backend}** ({model})")
    try:
        st.caption(f"Chunks indexed: **{embed_store.count()}**")
    except Exception:
        st.caption("Vector store not initialized yet -- run `python -m src.ingest`.")

    if st.session_state.get("history"):
        st.divider()
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.history = []
            st.rerun()

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    render_turn(turn)

if not st.session_state.history:
    st.markdown("**Try asking:**")
    cols = st.columns(2)
    for i, example in enumerate(EXAMPLE_QUESTIONS):
        if cols[i % 2].button(example, use_container_width=True, key=f"example_{i}"):
            ask(example)
            render_turn(st.session_state.history[-1])

question = st.chat_input("Ask about the papers above...")
if question:
    ask(question)
    render_turn(st.session_state.history[-1])
