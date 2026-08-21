"""
Streamlit chat UI for the RAG system.

TODO (implementation):
- Simple chat interface: text input for a question, display the generated answer plus
  which source chunks were used (builds trust / shows the retrieval step isn't hidden)
- A sidebar file uploader so a visitor can drop in their own PDFs and re-run ingestion
  live (nice-to-have, not required for v1)
- Should be runnable as: streamlit run app.py
"""

import streamlit as st

st.set_page_config(page_title="Portfolio RAG Q&A", page_icon="🔎")
st.title("Portfolio RAG Q&A")
st.caption("Ask a question about the documents in data/documents/")

st.info("TODO: wire this up to src.retrieve + src.generate once those are implemented.")

question = st.text_input("Your question")
if question:
    st.write("TODO: call retrieve() + generate_answer() and display the result here.")
