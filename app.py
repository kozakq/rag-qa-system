"""
Streamlit chat UI for the RAG system.

- Simple chat interface: text input for a question, display the generated answer plus
  which source chunks were used (builds trust / shows the retrieval step isn't hidden)
- Should be runnable as: streamlit run app.py
"""

import streamlit as st

from src import generate, retrieve

st.set_page_config(page_title="Portfolio RAG Q&A", page_icon="🔎")
st.title("Portfolio RAG Q&A")
st.caption("Ask a question about the documents in data/documents/")

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        if turn["chunks"]:
            with st.expander(f"Sources ({len(turn['chunks'])} chunk(s) used)"):
                for chunk in turn["chunks"]:
                    st.markdown(
                        f"**[{chunk['source']}, chunk {chunk['chunk_index']}]** "
                        f"(score: {chunk['score']:.3f})"
                    )
                    st.text(chunk["text"])
        else:
            st.caption("No relevant chunks were found in the document store.")

question = st.chat_input("Your question")
if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating an answer..."):
            chunks = retrieve.retrieve(question)
            context = retrieve.format_context(chunks)
            answer = generate.generate_answer(question, context)

        st.write(answer)
        if chunks:
            with st.expander(f"Sources ({len(chunks)} chunk(s) used)"):
                for chunk in chunks:
                    st.markdown(
                        f"**[{chunk['source']}, chunk {chunk['chunk_index']}]** "
                        f"(score: {chunk['score']:.3f})"
                    )
                    st.text(chunk["text"])
        else:
            st.caption("No relevant chunks were found in the document store.")

    st.session_state.history.append({"question": question, "answer": answer, "chunks": chunks})
