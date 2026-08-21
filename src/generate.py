"""
Answer generation, pluggable between a local model (Ollama) and a free-tier hosted API
(Groq or Hugging Face Inference), selected via the GENERATION_BACKEND env var so the
whole project runs at zero cost either way.

TODO (implementation):
- build_prompt(question, context): construct the final prompt, instructing the model to
  answer ONLY from the provided context and say "I don't know" if the context doesn't
  contain the answer (this matters — it's what keeps the demo honest instead of just
  letting the LLM hallucinate from its own training data)
- generate_answer(question, context): call the selected backend and return the answer text
- Support at least two backends behind one interface:
    - "ollama": POST to http://localhost:11434/api/generate
    - "groq" or "hf": call the free-tier hosted API using a key from .env
- Read the backend choice + any API key from environment variables (see .env.example)
"""

import os

GENERATION_BACKEND = os.getenv("GENERATION_BACKEND", "ollama")


def build_prompt(question: str, context: str) -> str:
    raise NotImplementedError


def generate_answer(question: str, context: str) -> str:
    raise NotImplementedError
