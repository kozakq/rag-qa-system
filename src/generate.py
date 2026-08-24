"""
Answer generation, pluggable between a local model (Ollama) and a free-tier hosted API
(Groq or Hugging Face Inference), selected via the GENERATION_BACKEND env var so the
whole project runs at zero cost either way.

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

import requests
from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL = "http://localhost:11434/api/generate"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
HF_URL = "https://api-inference.huggingface.co/models/{model}"

DEFAULT_OLLAMA_MODEL = "llama3.2:latest"
DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"
DEFAULT_HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"


def build_prompt(question: str, context: str) -> str:
    return (
        "You are a helpful assistant answering questions using ONLY the context below.\n"
        "If the answer is not contained in the context, say \"I don't know\" — do not use "
        "any outside knowledge and do not make anything up.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def _generate_ollama(prompt: str) -> str:
    model = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["response"].strip()
    except requests.exceptions.ConnectionError:
        return (
            "Error: could not reach Ollama at http://localhost:11434. "
            "Is Ollama running (`ollama serve`) with a model pulled?"
        )
    except requests.exceptions.RequestException as exc:
        return f"Error calling Ollama: {exc}"


def _generate_groq(prompt: str) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "Error: GROQ_API_KEY is not set in .env."
    model = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
    try:
        response = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except requests.exceptions.RequestException as exc:
        return f"Error calling Groq: {exc}"


def _generate_hf(prompt: str) -> str:
    api_key = os.getenv("HF_API_KEY")
    if not api_key:
        return "Error: HF_API_KEY is not set in .env."
    model = os.getenv("HF_MODEL", DEFAULT_HF_MODEL)
    try:
        response = requests.post(
            HF_URL.format(model=model),
            headers={"Authorization": f"Bearer {api_key}"},
            json={"inputs": prompt},
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list) and data and "generated_text" in data[0]:
            return data[0]["generated_text"].strip()
        return str(data)
    except requests.exceptions.RequestException as exc:
        return f"Error calling Hugging Face Inference API: {exc}"


def generate_answer(question: str, context: str) -> str:
    prompt = build_prompt(question, context)
    backend = os.getenv("GENERATION_BACKEND", "ollama").lower()

    if backend == "ollama":
        return _generate_ollama(prompt)
    if backend == "groq":
        return _generate_groq(prompt)
    if backend == "hf":
        return _generate_hf(prompt)
    return f"Error: unknown GENERATION_BACKEND '{backend}' (expected ollama, groq, or hf)."
